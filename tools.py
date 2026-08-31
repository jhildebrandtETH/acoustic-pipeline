import os
import shutil
import subprocess
import time
import numpy as np
import pandas as pd
import re
import signal
import threading
import traceback
from pathlib import Path
import json
from decimal import Decimal, InvalidOperation
from datetime import datetime



# ============================================================================
# TOOLS.PY STRUCTURE
# ============================================================================
# 1. Generic status / file-write helpers
# 2. CLI and configuration validation
# 3. OpenFOAM single-case execution helpers
# 4. Parallel batch scheduler / runtime dashboard
# 5. Resume / reconstruction helpers
# 6. Mesh, OpenFOAM dictionary, convergence and postprocessing utilities
# ============================================================================

# ============================================================================
# PARALLEL BATCH SCHEDULING / STATUS INFRASTRUCTURE
# ============================================================================

_SIMULATION_ORDER_FILE_LOCK = threading.RLock()
MATPLOTLIB_LOCK = threading.RLock()


def emit_status(status_callback=None, **fields):
    """Safely emit a structured runtime-status update for one case."""
    if status_callback is None:
        return

    try:
        status_callback(**fields)
    except Exception:
        # Monitoring must never be allowed to crash a simulation case.
        pass


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON through a temporary file and atomically replace the target."""
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=4)
        handle.flush()
        os.fsync(handle.fileno())

    os.replace(tmp_path, path)


def _write_pipeline_error_log(
    simulations_directory: Path,
    folder_name: str,
    traceback_text: str,
) -> Path | None:
    """Persist a full worker traceback where it survives dashboard redraws."""
    simulations_directory = Path(simulations_directory)
    safe_folder = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(folder_name))
    log_path = simulations_directory / f"{safe_folder}.pipeline_error.log"

    try:
        simulations_directory.mkdir(parents=True, exist_ok=True)
        log_path.write_text(str(traceback_text), encoding="utf-8")
        return log_path
    except OSError:
        # Error reporting must never hide the original worker exception.
        return None


def _case_path_state(path: Path) -> str:
    """Describe a case target without following a broken symlink away."""
    path = Path(path)
    if path.is_dir():
        return "directory"
    if path.is_symlink():
        return "symlink"
    if path.exists():
        return "non-directory entry"
    return "missing"


# ============================================================================
# CLI / CONFIGURATION VALIDATION
# ============================================================================

def validate_acoustic_arguments(parser, args):
    """Validate coupled acoustic CLI arguments and apply safe defaults."""
    if (
        args.acoustic_surface == "impermeable"
        and args.acoustic_sphere_diameter is not None
    ):
        parser.error(
            "Impermeable acoustic mode was selected but "
            "--acoustic-sphere-diameter was also supplied. The sphere "
            "diameter is only used for permeable mode."
        )

    if (
        args.acoustic_surface == "permeable"
        and args.acoustic_sphere_diameter is None
    ):
        args.acoustic_sphere_diameter = 2.5


# ============================================================================
# OPENFOAM SINGLE-CASE EXECUTION HELPERS
# ============================================================================

def report_case_stage(
    status_callback,
    stage,
    detail="",
    progress=None,
    error=None,
):
    """Report one OpenFOAM stage without letting monitoring affect execution."""
    fields = {
        "stage": stage,
        "detail": detail,
    }

    if progress is not None:
        fields["progress"] = progress
    if error is not None:
        fields["error"] = error

    emit_status(status_callback, **fields)


def run_openfoam_command(
    container,
    command,
    description,
    status_callback,
    stage,
    detail=None,
    print_output=False,
):
    """Execute one Docker/OpenFOAM command and publish start/end status."""
    report_case_stage(
        status_callback,
        stage,
        detail or f"{description} running",
    )

    success = safe_exec(
        container,
        command,
        description,
        print_output=print_output,
        status_callback=status_callback,
    )

    if success:
        report_case_stage(
            status_callback,
            stage,
            f"{description} finished",
            progress=100.0,
        )
    else:
        # safe_exec already reports the specific Docker/OpenFOAM failure.
        report_case_stage(
            status_callback,
            stage,
            f"{description} failed",
        )

    return success



def resolve_cfmesh_executable(explicit_path=None) -> Path:
    """
    Resolve the host-side cfMesh generateBoundaryLayers executable.

    Resolution order:
      1. explicit_path argument,
      2. CFMESH_BIN environment variable,
      3. generateBoundaryLayers available on PATH,
      4. default setup_cfmesh.sh installation under ~/.local/cfmesh.
    """
    candidates = []

    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())

    env_path = os.environ.get("CFMESH_BIN")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    path_match = shutil.which("generateBoundaryLayers")
    if path_match:
        candidates.append(Path(path_match))

    candidates.append(
        Path.home()
        / ".local"
        / "cfmesh"
        / "cfMesh-1.2.0"
        / "bin"
        / "generateBoundaryLayers"
    )
    candidates.append(
        Path.home()
        / "tools"
        / "cfmesh"
        / "cfMesh-1.2.0"
        / "bin"
        / "generateBoundaryLayers"
    )

    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except OSError:
            resolved = candidate.expanduser()

        if resolved.is_file() and os.access(resolved, os.X_OK):
            return resolved

    searched = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "cfMesh generateBoundaryLayers executable was not found.\n"
        "Run `bash setup_cfmesh.sh`, add generateBoundaryLayers to PATH, "
        "or set CFMESH_BIN to the executable path.\n"
        f"Searched:\n  - {searched}"
    )


def verify_openfoam_patch_exists(
    simulation_directory: Path,
    patch_name: str,
) -> None:
    """Fail early when the reconstructed root mesh does not contain a patch."""
    boundary_file = (
        Path(simulation_directory)
        / "constant"
        / "polyMesh"
        / "boundary"
    )

    if not boundary_file.is_file():
        raise FileNotFoundError(
            f"OpenFOAM boundary file not found: {boundary_file}"
        )

    text = boundary_file.read_text(encoding="utf-8", errors="ignore")
    pattern = rf"(?m)^\s*{re.escape(str(patch_name))}\s*$"

    if re.search(pattern, text) is None:
        raise ValueError(
            f"Patch '{patch_name}' was not found in {boundary_file}. "
            "The root mesh may not contain the reconstructed snappyHexMesh result."
        )


def prepare_case_for_cfmesh(
    simulation_directory: Path,
) -> None:
    """
    Prepare a fresh pre-solver OpenFOAM case for host-side cfMesh processing.

    cfMesh configuration is intentionally owned by system/meshDict (which may
    include files from Parameters/). The Python pipeline does not generate or
    modify nLayers, thicknessRatio, patch settings, or other layer controls.

    Diagnostic polyMesh/sets are disposable and can confuse older cfMesh
    runtimes. 0/uniform/time is restart metadata and is also unnecessary before
    the first solver run.
    """
    simulation_directory = Path(simulation_directory)

    mesh_dict = simulation_directory / "system" / "meshDict"
    if not mesh_dict.is_file():
        raise FileNotFoundError(
            f"cfMesh configuration not found: {mesh_dict}. "
            "Provide system/meshDict in the case template; it may include "
            "detailed settings from Parameters/."
        )

    sets_directory = (
        simulation_directory
        / "constant"
        / "polyMesh"
        / "sets"
    )
    if sets_directory.exists():
        shutil.rmtree(sets_directory)

    uniform_time = simulation_directory / "0" / "uniform" / "time"
    try:
        uniform_time.unlink(missing_ok=True)
    except OSError:
        pass


def run_cfmesh_boundary_layer_process(
    executable: Path,
    simulation_directory: Path,
    number_of_cores: int,
    status_callback=None,
) -> bool:
    """
    Run generateBoundaryLayers on the host with a strict per-case OpenMP limit.

    The process inherits the SLURM cpuset from the Python worker, while the
    OpenMP variables cap the cfMesh thread count to this case's allocation.
    """
    executable = Path(executable)
    simulation_directory = Path(simulation_directory)
    number_of_cores = int(number_of_cores)

    if number_of_cores < 1:
        raise ValueError("cfMesh number_of_cores must be at least 1")

    log_path = simulation_directory / "log.generateBoundaryLayers"
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": str(number_of_cores),
            "OMP_DYNAMIC": "FALSE",
            "OMP_MAX_ACTIVE_LEVELS": "1",
            "OMP_PROC_BIND": "close",
            "OMP_PLACES": "cores",
        }
    )

    report_case_stage(
        status_callback,
        "cfMesh",
        f"generateBoundaryLayers running | {number_of_cores} thread(s)",
    )

    command = [
        str(executable),
        "-case",
        str(simulation_directory),
    ]
    stdbuf_executable = shutil.which("stdbuf")
    if stdbuf_executable:
        command = [stdbuf_executable, "-oL", "-eL", *command]

    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                cwd=str(simulation_directory),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
            return_code = process.wait()
    except Exception as error:
        message = f"cfMesh launch failed: {error}"
        report_case_stage(
            status_callback,
            "cfMesh",
            message,
            error=message,
        )
        return False

    if return_code != 0:
        if return_code < 0:
            signal_number = -return_code
            try:
                signal_name = signal.Signals(signal_number).name
            except ValueError:
                signal_name = "UNKNOWN"
            message = (
                "generateBoundaryLayers terminated by signal "
                f"{signal_number} ({signal_name})"
            )
        else:
            message = f"generateBoundaryLayers exited with code {return_code}"

        try:
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write("\n\n" + "=" * 72 + "\n")
                log_file.write("PIPELINE DETECTED cfMesh FAILURE\n")
                log_file.write(f"{message}\n")
                log_file.write("=" * 72 + "\n")
        except OSError:
            pass

        report_case_stage(
            status_callback,
            "cfMesh",
            message,
            error=message,
        )
        return False

    log_text = log_path.read_text(encoding="utf-8", errors="ignore")
    if "Writing mesh" not in log_text or re.search(r"(?m)^\s*End\s*$", log_text) is None:
        message = (
            "generateBoundaryLayers returned zero but its log did not contain "
            "the expected 'Writing mesh' / 'End' completion markers"
        )
        report_case_stage(
            status_callback,
            "cfMesh",
            message,
            error=message,
        )
        return False

    report_case_stage(
        status_callback,
        "cfMesh",
        "boundary-layer generation finished",
        progress=100.0,
    )
    return True


def remove_stale_stopped_container(
    client,
    container_name,
    status_callback=None,
):
    """Remove an old stopped case container, but never an active one."""
    # Docker is imported locally so tools.py remains importable for CLI/order
    # operations even in environments where the Docker SDK is not installed.
    import docker

    try:
        existing = client.containers.get(container_name)
    except docker.errors.NotFound:
        return

    existing.reload()

    if existing.status == "running":
        raise RuntimeError(
            f"Docker container '{container_name}' is already running. "
            "Refusing to remove a potentially active simulation container."
        )

    report_case_stage(
        status_callback,
        "docker",
        "removing stale stopped container",
    )
    existing.remove(force=True)


class SimulationOrderStore:
    """
    Thread-safe owner of simulation_order.json.

    Worker threads must not read-modify-write the JSON independently. They report
    durable state transitions through this object, which serializes and atomically
    persists every update.
    """

    def __init__(self, simulations_directory: Path):
        self.simulations_directory = Path(simulations_directory)
        self.json_path = self.simulations_directory / "simulation_order.json"
        self._lock = threading.RLock()
        self._batch = load_simulation_order(self.simulations_directory)

    def snapshot(self) -> dict:
        with self._lock:
            # JSON round-trip provides a simple deep copy for this JSON-only data.
            return json.loads(json.dumps(self._batch))

    def get_case(self, folder_name: str) -> dict:
        with self._lock:
            for case in self._batch["cases"]:
                if case["folder"] == folder_name:
                    return dict(case)

        raise KeyError(f"Unknown simulation case: {folder_name}")

    def case_status(self, folder_name: str) -> str:
        return str(self.get_case(folder_name)["status"])

    def update_case(self, folder_name: str, **updates) -> dict:
        with self._lock:
            for case in self._batch["cases"]:
                if case["folder"] == folder_name:
                    case.update(updates)
                    case["updated_at"] = datetime.now().isoformat(timespec="seconds")
                    _atomic_write_json(self.json_path, self._batch)
                    return dict(case)

        raise KeyError(f"Unknown simulation case: {folder_name}")

    def set_status(self, folder_name: str, new_status: str, **updates) -> dict:
        return self.update_case(folder_name, status=new_status, **updates)

    def mark_failed(
        self,
        folder_name: str,
        error: str,
        resume_status: str,
    ) -> dict:
        return self.update_case(
            folder_name,
            status="failed",
            error=str(error),
            resume_status=str(resume_status),
        )

    def persist_batch_fields(self, **updates) -> None:
        with self._lock:
            self._batch.update(updates)
            _atomic_write_json(self.json_path, self._batch)


class RuntimeStatusRegistry:
    """In-memory, thread-safe runtime state used by the live batch dashboard."""

    def __init__(self, cases: list[dict]):
        self._lock = threading.RLock()
        self._states = {}

        for case in cases:
            folder = case["folder"]
            persisted_status = case.get("status", "pending")

            if persisted_status == "postprocessing_done":
                runtime_state = "DONE"
            elif persisted_status == "failed":
                runtime_state = "FAILED"
            else:
                runtime_state = "QUEUED"

            self._states[folder] = {
                "folder": folder,
                "mesh": case.get("mesh", ""),
                "rpm": case.get("rpm", ""),
                "cores": int(case.get("allocated_cores", case.get("cores", 1))),
                "state": runtime_state,
                "stage": persisted_status,
                "detail": "",
                "progress": None,
                "dependency": case.get("depends_on"),
                "error": case.get("error"),
                "updated_at": time.time(),
            }

    def update(self, folder_name: str, **updates) -> None:
        with self._lock:
            if folder_name not in self._states:
                self._states[folder_name] = {
                    "folder": folder_name,
                    "mesh": "",
                    "rpm": "",
                    "cores": 1,
                    "state": "QUEUED",
                    "stage": "",
                    "detail": "",
                    "progress": None,
                    "dependency": None,
                    "error": None,
                    "updated_at": time.time(),
                }

            self._states[folder_name].update(updates)
            self._states[folder_name]["updated_at"] = time.time()

    def callback_for(self, folder_name: str):
        def callback(**fields):
            normalized = dict(fields)

            # Case-level workers may only provide stage/detail/progress; the
            # scheduler owns the high-level RUNNING/QUEUED/DONE state.
            self.update(folder_name, **normalized)

        return callback

    def get(self, folder_name: str) -> dict:
        with self._lock:
            if folder_name not in self._states:
                raise KeyError(f"Unknown runtime case: {folder_name}")
            return dict(self._states[folder_name])

    def snapshot(self) -> list[dict]:
        with self._lock:
            return [dict(value) for value in self._states.values()]


class LiveBatchDashboard:
    """
    Render a self-refreshing terminal overview.

    Running cases are always shown first. Queued and dependency-waiting cases are
    grouped in a second table below them, followed by failed cases when present.
    """

    def __init__(
        self,
        registry: RuntimeStatusRegistry,
        total_cores: int,
        cores_per_case: int,
        max_cores_per_case: int | None = None,
        refresh_interval: float = 1.0,
    ):
        import sys

        self.registry = registry
        self.total_cores = int(total_cores)
        self.cores_per_case = int(cores_per_case)
        self.max_cores_per_case = int(
            max_cores_per_case
            if max_cores_per_case is not None
            else cores_per_case
        )
        self.refresh_interval = float(refresh_interval)
        self._stop_event = threading.Event()
        self._thread = None
        self._is_tty = bool(sys.stdout.isatty())
        self._last_non_tty_render = 0.0

    @staticmethod
    def _clip(value, width):
        text = "" if value is None else str(value)
        if len(text) <= width:
            return text
        if width <= 3:
            return text[:width]
        return text[: width - 3] + "..."

    @staticmethod
    def _progress_text(value):
        if value is None:
            return "-"

        try:
            value = float(value)
        except (TypeError, ValueError):
            return str(value)

        return f"{value:6.2f}%"

    def _table(self, title: str, rows: list[dict]) -> list[str]:
        lines = [title]
        lines.append(
            f"{'Case':36} {'Cores':>5} {'Stage':20} {'Progress':>10}  Detail"
        )
        lines.append("-" * 100)

        if not rows:
            lines.append("(none)")
            return lines

        for item in rows:
            detail = item.get("detail") or ""

            if item.get("state") == "WAITING_INIT" and item.get("dependency"):
                detail = f"waiting for {item['dependency']}"

            if item.get("state") == "FAILED" and item.get("error"):
                detail = item["error"]

            lines.append(
                f"{self._clip(item.get('folder'), 36):36} "
                f"{int(item.get('cores', 0)):>5} "
                f"{self._clip(item.get('stage'), 20):20} "
                f"{self._progress_text(item.get('progress')):>10}  "
                f"{self._clip(detail, 55)}"
            )

        return lines

    def render_text(self) -> str:
        states = self.registry.snapshot()

        running = sorted(
            (item for item in states if item.get("state") == "RUNNING"),
            key=lambda item: item["folder"],
        )
        queued = sorted(
            (
                item
                for item in states
                if item.get("state") in {"QUEUED", "WAITING_INIT"}
            ),
            key=lambda item: (
                0 if item.get("state") == "QUEUED" else 1,
                item["folder"],
            ),
        )
        failed = sorted(
            (item for item in states if item.get("state") in {"FAILED", "BLOCKED"}),
            key=lambda item: item["folder"],
        )
        done_count = sum(item.get("state") == "DONE" for item in states)
        used_cores = sum(int(item.get("cores", 0)) for item in running)

        lines = [
            "ACOUSTIC PIPELINE - PARALLEL SIMULATION ORDER",
            (
                f"Total cores: {self.total_cores} | Used: {used_cores} | "
                f"Case cores: "
                f"{self.cores_per_case}"
                f"{('-' + str(self.max_cores_per_case)) if self.max_cores_per_case != self.cores_per_case else ''} | "
                f"Running: {len(running)} | "
                f"Queued/waiting: {len(queued)} | Done: {done_count} | "
                f"Failed/blocked: {len(failed)}"
            ),
            "",
        ]

        lines.extend(self._table("RUNNING CASES", running))
        lines.append("")
        lines.extend(self._table("QUEUED / WAITING CASES", queued))

        if failed:
            lines.append("")
            lines.extend(self._table("FAILED / BLOCKED CASES", failed))

        return "\n".join(lines)

    def _run(self):
        while not self._stop_event.wait(self.refresh_interval):
            now = time.time()

            if self._is_tty:
                print("\033[H\033[J" + self.render_text(), end="", flush=True)
            elif now - self._last_non_tty_render >= 30.0:
                # In redirected output / SLURM logs, avoid ANSI escape spam.
                print("\n" + self.render_text(), flush=True)
                self._last_non_tty_render = now

    def start(self):
        if self._thread is not None:
            return

        self._thread = threading.Thread(
            target=self._run,
            name="batch-dashboard",
            daemon=True,
        )
        self._thread.start()

    def stop(self, final_render=True):
        self._stop_event.set()

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5)

        if final_render:
            if self._is_tty:
                print("\033[H\033[J" + self.render_text(), flush=True)
            else:
                print("\n" + self.render_text(), flush=True)


def calculate_scheduler_layout(
    cases: list[dict],
    total_cores: int,
    field_init: str,
    study: bool,
) -> dict:
    """Return slot count and near-equal core allocation for maximum throughput."""
    total_cores = int(total_cores)

    if total_cores < 1:
        raise ValueError("total_cores must be at least 1")
    if not cases:
        raise ValueError("Simulation order contains no cases")

    dependency_mode = str(field_init).lower() == "on" and not bool(study)

    if dependency_mode:
        parallel_units = len({case["mesh"] for case in cases})
    else:
        parallel_units = len(cases)

    max_parallel_cases = max(1, min(parallel_units, total_cores))
    base_cores = max(1, total_cores // max_parallel_cases)
    extra_core_slots = total_cores % max_parallel_cases
    max_cores = base_cores + (1 if extra_core_slots else 0)

    return {
        "total_cores": total_cores,
        "cores_per_case": base_cores,
        "max_cores_per_case": max_cores,
        "extra_core_slots": extra_core_slots,
        "max_parallel_cases": max_parallel_cases,
        "dependency_mode": dependency_mode,
    }


def assign_case_core_allocations(
    cases: list[dict],
    layout: dict,
    field_init: str,
    study: bool,
) -> None:
    """Assign each case its fixed MPI/serial core count for this order."""
    base_cores = int(layout["cores_per_case"])
    extra_slots = int(layout.get("extra_core_slots", 0))
    dependency_mode = str(field_init).lower() == "on" and not bool(study)

    if dependency_mode:
        mesh_order = []
        for case in cases:
            if case["mesh"] not in mesh_order:
                mesh_order.append(case["mesh"])

        mesh_cores = {
            mesh: base_cores + (1 if index < extra_slots else 0)
            for index, mesh in enumerate(mesh_order)
        }
        for case in cases:
            case["allocated_cores"] = mesh_cores[case["mesh"]]
        return

    # If there are more cases than cores, base_cores is 1 and extra_slots is 0,
    # so every queued case naturally uses one core as slots become available.
    for index, case in enumerate(cases):
        case["allocated_cores"] = base_cores + (1 if index < extra_slots else 0)


def add_field_initialization_dependencies(cases: list[dict], field_init: str, study: bool) -> None:
    """Mutate case entries so each mesh forms an explicit RPM dependency chain."""
    dependency_mode = str(field_init).lower() == "on" and not bool(study)
    previous_case_by_mesh = {}

    for case in cases:
        if dependency_mode:
            dependency = previous_case_by_mesh.get(case["mesh"])
            case["depends_on"] = dependency
            previous_case_by_mesh[case["mesh"]] = case["folder"]
        else:
            case["depends_on"] = None


def ensure_scheduler_metadata(
    order: dict,
    total_cores_override: int | None = None,
) -> dict:
    """Upgrade simulation_order.json data to the parallel-scheduler schema."""
    if "total_cores" not in order:
        # In the sequential pipeline, ``cores`` meant cores PER case. It is
        # therefore unsafe to reinterpret that legacy field as the new TOTAL
        # order budget. The caller must explicitly supply the new budget.
        if total_cores_override is None:
            raise ValueError(
                "Legacy simulation order detected. Its 'cores' field means "
                "cores per case, not total order cores. Resume it once with "
                "--total-cores <available cores> to migrate safely."
            )
        order["total_cores"] = int(total_cores_override)

    order["schema_version"] = 2

    # Boundary-layer metadata was introduced after scheduler schema v2.
    # Legacy orders keep their historical behavior unless the values were
    # explicitly stored when the order was created.
    order.setdefault("boundary_layers", "none")

    if any("depends_on" not in case for case in order.get("cases", [])):
        add_field_initialization_dependencies(
            order["cases"],
            order.get("field_init", "off"),
            order.get("study", False),
        )

    layout = calculate_scheduler_layout(
        order["cases"],
        order["total_cores"],
        order.get("field_init", "off"),
        order.get("study", False),
    )

    order.setdefault("cores_per_case", layout["cores_per_case"])
    order.setdefault("max_cores_per_case", layout["max_cores_per_case"])
    order.setdefault("extra_core_slots", layout["extra_core_slots"])
    order.setdefault("max_parallel_cases", layout["max_parallel_cases"])

    if any("allocated_cores" not in case for case in order["cases"]):
        assign_case_core_allocations(
            order["cases"],
            layout,
            order.get("field_init", "off"),
            order.get("study", False),
        )

    for case in order["cases"]:
        case.setdefault("resume_status", None)
        case.setdefault("error", None)

    return order


def dependency_state(case: dict, status_by_folder: dict[str, str]) -> tuple[str, str | None]:
    """Return READY, WAITING, or BLOCKED plus a human-readable reason."""
    dependency = case.get("depends_on")

    if not dependency:
        return "READY", None

    dependency_status = status_by_folder.get(dependency)

    if dependency_status == "postprocessing_done":
        return "READY", None

    if dependency_status in {"failed", "blocked"}:
        return "BLOCKED", f"initialization dependency failed: {dependency}"

    return "WAITING", f"waiting for initialization source: {dependency}"


def initialize_runtime_queue_states(order: dict, registry: RuntimeStatusRegistry) -> None:
    """Set initial QUEUED/WAITING/BLOCKED dashboard states from dependencies."""
    status_by_folder = {
        case["folder"]: case.get("status", "pending")
        for case in order["cases"]
    }

    for case in order["cases"]:
        folder = case["folder"]
        status = case.get("status", "pending")

        if status == "postprocessing_done":
            registry.update(folder, state="DONE", stage="complete", progress=100.0)
            continue

        if status == "failed":
            registry.update(
                folder,
                state="FAILED",
                stage="failed",
                error=case.get("error"),
            )
            continue

        dep_state, reason = dependency_state(case, status_by_folder)

        if dep_state == "READY":
            registry.update(folder, state="QUEUED", stage=status, detail="ready")
        elif dep_state == "WAITING":
            registry.update(folder, state="WAITING_INIT", stage="waiting_init", detail=reason)
        else:
            registry.update(folder, state="BLOCKED", stage="blocked", detail=reason, error=reason)


def runnable_cases(order: dict, running_folders: set[str]) -> list[dict]:
    """Return currently dependency-ready, non-terminal cases in stable order."""
    status_by_folder = {
        case["folder"]: case.get("status", "pending")
        for case in order["cases"]
    }

    ready = []

    for case in order["cases"]:
        folder = case["folder"]

        if folder in running_folders:
            continue

        if case.get("status") in {"postprocessing_done", "failed", "blocked"}:
            continue

        dep_state, _ = dependency_state(case, status_by_folder)

        if dep_state == "READY":
            ready.append(case)

    return ready


def refresh_dependency_runtime_states(order: dict, registry: RuntimeStatusRegistry, running_folders: set[str]) -> None:
    """Refresh queued/waiting/blocked dashboard state after a case transition."""
    status_by_folder = {
        case["folder"]: case.get("status", "pending")
        for case in order["cases"]
    }

    for case in order["cases"]:
        folder = case["folder"]

        if folder in running_folders:
            continue

        status = case.get("status", "pending")

        if status == "postprocessing_done":
            registry.update(folder, state="DONE", stage="complete", progress=100.0, detail="finished")
            continue

        if status == "failed":
            registry.update(folder, state="FAILED", stage="failed", error=case.get("error"))
            continue

        if status == "blocked":
            registry.update(folder, state="BLOCKED", stage="blocked", error=case.get("error"))
            continue

        dep_state, reason = dependency_state(case, status_by_folder)

        if dep_state == "READY":
            registry.update(folder, state="QUEUED", stage=status, detail="ready", progress=None)
        elif dep_state == "WAITING":
            registry.update(folder, state="WAITING_INIT", stage="waiting_init", detail=reason, progress=None)
        else:
            registry.update(folder, state="BLOCKED", stage="blocked", detail=reason, error=reason)


def block_cases_with_failed_dependencies(order_store: SimulationOrderStore, registry: RuntimeStatusRegistry) -> None:
    """
    Persist BLOCKED for every descendant of a failed initialization source.

    This is intentionally iterative. If A_3000 fails, A_4000 becomes blocked;
    that new durable state must then immediately block A_5000 in the same
    scheduler cycle instead of leaving a temporary dependency deadlock.
    """
    while True:
        order = order_store.snapshot()
        status_by_folder = {
            case["folder"]: case.get("status", "pending")
            for case in order["cases"]
        }
        newly_blocked = 0

        for case in order["cases"]:
            if case.get("status") in {"postprocessing_done", "failed", "blocked"}:
                continue

            dep_state, reason = dependency_state(case, status_by_folder)

            if dep_state == "BLOCKED":
                order_store.set_status(
                    case["folder"],
                    "blocked",
                    error=reason,
                    resume_status=case.get("status", "pending"),
                )
                registry.update(
                    case["folder"],
                    state="BLOCKED",
                    stage="blocked",
                    detail=reason,
                    error=reason,
                    progress=None,
                )
                newly_blocked += 1

        if newly_blocked == 0:
            return


def reactivate_failed_cases_for_resume(
    order_store: SimulationOrderStore,
    simulations_directory: Path,
) -> None:
    """Reactivate failed/blocked cases using their stored durable resume point."""
    order = order_store.snapshot()

    for case in order["cases"]:
        status = case.get("status")

        if status == "failed":
            resume_status = case.get("resume_status")

            if resume_status not in {
                "pending",
                "preprocessing_done",
                "solver_running",
                "solver_done",
            }:
                simulation_path = Path(simulations_directory) / case["folder"]
                processor0 = simulation_path / "processor0"

                if has_timestep(processor0) or has_timestep(simulation_path):
                    resume_status = "solver_running"
                else:
                    resume_status = "pending"

            order_store.set_status(
                case["folder"],
                resume_status,
                error=None,
                resume_status=None,
            )

        elif status == "blocked":
            # Dependency state is recalculated after failed parent cases are
            # reactivated, so descendants return to their prior durable state.
            fallback = case.get("resume_status") or "pending"
            order_store.set_status(
                case["folder"],
                fallback,
                error=None,
                resume_status=None,
            )


# ============================================================================
# CASE WORKER / RESOURCE-AWARE SCHEDULER
# ============================================================================

def resume_status_after_solver_failure(simulation_path: Path) -> str:
    processor0_path = simulation_path / "processor0"

    if has_timestep(processor0_path) or has_timestep(simulation_path):
        return "solver_running"

    return "pending"


def execute_simulation_case(
    case,
    pipeline_main_directory,
    simulations_directory,
    source_meshes_directory,
    source_meshes,
    order_store,
    registry,
    args,
    convergence_monitoring_revolutions_count=1000,
    convergence_tolerance=1e-3,
):
    """Execute one durable case state machine inside one worker thread."""
    # Local imports avoid circular imports: preprocessing/openfoamSimulation/
    # postprocessing themselves import helpers from tools.py.
    from openfoamSimulation import openfoamSimulation
    from postprocessing import postprocessing
    from preprocessing import preprocessing

    folder_name = case["folder"]
    mesh = case["mesh"]
    rpm = int(case["rpm"])
    mode = case["mode"]
    is_study_case = bool(case["study"])
    allocated_cores = int(case["allocated_cores"])
    dependency = case.get("depends_on")
    callback = registry.callback_for(folder_name)

    simulation_path = simulations_directory / folder_name
    status = order_store.case_status(folder_name)

    # Do NOT pre-create simulation_path here. For a new case, preprocessing()
    # owns creation of the case by copying the selected template with
    # dirs_exist_ok=True. The previous unconditional mkdir was redundant and,
    # on Windows-mounted WSL paths, could escape this worker's exception handler
    # as FileExistsError before preprocessing had even started.
    try:
        path_state = _case_path_state(simulation_path)
        if path_state in {"symlink", "non-directory entry"}:
            raise RuntimeError(
                f"Case path cannot be used because it is a {path_state}: "
                f"{simulation_path}"
            )

        if status != "pending" and not simulation_path.is_dir():
            raise FileNotFoundError(
                f"Cannot resume case '{folder_name}' from status '{status}' "
                f"because its case directory is missing: {simulation_path}"
            )

        registry.update(
            folder_name,
            state="RUNNING",
            stage="starting",
            detail=f"worker started | {allocated_cores} core(s)",
            progress=None,
            error=None,
        )

        if mesh not in source_meshes:
            raise FileNotFoundError(
                f"Source mesh '{mesh}' was not found in {source_meshes_directory}"
            )

        previous_simulation_path = (
            simulations_directory / dependency
            if dependency is not None
            else None
        )
        use_previous_init = (
            args.field_init == "on"
            and dependency is not None
            and not is_study_case
        )

        while status != "postprocessing_done":
            # --------------------------------------------------------------
            # PREPROCESSING
            # --------------------------------------------------------------
            if status == "pending":
                registry.update(
                    folder_name,
                    stage="preprocessing",
                    detail="preparing case",
                    progress=None,
                )

                preprocessing_kwargs = dict(
                    SIMULATION_NAME=folder_name,
                    RPM_COUNT=rpm,
                    MAIN_DIRECTORY=pipeline_main_directory,
                    TARGET_DIRECTORY=simulation_path,
                    CORES_TO_USE=allocated_cores,
                    MODE=mode,
                    INIT_FROM_PREVIOUS=use_previous_init,
                    PREVIOUS_SIMULATION_PATH=previous_simulation_path,
                    TURBULENCE_MODEL=args.turbulence,
                    ACOUSTIC_SURFACE=args.acoustic_surface,
                    ACOUSTIC_SPHERE_DIAMETER=args.acoustic_sphere_diameter,
                    STATUS_CALLBACK=callback,
                )

                if is_study_case:
                    preprocessing_kwargs.update(
                        STUDY_PARAMETER_NAME=case["study_parameter"],
                        STUDY_PARAMETER_FILE=case["study_file"],
                        STUDY_PARAMETER=case["study_value"],
                    )

                preprocessing(**preprocessing_kwargs)
                order_store.set_status(
                    folder_name,
                    "preprocessing_done",
                    error=None,
                    resume_status=None,
                )
                status = "preprocessing_done"
                continue

            # --------------------------------------------------------------
            # NEW SOLVER START
            # --------------------------------------------------------------
            if status == "preprocessing_done":
                order_store.set_status(
                    folder_name,
                    "solver_running",
                    error=None,
                    resume_status=None,
                )
                status = "solver_running"

                success = openfoamSimulation(
                    resume=False,
                    simulation_name=folder_name,
                    simulation_working_directory=simulation_path,
                    convergence_tolerance=convergence_tolerance,
                    rpm_count=rpm,
                    convergence_window_revolutions=(
                        convergence_monitoring_revolutions_count
                    ),
                    MODE=mode,
                    END_ON_MODE=args.end_on,
                    TURBULENCE_MODEL=args.turbulence,
                    initialize_from_previous=use_previous_init,
                    previous_simulation_path=previous_simulation_path,
                    NUMBER_OF_CORES=allocated_cores,
                    MESH_ONLY=args.mesh_only,
                    ALLOW_BAD_MESH=args.allow_bad_mesh,
                    BOUNDARY_LAYER_METHOD=args.boundary_layers,
                    STATUS_CALLBACK=callback,
                )

                if not success:
                    resume_status = resume_status_after_solver_failure(
                        simulation_path
                    )
                    error = (
                        registry.get(folder_name).get("error")
                        or "OpenFOAM stage returned failure"
                    )
                    order_store.mark_failed(
                        folder_name,
                        error=error,
                        resume_status=resume_status,
                    )
                    registry.update(
                        folder_name,
                        state="FAILED",
                        stage="failed",
                        detail=error,
                        error=error,
                        progress=None,
                    )
                    return False

                order_store.set_status(
                    folder_name,
                    "solver_done",
                    error=None,
                    resume_status=None,
                )
                status = "solver_done"
                continue

            # --------------------------------------------------------------
            # SOLVER RESUME
            # --------------------------------------------------------------
            if status == "solver_running":
                processor0_path = simulation_path / "processor0"
                has_any_timestep = (
                    has_timestep(processor0_path)
                    or has_timestep(simulation_path)
                )

                if not has_any_timestep:
                    callback(
                        stage="resume",
                        detail="no timestep found; moving case aside and restarting cleanly",
                        progress=None,
                    )
                    reset_case_folder(
                        simulation_path,
                        status_callback=callback,
                    )
                    order_store.set_status(
                        folder_name,
                        "pending",
                        error=None,
                        resume_status=None,
                    )
                    status = "pending"
                    continue

                safe_time = get_safe_timestep(simulation_path)

                if safe_time is None:
                    callback(
                        stage="resume",
                        detail="no usable safe timestep; restarting cleanly",
                    )
                    reset_case_folder(
                        simulation_path,
                        status_callback=callback,
                    )
                    order_store.set_status(
                        folder_name,
                        "pending",
                        error=None,
                        resume_status=None,
                    )
                    status = "pending"
                    continue

                callback(
                    stage="resume",
                    detail=f"resuming from safe timestep {safe_time}",
                )

                success = openfoamSimulation(
                    resume=True,
                    simulation_name=folder_name,
                    simulation_working_directory=simulation_path,
                    convergence_tolerance=convergence_tolerance,
                    rpm_count=rpm,
                    convergence_window_revolutions=(
                        convergence_monitoring_revolutions_count
                    ),
                    MODE=mode,
                    END_ON_MODE=args.end_on,
                    TURBULENCE_MODEL=args.turbulence,
                    initialize_from_previous=use_previous_init,
                    previous_simulation_path=previous_simulation_path,
                    NUMBER_OF_CORES=allocated_cores,
                    MESH_ONLY=args.mesh_only,
                    ALLOW_BAD_MESH=args.allow_bad_mesh,
                    BOUNDARY_LAYER_METHOD=args.boundary_layers,
                    STATUS_CALLBACK=callback,
                )

                if not success:
                    resume_status = resume_status_after_solver_failure(
                        simulation_path
                    )
                    error = (
                        registry.get(folder_name).get("error")
                        or "OpenFOAM resume stage returned failure"
                    )
                    order_store.mark_failed(
                        folder_name,
                        error=error,
                        resume_status=resume_status,
                    )
                    registry.update(
                        folder_name,
                        state="FAILED",
                        stage="failed",
                        detail=error,
                        error=error,
                    )
                    return False

                order_store.set_status(
                    folder_name,
                    "solver_done",
                    error=None,
                    resume_status=None,
                )
                status = "solver_done"
                continue

            # --------------------------------------------------------------
            # POSTPROCESSING
            # --------------------------------------------------------------
            if status == "solver_done":
                if args.mesh_only:
                    order_store.set_status(
                        folder_name,
                        "postprocessing_done",
                        error=None,
                        resume_status=None,
                    )
                    status = "postprocessing_done"
                    continue

                registry.update(
                    folder_name,
                    stage="postprocessing",
                    detail="starting postprocessing",
                    progress=0.0,
                )

                postprocessing(
                    ACOUSTIC_SURFACE=args.acoustic_surface,
                    SIMULATION_WORKING_DIRECTORY=simulation_path,
                    RPM_COUNT=rpm,
                    MODE=mode,
                    TURBULENCE_MODEL=args.turbulence,
                    STATUS_CALLBACK=callback,
                )

                order_store.set_status(
                    folder_name,
                    "postprocessing_done",
                    error=None,
                    resume_status=None,
                )
                status = "postprocessing_done"
                continue

            raise ValueError(
                f"Unknown case status for {folder_name}: {status}"
            )

        registry.update(
            folder_name,
            state="DONE",
            stage="complete",
            detail="finished",
            progress=100.0,
            error=None,
        )
        return True

    except Exception as error:
        full_traceback = traceback.format_exc()
        error_log = _write_pipeline_error_log(
            simulations_directory,
            folder_name,
            full_traceback,
        )

        if status == "solver_done":
            resume_status = "solver_done"
        elif status == "solver_running":
            resume_status = resume_status_after_solver_failure(
                simulation_path
            )
        elif status == "preprocessing_done":
            resume_status = "preprocessing_done"
        else:
            resume_status = "pending"

        error_text = f"{type(error).__name__}: {error}"
        if error_log is not None:
            error_text += f" | traceback: {error_log}"

        order_store.mark_failed(
            folder_name,
            error=error_text,
            resume_status=resume_status,
        )
        if error_log is not None:
            order_store.update_case(
                folder_name,
                error_log=str(error_log),
            )

        registry.update(
            folder_name,
            state="FAILED",
            stage="failed",
            detail=error_text,
            error=error_text,
            progress=None,
        )
        return False


def run_parallel_scheduler(
    pipeline_main_directory,
    simulations_directory,
    source_meshes_directory,
    source_meshes,
    order_store,
    registry,
    args,
    convergence_monitoring_revolutions_count=1000,
    convergence_tolerance=1e-3,
    scheduler_poll_interval=0.5,
):
    from concurrent.futures import ThreadPoolExecutor

    order = order_store.snapshot()
    max_parallel_cases = int(order["max_parallel_cases"])
    total_cores = int(order["total_cores"])
    cores_per_case = int(order["cores_per_case"])

    dashboard = LiveBatchDashboard(
        registry=registry,
        total_cores=total_cores,
        cores_per_case=cores_per_case,
        max_cores_per_case=int(order.get("max_cores_per_case", cores_per_case)),
        refresh_interval=1.0,
    )

    executor = ThreadPoolExecutor(
        max_workers=max_parallel_cases,
        thread_name_prefix="simulation-case",
    )
    running_futures = {}
    dashboard.start()

    try:
        while True:
            # Collect completed workers first so their slots can be reused
            # immediately by newly dependency-ready cases.
            for future, folder in list(running_futures.items()):
                if future.done():
                    try:
                        future.result()
                    except Exception as error:
                        # This is a last-resort guard. execute_simulation_case()
                        # should normally capture its own failures, but any error
                        # that escapes still gets a persistent full traceback.
                        full_traceback = traceback.format_exc()
                        error_log = _write_pipeline_error_log(
                            simulations_directory,
                            folder,
                            full_traceback,
                        )
                        error_text = (
                            f"Unhandled worker error: {type(error).__name__}: {error}"
                        )
                        if error_log is not None:
                            error_text += f" | traceback: {error_log}"

                        order_store.mark_failed(
                            folder,
                            error=error_text,
                            resume_status="pending",
                        )
                        if error_log is not None:
                            order_store.update_case(
                                folder,
                                error_log=str(error_log),
                            )
                        registry.update(
                            folder,
                            state="FAILED",
                            stage="failed",
                            detail=error_text,
                            error=error_text,
                        )

                    del running_futures[future]

            block_cases_with_failed_dependencies(order_store, registry)
            order = order_store.snapshot()
            running_folders = set(running_futures.values())

            refresh_dependency_runtime_states(
                order,
                registry,
                running_folders,
            )

            terminal_statuses = {
                "postprocessing_done",
                "failed",
                "blocked",
            }
            unfinished = [
                case
                for case in order["cases"]
                if case.get("status") not in terminal_statuses
            ]

            if not unfinished and not running_futures:
                break

            free_slots = max_parallel_cases - len(running_futures)

            if free_slots > 0:
                ready = runnable_cases(order, running_folders)

                for case in ready[:free_slots]:
                    folder = case["folder"]
                    registry.update(
                        folder,
                        state="RUNNING",
                        stage="starting",
                        detail="assigned scheduler slot",
                        progress=None,
                        error=None,
                    )

                    future = executor.submit(
                        execute_simulation_case,
                        case,
                        pipeline_main_directory,
                        simulations_directory,
                        source_meshes_directory,
                        source_meshes,
                        order_store,
                        registry,
                        args,
                        convergence_monitoring_revolutions_count,
                        convergence_tolerance,
                    )
                    running_futures[future] = folder

            if not running_futures:
                # If no worker is running and no case can be launched, the order
                # is dependency-deadlocked. Failed dependencies should already be
                # marked BLOCKED above, so this catches malformed dependency data.
                order = order_store.snapshot()
                ready = runnable_cases(order, set())
                unfinished = [
                    case
                    for case in order["cases"]
                    if case.get("status") not in terminal_statuses
                ]

                if unfinished and not ready:
                    raise RuntimeError(
                        "Scheduler deadlock: unfinished cases remain but none "
                        "are runnable. Check depends_on entries in simulation_order.json."
                    )

            time.sleep(scheduler_poll_interval)

    finally:
        executor.shutdown(wait=True, cancel_futures=False)
        dashboard.stop(final_render=True)



# ============================================================================
# OPENFOAM TIME-DIRECTORY / RESUME HELPERS
# ============================================================================

def _numeric_time_directories(directory, maximum_time=None):
    """Return numeric OpenFOAM time-directory names up to maximum_time."""
    directory = Path(directory)

    if not directory.is_dir():
        return set()

    upper_limit = (
        Decimal(str(maximum_time)) if maximum_time is not None else None
    )
    time_names = set()

    for path in directory.iterdir():
        if not path.is_dir():
            continue

        try:
            time_value = Decimal(path.name)
        except InvalidOperation:
            continue

        # Preserve the original 0/ initial-condition directory.
        if time_value <= 0:
            continue

        if upper_limit is None or time_value <= upper_limit:
            time_names.add(path.name)

    return time_names


def reconstructed_history_is_complete(
    simulation_directory,
    safe_time,
    status_callback=None,
):
    """Check that every processor0 time up to safe_time exists reconstructed."""
    simulation_directory = Path(simulation_directory)
    processor0_directory = simulation_directory / "processor0"

    expected_times = _numeric_time_directories(
        processor0_directory, maximum_time=safe_time
    )
    reconstructed_times = _numeric_time_directories(
        simulation_directory, maximum_time=safe_time
    )

    if not expected_times:
        emit_status(
            status_callback,
            stage="resume_check",
            detail=(
                f"No decomposed time directories up to {safe_time} were found "
                f"in {processor0_directory}"
            ),
        )
        return False

    missing_times = expected_times - reconstructed_times

    if missing_times:
        missing_times = sorted(missing_times, key=Decimal)
        emit_status(
            status_callback,
            stage="resume_check",
            detail=f"Reconstruction incomplete; missing {len(missing_times)} time directories",
        )
        return False

    emit_status(
        status_callback,
        stage="resume_check",
        detail=f"Verified {len(expected_times)} reconstructed time directories through {safe_time}",
    )
    return True


def _run_reconstruction_with_progress(
    container,
    command,
    description,
    simulation_directory,
    maximum_time=None,
    status_callback=None,
):
    """Run reconstructPar while reporting filesystem progress."""
    reconstruction_stop_event = threading.Event()

    reconstruction_thread = threading.Thread(
        target=run_reconstruction_progress_monitor,
        kwargs={
            "main_sim_folder": simulation_directory,
            "maximum_time": maximum_time,
            "check_interval": 2,
            "stop_event": reconstruction_stop_event,
            "status_callback": status_callback,
        },
        name="reconstruct-par-progress-monitor",
        daemon=True,
    )
    reconstruction_thread.start()

    try:
        return safe_exec(
            container,
            command,
            description,
            status_callback=status_callback,
        )
    finally:
        reconstruction_stop_event.set()

        if reconstruction_thread.is_alive():
            reconstruction_thread.join(timeout=10)

        if reconstruction_thread.is_alive():
            emit_status(
                status_callback,
                stage="reconstructing",
                detail="WARNING: reconstruction progress monitor did not stop within timeout",
            )


# ============================================================================
# GEOMETRY / OPENFOAM DICTIONARY PARSING
# ============================================================================

def find_source_stls(source_meshes_directory: Path) -> dict[str, Path]:
    """
    Find all .stl files inside "STL".

    Example:
        STL/11x7E.stl -> geometry name: 11x7E

    Returns
    -------
    dict[str, Path]
        Mapping from STL name to its complete source path.
    """

    if not source_meshes_directory.is_dir():
        raise FileNotFoundError(
            f"Source STL directory does not exist: "
            f"{source_meshes_directory}"
        )

    stl_files = sorted(
        path
        for path in source_meshes_directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".stl"
    )

    if not stl_files:
        raise FileNotFoundError(
            f"No .stl geometries files were found in: "
            f"{source_meshes_directory}"
        )

    source_meshes = {}

    for mesh_path in stl_files:
        mesh_name = mesh_path.stem

        if mesh_name in source_meshes:
            raise ValueError(
                f"Multiple source meshes use the name '{mesh_name}' in "
                f"{source_meshes_directory}"
            )

        source_meshes[mesh_name] = mesh_path

    return source_meshes

def get_y_domain_height(blockmeshdict_path):

    text = Path(blockmeshdict_path).read_text(errors="ignore")

    # Extract vertices block
    vertices_match = re.search(
        r"vertices\s*\((.*?)\);",
        text,
        re.DOTALL
    )

    if not vertices_match:
        raise ValueError("No vertices block found.")

    vertices_text = vertices_match.group(1)

    # Extract all y coordinates
    y_values = []

    for match in re.finditer(
        r"\(\s*[-+eE0-9\.]+\s+([-+eE0-9\.]+)\s+[-+eE0-9\.]+\s*\)",
        vertices_text
    ):
        y_values.append(float(match.group(1)))

    if not y_values:
        raise ValueError("No y values found.")

    return max(y_values) - min(y_values)


def read_openfoam_scalar(file_path, variable_name):
    """
    Reads a scalar OpenFOAM dictionary entry and returns it as float.

    Example supported lines:
        rhoInf         1.225;
        nu             1.5e-05;
        endTime        0.2;

    Parameters
    ----------
    file_path : str or Path
        Path to OpenFOAM dictionary file.

    variable_name : str
        Variable to search for.

    Returns
    -------
    float

    Raises
    ------
    FileNotFoundError
    ValueError
    """

    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    text = file_path.read_text(errors="ignore")

    pattern = rf"^\s*{re.escape(variable_name)}\s+([^\s;]+)\s*;"

    match = re.search(pattern, text, re.MULTILINE)

    if not match:
        raise ValueError(
            f"Variable '{variable_name}' not found in {file_path}"
        )

    value_string = match.group(1)

    try:
        return float(value_string)

    except ValueError:
        raise ValueError(
            f"Variable '{variable_name}' is not numeric: {value_string}"
        )



_OPENFOAM_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def read_openfoam_timestep_and_courant_statistics(
    log_path: str | Path,
    control_dict_path: str | Path | None = None,
) -> dict:
    """
    Read time-step and Courant-number statistics from an OpenFOAM solver log.

    The parser recognizes common OpenFOAM output such as::

        Time = 0.001
        deltaT = 1e-05
        Courant Number mean: 0.012 max: 0.84
        Interface Courant Number mean: 0.001 max: 0.12
        Mesh Courant Number mean: 0.004 max: 0.35

    If the log does not print ``deltaT = ...`` (common for fixed time-step
    runs), delta-t values are reconstructed from consecutive ``Time = ...``
    entries. Negative and zero differences caused by restarts are ignored.

    Parameters
    ----------
    log_path:
        Path to the OpenFOAM solver log, for example ``log.pimpleFoam``.
    control_dict_path:
        Optional path to ``system/controlDict``. When supplied, the configured
        ``adjustTimeStep``, ``deltaT``, ``maxDeltaT`` and ``maxCo`` values are
        included and max-Co exceedances are counted.

    Returns
    -------
    dict
        JSON-serializable statistics and raw histories. Missing values are
        represented by ``None`` rather than raising an exception.
    """

    log_path = Path(log_path)
    control_dict_path = (
        Path(control_dict_path)
        if control_dict_path is not None
        else None
    )

    result = {
        "status": "ok",
        "log_path": str(log_path),
        "control_dict_path": (
            str(control_dict_path)
            if control_dict_path is not None
            else None
        ),
        "adjust_time_step": None,
        "configured_delta_t_s": None,
        "configured_max_delta_t_s": None,
        "configured_max_co": None,
        "time_entries": 0,
        "delta_t": {
            "source": None,
            "samples": 0,
            "min_s": None,
            "average_s": None,
            "median_s": None,
            "max_s": None,
            "std_s": None,
        },
        "flow_courant": None,
        "interface_courant": None,
        "mesh_courant": None,
        "history": {
            "time_s": [],
            "delta_t_s": [],
            "flow_co_mean": [],
            "flow_co_max": [],
            "interface_co_mean": [],
            "interface_co_max": [],
            "mesh_co_mean": [],
            "mesh_co_max": [],
        },
    }

    def read_control_entry(text: str, name: str) -> str | None:
        match = re.search(
            rf"^\s*{re.escape(name)}\s+([^\s;]+)\s*;",
            text,
            re.MULTILINE,
        )
        return match.group(1) if match else None

    if control_dict_path is not None and control_dict_path.is_file():
        control_text = control_dict_path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        adjust_value = read_control_entry(
            control_text,
            "adjustTimeStep",
        )
        if adjust_value is not None:
            normalized = adjust_value.strip().lower()
            if normalized in {"yes", "true", "on", "1"}:
                result["adjust_time_step"] = True
            elif normalized in {"no", "false", "off", "0"}:
                result["adjust_time_step"] = False

        for entry_name, result_key in (
            ("deltaT", "configured_delta_t_s"),
            ("maxDeltaT", "configured_max_delta_t_s"),
            ("maxCo", "configured_max_co"),
        ):
            value = read_control_entry(control_text, entry_name)
            if value is not None:
                try:
                    result[result_key] = float(value)
                except ValueError:
                    pass

    if not log_path.is_file():
        result["status"] = "solver log not found"
        return result

    time_values: list[float] = []
    delta_t_values: list[float] = []
    flow_mean_values: list[float] = []
    flow_max_values: list[float] = []
    interface_mean_values: list[float] = []
    interface_max_values: list[float] = []
    mesh_mean_values: list[float] = []
    mesh_max_values: list[float] = []

    time_pattern = re.compile(
        rf"^\s*Time\s*=\s*({_OPENFOAM_NUMBER})\s*$"
    )
    delta_t_pattern = re.compile(
        rf"^\s*deltaT\s*=\s*({_OPENFOAM_NUMBER})\s*$"
    )
    flow_co_pattern = re.compile(
        rf"^\s*Courant Number mean:\s*({_OPENFOAM_NUMBER})"
        rf"\s+max:\s*({_OPENFOAM_NUMBER})"
    )
    interface_co_pattern = re.compile(
        rf"^\s*Interface Courant Number mean:\s*({_OPENFOAM_NUMBER})"
        rf"\s+max:\s*({_OPENFOAM_NUMBER})"
    )
    mesh_co_pattern = re.compile(
        rf"^\s*Mesh Courant Number mean:\s*({_OPENFOAM_NUMBER})"
        rf"\s+max:\s*({_OPENFOAM_NUMBER})"
    )

    with log_path.open("r", encoding="utf-8", errors="ignore") as log_file:
        for line in log_file:
            match = time_pattern.match(line)
            if match:
                time_values.append(float(match.group(1)))
                continue

            match = delta_t_pattern.match(line)
            if match:
                value = float(match.group(1))
                if value > 0.0:
                    delta_t_values.append(value)
                continue

            match = interface_co_pattern.match(line)
            if match:
                interface_mean_values.append(float(match.group(1)))
                interface_max_values.append(float(match.group(2)))
                continue

            match = mesh_co_pattern.match(line)
            if match:
                mesh_mean_values.append(float(match.group(1)))
                mesh_max_values.append(float(match.group(2)))
                continue

            match = flow_co_pattern.match(line)
            if match:
                flow_mean_values.append(float(match.group(1)))
                flow_max_values.append(float(match.group(2)))

    result["time_entries"] = int(len(time_values))

    if delta_t_values:
        delta_t_source = "deltaT entries in solver log"
    else:
        # Fixed-delta-t logs commonly omit "deltaT = ...". Reconstruct the
        # positive increments while ignoring duplicate/restarted time entries.
        reconstructed = []
        for previous_time, current_time in zip(
            time_values[:-1],
            time_values[1:],
        ):
            delta = current_time - previous_time
            if delta > 0.0:
                reconstructed.append(delta)

        delta_t_values = reconstructed
        delta_t_source = (
            "differences between consecutive Time entries"
            if delta_t_values
            else None
        )

    def finite_array(values: list[float]) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        return array[np.isfinite(array)]

    delta_array = finite_array(delta_t_values)
    if delta_array.size:
        result["delta_t"] = {
            "source": delta_t_source,
            "samples": int(delta_array.size),
            "min_s": float(np.min(delta_array)),
            "average_s": float(np.mean(delta_array)),
            "median_s": float(np.median(delta_array)),
            "max_s": float(np.max(delta_array)),
            "std_s": float(np.std(delta_array, ddof=0)),
        }

    def summarize_courant(
        mean_values: list[float],
        max_values: list[float],
    ) -> dict | None:
        mean_array = finite_array(mean_values)
        max_array = finite_array(max_values)

        if mean_array.size == 0 and max_array.size == 0:
            return None

        summary = {
            "samples": int(max(mean_array.size, max_array.size)),
            "mean_co_min": (
                float(np.min(mean_array))
                if mean_array.size
                else None
            ),
            "mean_co_average": (
                float(np.mean(mean_array))
                if mean_array.size
                else None
            ),
            "mean_co_max": (
                float(np.max(mean_array))
                if mean_array.size
                else None
            ),
            "max_co_min": (
                float(np.min(max_array))
                if max_array.size
                else None
            ),
            "max_co_average": (
                float(np.mean(max_array))
                if max_array.size
                else None
            ),
            "peak_max_co": (
                float(np.max(max_array))
                if max_array.size
                else None
            ),
            "configured_max_co_exceedance_count": None,
            "configured_max_co_exceedance_percent": None,
        }

        configured_max_co = result["configured_max_co"]
        if configured_max_co is not None and max_array.size:
            exceedance_count = int(
                np.count_nonzero(max_array > configured_max_co)
            )
            summary["configured_max_co_exceedance_count"] = (
                exceedance_count
            )
            summary["configured_max_co_exceedance_percent"] = (
                100.0 * exceedance_count / max_array.size
            )

        return summary

    result["flow_courant"] = summarize_courant(
        flow_mean_values,
        flow_max_values,
    )
    result["interface_courant"] = summarize_courant(
        interface_mean_values,
        interface_max_values,
    )
    result["mesh_courant"] = summarize_courant(
        mesh_mean_values,
        mesh_max_values,
    )

    result["history"] = {
        "time_s": [float(value) for value in time_values],
        "delta_t_s": [float(value) for value in delta_t_values],
        "flow_co_mean": [float(value) for value in flow_mean_values],
        "flow_co_max": [float(value) for value in flow_max_values],
        "interface_co_mean": [
            float(value)
            for value in interface_mean_values
        ],
        "interface_co_max": [
            float(value)
            for value in interface_max_values
        ],
        "mesh_co_mean": [float(value) for value in mesh_mean_values],
        "mesh_co_max": [float(value) for value in mesh_max_values],
    }

    if (
        result["delta_t"]["samples"] == 0
        and result["flow_courant"] is None
        and result["interface_courant"] is None
        and result["mesh_courant"] is None
    ):
        result["status"] = (
            "no time-step or Courant information found in solver log"
        )

    return result


# ============================================================================
# DOCKER EXECUTION / CASE INTEGRITY
# ============================================================================

def safe_exec(
    container,
    cmd,
    description="command",
    print_output=False,
    status_callback=None,
):
    """
    Execute a Docker command with streamed output and a real exit-code check.

    Only a small tail of command output is retained in Python memory. If a
    command fails, the last non-empty output line is sent to the batch
    dashboard so the user gets more than only an exit code. Full output still
    remains in the command-specific OpenFOAM log files.
    """
    from collections import deque

    output_tail = deque(maxlen=12)

    try:
        container.reload()

        if container.status != "running":
            message = f"Container is not running before {description}"
            emit_status(status_callback, detail=message, error=message)
            if status_callback is None:
                print(message + ".")
            return False

        api = container.client.api
        exec_data = api.exec_create(
            container.id,
            cmd,
            stdout=True,
            stderr=True,
        )
        exec_id = exec_data["Id"]
        output_stream = api.exec_start(exec_id, stream=True)

        for raw_line in output_stream:
            line = raw_line.decode("utf-8", errors="ignore").strip()
            if line:
                output_tail.append(line)
            if print_output and status_callback is None:
                print(line)

        inspect = api.exec_inspect(exec_id)
        exit_code = inspect.get("ExitCode")

        if exit_code != 0:
            last_line = output_tail[-1] if output_tail else "no command output"
            message = (
                f"{description} exited with code {exit_code}: {last_line}"
            )
            emit_status(status_callback, detail=message, error=message)
            if status_callback is None:
                print(message)
            return False

        return True

    except Exception as error:
        message = f"{description} failed: {error}"
        emit_status(
            status_callback,
            detail=message,
            error=message,
        )
        if status_callback is None:
            print(message)
        return False


def processor_deletion_is_safe(
    PATH_TO_CONTROL_DICT_PARAMETERS,
    SIMULATION_DIRECTORY,
    TURBULENCE_MODEL: str,
    RESUME: bool,
) -> bool:

    control_path = Path(PATH_TO_CONTROL_DICT_PARAMETERS)
    sim_dir = Path(SIMULATION_DIRECTORY)

    if not control_path.is_file() or not sim_dir.is_dir():
        return False

    text = control_path.read_text(errors="ignore")

    match = re.search(r"^\s*purgeWrite\s+(\d+)\s*;", text, re.MULTILINE)
    if not match:
        return False

    purge_write = int(match.group(1))

    # Resume only needs ONE valid reconstructed timestep.
    # Normal cleanup should validate all purgeWrite timesteps.
    if RESUME:
        required_number_of_times = 1
    else:
        required_number_of_times = max(purge_write, 1)

    def is_time_folder(path: Path) -> bool:
        if not path.is_dir() or path.name == "0":
            return False

        try:
            float(path.name)
            return True
        except ValueError:
            return False

    time_folders = sorted(
        [p for p in sim_dir.iterdir() if is_time_folder(p)],
        key=lambda p: float(p.name),
    )

    if len(time_folders) < required_number_of_times:
        return False

    folders_to_check = time_folders[-required_number_of_times:]

    # Solver-critical fields
    base_required_files = [
        "U",
        "p",
        "phi",
        "Uf",
        "nut",
    ]

    # For final cleanup validation (not resume),
    # also verify postprocessing fields exist.
    if not RESUME:
        base_required_files += [
            "Q",
            "vorticity",
        ]

    TURBULENCE_MODEL = TURBULENCE_MODEL.strip()

    if TURBULENCE_MODEL == "kEpsilon":
        turbulence_required_files = ["k", "epsilon"]

    elif TURBULENCE_MODEL == "kOmegaSST":
        turbulence_required_files = ["k", "omega"]

    elif TURBULENCE_MODEL == "DES":
            turbulence_required_files = ["k", "omega"]
    

    else:
        return False

    required_files = base_required_files + turbulence_required_files

    def file_is_healthy(file_path: Path) -> bool:

        if not file_path.is_file():
            return False

        if file_path.stat().st_size == 0:
            return False

        content = file_path.read_text(errors="ignore")

        # Basic OpenFOAM field structure checks
        if "FoamFile" not in content:
            return False

        if "dimensions" not in content:
            return False

        if "boundaryField" not in content and "internalField" not in content:
            return False

        return True

    for folder in folders_to_check:

        for filename in required_files:

            if not file_is_healthy(folder / filename):
                return False

    return True

# ============================================================================
# POSTPROCESSING FILE UTILITIES
# ============================================================================

def merge_postprocessing_dat_files(case_dir: Path, function_object_name: str, quiet=False) -> Path | None:
    """
    Merge all .dat files from postProcessing/<function_object_name>/<timeFolder>/ into
    one combined .dat file.

    Example:
        postProcessing/forcesBlades/0/force.dat
        postProcessing/forcesBlades/0.001/force.dat

    Output:
        postProcessing/forcesBlades/merged_force.dat
    """

    function_dir = Path(case_dir) / "postProcessing" / function_object_name

    if not function_dir.exists():
        if not quiet:
            print(f"No postProcessing folder found for: {function_object_name}")
        return None

    dat_files = []

    for time_folder in function_dir.iterdir():
        if not time_folder.is_dir():
            continue

        try:
            start_time = float(time_folder.name)
        except ValueError:
            continue

        for dat_file in time_folder.glob("*.dat"):
            dat_files.append((start_time, dat_file))

    if not dat_files:
        if not quiet:
            print(f"No .dat files found for: {function_object_name}")
        return None

    dat_files.sort(key=lambda item: item[0])

    # Group by filename, e.g. force.dat, residuals.dat, yPlus.dat
    files_by_name = {}

    for start_time, dat_file in dat_files:
        files_by_name.setdefault(dat_file.name, []).append((start_time, dat_file))

    last_output_path = None

    for dat_name, files in files_by_name.items():
        output_path = function_dir / f"merged_{dat_name}"

        header_written = False
        seen_times = set()

        with output_path.open("w") as out_file:
            for _, dat_file in files:
                with dat_file.open("r") as in_file:
                    for line in in_file:
                        stripped = line.strip()

                        if not stripped:
                            continue

                        # Header/comment lines
                        if stripped.startswith("#"):
                            if not header_written:
                                out_file.write(line)
                            continue

                        # Avoid duplicate time rows
                        first_column = stripped.split()[0]

                        try:
                            time_value = float(first_column)
                        except ValueError:
                            continue

                        if time_value in seen_times:
                            continue

                        seen_times.add(time_value)
                        out_file.write(line)

                header_written = True

        #print(f"Merged {function_object_name}: {output_path}")
        last_output_path = output_path

    return last_output_path



# ============================================================================
# SIMULATION ORDER PERSISTENCE / CASE STATE
# ============================================================================

def reset_case_folder(simulation_path: Path, status_callback=None):
    simulation_path = Path(simulation_path)

    if simulation_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        broken_path = simulation_path.with_name(
            simulation_path.name + f"_BROKEN_{timestamp}"
        )
        simulation_path.rename(broken_path)
        emit_status(
            status_callback,
            stage="reset",
            detail=f"Moved broken case to {broken_path.name}",
        )

        if status_callback is None:
            print(f"Moved broken case to: {broken_path}")

    simulation_path.mkdir(parents=True, exist_ok=True)


def make_folder_safe(value: str) -> str:
    return (
        value.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
    )


def load_simulation_order(simulations_directory: Path):
    json_path = Path(simulations_directory) / "simulation_order.json"

    if not json_path.exists():
        raise FileNotFoundError("No simulation_order.json found for resume")

    with _SIMULATION_ORDER_FILE_LOCK:
        with json_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)


def save_simulation_order(simulations_directory: Path, order: dict) -> None:
    """Atomically persist a complete simulation order."""
    json_path = Path(simulations_directory) / "simulation_order.json"
    with _SIMULATION_ORDER_FILE_LOCK:
        _atomic_write_json(json_path, order)


def update_case_status(simulations_directory: Path, folder_name: str, new_status: str):
    """Backward-compatible, concurrency-safe status update helper."""
    json_path = Path(simulations_directory) / "simulation_order.json"

    with _SIMULATION_ORDER_FILE_LOCK:
        with json_path.open("r", encoding="utf-8") as handle:
            batch = json.load(handle)

        found = False
        for case in batch["cases"]:
            if case["folder"] == folder_name:
                case["status"] = new_status
                case["updated_at"] = datetime.now().isoformat(timespec="seconds")
                found = True
                break

        if not found:
            raise KeyError(f"Unknown simulation case: {folder_name}")

        _atomic_write_json(json_path, batch)


def create_simulation_order(args, simulations_directory: Path):
    """Create the durable simulation order including scheduler metadata."""
    simulations_directory = Path(simulations_directory)
    simulations_directory.mkdir(parents=True, exist_ok=True)
    json_path = simulations_directory / "simulation_order.json"

    if json_path.exists():
        raise FileExistsError(
            f"Simulation order already exists in this directory:\n"
            f"{json_path}\n\n"
            f"One simulation order must have its own simulation_run folder. "
            f"Create a new directory or use --resume."
        )

    total_cores = int(args.total_cores)

    batch = {
        "schema_version": 2,
        "acoustic_surface": args.acoustic_surface,
        "acoustic_sphere_diameter": args.acoustic_sphere_diameter,
        "mode": args.mode,
        "turbulence": args.turbulence,
        "meshes": args.meshes,
        "rpms": args.rpms,
        "total_cores": total_cores,
        "field_init": args.field_init,
        "mesh_only": args.mesh_only,
        "end_on": args.end_on,
        "allow_bad_mesh": args.allow_bad_mesh,
        "boundary_layers": args.boundary_layers,
        "study": args.study,
        "study_file": getattr(args, "study_file", None),
        "study_parameter": getattr(args, "study_parameter", None),
        "study_values": getattr(args, "study_values", None),
        "cases": [],
    }

    if args.study:
        mesh = args.meshes[0]
        rpm = args.rpms[0]
        study_values = [
            value.strip()
            for value in args.study_values.split("...")
            if value.strip()
        ]

        for value in study_values:
            safe_value = make_folder_safe(value)
            folder = f"{mesh}_{rpm}RPM_{args.study_parameter}_{safe_value}"

            batch["cases"].append(
                {
                    "folder": folder,
                    "mesh": mesh,
                    "rpm": rpm,
                    "mode": args.mode,
                    "acoustic_surface": args.acoustic_surface,
                    "acoustic_sphere_diameter": args.acoustic_sphere_diameter,
                    "turbulence": args.turbulence,
                    "mesh_only": args.mesh_only,
                    "end_on": args.end_on,
                    "allow_bad_mesh": args.allow_bad_mesh,
                    "field_init": args.field_init,
                    "study": True,
                    "study_file": args.study_file,
                    "study_parameter": args.study_parameter,
                    "study_value": value,
                    "status": "pending",
                    "resume_status": None,
                    "error": None,
                }
            )
    else:
        for mesh in args.meshes:
            for rpm in args.rpms:
                folder = f"{mesh}_{rpm}RPM_{args.mode}"

                batch["cases"].append(
                    {
                        "folder": folder,
                        "mesh": mesh,
                        "rpm": rpm,
                        "mode": args.mode,
                        "turbulence": args.turbulence,
                        "mesh_only": args.mesh_only,
                        "end_on": args.end_on,
                        "acoustic_surface": args.acoustic_surface,
                        "acoustic_sphere_diameter": args.acoustic_sphere_diameter,
                        "allow_bad_mesh": args.allow_bad_mesh,
                        "field_init": args.field_init,
                        "study": False,
                        "study_file": None,
                        "study_parameter": None,
                        "study_value": None,
                        "status": "pending",
                        "resume_status": None,
                        "error": None,
                    }
                )

    add_field_initialization_dependencies(
        batch["cases"],
        field_init=args.field_init,
        study=args.study,
    )

    layout = calculate_scheduler_layout(
        cases=batch["cases"],
        total_cores=total_cores,
        field_init=args.field_init,
        study=args.study,
    )

    batch["cores_per_case"] = layout["cores_per_case"]
    batch["max_cores_per_case"] = layout["max_cores_per_case"]
    batch["extra_core_slots"] = layout["extra_core_slots"]
    batch["max_parallel_cases"] = layout["max_parallel_cases"]

    assign_case_core_allocations(
        batch["cases"],
        layout,
        args.field_init,
        args.study,
    )

    with _SIMULATION_ORDER_FILE_LOCK:
        _atomic_write_json(json_path, batch)

    print(
        f"Created simulation order file: {json_path}\n"
        f"Scheduler: total={layout['total_cores']} cores | "
        f"per-case={layout['cores_per_case']}"
        f"{('-' + str(layout['max_cores_per_case'])) if layout['max_cores_per_case'] != layout['cores_per_case'] else ''} | "
        f"max parallel={layout['max_parallel_cases']}"
    )


def is_mesh_ok(log_path, quiet=False):
    """
    Returns True if 'Mesh OK' is found in log.checkMesh, else False.
    """

    if not log_path.exists():
        if not quiet:
            print("Couldn't confirm mesh is OK because of path error...")
        return False

    log_text = log_path.read_text(errors="ignore")

    return "Mesh OK" in log_text


# ============================================================================
# CONVERGENCE / LIVE PROGRESS MONITORS
# ============================================================================

def check_residuals(
    residuals_file,
    revolution_time,
    use_log=True,
    min_points=10,
    quiet=False,
):
    """
    Returns True if all residuals satisfy slope criteria over the last revolution.

    The fitted regression slope is converted from "per second" to
    "per revolution" by multiplying with revolution_time.

    If use_log=True, the checked quantity is the change in log10(residual)
    over one revolution.
    """
    import builtins

    def _conditional_print(*args, **kwargs):
        if not quiet:
            builtins.print(*args, **kwargs)

    print = _conditional_print

    # SETTINGS
    # Bounds are now interpreted as slope/change OVER ONE REVOLUTION
    slope_bounds = {
        "p":  (-10e-2, 10e-2), #(-5e-2, 1e-2)
        "Ux": (-10e-2, 10e-3), #(-5e-2, 5e-3)
        "Uy": (-10e-2, 10e-3), #(-5e-2, 5e-3)
        "Uz": (-10e-2, 10e-3), #(-5e-2, 5e-3)
        "k":  (-10e-2, 10e-3), #(-5e-2, 5e-3)
    }
    ###

    # Read header explicitly from second line
    with open(residuals_file, "r") as f:
        lines = f.readlines()

    if len(lines) < 3:
        raise ValueError("Residual file is too short.")

    header = lines[1].lstrip("#").strip().split()

    df = pd.read_csv(
        residuals_file,
        sep=r"\s+",
        names=header,
        skiprows=2,
        na_values=["N/A"],
        engine="python",
    )

    if "Time" not in df.columns:
        raise ValueError("Residual file must contain a 'Time' column.")

    df = df.dropna(subset=["Time"]).sort_values("Time")

    if df.empty:
        raise ValueError("Residual file contains no valid data.")

    latest_time = df["Time"].iloc[-1]

    if latest_time <= revolution_time:
        print("Failed: not enough data for one full revolution.")
        return False

    # Last revolution window
    t_start = latest_time - revolution_time
    window_df = df[df["Time"] >= t_start].copy()

    if window_df.empty:
        print("Failed: no data in last revolution window.")
        return False

    failed_fields = []

    for field, bounds in slope_bounds.items():

        if field not in window_df.columns:
            failed_fields.append(field)
            continue

        if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
            raise ValueError(
                f"Bounds for '{field}' must be (lower_bound, upper_bound)."
            )

        lower_bound, upper_bound = bounds

        data = window_df[["Time", field]].dropna()

        if len(data) < min_points:
            failed_fields.append(field)
            continue

        t = data["Time"].to_numpy(dtype=float)
        y = data[field].to_numpy(dtype=float)

        if use_log:
            mask = y > 0.0
            t = t[mask]
            y = y[mask]

            if len(y) < min_points:
                failed_fields.append(field)
                continue

            y = np.log10(y)

        slope_per_second, _ = np.polyfit(t, y, 1)
        slope_per_revolution = slope_per_second * revolution_time

        if not (lower_bound <= slope_per_revolution <= upper_bound):
            failed_fields.append(field)

    # DEBUGGING ONLY
    print("\n--- Residual slopes per revolution (debug) ---")

    for field in slope_bounds.keys():

        if field not in window_df.columns:
            print(f"{field}: not found")
            continue

        data = window_df[["Time", field]].dropna()

        if len(data) < min_points:
            print(f"{field}: not enough data")
            continue

        t = data["Time"].to_numpy(dtype=float)
        y = data[field].to_numpy(dtype=float)

        if use_log:
            mask = y > 0.0
            t = t[mask]
            y = y[mask]

            if len(y) < min_points:
                print(f"{field}: not enough valid data after log filter")
                continue

            y = np.log10(y)

        slope_per_second, _ = np.polyfit(t, y, 1)
        slope_per_revolution = slope_per_second * revolution_time

        print(f"{field}: slope per revolution = {slope_per_revolution:.3e}")

    # END OF DEBUGGING

    if len(failed_fields) == 0:
        print("Passed: all residual slope checks satisfied.")
        return True
    else:
        print(f"Failed: residual slope check failed for {failed_fields}.")
        return False




def run_reconstruction_progress_monitor(
    main_sim_folder,
    maximum_time=None,
    check_interval=2.0,
    stop_event=None,
    status_callback=None,
):
    """Monitor reconstructPar from filesystem time directories."""
    case_path = Path(main_sim_folder)
    maximum_time_value = float(maximum_time) if maximum_time is not None else None

    def numeric_times(directory):
        values = set()
        directory = Path(directory)
        if not directory.is_dir():
            return values

        for path in directory.iterdir():
            if not path.is_dir():
                continue
            try:
                value = float(path.name)
            except ValueError:
                continue
            if value <= 0.0:
                continue
            if maximum_time_value is None or value <= maximum_time_value + 1e-12:
                values.add(round(value, 12))
        return values

    def find_reference_processor():
        processor0 = case_path / "processor0"
        if processor0.is_dir():
            return processor0
        processor_directories = sorted(
            path for path in case_path.glob("processor*") if path.is_dir()
        )
        return processor_directories[0] if processor_directories else None

    try:
        reference_processor = find_reference_processor()

        while reference_processor is None:
            emit_status(
                status_callback,
                stage="reconstructing",
                detail="waiting for processor directories",
                progress=0.0,
            )
            if stop_event is not None and stop_event.wait(check_interval):
                return None
            if stop_event is None:
                time.sleep(check_interval)
            reference_processor = find_reference_processor()

        expected_times = numeric_times(reference_processor)

        while not expected_times:
            emit_status(
                status_callback,
                stage="reconstructing",
                detail="waiting for processor time directories",
                progress=0.0,
            )
            if stop_event is not None and stop_event.wait(check_interval):
                return None
            if stop_event is None:
                time.sleep(check_interval)
            expected_times = numeric_times(reference_processor)

        expected_count = len(expected_times)

        while True:
            stopping = stop_event is not None and stop_event.is_set()
            reconstructed_times = numeric_times(case_path)
            completed_times = expected_times.intersection(reconstructed_times)
            completed_count = len(completed_times)
            percentage = min(max(100.0 * completed_count / expected_count, 0.0), 100.0)
            latest_time = max(completed_times) if completed_times else 0.0

            detail = (
                f"{completed_count}/{expected_count} time dirs | "
                f"latest {latest_time:.6f} s"
            )
            emit_status(
                status_callback,
                stage="reconstructing",
                detail=detail,
                progress=percentage,
            )

            if status_callback is None:
                print(
                    f"\rRECONSTRUCT | {detail} | {percentage:6.2f}%",
                    end="",
                    flush=True,
                )

            result = {
                "completed": completed_count,
                "expected": expected_count,
                "percentage": percentage,
                "latest_time": latest_time,
            }

            if completed_count >= expected_count or stopping:
                if status_callback is None:
                    print(flush=True)
                return result

            if stop_event is not None:
                stop_event.wait(check_interval)
            else:
                time.sleep(check_interval)

    except Exception as error:
        emit_status(
            status_callback,
            stage="reconstructing",
            detail=f"progress monitor failed: {error}",
            error=str(error),
        )
        if status_callback is None:
            print(f"Reconstruction progress monitor failed: {error}")
        return None


def run_time_progress_monitor(
    main_sim_folder,
    end_time,
    check_interval=5.0,
    log_file_name="log.pimpleFoam",
    stop_event=None,
    status_callback=None,
):
    """Monitor physical-time solver progress by incrementally reading the log."""
    log_path = Path(main_sim_folder) / log_file_name
    end_time = float(end_time)

    if end_time <= 0.0:
        raise ValueError("end_time must be greater than zero.")

    time_pattern = re.compile(rf"\bTime\s*=\s*({_OPENFOAM_NUMBER})")
    latest_time = 0.0
    file_position = 0
    unfinished_line = ""

    try:
        while True:
            stopping = stop_event is not None and stop_event.is_set()

            if log_path.is_file():
                current_size = log_path.stat().st_size
                if current_size < file_position:
                    file_position = 0
                    unfinished_line = ""

                with log_path.open("r", encoding="utf-8", errors="ignore") as log_file:
                    log_file.seek(file_position)
                    new_text = log_file.read()
                    file_position = log_file.tell()

                if new_text:
                    combined_text = unfinished_line + new_text
                    lines = combined_text.splitlines(keepends=True)
                    if lines and not lines[-1].endswith(("\n", "\r")):
                        unfinished_line = lines.pop()
                    else:
                        unfinished_line = ""

                    for line in lines:
                        match = time_pattern.search(line)
                        if match:
                            latest_time = float(match.group(1))

            percentage = min(max(100.0 * latest_time / end_time, 0.0), 100.0)
            detail = f"t={latest_time:.6f}/{end_time:.6f} s"
            emit_status(
                status_callback,
                stage="solving",
                detail=detail if log_path.is_file() else "waiting for solver log",
                progress=percentage,
            )

            if status_callback is None:
                print(
                    f"\rPIMPLE | {detail} | {percentage:6.2f}%",
                    end="",
                    flush=True,
                )

            if latest_time >= end_time - 1e-12:
                if status_callback is None:
                    print(flush=True)
                return latest_time

            if stopping:
                if status_callback is None:
                    print(flush=True)
                return latest_time

            if stop_event is not None:
                stop_event.wait(check_interval)
            else:
                time.sleep(check_interval)

    except Exception as error:
        emit_status(
            status_callback,
            stage="solving",
            detail=f"time-progress monitor failed: {error}",
            error=str(error),
        )
        if status_callback is None:
            print(f"Time-progress monitor failed: {error}")
        return None


def run_convergence_monitor(
    main_sim_folder,
    rpm,
    avg_history_count,
    tolerance,
    check_interval,
    timestep: str,
    convergence_mode: str = "convergence",
    stop_event=None,
    status_callback=None,
):
    """
    Monitor convergence and stop the OpenFOAM simulation by reducing endTime.

    convergence_mode options:
        "force_convergence"     -> stop when rolling 1-revolution thrust averages are stable
        "residual_convergence" -> stop when residual slopes over the last revolution are stable
        "convergence"      -> stop only after force convergence AND residual convergence

    In all modes, at least one full revolution of data is required before any
    convergence decision is made.
    """
    import builtins

    def _monitor_print(*args, **kwargs):
        message = " ".join(str(arg) for arg in args).strip()
        if status_callback is not None:
            emit_status(
                status_callback,
                stage="solving",
                detail=message[:120],
                progress=None,
            )
        else:
            builtins.print(*args, **kwargs)

    print = _monitor_print

    convergence_mode = convergence_mode.strip().lower()

    if convergence_mode not in {"force_convergence", "residual_convergence", "convergence"}:
        raise ValueError(
            "convergence_mode must be one of: 'force_convergence', 'residual_convergence', or 'convergence'."
        )

    force_file = os.path.join(
        main_sim_folder, "postProcessing", "forcesBlades", timestep, "forces.dat"
    )
    yplus_file = os.path.join(
        main_sim_folder, "postProcessing", "yPlus", timestep, "yPlus.dat"
    )
    residuals_file = os.path.join(
        main_sim_folder, "postProcessing", "residuals", timestep, "residuals.dat"
    )
    control_dict = os.path.join(main_sim_folder, "system", "controlDict")

    rev_time = 60.0 / rpm

    def should_stop_monitor() -> bool:
        return stop_event is not None and stop_event.is_set()

    def sleep_or_stop() -> bool:
        """
        Returns True if the monitor should stop, False if normal sleep finished.
        """
        if stop_event is not None:
            if stop_event.wait(check_interval):
                print("Convergence monitor stopped by main simulation.")
                return True
            return False

        time.sleep(check_interval)
        return False

    def get_control_dict_end_time(control_dict_path):
        if not os.path.exists(control_dict_path):
            return None

        with open(control_dict_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip().startswith("endTime"):
                    try:
                        return float(line.split()[1].replace(";", ""))
                    except Exception:
                        return None
        return None

    def set_control_dict_end_time(stop_time: float) -> bool:
        if not os.path.exists(control_dict):
            print(f"ERROR: controlDict not found at: {control_dict}")
            return False

        with open(control_dict, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        with open(control_dict, "w", encoding="utf-8") as f:
            for line in lines:
                if re.match(r"^\s*endTime\s+", line):
                    f.write(f"endTime         {stop_time + 1e-8};\n")
                else:
                    f.write(line)

        print("Simulation stop command sent to controlDict.")
        return True

    def read_latest_time_from_residuals():
        if not os.path.exists(residuals_file):
            return None

        latest_time = None

        with open(residuals_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()

                try:
                    latest_time = float(parts[0])
                except (ValueError, IndexError):
                    continue

        return latest_time

    def read_force_data():
        if not os.path.exists(force_file):
            return None, None

        times = []
        thrusts = []

        with open(force_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.replace("(", " ").replace(")", " ").split()

                if len(parts) < 3:
                    continue

                try:
                    t = float(parts[0])
                    thrust_y = float(parts[2])
                except ValueError:
                    continue

                times.append(t)
                thrusts.append(thrust_y)

        if not times:
            return None, None

        times = np.asarray(times, dtype=float)
        thrusts = np.asarray(thrusts, dtype=float)

        sort_idx = np.argsort(times)
        return times[sort_idx], thrusts[sort_idx]

    def get_yplus_stats(rev_window_start: float, rev_window_end: float):
        avg_yplus = float("nan")
        max_yplus = float("nan")
        min_yplus = float("nan")

        if os.path.exists(yplus_file):
            with open(yplus_file, "r", encoding="utf-8", errors="ignore") as f:
                yplus_lines = [
                    l.strip()
                    for l in f
                    if l.strip() and not l.strip().startswith("#")
                ]

            yplus_min_vals = []
            yplus_max_vals = []
            yplus_avg_vals = []

            for line in yplus_lines:
                parts = line.split()

                if len(parts) >= 5 and parts[1] == "propellerTip":
                    try:
                        t = float(parts[0])
                        y_min = float(parts[2])
                        y_max = float(parts[3])
                        y_avg = float(parts[4])
                    except ValueError:
                        continue

                    if rev_window_start <= t <= rev_window_end:
                        yplus_min_vals.append(y_min)
                        yplus_max_vals.append(y_max)
                        yplus_avg_vals.append(y_avg)

            if yplus_avg_vals:
                avg_yplus = float(np.mean(yplus_avg_vals))
                max_yplus = float(np.max(yplus_max_vals))
                min_yplus = float(np.min(yplus_min_vals))

        return avg_yplus, max_yplus, min_yplus

    print(f"RPM: {rpm}")
    print(f"One revolution time: {rev_time:.6f} s")
    print(f"Convergence mode: {convergence_mode}")

    while True:
        if should_stop_monitor():
            print("Convergence monitor stopped by main simulation.")
            return False

        try:
            # -----------------------------------------------------------------
            # RESIDUALS-ONLY MODE
            # -----------------------------------------------------------------
            if convergence_mode == "residual_convergence":
                latest_time = read_latest_time_from_residuals()

                if latest_time is None:
                    print("Waiting for residuals file to be created or filled...")
                    if sleep_or_stop():
                        return False
                    continue

                end_time = get_control_dict_end_time(control_dict)

                if end_time is not None and latest_time >= end_time - 1e-8:
                    print(
                        f"\n>>> Simulation reached endTime={end_time} "
                        f"without convergence <<<"
                    )
                    return False

                if latest_time < rev_time:
                    print(
                        f"Waiting for enough data: {latest_time:.4f}/{rev_time:.4f} s "
                        f"({latest_time / rev_time:.2f}/1.00 rev)"
                    )
                    if sleep_or_stop():
                        return False
                    continue

                avg_yplus, max_yplus, min_yplus = get_yplus_stats(
                    latest_time - rev_time,
                    latest_time,
                )

                print(
                    f"Time: {latest_time:.4f} | "
                    f"Checking residual convergence over last revolution | "
                    f"Avg y+: {avg_yplus:.2f} | "
                    f"Max y+: {max_yplus:.2f} | "
                    f"Min y+: {min_yplus:.2f}"
                )

                if check_residuals(residuals_file, rev_time, quiet=status_callback is not None):
                    print(
                        f"\n>>> SUFFICIENT RESIDUAL CONVERGENCE "
                        f"REACHED AT {latest_time}s <<<"
                    )

                    return set_control_dict_end_time(latest_time)

                if sleep_or_stop():
                    return False
                continue

            # -----------------------------------------------------------------
            # FORCE OR BOTH MODE
            # -----------------------------------------------------------------
            times, thrusts = read_force_data()

            if times is None:
                print("Waiting for force file to be created or filled...")
                if sleep_or_stop():
                    return False
                continue

            latest_time = float(times[-1])

            # Check endTime immediately after latest solver time is known.
            # Otherwise the monitor may wait forever if endTime is reached
            # before enough rolling averages exist.
            end_time = get_control_dict_end_time(control_dict)

            if end_time is not None and latest_time >= end_time - 1e-8:
                print(
                    f"\n>>> Simulation reached endTime={end_time} "
                    f"without convergence <<<"
                )
                return False

            if latest_time < rev_time:
                print(
                    f"Waiting for enough data: {latest_time:.4f}/{rev_time:.4f} s "
                    f"({latest_time / rev_time:.2f}/1.00 rev)"
                )
                if sleep_or_stop():
                    return False
                continue

            csum = np.concatenate(([0.0], np.cumsum(thrusts)))
            avg_times = []
            avg_vals = []

            for i in range(len(times)):
                t_end = times[i]
                t_start = t_end - rev_time

                if t_start < 0.0:
                    continue

                j = np.searchsorted(times, t_start, side="left")
                count = i - j + 1

                if count <= 0:
                    continue

                window_sum = csum[i + 1] - csum[j]
                avg_val = window_sum / count

                avg_times.append(t_end)
                avg_vals.append(avg_val)

            if not avg_vals:
                print("No valid rolling 1-rev averages available yet.")
                if sleep_or_stop():
                    return False
                continue

            avg_times = np.asarray(avg_times, dtype=float)
            avg_vals = np.asarray(avg_vals, dtype=float)

            latest_sim_time = float(avg_times[-1])
            current_avg_thrust = float(avg_vals[-1])

            rev_window_start = latest_sim_time - rev_time
            rev_window_end = latest_sim_time

            avg_yplus, max_yplus, min_yplus = get_yplus_stats(
                rev_window_start,
                rev_window_end,
            )

            if len(avg_vals) < avg_history_count:
                print(
                    f"Time: {latest_sim_time:.4f} | "
                    f"Current 1-rev Avg Thrust: {current_avg_thrust:.4f} | "
                    f"Waiting for enough averaged values: "
                    f"{len(avg_vals)} / {avg_history_count} | "
                    f"Avg y+: {avg_yplus:.2f} | "
                    f"Max y+: {max_yplus:.2f} | "
                    f"Min y+: {min_yplus:.2f}"
                )
                if sleep_or_stop():
                    return False
                continue

            avg_thrust_history = avg_vals[-avg_history_count:]
            std_dev = float(np.std(avg_thrust_history))
            avg_val = float(np.mean(avg_thrust_history))

            force_converged = std_dev < tolerance

            print(
                f"Time: {latest_sim_time:.4f} | "
                f"Current 1-rev Avg Thrust: {current_avg_thrust:.4f} | "
                f"Avg Thrust: {avg_val:.4f} | "
                f"StdDev(rolling 1-rev avgs): {std_dev:.6f} | "
                f"Force converged: {force_converged} | "
                f"Avg y+: {avg_yplus:.2f} | "
                f"Max y+: {max_yplus:.2f} | "
                f"Min y+: {min_yplus:.2f}"
            )

            if convergence_mode == "force_convergence":
                if force_converged:
                    print(
                        f"\n>>> SUFFICIENT FORCE CONVERGENCE "
                        f"REACHED AT {latest_sim_time}s <<<"
                    )

                    return set_control_dict_end_time(latest_sim_time)

            elif convergence_mode == "convergence":
                # Keep the previous logic: residuals are checked only after
                # force convergence has first been reached.
                if force_converged:
                    if check_residuals(residuals_file, rev_time, quiet=status_callback is not None):
                        print(
                            f"\n>>> SUFFICIENT FORCE AND RESIDUAL "
                            f"CONVERGENCE REACHED AT {latest_sim_time}s <<<"
                        )

                        return set_control_dict_end_time(latest_sim_time)
                else:
                    print("Force convergence not reached yet; residuals not checked.")

        except Exception as e:
            print(f"Error during monitoring: {e}")

        if sleep_or_stop():
            return False

# ============================================================================
# GENERAL CASE / PARAMETER UTILITIES
# ============================================================================

def get_latest_timestep(case_path):
    case_path = Path(case_path)

    time_dirs = []

    for item in case_path.iterdir():
        if item.is_dir():
            try:
                time_value = float(item.name)

                # Skip the initial "0" folder
                if time_value == 0.0:
                    continue

                time_dirs.append((time_value, item.name))

            except ValueError:
                pass

    if not time_dirs:
        raise FileNotFoundError(f"No time directories found in {case_path}")

    latest_time, latest_name = max(time_dirs, key=lambda x: x[0])
    return latest_time, latest_name

def has_timestep(case_path):
    try:
        get_latest_timestep(case_path)
        return True
    except FileNotFoundError:
        return False
    
def get_safe_timestep(case_dir: Path, required_fields=("U", "p")):
    """
    Returns safest timestep for resume:
    - Uses processor0 if parallel case exists
    - Falls back to case root if serial
    - Ignores timestep 0
    - Checks required fields exist
    - Picks newest valid timestep
    """

    # detect processor folders
    processor_dirs = sorted(case_dir.glob("processor*"))

    if processor_dirs:
        base_dir = processor_dirs[0]  # use processor0 as reference
    else:
        base_dir = case_dir

    times = []

    # collect numeric timestep folders
    for path in base_dir.iterdir():
        if not path.is_dir():
            continue

        try:
            t = float(path.name)
        except ValueError:
            continue

        if t > 0:
            times.append(t)

    if not times:
        return None

    times = sorted(times)

    # iterate newest → oldest
    for t in reversed(times):
        time_dir = base_dir / f"{t:.10g}"

        valid = True
        for field in required_fields:
            if not (time_dir / field).exists():
                valid = False
                break

        if valid:
            return t

    return None

def update_parameter(file_path, target_var, new_value, quiet=False):
    if not os.path.exists(file_path):
        if not quiet:
            print(f"Error: {file_path} not found.")
        return False

    lines = []
    updated = False

    with open(file_path, "r") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) >= 2 and parts[0] == target_var:
                lines.append(f"{target_var} {new_value};\n")
                updated = True
            else:
                lines.append(line)

    if updated:
        with open(file_path, "w") as handle:
            handle.writelines(lines)

        if not quiet:
            print(f"Successfully updated {target_var} to {new_value}.")
        return True

    if not quiet:
        print(f"Variable '{target_var}' not found in the file.")
    return False


def ensure_case_core_configuration(
    simulation_directory: Path,
    allocated_cores: int,
) -> None:
    """Ensure decomposePar uses the scheduler's core count for this case."""
    allocated_cores = int(allocated_cores)
    if allocated_cores < 1:
        raise ValueError("allocated_cores must be at least 1")

    parameter_file = (
        Path(simulation_directory)
        / "Parameters"
        / "decomposeParDict.cpp"
    )

    if not parameter_file.is_file():
        raise FileNotFoundError(
            f"decomposePar parameter file not found: {parameter_file}"
        )

    if not update_parameter(
        parameter_file,
        "numberOfSubdomains",
        allocated_cores,
        quiet=True,
    ):
        raise ValueError(
            f"numberOfSubdomains was not found in {parameter_file}"
        )



def create_reference_geometry_vtk_series(
    source_directory: Path,
    output_directory: Path,
    surface_file: str = "cubeWall.vtk",
) -> Path:
    time_directories = sorted(
        (
            path
            for path in source_directory.iterdir()
            if path.is_dir()
        ),
        key=lambda path: float(path.name),
    )

    if not time_directories:
        raise FileNotFoundError(
            f"No timestep directories found in {source_directory}"
        )

    reference_path = time_directories[0] / surface_file
    reference_data = reference_path.read_bytes()

    def split_vtk(data: bytes) -> tuple[bytes, bytes, bytes]:
        points_match = re.search(
            rb"(?m)^POINTS\s+\d+\s+\S+\s*$",
            data,
        )
        field_match = re.search(
            rb"(?m)^(?:CELL_DATA|POINT_DATA)\s+\d+\s*$",
            data,
        )

        if points_match is None or field_match is None:
            raise ValueError("Unsupported VTK file structure")

        header = data[:points_match.start()]
        geometry = data[points_match.start():field_match.start()]
        fields = data[field_match.start():]

        return header, geometry, fields

    _, reference_geometry, _ = split_vtk(reference_data)

    for time_directory in time_directories:
        source_path = time_directory / surface_file
        current_data = source_path.read_bytes()

        current_header, _, current_fields = split_vtk(current_data)

        target_directory = output_directory / time_directory.name
        target_directory.mkdir(parents=True, exist_ok=True)

        target_path = target_directory / surface_file
        target_path.write_bytes(
            current_header
            + reference_geometry
            + current_fields
        )

    return output_directory


# ============================================================================
# SIMULATION REPORT HELPERS
# ============================================================================
def create_yplus_distribution_plot(case_path, report_dir, patch_name="cubeWall"):
    import matplotlib.pyplot as plt


    def get_latest_time_dir(case_path):
        time_dirs = []

        for item in case_path.iterdir():
            if item.is_dir():
                try:
                    time_dirs.append((float(item.name), item))
                except ValueError:
                    pass

        if not time_dirs:
            return None

        return max(time_dirs, key=lambda x: x[0])[1]

    latest_time_dir = get_latest_time_dir(case_path)

    if latest_time_dir is None:
        return None, None

    yplus_file = latest_time_dir / "yPlus"

    if not yplus_file.exists():
        return None, None

    text = yplus_file.read_text(encoding="utf-8", errors="ignore")

    patch_pattern = rf"{re.escape(patch_name)}\s*\{{(.*?)\}}"
    patch_match = re.search(patch_pattern, text, re.DOTALL)

    if not patch_match:
        return None, None

    patch_block = patch_match.group(1)

    list_pattern = r"nonuniform\s+List<scalar>\s*(\d+)\s*\((.*?)\)"
    list_match = re.search(list_pattern, patch_block, re.DOTALL)

    if not list_match:
        return None, None

    values_block = list_match.group(2)

    yplus_values = np.array(
        [
            float(v)
            for v in re.findall(
                r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
                values_block,
            )
        ],
        dtype=float,
    )

    if len(yplus_values) == 0:
        return None, None

    total = len(yplus_values)

    # Compact wall-function quality classes
    class_counts = [
        int(np.sum(yplus_values < 5)),
        int(np.sum((yplus_values >= 5) & (yplus_values <= 30))),
        int(np.sum(yplus_values > 30)),
    ]
    class_percentages = [100.0 * c / total for c in class_counts]

    # Finer block diagram to make the high-y+ region visible
    block_bins = [0.0, 5.0, 30.0, 50.0, 100.0, 200.0, np.inf]
    block_labels = ["<5", "5-30", "30-50", "50-100", "100-200", ">200"]
    block_counts = []

    for lower, upper in zip(block_bins[:-1], block_bins[1:]):
        if np.isinf(upper):
            count = np.sum(yplus_values >= lower)
        elif lower == 0.0:
            count = np.sum(yplus_values < upper)
        else:
            count = np.sum((yplus_values >= lower) & (yplus_values < upper))
        block_counts.append(int(count))

    block_percentages = [100.0 * c / total for c in block_counts]

    yplus_stats = {
        "patch_name": patch_name,
        "time_dir": latest_time_dir.name,
        "n_faces": int(total),
        "average_yplus": float(np.mean(yplus_values)),
        "min_yplus": float(np.min(yplus_values)),
        "max_yplus": float(np.max(yplus_values)),
        "median_yplus": float(np.median(yplus_values)),
        "share_yplus_lt_5_percent": class_percentages[0],
        "share_yplus_5_to_30_percent": class_percentages[1],
        "share_yplus_gt_30_percent": class_percentages[2],
    }

    yplus_plot = report_dir / "yplus_distribution.png"

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    bars = ax.bar(block_labels, block_percentages, zorder=3)

    ax.set_ylabel("Surface face share [%]")
    ax.set_xlabel("y+ interval")
    ax.set_title(
        f"y+ Distribution of Propeller Surface "
        f"(avg. y+ = {yplus_stats['average_yplus']:.1f})"
    )

    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)

    # Extra vertical space prevents labels from touching the top frame or gridlines.
    max_percentage = max(block_percentages)
    ax.set_ylim(0, max_percentage * 1.18 + 3)

    # Labels are shifted above each bar and placed on a white background,
    # so the grid does not reduce readability.
    for bar, percentage, count in zip(bars, block_percentages, block_counts):
        label_y = bar.get_height() + 1.5

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            label_y,
            f"{percentage:.1f}%\n({count})",
            ha="center",
            va="bottom",
            fontsize=8,
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.9,
                pad=1.5,
            ),
            clip_on=False,
            zorder=5,
        )

    note = (
        f"Classes: <5 = {class_percentages[0]:.1f}%, "
        f"5-30 = {class_percentages[1]:.1f}%, "
        f">30 = {class_percentages[2]:.1f}%"
    )
    fig.text(0.5, 0.015, note, ha="center", fontsize=9)

    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(yplus_plot, dpi=200)
    plt.close(fig)

    return yplus_plot, yplus_stats


def read_mesh_element_types(case_path):
    log_checkmesh = case_path / "log.checkMesh"

    element_types = {
        "hexahedra": 0,
        "prisms": 0,
        "wedges": 0,
        "pyramids": 0,
        "tet wedges": 0,
        "tetrahedra": 0,
        "polyhedra": 0,
    }

    if not log_checkmesh.exists():
        return element_types

    text = log_checkmesh.read_text(encoding="utf-8", errors="ignore")

    patterns = {
        "hexahedra": r"hexahedra:\s*([0-9]+)",
        "prisms": r"prisms:\s*([0-9]+)",
        "wedges": r"wedges:\s*([0-9]+)",
        "pyramids": r"pyramids:\s*([0-9]+)",
        "tet wedges": r"tet wedges:\s*([0-9]+)",
        "tetrahedra": r"tetrahedra:\s*([0-9]+)",
        "polyhedra": r"polyhedra:\s*([0-9]+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            element_types[key] = int(match.group(1))

    return element_types


def create_mesh_element_plot(element_types, report_dir):
    import matplotlib.pyplot as plt

    nonzero = {
        key: value
        for key, value in element_types.items()
        if value > 0
    }

    if not nonzero:
        return None

    total = sum(nonzero.values())

    labels = list(nonzero.keys())
    values = [100.0 * value / total for value in nonzero.values()]

    mesh_plot = report_dir / "mesh_element_types.png"

    plt.figure(figsize=(5.0, 3.4))
    bars = plt.bar(labels, values)

    plt.ylabel("Cell share [%]")
    plt.title("Mesh Element Types")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y")

    # --- Add percentage labels on bars ---
    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(mesh_plot, dpi=200)
    plt.close()

    return mesh_plot


def read_mesh_information(case_path):
    log_checkmesh = case_path / "log.checkMesh"

    mesh_info = {
        "mesh_ok": False,
        "cells": None,
        "faces": None,
        "points": None,
        "boundary_patches": None,
        "max_aspect_ratio": None,
        "max_skewness": None,
        "max_non_orthogonality": None,
    }

    if not log_checkmesh.exists():
        mesh_info["status"] = "log.checkMesh not found"
        return mesh_info

    text = log_checkmesh.read_text(encoding="utf-8", errors="ignore")

    mesh_info["mesh_ok"] = "Mesh OK" in text
    mesh_info["status"] = "Mesh OK" if mesh_info["mesh_ok"] else "Mesh check failed / not confirmed"

    patterns = {
        "points": r"points:\s*([0-9]+)",
        "faces": r"faces:\s*([0-9]+)",
        "cells": r"cells:\s*([0-9]+)",
        "boundary_patches": r"boundary patches:\s*([0-9]+)",
        "max_aspect_ratio": r"Max aspect ratio\s*=\s*([0-9.eE+-]+)",
        "max_skewness": r"Max skewness\s*=\s*([0-9.eE+-]+)",
        "max_non_orthogonality": r"Mesh non-orthogonality Max:\s*([0-9.eE+-]+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            value = match.group(1)
            try:
                mesh_info[key] = float(value) if "." in value or "e" in value.lower() else int(value)
            except ValueError:
                mesh_info[key] = value

    return mesh_info


def format_seconds(seconds):
    if seconds is None:
        return "Not found"

    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    if h > 0:
        return f"{h} h {m} min {s} s"
    if m > 0:
        return f"{m} min {s} s"
    return f"{s} s"


def format_optional_number(value, format_spec=".6e"):
    if value is None:
        return "Not found"

    try:
        return format(float(value), format_spec)
    except (TypeError, ValueError):
        return str(value)


def format_optional_bool(value):
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "Not found"


def evaluate_thrust_convergence(times, thrusts, rev_time, threshold=1e-3):
    latest_time = float(times[-1])
    last_rev_start = latest_time - rev_time
    idx_start = np.searchsorted(times, last_rev_start, side="left")

    window_times = times[idx_start:]
    window_thrusts = thrusts[idx_start:]

    if len(window_thrusts) == 0:
        return {
            "passed": False,
            "reason": "No thrust samples found in final revolution window.",
            "window_start_s": last_rev_start,
            "window_end_s": latest_time,
            "mean_N": None,
            "std_N": None,
            "relative_std": None,
            "threshold": threshold,
            "n_samples": 0,
        }

    mean_thrust = float(np.mean(window_thrusts))
    std_thrust = float(np.std(window_thrusts, ddof=0))
    relative_std = std_thrust / max(abs(mean_thrust), 1e-12)

    return {
        "passed": bool(relative_std < threshold),
        "reason": None,
        "window_start_s": float(window_times[0]),
        "window_end_s": latest_time,
        "mean_N": mean_thrust,
        "std_N": std_thrust,
        "relative_std": float(relative_std),
        "threshold": float(threshold),
        "n_samples": int(len(window_thrusts)),
    }


def evaluate_moments(times, moments, rev_time):
    latest_time = float(times[-1])
    last_rev_start = latest_time - rev_time
    idx_start = np.searchsorted(times, last_rev_start, side="left")

    window_times = times[idx_start:]
    window_moments = moments[idx_start:]

    if len(window_moments) == 0:
        return {
            "passed": False,
            "reason": "No moments samples found in final revolution window.",
            "window_start_s": last_rev_start,
            "window_end_s": latest_time,
            "mean_N": None,
            "std_N": None,
            "relative_std": None
        }

    mean_moment = float(np.mean(window_moments))
    std_moment = float(np.std(window_moments, ddof=0))
    relative_std = std_moment / max(abs(mean_moment), 1e-12)

    return {
        "passed": bool,
        "reason": None,
        "window_start_s": float(window_times[0]),
        "window_end_s": latest_time,
        "mean_N": mean_moment,
        "std_N": std_moment,
        "relative_std": float(relative_std)
    }


def compute_thrust_stability_history(times, thrusts, rev_time):
    """
    Computes the relative thrust fluctuation over a sliding one-revolution window.

    metric(t) = std(F_window) / |mean(F_window)|

    The value at time t uses all force samples within [t - T_rev, t].
    """
    metric = np.full(len(times), np.nan, dtype=float)
    window_mean = np.full(len(times), np.nan, dtype=float)
    window_std = np.full(len(times), np.nan, dtype=float)
    sample_count = np.zeros(len(times), dtype=int)

    for i, time_value in enumerate(times):
        window_start = time_value - rev_time

        if window_start < times[0]:
            continue

        j = np.searchsorted(times, window_start, side="left")
        window_values = thrusts[j : i + 1]

        if len(window_values) < 2:
            continue

        mean_value = float(np.mean(window_values))
        std_value = float(np.std(window_values, ddof=0))

        window_mean[i] = mean_value
        window_std[i] = std_value
        metric[i] = std_value / max(abs(mean_value), 1e-12)
        sample_count[i] = int(len(window_values))

    return {
        "time": times,
        "relative_std": metric,
        "mean_N": window_mean,
        "std_N": window_std,
        "sample_count": sample_count,
    }


def create_force_plots(times, thrusts, report_dir, rev_time, thrust_convergence):
    import matplotlib.pyplot as plt

    force_plot = report_dir / "force_plot.png"
    conv_plot = report_dir / "force_convergence.png"

    latest_time = float(times[-1])
    last_rev_start = latest_time - rev_time

    # -----------------------------
    # Force history plot
    # -----------------------------
    # The raw force plot is kept as a general overview. Extreme initialization
    # spikes are excluded only from the axis scaling, not from the data itself.
    plot_mask = times > 0.001
    plot_thrusts = thrusts[plot_mask]

    if len(plot_thrusts) > 0:
        y_min = np.percentile(plot_thrusts, 1)
        y_max = np.percentile(plot_thrusts, 99)
        y_margin = 0.15 * max(y_max - y_min, 1e-12)
    else:
        y_min, y_max = np.min(thrusts), np.max(thrusts)
        y_margin = 0.15 * max(y_max - y_min, 1e-12)

    plt.figure(figsize=(12, 5))
    plt.plot(times, thrusts, label="Pressure force Fz")

    plt.axvspan(
        last_rev_start,
        latest_time,
        alpha=0.2,
        label="final revolution window",
    )

    plt.ylim(y_min - y_margin, y_max + y_margin)
    plt.xlabel("Time [s]")
    plt.ylabel("Force Fz [N]")
    plt.title("Pressure Force Fz")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(force_plot, dpi=200)
    plt.close()

    # -----------------------------
    # Thrust stability metric plot
    # -----------------------------
    # This plot directly visualizes the implemented convergence criterion:
    # std(F) / |mean(F)| evaluated over a sliding one-revolution window.
    threshold = thrust_convergence["threshold"]
    status = "PASSED" if thrust_convergence["passed"] else "FAILED"
    final_relative_std = thrust_convergence["relative_std"]

    stability_history = compute_thrust_stability_history(times, thrusts, rev_time)
    metric_time = stability_history["time"]
    metric = stability_history["relative_std"]
    valid = np.isfinite(metric) & (metric > 0.0)

    plt.figure(figsize=(12, 5))

    if np.any(valid):
        plt.plot(
            metric_time[valid],
            metric[valid],
            label=r"sliding 1-rev $\sigma_F / |\overline{F}|$",
        )

    plt.axhline(
        threshold,
        linestyle="--",
        label=f"criterion = {threshold:g}",
    )

    plt.axvspan(
        last_rev_start,
        latest_time,
        alpha=0.2,
        label="final evaluation window",
    )

    if final_relative_std is not None:
        text = (
            f"Final 1-rev result: {final_relative_std:.3e} → {status}\n"
            f"Criterion: relative thrust fluctuation < {threshold:g}"
        )
    else:
        text = f"Final 1-rev result could not be evaluated → {status}"

    plt.text(
        0.02,
        0.95,
        text,
        transform=plt.gca().transAxes,
        ha="left",
        va="top",
        bbox=dict(facecolor="white", edgecolor="black", alpha=0.85),
    )

    plt.yscale("log")
    plt.xlabel("Time [s]")
    plt.ylabel(r"Relative thrust fluctuation $\sigma_F / |\overline{F}|$")
    plt.title("Thrust Stability Criterion over Sliding One-Revolution Window")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(conv_plot, dpi=200)
    plt.close()

    return force_plot, conv_plot, stability_history


def create_moments_plots(times, moments, report_dir, rev_time):
    import matplotlib.pyplot as plt

    moments_plot = report_dir / "moments_plot.png"

    latest_time = float(times[-1])
    last_rev_start = latest_time - rev_time

    # -----------------------------
    # Moments history plot
    # -----------------------------

    plot_mask = times > 0.001
    plot_moments = moments[plot_mask]

    if len(plot_moments) > 0:
        y_min = np.percentile(plot_moments, 1)
        y_max = np.percentile(plot_moments, 99)
        y_margin = 0.15 * max(y_max - y_min, 1e-12)
    else:
        y_min, y_max = np.min(moments), np.max(moments)
        y_margin = 0.15 * max(y_max - y_min, 1e-12)

    plt.figure(figsize=(12, 5))
    plt.plot(times, moments, label="Moments (pressure & viscous) M_y")

    plt.axvspan(
        last_rev_start,
        latest_time,
        alpha=0.2,
        label="final revolution window",
    )

    plt.ylim(y_min - y_margin, y_max + y_margin)
    plt.xlabel("Time [s]")
    plt.ylabel("Moment My [Nm]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(moments_plot, dpi=200)
    plt.close()

    return moments_plot


def read_residual_dataframe(residual_file):
    if not residual_file.exists():
        return None

    with open(residual_file, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    if len(lines) < 3:
        return None

    header = lines[1].lstrip("#").split()

    df = pd.read_csv(
        residual_file,
        sep=r"\s+",
        names=header,
        skiprows=2,
        engine="python",
    )

    if "Time" not in df.columns:
        return None

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["Time"])
    df = df.sort_values("Time")

    return df


def evaluate_residual_slopes(df, rev_time, latest_time):
    if df is None or len(df) == 0:
        return None

    last_rev_start = latest_time - rev_time
    window = df[(df["Time"] >= last_rev_start) & (df["Time"] <= latest_time)].copy()

    if len(window) < 2:
        return {
            "window_start_s": last_rev_start,
            "window_end_s": latest_time,
            "n_samples": int(len(window)),
            "slopes_per_rev": {},
            "end_residuals": {},
            "mean_residuals": {},
            "reason": "Not enough residual samples in final revolution window.",
        }

    # Independent variable in revolutions relative to the start of the final window.
    x_rev = (window["Time"].to_numpy(dtype=float) - last_rev_start) / rev_time

    slopes_per_rev = {}
    end_residuals = {}
    mean_residuals = {}

    for col in window.columns:
        if col == "Time":
            continue

        values = window[col].to_numpy(dtype=float)
        valid = np.isfinite(values) & (values > 0.0) & np.isfinite(x_rev)

        if np.count_nonzero(valid) < 2:
            slopes_per_rev[col] = None
            end_residuals[col] = None
            mean_residuals[col] = None
            continue

        y_log = np.log10(values[valid])
        x_valid = x_rev[valid]

        # Slope of log10(residual) per propeller revolution.
        slope, _intercept = np.polyfit(x_valid, y_log, 1)

        slopes_per_rev[col] = float(slope)
        end_residuals[col] = float(values[valid][-1])
        mean_residuals[col] = float(np.mean(values[valid]))

    return {
        "window_start_s": float(window["Time"].iloc[0]),
        "window_end_s": float(window["Time"].iloc[-1]),
        "n_samples": int(len(window)),
        "slopes_per_rev": slopes_per_rev,
        "end_residuals": end_residuals,
        "mean_residuals": mean_residuals,
        "reason": None,
    }


def create_residual_plots(residual_file, report_dir, rev_time, latest_time):
    import matplotlib.pyplot as plt

    residual_plot = report_dir / "residuals.png"

    df = read_residual_dataframe(residual_file)

    if df is None:
        return None, None

    last_rev_start = latest_time - rev_time

    plt.figure(figsize=(12, 5))

    for col in df.columns:
        if col != "Time":
            plt.plot(df["Time"], df[col], label=col)

    plt.axvspan(
        last_rev_start,
        latest_time,
        alpha=0.2,
        label="final revolution window",
    )

    plt.yscale("log")
    plt.xlabel("Time [s]")
    plt.ylabel("Residual")
    plt.title("Residual Convergence")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(residual_plot, dpi=200)
    plt.close()

    residual_slope_info = evaluate_residual_slopes(df, rev_time, latest_time)

    return residual_plot, residual_slope_info


def draw_courant_summary(c, title, summary, y_position):
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_position, title)
    y_position -= 20

    c.setFont("Helvetica", 10)

    if summary is None:
        c.drawString(50, y_position, "No matching entries found in solver log.")
        return y_position - 26

    c.drawString(
        50,
        y_position,
        f"Samples: {summary['samples']} | "
        "Logged mean Co, average / maximum: "
        f"{format_optional_number(summary['mean_co_average'], '.4g')} / "
        f"{format_optional_number(summary['mean_co_max'], '.4g')}",
    )
    y_position -= 18

    c.drawString(
        50,
        y_position,
        "Logged maximum Co, average / peak: "
        f"{format_optional_number(summary['max_co_average'], '.4g')} / "
        f"{format_optional_number(summary['peak_max_co'], '.4g')}",
    )
    y_position -= 18

    exceedance_count = summary[
        "configured_max_co_exceedance_count"
    ]
    exceedance_percent = summary[
        "configured_max_co_exceedance_percent"
    ]

    if exceedance_count is not None:
        c.drawString(
            50,
            y_position,
            "Samples above configured maxCo: "
            f"{exceedance_count} "
            f"({format_optional_number(exceedance_percent, '.2f')}%)",
        )
        y_position -= 18

    return y_position - 18
