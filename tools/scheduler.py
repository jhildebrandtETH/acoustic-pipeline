"""Scheduler helpers for the acoustic pipeline."""

import os
import shutil
import time
import threading
import traceback
from pathlib import Path
import json
from datetime import datetime

from tools.common import _atomic_write_json
from tools.openfoam import load_simulation_order


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
        self._animate = (
            self._is_tty
            and os.environ.get("TERM") != "dumb"
            and os.environ.get("ACOUSTIC_DASHBOARD_ANIMATION", "1") != "0"
        )
        self._animation_color = "NO_COLOR" not in os.environ
        try:
            "▀▄█".encode(sys.stdout.encoding or "ascii")
            self._animation_unicode = True
        except (UnicodeError, LookupError):
            self._animation_unicode = False

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
        from tools.terminal_propeller import frame, status_overlay

        started = time.monotonic()
        next_status = 0.0
        last_size = None
        overlay = None
        interval = min(self.refresh_interval, 0.1) if self._animate else self.refresh_interval
        try:
            if self._animate:
                print("\033[?25l", end="", flush=True)
            while not self._stop_event.wait(interval):
                now = time.monotonic()
                if self._is_tty:
                    size = shutil.get_terminal_size()
                    output = ""
                    if now >= next_status or size != last_size:
                        status = self.render_text()
                        overlay = status_overlay(status, size.columns, size.lines) if self._animate else None
                        if overlay is None:
                            output = "\033[H\033[J" + status
                        elif size != last_size:
                            output = "\033[H\033[J"
                        next_status = now + self.refresh_interval
                        last_size = size
                    if overlay is not None:
                        output += "\033[H" + "\n".join(frame(
                            now - started,
                            width=size.columns - 1,
                            height=size.lines - 1,
                            overlay=overlay,
                            unicode=self._animation_unicode,
                            color=self._animation_color,
                        ))
                    if output:
                        print(output, end="", flush=True)
                elif now - self._last_non_tty_render >= 30.0:
                    # In redirected output / SLURM logs, avoid ANSI escape spam.
                    print("\n" + self.render_text(), flush=True)
                    self._last_non_tty_render = now
        finally:
            if self._animate:
                print("\033[0m\033[?25h", end="", flush=True)

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
    cores_per_case: int | None = None,
) -> dict:
    """Return fixed-size slots; retain automatic allocation for legacy orders."""
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

    if cores_per_case is not None:
        cores_per_case = int(cores_per_case)
        if not 1 <= cores_per_case <= total_cores:
            raise ValueError("cores_per_case must be between 1 and total_cores")
        return {
            "total_cores": total_cores,
            "cores_per_case": cores_per_case,
            "max_cores_per_case": cores_per_case,
            "extra_core_slots": 0,
            "max_parallel_cases": min(parallel_units, total_cores // cores_per_case),
            "dependency_mode": dependency_mode,
        }

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
        cores_per_case=order.get("target_cores_per_case"),
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
    from tools.openfoam import get_safe_timestep
    from tools.openfoam import has_timestep
    order = order_store.snapshot()

    for case in order["cases"]:
        status = case.get("status")

        if status == "failed":
            resume_status = case.get("resume_status")
            if order.get("meshing_backend") == "cfmesh-native-rotor-stator-v1":
                case_path = Path(simulations_directory) / case["folder"]
                if case.get("mesh_only") or not case_path.is_dir() or not get_safe_timestep(case_path):
                    resume_status = "pending"

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


def resume_status_after_solver_failure(simulation_path: Path) -> str:
    from tools.openfoam import has_timestep
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
    from tools.common import _case_path_state
    from tools.common import _write_pipeline_error_log
    from tools.openfoam import get_safe_timestep
    from tools.openfoam import has_timestep
    from tools.openfoam import reset_case_folder
    # Local imports avoid circular imports: preprocessing/openfoamSimulation/
    # postprocessing themselves import helpers from tools/.
    from openfoamSimulation import openfoamSimulation
    from tools.preprocessing import run_preprocessing as preprocessing

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
                    WALL_FUNCTIONS=args.wall_functions,
                    AERODYNAMICS_ONLY=args.aerodynamics_only,
                    STATUS_CALLBACK=callback,
                    LIVE_OUTPUT=getattr(args, "live_output", False),
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
                    END_ON_VALUE=getattr(args, "end_on_value", None),
                    TURBULENCE_MODEL=args.turbulence,
                    initialize_from_previous=use_previous_init,
                    previous_simulation_path=previous_simulation_path,
                    NUMBER_OF_CORES=allocated_cores,
                    MESH_ONLY=args.mesh_only,
                    ALLOW_BAD_MESH=args.allow_bad_mesh,
                    LIVE_OUTPUT=getattr(args, "live_output", False),
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
                    END_ON_VALUE=getattr(args, "end_on_value", None),
                    TURBULENCE_MODEL=args.turbulence,
                    initialize_from_previous=use_previous_init,
                    previous_simulation_path=previous_simulation_path,
                    NUMBER_OF_CORES=allocated_cores,
                    MESH_ONLY=args.mesh_only,
                    ALLOW_BAD_MESH=args.allow_bad_mesh,
                    LIVE_OUTPUT=getattr(args, "live_output", False),
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
                registry.update(
                    folder_name,
                    stage="postprocessing",
                    detail="starting postprocessing",
                    progress=0.0,
                )

                from postprocessing import postprocessing
                postprocessing(
                    MESH_ONLY=args.mesh_only,
                    AERODYNAMICS_ONLY=args.aerodynamics_only,
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
    from tools.common import _write_pipeline_error_log
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
    if not getattr(args, "live_output", False):
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
        if not getattr(args, "live_output", False):
            dashboard.stop(final_render=True)
