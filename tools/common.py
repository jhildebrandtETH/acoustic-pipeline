"""Common helpers for the acoustic pipeline."""

import os
import shutil
import subprocess
import re
import signal
import threading
from pathlib import Path
import json

_SIMULATION_ORDER_FILE_LOCK = threading.RLock()


MATPLOTLIB_LOCK = threading.RLock()


VISUALIZATION_LOCK = threading.RLock()


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


def validate_acoustic_arguments(parser, args):
    """Validate coupled acoustic CLI arguments and apply safe defaults."""
    if getattr(args, "aerodynamics_only", False):
        if args.acoustic_surface is not None or args.acoustic_sphere_diameter is not None:
            parser.error("--aerodynamics-only cannot be combined with acoustic surface options")
        return
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
    from tools.openfoam import safe_exec
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
            "The root mesh may not contain the assembled mesh."
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
