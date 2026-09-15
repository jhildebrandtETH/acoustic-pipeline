"""Common helpers for the acoustic pipeline."""

import os
import re
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
