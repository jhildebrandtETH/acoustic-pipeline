"""Openfoam helpers for the acoustic pipeline."""

import os
import gzip
import time
import numpy as np
import pandas as pd
import re
import threading
from pathlib import Path
import json
import math
from decimal import Decimal, InvalidOperation
from datetime import datetime

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
        if not time_value.is_finite() or time_value <= 0:
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
    from tools.common import emit_status
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

    reconstructed_values = {Decimal(name) for name in reconstructed_times}
    missing_times = {name for name in expected_times if Decimal(name) not in reconstructed_values}

    if missing_times:
        missing_times = sorted(missing_times, key=Decimal)
        emit_status(
            status_callback,
            stage="resume_check",
            detail=(f"Reconstruction incomplete; missing {len(missing_times)} time directories: "
                    + ", ".join(missing_times[:10])),
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
    from tools.common import emit_status
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
    from tools.common import emit_status
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
            if print_output:
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


def _openfoam_field_class(file_path):
    """Read a field header without loading a potentially very large field."""
    opener = gzip.open if file_path.suffix == ".gz" else open
    with opener(file_path, "rb") as handle:
        header = handle.read(65536)
    header = re.sub(rb"/\*.*?\*/|//[^\n]*", b"", header, flags=re.DOTALL)
    match = re.search(rb"\bFoamFile\s*\{([^}]+)\}", header)
    if match:
        field_class = re.search(rb"\bclass\s+([^;\s]+)\s*;", match.group(1))
        if field_class:
            return field_class.group(1).decode("ascii")
    return None


def _openfoam_field_health_error(file_path, expected_class=None):
    """Basic structure check, including gzip/binary; not a numerical validation."""
    if not file_path.is_file():
        file_path = file_path.with_name(file_path.name + ".gz")
    if not file_path.is_file():
        return "missing field"
    try:
        field_class = _openfoam_field_class(file_path)
        if not field_class:
            return "missing OpenFOAM field header/class"
        if expected_class and field_class != expected_class:
            return f"field class {field_class} differs from {expected_class}"
        markers = {b"dimensions", b"value"} if field_class.endswith("::Internal") else {
            b"dimensions", b"internalField", b"boundaryField"
        }
        opener = gzip.open if file_path.suffix == ".gz" else open
        with opener(file_path, "rb") as handle:
            tail = b""
            while chunk := handle.read(1024 * 1024):
                data = tail + chunk
                markers = {marker for marker in markers if marker not in data}
                tail = data[-32:]
        if markers:
            return "missing " + ", ".join(sorted(marker.decode() for marker in markers))
    except (OSError, EOFError, UnicodeError) as error:
        return f"unreadable field: {error}"
    return None


def processor_deletion_is_safe(
    PATH_TO_CONTROL_DICT_PARAMETERS,
    SIMULATION_DIRECTORY,
    TURBULENCE_MODEL: str,
    RESUME: bool,
    status_callback=None,
    maximum_time=None,
) -> bool:
    """Verify saved processor history before cleanup after successful reconstructPar.

    purgeWrite is a retention limit, not a minimum output count. Keep the
    parameter-path argument for caller compatibility, but use actual processor
    outputs as the source of truth. On resume, maximum_time is the selected
    safe timestep; newer, potentially interrupted output is deliberately excluded.
    """
    from tools.common import emit_status
    sim_dir = Path(SIMULATION_DIRECTORY)
    stage = "resume_check" if RESUME else "cleanup"

    def report(detail):
        # Dashboard details are overwritten by later stages. Keep the reason
        # beside reconstructPar's log so failed cleanup remains diagnosable.
        try:
            (sim_dir / "log.processor_cleanup_check").write_text(detail + "\n", encoding="utf-8")
        except OSError:
            pass
        emit_status(status_callback, stage=stage, detail=detail)

    def reject(reason):
        detail = f"Processor folders preserved: {reason}"
        report(detail)
        if status_callback is None:
            print(detail)
        return False

    turbulence_fields = {
        "kEpsilon": {"k", "epsilon"},
        "kOmegaSST": {"k", "omega"},
        "DES": {"k", "omega"},
    }.get(TURBULENCE_MODEL.strip())
    if turbulence_fields is None:
        return reject(f"unsupported turbulence model {TURBULENCE_MODEL!r}")
    required_fields = {"U", "p", "phi", "nut"} | turbulence_fields

    def time_paths(directory):
        return {
            Decimal(name): directory / name
            for name in sorted(_numeric_time_directories(directory, maximum_time))
        }

    try:
        if not sim_dir.is_dir():
            return reject(f"case directory is missing: {sim_dir}")
        processors = sorted(
            (p for p in sim_dir.iterdir() if p.is_dir()
             and re.fullmatch(r"processor\d+", p.name)),
            key=lambda p: int(p.name[9:]),
        )
        if not processors:
            return reject("no processor directories found to verify")
        processor_times = {p: time_paths(p) for p in processors}
        expected_times = set().union(*(set(times) for times in processor_times.values()))
        if not expected_times:
            return reject("no saved processor timesteps found in the reconstruction range")
        root_times = time_paths(sim_dir)
        for time_value in sorted(expected_times):
            for processor, times in processor_times.items():
                if time_value not in times:
                    return reject(f"{processor.name} is missing timestep {time_value}")
            if time_value not in root_times:
                return reject(f"missing reconstructed timestep {time_value}")

            # Uf and function-object outputs are required only when written.
            # Inspect every rank so a field missing on processor0 is not lost.
            written_fields = {}
            for times in processor_times.values():
                for field_path in times[time_value].iterdir():
                    if not field_path.is_file():
                        continue
                    field_class = _openfoam_field_class(field_path)
                    if field_class and re.fullmatch(
                        r"(?:vol|surface|point)\w*Field(?:::Internal)?", field_class
                    ):
                        name = field_path.name.removesuffix(".gz")
                        if name in written_fields and written_fields[name] != field_class:
                            return reject(f"conflicting processor field classes at {time_value}/{name}")
                        written_fields[name] = field_class

            folder = root_times[time_value]
            for name in sorted(required_fields | written_fields.keys()):
                error = _openfoam_field_health_error(folder / name, written_fields.get(name))
                if error:
                    return reject(f"{folder.name}/{name}: {error}")
    except (OSError, EOFError, UnicodeError) as error:
        return reject(f"could not inspect reconstruction: {error}")

    report(f"Verified fields in all {len(expected_times)} reconstructed processor timesteps")
    return True


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


def reset_case_folder(simulation_path: Path, status_callback=None):
    from tools.common import emit_status
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
    from tools.common import _SIMULATION_ORDER_FILE_LOCK
    json_path = Path(simulations_directory) / "simulation_order.json"

    if not json_path.exists():
        raise FileNotFoundError("No simulation_order.json found for resume")

    with _SIMULATION_ORDER_FILE_LOCK:
        with json_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)


def save_simulation_order(simulations_directory: Path, order: dict) -> None:
    """Atomically persist a complete simulation order."""
    from tools.common import _SIMULATION_ORDER_FILE_LOCK
    from tools.common import _atomic_write_json
    json_path = Path(simulations_directory) / "simulation_order.json"
    with _SIMULATION_ORDER_FILE_LOCK:
        _atomic_write_json(json_path, order)


def update_case_status(simulations_directory: Path, folder_name: str, new_status: str):
    """Backward-compatible, concurrency-safe status update helper."""
    from tools.common import _SIMULATION_ORDER_FILE_LOCK
    from tools.common import _atomic_write_json
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


def parse_end_on(values):
    """Parse the CLI stop condition while retaining legacy bare 'time'."""
    mode = values[0]
    if mode not in {"time", "rev", "convergence", "force_convergence", "residual_convergence"}:
        raise ValueError("--end-on expects time SECONDS, rev REVOLUTIONS, or a convergence mode")
    if len(values) == 1 and mode != "rev":
        return mode, None
    if len(values) != 2 or mode not in {"time", "rev"}:
        raise ValueError("Only --end-on time and --end-on rev accept one numeric value; rev requires it")
    try:
        value = float(values[1])
    except ValueError:
        raise ValueError("--end-on value must be a finite positive number") from None
    if not math.isfinite(value) or value <= 0:
        raise ValueError("--end-on value must be a finite positive number")
    return mode, value


def create_simulation_order(args, simulations_directory: Path):
    """Create the durable simulation order including scheduler metadata."""
    from tools.common import _SIMULATION_ORDER_FILE_LOCK
    from tools.common import _atomic_write_json
    from tools.scheduler import add_field_initialization_dependencies
    from tools.scheduler import assign_case_core_allocations
    from tools.scheduler import calculate_scheduler_layout
    simulations_directory = Path(simulations_directory)
    simulations_directory.mkdir(parents=True, exist_ok=True)
    json_path = simulations_directory / "simulation_order.json"

    total_cores = int(args.total_cores)

    batch = {
        "schema_version": 3,
        "wall_functions": args.wall_functions,
        "aerodynamics_only": args.aerodynamics_only,
        "meshing_backend": "cfmesh-native-rotor-stator-v1",
        "acoustic_surface": args.acoustic_surface,
        "acoustic_sphere_diameter": args.acoustic_sphere_diameter,
        "mode": args.mode,
        "turbulence": args.turbulence,
        "meshes": args.meshes,
        "rpms": args.rpms,
        "total_cores": total_cores,
        "target_cores_per_case": int(args.cores_per_case),
        "field_init": args.field_init,
        "mesh_only": args.mesh_only,
        "end_on": args.end_on,
        "end_on_value": getattr(args, "end_on_value", None),
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
            folder = f"{mesh}_{rpm}RPM_{make_folder_safe(args.study_parameter)}_{safe_value}"

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
                    "end_on_value": getattr(args, "end_on_value", None),
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
                        "end_on_value": getattr(args, "end_on_value", None),
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

    folders = [case["folder"] for case in batch["cases"]]
    if not folders or len(set(folders)) != len(folders):
        raise ValueError("Order needs distinct, non-empty case/study values")

    add_field_initialization_dependencies(
        batch["cases"],
        field_init=args.field_init,
        study=args.study,
    )

    layout = calculate_scheduler_layout(
        cases=batch["cases"],
        total_cores=total_cores,
        cores_per_case=args.cores_per_case,
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

    from tools.cfmesh_orders import require_new_order

    with _SIMULATION_ORDER_FILE_LOCK:
        require_new_order(simulations_directory)
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


slope_bounds = {
    "p":  (-10e-2, 10e-2), #(-5e-2, 1e-2)
    "Ux": (-10e-2, 10e-3), #(-5e-2, 5e-3)
    "Uy": (-10e-2, 10e-3), #(-5e-2, 5e-3)
    "Uz": (-10e-2, 10e-3), #(-5e-2, 5e-3)
    "k":  (-10e-2, 10e-3), #(-5e-2, 5e-3)
}


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
    from tools.common import emit_status
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
    from tools.common import emit_status
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
    from tools.common import emit_status
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
