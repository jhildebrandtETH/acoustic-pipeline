import os
import time
import numpy as np
import pandas as pd
import re
from pathlib import Path
import json
from datetime import datetime

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


def safe_exec(container, cmd, description="command", print_output=False):
    try:
        container.reload()

        if container.status != "running":
            print(f"Container is not running before {description}.")
            return False

        result = container.exec_run(cmd, stream=True)

        for line in result.output:
            if print_output:
                print(line.decode("utf-8", errors="ignore").strip())

        return True

    except Exception as e:
        print(f"{description} failed: {e}")
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

        if "internalField" not in content:
            return False

        if "boundaryField" not in content:
            return False

        return True

    for folder in folders_to_check:

        for filename in required_files:

            if not file_is_healthy(folder / filename):
                return False

    return True

def merge_postprocessing_dat_files(case_dir: Path, function_object_name: str) -> Path | None:
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



def reset_case_folder(simulation_path: Path):
    if simulation_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        broken_path = simulation_path.with_name(
            simulation_path.name + f"_BROKEN_{timestamp}"
        )

        simulation_path.rename(broken_path)
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
    json_path = simulations_directory / "simulation_order.json"

    if not json_path.exists():
        raise FileNotFoundError("No simulation_batch.json found for resume")

    with open(json_path, "r") as f:
        return json.load(f)
    


def update_case_status(simulations_directory: Path, folder_name: str, new_status: str):
    json_path = simulations_directory / "simulation_order.json"

    with open(json_path, "r") as f:
        batch = json.load(f)

    for case in batch["cases"]:
        if case["folder"] == folder_name:
            case["status"] = new_status
            break

    with open(json_path, "w") as f:
        json.dump(batch, f, indent=4)



def create_simulation_order(args, simulations_directory: Path):

    simulations_directory.mkdir(parents=True, exist_ok=True)

    json_path = simulations_directory / "simulation_order.json"

    # Enforce: one order = one directory
    if json_path.exists():
        raise FileExistsError(
            f"Simulation order already exists in this directory:\n"
            f"{json_path}\n\n"
            f"One simulation order must have its own simulation_run folder. "
            f"Create a new directory or use --resume."
        )

    batch = {
        "mode": args.mode,
        "turbulence" : args.turbulence,
        "meshes": args.meshes,
        "rpms": args.rpms,
        "cores": args.cores,
        "field_init": args.field_init,
        "mesh_only" : args.mesh_only,
        "end_on" : args.end_on,
        "allow_bad_mesh" : args.allow_bad_mesh,
        "study": args.study,
        "study_file": getattr(args, "study_file", None),
        "study_parameter": getattr(args, "study_parameter", None),
        "study_values": getattr(args, "study_values", None),
        "cases": []
    }

    # -------- STUDY ON --------
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

            batch["cases"].append({
                "folder": folder,
                "mesh": mesh,
                "rpm": rpm,
                "mode": args.mode,
                "turbulence" : args.turbulence,
                "cores": args.cores,
                "mesh_only" : args.mesh_only,
                "end_on" : args.end_on,
                "allow_bad_mesh" : args.allow_bad_mesh,
                "field_init": args.field_init,
                "study": args.study,
                "study_file": args.study_file,
                "study_parameter": args.study_parameter,
                "study_value": value,
                "status": "pending"
            })

    # -------- STUDY OFF --------
    else:
        for mesh in args.meshes:
            for rpm in args.rpms:
                folder = f"{mesh}_{rpm}RPM_{args.mode}"

                batch["cases"].append({
                    "folder": folder,
                    "mesh": mesh,
                    "rpm": rpm,
                    "mode": args.mode,
                    "turbulence" : args.turbulence,
                    "cores": args.cores,
                    "mesh_only" : args.mesh_only,
                    "end_on" : args.end_on,
                    "allow_bad_mesh" : args.allow_bad_mesh,
                    "field_init": args.field_init,
                    "study": args.study,
                    "study_file": None,
                    "study_parameter": None,
                    "study_value": None,
                    "status": "pending"
                })


    with open(json_path, "w") as f:
        json.dump(batch, f, indent=4)

    print(f"Created simulation order file: {json_path}")


def is_mesh_ok(log_path):
    """
    Returns True if 'Mesh OK' is found in log.checkMesh, else False.
    """

    if not log_path.exists():
        print("Coudn't confirm mesh is OK because of path error...")
        return False

    log_text = log_path.read_text(errors="ignore")

    return "Mesh OK" in log_text


def check_residuals(
    residuals_file,
    revolution_time,
    use_log=True,
    min_points=10,
):
    """
    Returns True if all residuals satisfy slope criteria over the last revolution.

    The fitted regression slope is converted from "per second" to
    "per revolution" by multiplying with revolution_time.

    If use_log=True, the checked quantity is the change in log10(residual)
    over one revolution.
    """

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
):
    """
    Display reconstructPar progress on one terminal line.

    Progress is calculated from the number of numeric time directories that
    exist in the case root compared with the matching directories in
    processor0. Set maximum_time for a limited resume reconstruction such as
    reconstructPar -time ":0.05".

    Example:
        RECONSTRUCT | 42/120 time dirs | 35.00% |
        latest 0.021000 s | [##########--------------------]
    """

    case_path = Path(main_sim_folder)
    maximum_time_value = (
        float(maximum_time)
        if maximum_time is not None
        else None
    )

    previous_print_width = 0

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

            if (
                maximum_time_value is None
                or value <= maximum_time_value + 1e-12
            ):
                # Rounded keys avoid mismatches such as 0.01 versus 1e-2.
                values.add(round(value, 12))

        return values

    def find_reference_processor():
        processor0 = case_path / "processor0"

        if processor0.is_dir():
            return processor0

        processor_directories = sorted(
            path
            for path in case_path.glob("processor*")
            if path.is_dir()
        )

        return (
            processor_directories[0]
            if processor_directories
            else None
        )

    def update_status_line(message):
        nonlocal previous_print_width

        print_width = max(previous_print_width, len(message))

        print(
            "\r" + message.ljust(print_width),
            end="",
            flush=True,
        )

        previous_print_width = print_width

    def close_status_line(final_message=None):
        if final_message is not None:
            update_status_line(final_message)

        if previous_print_width > 0:
            print(flush=True)

    try:
        reference_processor = find_reference_processor()

        while reference_processor is None:
            update_status_line(
                "RECONSTRUCT | Waiting for processor directories..."
            )

            if stop_event is not None and stop_event.wait(check_interval):
                close_status_line()
                return None

            if stop_event is None:
                time.sleep(check_interval)

            reference_processor = find_reference_processor()

        expected_times = numeric_times(reference_processor)

        while not expected_times:
            limit_text = (
                f" up to {maximum_time_value:.6f} s"
                if maximum_time_value is not None
                else ""
            )

            update_status_line(
                "RECONSTRUCT | Waiting for processor time directories"
                f"{limit_text}..."
            )

            if stop_event is not None and stop_event.wait(check_interval):
                close_status_line()
                return None

            if stop_event is None:
                time.sleep(check_interval)

            expected_times = numeric_times(reference_processor)

        expected_count = len(expected_times)
        bar_width = 30

        while True:
            stopping = (
                stop_event is not None
                and stop_event.is_set()
            )

            reconstructed_times = numeric_times(case_path)
            completed_times = expected_times.intersection(
                reconstructed_times
            )

            completed_count = len(completed_times)
            percentage = min(
                max(
                    100.0 * completed_count / expected_count,
                    0.0,
                ),
                100.0,
            )

            bar_completed = round(
                bar_width * percentage / 100.0
            )
            progress_bar = (
                "#" * bar_completed
                + "-" * (bar_width - bar_completed)
            )

            latest_time = (
                max(completed_times)
                if completed_times
                else 0.0
            )

            message = (
                f"RECONSTRUCT | {completed_count}/{expected_count} "
                f"time dirs | {percentage:6.2f}% | "
                f"latest {latest_time:.6f} s | "
                f"[{progress_bar}]"
            )

            update_status_line(message)

            if completed_count >= expected_count:
                close_status_line(message)
                return {
                    "completed": completed_count,
                    "expected": expected_count,
                    "percentage": 100.0,
                    "latest_time": latest_time,
                }

            # The reconstruction command has ended. This iteration already
            # performed the final filesystem scan, so close the dynamic line.
            if stopping:
                close_status_line()
                return {
                    "completed": completed_count,
                    "expected": expected_count,
                    "percentage": percentage,
                    "latest_time": latest_time,
                }

            if stop_event is not None:
                stop_event.wait(check_interval)
            else:
                time.sleep(check_interval)

    except Exception as error:
        close_status_line()
        print(f"Reconstruction progress monitor failed: {error}")
        return None


def run_time_progress_monitor(
    main_sim_folder,
    end_time,
    check_interval=5.0,
    log_file_name="log.pimpleFoam",
    stop_event=None,
):
    """
    Display OpenFOAM physical-time progress on one terminal line.

    Example:
        PIMPLE | Time 0.042500 / 0.090000 s | 47.22% |
        [##############----------------]

    The solver log is read incrementally, so the complete file is not scanned
    again on every update.
    """

    log_path = Path(main_sim_folder) / log_file_name
    end_time = float(end_time)

    if end_time <= 0.0:
        raise ValueError("end_time must be greater than zero.")

    # Be tolerant of prefixes, suffixes, and additional whitespace in
    # OpenFOAM or MPI output lines.
    time_pattern = re.compile(
        rf"\bTime\s*=\s*({_OPENFOAM_NUMBER})"
    )

    latest_time = 0.0
    file_position = 0
    unfinished_line = ""
    previous_print_width = 0

    def update_status_line(message):
        nonlocal previous_print_width

        # Padding removes characters left by a previously longer message.
        print_width = max(previous_print_width, len(message))

        print(
            "\r" + message.ljust(print_width),
            end="",
            flush=True,
        )

        previous_print_width = print_width

    def close_status_line(final_message=None):
        if final_message is not None:
            update_status_line(final_message)

        # Finish the dynamic line so following print() calls start normally.
        if previous_print_width > 0:
            print(flush=True)

    try:
        while True:
            stopping = (
                stop_event is not None
                and stop_event.is_set()
            )

            if log_path.is_file():
                current_size = log_path.stat().st_size

                # tee truncates the file when a new solver run begins.
                if current_size < file_position:
                    file_position = 0
                    unfinished_line = ""

                with log_path.open(
                    "r",
                    encoding="utf-8",
                    errors="ignore",
                ) as log_file:
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

            percentage = min(
                max(100.0 * latest_time / end_time, 0.0),
                100.0,
            )

            bar_width = 30
            completed = round(bar_width * percentage / 100.0)
            progress_bar = (
                "#" * completed
                + "-" * (bar_width - completed)
            )

            if log_path.is_file():
                message = (
                    f"PIMPLE | Time {latest_time:.6f} / "
                    f"{end_time:.6f} s | "
                    f"{percentage:6.2f}% | "
                    f"[{progress_bar}]"
                )
            else:
                message = "PIMPLE | Waiting for solver log..."

            update_status_line(message)

            if latest_time >= end_time - 1e-12:
                close_status_line(
                    f"PIMPLE | Time {latest_time:.6f} / "
                    f"{end_time:.6f} s | "
                    f"100.00% | [{'#' * bar_width}]"
                )
                return latest_time

            # After the solver sets the event, this loop still reads the log
            # once more before terminating.
            if stopping:
                close_status_line()
                return latest_time

            if stop_event is not None:
                stop_event.wait(check_interval)
            else:
                time.sleep(check_interval)

    except Exception as error:
        close_status_line()
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

                if check_residuals(residuals_file, rev_time):
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
                    if check_residuals(residuals_file, rev_time):
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

def update_parameter(file_path, target_var, new_value):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    lines = []
    updated = False

    # Read the file and modify the specific line
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2 and parts[0] == target_var:
                lines.append(f"{target_var} {new_value};\n")
                updated = True
            else:
                lines.append(line)

    # Write the changes back to the file
    if updated:
        with open(file_path, 'w') as f:
            f.writelines(lines)
        print(f"Successfully updated {target_var} to {new_value}.")
    else:
        print(f"Variable '{target_var}' not found in the file.")




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
