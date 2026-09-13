from datetime import datetime
import json
import math
import signal
import subprocess
import sys
import shutil
import tempfile
from pathlib import Path


from tools import emit_status
from tools import get_latest_timestep
from tools import update_parameter


def prepare_case_directory(template_directory, parameters_directory, target_directory):
    """Copy a fresh case, preserving any previous results in a sibling folder."""
    from cfmesh_pipeline import safe_path

    target_directory = safe_path(target_directory)
    # DrvFS can return EEXIST for a deleted case name that stat cannot see.
    # Build under a unique sibling name, then rename the populated directory.
    with tempfile.TemporaryDirectory(
        prefix=".case-", dir=target_directory.parent
    ) as temporary_directory:
        staging_directory = Path(temporary_directory)
        shutil.copytree(template_directory, staging_directory, dirs_exist_ok=True)
        # A template may contain a saved timestep that overrides deltaT.
        # New cases must start from the current control dictionary.
        (staging_directory / "0/uniform/time").unlink(missing_ok=True)
        shutil.copytree(
            parameters_directory, staging_directory / "Parameters", dirs_exist_ok=True
        )

        if target_directory.exists():
            if not target_directory.is_dir():
                raise FileExistsError(
                    f"Case path is not a directory: {target_directory}"
                )
            archived_directory = safe_path(
                target_directory.with_name(
                    target_directory.name
                    + "_PREVIOUS_"
                    + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                )
            )
            target_directory.rename(archived_directory)
            print(f"Previous case preserved in {archived_directory}")

        staging_directory.rename(target_directory)


def preprocessing(
    SIMULATION_NAME,
    RPM_COUNT,
    MAIN_DIRECTORY,
    TARGET_DIRECTORY,
    CORES_TO_USE,
    MODE,
    INIT_FROM_PREVIOUS,
    PREVIOUS_SIMULATION_PATH,
    TURBULENCE_MODEL,
    ACOUSTIC_SURFACE,
    ACOUSTIC_SPHERE_DIAMETER,
    STUDY_PARAMETER_NAME=None,
    STUDY_PARAMETER_FILE=None,
    STUDY_PARAMETER=None,
    STATUS_CALLBACK=None,
    LIVE_OUTPUT=False,
):
    """Run geometry preparation in its own process and retain native crash output."""
    from cfmesh_pipeline import safe_path

    arguments = dict(
        SIMULATION_NAME=SIMULATION_NAME,
        RPM_COUNT=RPM_COUNT,
        MAIN_DIRECTORY=MAIN_DIRECTORY,
        TARGET_DIRECTORY=TARGET_DIRECTORY,
        CORES_TO_USE=CORES_TO_USE,
        MODE=MODE,
        INIT_FROM_PREVIOUS=INIT_FROM_PREVIOUS,
        PREVIOUS_SIMULATION_PATH=PREVIOUS_SIMULATION_PATH,
        TURBULENCE_MODEL=TURBULENCE_MODEL,
        ACOUSTIC_SURFACE=ACOUSTIC_SURFACE,
        ACOUSTIC_SPHERE_DIAMETER=ACOUSTIC_SPHERE_DIAMETER,
        STUDY_PARAMETER_NAME=STUDY_PARAMETER_NAME,
        STUDY_PARAMETER_FILE=STUDY_PARAMETER_FILE,
        STUDY_PARAMETER=STUDY_PARAMETER,
    )
    case_directory = safe_path(TARGET_DIRECTORY)
    log_path = safe_path(
        case_directory.parent / (case_directory.name + ".preprocessing.log")
    )
    emit_status(STATUS_CALLBACK, stage="preprocessing", detail="preparing case")
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-u", "-X", "faulthandler", str(Path(__file__).resolve())],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        process.stdin.write(json.dumps(arguments, default=str))
        process.stdin.close()
        for line in process.stdout:
            log.write(line)
            log.flush()
            if line.startswith("PIPELINE_STATUS "):
                emit_status(
                    STATUS_CALLBACK, **json.loads(line[len("PIPELINE_STATUS ") :])
                )
            elif LIVE_OUTPUT:
                print(line, end="", flush=True)
        return_code = process.wait()
    if return_code:
        reason = (
            signal.Signals(-return_code).name
            if return_code < 0
            else f"exit code {return_code}"
        )
        raise RuntimeError(f"Preprocessing failed ({reason}); see {log_path}")


def _prepare_case(
    SIMULATION_NAME,
    RPM_COUNT,
    MAIN_DIRECTORY,
    TARGET_DIRECTORY,
    CORES_TO_USE,
    MODE,
    INIT_FROM_PREVIOUS,
    PREVIOUS_SIMULATION_PATH,
    TURBULENCE_MODEL,
    ACOUSTIC_SURFACE,
    ACOUSTIC_SPHERE_DIAMETER,
    STUDY_PARAMETER_NAME=None,
    STUDY_PARAMETER_FILE=None,
    STUDY_PARAMETER=None,
    STATUS_CALLBACK=None,
):
    """Prepare one case directory for the OpenFOAM execution stage."""
    from cfmesh_pipeline import safe_path, prepare_geometry

    target_directory = safe_path(TARGET_DIRECTORY)
    main_directory = Path(MAIN_DIRECTORY)

    emit_status(
        STATUS_CALLBACK,
        stage="preprocessing",
        detail="copying case template",
    )

    if MODE != "AMI":
        raise ValueError(f"Unsupported OpenFOAM mode: {MODE}")

    template_by_turbulence = {
        "kOmegaSST": "Core Template AMI - kOmegaSST",
        "kEpsilon": "Core Template AMI - kEpsilon",
        "DES": "Core Template DES - kOmegaSST",
    }

    try:
        core_template_directory = (
            main_directory / template_by_turbulence[TURBULENCE_MODEL]
        )
    except KeyError as error:
        raise ValueError(f"Unsupported turbulence model: {TURBULENCE_MODEL}") from error

    prepare_case_directory(
        core_template_directory, main_directory / "Parameters", target_directory
    )

    if INIT_FROM_PREVIOUS:
        if PREVIOUS_SIMULATION_PATH is None:
            raise ValueError(
                "INIT_FROM_PREVIOUS=True requires PREVIOUS_SIMULATION_PATH."
            )

        previous_path = Path(PREVIOUS_SIMULATION_PATH)
        emit_status(
            STATUS_CALLBACK,
            stage="preprocessing",
            detail=f"copying initialization from {previous_path.name}",
        )

        init_path = target_directory / "init"
        init_path.mkdir(parents=True, exist_ok=True)

        _, latest_name = get_latest_timestep(previous_path)

        for source_name in ("constant", "system", "Parameters"):
            shutil.copytree(
                previous_path / source_name,
                init_path / source_name,
                dirs_exist_ok=True,
            )

        shutil.copytree(
            previous_path / latest_name,
            init_path / latest_name,
            dirs_exist_ok=True,
        )

    omega = RPM_COUNT * 2.0 * math.pi / 60.0
    update_parameter(
        target_directory / "Parameters" / "rotational_parameters.cpp",
        "omega_val",
        omega,
        quiet=True,
    )
    update_parameter(
        target_directory / "Parameters" / "decomposeParDict.cpp",
        "numberOfSubdomains",
        CORES_TO_USE,
        quiet=True,
    )

    control_parameters = target_directory / "Parameters" / "controlDict.cpp"

    if ACOUSTIC_SURFACE == "permeable":
        update_parameter(control_parameters, "impermeableEnabled", "no", quiet=True)
        update_parameter(control_parameters, "permeableEnabled", "yes", quiet=True)
    elif ACOUSTIC_SURFACE == "impermeable":
        update_parameter(control_parameters, "impermeableEnabled", "yes", quiet=True)
        update_parameter(control_parameters, "permeableEnabled", "no", quiet=True)
    elif ACOUSTIC_SURFACE is None:
        update_parameter(control_parameters, "impermeableEnabled", "no", quiet=True)
        update_parameter(control_parameters, "permeableEnabled", "no", quiet=True)
    else:
        raise ValueError(f"Unsupported acoustic surface: {ACOUSTIC_SURFACE}")

    tri_surface_path = target_directory / "constant" / "triSurface"
    tri_surface_path.mkdir(parents=True, exist_ok=True)

    geometry_name = SIMULATION_NAME.split(f"_{RPM_COUNT}RPM_", 1)[0]
    from tools import find_source_stls

    source = find_source_stls(target_directory.parent / "STL")[geometry_name]
    study = None
    if STUDY_PARAMETER_NAME is not None:
        study = (STUDY_PARAMETER_FILE, STUDY_PARAMETER_NAME, STUDY_PARAMETER)
    prepare_geometry(
        target_directory,
        source,
        ACOUSTIC_SURFACE,
        ACOUSTIC_SPHERE_DIAMETER,
        study=study,
    )

    emit_status(
        STATUS_CALLBACK,
        stage="preprocessing",
        detail="preprocessing complete",
        progress=100.0,
    )
    return None


if __name__ == "__main__":
    request = json.load(sys.stdin)
    request["STATUS_CALLBACK"] = lambda **fields: print(
        "PIPELINE_STATUS " + json.dumps(fields), flush=True
    )
    _prepare_case(**request)
