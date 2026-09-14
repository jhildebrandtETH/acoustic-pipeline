"""Prepare a case: template, initial fields, parameters, and cfMesh geometry."""
import json
import math
from pathlib import Path
import shutil
import sys

from tools import emit_status, get_latest_timestep, update_parameter, find_source_stls
from tools.preprocessing import prepare_case_directory

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
    WALL_FUNCTIONS="legacy",
    AERODYNAMICS_ONLY=False,
):
    """Prepare one case directory for the OpenFOAM execution stage."""
    from tools.cfmesh_pipeline import safe_path, prepare_geometry

    target_directory = safe_path(TARGET_DIRECTORY)
    main_directory = Path(MAIN_DIRECTORY)

    emit_status(
        STATUS_CALLBACK,
        stage="preprocessing",
        detail="copying case template",
    )

    from tools.templates import template_directory
    core_template_directory = template_directory(
        MODE, TURBULENCE_MODEL, WALL_FUNCTIONS, root=main_directory
    )
    if AERODYNAMICS_ONLY:
        ACOUSTIC_SURFACE = None
        ACOUSTIC_SPHERE_DIAMETER = None

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
    preprocessing(**request)
