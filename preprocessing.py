import math
import shutil
from pathlib import Path

import numpy as np
import trimesh

from tools import emit_status
from tools import get_latest_timestep
from tools import update_parameter


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
):
    """Prepare one case directory for the OpenFOAM execution stage."""
    target_directory = Path(TARGET_DIRECTORY)
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
        raise ValueError(
            f"Unsupported turbulence model: {TURBULENCE_MODEL}"
        ) from error

    shutil.copytree(
        core_template_directory,
        target_directory,
        dirs_exist_ok=True,
    )
    shutil.copytree(
        main_directory / "Parameters",
        target_directory / "Parameters",
        dirs_exist_ok=True,
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

    if (
        STUDY_PARAMETER_NAME is not None
        and STUDY_PARAMETER_FILE is not None
        and STUDY_PARAMETER is not None
    ):
        study_file = (
            target_directory
            / "Parameters"
            / f"{STUDY_PARAMETER_FILE}.cpp"
        )
        update_parameter(
            study_file,
            STUDY_PARAMETER_NAME,
            STUDY_PARAMETER,
            quiet=True,
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

    control_parameters = (
        target_directory / "Parameters" / "controlDict.cpp"
    )

    if ACOUSTIC_SURFACE == "permeable":
        update_parameter(
            control_parameters, "impermeableEnabled", "no", quiet=True
        )
        update_parameter(
            control_parameters, "permeableEnabled", "yes", quiet=True
        )
    elif ACOUSTIC_SURFACE == "impermeable":
        update_parameter(
            control_parameters, "impermeableEnabled", "yes", quiet=True
        )
        update_parameter(
            control_parameters, "permeableEnabled", "no", quiet=True
        )
    else:
        raise ValueError(
            f"Unsupported acoustic surface: {ACOUSTIC_SURFACE}"
        )

    tri_surface_path = target_directory / "constant" / "triSurface"
    tri_surface_path.mkdir(parents=True, exist_ok=True)

    geometry_name = SIMULATION_NAME.split(
        f"_{RPM_COUNT}RPM_", 1
    )[0]
    diameter_inch = float(geometry_name.split("x", 1)[0])
    diameter_meter = diameter_inch * 0.0254

    if ACOUSTIC_SURFACE == "permeable":
        if ACOUSTIC_SPHERE_DIAMETER is None:
            raise ValueError(
                "Permeable acoustic mode requires "
                "ACOUSTIC_SPHERE_DIAMETER."
            )

        acoustic_sphere_radius = (
            0.5
            * float(ACOUSTIC_SPHERE_DIAMETER)
            * diameter_meter
        )
        sphere = trimesh.creation.icosphere(
            subdivisions=6,
            radius=acoustic_sphere_radius,
        )
        sphere.export(tri_surface_path / "permeableSurface.stl")

    rotation = trimesh.transformations.rotation_matrix(
        np.pi / 2.0,
        [1, 0, 0],
    )

    rotary_cylinder = trimesh.creation.cylinder(
        radius=0.5 * diameter_meter * 1.2,
        height=0.05,
        sections=128,
    )
    rotary_cylinder.apply_transform(rotation)
    rotary_cylinder.export(tri_surface_path / "rotaryCylinder.stl")

    wake_offset = -0.1

    inner_cylinder = trimesh.creation.cylinder(
        radius=0.2,
        height=0.5,
        sections=128,
    )
    inner_cylinder.apply_transform(rotation)
    inner_cylinder.apply_translation([0.0, wake_offset, 0.0])
    inner_cylinder.export(tri_surface_path / "innerCylinder.stl")

    outer_cylinder = trimesh.creation.cylinder(
        radius=0.25,
        height=0.6,
        sections=128,
    )
    outer_cylinder.apply_transform(rotation)
    outer_cylinder.apply_translation([0.0, wake_offset, 0.0])
    outer_cylinder.export(tri_surface_path / "outerCylinder.stl")

    shutil.copy(
        target_directory.parent / "STL" / f"{geometry_name}.stl",
        tri_surface_path / "propeller.stl",
    )

    features_path = target_directory.parent / "FEATURES"
    feature_files = {
        "other": features_path / f"{geometry_name}_other.obj",
        "tip": features_path / f"{geometry_name}_tip.obj",
    }

    for feature_name, feature_path in feature_files.items():
        if not feature_path.is_file():
            raise FileNotFoundError(
                f"Feature file not found: {feature_path}"
            )

        shutil.copy(
            feature_path,
            tri_surface_path / f"propeller_{feature_name}.obj",
        )

    emit_status(
        STATUS_CALLBACK,
        stage="preprocessing",
        detail="preprocessing complete",
        progress=100.0,
    )
    return None
