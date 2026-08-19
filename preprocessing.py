## --->  PE0_FILE_PATH, RPM, TARGET_DIRECTORY, CORE_TEMPLATE_DIRECTORY... --> THIS FUNCTION ---> READY TO RUN OPENFOAM SIMULATION FOR PARTRICULAR CASE

import shutil
import os
import math
import numpy as np
import trimesh
from pathlib import Path
from tools import get_latest_timestep
from tools import update_parameter
from tools import read_openfoam_scalar
from tools import get_y_domain_height


interpolation_points = 100




# Usage


def preprocessing(SIMULATION_NAME, RPM_COUNT, MAIN_DIRECTORY, TARGET_DIRECTORY, CORES_TO_USE, MODE, INIT_FROM_PREVIOUS, PREVIOUS_SIMULATION_PATH, TURBULENCE_MODEL, ACOUSTIC_SURFACE, ACOUSTIC_SPHERE_DIAMETER, STUDY_PARAMETER_NAME = None, STUDY_PARAMETER_FILE = None, STUDY_PARAMETER = None):

 
    #1. duplicate right Core Template to target directory (AMI or RMF approach)

    
    if MODE == "AMI":

        if TURBULENCE_MODEL == "kOmegaSST":
            core_template_directory = os.path.join(MAIN_DIRECTORY, "Core Template AMI - kOmegaSST")
        
        elif TURBULENCE_MODEL == "kEpsilon":
            core_template_directory = os.path.join(MAIN_DIRECTORY, "Core Template AMI - kEpsilon")

        elif TURBULENCE_MODEL == "DES":
            core_template_directory = os.path.join(MAIN_DIRECTORY, "Core Template DES - kOmegaSST")

    else:
        print("Unknown mode was passed to the pipeline...")
        return None


    shutil.copytree(core_template_directory, TARGET_DIRECTORY, dirs_exist_ok=True)
    
    # copy parameters folder to case folder

    parameters_path_main = os.path.join(MAIN_DIRECTORY, 'Parameters')

    shutil.copytree(parameters_path_main, os.path.join(TARGET_DIRECTORY, 'Parameters'))

    #

    ## Copy init related files to target


    if INIT_FROM_PREVIOUS:

        # create init folder in case
        
        init_path = Path(TARGET_DIRECTORY) / "init"
        init_path.mkdir()


        # copy relevant subfolders of previous case to new init folder to this case
        constant_init_path = Path(PREVIOUS_SIMULATION_PATH) / "constant"
        system_init_path = Path(PREVIOUS_SIMULATION_PATH) / "system"
        parameters_init_path = Path(PREVIOUS_SIMULATION_PATH) / "Parameters"

        # get latest Timestep that is then initialized
        latest_time, latest_name = get_latest_timestep(PREVIOUS_SIMULATION_PATH)
        timestep_init_path = Path(PREVIOUS_SIMULATION_PATH) / latest_name

        shutil.copytree(constant_init_path, init_path / "constant")
        shutil.copytree(system_init_path, init_path / "system")
        shutil.copytree(parameters_init_path, init_path / "Parameters")
        shutil.copytree(timestep_init_path, init_path / latest_name)
    ##


    #2. know about what simulation we are talking about (geometry facts & RPM)
    
    # adapt study parameter in case file in study mode

    if STUDY_PARAMETER_NAME is not None and STUDY_PARAMETER_FILE is not None and STUDY_PARAMETER is not None:

        file_name = STUDY_PARAMETER_FILE + ".cpp"

        file_path = Path(TARGET_DIRECTORY) / "Parameters" / file_name

        update_parameter(file_path, STUDY_PARAMETER_NAME, STUDY_PARAMETER)


    #3. adapt all exisiting parameters based on certain rules

    rotational_parameters_file_path = os.path.join(TARGET_DIRECTORY, 'Parameters', 'rotational_parameters.cpp')

    omega = RPM_COUNT * 2 * math.pi / 60

    update_parameter(rotational_parameters_file_path, 'omega_val', omega)


    decomposeParDict_parameters_file_path = os.path.join(TARGET_DIRECTORY, 'Parameters', 'decomposeParDict.cpp')

    update_parameter(decomposeParDict_parameters_file_path, 'numberOfSubdomains', CORES_TO_USE)


    """
    
   # Autonomous y+ targeting


    #applying boudary layer theory (prantl & schlichting)

    rho_file_path = Path(TARGET_DIRECTORY) / 'system' / 'forces'
    nu_file_path = Path(TARGET_DIRECTORY) / 'constant' / 'transportProperties'
    block_mesh_dict_path = Path(TARGET_DIRECTORY) / 'system' / 'blockMeshDict'
    block_mesh_parameters_path = Path(TARGET_DIRECTORY) / 'Parameters' / 'blockMeshDict.cpp'


    inner_reference_radius = 0.015
    reference_chord_length = 0.016
    y_plus_target = 30

    U_rel = inner_reference_radius * omega
    rho = read_openfoam_scalar(rho_file_path, 'rhoInf')
    nu = read_openfoam_scalar(nu_file_path, 'nu')
    Re = (U_rel*reference_chord_length)/(nu)
    C_f = 0.0592*math.pow(Re,(-1/5)) # Prantl-Schlichting equation
    tau_w = 0.5*rho*(U_rel**2)*C_f
    u_tau = math.sqrt((tau_w)/(rho))

    h = (y_plus_target * nu)/(u_tau)#y+ definition


    L_y = get_y_domain_height(block_mesh_dict_path)
    
    text = block_mesh_parameters_path.read_text(errors="ignore")
    match = re.search(
    r"blocks_resolution\s*\(\s*(\d+)\s+(\d+)\s+(\d+)\s*\)\s*;",
    text
    )

    N_y = int(match.group(2))
    
    delta_y_0 = (L_y)/(N_y)

    n = math.log((delta_y_0)/(h), 2)
    n_floor = math.floor(n)
    print(f"Autonomous y+ targeting: First layer thickness determined to: {h}m")
    print(f"Autonomous y+ targeting: Refinement levels determined to: {n} -> {n_floor}")

    first_layer_thickness = h
    first_layer_thickness_string = f'{first_layer_thickness}'

    propeller_region_refinement_level = n_floor
    propeller_region_refinement_level_string = f'{propeller_region_refinement_level}'

    propeller_surface_refinement_level = n_floor
    propeller_surface_refinement_level_string = f'({propeller_surface_refinement_level} {propeller_surface_refinement_level})'

    snappyHexMeshDict_parameters_file_path = os.path.join(TARGET_DIRECTORY, 'Parameters', 'snappyHexMeshDict.cpp')
    
    update_parameter(snappyHexMeshDict_parameters_file_path, 'firstLayerThickness', first_layer_thickness_string)
    update_parameter(snappyHexMeshDict_parameters_file_path, 'propellerTipRegionLevel', propeller_region_refinement_level_string)
    update_parameter(snappyHexMeshDict_parameters_file_path, 'propellerTipSurfaceRefinementLevel', propeller_surface_refinement_level_string)
    
    """
    # Acoustic calculations setup

    controlDict_parameters_file_path = os.path.join(TARGET_DIRECTORY, 'Parameters', 'controlDict.cpp')

    if ACOUSTIC_SURFACE == "permeable":
        update_parameter(controlDict_parameters_file_path, 'impermeableEnabled', 'no')
        update_parameter(controlDict_parameters_file_path, 'permeableEnabled', 'yes')
    elif ACOUSTIC_SURFACE == "impermeable":
        update_parameter(controlDict_parameters_file_path, 'impermeableEnabled', 'yes')
        update_parameter(controlDict_parameters_file_path, 'permeableEnabled', 'no')


    triSurface_path = TARGET_DIRECTORY / "constant" / "triSurface"

    triSurface_path.mkdir(parents = True, exist_ok = True)

    diameter_inch = float(SIMULATION_NAME.split("x", 1)[0])
    diameter_meter = diameter_inch * 0.0254

    acoustic_sphere_diameter = 0.5*ACOUSTIC_SPHERE_DIAMETER * diameter_meter

    sphere = trimesh.creation.icosphere(
        subdivisions= 6,
        radius=acoustic_sphere_diameter,
    )
    sphere.export(triSurface_path / "permeableSurface.stl")


    rotation = trimesh.transformations.rotation_matrix(
    np.pi / 2,
    [1, 0, 0]
    )



    rotaryCylinder = trimesh.creation.cylinder(
        radius= 0.5*diameter_meter*1.2,
        height=0.05,
        sections=128,
    )
    rotaryCylinder.apply_transform(rotation)
    rotaryCylinder.export(triSurface_path / "rotaryCylinder.stl")

    innerCylinder = trimesh.creation.cylinder(
        radius= 0.2,
        height=0.5,
        sections=128,
    )

    wake_offset = -0.1

    innerCylinder.apply_transform(rotation)
    innerCylinder.apply_translation([0.0, wake_offset, 0.0])
    innerCylinder.export(triSurface_path / "innerCylinder.stl")

    outerCylinder = trimesh.creation.cylinder(
        radius= 0.25,
        height=0.6,
        sections=128,
    )
    outerCylinder.apply_transform(rotation)
    outerCylinder.apply_translation([0.0, wake_offset, 0.0])
    outerCylinder.export(triSurface_path / "outerCylinder.stl")



    stl_name = SIMULATION_NAME.rsplit("_", 2)[0]

    # Copy propeller STL
    stl_path = TARGET_DIRECTORY.parent / "STL" / f"{stl_name}.stl"
    target_stl_path = triSurface_path / "propeller.stl"

    shutil.copy(stl_path, target_stl_path)


    # Copy propeller feature edges
    features_path = TARGET_DIRECTORY.parent / "FEATURES"

    feature_files = {
        "other": features_path / f"{stl_name}_other.obj",
        "tip": features_path / f"{stl_name}_tip.obj",
    }

    for feature_name, feature_path in feature_files.items():

        if not feature_path.is_file():
            raise FileNotFoundError(
                f"Feature file not found: {feature_path}"
            )

        target_feature_path = triSurface_path / f"propeller_{feature_name}.obj"
        shutil.copy(feature_path, target_feature_path)

    return None