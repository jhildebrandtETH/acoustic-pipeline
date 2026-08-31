from pathlib import Path

from tools import prepare_case_for_cfmesh
from tools import report_case_stage
from tools import resolve_cfmesh_executable
from tools import run_cfmesh_boundary_layer_process


def generate_boundary_layers(
    simulation_working_directory,
    number_of_cores,
    status_callback=None,
):
    """Generate boundary layers with host-side cfMesh using system/meshDict."""
    case_directory = Path(simulation_working_directory)

    executable = resolve_cfmesh_executable()

    report_case_stage(
        status_callback,
        "cfMesh",
        "preparing case | settings from system/meshDict",
    )

    prepare_case_for_cfmesh(case_directory)

    success = run_cfmesh_boundary_layer_process(
        executable=executable,
        simulation_directory=case_directory,
        number_of_cores=number_of_cores,
        status_callback=status_callback,
    )

    if not success:
        return False

    report_case_stage(
        status_callback,
        "cfMesh",
        "cfMesh layer stage complete",
        progress=100.0,
    )
    return True
