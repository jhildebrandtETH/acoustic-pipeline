from acoustic_propagation import run_acoustic_solver
from createSimulationReport import create_simulation_report
from tools import MATPLOTLIB_LOCK
from tools import emit_status
from tools import merge_postprocessing_dat_files


def postprocessing(
    ACOUSTIC_SURFACE,
    SIMULATION_WORKING_DIRECTORY,
    RPM_COUNT,
    MODE,
    TURBULENCE_MODEL,
    STATUS_CALLBACK=None,
):
    emit_status(
        STATUS_CALLBACK,
        stage="postprocessing",
        detail="starting postprocessing",
        progress=0.0,
    )

    run_acoustic_solver(
        ACOUSTIC_SURFACE,
        SIMULATION_WORKING_DIRECTORY,
        RPM_COUNT,
        STATUS_CALLBACK=STATUS_CALLBACK,
    )

    emit_status(
        STATUS_CALLBACK,
        stage="postprocessing",
        detail="merging OpenFOAM function-object output",
        progress=55.0,
    )
    merge_postprocessing_dat_files(SIMULATION_WORKING_DIRECTORY, "forcesBlades", quiet=True)
    merge_postprocessing_dat_files(SIMULATION_WORKING_DIRECTORY, "residuals", quiet=True)
    merge_postprocessing_dat_files(SIMULATION_WORKING_DIRECTORY, "yPlus", quiet=True)

    emit_status(
        STATUS_CALLBACK,
        stage="report",
        detail="creating simulation report",
        progress=75.0,
    )

    # The report generator creates many Matplotlib figures and ReportLab pages.
    # Serialize that plotting section while other CFD cases keep running.
    with MATPLOTLIB_LOCK:
        create_simulation_report(
            case_path=SIMULATION_WORKING_DIRECTORY,
            turbulence_model=TURBULENCE_MODEL,
            rpm=RPM_COUNT,
            mode=MODE,
            quiet=True,
        )

    emit_status(
        STATUS_CALLBACK,
        stage="postprocessing",
        detail="postprocessing complete",
        progress=100.0,
    )
    return None
