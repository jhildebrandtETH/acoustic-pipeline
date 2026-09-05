"""Independent server-side ParaView stage; all implementation helpers live in tools."""

from tools import emit_status, run_visualization_job


def run_visualization(
    ACOUSTIC_SURFACE,
    SIMULATION_WORKING_DIRECTORY,
    RPM,
    STATUS_CALLBACK=None,
    config=None,
):
    """Build an auditable visual atlas without deleting simulation data.

    ``config`` overrides the optional case-local ``visualization.json`` settings.
    A failed/unavailable renderer is recorded in the atlas manifest and PDF;
    ``required=True`` additionally makes the postprocessing stage fail.
    """
    emit_status(
        STATUS_CALLBACK, stage="visualization",
        detail="waiting for server-side visualization resources", progress=0.0,
    )
    manifest = run_visualization_job(
        SIMULATION_WORKING_DIRECTORY, RPM, ACOUSTIC_SURFACE,
        config=config, status_callback=STATUS_CALLBACK,
    )
    emit_status(
        STATUS_CALLBACK, stage="visualization",
        detail=f"visual atlas {manifest['status']}: {len(manifest['views'])} views",
        progress=100.0,
    )
    return manifest
