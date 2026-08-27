import math

import matplotlib.pyplot as plt
import torch

from acousticSolver.src.foamacoustics import F1ASolver
from acousticSolver.src.foamacoustics.plotting import plot_spl_spectrum
from tools import MATPLOTLIB_LOCK
from tools import create_reference_geometry_vtk_series
from tools import emit_status


def run_acoustic_solver(
    ACOUSTIC_SURFACE,
    SIMULATION_WORKING_DIRECTORY,
    RPM,
    STATUS_CALLBACK=None,
):
    emit_status(
        STATUS_CALLBACK,
        stage="acoustics",
        detail="preparing acoustic surface data",
        progress=0.0,
    )

    if ACOUSTIC_SURFACE == "impermeable":
        surface_directory = (
            SIMULATION_WORKING_DIRECTORY
            / "postProcessing"
            / "writePatchFields"
        )
        acoustic_surface_directory = (
            SIMULATION_WORKING_DIRECTORY
            / "postProcessing"
            / "writePatchFields_referenceGeometry"
        )
        surface_file = "propeller.vtk"
        permeable = False
    else:
        surface_directory = (
            SIMULATION_WORKING_DIRECTORY
            / "postProcessing"
            / "writePermeableSurfaceFields"
        )
        acoustic_surface_directory = (
            SIMULATION_WORKING_DIRECTORY
            / "postProcessing"
            / "writePermeableSurfaceFields_referenceGeometry"
        )
        surface_file = "permeableSurface.vtk"
        permeable = True

    create_reference_geometry_vtk_series(
        surface_directory,
        acoustic_surface_directory,
        surface_file=surface_file,
    )

    rpm = RPM
    omega_rad_s = rpm * 2.0 * math.pi / 60.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    emit_status(
        STATUS_CALLBACK,
        stage="acoustics",
        detail=f"loading FW-H data on {device}",
        progress=25.0,
    )

    cache_name = (
        "writePermeableSurfaceFields_referenceGeometry"
        if permeable
        else "writePatchFields_referenceGeometry"
    )

    solver = F1ASolver.from_openfoam_vtk(
        acoustic_surface_directory,
        surface_file=surface_file,
        rpm=rpm,
        permeable=permeable,
        moving_surface=True,
        rotation_center_m=[0.0, 0.0, 0.0],
        omega_rad_s=[0.0, omega_rad_s, 0.0],
        device=device,
        cache_dir=SIMULATION_WORKING_DIRECTORY / ".cache" / cache_name,
    )

    observer_m = torch.tensor([1.0, 0.0, 0.0], device=device)

    emit_status(
        STATUS_CALLBACK,
        stage="acoustics",
        detail="predicting observer pressure",
        progress=55.0,
    )
    result = solver.predict(observer_m)

    output_path = SIMULATION_WORKING_DIRECTORY / "report" / "spl_spectrum.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Matplotlib uses global process state. The CFD cases may run concurrently,
    # but plotting is serialized to prevent figures from different worker
    # threads interfering with each other.
    emit_status(
        STATUS_CALLBACK,
        stage="acoustics",
        detail="creating SPL spectrum",
        progress=85.0,
    )
    with MATPLOTLIB_LOCK:
        ax = plot_spl_spectrum(
            result,
            rotations=5.0,
            blade_count=2,
        )
        fig = ax.figure if hasattr(ax, "figure") else plt.gcf()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    emit_status(
        STATUS_CALLBACK,
        stage="acoustics",
        detail="acoustic spectrum saved",
        progress=100.0,
    )

    return None
