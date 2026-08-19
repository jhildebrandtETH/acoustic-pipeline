import math
import torch
import matplotlib.pyplot as plt

from acousticSolver.src.foamacoustics import F1ASolver
from acousticSolver.src.foamacoustics.plotting import plot_spl_spectrum
from tools import create_reference_geometry_vtk_series


def run_acoustic_solver(SIMULATION_WORKING_DIRECTORY, RPM):

    print("Acoustic solver started...")

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

    create_reference_geometry_vtk_series(
        surface_directory,
        acoustic_surface_directory,
        surface_file="propeller.vtk",
    )


    rpm = RPM
    omega_rad_s = rpm * 2.0 * math.pi / 60.0

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    solver_path = SIMULATION_WORKING_DIRECTORY / "postProcessing" / "writePatchFields"

    solver = F1ASolver.from_openfoam_vtk(
        acoustic_surface_directory,
        surface_file="propeller.vtk",
        rpm=rpm,
        permeable=False,
        moving_surface=True,
        rotation_center_m=[0.0, 0.0, 0.0],
        omega_rad_s=[0.0, 0.0, omega_rad_s],
        device=device,
        cache_dir=SIMULATION_WORKING_DIRECTORY
        / ".cache"
        / "writePatchFields_referenceGeometry",
    )

    observer_m = torch.tensor(
        [1.0, 0.0, 0.0],
        device=device,
    )

    result = solver.predict(observer_m)

    plot_spl_spectrum(
        result,
        rotations=5.0,
        blade_count=2,
    )

    output_path = SIMULATION_WORKING_DIRECTORY / "report" / "spl_spectrum.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ax = plot_spl_spectrum(
        result,
        rotations=5.0,
        blade_count=2,
    )

    # Usually the plotting helper returns a Matplotlib Axes object
    fig = ax.figure if hasattr(ax, "figure") else plt.gcf()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Acoustic spectrum saved to: {output_path.resolve()}")

    print("Acoustic solver finished...")

    return None
