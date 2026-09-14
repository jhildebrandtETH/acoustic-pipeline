"""Command-line options and terminal recovery."""
import argparse
import os
from pathlib import Path

def restore_working_directory():
    """Recover a stale WSL working directory using the shell's absolute PWD."""
    try:
        Path.cwd()
        return
    except FileNotFoundError:
        # A terminal can retain a stale directory handle after a folder was
        # moved or recreated from Windows. Re-enter its current path.
        shell_directory = Path(os.environ.get("PWD", ""))

    if shell_directory.is_absolute():
        try:
            os.chdir(shell_directory)
            Path.cwd()
            print(f"Re-entered terminal working directory: {shell_directory}")
            return
        except OSError:
            pass

    raise SystemExit(
        "The terminal's working directory is no longer available. "
        "Run 'cd /', then cd back into the pipeline repository "
        "and retry the command."
    )



def create_parser():
    parser = argparse.ArgumentParser(
        description="Dispatch OpenFOAM simulations with throughput-oriented parallel scheduling."
    )
    parser.add_argument(
        "--sim-dir",
        type=Path,
        required=True,
        help="Order directory in WSL or on a mounted drive, in a dedicated output folder. Empty orders use the selected propeller. Existing directories are accepted; an existing simulation_order.json requires --resume.",
    )
    parser.add_argument("--rpms", nargs="+", type=int)
    parser.add_argument("--mode", choices=["AMI", "MRF"])
    parser.add_argument(
        "--turbulence",
        choices=["kEpsilon", "kOmegaSST", "DES"],
    )
    parser.add_argument(
        "--field-init",
        default="off",
        choices=["on", "off"],
        help=(
            "off: all cases are independent (maximum throughput). "
            "on: each geometry forms a sequential RPM initialization chain."
        ),
    )
    parser.add_argument("--wall-functions", choices=["yes", "no"],
                        help="Required for new orders: yes uses wall functions; no resolves the viscous sublayer.")
    parser.add_argument("--aerodynamics-only", action="store_true",
                        help="Disable acoustic sampling, solving and propagation; retain aerodynamic results and reports.")
    parser.add_argument("--study", action="store_true")
    parser.add_argument("--study-file")
    parser.add_argument("--study-parameter")
    parser.add_argument(
        "--study-values",
        help="Study values separated by '...'. Example: '(8 24 8)...(16 48 16)'",
    )
    parser.add_argument(
        "--total-cores",
        "--cores",
        dest="total_cores",
        type=int,
        help=(
            "Total CPU-core budget for the complete simulation order. "
            "--cores remains accepted as a backward-compatible alias."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--mesh-only",
        action="store_true",
        help="cfMesh + rotating zone + NCC + checks; stop before solver/acoustics.",
    )
    parser.add_argument(
        "--live-output",
        action="store_true",
        help="Stream full meshing/solver output instead of the dashboard.",
    )
    parser.add_argument("--allow-bad-mesh", action="store_true")
    parser.add_argument(
        "--boundary-layers",
        choices=["dict", "cfmesh", "none"],
        default="dict",
        help=(
            "dict: use cfmeshRotorDict/cfmeshStatorDict unchanged (default); "
            "none: force no layers in both; cfmesh: enable the complete native "
            "rotor workflow, with layer subdivision from its dictionary."
        ),
    )
    parser.add_argument(
        "--end-on",
        nargs="+",
        default=["convergence"],
        metavar="MODE_OR_VALUE",
        help="time SECONDS | rev REVOLUTIONS | convergence (default) | force_convergence | residual_convergence. Bare time uses controlDict.cpp.",
    )
    parser.add_argument(
        "--acoustic-surface",
        choices=["permeable", "impermeable"],
        help=("impermeable: propeller surface | permeable: enclosing sphere"),
    )
    parser.add_argument(
        "--acoustic-sphere-diameter",
        default=None,
        type=float,
        help="Permeable sphere diameter as a multiple of propeller diameter.",
    )

    return parser
