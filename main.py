import argparse
import faulthandler
import os
from pathlib import Path
import sys


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
        "Run 'cd /', then cd back into the cfMesh pipeline duplicate "
        "and retry the command."
    )


# Restore the directory before resolving a relative __file__ or --sim-dir.
restore_working_directory()
_PIPELINE_DIRECTORY = Path(__file__).resolve().parent

# Find this checkout's modules even with Python -P/-I or PYTHONSAFEPATH.
sys.path.insert(0, str(_PIPELINE_DIRECTORY))

from tools import RuntimeStatusRegistry
from tools import SimulationOrderStore
from tools import create_simulation_order
from tools import ensure_scheduler_metadata
from tools import find_source_stls
from tools import initialize_runtime_queue_states
from tools import load_simulation_order
from tools import parse_end_on
from tools import reactivate_failed_cases_for_resume
from cfmesh_pipeline import preflight, safe_path, BACKEND
from cfmesh_orders import order_lock, seed_current_stl, require_new_order
from tools import run_parallel_scheduler
from tools import save_simulation_order
from tools import validate_acoustic_arguments


def main() -> None:
    pipeline_main_directory = _PIPELINE_DIRECTORY

    parser = argparse.ArgumentParser(
        description="Dispatch OpenFOAM simulations with throughput-oriented parallel scheduling."
    )
    parser.add_argument(
        "--sim-dir",
        type=Path,
        required=True,
        help="Order directory in WSL or on a mounted drive, inside or outside the duplicate. Empty orders use the selected propeller. Existing directories are accepted; an existing simulation_order.json requires --resume.",
    )
    parser.add_argument("--rpms", nargs="+", type=int)
    parser.add_argument("--mode", choices=["AMI"])
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

    args = parser.parse_args()
    try:
        with order_lock(args.sim_dir) as order_directory:
            with (order_directory / "pipeline-native-fault.log").open("a") as fault_log:
                faulthandler.enable(file=fault_log, all_threads=True)
                try:
                    _run(args, parser, pipeline_main_directory)
                finally:
                    faulthandler.enable()  # Restore stderr before the log closes.
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        parser.error(str(error))


def _run(args, parser, pipeline_main_directory):
    convergence_monitoring_revolutions_count = 1000
    convergence_tolerance = 1e-2
    scheduler_poll_interval = 0.5
    try:
        args.end_on, args.end_on_value = parse_end_on(args.end_on)
    except ValueError as error:
        parser.error(str(error))
    try:
        simulations_directory = safe_path(args.sim_dir)
    except ValueError as error:
        parser.error(str(error))
    if not args.resume:
        require_new_order(simulations_directory)
        seed_current_stl(simulations_directory)
    source_meshes_directory = simulations_directory / "STL"

    try:
        source_meshes = find_source_stls(source_meshes_directory)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    # ----------------------------------------------------------------------
    # RESUME EXISTING ORDER
    # ----------------------------------------------------------------------
    if args.resume:
        if not simulations_directory.exists():
            parser.error(f"--sim-dir does not exist: {simulations_directory}")

        order_file = simulations_directory / "simulation_order.json"
        if not order_file.exists():
            parser.error(
                "--resume was used, but no simulation_order.json was found in "
                f"{simulations_directory}"
            )

        raw_order = load_simulation_order(simulations_directory)
        if raw_order.get("meshing_backend") != BACKEND:
            parser.error(
                "This order was not created by the native cfMesh pipeline; use a fresh order directory."
            )
        for stored_case in raw_order.get("cases", []):
            try:
                resolved_case = safe_path(simulations_directory / stored_case["folder"])
                if resolved_case.parent != simulations_directory:
                    raise ValueError(
                        "Stored case folder must be directly inside this order."
                    )
            except (ValueError, KeyError) as error:
                parser.error(str(error))
        legacy_order = "total_cores" not in raw_order

        if legacy_order and args.total_cores is None:
            parser.error(
                "Legacy simulation_order.json detected: its stored 'cores' "
                "value meant cores per case. Resume this order once with "
                "--total-cores <available cores> so the new scheduler can "
                "migrate it without guessing."
            )

        if (
            not legacy_order
            and args.total_cores is not None
            and int(args.total_cores) != int(raw_order["total_cores"])
        ):
            parser.error(
                "This simulation order already has a stored total-core budget "
                f"of {raw_order['total_cores']}. Changing the allocation during "
                "--resume is intentionally disabled because cases may already "
                "be decomposed with the stored core count."
            )

        order = ensure_scheduler_metadata(
            raw_order,
            total_cores_override=args.total_cores if legacy_order else None,
        )

        args.mode = order["mode"]
        args.meshes = order["meshes"]
        args.rpms = order["rpms"]
        args.field_init = order["field_init"]
        args.study = order["study"]
        args.study_file = order["study_file"]
        args.study_parameter = order["study_parameter"]
        args.study_values = order["study_values"]
        args.total_cores = int(order["total_cores"])
        args.mesh_only = order["mesh_only"]
        args.allow_bad_mesh = order["allow_bad_mesh"]
        args.boundary_layers = order.get("boundary_layers", "none")
        args.turbulence = order["turbulence"]
        args.end_on = order["end_on"]
        args.end_on_value = order.get("end_on_value")
        args.acoustic_surface = order.get("acoustic_surface")
        args.acoustic_sphere_diameter = order.get("acoustic_sphere_diameter")

        missing_source_meshes = [
            mesh for mesh in args.meshes if mesh not in source_meshes
        ]
        if missing_source_meshes:
            parser.error(
                "The following meshes from simulation_order.json are missing "
                "from STL/: " + ", ".join(missing_source_meshes)
            )

        try:
            preflight(args)
        except Exception as error:
            parser.error(str(error))
        save_simulation_order(simulations_directory, order)
        order_store = SimulationOrderStore(simulations_directory)
        reactivate_failed_cases_for_resume(
            order_store,
            simulations_directory,
        )
        order = order_store.snapshot()

    # ----------------------------------------------------------------------
    # CREATE NEW ORDER
    # ----------------------------------------------------------------------
    else:
        args.meshes = list(source_meshes.keys())
        missing = []

        if args.rpms is None:
            missing.append("--rpms")
        if args.mode is None:
            missing.append("--mode")
        if args.total_cores is None:
            missing.append("--total-cores")
        if args.turbulence is None:
            missing.append("--turbulence")
        if not args.mesh_only and args.acoustic_surface is None:
            missing.append("--acoustic-surface")

        if missing:
            parser.error(
                "The following arguments are required for a new simulation run: "
                + ", ".join(missing)
            )

        if args.total_cores is not None and args.total_cores < 1:
            parser.error("--total-cores must be at least 1")

        if len(set(args.rpms or [])) != len(args.rpms or []):
            parser.error("--rpms must not contain duplicate values")
        if any(rpm <= 0 for rpm in args.rpms):
            parser.error("--rpms must be positive")

        if args.field_init == "on" and args.rpms != sorted(args.rpms):
            parser.error(
                "--field-init on requires RPM values in ascending order because "
                "each case is initialized from the preceding RPM case."
            )

        validate_acoustic_arguments(parser, args)

        if args.study:
            study_missing = []
            if args.study_file is None:
                study_missing.append("--study-file")
            if args.study_parameter is None:
                study_missing.append("--study-parameter")
            if args.study_values is None:
                study_missing.append("--study-values")

            if study_missing:
                parser.error(
                    "The following arguments are required when --study is set: "
                    + ", ".join(study_missing)
                )

            if len(args.meshes) != 1 or len(args.rpms) != 1:
                parser.error(
                    "When --study is set, STL/ must contain exactly one mesh "
                    "and exactly one RPM must be provided."
                )

            if args.field_init == "on":
                parser.error(
                    "--field-init on is not supported together with --study. "
                    "Study cases are independent by design."
                )

        try:
            preflight(args)
        except Exception as error:
            parser.error(str(error))
        simulations_directory.mkdir(parents=True, exist_ok=True)
        create_simulation_order(
            args=args,
            simulations_directory=simulations_directory,
        )
        order_store = SimulationOrderStore(simulations_directory)
        order = order_store.snapshot()

    # ----------------------------------------------------------------------
    # START SCHEDULER
    # ----------------------------------------------------------------------
    registry = RuntimeStatusRegistry(order["cases"])
    initialize_runtime_queue_states(order, registry)

    print(
        "\nParallel scheduler configured:\n"
        f"  total cores       : {order['total_cores']}\n"
        f"  cores per case    : {order['cores_per_case']}"
        f"{('-' + str(order.get('max_cores_per_case'))) if order.get('max_cores_per_case', order['cores_per_case']) != order['cores_per_case'] else ''}\n"
        f"  max parallel cases: {order['max_parallel_cases']}\n"
        f"  field init        : {args.field_init}\n"
        f"  boundary layers   : {args.boundary_layers}\n"
    )

    run_parallel_scheduler(
        pipeline_main_directory=pipeline_main_directory,
        simulations_directory=simulations_directory,
        source_meshes_directory=source_meshes_directory,
        source_meshes=source_meshes,
        order_store=order_store,
        registry=registry,
        args=args,
        convergence_monitoring_revolutions_count=(
            convergence_monitoring_revolutions_count
        ),
        convergence_tolerance=convergence_tolerance,
        scheduler_poll_interval=scheduler_poll_interval,
    )

    final_order = order_store.snapshot()
    failed = [case for case in final_order["cases"] if case["status"] == "failed"]
    blocked = [case for case in final_order["cases"] if case["status"] == "blocked"]

    if failed or blocked:
        print(
            f"\nSimulation order finished with {len(failed)} failed and "
            f"{len(blocked)} blocked case(s). Use --resume after correcting "
            "the underlying issue."
        )
        raise SystemExit(1)
    else:
        import json

        overridden = []
        for case in final_order["cases"]:
            report = simulations_directory / case["folder"] / "cfmesh/mesh-status.json"
            if (
                report.exists()
                and not json.loads(report.read_text())["mesh_quality_ok"]
            ):
                overridden.append(case["folder"])
        for case in final_order["cases"]:
            acoustic_status_path = (
                simulations_directory / case["folder"] / "report/acoustic-status.json"
            )
            if acoustic_status_path.is_file():
                acoustic_status = json.loads(acoustic_status_path.read_text())
                if acoustic_status["status"] != "complete":
                    print(f"{case['folder']}: {acoustic_status['detail']}")
        if overridden:
            print(
                "\nOrder completed with explicit mesh-quality overrides: "
                + ", ".join(overridden)
            )
        else:
            print("\nAll simulations completed successfully.")


if __name__ == "__main__":
    main()
