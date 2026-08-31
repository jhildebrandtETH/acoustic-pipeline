import argparse
from pathlib import Path

from tools import RuntimeStatusRegistry
from tools import SimulationOrderStore
from tools import create_simulation_order
from tools import ensure_scheduler_metadata
from tools import find_source_stls
from tools import initialize_runtime_queue_states
from tools import load_simulation_order
from tools import reactivate_failed_cases_for_resume
from tools import resolve_cfmesh_executable
from tools import run_parallel_scheduler
from tools import save_simulation_order
from tools import validate_acoustic_arguments


def main() -> None:
    convergence_monitoring_revolutions_count = 1000
    convergence_tolerance = 1e-3
    scheduler_poll_interval = 0.5

    pipeline_main_directory = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Dispatch OpenFOAM simulations with throughput-oriented parallel scheduling."
    )
    parser.add_argument(
        "--sim-dir",
        type=Path,
        required=True,
        help="Simulation-order directory containing STL/ and FEATURES/.",
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
    parser.add_argument("--mesh-only", action="store_true")
    parser.add_argument("--allow-bad-mesh", action="store_true")
    parser.add_argument(
        "--boundary-layers",
        choices=["cfmesh", "none"],
        default="cfmesh",
        help=(
            "Boundary-layer method. cfmesh: reconstruct the snappy mesh, run "
            "host-side cfMesh generateBoundaryLayers, validate with OpenFOAM "
            "13, then continue. none: keep the snappy mesh unchanged."
        ),
    )
    parser.add_argument(
        "--end-on",
        choices=[
            "time",
            "convergence",
            "force_convergence",
            "residual_convergence",
        ],
        default="convergence",
    )
    parser.add_argument(
        "--acoustic-surface",
        choices=["permeable", "impermeable"],
        help=(
            "impermeable: propeller surface | permeable: enclosing sphere"
        ),
    )
    parser.add_argument(
        "--acoustic-sphere-diameter",
        default=None,
        type=float,
        help="Permeable sphere diameter as a multiple of propeller diameter.",
    )

    args = parser.parse_args()
    simulations_directory = args.sim_dir.resolve()
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
        save_simulation_order(simulations_directory, order)

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

        if args.boundary_layers == "cfmesh":
            try:
                resolve_cfmesh_executable()
            except FileNotFoundError as error:
                parser.error(str(error))

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

        if args.field_init == "on" and args.rpms != sorted(args.rpms):
            parser.error(
                "--field-init on requires RPM values in ascending order because "
                "each case is initialized from the preceding RPM case."
            )

        if args.boundary_layers == "cfmesh":
            try:
                resolve_cfmesh_executable()
            except FileNotFoundError as error:
                parser.error(str(error))

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
    failed = [
        case for case in final_order["cases"] if case["status"] == "failed"
    ]
    blocked = [
        case for case in final_order["cases"] if case["status"] == "blocked"
    ]

    if failed or blocked:
        print(
            f"\nSimulation order finished with {len(failed)} failed and "
            f"{len(blocked)} blocked case(s). Use --resume after correcting "
            "the underlying issue."
        )
    else:
        print("\nAll simulations completed successfully.")


if __name__ == "__main__":
    main()
