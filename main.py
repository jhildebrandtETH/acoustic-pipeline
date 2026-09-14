"""Run an order: configure cases, then preprocess, mesh/solve, and postprocess.

Each case follows preprocessing.py -> openfoamSimulation.py -> postprocessing.py.
The scheduler in tools/scheduler.py handles parallel cases, status, and resume.
"""
from pathlib import Path
import sys
import faulthandler

sys.path.insert(0, str(Path(__file__).absolute().parent))
from tools.cli import create_parser, restore_working_directory
from tools.orders import prepare_simulation_order
from tools import RuntimeStatusRegistry
from tools import initialize_runtime_queue_states
from tools.cfmesh_orders import order_lock
from tools import run_parallel_scheduler


def main() -> None:
    restore_working_directory()
    pipeline_main_directory = Path(__file__).resolve().parent
    parser = create_parser()
    args = parser.parse_args()

    def run_order():
        convergence_monitoring_revolutions_count = 1000
        convergence_tolerance = 1e-2
        scheduler_poll_interval = 0.5
        # Load saved options or create cases from the selected inputs and settings.
        (simulations_directory, source_meshes_directory, source_meshes,
         order_store, order) = prepare_simulation_order(args, parser)

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


    # One launcher owns the order; preserve a log for native-library failures.
    try:
        with order_lock(args.sim_dir) as order_directory:
            with (order_directory / "pipeline-native-fault.log").open("a") as fault_log:
                faulthandler.enable(file=fault_log, all_threads=True)
                try:
                    run_order()
                finally:
                    faulthandler.enable()
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
