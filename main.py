import argparse
from pathlib import Path

from preprocessing import preprocessing
from openfoamSimulation import openfoamSimulation
from postprocessing import postprocessing
from tools import create_simulation_order
from tools import load_simulation_order
from tools import update_case_status
from tools import has_timestep
from tools import reset_case_folder
from tools import get_safe_timestep


def find_source_meshes(source_meshes_directory: Path) -> dict[str, Path]:
    """
    Find all .unv meshes inside source_meshes.

    Example:
        source_meshes/mesh_a.unv -> mesh name: mesh_a
        source_meshes/mesh_b.unv -> mesh name: mesh_b

    Returns
    -------
    dict[str, Path]
        Mapping from mesh name to its complete source path.
    """

    if not source_meshes_directory.is_dir():
        raise FileNotFoundError(
            f"Source mesh directory does not exist: "
            f"{source_meshes_directory}"
        )

    mesh_files = sorted(
        path
        for path in source_meshes_directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".msh"
    )

    if not mesh_files:
        raise FileNotFoundError(
            f"No .msh mesh files were found in: "
            f"{source_meshes_directory}"
        )

    source_meshes = {}

    for mesh_path in mesh_files:
        mesh_name = mesh_path.stem

        if mesh_name in source_meshes:
            raise ValueError(
                f"Multiple source meshes use the name '{mesh_name}' in "
                f"{source_meshes_directory}"
            )

        source_meshes[mesh_name] = mesh_path

    return source_meshes


def main() -> None:
    pipeline_main_directory = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Dispatch OpenFOAM simulations."
    )

    parser.add_argument(
        "--sim-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing source_meshes and where simulation "
            "cases will be created or resumed."
        ),
    )

    parser.add_argument("--rpms", nargs="+", type=int)
    parser.add_argument("--mode", choices=["AMI", "MRF"])
    parser.add_argument(
        "--turbulence",
        choices=["kEpsilon", "kOmegaSST", "DES"],
    )
    parser.add_argument(
        "--field-init",
        default="on",
        choices=["on", "off"],
    )
    parser.add_argument("--study", action="store_true")
    parser.add_argument("--study-file")
    parser.add_argument("--study-parameter")
    parser.add_argument(
        "--study-values",
        help=(
            "Study values separated by '...'. "
            "Example: '(8 24 8)...(16 48 16)'"
        ),
    )
    parser.add_argument("--cores", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mesh-only", action="store_true")
    parser.add_argument("--allow-bad-mesh", action="store_true")
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

    args = parser.parse_args()

    simulations_directory = args.sim_dir.resolve()
    source_meshes_directory = simulations_directory / "source_meshes"

    # All available source meshes are determined from the source_meshes folder.
    try:
        source_meshes = find_source_meshes(source_meshes_directory)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    # -------- RESUME / NEW RUN VALIDATION --------
    if args.resume:
        if not simulations_directory.exists():
            parser.error(
                f"--sim-dir does not exist: {simulations_directory}"
            )

        order_file = simulations_directory / "simulation_order.json"

        if not order_file.exists():
            parser.error(
                "--resume was used, but no simulation_order.json was found "
                f"in {simulations_directory}"
            )

        order = load_simulation_order(simulations_directory)

        # The resumed order must use the new mesh-based schema.
        args.mode = order["mode"]
        args.meshes = order["meshes"]
        args.rpms = order["rpms"]
        args.field_init = order["field_init"]
        args.study = order["study"]
        args.study_file = order["study_file"]
        args.study_parameter = order["study_parameter"]
        args.study_values = order["study_values"]
        args.cores = order["cores"]
        args.mesh_only = order["mesh_only"]
        args.allow_bad_mesh = order["allow_bad_mesh"]
        args.turbulence = order["turbulence"]
        args.end_on = order["end_on"]

        missing_source_meshes = [
            mesh
            for mesh in args.meshes
            if mesh not in source_meshes
        ]

        if missing_source_meshes:
            parser.error(
                "The following meshes from simulation_order.json are "
                "missing from source_meshes: "
                + ", ".join(missing_source_meshes)
            )

        print(
            f"\n--- Resuming simulation batch from: "
            f"{simulations_directory} ---"
        )
        print(f"Mode: {args.mode}")
        print(f"Meshes: {args.meshes}")
        print(f"RPMs: {args.rpms}")
        print(f"Cores: {args.cores}")
        print(f"Study: {args.study}")

    else:
        # For a new order, all meshes found in source_meshes are used.
        args.meshes = list(source_meshes.keys())

        missing = []

        if args.rpms is None:
            missing.append("--rpms")

        if args.mode is None:
            missing.append("--mode")

        if args.cores is None:
            missing.append("--cores")

        if args.turbulence is None:
            missing.append("--turbulence")

        if missing:
            parser.error(
                "The following arguments are required for a new "
                "simulation run: "
                + ", ".join(missing)
            )

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
                    "The following arguments are required when "
                    "--study is set: "
                    + ", ".join(study_missing)
                )

            if len(args.meshes) != 1 or len(args.rpms) != 1:
                parser.error(
                    "When --study is set, source_meshes must contain "
                    "exactly one mesh and exactly one RPM must be provided."
                )

        simulations_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        create_simulation_order(
            args=args,
            simulations_directory=simulations_directory,
        )

        order = load_simulation_order(simulations_directory)

        print(
            f"\nFound source meshes: {args.meshes}"
        )

    convergence_monitoring_revolutions_count = 1000
    convergence_tolerance = 1e-3

    previous_simulation_by_mesh = {}

    # -------- UNIFIED CASE-BASED PIPELINE --------
    for case in order["cases"]:
        folder_name = case["folder"]
        mesh = case["mesh"]
        rpm = int(case["rpm"])
        mode = case["mode"]
        status = case["status"]
        is_study_case = case["study"]

        if mesh not in source_meshes:
            raise FileNotFoundError(
                f"Source mesh '{mesh}' was not found in "
                f"{source_meshes_directory}"
            )

        source_mesh_path = source_meshes[mesh]

        simulation_path = simulations_directory / folder_name
        simulation_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            f"\n--- Case: {folder_name} | Status: {status} ---"
        )
        print(f"Source mesh: {source_mesh_path}")

        # Failed cases are terminal in normal mode to avoid infinite
        # retry loops. With --resume, failed cases are reactivated.
        if status == "failed":
            if args.resume:
                print("Reactivating failed case for resume...")

                update_case_status(
                    simulations_directory,
                    folder_name,
                    "solver_running",
                )

                status = "solver_running"

            else:
                print(
                    "Skipping failed case. Use --resume to resume it."
                )
                continue

        if status == "postprocessing_done":
            print("Skipping completed case.")

            if not is_study_case:
                previous_simulation_by_mesh[mesh] = simulation_path

            continue

        previous_simulation_path = previous_simulation_by_mesh.get(mesh)

        use_previous_init = (
            args.field_init == "on"
            and previous_simulation_path is not None
            and not is_study_case
        )

        # Inner loop allows a clean restart to return to preprocessing
        # for the same case instead of moving to the next case.
        while status != "postprocessing_done":

            # ---------------- PREPROCESSING ----------------
            if status == "pending":
                print("Starting preprocessing...")

                preprocessing_kwargs = dict(
                    RPM_COUNT=rpm,
                    MAIN_DIRECTORY=pipeline_main_directory,
                    TARGET_DIRECTORY=simulation_path,
                    CORES_TO_USE=args.cores,
                    MODE=mode,
                    INIT_FROM_PREVIOUS=use_previous_init,
                    PREVIOUS_SIMULATION_PATH=previous_simulation_path,
                    TURBULENCE_MODEL=args.turbulence,
                )

                if is_study_case:
                    preprocessing_kwargs.update(
                        STUDY_PARAMETER_NAME=case["study_parameter"],
                        STUDY_PARAMETER_FILE=case["study_file"],
                        STUDY_PARAMETER=case["study_value"],
                    )

                preprocessing(**preprocessing_kwargs)

                update_case_status(
                    simulations_directory,
                    folder_name,
                    "preprocessing_done",
                )

                status = "preprocessing_done"
                continue

            # ---------------- SOLVER START ----------------
            if status == "preprocessing_done":
                print("Starting OpenFOAM...")

                update_case_status(
                    simulations_directory,
                    folder_name,
                    "solver_running",
                )

                success = openfoamSimulation(
                    resume=False,
                    simulation_name=folder_name,
                    simulation_working_directory=simulation_path,
                    convergence_tolerance=convergence_tolerance,
                    rpm_count=rpm,
                    convergence_window_revolutions=(
                        convergence_monitoring_revolutions_count
                    ),
                    MODE=mode,
                    END_ON_MODE=args.end_on,
                    TURBULENCE_MODEL=args.turbulence,
                    initialize_from_previous=use_previous_init,
                    previous_simulation_path=previous_simulation_path,
                    NUMBER_OF_CORES=args.cores,
                    MESH_ONLY=args.mesh_only,
                    ALLOW_BAD_MESH=args.allow_bad_mesh,
                )

                if success:
                    update_case_status(
                        simulations_directory,
                        folder_name,
                        "solver_done",
                    )

                    status = "solver_done"

                    if args.mesh_only:
                        update_case_status(
                            simulations_directory,
                            folder_name,
                            "postprocessing_done",
                        )

                        status = "postprocessing_done"

                else:
                    update_case_status(
                        simulations_directory,
                        folder_name,
                        "failed",
                    )

                    status = "failed"
                    break

                continue

            # ---------------- SOLVER RESUME ----------------
            if status == "solver_running":
                processor0_path = simulation_path / "processor0"

                if not has_timestep(processor0_path):
                    print(
                        "Solver marked as running but no timesteps "
                        "were found -> clean restart"
                    )

                    reset_case_folder(simulation_path)

                    update_case_status(
                        simulations_directory,
                        folder_name,
                        "pending",
                    )

                    status = "pending"
                    continue

                safe_time = get_safe_timestep(simulation_path)

                if safe_time is None:
                    print(
                        "Timesteps exist but none are usable "
                        "-> clean restart"
                    )

                    reset_case_folder(simulation_path)

                    update_case_status(
                        simulations_directory,
                        folder_name,
                        "pending",
                    )

                    status = "pending"
                    continue

                print(
                    f"Resuming solver from safe timestep: {safe_time}"
                )

                success = openfoamSimulation(
                    resume=True,
                    simulation_name=folder_name,
                    simulation_working_directory=simulation_path,
                    convergence_tolerance=convergence_tolerance,
                    rpm_count=rpm,
                    convergence_window_revolutions=(
                        convergence_monitoring_revolutions_count
                    ),
                    MODE=mode,
                    END_ON_MODE=args.end_on,
                    TURBULENCE_MODEL=args.turbulence,
                    initialize_from_previous=use_previous_init,
                    previous_simulation_path=previous_simulation_path,
                    NUMBER_OF_CORES=args.cores,
                    MESH_ONLY=args.mesh_only,
                    ALLOW_BAD_MESH=args.allow_bad_mesh,
                )

                if success:
                    update_case_status(
                        simulations_directory,
                        folder_name,
                        "solver_done",
                    )

                    status = "solver_done"

                    if args.mesh_only:
                        update_case_status(
                            simulations_directory,
                            folder_name,
                            "postprocessing_done",
                        )

                        status = "postprocessing_done"

                else:
                    update_case_status(
                        simulations_directory,
                        folder_name,
                        "failed",
                    )

                    status = "failed"
                    break

                continue

            # ---------------- POSTPROCESSING ----------------
            if status == "solver_done":
                print("Starting postprocessing...")

                postprocessing(
                    SIMULATION_WORKING_DIRECTORY=simulation_path,
                    RPM_COUNT=rpm,
                    MODE=mode,
                    TURBULENCE_MODEL=args.turbulence,
                )

                update_case_status(
                    simulations_directory,
                    folder_name,
                    "postprocessing_done",
                )

                status = "postprocessing_done"
                continue

            raise ValueError(
                f"Unknown case status for {folder_name}: {status}"
            )

        if status == "postprocessing_done" and not is_study_case:
            previous_simulation_by_mesh[mesh] = simulation_path

    print("\nAll simulations completed.")


if __name__ == "__main__":
    main()