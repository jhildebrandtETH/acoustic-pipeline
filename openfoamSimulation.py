import docker
from pathlib import Path
from decimal import Decimal, InvalidOperation
import threading

from tools import run_convergence_monitor
from tools import run_time_progress_monitor
from tools import run_reconstruction_progress_monitor
from tools import read_openfoam_scalar
from tools import is_mesh_ok
from tools import get_safe_timestep
from tools import processor_deletion_is_safe
from tools import safe_exec


convergence_check_interval = 1
reconstruction_check_interval = 2


def _numeric_time_directories(directory, maximum_time=None):
    """Return numeric OpenFOAM time-directory names up to maximum_time."""
    directory = Path(directory)

    if not directory.is_dir():
        return set()

    upper_limit = (
        Decimal(str(maximum_time)) if maximum_time is not None else None
    )
    time_names = set()

    for path in directory.iterdir():
        if not path.is_dir():
            continue

        try:
            time_value = Decimal(path.name)
        except InvalidOperation:
            continue

        # Preserve the original 0/ initial-condition directory.
        if time_value <= 0:
            continue

        if upper_limit is None or time_value <= upper_limit:
            time_names.add(path.name)

    return time_names


def reconstructed_history_is_complete(simulation_directory, safe_time):
    """Check that every processor0 time up to safe_time exists reconstructed."""
    simulation_directory = Path(simulation_directory)
    processor0_directory = simulation_directory / "processor0"

    expected_times = _numeric_time_directories(
        processor0_directory, maximum_time=safe_time
    )
    reconstructed_times = _numeric_time_directories(
        simulation_directory, maximum_time=safe_time
    )

    if not expected_times:
        print(
            f"No decomposed time directories up to {safe_time} were found in "
            f"'{processor0_directory}'."
        )
        return False

    missing_times = expected_times - reconstructed_times

    if missing_times:
        missing_times = sorted(missing_times, key=Decimal)
        print(
            "Resume reconstruction is incomplete. The following time "
            f"directories are still missing in the case root: {missing_times}"
        )
        return False

    print(
        f"Verified {len(expected_times)} reconstructed time directories "
        f"through safe time {safe_time}."
    )
    return True



def _run_reconstruction_with_progress(
    container,
    command,
    description,
    simulation_directory,
    maximum_time=None,
):
    """
    Run reconstructPar while displaying one-line filesystem progress.
    """

    reconstruction_stop_event = threading.Event()

    reconstruction_thread = threading.Thread(
        target=run_reconstruction_progress_monitor,
        kwargs={
            "main_sim_folder": simulation_directory,
            "maximum_time": maximum_time,
            "check_interval": reconstruction_check_interval,
            "stop_event": reconstruction_stop_event,
        },
        name="reconstruct-par-progress-monitor",
        daemon=True,
    )

    reconstruction_thread.start()

    reconstruction_successful = False

    try:
        reconstruction_successful = safe_exec(
            container,
            command,
            description,
        )

    finally:
        reconstruction_stop_event.set()

        if reconstruction_thread.is_alive():
            reconstruction_thread.join(timeout=10)

        if reconstruction_thread.is_alive():
            print(
                "WARNING: Reconstruction progress monitor did not "
                "stop within timeout."
            )

    return reconstruction_successful



def openfoamSimulation(
    simulation_name,
    simulation_working_directory,
    convergence_tolerance,
    rpm_count,
    convergence_window_revolutions,
    MODE,
    TURBULENCE_MODEL,
    NUMBER_OF_CORES,
    resume,
    MESH_ONLY,
    END_ON_MODE,
    ALLOW_BAD_MESH,
    initialize_from_previous=False,
    previous_simulation_path=None,
):
    status = False
    container = None
    monitor_thread = None
    monitor_stop_event = None

    try:
        client = docker.from_env()

        simulation_working_directory = Path(simulation_working_directory)

        source_meshes_directory = (
            simulation_working_directory.parent / "source_meshes"
        )

        my_volumes = {
            str(simulation_working_directory): {
                "bind": "/simulation",
                "mode": "rw",
            },
            str(source_meshes_directory): {
                "bind": "/source_meshes",
                "mode": "ro",
            },
        }

        container = client.containers.run(
            image="microfluidica/openfoam:13",
            name=simulation_name,
            volumes=my_volumes,
            working_dir="/simulation",
            command="bash",
            detach=True,
            tty=True,
            stdin_open=True,
        )

        print(f"Container '{container.name}' created successfully!")
        print(f"Status: {container.status}")

        # ---------------- NOT RESUME ----------------
        if not resume:

            case_separator = f"_{rpm_count}RPM_"
            mesh_name, separator_found, _ = simulation_name.partition(case_separator)

            if not separator_found or not mesh_name:
                raise ValueError(
                    f"Could not extract mesh name from simulation name "
                    f"'{simulation_name}'. Expected '{case_separator}'."
                )

            mesh_file_string = f"{mesh_name}.unv"

            ideasUnv_cmd = (
                "bash -c '"
                "source /opt/openfoam13/etc/bashrc && "
                f'ideasUnvToFoam -case /simulation '
                f'"/source_meshes/{mesh_file_string}"'
                "'"
            )

            print("ideasUnvToFoam started...")
            if not safe_exec(
                container,
                ideasUnv_cmd,
                "ideasUnvToFoam",
                print_output=True,
            ):
                return False

            print("ideasUnvToFoam finished...")

            transformPoints_cmd = (
                "bash -c 'source /opt/openfoam13/etc/bashrc && "
                "transformPoints \"scale=(0.001 0.001 0.001)\" "
                "| tee log.transformPoints'"
            )

            print("Scaling mesh from millimetres to metres...")
            if not safe_exec(
                container,
                transformPoints_cmd,
                "transformPoints",
                print_output=True,
            ):
                return False

            print("Mesh scaling finished...")

            set_wall_patch_cmd = (
            "bash -c '"
            "set -o pipefail; "
            "source /opt/openfoam13/etc/bashrc && "
            "foamDictionary constant/polyMesh/boundary "
            "-entry entry0/cubeWall/type "
            "-set wall "
            "2>&1 | tee log.setPatchTypes"
            "'"
            )

            print("Setting OpenFOAM patch types...")
            if not safe_exec(
                container,
                set_wall_patch_cmd,
                "set cubeWall patch type",
                print_output=True,
            ):
                return False
            print("Patch types configured.")


            checkMesh_cmd = "bash -c 'source /opt/openfoam13/etc/bashrc && checkMesh -allGeometry -allTopology -writeSets -setFormat vtk | tee log.checkMesh'"
            print("checkMesh started...")
            if not safe_exec(container, checkMesh_cmd, "checkMesh", print_output=True):
                return False
            print("checkMesh finished...")

            


            vtk_cell_sets_cmd = (
            "bash -c 'source /opt/openfoam13/etc/bashrc && "
            "for cell_set in "
            "underdeterminedCells "
            "oneInternalFaceCells "
            "twoInternalFacesCells; do "
            "if [ -f constant/polyMesh/sets/$cell_set ]; then "
            "echo \"Converting $cell_set to VTK...\"; "
            "foamToVTK -constant -cellSet $cell_set; "
            "else "
            "echo \"WARNING: constant/polyMesh/sets/$cell_set not found\"; "
            "fi; "
            "done'"
            )

            print("Converting problematic cell sets to VTK...")
            if not safe_exec(
                container,
                vtk_cell_sets_cmd,
                "foamToVTK cell-set conversion",
                print_output=True,
            ):
                return False
            print("Cell-set VTK conversion finished.")

            checkMesh_log_path = Path(simulation_working_directory) / "log.checkMesh"

            if not (is_mesh_ok(checkMesh_log_path) or ALLOW_BAD_MESH):
                print("Mesh is not OK... stopping this case")
                return False

            if MODE == "AMI":
                createNonConformalCouples_cmd = (
                    "bash -c 'source /opt/openfoam13/etc/bashrc && "
                    "createNonConformalCouples -fields statorAMI rotorAMI"
                    "> log.createNonConformalCouples'"
                )
                print("createNonConformalCouples started...")
                if not safe_exec(container, createNonConformalCouples_cmd, "createNonConformalCouples"):
                    return False
                print("createNonConformalCouples finished...")

            if initialize_from_previous:
                print(f"Initializing from previous case: {previous_simulation_path}")

                mapFields_cmd = (
                    "bash -c 'source /opt/openfoam13/etc/bashrc && "
                    "mapFields /simulation/init/ -consistent -sourceTime latestTime "
                    "> log.mapFields'"
                )
                print("mapFields started...")
                if not safe_exec(container, mapFields_cmd, "mapFields"):
                    return False
                print("mapFields finished...")

            if not MESH_ONLY:
                
                decomposePar_cmd = "bash -c 'source /opt/openfoam13/etc/bashrc && decomposePar > log.decomposePar'"
                print("decomposePar started...")
                if not safe_exec(container, decomposePar_cmd, "decomposePar"):
                    return False
                print("decomposePar finished...")
                

        # ---------------- RESUME ----------------
        else:
            print("Preparing to resume...")

            safe_time = get_safe_timestep(Path(simulation_working_directory))

            if safe_time is None:
                print("No safe timestep found for resume.")
                return False

            # Reconstruct every decomposed result from the first written time
            # through the selected safe time. Do not use reconstructPar -rm here:
            # processor folders must remain available until all checks pass.
            reconstruct_time_range = f":{safe_time}"
            reconstructPar_resume_cmd = (
                "bash -c 'source /opt/openfoam13/etc/bashrc && "
                f'reconstructPar -time "{reconstruct_time_range}" -noZero '
                "> log_resume.reconstructPar 2>&1'"
            )
            print(
                "Reconstructing all decomposed timesteps through "
                f"safe time {safe_time}..."
            )

            if not _run_reconstruction_with_progress(
                container=container,
                command=reconstructPar_resume_cmd,
                description="resume reconstructPar history",
                simulation_directory=simulation_working_directory,
                maximum_time=safe_time,
            ):
                return False

            print("Resume history reconstruction finished.")

            print("Checking whether the complete history was reconstructed...")
            if not reconstructed_history_is_complete(
                simulation_working_directory, safe_time
            ):
                print(
                    "Resume operation aborted. Processor folders were preserved."
                )
                return False

            print("Checking if the safe-time reconstruction is healthy...")

            path_to_control_dict_parameter = (
                Path(simulation_working_directory) / "Parameters" / "controlDict.cpp"
            )

            is_processor_deletion_safe = processor_deletion_is_safe(
                PATH_TO_CONTROL_DICT_PARAMETERS=path_to_control_dict_parameter,
                SIMULATION_DIRECTORY=simulation_working_directory,
                RESUME=True,
                TURBULENCE_MODEL=TURBULENCE_MODEL,
            )

            if not is_processor_deletion_safe:
                print(
                    f"Reconstructed data in '{simulation_working_directory}' failed integrity checks. "
                    "Resume operation aborted. Case marked as failed."
                )
                return False

            print("Reconstruction looks healthy, continue to clean up...")
            print("Deleting processor folders...")

            delete_processor_folders_cmd = "bash -c 'source /opt/openfoam13/etc/bashrc && rm -rf processor*'"
            if not safe_exec(container, delete_processor_folders_cmd, "delete processor folders after resume reconstruction"):
                return False
            print("Deleted processor folder...")

            decomposePar_resume_cmd = "bash -c 'source /opt/openfoam13/etc/bashrc && decomposePar > log_resume.decomposePar'"
            print("decomposePar started...")
            if not safe_exec(container, decomposePar_resume_cmd, "resume decomposePar"):
                return False
            print("decomposePar finished...")

        # ---------------- SOLVER ----------------
        if not MESH_ONLY:

            if resume:
                timestep_str = str(safe_time)
            else:
                timestep_str = "0"

            end_on_mode = str(END_ON_MODE).strip().lower()
            monitor_stop_event = threading.Event()

            # Avoid displaying progress from a solver log left by an older run.
            solver_log_path = (
                Path(simulation_working_directory)
                / "log.pimpleFoam"
            )

            try:
                solver_log_path.unlink(missing_ok=True)
            except OSError as error:
                print(
                    "WARNING: Could not remove old solver log before "
                    f"starting: {error}"
                )

            if end_on_mode == "time":
                parameter_control_dict = (
                    Path(simulation_working_directory)
                    / "Parameters"
                    / "controlDict.cpp"
                )

                runtime_control_dict = (
                    Path(simulation_working_directory)
                    / "system"
                    / "controlDict"
                )

                try:
                    target_end_time = read_openfoam_scalar(
                        parameter_control_dict,
                        "endTime",
                    )
                    end_time_source = parameter_control_dict
                except (FileNotFoundError, ValueError):
                    target_end_time = read_openfoam_scalar(
                        runtime_control_dict,
                        "endTime",
                    )
                    end_time_source = runtime_control_dict

                monitor_thread = threading.Thread(
                    target=run_time_progress_monitor,
                    kwargs={
                        "main_sim_folder": simulation_working_directory,
                        "end_time": target_end_time,
                        "check_interval": 5.0,
                        "stop_event": monitor_stop_event,
                    },
                    name="pimple-time-progress-monitor",
                    daemon=True,
                )

                print(
                    "Launching time-progress monitor | "
                    f"endTime={target_end_time:.6f} s | "
                    f"source={end_time_source}"
                )

            else:
                monitor_thread = threading.Thread(
                    target=run_convergence_monitor,
                    kwargs={
                        "main_sim_folder": simulation_working_directory,
                        "rpm": rpm_count,
                        "avg_history_count": convergence_window_revolutions,
                        "tolerance": convergence_tolerance,
                        "convergence_mode": end_on_mode,
                        "check_interval": convergence_check_interval,
                        "timestep": timestep_str,
                        "stop_event": monitor_stop_event,
                    },
                    name="pimple-convergence-monitor",
                    daemon=True,
                )

                print(
                    "Launching convergence monitor | "
                    f"mode={end_on_mode} | "
                    f"start timestep={timestep_str}"
                )

            monitor_thread.start()

            simRun_cmd = (
                "bash -c '"
                "set -o pipefail; "
                "source /opt/openfoam13/etc/bashrc && "
                f"mpirun --allow-run-as-root --use-hwthread-cpus "
                f"-np {NUMBER_OF_CORES} "
                "stdbuf -oL -eL "
                "foamRun -solver incompressibleFluid -parallel "
                "2>&1 | stdbuf -oL tee log.pimpleFoam"
                "'"
            )
            """
            simRun_cmd = (
                            f"bash -c 'source /opt/openfoam13/etc/bashrc && "
                            f"foamRun -solver incompressibleFluid 2>&1 | tee log.pimpleFoam'"
                        )
            """

            print("pimpleFoam solver started.")

            solver_successful = safe_exec(
                container,
                simRun_cmd,
                "pimpleFoam solver",
            )

            # Stop either monitor and wait until its dynamic terminal line has
            # been closed before printing the final solver status.
            if monitor_stop_event is not None:
                monitor_stop_event.set()

            if monitor_thread is not None and monitor_thread.is_alive():
                monitor_thread.join(timeout=10)

            if monitor_thread is not None and monitor_thread.is_alive():
                print("WARNING: Solver monitor did not stop within timeout.")

            if not solver_successful:
                return False

            print("pimpleFoam solver finished.")

            reconstructPar_cmd = (
                "bash -c 'source /opt/openfoam13/etc/bashrc && "
                "reconstructPar > log.reconstructPar 2>&1'"
            )

            print("Final reconstruction started.")

            if not _run_reconstruction_with_progress(
                container=container,
                command=reconstructPar_cmd,
                description="final reconstructPar",
                simulation_directory=simulation_working_directory,
                maximum_time=None,
            ):
                return False

            print("Final reconstruction finished.")

        else:
            print("Mesh-only mode: skipping solver.")

        # ---------------- FOAM FILE ----------------
        foam_file_cmd = "bash -c 'source /opt/openfoam13/etc/bashrc && touch sim.foam'"
        print("Creating FOAM file...")
        if not safe_exec(container, foam_file_cmd, "create FOAM file"):
            return False
        print("FOAM File created...")

        # ---------------- PROCESSOR CLEANUP CHECK ----------------
        path_to_control_dict_parameter = (
            Path(simulation_working_directory) / "Parameters" / "controlDict.cpp"
        )

        is_processor_deletion_safe = processor_deletion_is_safe(
            PATH_TO_CONTROL_DICT_PARAMETERS=path_to_control_dict_parameter,
            SIMULATION_DIRECTORY=simulation_working_directory,
            RESUME=False,
            TURBULENCE_MODEL=TURBULENCE_MODEL,
        )

        if is_processor_deletion_safe:
            print("Reconstruction looks healthy, continue to clean up...")
            print("Deleting processor folders...")

            delete_processor_folders_cmd = "bash -c 'source /opt/openfoam13/etc/bashrc && rm -rf processor*'"
            if not safe_exec(container, delete_processor_folders_cmd, "final delete processor folders"):
                return False

            print("Deleted processor folder...")
            print("Cleanup complete. System ready for the next simulation.")

        else:
            print(
                f"Reconstructed data in '{simulation_working_directory}' failed integrity checks. "
                "Processor source files were preserved for safety and manual inspection."
            )

        status = True
        return status

    except Exception as e:
        print(f"Simulation case failed unexpectedly: {e}")
        return False

    finally:
        # Always stop the monitor belonging to this case before leaving this function.
        try:
            if monitor_stop_event is not None:
                monitor_stop_event.set()

            if monitor_thread is not None and monitor_thread.is_alive():
                monitor_thread.join(timeout=5)

        except Exception as e:
            print(f"Monitor cleanup skipped/failed: {e}")

        # Always attempt container cleanup, but never let cleanup crash the pipeline.
        if container is not None:
            try:
                container.reload()

                if container.status == "running":
                    print(f"Stopping container '{container.name}'...")
                    container.stop()

                print(f"Removing container '{container.name}'...")
                container.remove(force=True)

            except Exception as e:
                print(f"Container cleanup skipped/failed: {e}")

        print(f"openFoamSimulation returns status: {status}")
