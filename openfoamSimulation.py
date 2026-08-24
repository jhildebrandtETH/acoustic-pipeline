import os
import docker
from pathlib import Path

import threading

from tools import run_convergence_monitor
from tools import run_time_progress_monitor
from tools import run_reconstruction_progress_monitor
from tools import read_openfoam_scalar
from tools import is_mesh_ok
from tools import get_safe_timestep
from tools import processor_deletion_is_safe
from tools import safe_exec
from tools import _numeric_time_directories
from tools import reconstructed_history_is_complete
from tools import _run_reconstruction_with_progress


convergence_check_interval = 1

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


        my_volumes = {
            str(simulation_working_directory): {
                "bind": "/simulation",
                "mode": "rw",
            },
        }

        # On Linux, run the container with the host user's UID/GID so files
        # created in bind-mounted case directories are owned by the user
        # instead of root. On Windows, keep Docker's default user behavior.
        docker_user = None
        if hasattr(os, "getuid") and hasattr(os, "getgid"):
            docker_user = f"{os.getuid()}:{os.getgid()}"

        print(f"Docker user: {docker_user or 'default'}")

        container = client.containers.run(
            image="microfluidica/openfoam:13",
            name=simulation_name,
            volumes=my_volumes,
            working_dir="/simulation",
            command="bash",
            detach=True,
            tty=True,
            stdin_open=True,
            user=docker_user,
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

            mesh_file_string = f"{mesh_name}.msh"

            blockMesh_cmd = "bash -c 'source /opt/openfoam13/etc/bashrc && blockMesh > log.blockMesh'"
            print("blockMesh started...")
            if not safe_exec(container, blockMesh_cmd, "blockMesh"):
                return False
            print("blockMesh finished...")


            surfaceFeatures_cmd = "bash -c 'source /opt/openfoam13/etc/bashrc && surfaceFeatures > log.surfaceFeatures'"
            print("surfaceFeatures started...")
            if not safe_exec(container, surfaceFeatures_cmd, "surfaceFeatures"):
                return False
            print("surfaceFeatures finished...")

            decomposePar_cmd = "bash -c 'source /opt/openfoam13/etc/bashrc && decomposePar -copyZero > log.decomposePar'"
            print("decomposePar started...")
            if not safe_exec(container, decomposePar_cmd, "decomposePar"):
                return False
            print("decomposePar finished...")

            snappyHexMesh_cmd = (
                "bash -c '"
                "set -o pipefail; "
                "source /opt/openfoam13/etc/bashrc && "
                f"mpirun --allow-run-as-root --use-hwthread-cpus "
                f"-np {NUMBER_OF_CORES} "
                "snappyHexMesh -parallel -overwrite "
                "2>&1 | tee log.snappyHexMesh"
                "'"
            )
            print("snappyHexMesh started...")
            if not safe_exec(container, snappyHexMesh_cmd, "snappyHexMesh"):
                return False
            print("snappyHexMesh finished...")

            """
            ideasUnv_cmd = (
                'bash -c "'
                "source /opt/openfoam13/etc/bashrc && "
                'fluentMeshToFoam -case /simulation '
                f'"/source_meshes/{mesh_file_string}"'
                '"'
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

            

            toposet_cmd = (
             "bash -c 'source /opt/openfoam13/etc/bashrc && splitMeshRegions -makeCellZones -noFields | tee log.splitMeshRegions'"
            )

            print("TopoSet started...")
            if not safe_exec(
                container,
                toposet_cmd,
                "topoSet",
                print_output=True,
            ):
                return False
            print("toposet finished...")
            """


            
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
                    f"mpirun --oversubscribe -np {NUMBER_OF_CORES} "
                    "createNonConformalCouples -parallel rotaryRegion_slave rotaryRegion"
                    "> log.createNonConformalCouples 2>&1'"
                )

                print("createNonConformalCouples started...")

                if not safe_exec(
                    container,
                    createNonConformalCouples_cmd,
                    "createNonConformalCouples"
                ):
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

            #if not MESH_ONLY:
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
            
            reconstructPar_cmd = (
                            "bash -c 'source /opt/openfoam13/etc/bashrc && "
                            "reconstructPar > log.reconstructPar 2>&1'"
                        )
            print("Reconstructing mesh started...")
            if not safe_exec(container, reconstructPar_cmd, "reconstructPar"):
                return False
            print("Reconstructing mesh finished...")

            checkMesh_cmd = "bash -c 'source /opt/openfoam13/etc/bashrc && checkMesh -allGeometry -allTopology -writeSets -setFormat vtk | tee log.checkMesh'"
            print("checkMesh after reconstructing started...")
            if not safe_exec(container, checkMesh_cmd, "checkMesh", print_output=True):
                return False
            print("checkMesh after reconstructing finished...")
                        


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
