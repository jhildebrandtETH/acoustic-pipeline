import os
import math
import threading
from pathlib import Path

import docker

from cfmesh import cfmesh
from tools import _run_reconstruction_with_progress
from tools import ensure_case_core_configuration
from tools import remove_stale_stopped_container
from tools import report_case_stage
from tools import run_openfoam_command
from tools import get_safe_timestep
from tools import processor_deletion_is_safe
from tools import read_openfoam_scalar
from tools import reconstructed_history_is_complete
from tools import run_convergence_monitor
from tools import run_time_progress_monitor
from tools import safe_exec
from tools import update_parameter


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
    STATUS_CALLBACK=None,
    END_ON_VALUE=None,
    LIVE_OUTPUT=False,
):
    """
    Run native cfMesh rotor/stator meshing, Foundation 13 NCC, and the existing
    solver/reconstruction route. Mesh-only includes the finished rotating zone
    and NCC interface. cfMesh uses allocated OpenMP threads; solver uses MPI.
    """
    convergence_check_interval = 1

    status = False
    container = None
    monitor_thread = None
    monitor_stop_event = None

    simulation_working_directory = Path(simulation_working_directory)
    number_of_cores = int(NUMBER_OF_CORES)
    parallel_run = number_of_cores > 1
    if number_of_cores < 1:
        raise ValueError("NUMBER_OF_CORES must be at least 1")

    if MESH_ONLY and resume:
        raise ValueError("--mesh-only cannot be combined with --resume")

    try:
        # The scheduler is the single source of truth for MPI decomposition.
        # Re-assert it here as well so resumed / migrated cases cannot launch
        # with a stale numberOfSubdomains value.
        ensure_case_core_configuration(
            simulation_working_directory,
            number_of_cores,
        )

        def start_foundation():
            nonlocal container
            import hashlib
            import re
            key = hashlib.sha256(str(simulation_working_directory.resolve()).encode()).hexdigest()[:10]
            container_name = "cfmesh-" + re.sub(r"[^a-zA-Z0-9_.-]", "-", simulation_name)[:80] + "-" + key
            client = docker.from_env()
            remove_stale_stopped_container(client, container_name, STATUS_CALLBACK)

            my_volumes = {
                str(simulation_working_directory): {
                    "bind": "/simulation",
                    "mode": "rw",
                },
            }

            docker_user = None
            if hasattr(os, "getuid") and hasattr(os, "getgid"):
                docker_user = f"{os.getuid()}:{os.getgid()}"

            report_case_stage(
                STATUS_CALLBACK,
                "docker",
                f"creating container | cores={number_of_cores} | "
                f"mode={'MPI' if parallel_run else 'serial'}",
            )

            container = client.containers.run(
                image="microfluidica/openfoam:13",
                name=container_name,
                volumes=my_volumes,
                working_dir="/simulation",
                command="bash",
                detach=True,
                tty=True,
                stdin_open=True,
                user=docker_user,
                labels={"acoustic-pipeline-case": simulation_name},
            )

            return container

        # ------------------------------------------------------------------
        # NEW CASE: mesh preparation
        # ------------------------------------------------------------------
        if not resume:
            from tools.cfmesh_pipeline import docker_run
            mesh_ok = cfmesh(
                start_foundation, simulation_working_directory, number_of_cores,
                ALLOW_BAD_MESH, STATUS_CALLBACK, LIVE_OUTPUT,
            )
            if MESH_ONLY:
                report_case_stage(
                    STATUS_CALLBACK, "meshOnly",
                    "cfMesh, rotating cellZone, NCC and checks complete"
                    + ("" if mesh_ok else " | QUALITY FAILED (explicit override)"),
                    progress=100.0,
                )
                status = True
                return True
            if initialize_from_previous:
                if previous_simulation_path is None:
                    raise ValueError(
                        "initialize_from_previous=True but no previous simulation path was supplied"
                    )

                map_fields_cmd = (
                    "bash -c 'source /opt/openfoam13/etc/bashrc && "
                    "mapFields /simulation/init/ -consistent -sourceTime latestTime "
                    "> log.mapFields 2>&1'"
                )
                if not run_openfoam_command(
                    container,
                    map_fields_cmd,
                    "mapFields",
                    STATUS_CALLBACK,
                    "mapFields",
                    detail=f"initializing from {Path(previous_simulation_path).name}",
                ):
                    return False

            if parallel_run:
                docker_run(container, simulation_working_directory,
                           ["decomposePar"], "log.decomposePar",
                           STATUS_CALLBACK, LIVE_OUTPUT)

        # ------------------------------------------------------------------
        # RESUME CASE
        # ------------------------------------------------------------------
        else:
            start_foundation()
            report_case_stage(STATUS_CALLBACK, "resume", "finding safe timestep")
            safe_time = get_safe_timestep(simulation_working_directory)

            if safe_time is None:
                report_case_stage(
                    STATUS_CALLBACK,
                    "resume",
                    "no safe timestep found",
                    error="No safe timestep found for resume",
                )
                return False

            processor_directories = [
                path
                for path in simulation_working_directory.glob("processor*")
                if path.is_dir()
            ]

            # A resumed case may have been decomposed with a different core
            # count by the old sequential pipeline. Reconstruct the existing
            # processor data first, independent of the NEW allocation.
            if processor_directories:
                reconstruct_time_range = f":{safe_time}"
                reconstruct_resume_cmd = (
                    "bash -c 'source /opt/openfoam13/etc/bashrc && "
                    f'reconstructPar -time "{reconstruct_time_range}" -noZero '
                    "> log_resume.reconstructPar 2>&1'"
                )

                if not _run_reconstruction_with_progress(
                    container=container,
                    command=reconstruct_resume_cmd,
                    description="resume reconstructPar history",
                    simulation_directory=simulation_working_directory,
                    maximum_time=safe_time,
                    status_callback=STATUS_CALLBACK,
                ):
                    return False

                if not reconstructed_history_is_complete(
                    simulation_working_directory,
                    safe_time,
                    status_callback=STATUS_CALLBACK,
                ):
                    return False

                path_to_control_dict_parameter = (
                    simulation_working_directory / "Parameters" / "controlDict.cpp"
                )
                if not processor_deletion_is_safe(
                    PATH_TO_CONTROL_DICT_PARAMETERS=path_to_control_dict_parameter,
                    SIMULATION_DIRECTORY=simulation_working_directory,
                    RESUME=True,
                    TURBULENCE_MODEL=TURBULENCE_MODEL,
                    status_callback=STATUS_CALLBACK,
                    maximum_time=safe_time,
                ):
                    report_case_stage(
                        STATUS_CALLBACK,
                        "resume_check",
                        "reconstructed history failed integrity checks; see log.processor_cleanup_check",
                        error="Resume reconstruction integrity check failed",
                    )
                    return False

                delete_processors_cmd = (
                    "bash -c 'source /opt/openfoam13/etc/bashrc && rm -rf processor*'"
                )
                if not run_openfoam_command(
                    container,
                    delete_processors_cmd,
                    "delete processor folders",
                    STATUS_CALLBACK,
                    "resumeCleanup",
                ):
                    return False
            else:
                report_case_stage(
                    STATUS_CALLBACK,
                    "resume",
                    f"complete fields already available at safe timestep {safe_time}",
                    progress=100.0,
                )

            # Decompose again only if this scheduler assigned more than one
            # core. A one-core resumed case continues directly from the root
            # fields after any old processor data has been reconstructed.
            if parallel_run:
                decompose_resume_cmd = (
                    "bash -c 'source /opt/openfoam13/etc/bashrc && "
                    "decomposePar > log_resume.decomposePar 2>&1'"
                )
                if not run_openfoam_command(
                    container,
                    decompose_resume_cmd,
                    "resume decomposePar",
                    STATUS_CALLBACK,
                    "decomposePar",
                ):
                    return False
            else:
                report_case_stage(
                    STATUS_CALLBACK,
                    "resume",
                    f"serial resume from safe timestep {safe_time}",
                    progress=100.0,
                )

        # ------------------------------------------------------------------
        # SOLVER
        # ------------------------------------------------------------------
        # Mesh-only cases have already returned immediately after checkMesh.
        timestep_str = str(safe_time) if resume else "0"
        end_on_mode = str(END_ON_MODE).strip().lower()
        if end_on_mode == "rev":
            end_on_mode = "time"
        monitor_stop_event = threading.Event()

        solver_log_path = simulation_working_directory / "log.pimpleFoam"
        try:
            solver_log_path.unlink(missing_ok=True)
        except OSError:
            pass

        if end_on_mode == "time":
            parameter_control_dict = (
                simulation_working_directory / "Parameters" / "controlDict.cpp"
            )
            runtime_control_dict = (
                simulation_working_directory / "system" / "controlDict"
            )

            if END_ON_VALUE is not None:
                target_end_time = float(END_ON_VALUE)
                if END_ON_MODE == "rev":
                    target_end_time *= 60.0 / float(rpm_count)
                if not math.isfinite(target_end_time) or target_end_time <= 0:
                    raise ValueError("The requested end time must be finite and positive")
                if not update_parameter(parameter_control_dict, "endTime", target_end_time, quiet=True):
                    raise ValueError(f"Cannot update endTime in {parameter_control_dict}")

            try:
                target_end_time = read_openfoam_scalar(
                    parameter_control_dict,
                    "endTime",
                )
            except (FileNotFoundError, ValueError):
                target_end_time = read_openfoam_scalar(
                    runtime_control_dict,
                    "endTime",
                )

            monitor_thread = threading.Thread(
                target=run_time_progress_monitor,
                kwargs={
                    "main_sim_folder": simulation_working_directory,
                    "end_time": target_end_time,
                    "check_interval": 5.0,
                    "stop_event": monitor_stop_event,
                    "status_callback": STATUS_CALLBACK,
                },
                name=f"{simulation_name}-time-monitor",
                daemon=True,
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
                    "status_callback": STATUS_CALLBACK,
                },
                name=f"{simulation_name}-convergence-monitor",
                daemon=True,
            )

        monitor_thread.start()
        report_case_stage(
            STATUS_CALLBACK,
            "solving",
            f"solver started | {number_of_cores} core(s)",
            progress=0.0 if end_on_mode == "time" else None,
        )

        if parallel_run:
            sim_run_cmd = (
                "bash -c '"
                "set -o pipefail; "
                "source /opt/openfoam13/etc/bashrc && "
                f"mpirun --allow-run-as-root --use-hwthread-cpus -np {number_of_cores} "
                "stdbuf -oL -eL foamRun -solver incompressibleFluid -parallel "
                "2>&1 | stdbuf -oL tee log.pimpleFoam'"
            )
        else:
            sim_run_cmd = (
                "bash -c '"
                "set -o pipefail; "
                "source /opt/openfoam13/etc/bashrc && "
                "stdbuf -oL -eL foamRun -solver incompressibleFluid "
                "2>&1 | stdbuf -oL tee log.pimpleFoam'"
            )

        solver_successful = safe_exec(
            container,
            sim_run_cmd,
            "OpenFOAM solver",
            status_callback=STATUS_CALLBACK,
            print_output=LIVE_OUTPUT,
        )

        if monitor_stop_event is not None:
            monitor_stop_event.set()
        if monitor_thread is not None and monitor_thread.is_alive():
            monitor_thread.join(timeout=10)

        if not solver_successful:
            return False

        # A zero-exit process is not proof that the solver advanced any time.
        solver_text = solver_log_path.read_text(errors="replace")
        solved_times = re.findall(r"(?m)^Time = ([0-9.eE+-]+)", solver_text)
        if not resume and not solved_times:
            raise RuntimeError(
                "OpenFOAM exited without advancing time. Check endTime, deltaT "
                "and stale 0/uniform/time metadata."
            )

        report_case_stage(STATUS_CALLBACK, "solving", "solver finished", progress=100.0)

        if parallel_run:
            reconstruct_cmd = (
                "bash -c 'source /opt/openfoam13/etc/bashrc && "
                "reconstructPar > log.reconstructPar 2>&1'"
            )
            if not _run_reconstruction_with_progress(
                container=container,
                command=reconstruct_cmd,
                description="final reconstructPar",
                simulation_directory=simulation_working_directory,
                maximum_time=None,
                status_callback=STATUS_CALLBACK,
            ):
                return False
        # ------------------------------------------------------------------
        # Final bookkeeping / cleanup
        # ------------------------------------------------------------------
        foam_file_cmd = (
            "bash -c 'source /opt/openfoam13/etc/bashrc && touch sim.foam'"
        )
        if not run_openfoam_command(
            container,
            foam_file_cmd,
            "create FOAM file",
            STATUS_CALLBACK,
            "finalizing",
        ):
            return False

        if parallel_run:
            path_to_control_dict_parameter = (
                simulation_working_directory / "Parameters" / "controlDict.cpp"
            )
            is_processor_deletion_safe = processor_deletion_is_safe(
                PATH_TO_CONTROL_DICT_PARAMETERS=path_to_control_dict_parameter,
                SIMULATION_DIRECTORY=simulation_working_directory,
                RESUME=False,
                TURBULENCE_MODEL=TURBULENCE_MODEL,
                status_callback=STATUS_CALLBACK,
            )

            if is_processor_deletion_safe:
                delete_processors_cmd = (
                    "bash -c 'source /opt/openfoam13/etc/bashrc && rm -rf processor*'"
                )
                if not run_openfoam_command(
                    container,
                    delete_processors_cmd,
                    "final processor cleanup",
                    STATUS_CALLBACK,
                    "cleanup",
                ):
                    return False
            else:
                report_case_stage(
                    STATUS_CALLBACK,
                    "cleanup",
                    "processor folders preserved; see log.processor_cleanup_check for the failed check",
                )

        status = True
        report_case_stage(STATUS_CALLBACK, "openfoam_done", "OpenFOAM stage complete", progress=100.0)
        return True

    except Exception as error:
        report_case_stage(
            STATUS_CALLBACK,
            "failed",
            f"OpenFOAM case failed: {error}",
            error=str(error),
        )
        return False

    finally:
        try:
            if monitor_stop_event is not None:
                monitor_stop_event.set()
            if monitor_thread is not None and monitor_thread.is_alive():
                monitor_thread.join(timeout=5)
        except Exception:
            pass

        if container is not None:
            try:
                container.reload()
                if container.status == "running":
                    report_case_stage(STATUS_CALLBACK, "docker", "stopping container")
                    container.stop()

                report_case_stage(STATUS_CALLBACK, "docker", "removing container")
                container.remove(force=True)
            except Exception as error:
                report_case_stage(
                    STATUS_CALLBACK,
                    "docker",
                    f"container cleanup warning: {error}",
                )

        if not status:
            report_case_stage(STATUS_CALLBACK, "failed", "OpenFOAM returned failure")
