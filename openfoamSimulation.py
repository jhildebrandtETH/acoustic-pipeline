import os
import threading
from pathlib import Path

import docker

from cfmesh import generate_boundary_layers
from tools import _run_reconstruction_with_progress
from tools import ensure_case_core_configuration
from tools import remove_stale_stopped_container
from tools import report_case_stage
from tools import run_openfoam_command
from tools import get_safe_timestep
from tools import is_mesh_ok
from tools import processor_deletion_is_safe
from tools import read_openfoam_scalar
from tools import reconstructed_history_is_complete
from tools import run_convergence_monitor
from tools import run_time_progress_monitor
from tools import safe_exec
from tools import verify_openfoam_patch_exists


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
    BOUNDARY_LAYER_METHOD="cfmesh",
    initialize_from_previous=False,
    previous_simulation_path=None,
    STATUS_CALLBACK=None,
):
    """
    Run one complete OpenFOAM case inside its own Docker container.

    NUMBER_OF_CORES == 1 uses a true serial OpenFOAM path. Two or more cores use
    the decomposed MPI path. When cfMesh boundary layers are enabled, the
    parallel snappy mesh is reconstructed to the root case, layered by
    host-side cfMesh using this case's allocated core count, checked by
    OpenFOAM 13, and re-decomposed before NCC/solver execution. Mesh-only mode
    stops after the final selected mesh workflow, exports native snappy layer
    cells from addedCells to VTK, and leaves sim.foam for ParaView.
    """
    convergence_check_interval = 1

    status = False
    container = None
    monitor_thread = None
    monitor_stop_event = None

    simulation_working_directory = Path(simulation_working_directory)
    number_of_cores = int(NUMBER_OF_CORES)
    parallel_run = number_of_cores > 1
    boundary_layer_method = str(BOUNDARY_LAYER_METHOD).strip().lower()

    if boundary_layer_method not in {"none", "cfmesh"}:
        raise ValueError(
            "BOUNDARY_LAYER_METHOD must be one of: 'none', 'cfmesh'"
        )

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

        client = docker.from_env()
        remove_stale_stopped_container(client, simulation_name, STATUS_CALLBACK)

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
            name=simulation_name,
            volumes=my_volumes,
            working_dir="/simulation",
            command="bash",
            detach=True,
            tty=True,
            stdin_open=True,
            user=docker_user,
            labels={"acoustic-pipeline-case": simulation_name},
        )

        # ------------------------------------------------------------------
        # NEW CASE: mesh preparation
        # ------------------------------------------------------------------
        if not resume:
            block_mesh_cmd = (
                "bash -c 'source /opt/openfoam13/etc/bashrc && "
                "blockMesh > log.blockMesh 2>&1'"
            )
            if not run_openfoam_command(
                container,
                block_mesh_cmd,
                "blockMesh",
                STATUS_CALLBACK,
                "blockMesh",
            ):
                return False

            surface_features_cmd = (
                "bash -c 'source /opt/openfoam13/etc/bashrc && "
                "surfaceFeatures > log.surfaceFeatures 2>&1'"
            )
            if not run_openfoam_command(
                container,
                surface_features_cmd,
                "surfaceFeatures",
                STATUS_CALLBACK,
                "surfaceFeatures",
            ):
                return False

            if parallel_run:
                clear_initial_sets_cmd = (
                    "bash -c 'rm -rf constant/polyMesh/sets'"
                )
                if not run_openfoam_command(
                    container,
                    clear_initial_sets_cmd,
                    "clear stale mesh sets before initial decomposition",
                    STATUS_CALLBACK,
                    "decomposeParPrep",
                ):
                    return False

                decompose_cmd = (
                    "bash -c 'source /opt/openfoam13/etc/bashrc && "
                    "decomposePar -copyZero > log.decomposePar 2>&1'"
                )
                if not run_openfoam_command(
                    container,
                    decompose_cmd,
                    "decomposePar",
                    STATUS_CALLBACK,
                    "decomposePar",
                ):
                    return False

                snappy_cmd = (
                    "bash -c '"
                    "set -o pipefail; "
                    "source /opt/openfoam13/etc/bashrc && "
                    f"mpirun --allow-run-as-root --use-hwthread-cpus -np {number_of_cores} "
                    "snappyHexMesh -parallel -overwrite "
                    "2>&1 | tee log.snappyHexMesh'"
                )
            else:
                snappy_cmd = (
                    "bash -c '"
                    "set -o pipefail; "
                    "source /opt/openfoam13/etc/bashrc && "
                    "snappyHexMesh -overwrite 2>&1 | tee log.snappyHexMesh'"
                )

            if not run_openfoam_command(
                container,
                snappy_cmd,
                "snappyHexMesh",
                STATUS_CALLBACK,
                "snappyHexMesh",
            ):
                return False

            # cfMesh operates on the root constant/polyMesh. Mesh-only cases
            # also need a root mesh for checkMesh/ParaView. Therefore rebuild
            # the root mesh after parallel snappy whenever either condition
            # applies.
            root_mesh_required = MESH_ONLY or boundary_layer_method == "cfmesh"

            if parallel_run and root_mesh_required:
                reconstruct_mesh_cmd = (
                    "bash -c 'source /opt/openfoam13/etc/bashrc && "
                    "reconstructPar > log.reconstructParMesh 2>&1'"
                )
                if not run_openfoam_command(
                    container,
                    reconstruct_mesh_cmd,
                    "reconstructPar mesh",
                    STATUS_CALLBACK,
                    "reconstructPar",
                ):
                    return False

                try:
                    verify_openfoam_patch_exists(
                        simulation_working_directory,
                        "propeller",
                    )
                except (FileNotFoundError, ValueError) as error:
                    report_case_stage(
                        STATUS_CALLBACK,
                        "meshReconstructionCheck",
                        str(error),
                        error=(
                            "Parallel snappy mesh was not reconstructed to "
                            "constant/polyMesh"
                        ),
                    )
                    return False

            # Keep a separate snappy baseline when cfMesh is enabled. The
            # baseline check deliberately does not write sets because cfMesh
            # should receive a clean reconstructed root mesh.
            if boundary_layer_method == "cfmesh":
                check_mesh_log_name = "log.checkMesh.snappy"
                check_mesh_cmd = (
                    "bash -c 'source /opt/openfoam13/etc/bashrc && "
                    "checkMesh -allGeometry -allTopology "
                    "| tee log.checkMesh.snappy'"
                )
            elif MESH_ONLY:
                # Mesh-only cases have already reconstructed the parallel
                # snappy mesh to constant/polyMesh above. Run diagnostics on
                # that root mesh and write VTK surfaces/sets so the failed
                # checkMesh regions can be inspected directly in ParaView.
                check_mesh_log_name = "log.checkMesh"
                check_mesh_cmd = (
                    "bash -c 'source /opt/openfoam13/etc/bashrc && "
                    "checkMesh -allGeometry -allTopology "
                    "-writeSurfaces -surfaceFormat vtk "
                    "-writeSets -setFormat vtk "
                    "| tee log.checkMesh'"
                )
            elif parallel_run:
                # Normal non-mesh-only snappy path: validate the actual
                # decomposed processor meshes directly without reconstructing.
                check_mesh_log_name = "log.checkMesh"
                check_mesh_cmd = (
                    "bash -c '"
                    "set -o pipefail; "
                    "source /opt/openfoam13/etc/bashrc && "
                    f"mpirun --allow-run-as-root --use-hwthread-cpus "
                    f"-np {number_of_cores} "
                    "checkMesh -parallel -allGeometry -allTopology "
                    "2>&1 | tee log.checkMesh'"
                )
            else:
                check_mesh_log_name = "log.checkMesh"
                check_mesh_cmd = (
                    "bash -c 'source /opt/openfoam13/etc/bashrc && "
                    "checkMesh -allGeometry -allTopology "
                    "-writeSurfaces -surfaceFormat vtk "
                    "-writeSets -setFormat vtk "
                    "| tee log.checkMesh'"
                )

            if not run_openfoam_command(
                container,
                check_mesh_cmd,
                "checkMesh",
                STATUS_CALLBACK,
                "checkMesh",
            ):
                return False

            check_mesh_log_path = (
                simulation_working_directory / check_mesh_log_name
            )
            if not (is_mesh_ok(check_mesh_log_path, quiet=True) or ALLOW_BAD_MESH):
                report_case_stage(
                    STATUS_CALLBACK,
                    "checkMesh",
                    "mesh check failed and --allow-bad-mesh is not set",
                    error="Mesh is not OK",
                )
                return False

            # --------------------------------------------------------------
            # cfMesh boundary-layer generation
            # --------------------------------------------------------------
            if boundary_layer_method == "cfmesh":
                # The processor meshes contain the pre-cfMesh snappy result.
                # They become invalid as soon as cfMesh modifies the root mesh,
                # so discard them before the host-side layer operation.
                if parallel_run:
                    discard_snappy_processors_cmd = (
                        "bash -c 'rm -rf processor*'"
                    )
                    if not run_openfoam_command(
                        container,
                        discard_snappy_processors_cmd,
                        "discard pre-cfMesh processor meshes",
                        STATUS_CALLBACK,
                        "cfMeshPrep",
                    ):
                        return False

                if not generate_boundary_layers(
                    simulation_working_directory=simulation_working_directory,
                    number_of_cores=number_of_cores,
                    status_callback=STATUS_CALLBACK,
                ):
                    return False

                # cfMesh/polyMeshGen rewrites constant/polyMesh and removes
                # OpenFOAM zones. Recreate the rotating disconnected region
                # as the cellZone required by the rotor motion solver.
                topo_set_dict_path = (
                    simulation_working_directory / "system" / "topoSetDict"
                )
                topo_set_dict_path.write_text(
                    """FoamFile
{
    format      ascii;
    class       dictionary;
    object      topoSetDict;
}

actions
(
    {
        name    rotaryRegionCells;
        type    cellSet;
        action  new;

        source  regionToCell;
        sourceInfo
        {
            insidePoints ((0.1 0 0));
            nErode 0;
        }
    }

    {
        name    rotaryRegion;
        type    cellZoneSet;
        action  new;

        source  setToCellZone;
        sourceInfo
        {
            set rotaryRegionCells;
        }
    }
);
""",
                    encoding="utf-8",
                )

                restore_rotary_zone_cmd = (
                    "bash -c 'source /opt/openfoam13/etc/bashrc && "
                    "topoSet > log.restoreRotaryRegion 2>&1'"
                )
                if not run_openfoam_command(
                    container,
                    restore_rotary_zone_cmd,
                    "restore rotaryRegion cellZone",
                    STATUS_CALLBACK,
                    "restoreRotaryRegion",
                ):
                    return False

                rotary_zone_path = (
                    simulation_working_directory
                    / "constant"
                    / "polyMesh"
                    / "cellZones"
                )
                try:
                    rotary_zone_text = rotary_zone_path.read_text(
                        encoding="utf-8",
                        errors="ignore",
                    )
                except OSError as error:
                    report_case_stage(
                        STATUS_CALLBACK,
                        "restoreRotaryRegion",
                        f"could not read {rotary_zone_path}: {error}",
                        error="rotaryRegion cellZone verification failed",
                    )
                    return False

                if "rotaryRegion" not in rotary_zone_text:
                    report_case_stage(
                        STATUS_CALLBACK,
                        "restoreRotaryRegion",
                        "rotaryRegion was not recreated after cfMesh",
                        error="rotaryRegion cellZone missing after cfMesh",
                    )
                    return False

                # Final mesh validation is always performed by OpenFOAM 13,
                # not by cfMesh's bundled OpenFOAM runtime. Surface diagnostics
                # make skew/non-orthogonal problem locations directly
                # inspectable in ParaView.
                check_mesh_after_cfmesh_cmd = (
                    "bash -c 'source /opt/openfoam13/etc/bashrc && "
                    "checkMesh -allGeometry -allTopology "
                    "-writeSurfaces -surfaceFormat vtk "
                    "-writeSets -setFormat vtk "
                    "| tee log.checkMesh'"
                )
                if not run_openfoam_command(
                    container,
                    check_mesh_after_cfmesh_cmd,
                    "checkMesh after cfMesh",
                    STATUS_CALLBACK,
                    "checkMeshCfMesh",
                ):
                    return False

                final_mesh_log_path = (
                    simulation_working_directory / "log.checkMesh"
                )
                if not (
                    is_mesh_ok(final_mesh_log_path, quiet=True)
                    or ALLOW_BAD_MESH
                ):
                    report_case_stage(
                        STATUS_CALLBACK,
                        "checkMeshCfMesh",
                        (
                            "cfMesh-layered mesh check failed and "
                            "--allow-bad-mesh is not set"
                        ),
                        error="cfMesh-layered mesh is not OK",
                    )
                    return False

            # --mesh-only is a strict early-exit path after the requested mesh
            # generation method and final checkMesh. Never create NCCs or run
            # solver/postprocessing work. Keep sim.foam for direct ParaView.
            if MESH_ONLY:
                if boundary_layer_method == "none":
                    layer_cells_vtk_cmd = (
                        "bash -c 'source /opt/openfoam13/etc/bashrc && "
                        "foamToVTK -constant -cellSet addedCells "
                        "> log.foamToVTK.addedCells 2>&1'"
                    )
                    if not run_openfoam_command(
                        container,
                        layer_cells_vtk_cmd,
                        "export boundary-layer cells to VTK",
                        STATUS_CALLBACK,
                        "layerCellsVTK",
                    ):
                        return False

                foam_file_cmd = "bash -c 'touch sim.foam'"
                if not run_openfoam_command(
                    container,
                    foam_file_cmd,
                    "create FOAM file",
                    STATUS_CALLBACK,
                    "finalizing",
                ):
                    return False

                status = True
                report_case_stage(
                    STATUS_CALLBACK,
                    "meshOnly",
                    "mesh-only case complete after final checkMesh",
                    progress=100.0,
                )
                report_case_stage(
                    STATUS_CALLBACK,
                    "openfoam_done",
                    "OpenFOAM mesh-only stage complete",
                    progress=100.0,
                )
                return True

            # For a normal parallel cfMesh case, the processor meshes were
            # intentionally discarded before layer generation. Re-decompose
            # the final layered root mesh before NCC and the solver continue.
            if boundary_layer_method == "cfmesh" and parallel_run:
                clear_stale_sets_cmd = (
                    "bash -c 'rm -rf constant/polyMesh/sets'"
                )
                if not run_openfoam_command(
                    container,
                    clear_stale_sets_cmd,
                    "clear stale mesh sets",
                    STATUS_CALLBACK,
                    "decomposeParAfterCfMeshPrep",
                ):
                    return False

                decompose_layered_cmd = (
                    "bash -c 'source /opt/openfoam13/etc/bashrc && "
                    "decomposePar -copyZero > log.decomposeParAfterCfMesh 2>&1'"
                )
                if not run_openfoam_command(
                    container,
                    decompose_layered_cmd,
                    "decompose layered mesh",
                    STATUS_CALLBACK,
                    "decomposeParAfterCfMesh",
                ):
                    return False

            if MODE == "AMI":
                if parallel_run:
                    ncc_cmd = (
                        "bash -c 'source /opt/openfoam13/etc/bashrc && "
                        f"mpirun --oversubscribe -np {number_of_cores} "
                        "createNonConformalCouples -parallel "
                        "rotaryRegion_slave rotaryRegion "
                        "> log.createNonConformalCouples 2>&1'"
                    )
                else:
                    ncc_cmd = (
                        "bash -c 'source /opt/openfoam13/etc/bashrc && "
                        "createNonConformalCouples rotaryRegion_slave rotaryRegion "
                        "> log.createNonConformalCouples 2>&1'"
                    )

                if not run_openfoam_command(
                    container,
                    ncc_cmd,
                    "createNonConformalCouples",
                    STATUS_CALLBACK,
                    "createNonConformalCouples",
                ):
                    return False

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

        # ------------------------------------------------------------------
        # RESUME CASE
        # ------------------------------------------------------------------
        else:
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
        )

        if monitor_stop_event is not None:
            monitor_stop_event.set()
        if monitor_thread is not None and monitor_thread.is_alive():
            monitor_thread.join(timeout=10)

        if not solver_successful:
            return False

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
