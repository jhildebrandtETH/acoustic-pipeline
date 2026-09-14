from datetime import datetime
import json
import signal
import subprocess
import sys
import shutil
import tempfile
from pathlib import Path


from tools import emit_status


def prepare_case_directory(template_directory, parameters_directory, target_directory):
    """Copy a fresh case, preserving any previous results in a sibling folder."""
    from tools.cfmesh_pipeline import safe_path

    target_directory = safe_path(target_directory)
    # DrvFS can return EEXIST for a deleted case name that stat cannot see.
    # Build under a unique sibling name, then rename the populated directory.
    with tempfile.TemporaryDirectory(
        prefix=".case-", dir=target_directory.parent
    ) as temporary_directory:
        staging_directory = Path(temporary_directory)
        shutil.copytree(template_directory, staging_directory, dirs_exist_ok=True)
        # A template may contain a saved timestep that overrides deltaT.
        # New cases must start from the current control dictionary.
        (staging_directory / "0/uniform/time").unlink(missing_ok=True)
        shutil.copytree(
            parameters_directory, staging_directory / "Parameters", dirs_exist_ok=True
        )

        if target_directory.exists():
            if not target_directory.is_dir():
                raise FileExistsError(
                    f"Case path is not a directory: {target_directory}"
                )
            archived_directory = safe_path(
                target_directory.with_name(
                    target_directory.name
                    + "_PREVIOUS_"
                    + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                )
            )
            target_directory.rename(archived_directory)
            print(f"Previous case preserved in {archived_directory}")

        staging_directory.rename(target_directory)

def run_preprocessing(
    SIMULATION_NAME,
    RPM_COUNT,
    MAIN_DIRECTORY,
    TARGET_DIRECTORY,
    CORES_TO_USE,
    MODE,
    INIT_FROM_PREVIOUS,
    PREVIOUS_SIMULATION_PATH,
    TURBULENCE_MODEL,
    ACOUSTIC_SURFACE,
    ACOUSTIC_SPHERE_DIAMETER,
    STUDY_PARAMETER_NAME=None,
    STUDY_PARAMETER_FILE=None,
    STUDY_PARAMETER=None,
    STATUS_CALLBACK=None,
    LIVE_OUTPUT=False,
    WALL_FUNCTIONS="legacy",
    AERODYNAMICS_ONLY=False,
):
    """Run geometry preparation in its own process and retain native crash output."""
    from tools.cfmesh_pipeline import safe_path

    arguments = dict(
        WALL_FUNCTIONS=WALL_FUNCTIONS,
        AERODYNAMICS_ONLY=AERODYNAMICS_ONLY,
        SIMULATION_NAME=SIMULATION_NAME,
        RPM_COUNT=RPM_COUNT,
        MAIN_DIRECTORY=MAIN_DIRECTORY,
        TARGET_DIRECTORY=TARGET_DIRECTORY,
        CORES_TO_USE=CORES_TO_USE,
        MODE=MODE,
        INIT_FROM_PREVIOUS=INIT_FROM_PREVIOUS,
        PREVIOUS_SIMULATION_PATH=PREVIOUS_SIMULATION_PATH,
        TURBULENCE_MODEL=TURBULENCE_MODEL,
        ACOUSTIC_SURFACE=ACOUSTIC_SURFACE,
        ACOUSTIC_SPHERE_DIAMETER=ACOUSTIC_SPHERE_DIAMETER,
        STUDY_PARAMETER_NAME=STUDY_PARAMETER_NAME,
        STUDY_PARAMETER_FILE=STUDY_PARAMETER_FILE,
        STUDY_PARAMETER=STUDY_PARAMETER,
    )
    case_directory = safe_path(TARGET_DIRECTORY)
    log_path = safe_path(
        case_directory.parent / (case_directory.name + ".preprocessing.log")
    )
    emit_status(STATUS_CALLBACK, stage="preprocessing", detail="preparing case")
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-u", "-X", "faulthandler", str(Path(__file__).resolve().parent.parent / "preprocessing.py")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        process.stdin.write(json.dumps(arguments, default=str))
        process.stdin.close()
        for line in process.stdout:
            log.write(line)
            log.flush()
            if line.startswith("PIPELINE_STATUS "):
                emit_status(
                    STATUS_CALLBACK, **json.loads(line[len("PIPELINE_STATUS ") :])
                )
            elif LIVE_OUTPUT:
                print(line, end="", flush=True)
        return_code = process.wait()
    if return_code:
        reason = (
            signal.Signals(-return_code).name
            if return_code < 0
            else f"exit code {return_code}"
        )
        raise RuntimeError(f"Preprocessing failed ({reason}); see {log_path}")
