"""Exercise the solver-to-reconstruction transition without running Docker."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import openfoamSimulation as stage


class SolverCompletionTests(unittest.TestCase):
    def run_case(self, cores, resume, log_text):
        events = []
        container = Mock(status="running")
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            case = Path(tmp)
            for name, value in (
                ("ensure_case_core_configuration", None),
                ("remove_stale_stopped_container", None),
                ("read_openfoam_scalar", 1.0),
                ("get_safe_timestep", "1"),
                ("processor_deletion_is_safe", True),
            ):
                stack.enter_context(patch.object(stage, name, return_value=value))
            stack.enter_context(patch.object(stage.threading, "Thread"))
            client = stack.enter_context(patch.object(stage.docker, "from_env"))
            client.return_value.containers.run.return_value = container
            stack.enter_context(patch.object(
                stage, "cfmesh", side_effect=lambda start, *args: bool(start())
            ))
            stack.enter_context(patch("tools.cfmesh_pipeline.docker_run"))
            reconstruct = stack.enter_context(patch.object(
                stage, "_run_reconstruction_with_progress", return_value=True
            ))
            command = stack.enter_context(patch.object(
                stage, "run_openfoam_command", return_value=True
            ))

            def solve(*args, **kwargs):
                (case / "log.pimpleFoam").write_text(log_text)
                return True

            stack.enter_context(patch.object(stage, "safe_exec", side_effect=solve))
            result = stage.openfoamSimulation(
                "case", case, 1e-2, 4000, 1000, "AMI", "kOmegaSST",
                cores, resume, False, "time", False,
                STATUS_CALLBACK=lambda **event: events.append(event),
            )
            container.stop.assert_called_once()
            container.remove.assert_called_once_with(force=True)
        return result, events, reconstruct, command

    def test_successful_solver_reaches_finalization(self):
        for cores in (1, 2):
            for resume in (False, True):
                with self.subTest(cores=cores, resume=resume):
                    result, events, reconstruct, command = self.run_case(
                        cores, resume, "Time = 1e-03\nTime = 1\nEnd\n"
                    )
                    self.assertTrue(result, events)
                    self.assertIn("openfoam_done", [event["stage"] for event in events])
                    self.assertEqual(reconstruct.call_count, int(cores > 1))
                    descriptions = [call.args[2] for call in command.call_args_list]
                    self.assertIn("create FOAM file", descriptions)
                    if cores > 1:
                        self.assertEqual(descriptions[-2:], ["create FOAM file", "final processor cleanup"])

    def test_new_run_without_timesteps_reports_actionable_error(self):
        result, events, reconstruct, command = self.run_case(2, False, "End\n")
        self.assertFalse(result)
        self.assertTrue(any(
            "OpenFOAM exited without advancing time" in event.get("error", "")
            for event in events
        ), events)
        reconstruct.assert_not_called()
        command.assert_not_called()

    def test_completed_resume_can_finalize_without_new_timesteps(self):
        result, events, reconstruct, command = self.run_case(2, True, "End\n")
        self.assertTrue(result, events)
        reconstruct.assert_called_once()
        self.assertIn("create FOAM file", [call.args[2] for call in command.call_args_list])
