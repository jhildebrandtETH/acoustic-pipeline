"""Editable meshing controls without requiring a mesher or Docker daemon."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

from tools.cfmesh_controls import DEFAULTS, read_controls
from tools import cfmesh_pipeline as pipeline


class ControlsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.case = Path(self.temp.name)
        batch = patch.object(pipeline, "query_entries", side_effect=lambda path, entries: {entry: self.query(path, entry) for entry in entries})
        batch.start()
        self.addCleanup(batch.stop)
        self.parameters = self.case / "Parameters"
        self.parameters.mkdir()
        (self.parameters / "cfmeshPipelineDict").touch()
        self.values = {
            f"{section}/{key}": str(value).lower()
            for section, entries in DEFAULTS.items() for key, value in entries.items()
        }

    def query(self, path, entry=None):
        if Path(path).name == "cfmeshPipelineDict":
            return self.values[entry]
        if entry is None:
            return "customNativeControl 42; workflowControls { stopAfter edgeExtraction; }"
        if entry == "workflowControls/stopAfter":
            return "edgeExtraction"
        raise ValueError(entry)

    def test_old_snapshot_defaults_are_independent(self):
        (self.parameters / "cfmeshPipelineDict").unlink()
        with patch.object(pipeline, "query") as query:
            controls = read_controls(self.parameters)
        query.assert_not_called()
        self.assertEqual(controls, DEFAULTS)
        controls["improveMeshQuality"]["nLoops"] = 99
        self.assertEqual(read_controls(self.parameters)["improveMeshQuality"]["nLoops"], 2)

    def test_invalid_values_report_the_entry(self):
        for entry, value in (
            ("improveMeshQuality/enabled", "maybe"),
            ("improveMeshQuality/nLoops", "-1"),
            ("improveMeshQuality/nIterations", "1.5"),
            ("interfaceProjection/movementLimitFactor", "nan"),
            ("interfaceProjection/absoluteTolerance", "-1e-10"),
            ("acceptance/minimumFaceCoverage", "1.01"),
        ):
            with self.subTest(entry=entry), patch.dict(self.values, {entry: value}), \
                    patch.object(pipeline, "query", side_effect=self.query):
                with self.assertRaisesRegex(ValueError, entry):
                    read_controls(self.parameters)

    def test_custom_controls_reach_mesh_command_and_snapshot(self):
        self.values["improveMeshQuality/nLoops"] = "7"
        self.values["improveMeshQuality/nIterations"] = "35"
        self.values["improveMeshQuality/nSurfaceIterations"] = "4"
        self.values["acceptance/minimumFaceCoverage"] = "0.97"
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                self.values["improveMeshQuality/enabled"] = str(enabled).lower()
                for role in ("rotor", "stator"):
                    region = self.case / "cfmesh" / role
                    region.mkdir(parents=True, exist_ok=True)
                    (region / "log.cartesianMesh").write_text("Stopping after step edgeExtraction")
                with patch.object(pipeline, "query", side_effect=self.query), \
                        patch.object(pipeline, "native_mesh_run") as native, \
                        patch.object(pipeline, "native_run") as edit, \
                        patch.object(pipeline, "docker_run", side_effect=RuntimeError("assembly reached")):
                    def start_foundation():
                        from tools.cfmesh_runtime import _PHASE
                        self.assertIsNone(_PHASE.get())
                        self.assertEqual([call.args[1] for call in native.call_args_list].count("stator"),
                                         2 if enabled else 1)
                        return None
                    with self.assertRaisesRegex(RuntimeError, "assembly reached"):
                        pipeline.run_mesh(start_foundation, self.case, 2, False)
                edit.assert_not_called()  # Dictionary mode preserves native controls.
                commands = [call.args[2] for call in native.call_args_list]
                expected = [["cartesianMesh"]]
                if enabled:
                    expected.append(["improveMeshQuality", "-nLoops", "7", "-nIterations", "35", "-nSurfaceIterations", "4"])
                self.assertEqual(commands, expected * 2)
                self.assertIn("customNativeControl 42;", (region / "system/meshDict").read_text())
                snapshot = json.loads((self.case / "cfmesh/pipeline-controls.json").read_text())
                self.assertEqual(snapshot["controls"]["acceptance"]["minimumFaceCoverage"], 0.97)
                self.assertNotIn("boundary_layers", snapshot)

    def test_meshing_failure_does_not_start_foundation(self):
        start_foundation = Mock()
        with patch.object(pipeline, "query", side_effect=self.query), \
                patch.object(pipeline, "native_mesh_run", side_effect=RuntimeError("meshing failed")):
            with self.assertRaisesRegex(RuntimeError, "meshing failed"):
                pipeline.run_mesh(start_foundation, self.case, 2, False)
        start_foundation.assert_not_called()
        from tools.cfmesh_runtime import _PHASE
        self.assertIsNone(_PHASE.get())


if __name__ == "__main__":
    unittest.main()
