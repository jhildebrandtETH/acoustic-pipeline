"""Editable meshing controls without requiring a mesher or Docker daemon."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.cfmesh_controls import DEFAULTS, read_controls
from tools import cfmesh_pipeline as pipeline


class ControlsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.case = Path(self.temp.name)
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
                region = self.case / "cfmesh/rotor"
                region.mkdir(parents=True, exist_ok=True)
                (region / "log.cartesianMesh").write_text("Stopping after step edgeExtraction")
                with patch.object(pipeline, "query", side_effect=self.query), \
                        patch.object(pipeline, "native_mesh_run") as native, \
                        patch.object(pipeline, "native_run") as edit, \
                        patch.object(pipeline, "docker_run", side_effect=RuntimeError("assembly reached")):
                    with self.assertRaisesRegex(RuntimeError, "assembly reached"):
                        pipeline.run_mesh(None, self.case, 2, False)
                edit.assert_not_called()  # Dictionary mode preserves native controls.
                commands = [call.args[2] for call in native.call_args_list]
                expected = [["cartesianMesh"]]
                if enabled:
                    expected.append(["improveMeshQuality", "-nLoops", "7", "-nIterations", "35", "-nSurfaceIterations", "4"])
                self.assertEqual(commands, expected)
                self.assertIn("customNativeControl 42;", (region / "system/meshDict").read_text())
                snapshot = json.loads((self.case / "cfmesh/pipeline-controls.json").read_text())
                self.assertEqual(snapshot["controls"]["acceptance"]["minimumFaceCoverage"], 0.97)
                self.assertNotIn("boundary_layers", snapshot)


if __name__ == "__main__":
    unittest.main()
