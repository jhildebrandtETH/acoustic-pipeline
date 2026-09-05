"""Host checks plus an opt-in analytic end-to-end test with real ParaView.

Run: python -m unittest discover -s tests -v
Include rendering: RUN_PARAVIEW_TESTS=1 (PowerShell: $env:RUN_PARAVIEW_TESTS='1').
"""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import tools
from visualization import run_visualization


class VisualizationTests(unittest.TestCase):
    def test_configuration_and_units(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "10x7E_600RPM_AMI"
            (case / "0").mkdir(parents=True)
            (case / "0" / "p").write_text("dimensions [0 2 -2 0 0 0 0]; // NOT Pa")
            self.assertEqual(tools._pvvis_pressure_units(case)[0], "m^2/s^2")
            self.assertAlmostEqual(tools.visualization_settings(case, 600, "impermeable")["diameter_m"], .254)
            (case / "visualization.json").write_text(json.dumps({"surface_phases": 6}))
            self.assertEqual(tools.visualization_settings(case, 600, "impermeable")["surface_phases"], 6)
            self.assertEqual(tools.visualization_settings(case, 600, "impermeable", {"surface_phases": 8})["surface_phases"], 8)
            for config in ({"surface_phases": 0}, {"enabled": "false"}, {"q_over_omega2": [-1]},
                           {"diameter_m": float("nan")}, {"color_ranges": {"p": [2, 1]}},
                           {"color_ranges": {"misspelled": [0, 1]}}, {"unknown": 2}):
                with self.assertRaises(ValueError):
                    tools.visualization_settings(case, 600, "impermeable", config)

    def test_sparse_time_selection_does_not_duplicate_frames(self):
        entries = [(0.0, "zero"), (.01, "first"), (.2, "last")]
        selected = tools._pvvis_select(entries, 600, 12)
        self.assertEqual(len(selected), len({t for t, _ in selected}))
        self.assertEqual(selected[-1][0], .2)
        self.assertTrue(all(t >= .1 for t, _ in selected))

    def test_missing_renderer_replaces_stale_manifest_and_strict_mode_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "report" / "visuals"
            root.mkdir(parents=True)
            (root / "manifest.json").write_text(json.dumps({"status": "complete", "views": [{"image": "stale.png"}]}))
            with patch.object(tools, "find_paraview_executable", side_effect=FileNotFoundError("No ParaView")):
                result = run_visualization("impermeable", tmp, 600)
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["views"], [])
                self.assertIn("No ParaView", result["warnings"])
                with self.assertRaises(RuntimeError):
                    run_visualization("impermeable", tmp, 600, config={"required": True})
            with patch.object(tools, "find_paraview_executable") as find:
                self.assertEqual(run_visualization("impermeable", tmp, 600, config={"enabled": False})["status"], "disabled")
                find.assert_not_called()

    def test_pipeline_orders_visuals_before_report(self):
        events = []
        acoustic = types.ModuleType("acoustic_propagation")
        acoustic.run_acoustic_solver = lambda *a, **k: events.append("acoustics")
        report = types.ModuleType("createSimulationReport")
        report.create_simulation_report = lambda *a, **k: events.append("report")
        spec = importlib.util.spec_from_file_location("postprocessing_test", Path(__file__).parents[1] / "postprocessing.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, acoustic_propagation=acoustic, createSimulationReport=report):
            spec.loader.exec_module(module)
        with patch.object(module, "merge_postprocessing_dat_files", side_effect=lambda *a, **k: events.append("merge")), \
             patch.object(module, "run_visualization", side_effect=lambda *a, **k: events.append("visuals")):
            module.postprocessing("impermeable", Path("case"), 600, "AMI", "kOmegaSST")
        self.assertEqual(events, ["acoustics", "merge", "merge", "merge", "visuals", "report"])

    @unittest.skipUnless(os.environ.get("RUN_PARAVIEW_TESTS") == "1", "Opt-in real ParaView rendering test")
    def test_real_paraview_analytic_statistics_and_pdf(self):
        from reportlab.pdfgen.canvas import Canvas
        from pypdf import PdfReader

        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / "10x7E_600RPM_AMI"
            subprocess.run([sys.executable, str(Path(__file__).parent / "fixtures" / "visualization_case.py"), str(case)], check=True, capture_output=True)
            original_files = {str(p.relative_to(case)): hashlib.sha256(p.read_bytes()).hexdigest() for p in case.rglob("*") if p.is_file()}
            result = run_visualization("impermeable", case, 600, config={
                "image_resolution": [1200, 720], "surface_phases": 3, "volume_phases": 2,
                "wake_stations_D": [0, .5], "q_over_omega2": [.5], "statistics_revolutions": 2,
            })
            self.assertEqual(result["status"], "complete", result["warnings"])
            self.assertGreaterEqual(len(result["views"]), 75)
            for relative, digest in original_files.items():
                self.assertEqual(hashlib.sha256((case / relative).read_bytes()).hexdigest(), digest)
            volume = [v for v in result["views"] if v["title"].startswith("Flow slice - speed")]
            self.assertEqual({v["time_s"] for v in volume}, {.1, .2})
            self.assertEqual(len({tuple(v["color_range"]) for v in volume}), 1)
            checks = Path(tmp) / "analytic_checks.py"
            checks.write_text('''import json, runpy, sys
from pathlib import Path
import numpy as np
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader
from vtkmodules.util.numpy_support import vtk_to_numpy
run_dir = Path(sys.argv[1])
worker = runpy.run_path(str(run_dir / "render_visuals.py"))
settings = json.loads((run_dir / "settings.json").read_text())
reader = vtkXMLPolyDataReader()
reader.SetFileName(str(run_dir / "surface_statistics.vtp"))
reader.Update()
attrs = reader.GetOutput().GetCellData()
np.testing.assert_allclose(vtk_to_numpy(attrs.GetArray("p_mean")), [101325.2,101325.4], atol=1e-7, rtol=0)
np.testing.assert_allclose(vtk_to_numpy(attrs.GetArray("p_rms")), np.array([2,4])*np.sqrt(.0034375), rtol=1e-7)
np.testing.assert_allclose(vtk_to_numpy(attrs.GetArray("dpdt_rms")), [2,4], rtol=1e-7)
entries = worker["_pvvis_times"](Path(settings["case_path"])/"postProcessing"/"writePatchFields", "propeller.vtk")
# Nonuniform physical time integration: linear p has an exact midpoint mean
# and constant derivative independently of the spacing.
worker["_pvvis_statistics"]([entries[i] for i in [0,1,3,8]], settings, run_dir, "m^2/s^2")
reader.Modified(); reader.Update()
attrs = reader.GetOutput().GetCellData()
np.testing.assert_allclose(vtk_to_numpy(attrs.GetArray("p_mean")), [101325.2,101325.4], atol=1e-7, rtol=0)
np.testing.assert_allclose(vtk_to_numpy(attrs.GetArray("dpdt_rms")), [2,4], rtol=1e-7)
# A changed panel order must not produce a plausible but invalid RMS map.
path = Path(entries[-1][1])
text = path.read_text()
path.write_text(text.replace("4 0 1 2 3", "4 1 0 2 3"))
try:
    worker["_pvvis_statistics"](entries, settings, run_dir, "m^2/s^2")
except ValueError as exc:
    assert "topology" in str(exc)
else:
    raise AssertionError("Changed panel topology was accepted")
path.write_text(text)
print("Analytic and topology checks passed")
''', encoding="utf-8")
            checked = subprocess.run([tools.find_paraview_executable(), "--disable-registry", str(checks), result["run_directory"]], capture_output=True, text=True)
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)
            output = Path(tmp) / "report.pdf"
            canvas = Canvas(str(output))
            canvas.drawString(50, 700, "Existing report content")
            summary = tools.append_visualization_report(canvas, case)
            canvas.save()
            self.assertEqual(summary["views"], len(result["views"]))
            pages = PdfReader(output).pages
            self.assertEqual(len(pages), len(result["views"]) + 2)
            self.assertIn("Scientific Visual Atlas", pages[1].extract_text())
            self.assertIn("not", pages[1].extract_text())
            self.assertIn("Rotor phase", pages[5].extract_text())
            permeable = run_visualization("permeable", case, 600, config={
                "image_resolution": [1200, 720], "surface_phases": 1, "volume_phases": 1,
                "wake_stations_D": [0], "q_over_omega2": [.5], "statistics_revolutions": 2,
            })
            self.assertEqual(permeable["status"], "complete", permeable["warnings"])
            self.assertEqual(permeable["surface_statistics"]["frame"], "stationary panels")
            self.assertTrue(any("blade enclosure" in v["title"] for v in permeable["views"]))


if __name__ == "__main__":
    unittest.main()
