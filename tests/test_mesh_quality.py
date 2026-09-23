"""Quality parsing, conservative scoring, and PDF integration regressions."""
from pathlib import Path
import tempfile
import unittest

from tools.reporting import read_mesh_information, read_mesh_element_types
from tools.mesh_quality import read_mesh_quality


BASE = """Mesh stats
    points: 12000
    faces: 30000
    cells: 10000
    boundary patches: 4
    hexahedra: 9990
    wedges: 0
    tet wedges: 10
Checking geometry...
    Max aspect ratio = 12 OK.
    Mesh non-orthogonality Max: 72 average: 8
    Max skewness = 2 OK.
    Min volume = 1e-12. Max volume = 0.001. Total volume = 1
    Cell determinant (wellposedness) : minimum: 0.02 average: 1.2
"""
DEFECTS = """    *Number of severely non-orthogonal (> 70 degrees) faces: 30.
    <<Writing 30 non-orthogonal faces to set nonOrthoFaces
    <<Writing 10 under-determined cells to set underdeterminedCells
    <<Writing 20 concave cells to set concaveCells
    <<Writing 5 faces with low volume ratio cells to set lowVolRatioFaces
"""


class MeshQualityTests(unittest.TestCase):
    def test_openfoam13_actual_log_all_failures_and_warnings(self):
        text = (Path(__file__).parent / "fixtures" / "checkMesh_openfoam13.log").read_text()
        info, quality = self.parse(text)
        rows = {row["name"]: row for row in quality["diagnostics"]}
        expected_failures = {"skewFaces": 104, "lowQualityTetFaces": 1938,
                             "underdeterminedCells": 235, "concaveCells": 914,
                             "lowWeightFaces": 153}
        self.assertEqual(quality["failed_checks"], 5)
        self.assertEqual(len(quality["failed_messages"]), 5)
        self.assertEqual({row["name"]: row["count"] for row in rows.values()
                          if row["status"] == "Failed"}, expected_failures)
        self.assertEqual(rows["nonOrthoFaces"]["status"], "Reported")
        self.assertEqual(rows["nonOrthoFaces"]["count"], 290)
        self.assertEqual(rows["concaveFaces"]["count"], 88)
        self.assertEqual(rows["warpedFaces"]["count"], 71)
        self.assertEqual(rows["shortEdges"]["count"], 2)
        self.assertIsNone(rows["shortEdges"]["percent"])
        self.assertIsNone(rows["shortEdges"]["penalty"])
        self.assertIsNotNone(quality["score"])
        self.assertLess(quality["score"], 45.3)
        self.assertEqual(quality["cell_bounds"], (914, 1149))
        self.assertAlmostEqual(rows["concaveCells"]["percent"], 100 * 914 / 544966)
        # If a version also writes sets, do not charge the same category twice.
        _, duplicate = self.parse(text.replace("End", "<<Writing 235 cells to set underdeterminedCells\nEnd"))
        self.assertEqual(quality["score"], duplicate["score"])

    def test_openfoam13_pdf_shows_missing_cell_and_face_diagnostics(self):
        from createSimulationReport import create_simulation_report
        from pypdf import PdfReader
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "log.checkMesh").write_text((Path(__file__).parent / "fixtures" / "checkMesh_openfoam13.log").read_text())
            result = create_simulation_report(root, 4000, "AMI", "kOmegaSST", mesh_only=True, quiet=True)
            text = "\n".join(page.extract_text() for page in PdfReader(result["output_pdf"]).pages)
            for name in ("underdeterminedCells", "concaveCells", "lowQualityTetFaces", "lowWeightFaces"):
                self.assertIn(name, text)
            self.assertIn("914 to 1,149", text)
            self.assertNotIn("No cell-defect counts reported", text)
            self.assertIn("5 explicit failure messages", text)

    def parse(self, text):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            if text is not None:
                (root / "log.checkMesh").write_text(text)
            info = read_mesh_information(root)
            return info, read_mesh_quality(root, info)

    def test_clean_missing_and_unfinished_logs(self):
        self.assertEqual(self.parse(BASE + "Mesh OK.\n")[1]["score"], 100)
        for text in (None, BASE, "Mesh OK.", BASE + "Mesh OK.\nExec : checkMesh\nStarting"):
            with self.subTest(text=text):
                self.assertIsNone(self.parse(text)[1]["score"])

    def test_entity_denominators_overlap_and_deduplication(self):
        info, quality = self.parse(BASE + DEFECTS + "Failed 2 mesh checks.\n")
        rows = {row["name"]: row for row in quality["diagnostics"]}
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows["nonOrthoFaces"]["percent"], .1)
        self.assertEqual(rows["nonOrthoFaces"]["per_100_cells"], .3)
        self.assertEqual(rows["lowVolRatioFaces"]["entity"], "faces")
        self.assertEqual(quality["cell_bounds"], (20, 30))
        self.assertLessEqual(quality["score"], 59)
        self.assertEqual(info["mean_non_orthogonality"], 8)
        self.assertEqual(info["min_volume"], 1e-12)

    def test_critical_and_uncounted_failures(self):
        _, quality = self.parse(BASE + "<<Writing 1 cells to set zeroVolumeCells\nFailed 1 mesh checks.")
        self.assertLessEqual(quality["score"], 19)
        self.assertTrue(quality["critical"])
        self.assertEqual(self.parse(BASE + "Failed 1 mesh checks.")[1]["score"], round(95 * .59, 1))

    def test_latest_mesh_does_not_reuse_previous_success(self):
        info, quality = self.parse(BASE + "Mesh OK.\n" + BASE + "Failed 3 mesh checks.")
        self.assertFalse(info["mesh_ok"])
        self.assertEqual(quality["failed_checks"], 3)
        self.assertIsNone(self.parse(BASE + "Mesh OK.\n" + BASE)[1]["score"])

    def test_prevalence_monotonic_and_scale_invariant(self):
        def score(count, base=BASE):
            return self.parse(base + f"<<Writing {count} concave cells to set concaveCells\nMesh OK.")[1]["score"]
        self.assertGreater(score(1), score(10))
        self.assertGreater(score(10), score(100))
        self.assertEqual(score(10), score(100, BASE.replace("cells: 10000", "cells: 100000")))

    def test_mesh_only_pdf_contains_stats_quality_and_all_overflow_rows(self):
        from createSimulationReport import create_simulation_report
        from pypdf import PdfReader
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            extras = "".join(f"<<Writing 1 faces to set extraCheck{i}\n" for i in range(45))
            (root / "log.checkMesh").write_text(BASE + DEFECTS + extras + "Failed 2 mesh checks.")
            result = create_simulation_report(root, 4000, "AMI", "kOmegaSST", mesh_only=True, quiet=True)
            pages = PdfReader(result["output_pdf"]).pages
            text = "\n".join(page.extract_text() for page in pages)
            self.assertIn("Cells: 10000", pages[0].extract_text())
            self.assertIn("Mesh Quality Metrics", pages[1].extract_text())
            self.assertIn("extraCheck44", text)
            self.assertIn("heuristic v2", text)
            self.assertEqual(result["mesh_quality"]["failed_checks"], 2)
            self.assertEqual(read_mesh_element_types(root)["wedges"], 0)

    def test_normal_report_also_contains_quality_page(self):
        from createSimulationReport import create_simulation_report
        from pypdf import PdfReader
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "log.checkMesh").write_text(BASE + "Mesh OK.")
            forces = root / "postProcessing" / "forcesBlades"
            forces.mkdir(parents=True)
            (forces / "merged_forces.dat").write_text("".join(
                f"{i * .005} 0 1 0 0 1 0 0 1 0 0 1 0\n" for i in range(1, 41)))
            result = create_simulation_report(root, 4000, "AMI", "kOmegaSST", aerodynamics_only=True, quiet=True)
            text = "\n".join(page.extract_text() for page in PdfReader(result["pdf_path"]).pages)
            self.assertIn("Mesh Quality Metrics", text)
            self.assertIn("CFD Simulation Report", text)
            self.assertEqual(result["mesh_quality"]["score"], 100)

    def test_failed_and_critical_scores_do_not_plateau_at_caps(self):
        for name, factor in (("concaveCells", .59), ("zeroVolumeCells", .19)):
            scores = []
            for count in (1, 10, 100, 1000):
                _, quality = self.parse(BASE + f"<<Writing {count} cells to set {name}\nFailed 1 mesh checks.")
                self.assertAlmostEqual(quality["score"], round(quality["raw_score"] * factor, 1))
                scores.append(quality["score"])
            self.assertTrue(all(a > b for a, b in zip(scores, scores[1:])), scores)

    def test_failed_score_scales_with_mesh_size_and_failure_count(self):
        tail = "<<Writing 100 cells to set concaveCells\nFailed 1 mesh checks."
        original = self.parse(BASE + tail)[1]["score"]
        larger = self.parse(BASE.replace("cells: 10000", "cells: 100000") + tail)[1]["score"]
        more_failures = self.parse(BASE + tail.replace("Failed 1", "Failed 2"))[1]["score"]
        self.assertGreater(larger, original)
        self.assertGreater(original, more_failures)

    def test_regenerated_pdf_updates_score_text_and_scale_marker(self):
        from createSimulationReport import create_simulation_report
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        from pypdf import PdfReader
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scores, positions = [], []
            original_line = canvas.Canvas.line
            for count in (1, 1000):
                (root / "log.checkMesh").write_text(BASE + f"<<Writing {count} cells to set concaveCells\nFailed 1 mesh checks.")
                lines = []

                def capture_line(pdf, x1, y1, x2, y2):
                    lines.append((x1, y1, x2, y2))
                    return original_line(pdf, x1, y1, x2, y2)

                with patch.object(canvas.Canvas, "line", capture_line):
                    result = create_simulation_report(root, 4000, "AMI", "kOmegaSST", mesh_only=True, quiet=True)
                score = result["mesh_quality"]["score"]
                text = "\n".join(page.extract_text() for page in PdfReader(result["output_pdf"]).pages)
                self.assertIn(f"Mesh Quality Index: {score:.1f} / 100", text)
                self.assertNotIn("Mesh Quality Index: 59.0", text)
                marker = next(line for line in lines if line[0] == line[2] and abs(line[1] - line[3] - 23) < .001)
                self.assertAlmostEqual(marker[0], 50 + (A4[0] - 100) * score / 100)
                scores.append(score)
                positions.append(marker[0])
            self.assertGreater(scores[0], scores[1])
            self.assertGreater(positions[0], positions[1])


if __name__ == "__main__":
    unittest.main()
