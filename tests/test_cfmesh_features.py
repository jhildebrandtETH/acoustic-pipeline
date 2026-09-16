"""Explicit feature input validation and per-case staging."""

from pathlib import Path
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from tools.cfmesh_features import prepare_features, scaled_obj
from tools.cfmesh_pipeline import ROOT, header, native_run, query, strip_header, write_ftr


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "STL/12x6.stl"
        self.case = self.root / "case"
        (self.case / "Parameters").mkdir(parents=True)
        (self.case / "Parameters/cfmeshFeatureDict").touch()
        (self.root / "FEATURES").mkdir()
        self.obj = self.root / "FEATURES/12x6_tip.obj"
        self.obj.write_text("v 0 0 0\nv 10 0 0\nv 10 20 0\nl -3 -2 -1\n")
        self.values = {"enabled": "true", "groups": "(le te tip hub)",
                       "le/level": "6", "te/level": "6",
                       "le/refinementThickness": "0.0005",
                       "te/refinementThickness": "0.0005",
                       "tip/level": "6", "hub/level": "7",
                       "tip/refinementThickness": "0.001",
                       "hub/refinementThickness": "0"}
        for group in ("le", "te"):
            (self.root / f"FEATURES/{self.source.stem}_{group}.obj").write_text(self.obj.read_text())
        self.query = patch("tools.cfmesh_features.query", side_effect=lambda p, e: self.values[e]).start()
        self.addCleanup(patch.stopall)

    def prepare(self):
        return prepare_features(self.case, self.source, 0.001, {"base_cell_size_m": 0.02})

    def test_groups_scaling_and_snapshots(self):
        hub = self.root / "FEATURES/12x6_hub.obj"
        hub.write_text("v 0 0 0\nv 0 10 0\nl 1/1 2/2\n")
        report = self.prepare()
        self.assertEqual(set(report), {"le", "te", "tip", "hub"})
        self.assertTrue(all(group["status"] == "active" for group in report.values()))
        self.assertEqual(report["tip"]["edges"], 2)
        self.assertEqual(report["hub"]["level"], 7)
        self.assertEqual(report["tip"]["cell_size_m"], 0.02 / 64)
        staged = self.case / "cfmesh/rotor" / report["tip"]["edge_file"]
        self.assertIn("v 0.01 0.02 0", staged.read_text())
        self.assertIn("l 1 2\nl 2 3", staged.read_text())
        self.assertEqual((self.case / "FEATURES" / self.obj.name).read_text(), self.obj.read_text())
        native = (self.case / "Parameters/cfmeshFeatures.generated").read_text()
        self.assertIn("additionalRefinementLevels 7;", native)
        self.assertNotIn("cellSize", native)
        self.assertFalse((self.case / "cfmesh/stator").exists())

    def test_missing_and_unrelated_files(self):
        (self.root / "FEATURES/another_hub.obj").write_text(self.obj.read_text())
        with self.assertWarnsRegex(UserWarning, "12x6_hub.obj"):
            report = self.prepare()
        self.assertEqual(report["hub"]["status"], "missing")
        self.assertEqual(report["tip"]["status"], "active")

    def test_disabled_and_legacy(self):
        self.values["enabled"] = "false"
        self.assertEqual(self.prepare(), {})
        self.query.assert_called_once()
        (self.case / "Parameters/cfmeshFeatureDict").unlink()
        self.assertEqual(self.prepare(), {})

    def test_invalid_controls(self):
        for key, value in (("tip/level", "-1"), ("tip/level", "1.5"),
                           ("tip/refinementThickness", "nan"),
                           ("tip/refinementThickness", "-0.01"),
                           ("groups", "(../tip)"), ("groups", "(tip tip)")):
            with self.subTest(key=key, value=value):
                previous = self.values[key]
                self.values[key] = value
                with self.assertRaises(ValueError):
                    self.prepare()
                self.values[key] = previous

    def test_invalid_curves(self):
        for body in ("f 1 2 3", "l 0 1", "l 1 3", "l -3 -1", "l 1 1", "l 1", "v nan 0 0\nl 1 3"):
            with self.subTest(body=body):
                self.obj.write_text("v 0 0 0\nv 1 0 0\n" + body + "\n")
                with self.assertRaises(ValueError):
                    scaled_obj(self.obj, 0.001)

    @unittest.skipUnless(os.environ.get("CFMESH_NATIVE_FEATURE_TESTS") == "1", "requires native cfMesh")
    def test_native_feature_level_increases_cell_count(self):
        self.values["groups"] = "(tip)"
        self.obj.write_text("v -100 0 0\nv 100 0 0\nl 1 2\n")
        counts = []
        for level in (0, 2):
            self.case = self.root / f"level{level}"
            shutil.copytree(ROOT / "Parameters", self.case / "Parameters")
            self.assertEqual(query(self.case / "Parameters/cfmeshFeatureDict", "tip/level"), "6")
            self.values["tip/level"] = str(level)
            self.values["tip/refinementThickness"] = "0.025"
            prepare_features(self.case, self.source, 0.001, {"base_cell_size_m": 0.05})
            rotor = self.case / "cfmesh/rotor"
            (rotor / "system").mkdir()
            vertices = [(x, y, z) for x in (-0.2, 0.2) for y in (-0.2, 0.2) for z in (-0.2, 0.2)]
            quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
                     (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
            faces = [triangle for a, b, c, d in quads for triangle in ((a, b, c), (a, c, d))]
            write_ftr(rotor / "constant/triSurface/domain.ftr", [("walls", "wall", vertices, faces, False)])
            shutil.copytree(self.case / "Parameters", rotor / "Parameters")
            (rotor / "system/fullMeshDict").write_text(header("meshDict") + '#include "../Parameters/cfmeshRotorDict"\n')
            expanded = query(rotor / "system/fullMeshDict")
            self.assertIn("edgeMeshRefinement", expanded)
            self.assertIn("features/tip.obj", expanded)
            (rotor / "system/meshDict").write_text(header("meshDict") + '''
surfaceFile "constant/triSurface/domain.ftr";
maxCellSize 0.05;
workflowControls { stopAfter edgeExtraction; }
edgeMeshRefinement
{
    #include "../Parameters/cfmeshFeatures.generated"
}
''')
            (rotor / "system/controlDict").write_text(header("controlDict") + '''
application cartesianMesh;
startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1;
writeControl timeStep; writeInterval 1; writeFormat ascii;
''')
            result = native_run(["cartesianMesh"], cwd=rotor, cores=1, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            labels = []
            for filename in ("owner", "neighbour"):
                body = strip_header(rotor / "constant/polyMesh" / filename)
                labels.extend(map(int, body[body.index("(") + 1:body.rindex(")")].split()))
            counts.append(max(labels) + 1)
        self.assertGreater(counts[1], counts[0], counts)
        print("Native feature level 0 / level 2 cell counts:", counts)


if __name__ == "__main__":
    unittest.main()
