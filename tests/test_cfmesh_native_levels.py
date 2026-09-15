"""Verify native level zero against an otherwise identical unrefined mesh."""

import os
from pathlib import Path
import re
import tempfile
import unittest

from tools.cfmesh_pipeline import header, native_run, strip_header, write_ftr


@unittest.skipUnless(os.environ.get("CFMESH_NATIVE_LEVEL_TESTS") == "1", "requires trimesh and native cfMesh")
class NativeLevelTests(unittest.TestCase):
    def test_zero_is_background_and_one_adds_exactly_one_level(self):
        import trimesh

        counts = {}
        with tempfile.TemporaryDirectory(prefix="cfmesh-level-check-") as tmp:
            for name, control in (
                ("baseline", None),
                ("zero", "additionalRefinementLevels 0;"),
                ("one", "additionalRefinementLevels 1;"),
                ("old_size_conversion", "cellSize 0.05;"),
            ):
                case = Path(tmp) / name
                (case / "system").mkdir(parents=True)
                (case / "constant/triSurface").mkdir(parents=True)
                cube = trimesh.creation.box(extents=[0.4, 0.4, 0.4])
                write_ftr(case / "constant/triSurface/domain.ftr", [
                    ("walls", "wall", cube.vertices, cube.faces, False),
                ])
                objects = ""
                if control:
                    objects = "objectRefinements { testSphere { type sphere; centre (0 0 0); radius 0.12; " + control + " } }\n"
                (case / "system/meshDict").write_text(header("meshDict") + """
surfaceFile "constant/triSurface/domain.ftr";
maxCellSize 0.05;
workflowControls { stopAfter edgeExtraction; }
""" + objects)
                (case / "system/controlDict").write_text(header("controlDict") + """
application cartesianMesh;
startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1;
writeControl timeStep; writeInterval 1; writeFormat ascii;
""")
                result = native_run(["cartesianMesh"], cwd=case, cores=1, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                base = int(re.search(r"Requested cell size corresponds to octree level (\d+)", result.stdout)[1])
                if control:
                    actual = int(re.search(r"Ref level for object testSphere is (\d+)", result.stdout)[1])
                    expected_delta = 0 if name == "zero" else 1
                    self.assertEqual(actual, base + expected_delta, result.stdout)
                labels = []
                for filename in ("owner", "neighbour"):
                    contents = strip_header(case / "constant/polyMesh" / filename)
                    labels.extend(map(int, contents[contents.index("(") + 1:contents.rindex(")")].split()))
                counts[name] = max(labels) + 1
        self.assertEqual(counts["zero"], counts["baseline"], counts)
        self.assertGreater(counts["one"], counts["baseline"], counts)
        self.assertEqual(counts["old_size_conversion"], counts["one"], counts)
        print("Native level comparison cell counts:", counts)


if __name__ == "__main__":
    unittest.main()
