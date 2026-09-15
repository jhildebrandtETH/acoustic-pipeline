"""Small native layer smoke test, independent of a production propeller mesh."""

import os
from pathlib import Path
import shutil
import tempfile
import unittest

from tools.cfmesh_pipeline import ROOT, header, native_run, write_ftr


@unittest.skipUnless(os.environ.get("CFMESH_NATIVE_LAYER_TESTS") == "1", "requires trimesh and OpenFOAM Docker runtime")
class NativeLayerTests(unittest.TestCase):
    def test_cartesian_mesh_consumes_layer_optimisation_controls(self):
        import trimesh

        with tempfile.TemporaryDirectory(prefix="cfmesh-native-layer-") as tmp:
            case = Path(tmp)
            shutil.copytree(ROOT / "Parameters", case / "Parameters")
            common = case / "Parameters/cfmeshCommon.cpp"
            text = common.read_text().replace("baseCellSize 0.02;", "baseCellSize 0.1;")
            text = text.replace("propellerLevel 5;", "propellerLevel 0;")
            text = text.replace("layerOptimise 0;", "layerOptimise 1;")
            common.write_text(text)
            (case / "system").mkdir()
            (case / "constant/triSurface").mkdir(parents=True)
            cube = trimesh.creation.box(extents=[0.4, 0.4, 0.4])
            write_ftr(case / "constant/triSurface/domain.ftr", [
                ("propeller", "wall", cube.vertices, cube.faces, False),
            ])
            (case / "system/meshDict").write_text(header("meshDict") + '#include "../Parameters/cfmeshRotorDict"\n')
            (case / "system/controlDict").write_text(header("controlDict") + """
application cartesianMesh;
startFrom startTime; startTime 0; stopAt endTime; endTime 1; deltaT 1;
writeControl timeStep; writeInterval 1; writeFormat ascii;
""")
            (case / "system/fvSchemes").write_text(header("fvSchemes") + """
ddtSchemes { default steadyState; } gradSchemes { default Gauss linear; }
divSchemes { default none; } laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; } snGradSchemes { default corrected; }
""")
            (case / "system/fvSolution").write_text(header("fvSolution") + "solvers {}\n")
            result = native_run(["cartesianMesh"], cwd=case, cores=1, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("Starting optimising boundary layer", result.stdout)
            self.assertTrue((case / "constant/polyMesh/points").is_file())
            result = native_run(["checkMesh"], cwd=case, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("Mesh OK.", result.stdout)


if __name__ == "__main__":
    unittest.main()
