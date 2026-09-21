import argparse
import tempfile
import unittest
from pathlib import Path

from diagnoseCourant import analyse, choose_time, load_case, log_history, main, read_ascii_internal

try:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk
    import numpy as np
except ImportError:
    vtk = None


class HistoryTests(unittest.TestCase):
    def test_plateau_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "log"
            log.write_text("deltaT = 1e-5\nCourant Number mean: 0.1 max: 5\n" * 20)
            result = log_history(log)
            self.assertTrue(result["delta_t_plateau"])
            self.assertEqual(result["max_co_tail"], [5.] * 20)

    def test_exact_saved_time_required(self):
        self.assertEqual(choose_time([0, .1, .2], "latest"), .2)
        with self.assertRaises(ValueError):
            choose_time([0, .1, .2], ".15")


@unittest.skipIf(vtk is None, "Install vtk and numpy for geometry tests")
class GeometryTests(unittest.TestCase):
    def test_internal_fields_ignore_ncc_boundary_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "U"
            path.write_text("FoamFile { format ascii; }\ninternalField nonuniform List<vector> 2 ((1 2 3)(4 5 6));\nboundaryField { virtualPatch { value nonuniform List<vector> 99 (); } }")
            np.testing.assert_array_equal(read_ascii_internal(path, 2, 3), [[1, 2, 3], [4, 5, 6]])
            with self.assertRaisesRegex(ValueError, "length"):
                read_ascii_internal(path, 3, 3)

    def test_ranking_neighbours_and_original_ids(self):
        points = vtk.vtkPoints()
        for x in range(3):
            for y, z in [(0, 0), (1, 0), (1, 1), (0, 1)]:
                points.InsertNextPoint(x, y, z)
        mesh = vtk.vtkUnstructuredGrid()
        mesh.SetPoints(points)
        for base in [0, 4]:
            ids = vtk.vtkIdList()
            for index in [0, 4, 5, 1, 3, 7, 6, 2]:
                ids.InsertNextId(base + index)
            mesh.InsertNextCell(vtk.VTK_HEXAHEDRON, ids)
        for name, values in [("Co", np.array([.2, 5.])), ("U", np.array([[1., 0, 0], [2., 0, 0]]))]:
            array = numpy_to_vtk(values, deep=True)
            array.SetName(name)
            mesh.GetCellData().AddArray(array)
        args = argparse.Namespace(co_field=None, patch="propeller", axis="y", origin=[0, 0, 0], tip_radius=None, top=1, threshold=4, target_co=1)
        summary, rows, subset = analyse(mesh, [], args)
        self.assertEqual(rows[0]["cell_id"], 1)
        self.assertAlmostEqual(rows[0]["volume_m3"], 1)
        self.assertAlmostEqual(rows[0]["edge_ratio"], 1)
        self.assertEqual(rows[0]["U_to_neighbour_ratio"], 2)
        self.assertEqual(subset.GetNumberOfCells(), 2)
        self.assertEqual(summary["cells_above_threshold"], 1)
        self.assertAlmostEqual(summary["same_flux_timestep_multiplier"], .2)
        mesh.GetCellData().RemoveArray("Co")
        with self.assertRaisesRegex(ValueError, "No saved cell Courant"):
            analyse(mesh, [], args)

    def test_reads_foam_and_writes_report(self):
        def foam(cls, name, body):
            return f"FoamFile {{ version 2.0; format ascii; class {cls}; object {name}; }}\n{body}\n"
        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory)
            poly = case / "constant/polyMesh"
            poly.mkdir(parents=True)
            (case / "system").mkdir()
            (case / "system/controlDict").write_text(foam("dictionary", "controlDict", "startTime 0; endTime 1; deltaT 0.1; writeControl timeStep; writeInterval 1;"))
            (poly / "points").write_text(foam("vectorField", "points", "8((0 0 0)(1 0 0)(1 1 0)(0 1 0)(0 0 1)(1 0 1)(1 1 1)(0 1 1))"))
            (poly / "faces").write_text(foam("faceList", "faces", "6(4(0 3 2 1) 4(4 5 6 7) 4(0 1 5 4) 4(1 2 6 5) 4(2 3 7 6) 4(3 0 4 7))"))
            (poly / "owner").write_text(foam("labelList", "owner", "6(0 0 0 0 0 0)"))
            (poly / "neighbour").write_text(foam("labelList", "neighbour", "0()"))
            (poly / "boundary").write_text(foam("polyBoundaryMesh", "boundary", "1(propeller { type wall; nFaces 6; startFace 0; })"))
            time = case / "0.1"
            time.mkdir()
            for field, cls, dims, value in [("Co", "volScalarField", "0 0 0 0 0 0 0", "5"), ("U", "volVectorField", "0 1 -1 0 0 0 0", "(1 0 0)")]:
                (time / field).write_text(foam(cls, field, f"dimensions [{dims}]; internalField uniform {value}; boundaryField {{ propeller {{ type zeroGradient; }} }}"))
            mesh, blocks, selected, reader = load_case(case, "latest")
            self.assertEqual(selected, .1)
            self.assertEqual(mesh.GetNumberOfCells(), 1)
            out = case / "diagnostics"
            main([str(case), "--output", str(out)])
            self.assertTrue((out / "report.md").is_file())
            self.assertTrue((out / "hotspots.vtu").is_file())
            self.assertFalse((case / "diagnostics.foam").exists())


if __name__ == "__main__":
    unittest.main()
