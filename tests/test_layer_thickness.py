"""Independent geometric and report checks for first-cell measurements."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from tools.layer_thickness import measure_first_cells, thickness_statistics, create_thickness_report_data


def write_boxes(root):
    """Two disjoint prisms: wall areas 1, 3 m2 and depths 1, 2 mm."""
    mesh = Path(root)/'constant/polyMesh'
    mesh.mkdir(parents=True)
    points, outer, walls = [], [], []
    for cell, (origin, width, depth) in enumerate(((0, 1, .001), (2, 3, .002))):
        offset = len(points)
        points.extend([(origin+width*x, depth*y, z) for x,y,z in
                       ((0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1))])
        for face in ((0,3,2,1),(4,5,6,7),(3,7,6,2),(0,4,7,3),(1,2,6,5)):
            outer.append(([i+offset for i in face], cell))
        walls.append(([i+offset for i in (0,1,5,4)], cell))
    faces = outer+walls

    def write(name, values):
        (mesh/name).write_text(f'FoamFile {{ format ascii; object {name}; }}\n{len(values)}\n(\n'+'\n'.join(values)+'\n)\n')

    write('points', ['('+' '.join(map(str, p))+')' for p in points])
    write('faces', ['4('+' '.join(map(str, f))+')' for f, _ in faces])
    write('owner', [str(c) for _, c in faces])
    write('neighbour', [])
    write('boundary', ['outer { type patch; nFaces 10; startFace 0; }',
                       'propeller { type wall; nFaces 2; startFace 10; }'])
    return mesh


class LayerThicknessTests(unittest.TestCase):
    def test_projected_geometry_and_area_weighted_mean(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = measure_first_cells(write_boxes(tmp))
            np.testing.assert_allclose(rows[:, 3:], [[1, 1], [3, 2]])
            stats, valid = thickness_statistics(rows, [1, 2])
            self.assertTrue(valid.all())
            self.assertEqual((stats['min_mm'], stats['mean_mm'], stats['max_mm']), (1, 1.75, 2))
            self.assertEqual(stats['area_fraction_in_target'], 1)

    def test_gzip_and_rotated_prisms_keep_normal_not_axis_height(self):
        with tempfile.TemporaryDirectory() as tmp:
            mesh = write_boxes(tmp)
            points = (mesh/'points').read_text()
            import re
            rotation = np.array([[1,0,0],[0,.6,-.8],[0,.8,.6]])
            def rotate(match):
                p = np.fromstring(match[1], sep=' ')
                return '('+' '.join(map(str, rotation@p))+')'
            points = re.sub(r'\(([-\d.]+ [-\d.]+ [-\d.]+)\)', rotate, points)
            (mesh/'points.gz').write_bytes(gzip.compress(points.encode()))
            (mesh/'points').unlink()
            np.testing.assert_allclose(measure_first_cells(mesh)[:, 4], [1,2])

    def test_excluded_faces_are_not_silently_counted_as_zero(self):
        rows = np.array([[0,0,0,1,1], [0,0,0,3,np.nan], [0,0,0,1,-2]])
        result, _ = thickness_statistics(rows)
        self.assertEqual(result['excluded_faces'], 2)
        self.assertEqual(result['measured_area_fraction'], .2)
        self.assertEqual(result['mean_mm'], 1)

    def test_report_recomputes_and_removes_stale_histogram(self):
        with tempfile.TemporaryDirectory() as tmp:
            mesh = write_boxes(tmp)
            result = create_thickness_report_data(tmp)
            self.assertEqual(result['status'], 'complete')
            image = Path(result['image'])
            self.assertTrue(image.is_file())
            self.assertIsNone(result['target_thickness_mm'])
            (mesh/'points').write_bytes(b'FoamFile { format binary; }\n')
            failed = create_thickness_report_data(tmp)
            self.assertEqual(failed['status'], 'unavailable')
            self.assertIn('ASCII', failed['reason'])
            self.assertFalse(image.exists())

    def test_mesh_and_solve_reports_both_include_numbers_and_histogram(self):
        from createSimulationReport import create_simulation_report
        from pypdf import PdfReader
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_boxes(root)
            (root/'Parameters').mkdir()
            (root/'Parameters/meshReport.json').write_text(json.dumps({'target_thickness_mm': [1,2]}))
            forces = root/'postProcessing/forcesBlades'
            forces.mkdir(parents=True)
            (forces/'merged_forces.dat').write_text(''.join(
                f'{i*.005} 0 1 0 0 1 0 0 1 0 0 1 0\n' for i in range(1,41)))
            for mesh_only in (True, False):
                with self.subTest(mesh_only=mesh_only):
                    path = root/f'report-{mesh_only}.pdf'
                    result = create_simulation_report(root, 4000, 'AMI', 'kOmegaSST', output_pdf=path,
                                                      mesh_only=mesh_only, aerodynamics_only=True, quiet=True)
                    self.assertEqual(result['layer_thickness']['mean_mm'], 1.75)
                    page = next(p for p in PdfReader(path).pages if 'First-Cell Thickness' in p.extract_text())
                    self.assertIn('1.75 mm', page.extract_text())
                    self.assertIn('Maximum', page.extract_text())
                    self.assertEqual(len(page.images), 1)


if __name__ == '__main__':
    unittest.main()
