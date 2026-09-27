"""Volume geometry, tail prevalence, stale data and both PDF report paths."""
import gzip
import re
from pathlib import Path
import tempfile
import unittest

import numpy as np
from pypdf import PdfReader

from tests.mesh_case_fixture import write_mesh_case
from tools.mesh_volumes import measure_cell_volumes, volume_statistics, create_volume_report_data


class MeshVolumeTests(unittest.TestCase):
    def test_small_cells_in_translated_mesh(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_mesh_case(tmp)
            mesh = Path(tmp) / 'constant/polyMesh'
            path = mesh / 'points'

            def transform(match):
                xyz = list(map(float, match[1].split()))
                if abs(xyz[0] - (-1 + 2/12)) < 1e-12:
                    xyz[0] = -1 + 1e-7
                return '(' + ' '.join(str(v + 10) for v in xyz) + ')'

            path.write_text(re.sub(r'\(([^()]+)\)', transform, path.read_text()))
            volumes = measure_cell_volumes(mesh)
            self.assertAlmostEqual(volumes.sum(), 2.3)
            np.testing.assert_allclose(volumes[volumes < 1e-8], 1e-7 * .15 * .125, rtol=1e-7)
            stats = volume_statistics(volumes)
            self.assertEqual(stats['small_cell_shares'][0]['count'], 64)
            self.assertAlmostEqual(stats['small_cell_shares'][0]['cell_percent'], 100 * 64 / 736)

    def test_mesh_geometry_and_gzip(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_mesh_case(tmp)
            mesh = Path(tmp) / 'constant/polyMesh'
            volumes = measure_cell_volumes(mesh)
            self.assertEqual(len(volumes), 736)
            np.testing.assert_allclose(volumes, .003125, rtol=1e-12)
            self.assertAlmostEqual(volumes.sum(), 2.3)
            for name in ('points', 'faces', 'owner', 'neighbour'):
                path = mesh / name
                path.with_suffix('.gz').write_bytes(gzip.compress(path.read_bytes()))
                path.unlink()
            np.testing.assert_allclose(measure_cell_volumes(mesh), volumes)

    def test_tiny_tail_and_invalid_denominator(self):
        stats = volume_statistics([1] * 997 + [1e-8, 0, float('nan')])
        self.assertEqual(stats['nonpositive_cells'], 1)
        self.assertEqual(stats['nonfinite_cells'], 1)
        for row in stats['small_cell_shares']:
            self.assertEqual(row['count'], 1)
            self.assertEqual(row['cell_percent'], .1)
            self.assertLess(row['volume_percent'], 1e-8)
        self.assertEqual(volume_statistics([1] * 100)['small_cell_shares'][0]['count'], 0)

    def test_missing_mesh_removes_stale_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_mesh_case(tmp)
            result = create_volume_report_data(tmp)
            self.assertEqual(result['status'], 'complete')
            (Path(tmp) / 'constant/polyMesh/owner').unlink()
            missing = create_volume_report_data(tmp)
            self.assertEqual(missing['status'], 'unavailable')
            self.assertFalse(Path(result['image']).exists())
            self.assertFalse(Path(result['csv']).exists())

    def test_both_report_paths_and_solver_modes(self):
        from createSimulationReport import create_simulation_report
        for mesh_only, mode in ((True, 'AMI'), (False, 'MRF')):
            with self.subTest(mesh_only=mesh_only, mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                write_mesh_case(root)
                forces = root / 'postProcessing/forcesBlades'
                forces.mkdir(parents=True)
                (forces / 'merged_forces.dat').write_text(''.join(
                    f'{i * .005} 0 1 0 0 1 0 0 1 0 0 1 0\n' for i in range(1, 41)))
                result = create_simulation_report(root, 4000, mode, 'kOmegaSST',
                                                  mesh_only=mesh_only, aerodynamics_only=True, quiet=True)
                text = '\n'.join(page.extract_text() for page in PdfReader(
                    result.get('output_pdf', result.get('pdf_path'))).pages)
                self.assertIn('Mesh Volume Distribution', text)
                self.assertIn('Mesh and Solver Settings', text)
                self.assertIn('Report arguments', text)
                self.assertIn('736 cells', text)
                self.assertIn('0.01% of median', text)
                self.assertEqual(result['mesh_volumes']['status'], 'complete')


if __name__ == '__main__':
    unittest.main()
