"""Cell prevalence, saved-field selection and normal-report integration."""
import gzip
from pathlib import Path
import tempfile
import unittest
import numpy as np

from pypdf import PdfReader
from reportlab.pdfgen import canvas

from tests.mesh_case_fixture import write_mesh_case
from tools.courant_distribution import (
    courant_statistics, create_courant_report_data, append_courant_report,
    create_highest_courant_volume_plot,
)


class CourantDistributionTests(unittest.TestCase):
    def test_highest_cells_keep_volume_alignment_and_break_ties(self):
        with tempfile.TemporaryDirectory() as tmp:
            image, csv = Path(tmp) / 'plot.png', Path(tmp) / 'cells.csv'
            co = np.r_[np.zeros(500), np.ones(1001)]
            volumes = np.geomspace(1e-12, 1e-3, len(co))
            result = create_highest_courant_volume_plot(co, volumes, image, csv)
            rows = np.loadtxt(csv, delimiter=',', skiprows=1)
            self.assertEqual(result['cells'], 1000)
            np.testing.assert_array_equal(rows[:, 0], np.arange(500, 1500))
            np.testing.assert_allclose(rows[:, 2], volumes[500:1500])
            self.assertGreater(result['median_volume_m3'], result['mesh_median_volume_m3'])
            for bad in (volumes[:-1], np.zeros(len(co)), np.full(len(co), np.nan)):
                with self.assertRaises(ValueError):
                    create_highest_courant_volume_plot(co, bad, image, csv)

    def test_snapshot_points_and_missing_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_mesh_case(root)
            time = root / '0.1'
            time.mkdir()
            (time / 'Co').write_text('FoamFile { format ascii; } internalField uniform 1;')
            initial = create_courant_report_data(root)['highest_courant_volumes']
            # A points-only mesh update must inherit topology but use new geometry.
            from tools.layer_thickness import _read_list
            count, text = _read_list(root / 'constant/polyMesh/points')
            points = 2 * np.fromstring(text.replace('(', ' ').replace(')', ' '), sep=' ').reshape(-1, 3)
            (time / 'polyMesh').mkdir()
            (time / 'polyMesh/points').write_text('FoamFile { format ascii; }\n' + str(count) + '\n(\n' +
                '\n'.join('(' + ' '.join(map(str, row)) + ')' for row in points) + '\n)\n')
            updated = create_courant_report_data(root)['highest_courant_volumes']
            self.assertAlmostEqual(updated['median_volume_m3'] / initial['median_volume_m3'], 8)
            (root / 'constant/polyMesh/faces').unlink()
            unavailable = create_courant_report_data(root)
            self.assertEqual(unavailable['status'], 'complete')
            self.assertEqual(unavailable['highest_courant_volumes']['status'], 'unavailable')
            self.assertFalse(Path(updated['image']).exists())

    def test_small_tail_and_strict_thresholds(self):
        stats = courant_statistics([0] + [.1] * 997 + [2.5, 5], 5)
        self.assertEqual([r['count'] for r in stats['timestep_scenarios']], [0, 1, 2, 2])
        self.assertEqual(stats['timestep_scenarios'][1]['cell_percent'], .1)
        self.assertEqual(stats['same_flux_multiplier'], 1)
        self.assertEqual(courant_statistics([0, 0])['same_flux_multiplier'], None)
        self.assertEqual(courant_statistics([1, 2])['target_co'], 2)
        for values in ([], [-1], [float('nan')], [float('inf')]):
            with self.assertRaises(ValueError):
                courant_statistics(values)

    def test_saved_fields_pdf_and_stale_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_mesh_case(root)
            time = root / '0.1'
            time.mkdir()
            field = time / 'Co.gz'
            for value in (0, 5):
                field.write_bytes(gzip.compress(f'FoamFile {{ format ascii; }}\ninternalField uniform {value};'.encode()))
                result = create_courant_report_data(root, 5)
                self.assertEqual(result['status'], 'complete', result)
                self.assertEqual(result['cells'], 736)
                self.assertEqual(result['quantiles']['max'], value)
                self.assertEqual(result['highest_courant_volumes']['cells'], 736)
            pdf = root / 'report/test.pdf'
            c = canvas.Canvas(str(pdf))
            append_courant_report(c, result)
            c.save()
            text = PdfReader(pdf).pages[-1].extract_text()
            self.assertIn('Cell Courant Number Distribution', text)
            self.assertIn('736 cells', text)
            self.assertIn('10x', text)
            (root / '0.2').mkdir()
            missing = create_courant_report_data(root, 5)
            self.assertEqual(missing['status'], 'unavailable')
            self.assertIn('latest saved time 0.2', missing['reason'])
            self.assertFalse(Path(result['image']).exists())
            self.assertFalse(Path(result['csv']).exists())
            self.assertFalse(Path(result['highest_courant_volumes']['image']).exists())

    def test_nonuniform_and_newer_processors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_mesh_case(root)
            time = root / '0.1'
            time.mkdir()
            values = ' '.join(['0.1'] * 735 + ['5'])
            (time / 'Co').write_text(f'FoamFile {{ format ascii; }}\ninternalField nonuniform List<scalar> 736 ({values});')
            result = create_courant_report_data(root, 5)
            self.assertEqual(result['status'], 'complete', result)
            self.assertEqual(result['timestep_scenarios'][1]['count'], 1)
            (root / 'processor0/0.2').mkdir(parents=True)
            result = create_courant_report_data(root)
            self.assertEqual(result['status'], 'unavailable')
            self.assertIn('Processor output is newer', result['reason'])


if __name__ == '__main__':
    unittest.main()
