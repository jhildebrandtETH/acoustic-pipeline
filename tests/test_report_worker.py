"""Report process isolation, failure diagnostics and numerical equivalence."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools.mesh_geometry import face_geometry
from tools import report_worker


class ReportWorkerTests(unittest.TestCase):
    def test_real_worker_generates_pdf_and_records_steps(self):
        from tests.mesh_case_fixture import write_mesh_case
        with tempfile.TemporaryDirectory() as tmp:
            write_mesh_case(tmp)
            report_worker.create_simulation_report(case_path=tmp, rpm=4000, mode='AMI',
                                                   turbulence_model='kOmegaSST', mesh_only=True, quiet=True,
                                                   latest_report_path=Path(tmp) / 'output/latest.pdf')
            report = Path(tmp) / 'report'
            self.assertTrue((report / 'simulation_report.pdf').read_bytes().startswith(b'%PDF'))
            self.assertEqual((Path(tmp) / 'output/latest.pdf').read_bytes(),
                             (report / 'simulation_report.pdf').read_bytes())
            self.assertEqual(json.loads((report / 'report-generation-status.json').read_text())['status'], 'complete')
            log = (report / 'report-generation.log').read_text()
            self.assertIn('measuring first-cell thickness', log)
            self.assertIn('measuring cell volumes', log)

    def test_latest_copy_refreshes_and_failed_copy_preserves_previous(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, latest = Path(tmp) / 'case.pdf', Path(tmp) / 'output/latest.pdf'
            source.write_bytes(b'first complete report')
            report_worker.publish_latest_report(source, latest)
            source.write_bytes(b'new mesh report')
            report_worker.publish_latest_report(source, latest)
            self.assertEqual(latest.read_bytes(), b'new mesh report')

            def fail_copy(source, staged):
                Path(staged).write_bytes(b'partial')
                raise OSError('copy failed')

            with patch.object(report_worker.shutil, 'copyfile', side_effect=fail_copy):
                with self.assertRaises(OSError):
                    report_worker.publish_latest_report(source, latest)
            self.assertEqual(latest.read_bytes(), b'new mesh report')
            self.assertEqual(list(latest.parent.iterdir()), [latest])

    def test_abrupt_child_exit_preserves_pdf_and_becomes_exception(self):
        real_popen = subprocess.Popen

        def crash(command, **kwargs):
            return real_popen([sys.executable, '-c', 'import os; os._exit(139)'], **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / 'report'
            report.mkdir()
            pdf = report / 'simulation_report.pdf'
            pdf.write_bytes(b'previous PDF')
            latest = Path(tmp) / 'latest.pdf'
            latest.write_bytes(b'previous latest PDF')
            with patch.object(report_worker.subprocess, 'Popen', side_effect=crash):
                with self.assertRaisesRegex(RuntimeError, 'code 139.*report-generation.log'):
                    report_worker.create_simulation_report(case_path=tmp, latest_report_path=latest)
            self.assertEqual(pdf.read_bytes(), b'previous PDF')
            self.assertEqual(latest.read_bytes(), b'previous latest PDF')
            self.assertEqual(json.loads((report / 'report-generation-status.json').read_text())['status'], 'failed')

    def test_python_failure_does_not_publish_partial_pdf(self):
        def fail(**kwargs):
            Path(kwargs['output_pdf']).write_bytes(b'partial PDF')
            raise ValueError('invalid report data')

        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / 'report'
            report.mkdir()
            pdf = report / 'simulation_report.pdf'
            pdf.write_bytes(b'previous PDF')
            request = Path(tmp) / 'request.json'
            request.write_text(json.dumps({'case_path': tmp}))
            with patch('createSimulationReport.create_simulation_report', side_effect=fail):
                with self.assertRaisesRegex(ValueError, 'invalid report data'):
                    report_worker.main(request)
            self.assertEqual(pdf.read_bytes(), b'previous PDF')

    def test_batched_geometry_matches_scalar_nonplanar_and_degenerate_faces(self):
        rng = np.random.default_rng(12)
        sizes = rng.integers(3, 9, size=37)
        offsets = np.r_[0, np.cumsum(sizes)]
        points = rng.normal(size=(offsets[-1], 3))
        points[:sizes[0]] = 0
        vertices = np.arange(len(points))
        ids = np.arange(len(sizes))[::-1]
        result = face_geometry(points, vertices, offsets, ids, batch_size=5)
        for i in ids:
            p = points[offsets[i]:offsets[i + 1]]
            ref = p.mean(axis=0)
            triangles = np.cross(p - ref, np.roll(p - ref, -1, axis=0)) / 2
            weights = np.linalg.norm(triangles, axis=1)
            centre = (np.average((p + np.roll(p, -1, axis=0) + ref) / 3, weights=weights, axis=0)
                      if weights.sum() else ref)
            np.testing.assert_allclose(result[i][0], centre, atol=1e-14)
            np.testing.assert_allclose(result[i][1], triangles.sum(axis=0), atol=1e-14)
