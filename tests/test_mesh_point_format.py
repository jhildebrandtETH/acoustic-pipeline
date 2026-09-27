import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.mesh_point_format import normalize_point_signs
from tools import cfmesh_pipeline

HEADER = b'FoamFile\n{\nformat ascii;\nclass vectorField;\nobject points;\n}\n'


class PointFormatTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.case = Path(self.temp.name)
        self.path = self.case / 'constant/polyMesh/points'
        self.path.parent.mkdir(parents=True)

    def test_fix_preserves_exponents_negatives_and_every_other_byte(self):
        original = HEADER + b'2\n(\n(+0.033 -1.20e+03 +.7e-02)\n(1e+05 -0.0 2)\n)\n'
        expected = original.replace(b'+0.033', b'0.033').replace(b'+.7', b'.7')
        self.path.write_bytes(original)
        self.assertEqual(normalize_point_signs(self.path), 1)
        self.assertEqual(self.path.read_bytes(), expected)
        record = json.loads(self.path.with_name('point-sign-repairs.jsonl').read_text())
        self.assertEqual(Path(record['backup']).read_bytes(), original)
        self.assertEqual(normalize_point_signs(self.path), 0)
        self.assertEqual(len(list(self.path.parent.glob('points.before-sign-fix-*'))), 1)

    def test_binary_and_clean_files_are_untouched(self):
        for data in (HEADER + b'1\n(\n(1e+03 -2 3)\n)\n',
                     HEADER.replace(b'ascii', b'binary') + b'\x00(+3 garbage\xff'):
            self.path.write_bytes(data)
            self.assertEqual(normalize_point_signs(self.path), 0)
            self.assertEqual(self.path.read_bytes(), data)

    def test_malformed_input_is_not_silently_repaired(self):
        data = HEADER + b'1\n(\n(+0.5 nan 3)\n)\n'
        self.path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, 'non-vector'):
            normalize_point_signs(self.path)
        self.assertEqual(self.path.read_bytes(), data)

    def test_wrapper_repairs_before_reader_and_after_writer(self):
        bad = HEADER + b'1\n(\n(1 2 +3)\n)\n'
        good = bad.replace(b'+3', b'3')
        self.path.write_bytes(bad)
        def execute(*args, **kwargs):
            self.assertEqual(self.path.read_bytes(), good)
            self.path.write_bytes(bad)  # Simulate the observed writer output.
            return True
        with patch('tools.safe_exec', side_effect=execute), patch('tools.report_case_stage'):
            cfmesh_pipeline.docker_run(None, self.case, ['mergeMeshes'], 'log.merge', None, False)
        self.assertEqual(self.path.read_bytes(), good)

    def test_wrapper_preserves_command_failure(self):
        self.path.write_bytes(HEADER + b'0\n(\n)\n')
        with patch('tools.safe_exec', return_value=False), patch('tools.report_case_stage'):
            with self.assertRaisesRegex(RuntimeError, 'checkMesh failed'):
                cfmesh_pipeline.docker_run(None, self.case, ['checkMesh'], 'log.check', None, False)


if __name__ == '__main__':
    unittest.main()
