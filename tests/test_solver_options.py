"""CLI, template matrix and acoustic bypass regressions without a CFD runtime."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools.cli import create_parser
from tools.orders import prepare_simulation_order
from tools.templates import template_directory
from tools.cfmesh_pipeline import safe_path


class SolverOptionsTests(unittest.TestCase):
    def test_template_matrix_and_boundary_conditions(self):
        count = 0
        for mode in ('AMI', 'MRF'):
            for model in ('kOmegaSST', 'kEpsilon', 'DES'):
                for walls in ('no', 'yes'):
                    if model == 'DES' and (mode == 'MRF' or walls == 'yes'):
                        with self.assertRaises(ValueError):
                            template_directory(mode, model, walls)
                        continue
                    template = template_directory(mode, model, walls)
                    self.assertTrue((template / 'system/controlDict').is_file())
                    self.assertEqual((template / 'constant/dynamicMeshDict').exists(), mode == 'AMI')
                    self.assertEqual((template / 'constant/MRFProperties').exists(), mode == 'MRF')
                    fields = '\n'.join(p.read_text() for p in (template / '0').iterdir() if p.is_file())
                    if walls == 'no':
                        self.assertNotIn('WallFunction', fields)
                    else:
                        self.assertIn('nutkWallFunction', fields)
                        self.assertIn('kqRWallFunction', fields)
                    if mode == 'MRF':
                        self.assertIn('MRFnoSlip', (template / '0/U').read_text())
                    if model == 'kEpsilon' and walls == 'no':
                        self.assertIn('LaunderSharmaKE;', (template / 'constant/turbulenceProperties').read_text())
                    if model != 'kEpsilon':
                        self.assertIn('object      omega;', (template / '0/omega').read_text())
                    with self.assertRaises(ValueError):
                        safe_path(template / 'case')
                    count += 1
        self.assertEqual(count, 9)

    def order(self, directory, *options):
        parser = create_parser()
        args = parser.parse_args(['--sim-dir', directory, *options])
        with patch('tools.orders.preflight'), patch('tools.orders.seed_current_stl'), \
             patch('tools.orders.find_source_stls', return_value={'test': Path(directory) / 'STL/test.stl'}):
            result = prepare_simulation_order(args, parser)
        return args, result[-1]

    def test_aerodynamic_order_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, order = self.order(tmp, '--rpms', '4000', '--total-cores', '2', '--cores-per-case', '2',
                '--mode', 'MRF', '--turbulence', 'kEpsilon', '--wall-functions', 'no',
                '--aerodynamics-only')
            self.assertTrue(order['aerodynamics_only'])
            self.assertIsNone(order['acoustic_surface'])
            resumed, restored = self.order(tmp, '--resume', '--wall-functions', 'yes')
            self.assertEqual(resumed.wall_functions, 'no')
            self.assertTrue(resumed.aerodynamics_only)
            self.assertEqual(resumed.mode, 'MRF')
            self.assertEqual(restored['target_cores_per_case'], 2)
            self.assertEqual(resumed.cores_per_case, 2)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.order(tmp, '--resume', '--cores-per-case', '1')

    def test_new_order_requires_valid_case_cores(self):
        base = ['--rpms', '4000', '--total-cores', '2', '--mode', 'MRF',
                '--turbulence', 'kEpsilon', '--wall-functions', 'no', '--mesh-only']
        for extra in [[], ['--cores-per-case', '0'], ['--cores-per-case', '3']]:
            with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.order(tmp, *base, *extra)

    def test_invalid_new_order_options(self):
        base = ['--rpms', '4000', '--total-cores', '2', '--cores-per-case', '2', '--mode', 'AMI', '--turbulence', 'DES']
        for extra in [[], ['--wall-functions', 'yes'],
                      ['--wall-functions', 'no', '--acoustic-surface', 'impermeable']]:
            with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.order(tmp, *base, '--aerodynamics-only', *extra)

    def test_legacy_resume_preserves_original_templates(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, order = self.order(tmp, '--rpms', '4000', '--total-cores', '2', '--cores-per-case', '2',
                '--mode', 'AMI', '--turbulence', 'kOmegaSST', '--wall-functions', 'no', '--mesh-only')
            del order['wall_functions']
            del order['aerodynamics_only']
            (Path(tmp) / 'simulation_order.json').write_text(json.dumps(order))
            args, _ = self.order(tmp, '--resume')
            self.assertFalse(args.aerodynamics_only)
            self.assertIn('legacy', template_directory(args.mode, args.turbulence, args.wall_functions).parts)

    def test_aerodynamic_postprocessing_never_loads_acoustic_solver(self):
        import postprocessing as stage
        with tempfile.TemporaryDirectory() as tmp, \
             patch.dict(sys.modules, {'acoustic_propagation': None}), \
             patch.object(stage, 'merge_postprocessing_dat_files') as merge, \
             patch.object(stage, 'run_visualization') as visualize, \
             patch.object(stage, 'create_simulation_report') as report:
            stage.postprocessing(None, Path(tmp), 4000, 'MRF', 'kOmegaSST', AERODYNAMICS_ONLY=True)
            self.assertEqual(merge.call_count, 3)
            visualize.assert_called_once()
            self.assertTrue(report.call_args.kwargs['aerodynamics_only'])
            status = json.loads((Path(tmp) / 'report/acoustic-status.json').read_text())
            self.assertEqual(status['status'], 'skipped')

    def test_aerodynamic_preprocessing_disables_sampling(self):
        from preprocessing import preprocessing
        with tempfile.TemporaryDirectory() as tmp, \
             patch('preprocessing.find_source_stls', return_value={'test': Path(tmp) / 'test.stl'}), \
             patch('tools.find_source_stls', return_value={'test': Path(tmp) / 'test.stl'}), \
             patch('tools.cfmesh_pipeline.prepare_geometry') as geometry:
            root = Path(__file__).resolve().parent.parent
            case = Path(tmp) / 'test_4000RPM_MRF'
            preprocessing(case.name, 4000, root, case, 2, 'MRF', False, None,
                          'kOmegaSST', 'permeable', 2.5, WALL_FUNCTIONS='no', AERODYNAMICS_ONLY=True)
            self.assertIsNone(geometry.call_args.args[2])
            self.assertIsNone(geometry.call_args.args[3])
            controls = (case / 'Parameters/controlDict.cpp').read_text()
            self.assertRegex(controls, r'impermeableEnabled\s+no;')
            self.assertRegex(controls, r'permeableEnabled\s+no;')
