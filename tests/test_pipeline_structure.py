"""Regression checks for moved helpers, case preservation and process entry points."""
import ast
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import tools
from tools.cfmesh_pipeline import ROOT, safe_path
from tools.cfmesh_orders import seed_current_stl
from tools.geometry import read_stl
from tools.preprocessing import prepare_case_directory, run_preprocessing
from tools.scheduler import SimulationOrderStore, calculate_scheduler_layout
from tools.cli import create_parser
from tools.orders import prepare_simulation_order


class PipelineStructureTests(unittest.TestCase):
    def test_stage_files_have_one_top_level_function(self):
        for path in ROOT.glob('*.py'):
            tree = ast.parse(path.read_text(encoding='utf-8'))
            self.assertEqual(len([n for n in tree.body if isinstance(n, ast.FunctionDef)]), 1, path.name)

    def test_existing_helper_imports_and_shared_locks(self):
        for name in tools._EXPORTS:
            getattr(tools, name)
        from tools.common import MATPLOTLIB_LOCK
        self.assertIs(tools.MATPLOTLIB_LOCK, MATPLOTLIB_LOCK)

    def test_order_store_persists_and_snapshots_are_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            tools.save_simulation_order(directory, {'cases': [{'folder': 'case', 'status': 'pending'}]})
            store = SimulationOrderStore(directory)
            snapshot = store.snapshot()
            snapshot['cases'][0]['status'] = 'changed'
            self.assertEqual(store.case_status('case'), 'pending')
            store.set_status('case', 'preprocessing_done')
            self.assertEqual(SimulationOrderStore(directory).case_status('case'), 'preprocessing_done')

    def test_scheduler_preserves_core_budget(self):
        cases = [{'mesh': 'propeller', 'folder': f'rpm{i}'} for i in range(3)]
        independent = calculate_scheduler_layout(cases, 8, 'off', False)
        chained = calculate_scheduler_layout(cases, 8, 'on', False)
        self.assertEqual(independent['max_parallel_cases'], 3)
        self.assertEqual(independent['cores_per_case'] * 3 + independent['extra_core_slots'], 8)
        self.assertEqual(chained['max_parallel_cases'], 1)
        self.assertEqual(chained['cores_per_case'], 8)

    def test_explicit_case_cores_limit_concurrency(self):
        cases = [{'mesh': 'propeller', 'folder': f'rpm{i}'} for i in range(200)]
        for total, target, slots in [(100, 20, 5), (100, 10, 10), (103, 20, 5)]:
            layout = calculate_scheduler_layout(cases, total, 'off', False, target)
            self.assertEqual(layout['max_parallel_cases'], slots)
            tools.assign_case_core_allocations(cases, layout, 'off', False)
            self.assertEqual({case['allocated_cores'] for case in cases}, {target})
        chained = calculate_scheduler_layout(cases, 100, 'on', False, 20)
        self.assertEqual(chained['max_parallel_cases'], 1)
        self.assertEqual(chained['cores_per_case'], 20)
        for invalid in [0, -1, 101]:
            with self.assertRaises(ValueError):
                calculate_scheduler_layout(cases, 100, 'off', False, invalid)

    def test_case_copy_preserves_previous_results_and_uses_current_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            template = base / 'template'
            (template / '0/uniform').mkdir(parents=True)
            (template / '0/uniform/time').write_text('stale restart metadata')
            parameters = base / 'parameters'
            parameters.mkdir()
            (parameters / 'controlDict.cpp').write_text('deltaT 0.001;')
            case = base / 'case'
            case.mkdir()
            (case / 'result').write_text('keep this')
            prepare_case_directory(template, parameters, case)
            archived = list(base.glob('case_PREVIOUS_*'))
            self.assertEqual(len(archived), 1)
            self.assertEqual((archived[0] / 'result').read_text(), 'keep this')
            self.assertFalse((case / '0/uniform/time').exists())
            self.assertEqual((case / 'Parameters/controlDict.cpp').read_text(), 'deltaT 0.001;')

    def test_example_selection_and_geometry_reader(self):
        vertices, faces, volume, digest = read_stl(ROOT / 'examples/STL/10x7E.stl', 0.001)
        self.assertTrue(vertices and faces)
        self.assertNotEqual(volume, 0)
        self.assertEqual(len(digest), 64)
        with tempfile.TemporaryDirectory() as tmp:
            seed_current_stl(tmp)
            target = Path(tmp) / 'STL/10x7E.stl'
            self.assertEqual(target.read_bytes(), (ROOT / 'examples/STL/10x7E.stl').read_bytes())
            target.write_text('user input')
            seed_current_stl(tmp)
            self.assertEqual(target.read_text(), 'user input')

    def test_output_guard_accepts_dedicated_local_run(self):
        self.assertEqual(safe_path(ROOT / 'runs/my-case'), ROOT / 'runs/my-case')
        for path in [ROOT, ROOT / 'tools/case', ROOT / 'Parameters/case']:
            with self.assertRaises(ValueError):
                safe_path(path)

    def test_moved_preprocessing_launcher_logs_and_forwards_status(self):
        class Process:
            stdin = io.StringIO()
            stdout = io.StringIO('PIPELINE_STATUS {"stage": "geometry"}\nmeshed\n')
            def wait(self):
                return 0

        with tempfile.TemporaryDirectory() as tmp, patch('tools.preprocessing.subprocess.Popen', return_value=Process()) as start:
            events = []
            run_preprocessing('case', 4000, ROOT, Path(tmp) / 'case', 1, 'AMI',
                              False, None, 'kOmegaSST', None, None,
                              STATUS_CALLBACK=lambda **event: events.append(event))
            self.assertEqual(Path(start.call_args.args[0][-1]), ROOT / 'preprocessing.py')
            self.assertIn({'stage': 'geometry'}, events)
            self.assertIn('meshed', (Path(tmp) / 'case.preprocessing.log').read_text())

    def test_help_from_external_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, '-I', str(ROOT / 'main.py'), '--help'],
                                    cwd=tmp, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('--boundary-layers', result.stdout)

    def test_new_order_and_resume_keep_stored_settings(self):
        with tempfile.TemporaryDirectory() as tmp, patch('tools.orders.preflight'):
            parser = create_parser()
            args = parser.parse_args(['--sim-dir', tmp, '--rpms', '4000',
                                      '--mode', 'AMI', '--turbulence', 'kOmegaSST', '--wall-functions', 'no',
                                      '--total-cores', '4', '--cores-per-case', '4', '--mesh-only'])
            _, _, meshes, _, order = prepare_simulation_order(args, parser)
            self.assertIn('10x7E', meshes)
            self.assertTrue(order['mesh_only'])
            self.assertEqual(order['total_cores'], 4)
            resumed = parser.parse_args(['--sim-dir', tmp, '--resume'])
            _, _, _, _, restored = prepare_simulation_order(resumed, parser)
            self.assertTrue(resumed.mesh_only)
            self.assertEqual(resumed.rpms, [4000])
            self.assertEqual(restored['cases'], order['cases'])
