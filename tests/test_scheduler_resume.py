"""Native-mesh recovery must preserve completed work after report failures."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.scheduler import SimulationOrderStore, reactivate_failed_cases_for_resume


class SchedulerResumeTests(unittest.TestCase):
    def test_native_recovery_checkpoints(self):
        scenarios = [
            # mesh_only, checkpoint, case exists, safe timestep, expected
            (True, 'solver_done', True, None, 'solver_done'),
            (False, 'solver_done', True, None, 'solver_done'),
            (True, 'solver_done', False, None, 'pending'),
            (True, 'solver_running', True, None, 'pending'),
            (False, 'solver_running', True, None, 'pending'),
            (False, 'solver_running', True, '0.1', 'solver_running'),
        ]
        for mesh_only, checkpoint, exists, timestep, expected in scenarios:
            with self.subTest(scenario=(mesh_only, checkpoint, exists, timestep)):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    if exists:
                        (root / 'case').mkdir()
                        (root / 'case' / 'completed-mesh').write_text('preserve')
                    (root / 'simulation_order.json').write_text(json.dumps({
                        'meshing_backend': 'cfmesh-native-rotor-stator-v1',
                        'cases': [{'folder': 'case', 'mesh_only': mesh_only,
                                   'status': 'failed', 'resume_status': checkpoint,
                                   'error': 'report or solver failed'}],
                    }))
                    store = SimulationOrderStore(root)
                    with patch('tools.openfoam.get_safe_timestep', return_value=timestep) as safe:
                        reactivate_failed_cases_for_resume(store, root)
                    case = store.get_case('case')
                    self.assertEqual(case['status'], expected)
                    self.assertIsNone(case['resume_status'])
                    self.assertIsNone(case['error'])
                    if checkpoint == 'solver_done':
                        safe.assert_not_called()
                    if exists:
                        self.assertEqual((root / 'case' / 'completed-mesh').read_text(), 'preserve')


if __name__ == '__main__':
    unittest.main()
