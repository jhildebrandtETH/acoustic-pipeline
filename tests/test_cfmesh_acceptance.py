"""Acceptance uses standard checks while preserving extended diagnostics."""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import cfmesh_pipeline as pipeline
from tools.cfmesh_controls import DEFAULTS


class MeshAcceptanceTests(unittest.TestCase):
    def run_case(self, *, extended_ok=False, standard_ok=True, ncc_ok=True,
                 volume=1.0, coverage=1.0, allow_bad=False):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        case = Path(temp.name)
        for role in ('rotor', 'stator'):
            mesh = case / 'cfmesh' / role / 'constant/polyMesh'
            mesh.mkdir(parents=True)
            (mesh / 'owner').write_text('FoamFile {}\n1\n(0)\n')
            (mesh / 'neighbour').write_text('FoamFile {}\n0\n()\n')
            (mesh / 'cellZones').write_text('rotaryRegion')
        (case / 'cfmesh/geometry.json').write_text(
            json.dumps({'expected_fluid_volume_m3': 1.0}))
        patches = {name: {'nFaces': 1, 'type': 'patch'} for name in (
            'inlet', 'outlet', 'walls', 'propeller', 'rotaryRegion', 'rotaryRegion_slave')}
        coupled = dict(patches, ncc1={'nFaces': 0, 'type': 'nonConformalCyclic'},
                       ncc2={'nFaces': 0, 'type': 'nonConformalCyclic'})
        commands = []

        def run(container, root, command, log_name, *args):
            commands.append((command, log_name))
            if command[0] == 'checkMesh':
                success = (extended_ok if log_name.endswith('.extended') else
                           ncc_ok if log_name.endswith('.NCC') else standard_ok)
                text = f'Number of regions: 2\nTotal volume = {volume}\n'
                text += 'Mesh OK.\n' if success else 'Failed 2 mesh checks.\n'
                (root / log_name).write_text(text)
            elif command[0] == 'createNonConformalCouples':
                (root / log_name).write_text(''.join(
                    f'{side} min/average/max coverage = {coverage}/1/1\n'
                    for side in ('Source', 'Target')))

        controls = copy.deepcopy(DEFAULTS)
        controls['interfaceProjection']['enabled'] = False
        with ExitStack() as stack:
            for name, kwargs in (
                ('query', {'return_value': 'surfaceFile "domain.ftr";'}),
                ('optional', {'return_value': None}),
                ('native_mesh_run', {}),
                ('docker_run', {'side_effect': run}),
                ('patch_info', {'side_effect': [patches, coupled]}),
            ):
                stack.enter_context(patch.object(pipeline, name, **kwargs))
            stack.enter_context(patch('tools.cfmesh_controls.read_controls', return_value=controls))
            result = pipeline.run_mesh(None, case, 4, allow_bad)
        return result, case, commands

    def test_extended_failure_is_recorded_but_standard_success_is_accepted(self):
        result, case, commands = self.run_case()
        self.assertTrue(result)
        status = json.loads((case / 'cfmesh/mesh-status.json').read_text())
        self.assertFalse(status['extended_mesh_diagnostics_ok'])
        self.assertTrue(status['mesh_quality_ok'])
        self.assertTrue(status['ncc_mesh_quality_ok'])
        self.assertEqual([item for item in commands if item[0][0] == 'checkMesh'], [
            (['checkMesh', '-allGeometry', '-allTopology'], 'log.checkMesh.extended'),
            (['checkMesh'], 'log.checkMesh'),
            (['checkMesh'], 'log.checkMesh.NCC'),
        ])
        self.assertIn('Failed 2', (case / 'log.checkMesh.extended').read_text())
        self.assertIn('Mesh OK.', (case / 'log.checkMesh').read_text())

    def test_extended_success_is_recorded(self):
        _, case, _ = self.run_case(extended_ok=True)
        self.assertTrue(json.loads((case / 'cfmesh/mesh-status.json').read_text())[
            'extended_mesh_diagnostics_ok'])

    def test_standard_and_ncc_failures_still_block(self):
        for options in ({'standard_ok': False}, {'ncc_ok': False}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, 'checks failed'):
                self.run_case(**options)

    def test_volume_and_coverage_remain_mandatory_even_with_allow_bad(self):
        for options, message in (({'volume': 1.1}, 'maxRelativeVolumeError'),
                                 ({'coverage': 0.5}, 'coverage is inadequate')):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, message):
                self.run_case(allow_bad=True, **options)


if __name__ == '__main__':
    unittest.main()
