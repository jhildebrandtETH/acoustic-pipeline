"""Opt-in OpenFOAM 13 NCC smoke test: OPENFOAM_TEMPLATE_SMOKE=1 in Linux/WSL."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tools.templates import template_directory


@unittest.skipUnless(os.environ.get('OPENFOAM_TEMPLATE_SMOKE') == '1',
                     'requires Docker and the OpenFOAM 13 image')
class TemplateNccTests(unittest.TestCase):
    def test_all_initial_fields_survive_coupling(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='template-ncc-') as temporary:
            for mode, model, treatment in [
                (mode, model, treatment)
                for mode in ('AMI', 'MRF')
                for model in ('kOmegaSST', 'kEpsilon')
                for treatment in ('yes', 'no')
            ] + [('AMI', 'DES', 'no')]:
                with self.subTest(mode=mode, model=model, treatment=treatment):
                    case = Path(temporary) / f'{mode}-{model}-{treatment}'
                    shutil.copytree(template_directory(mode, model, treatment), case)
                    shutil.copytree(root / 'Parameters', case / 'Parameters')
                    # Static two-block fixture isolates field loading and NCC creation.
                    (case / 'constant/dynamicMeshDict').unlink(missing_ok=True)
                    (case / 'system/controlDict').write_text(
                        'FoamFile { format ascii; class dictionary; object controlDict; }\n'
                        'application foamRun; startFrom startTime; startTime 0; '
                        'stopAt endTime; endTime 1; deltaT 1; '
                        'writeControl timeStep; writeInterval 1;\n')
                    (case / 'system/blockMeshDict').write_text('''
FoamFile { format ascii; class dictionary; object blockMeshDict; }
convertToMeters 1;
vertices
(
 (0 0 0) (1 0 0) (1 1 0) (0 1 0)
 (0 0 1) (1 0 1) (1 1 1) (0 1 1)
 (1 0 0) (2 0 0) (2 1 0) (1 1 0)
 (1 0 1) (2 0 1) (2 1 1) (1 1 1)
);
blocks
(
 hex (0 1 2 3 4 5 6 7) (2 2 2) simpleGrading (1 1 1)
 hex (8 9 10 11 12 13 14 15) (2 2 2) simpleGrading (1 1 1)
);
edges ();
boundary
(
 inlet { type patch; faces ((0 4 7 3)); }
 outlet { type patch; faces ((9 10 14 13)); }
 propeller { type wall; faces ((0 1 5 4) (8 9 13 12)); }
 walls { type patch; faces ((3 7 6 2) (11 15 14 10)
                           (0 3 2 1) (8 11 10 9)
                           (4 5 6 7) (12 13 14 15)); }
 rotaryRegion { type patch; faces ((1 2 6 5)); }
 rotaryRegion_slave { type patch; faces ((8 12 15 11)); }
);
mergePatchPairs ();
''')
                    result = subprocess.run([
                        'docker', 'run', '--rm', '--user', f'{os.getuid()}:{os.getgid()}',
                        '-v', f'{case}:/case', '-w', '/case', '--entrypoint', 'bash',
                        'microfluidica/openfoam:13', '-lc',
                        'source /opt/openfoam13/etc/bashrc && '
                        'blockMesh > log.blockMesh 2>&1 && '
                        'createNonConformalCouples -fields rotaryRegion_slave rotaryRegion'
                    ], text=True, capture_output=True, timeout=120)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertNotIn('Using dynamicCode', result.stdout)
                    self.assertFalse((case / 'dynamicCode').exists())
                    self.assertIn('nonConformalCyclic',
                                  (case / 'constant/polyMesh/boundary').read_text())


if __name__ == '__main__':
    unittest.main()
