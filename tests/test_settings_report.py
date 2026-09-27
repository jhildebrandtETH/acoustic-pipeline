"""Read saved settings without falling back to current repository defaults."""
import json
from pathlib import Path
import tempfile
import unittest

from tools.settings_report import create_settings_report_data, read_dictionary, resolve_dictionary


class SettingsReportTests(unittest.TestCase):
    def test_includes_overrides_and_inherited_solvers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'params').write_text('base 1e-7; dt $base; deltaT $dt; maxCo 2;')
            (root / 'dict').write_text('''#include "params"
                deltaT $deltaT; maxCo 3; // current override
                solvers { "pcorr.*" { solver GAMG; tolerance 1e-2; }
                  p { $pcorr; tolerance 1e-5; }
                  pFinal { $p; relTol 0; } }
                ''')
            sources, warnings = {}, []
            result = resolve_dictionary(read_dictionary(root/'dict', root, sources, warnings))
            self.assertEqual(result['deltaT'], '1e-7')
            self.assertEqual(result['maxCo'], '3')
            self.assertEqual(result['solvers']['pFinal'], dict(solver='GAMG', tolerance='1e-5', relTol='0'))
            self.assertEqual(len(sources), 2)
            self.assertFalse(warnings)

    def test_case_snapshots_and_regeneration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'system').mkdir()
            (root/'system/controlDict').write_text('application pimpleFoam; maxCo 0.4; deltaT 1e-8;')
            (root/'0').mkdir()
            (root/'0/U').write_text('boundaryField { inlet { type fixedValue; value uniform (0 3 0); } }')
            mesh = root/'cfmesh/rotor/system'
            mesh.mkdir(parents=True)
            (mesh/'meshDict').write_text('maxCellSize 0.123; boundaryLayers { nLayers 7; }')
            result = create_settings_report_data(root, 4000, 'AMI', 'kOmegaSST', True)
            self.assertEqual(result['documents']['cfmesh/rotor/system/meshDict']['maxCellSize'], '0.123')
            self.assertEqual(result['documents']['system/controlDict']['maxCo'], '0.4')
            self.assertIn('Not recorded', str(result['sections']))
            self.assertIn('uniform (0 3 0)', str(result['sections']))
            (root/'system/controlDict').write_text('maxCo 0.2;')
            updated = create_settings_report_data(root, 4000, 'AMI', 'kOmegaSST')
            self.assertNotEqual(result['sources'], updated['sources'])
            self.assertEqual(json.loads((root/'report/mesh_solver_settings.json').read_text()), updated)

    def test_unresolved_expressions_cycles_and_external_includes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'dict').write_text('#include "dict"\nomega #calc "2*pi"; other $missing;')
            warnings = []
            result = resolve_dictionary(read_dictionary(root/'dict', root, {}, warnings))
            self.assertIn('Cyclic include', warnings[0])
            self.assertIn('unresolved', result['omega'])
            self.assertIn('unresolved', result['other'])
            (root/'dict').write_text('#includeEtc "external"\nmaxCo 2;')
            warnings = []
            result = read_dictionary(root/'dict', root, {}, warnings)
            self.assertEqual(result['maxCo'], '2')
            self.assertIn('not resolved', warnings[0])


if __name__ == '__main__':
    unittest.main()
