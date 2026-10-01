"""Native regression: each surface-distance control must affect a real mesh."""
import os
from pathlib import Path
import re
import shutil
import tempfile
import unittest

from tools.cfmesh_pipeline import ROOT, header, native_run, strip_header, write_ftr


@unittest.skipUnless(os.environ.get('CFMESH_DISTANCE_TESTS') == '1', 'requires native cfMesh')
class DistanceBandTests(unittest.TestCase):
    def test_outer_level_outer_distance_and_inner_level_affect_mesh(self):
        import trimesh

        counts = {}
        with tempfile.TemporaryDirectory(prefix='cfmesh-distance-') as tmp:
            for name, outer, distance, inner, inner_distance in (
                ('baseline', 1, 0.004, 3, 0.0006), ('outer_level', 2, 0.004, 3, 0.0006),
                ('outer_distance', 1, 0.008, 3, 0.0006), ('inner_level', 1, 0.004, 4, 0.0006),
                ('inner_distance', 1, 0.004, 3, 0.002),
            ):
                case = Path(tmp) / name
                shutil.copytree(ROOT/'Parameters', case/'Parameters')
                common = case/'Parameters/cfmeshCommon.cpp'
                text = common.read_text()
                for key, value in dict(baseCellSize=0.004, propellerLevel=outer,
                                       propellerRefinementThickness=distance,
                                       propellerNearLevel=inner,
                                       propellerNearRefinementThickness=inner_distance).items():
                    text = re.sub(r'\b' + key + r'\s+[^;]+;', f'{key} {value};', text)
                common.write_text(text)
                (case/'system').mkdir()
                (case/'constant/triSurface').mkdir(parents=True)
                domain = trimesh.creation.box(extents=[0.04]*3)
                blade = trimesh.creation.box(extents=[0.012, 0.002, 0.006])
                blade.export(case/'constant/triSurface/propeller.stl')
                write_ftr(case/'constant/triSurface/domain.ftr', [
                    ('walls', 'wall', domain.vertices, domain.faces, False),
                    ('propeller', 'wall', blade.vertices, blade.faces, True),
                ])
                (case/'system/meshDict').write_text(header('meshDict') +
                    '#include "../Parameters/cfmeshRotorDict"\nworkflowControls { stopAfter edgeExtraction; }\n')
                (case/'system/controlDict').write_text(header('controlDict') + '''
application cartesianMesh; startFrom startTime; startTime 0; stopAt endTime;
endTime 1; deltaT 1; writeControl timeStep; writeInterval 1; writeFormat ascii;
''')
                result = native_run(['cartesianMesh'], cwd=case, cores=2, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                maximum = -1
                for filename in ('owner', 'neighbour'):
                    data = strip_header(case/'constant/polyMesh'/filename)
                    maximum = max(maximum, max(map(int, data[data.index('(')+1:data.rindex(')')].split())))
                counts[name] = maximum + 1
            for name in ('outer_level', 'outer_distance', 'inner_level', 'inner_distance'):
                self.assertGreater(counts[name], counts['baseline'], counts)
        print('Native two-band counts:', counts)
