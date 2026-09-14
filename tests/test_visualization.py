"""Report selection, flow-only settings and optional real ParaView rendering."""
import inspect
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import tools.visualization as visualization


class VisualizationTests(unittest.TestCase):
    def test_flow_only_settings_and_log_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = visualization.visualization_settings(tmp, 4000, None)
            self.assertIsNone(settings['acoustic_surface'])
            self.assertEqual(settings['report_max_views'], 32)
            with self.assertRaises(ValueError):
                visualization.visualization_settings(tmp, 4000, None, {'log_fields': ['p']})

    def test_report_selection_keeps_distinct_quantities_and_latest_time(self):
        views = [{'title': f'Flow slice - {field} ({station})', 'time_s': t}
                 for t in [1, 2, 3] for station in ['x=0', 'y=0']
                 for field in ['speed', 'p', 'vorticity']]
        selected = visualization.select_report_views(views, 3)
        self.assertEqual(len(selected), 3)
        self.assertEqual({item['time_s'] for item in selected}, {3})
        self.assertEqual(len({item['title'] for item in selected}), 3)
        self.assertEqual(visualization.select_report_views(views, 50), views)

    @unittest.skipUnless(os.environ.get('PARAVIEW_RENDER_TESTS') == '1', 'requires offscreen ParaView')
    def test_offscreen_cell_point_and_large_offset_color_rendering(self):
        executable = visualization.find_paraview_executable()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = ('import json, math, traceback, sys\nfrom pathlib import Path\n'
                      'import numpy as np\nfrom paraview import simple as pvs, servermanager\n')
            script += '\n\n'.join(inspect.getsource(value) for name, value in vars(visualization).items()
                                  if name.startswith('_pvvis_') and inspect.isfunction(value))
            script += '''
run_dir = Path(__file__).resolve().parent
pvs._DisableFirstRenderCameraReset()
view = pvs.CreateView('RenderView')
view.ViewSize = [1200, 800]
view.UseColorPaletteForBackground = 0
view.Background = [1, 1, 1]
view.BackgroundColorMode = 'Single Color'
settings = {'image_resolution': [1200, 800], 'color_ranges': {}, 'rpm': 4000,
            'log_fields': ['vorticity_magnitude']}
result = {'views': [], 'warnings': []}
sphere = pvs.Sphere(ThetaResolution=48, PhiResolution=48)
pressure = pvs.Calculator(Input=sphere)
pressure.ResultArrayName = 'p'
pressure.Function = '101325 + 10*coordsX'
cells = pvs.PointDatatoCellData(Input=pressure)
_pvvis_render(cells, result, settings, run_dir, view, 'Cell pressure', 'Synthetic',
              camera='front', field=('CELLS', 'p', 'Kinematic pressure'))
lut = pvs.GetColorTransferFunction('p')
bar = pvs.GetScalarBar(lut, view)
fmt = str(bar.LabelFormat)
labels = [fmt.format(v) if '{' in fmt else fmt % v for v in bar.CustomLabels]
assert len(set(labels)) == len(labels), labels
vortex = pvs.Calculator(Input=sphere)
vortex.ResultArrayName = 'vorticity_magnitude'
vortex.Function = 'exp(12*coordsX)'
_pvvis_render(vortex, result, settings, run_dir, view, 'Vorticity', 'Synthetic',
              field=('POINTS', 'vorticity_magnitude', 'Vorticity [1/s]'))
assert result['views'][-1]['color_scale'] == 'logarithmic'
# A remote observer changes the camera scale; the next close-up must remain visible.
observer = pvs.Sphere(Center=[10, 0, 0], Radius=.05)
group = pvs.GroupDatasets(Input=[sphere, observer])
_pvvis_render(group, result, settings, run_dir, view, 'Observer context', 'Synthetic')
_pvvis_render(cells, result, settings, run_dir, view, 'Mesh and scalar', 'Synthetic',
              camera='back', field=('CELLS', 'p', 'Kinematic pressure'), edges=True)
# An empty plot with a surviving scalar bar must not pass the image check.
pvs.HideAll(view)
view.OrientationAxesVisibility = 0
view.AxesGrid.Visibility = 0
bar.Visibility = 1
pvs.Render(view)
pvs.SaveScreenshot(str(run_dir / 'blank.png'), view, ImageResolution=[1200,800])
try:
    _pvvis_check_foreground(run_dir / 'blank.png')
except ValueError:
    pass
else:
    raise AssertionError('Blank plot with scalar bar was accepted')
'''
            (root / 'check.py').write_text(script, encoding='utf-8')
            run = subprocess.run([executable, '--disable-registry', '--force-offscreen-rendering',
                                  str(root / 'check.py')], capture_output=True, text=True, timeout=120)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            result = json.loads((root / 'result.json').read_text())
            self.assertEqual(len(result['views']), 4)
