"""Report selection, flow-only settings and optional real ParaView rendering."""
import inspect
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, PropertyMock, patch

import tools.visualization as visualization


class VisualizationTests(unittest.TestCase):
    def test_renderer_discovery_prefers_native_then_windows_fallback(self):
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(visualization.shutil, 'which', return_value='/usr/bin/pvpython'), \
                patch.object(visualization, 'windows_paraview_candidates') as windows:
            self.assertEqual(visualization.find_paraview_executable(), '/usr/bin/pvpython')
            windows.assert_not_called()
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(visualization.shutil, 'which', return_value=None), \
                patch.object(visualization, 'windows_paraview_candidates', return_value=iter(['/mnt/c/Program Files/ParaView/bin/pvpython.exe'])):
            self.assertTrue(visualization.find_paraview_executable().endswith('pvpython.exe'))
            with self.assertRaisesRegex(FileNotFoundError, 'ParaView executable not found'):
                visualization.find_paraview_executable('/explicit/missing/pvpython')

    def test_wsl_windows_launch_translates_paths_and_keeps_host_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()

            def launch(command, **kwargs):
                run_dir = kwargs['cwd']
                settings = json.loads((run_dir / 'settings.json').read_text())
                self.assertEqual(settings['case_path'], 'windows:' + str(root))
                self.assertEqual(command[-2], 'windows:' + str(run_dir / 'render_visuals.py'))
                self.assertEqual(command[-1], 'windows:' + str(run_dir / 'settings.json'))
                self.assertIn('OMP_NUM_THREADS', kwargs['env']['WSLENV'])
                (run_dir / 'result.json').write_text(json.dumps({
                    'status': 'complete', 'views': [{'image': 'example.png'}], 'warnings': []}))
                return MagicMock(returncode=0)

            with patch.object(visualization, 'running_in_wsl', return_value=True), \
                    patch.object(visualization, 'find_paraview_executable', return_value='/mnt/c/ParaView/pvpython.exe'), \
                    patch.object(visualization, 'paraview_runtime_path', side_effect=lambda p, bridge: 'windows:' + str(p)), \
                    patch.object(visualization.subprocess, 'run', side_effect=launch), \
                    patch.object(visualization, 'validate_visualization_images'):
                result = visualization.run_visualization_job(root, 4000, None, {'mesh_only': True})
            self.assertEqual(result['status'], 'complete', result['warnings'])
            self.assertEqual(result['settings']['case_path'], str(root))
            self.assertTrue(result['renderer']['windows_wsl_bridge'])

    def test_wsl_path_conversion_preserves_spaces(self):
        path = Path('case with spaces').resolve()
        with patch.object(visualization.subprocess, 'check_output', return_value='C:\\case with spaces\n') as convert:
            self.assertEqual(visualization.paraview_runtime_path(path, True), 'C:\\case with spaces')
            convert.assert_called_once_with(['wslpath', '-w', str(path)], text=True)

    def test_mesh_only_dispatch_does_not_read_flow_or_pressure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'settings.json'
            path.write_text(json.dumps(visualization.visualization_settings(tmp, 4000, None, {'mesh_only': True})))
            with patch.object(visualization, 'pvs', MagicMock(), create=True), \
                    patch.object(visualization, '_pvvis_mesh') as mesh, \
                    patch.object(visualization, '_pvvis_volume') as volume, \
                    patch.object(visualization, '_pvvis_surface') as surface, \
                    patch.object(visualization, '_pvvis_pressure_units') as pressure:
                visualization._pvvis_main(path)
            mesh.assert_called_once()
            volume.assert_not_called()
            surface.assert_not_called()
            pressure.assert_not_called()

    def test_full_atlas_keeps_mesh_when_flow_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'settings.json'
            path.write_text(json.dumps(visualization.visualization_settings(tmp, 4000, None)))

            def mesh(result, *_):
                result['views'].append({'chapter': 'mesh', 'title': 'Mesh domain overview'})

            with patch.object(visualization, 'pvs', MagicMock(), create=True), \
                    patch.object(visualization, '_pvvis_mesh', side_effect=mesh), \
                    patch.object(visualization, '_pvvis_volume', side_effect=ValueError('no flow fields')), \
                    patch.object(visualization, '_pvvis_pressure_units', return_value=('Pa', 'Pressure')), \
                    patch.object(visualization.traceback, 'print_exc'):
                visualization._pvvis_main(path)
            result = json.loads((Path(tmp) / 'result.json').read_text())
            self.assertEqual(result['status'], 'partial')
            self.assertEqual(result['views'][0]['chapter'], 'mesh')

    def test_report_chapters_are_separate_and_mesh_only_filters_flow(self):
        from reportlab.pdfgen import canvas
        from pypdf import PdfReader

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            visuals = root / 'report' / 'visuals'
            visuals.mkdir(parents=True)
            views = [{'title': title, 'caption': 'Example', 'image': 'missing.png'}
                     for title in ('Flow slice - speed', 'Blade surface mesh (front)',
                                   'Mesh section (xy)', 'Surface pressure')]
            (visuals / 'manifest.json').write_text(json.dumps({
                'status': 'complete', 'views': views, 'warnings': [],
                'settings': {'report_max_views': 1}}))
            for mesh_only in (True, False):
                output = root / f'{mesh_only}.pdf'
                pdf = canvas.Canvas(str(output))
                visualization.append_visualization_report(pdf, root, mesh_only=mesh_only)
                pdf.save()
                content = '\n'.join(p.extract_text() for p in PdfReader(output).pages)
                self.assertIn('Mesh - Geometry, Refinement and Near-Wall Layers', content)
                self.assertEqual('Flow and Blade-Wall Diagnostics' in content, not mesh_only)
                self.assertEqual('Acoustic Surface Diagnostics' in content, not mesh_only)

    @unittest.skipUnless(os.environ.get('PARAVIEW_RENDER_TESTS') == '1', 'requires offscreen ParaView')
    def test_real_mesh_only_atlas_without_fields_times_or_diameter(self):
        from mesh_case_fixture import write_mesh_case
        from createSimulationReport import create_simulation_report
        from pypdf import PdfReader

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'unnamed-case'
            write_mesh_case(root)
            manifest = visualization.run_visualization_job(root, 4000, None, {
                'mesh_only': True, 'image_resolution': [1200, 800]})
            self.assertNotEqual(manifest['status'], 'failed', manifest['warnings'])
            self.assertGreaterEqual(len(manifest['views']), 20, manifest['warnings'])
            self.assertEqual({v['chapter'] for v in manifest['views']}, {'mesh'})
            self.assertFalse(any('Near-wall flow' in w for w in manifest['warnings']))
            self.assertEqual(manifest['mesh_summary']['cells'], 736)
            report = create_simulation_report(root, 4000, 'AMI', 'kOmegaSST', mesh_only=True, quiet=True)
            pdf = PdfReader(report['output_pdf'])
            self.assertEqual(sum(len(page.images) for page in pdf.pages), len(manifest['views']))
            text = '\n'.join(page.extract_text() for page in pdf.pages)
            self.assertIn('Mesh - Geometry, Refinement and Near-Wall Layers', text)
            self.assertNotIn('Flow and Blade-Wall Diagnostics', text)

    def test_mesh_views_accept_zero_time_without_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'constant' / 'polyMesh').mkdir(parents=True)
            settings = visualization.visualization_settings(
                tmp, 4000, None, {'mesh_only': True, 'diameter_m': 0.25})
            reader = MagicMock()
            reader.CellArrays.Available = []
            reader.TimestepValues = [0.0]
            reader.GetDataInformation.return_value.GetBounds.return_value = [-1, 1] * 3
            reader.GetDataInformation.return_value.GetNumberOfCells.return_value = 10
            reader.GetDataInformation.return_value.GetNumberOfPoints.return_value = 20
            pvs = MagicMock()
            pvs.OpenFOAMReader.return_value = reader
            type(pvs.Slice.return_value).SliceType = PropertyMock(return_value=MagicMock())
            result = {'views': [], 'warnings': []}
            with patch.object(visualization, 'pvs', pvs, create=True), \
                    patch.object(visualization, '_pvvis_crop', return_value=MagicMock()), \
                    patch.object(visualization, '_pvvis_render') as render, \
                    patch.object(visualization, '_pvvis_wall') as wall:
                visualization._pvvis_volume(result, settings, Path(tmp), MagicMock(), ('Pa', 'Pressure'))
            self.assertEqual(reader.SkipZeroTime, 0)
            self.assertEqual(reader.CellArrays, [])
            self.assertTrue(render.called)
            self.assertTrue(all(call.args[5].startswith(('Mesh section', 'Mesh domain')) for call in render.call_args_list))
            self.assertEqual(wall.call_args.args[1], 0.0)
            self.assertEqual(result['warnings'], [])

    def test_mesh_only_report_without_solver_outputs(self):
        from createSimulationReport import create_simulation_report

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            visuals = root / 'report' / 'visuals'
            visuals.mkdir(parents=True)
            (visuals / 'manifest.json').write_text(json.dumps({
                'status': 'disabled', 'views': [], 'warnings': [],
                'settings': {'mesh_only': True},
            }))
            create_simulation_report(root, 4000, 'AMI', 'kOmegaSST', mesh_only=True, quiet=True)
            self.assertTrue((root / 'report' / 'simulation_report.pdf').read_bytes().startswith(b'%PDF'))

    def test_mesh_only_postprocessing_skips_solver_data(self):
        import postprocessing as module

        with patch.object(module, 'run_visualization') as render, \
                patch.object(module, 'create_simulation_report') as report, \
                patch.object(module, 'merge_postprocessing_dat_files') as merge:
            module.postprocessing(None, 'case', 4000, 'AMI', 'kOmegaSST', MESH_ONLY=True)
        self.assertEqual(render.call_args.kwargs['config'], {'mesh_only': True})
        self.assertTrue(report.call_args.kwargs['mesh_only'])
        merge.assert_not_called()

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
            script_path = visualization.paraview_runtime_path(
                root / 'check.py', visualization.running_in_wsl() and executable.lower().endswith('.exe'))
            run = subprocess.run([executable, '--disable-registry', '--force-offscreen-rendering',
                                  script_path], capture_output=True, text=True, timeout=120)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            result = json.loads((root / 'result.json').read_text())
            self.assertEqual(len(result['views']), 4)
