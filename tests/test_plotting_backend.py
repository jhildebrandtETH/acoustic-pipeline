"""Worker reports must not create Tk objects, even with a GUI backend selected."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class PlottingBackendTests(unittest.TestCase):
    def test_gui_environment_cannot_enable_tk_in_worker_reports(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            script = r'''
import gc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd() / 'tests'))
from test_layer_thickness import write_boxes
from tools.common import MATPLOTLIB_LOCK

def report(index):
    # Import report/plotting code from the worker, as standalone callers may do.
    with MATPLOTLIB_LOCK:
        from createSimulationReport import create_simulation_report
        case = Path(sys.argv[1]) / str(index)
        write_boxes(case)
        result = create_simulation_report(case, 4000, 'AMI', 'kOmegaSST', mesh_only=True, quiet=True)
        assert result['layer_thickness']['status'] == 'complete'
        assert Path(result['output_pdf']).read_bytes().startswith(b'%PDF')
        gc.collect()

with ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(report, range(2)))
gc.collect()
import matplotlib
from tools.plotting import pyplot
assert matplotlib.get_backend().lower() == 'agg', matplotlib.get_backend()
assert not pyplot.get_fignums(), 'Plot figures were left open'
assert 'matplotlib.backends.backend_tkagg' not in sys.modules
assert 'tkinter' not in sys.modules
print('Worker reports and cleanup completed using Agg')
'''
            env = dict(os.environ, MPLBACKEND='TkAgg', MPLCONFIGDIR=str(Path(tmp)/'mpl'))
            run = subprocess.run([sys.executable, '-c', script, tmp], cwd=repo, env=env,
                                 capture_output=True, text=True, timeout=90)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn('completed using Agg', run.stdout)
            self.assertNotIn('Exception ignored', run.stderr)
            self.assertNotIn('main thread is not in main loop', run.stderr)


if __name__ == '__main__':
    unittest.main()
