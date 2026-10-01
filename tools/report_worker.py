"""Run PDF generation outside the scheduler so native crashes are recoverable."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from tools.common import emit_status, _atomic_write_json


LATEST_REPORT = Path(__file__).resolve().parents[1] / 'output' / 'simulation_report.pdf'


def publish_latest_report(source, destination):
    """Replace the convenience copy only after a complete copy is available."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.latest-report-', dir=destination.parent) as temporary:
        staged = Path(temporary) / 'report.pdf'
        shutil.copyfile(source, staged)
        os.replace(staged, destination)


def create_simulation_report(*, status_callback=None, latest_report_path=None, **kwargs):
    case = Path(kwargs['case_path']).resolve()
    report = case / 'report'
    report.mkdir(parents=True, exist_ok=True)
    kwargs['case_path'] = str(case)
    if kwargs.get('output_pdf') is not None:
        kwargs['output_pdf'] = str(Path(kwargs['output_pdf']).resolve())
    log_path = report / 'report-generation.log'
    status_path = report / 'report-generation-status.json'
    _atomic_write_json(status_path, {'detail': 'starting report worker'})
    with tempfile.TemporaryDirectory(prefix='.report-', dir=report) as temporary:
        request = Path(temporary) / 'request.json'
        request.write_text(json.dumps(kwargs), encoding='utf-8')
        env = dict(os.environ, MPLBACKEND='Agg', PYTHONFAULTHANDLER='1',
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        started = time.monotonic()
        with log_path.open('w', encoding='utf-8') as log:
            process = subprocess.Popen(
                [sys.executable, '-u', '-X', 'faulthandler', '-m', 'tools.report_worker', str(request)],
                cwd=Path(__file__).resolve().parents[1], env=env,
                stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
            )
            try:
                while True:
                    try:
                        code = process.wait(timeout=1)
                        break
                    except subprocess.TimeoutExpired:
                        try:
                            status = json.loads(status_path.read_text(encoding='utf-8'))
                        except (OSError, ValueError):
                            status = {'detail': 'generating report'}
                        emit_status(status_callback, stage='report', progress=status.get('progress', 75),
                                    detail=f"{status['detail']} ({time.monotonic()-started:.0f}s elapsed)")
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
        if code:
            status = json.loads(status_path.read_text(encoding='utf-8'))
            message = f"Report worker exited with code {code} during {status['detail']}; see {log_path}"
            _atomic_write_json(status_path, dict(status, status='failed', error=message))
            raise RuntimeError(message)
    destination = Path(kwargs.get('output_pdf') or report / 'simulation_report.pdf')
    latest = Path(latest_report_path) if latest_report_path is not None else LATEST_REPORT
    try:
        publish_latest_report(destination, latest)
    except OSError as exc:
        message = f'Case report saved at {destination}, but could not refresh {latest}: {exc}'
        _atomic_write_json(status_path, {'status': 'failed', 'detail': message})
        raise RuntimeError(message) from exc


def main(request):
    kwargs = json.loads(Path(request).read_text(encoding='utf-8'))
    report = Path(kwargs['case_path']) / 'report'
    started = time.monotonic()

    def status_callback(**status):
        status['elapsed_seconds'] = time.monotonic() - started
        _atomic_write_json(report / 'report-generation-status.json', status)
        print(f"{status['elapsed_seconds']:.3f}s: {status['detail']}", flush=True)

    status_callback(detail='loading report libraries', progress=75)
    from createSimulationReport import create_simulation_report as generate

    destination = Path(kwargs.get('output_pdf') or report / 'simulation_report.pdf')
    # Keep an existing PDF intact if generation crashes or raises an exception.
    with tempfile.TemporaryDirectory(prefix='.pdf-', dir=destination.parent) as temporary:
        kwargs['output_pdf'] = str(Path(temporary) / 'report.pdf')
        generate(**kwargs, status_callback=status_callback)
        os.replace(kwargs['output_pdf'], destination)
    status_callback(detail='report complete', progress=100, status='complete')


if __name__ == '__main__':
    main(sys.argv[1])
