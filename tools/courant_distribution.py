"""Whole-cell Courant distributions from the latest reconstructed snapshot."""
import json
import re
from pathlib import Path

import numpy as np

from diagnoseCourant import read_ascii_internal
from .layer_thickness import _read_list
from .mesh_volumes import measure_cell_volumes
from .plotting import pyplot as plt


def _times(root):
    return sorted((float(p.name), p) for p in root.iterdir()
                  if p.is_dir() and re.fullmatch(r'\d+(?:\.\d*)?(?:[eE][-+]?\d+)?', p.name)
                  and np.isfinite(float(p.name)) and float(p.name) > 0)


def courant_statistics(values, target=None):
    co = np.asarray(values, dtype=float)
    if co.ndim != 1 or not co.size or not np.isfinite(co).all() or np.any(co < 0):
        raise ValueError('Courant field must contain finite nonnegative scalars for every cell.')
    maximum = float(co.max())
    configured = target is not None and np.isfinite(target) and target > 0
    target = float(target) if configured else maximum
    rows = []
    for multiplier in (1, 2, 5, 10):
        cutoff = target / multiplier
        count = int(np.count_nonzero(co > cutoff))
        rows.append(dict(multiplier=multiplier, cutoff=cutoff, count=count,
                         cell_percent=100 * count / co.size))
    return dict(cells=int(co.size), zero_cells=int(np.count_nonzero(co == 0)),
                quantiles=dict(zip(('min', 'median', 'p95', 'p99', 'p99.9', 'max'),
                                   map(float, np.percentile(co, [0, 50, 95, 99, 99.9, 100])))),
                target_co=target, target_source='configured maxCo' if configured else 'snapshot maximum (comparison reference)',
                same_flux_multiplier=target / maximum if maximum > 0 else None,
                timestep_scenarios=rows)


def create_courant_report_data(case_path, target=None):
    case = Path(case_path)
    report = case / 'report'
    report.mkdir(parents=True, exist_ok=True)
    image, csv = report / 'courant_distribution.png', report / 'cell_courant_numbers.csv'
    volume_image = report / 'highest_courant_cell_volumes.png'
    volume_csv = report / 'highest_courant_cell_volumes.csv'
    for path in (image, csv, volume_image, volume_csv):
        path.unlink(missing_ok=True)
    result = dict(status='unavailable', image=None,
                  scope='Latest saved reconstructed time; all internal cells, counted equally.')
    try:
        times = _times(case)
        if not times:
            raise ValueError('No reconstructed simulation snapshot above time zero.')
        selected, directory = times[-1]
        for processor in case.glob('processor*'):
            if processor.is_dir():
                processor_times = _times(processor)
                if processor_times and processor_times[-1][0] > selected:
                    raise ValueError('Processor output is newer; reconstruct the latest time including Co first.')
        field = next((directory / name for name in ('Co', 'CourantNo', 'CourantNumber')
                      if (directory / name).is_file() or (directory / (name + '.gz')).is_file()), None)
        if field is None:
            raise ValueError(f'No cell Courant field at latest saved time {directory.name}; enable CourantNo output and reconstruct Co.')
        # Resolve topology at this time, including meshes changed at earlier times.
        labels = []
        mesh_files = {}
        for name in ('owner', 'neighbour', 'points', 'faces'):
            candidates = [case / 'constant/polyMesh' / name] + [p / 'polyMesh' / name for _, p in times]
            path = next((p for p in reversed(candidates) if p.is_file() or p.with_name(name + '.gz').is_file()), None)
            if path is None:
                if name in ('points', 'faces'):
                    continue
                raise ValueError(f'Missing reconstructed mesh {name}.')
            mesh_files[name] = path
            if name in ('points', 'faces'):
                continue
            count, text = _read_list(path)
            array = np.fromstring(text.strip(), sep=' ', dtype=np.int64)
            if len(array) != count or np.any(array < 0):
                raise ValueError('Invalid mesh cell addressing.')
            labels.append(array)
        cell_ids = np.union1d(*labels)
        if not cell_ids.size or not np.array_equal(cell_ids, np.arange(len(cell_ids))):
            raise ValueError('Mesh cell addressing is empty or not contiguous.')
        co = read_ascii_internal(field, len(cell_ids), 1)
        result.update(courant_statistics(co, target))
        result.update(time_s=selected, field=str(field))
        # symlog preserves zero cells while showing both the bulk and a long tail.
        positive = co[co > 0]
        linear = float(positive.min()) if positive.size else 1.
        hi = max(float(co.max()), linear) * 1.05
        edges = np.unique(np.r_[0, np.geomspace(linear, hi, 41)])
        counts, _ = np.histogram(co, edges)
        fig, axes = plt.subplots(2, 1, figsize=(8, 3.4), sharex=True)
        try:
            axes[0].stairs(100 * counts / len(co), edges, fill=True, color='#4a8195')
            axes[0].set_ylabel('Cell share [%]')
            # Exact exceedance at every bin edge, retaining the maximum endpoint.
            x = np.unique(np.r_[edges, co.max()])
            exceed = len(co) - np.searchsorted(np.sort(co), x, side='right')
            axes[1].plot(x, 100 * exceed / len(co), color='#4a8195', marker='.', markersize=3)
            axes[1].set(xlabel='Cell Courant number [-] (log scale; linear near zero)',
                        ylabel='Above value [%]', ylim=(0, 101))
            for ax in axes:
                ax.set_xscale('symlog', linthresh=linear)
                ax.set_xlim(0, hi)
                ax.grid(alpha=.25)
            fig.tight_layout()
            fig.savefig(image, dpi=180)
        finally:
            plt.close(fig)
        np.savetxt(csv, co, delimiter=',', header='cell_courant_number', comments='')
        result.update(status='complete', image=str(image), csv=str(csv))
        try:
            volumes = measure_cell_volumes(case / 'constant/polyMesh', mesh_files)
            result['highest_courant_volumes'] = create_highest_courant_volume_plot(
                co, volumes, volume_image, volume_csv)
        except (OSError, ValueError, UnicodeError) as exc:
            result['highest_courant_volumes'] = dict(status='unavailable', reason=str(exc))
    except (OSError, ValueError, UnicodeError) as exc:
        result['reason'] = str(exc)
    (report / 'courant_distribution.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def create_highest_courant_volume_plot(co, volumes, image, csv):
    """Compare count-normalized distributions using matching snapshot cell IDs."""
    co, volumes = np.asarray(co), np.asarray(volumes)
    if volumes.shape != co.shape or not np.isfinite(volumes).all() or np.any(volumes <= 0):
        raise ValueError('Snapshot volumes must match the Courant cells and be finite and positive.')
    # Stable ordering resolves cutoff ties by ascending cell ID.
    ids = np.argsort(-co, kind='stable')[:1000]
    selected = volumes[ids]
    lo, hi = float(volumes.min()), float(volumes.max())
    uniform = hi / lo < 1.000001
    if uniform:
        lo, hi = lo / 1.1, hi * 1.1
    edges = np.geomspace(lo, hi, 32 if uniform else 31)
    all_counts, _ = np.histogram(volumes, edges)
    top_counts, _ = np.histogram(selected, edges)
    fig, ax = plt.subplots(figsize=(8, 2.5))
    try:
        ax.stairs(100 * all_counts / len(volumes), edges, fill=True, alpha=.3,
                  color='#4a8195', label='All mesh cells')
        ax.stairs(100 * top_counts / len(ids), edges, color='#d8822b', linewidth=1.8,
                  label=f'Highest {len(ids):,} Courant cells')
        ax.set(xscale='log', xlim=(lo, hi), ylabel='Cells per bin [%]',
               xlabel='Cell volume [m³]: smaller / finer  ←  →  larger / coarser',
               title=f'Volume regime of the highest {len(ids):,} Courant cells')
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(image, dpi=180)
    finally:
        plt.close(fig)
    np.savetxt(csv, np.column_stack((ids, co[ids], selected)), delimiter=',',
               fmt=['%d', '%.12g', '%.12g'],
               header='cell_id,cell_courant_number,cell_volume_m3', comments='')
    return dict(status='complete', image=str(image), csv=str(csv), cells=int(len(ids)),
                cutoff_co=float(co[ids[-1]]), tie_break='Ascending cell ID',
                median_volume_m3=float(np.median(selected)),
                mesh_median_volume_m3=float(np.median(volumes)))


def append_courant_report(c, result):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph, Table, TableStyle
    from xml.sax.saxutils import escape

    width, height = A4
    c.showPage()
    c.setPageSize(A4)
    c.setFillColor(colors.HexColor('#142f43'))
    c.setFont('Helvetica-Bold', 18)
    c.drawString(50, height - 52, 'Cell Courant Number Distribution')
    c.setFillColor(colors.black)
    style = ParagraphStyle('courant-distribution', fontName='Helvetica', fontSize=9, leading=12)
    y = height - 76

    def paragraph(text):
        nonlocal y
        p = Paragraph(escape(text), style)
        _, h = p.wrap(width - 100, height)
        p.drawOn(c, 50, y - h)
        y -= h + 7

    paragraph(result['scope'])
    if result['status'] != 'complete':
        paragraph('Distribution unavailable: ' + result['reason'])
        paragraph('Requires a saved ASCII cell Courant field (gzip supported) and reconstructed mesh addressing. Solver-log mean and maximum values cannot reveal how many cells limit the timestep.')
        return
    paragraph(f"Saved time: {result['time_s']:g} s | {result['cells']:,} cells | Zero Co: {result['zero_cells']:,}")
    paragraph('Co: ' + ' | '.join(f'{k}: {v:.4g}' for k, v in result['quantiles'].items()))
    ph = (width - 100) * 3.4 / 8
    c.drawImage(result['image'], 50, y - ph, width=width - 100, height=ph)
    y -= ph + 8
    volume = result.get('highest_courant_volumes', {})
    if volume.get('status') == 'complete':
        ph = (width - 100) * 2.5 / 8
        c.drawImage(volume['image'], 50, y - ph, width=width - 100, height=ph)
        y -= ph + 5
        paragraph('Each curve sums to 100% of its group. Volumes use the saved snapshot mesh. '
                  'Cutoff ties use cell ID; meshes under 1,000 cells include all cells.')
    else:
        paragraph('High-Co volume comparison unavailable: ' + volume.get('reason', 'No snapshot volumes.'))
    paragraph(f"Comparison limit: {result['target_co']:.4g} ({result['target_source']}). Counts below exceed this limit if the saved timestep is multiplied, assuming unchanged fluxes and mesh.")
    rows = [['Timestep multiplier', 'Current Co above', 'Cells exceeding', 'Cell share']]
    rows += [[f"{r['multiplier']}x", f"{r['cutoff']:.4g}", f"{r['count']:,}", f"{r['cell_percent']:.4g}%"]
             for r in result['timestep_scenarios']]
    table = Table(rows, colWidths=[135, 115, 130, width - 480])
    table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e5edf2')),
                               ('FONTSIZE', (0, 0), (-1, -1), 8),
                               ('TOPPADDING', (0, 0), (-1, -1), 5),
                               ('BOTTOMPADDING', (0, 0), (-1, -1), 5)]))
    _, th = table.wrap(width - 100, height)
    table.drawOn(c, 50, y - th)
    y -= th + 10
    multiplier = result['same_flux_multiplier']
    paragraph(f"Estimated multiplier to reach the limit at the worst cell: {multiplier:.3g}x." if multiplier is not None
              else 'All cells have zero Co; this snapshot gives no finite flow-Courant timestep bound.')
    paragraph('High-Co cells toward larger volumes warrant inspection of local fluxes and cell shape. Volume indicates relative cell size, not named refinement zones. Even one cell can limit the global timestep; this snapshot can miss transient peaks.')
    c.setFont('Helvetica', 8)
    c.drawString(50, 32, 'Data: report/courant_distribution.json and cell_courant_numbers.csv')
