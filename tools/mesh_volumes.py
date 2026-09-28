"""Initial-mesh cell volumes and count/volume-weighted distribution reporting."""
import json
import re
from pathlib import Path

import numpy as np

from .layer_thickness import _read_list
from .plotting import pyplot as plt


def measure_cell_volumes(mesh, mesh_files=None):
    """Signed volumes of the assembled ASCII (optionally gzip) polyMesh.

    Face-centre triangle fans form oriented tetrahedra about a local reference
    per cell. Neighbour contributions have reversed orientation. Local origins
    avoid cancellation when tiny cells are far from the coordinate origin.
    """
    mesh = Path(mesh)
    mesh_files = mesh_files or {}
    npoints, text = _read_list(mesh_files.get('points', mesh / 'points'))
    points = np.fromstring(text.replace('(', ' ').replace(')', ' '), sep=' ')
    if points.size != 3 * npoints or not np.isfinite(points).all():
        raise ValueError('Invalid mesh points.')
    points = points.reshape(-1, 3)
    nfaces, text = _read_list(mesh_files.get('faces', mesh / 'faces'))
    sizes = np.array([int(v) for v in re.findall(r'(\d+)\s*\(', text)], dtype=np.int64)
    vertices = np.fromstring(re.sub(r'\d+\s*\(', ' ', text).replace(')', ' '), sep=' ', dtype=np.int64)
    if (len(sizes) != nfaces or sizes.sum() != len(vertices) or np.any(sizes < 3)
            or np.any(vertices < 0) or np.any(vertices >= npoints)):
        raise ValueError('Invalid mesh faces.')
    offsets = np.r_[0, np.cumsum(sizes)]
    nowner, text = _read_list(mesh_files.get('owner', mesh / 'owner'))
    owner = np.fromstring(text.strip(), sep=' ', dtype=np.int64)
    nneighbour, text = _read_list(mesh_files.get('neighbour', mesh / 'neighbour'))
    neighbour = np.fromstring(text.strip(), sep=' ', dtype=np.int64)
    if (nowner != nfaces or len(owner) != nfaces or len(neighbour) != nneighbour
            or nneighbour > nfaces or not nfaces or np.any(owner < 0)
            or np.any(neighbour < 0)):
        raise ValueError('Invalid mesh cell addressing.')
    labels = np.union1d(owner, neighbour)
    if not np.array_equal(labels, np.arange(len(labels))):
        raise ValueError('Cell addressing is not contiguous.')
    origins = np.empty((len(labels), 3))
    origins[owner] = points[vertices[offsets[:-1]]]
    origins[neighbour] = points[vertices[offsets[:nneighbour]]]
    volumes = np.zeros(len(labels))
    # Bound temporary geometry arrays for large meshes.
    for start in range(0, nfaces, 50000):
        end = min(start + 50000, nfaces)
        local_offsets = offsets[start:end+1] - offsets[start]
        face_ids = np.repeat(np.arange(end-start), sizes[start:end])
        p = points[vertices[offsets[start]:offsets[end]]]
        next_ids = np.arange(len(p)) + 1
        next_ids[local_offsets[1:]-1] = local_offsets[:-1]
        q = p[next_ids]
        centres = np.add.reduceat(p, local_offsets[:-1]) / sizes[start:end, None]
        centre = centres[face_ids]
        cross = np.cross(p-centre, q-centre)
        cell_ids = owner[start:end][face_ids]
        tetra = np.einsum('ij,ij->i', cross, centre-origins[cell_ids]) / 6
        np.add.at(volumes, cell_ids, tetra)
        internal = face_ids + start < nneighbour
        if internal.any():
            cell_ids = neighbour[(face_ids+start)[internal]]
            tetra = np.einsum('ij,ij->i', cross[internal], centre[internal]-origins[cell_ids]) / 6
            np.add.at(volumes, cell_ids, -tetra)
    return volumes


def volume_statistics(volumes):
    volumes = np.asarray(volumes, dtype=float)
    positive = volumes[np.isfinite(volumes) & (volumes > 0)]
    result = dict(cells=int(volumes.size), positive_cells=int(positive.size),
                  nonpositive_cells=int(np.sum(np.isfinite(volumes) & (volumes <= 0))),
                  nonfinite_cells=int(np.sum(~np.isfinite(volumes))))
    if not positive.size:
        raise ValueError('No finite positive cell volumes; distribution unavailable.')
    quantiles = np.percentile(positive, [0, 1, 5, 50, 95, 99, 100])
    result['quantiles_m3'] = dict(zip(('min', 'p01', 'p05', 'median', 'p95', 'p99', 'max'), map(float, quantiles)))
    result['positive_volume_m3'] = float(positive.sum())
    result['small_cell_shares'] = []
    for fraction in (.01, .001, .0001):
        threshold = float(quantiles[3] * fraction)
        small = positive[positive < threshold]
        result['small_cell_shares'].append(dict(
            median_fraction=fraction, threshold_m3=threshold, count=int(small.size),
            cell_percent=float(100 * small.size / volumes.size),
            volume_percent=float(100 * small.sum() / positive.sum())))
    return result


def create_volume_report_data(case_path):
    """Always recompute; never substitute a rotor-only mesh or log extrema."""
    case = Path(case_path)
    report = case / 'report'
    report.mkdir(parents=True, exist_ok=True)
    image, csv = report / 'mesh_volume_distribution.png', report / 'mesh_cell_volumes.csv'
    for path in (image, csv):
        path.unlink(missing_ok=True)
    result = dict(status='unavailable', image=None,
                  scope='Assembled initial mesh: constant/polyMesh; all cells.')
    try:
        volumes = measure_cell_volumes(case / 'constant/polyMesh')
        result.update(volume_statistics(volumes))
        positive = np.sort(volumes[np.isfinite(volumes) & (volumes > 0)])
        lo, hi = positive[0], positive[-1]
        # A uniform mesh still needs a nonzero plotting interval.
        uniform = hi / lo < 1.000001
        if uniform:
            lo, hi = lo / 1.1, hi * 1.1
        # An odd bin count puts an effectively uniform population at bin centre.
        edges = np.geomspace(lo, hi, 42 if uniform else 41)
        counts, _ = np.histogram(positive, edges)
        weights, _ = np.histogram(positive, edges, weights=positive)
        fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
        try:
            axes[0].stairs(100 * counts / len(volumes), edges, fill=True, color='#4a8195', label='Share of all cells')
            axes[0].stairs(100 * weights / positive.sum(), edges, color='#d8822b', linewidth=1.5, label='Share of positive mesh volume')
            axes[0].set_ylabel('Share per bin [%]')
            axes[0].legend(fontsize=8)
            # Downsample only the plotted CDF, retaining endpoints and tail detail.
            indices = np.unique(np.r_[0, np.geomspace(1, len(positive), min(2000, len(positive))).astype(int)-1, len(positive)-1])
            axes[1].step(np.r_[lo, positive[indices], hi],
                         np.r_[0, 100 * (indices+1) / len(volumes), 100 * len(positive) / len(volumes)],
                         where='post', color='#4a8195')
            axes[1].set(xlabel='Cell volume [m³] (logarithmic scale)', ylabel='Cumulative cell share [%]', ylim=(0, 101))
            for ax in axes:
                ax.set_xscale('log')
                ax.set_xlim(lo, hi)
                ax.grid(alpha=.25)
            fig.tight_layout()
            fig.savefig(image, dpi=180)
        finally:
            plt.close(fig)
        np.savetxt(csv, volumes, delimiter=',', header='cell_volume_m3', comments='')
        result.update(status='complete', image=str(image), csv=str(csv))
    except (OSError, ValueError, UnicodeError) as exc:
        result['reason'] = str(exc).replace('First-cell measurement', 'Cell-volume measurement')
    (report / 'mesh_volume_distribution.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def append_volume_report(c, result):
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
    c.drawString(50, height-52, 'Mesh Volume Distribution')
    c.setFillColor(colors.black)
    style = ParagraphStyle('mesh-volume', fontName='Helvetica', fontSize=9, leading=12)
    y = height-78

    def paragraph(text):
        nonlocal y
        p = Paragraph(escape(text), style)
        _, h = p.wrap(width-100, height)
        p.drawOn(c, 50, y-h)
        y -= h+8

    paragraph(result['scope'])
    if result['status'] == 'unavailable':
        paragraph('Distribution unavailable: ' + result['reason'])
        paragraph('Requires reconstructed ASCII mesh points, faces, owner and neighbour (gzip supported). No distribution is inferred from checkMesh minimum/maximum values.')
        return
    paragraph(f"{result['cells']:,} cells | Nonpositive: {result['nonpositive_cells']:,} | Nonfinite: {result['nonfinite_cells']:,}. "
              'Invalid volumes are excluded from log plots and listed separately; cell percentages use all cells.')
    q = result['quantiles_m3']
    paragraph('Positive-cell volumes [m3]: ' + ' | '.join(f'{k}: {q[k]:.4g}' for k in ('min', 'median', 'max')))
    paragraph('Percentiles [m3]: ' + ' | '.join(f'{k}: {q[k]:.4g}' for k in ('p01', 'p05', 'p95', 'p99')))
    ph = (width-100) * 5.5/8
    c.drawImage(result['image'], 50, y-ph, width=width-100, height=ph)
    y -= ph+10
    rows = [['Below median fraction', 'Cutoff [m3]', 'Cells', 'Cell share', 'Volume share']]
    for row in result['small_cell_shares']:
        rows.append([f"{100*row['median_fraction']:g}% of median", f"{row['threshold_m3']:.3g}",
                     f"{row['count']:,}", f"{row['cell_percent']:.4g}%", f"{row['volume_percent']:.4g}%"])
    table = Table(rows, colWidths=[140, 90, 75, 90, width-495])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e5edf2')),
        ('FONTSIZE', (0, 0), (-1, -1), 8), ('TOPPADDING', (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7)]))
    _, th = table.wrap(width-100, height)
    table.drawOn(c, 50, y-th)
    y -= th+12
    paragraph('Small-cell cutoffs are diagnostic guides, not pass/fail criteria. Intentional refinement can produce small cells. A narrow peak indicates similar sizes; a long left tail indicates a small-volume population.')
    paragraph('Bins have equal logarithmic width. Cell share measures prevalence; volume share measures occupied space. Volumes use oriented face-centre triangle fans on the initial mesh.')
    c.setFont('Helvetica', 8)
    c.drawString(50, 32, 'Data: report/mesh_volume_distribution.json and mesh_cell_volumes.csv')
