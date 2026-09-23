"""Initial-mesh first-cell normal depths, area-weighted statistics and PDF page.

The full cell depth is wall-face to opposite-face centroid separation projected
onto the wall normal. It is neither wall-to-cell-centre distance nor total stack
thickness. Cells without one unambiguous opposite face are explicitly excluded.
"""
import gzip
import json
from pathlib import Path
import re

import numpy as np
from .plotting import pyplot as plt


def _read_list(path):
    path = Path(path)
    if path.is_file():
        raw = path.read_bytes()
    elif path.with_name(path.name + '.gz').is_file():
        raw = gzip.decompress(path.with_name(path.name + '.gz').read_bytes())
    else:
        raise FileNotFoundError(f'Missing mesh file: {path}')
    if re.search(rb'\bformat\s+binary\s*;', raw[:4096]):
        raise ValueError('First-cell measurement requires an ASCII mesh (gzip is supported).')
    text = raw.decode('utf-8')
    text = re.sub(r'/\*.*?\*/|//[^\n]*', '', text, flags=re.S)
    match = re.search(r'(?m)^\s*(\d+)\s*\(', text)
    if not match or text.rfind(')') < match.end():
        raise ValueError(f'Unrecognised OpenFOAM list: {path.name}')
    return int(match[1]), text[match.end():text.rfind(')')]


def measure_first_cells(mesh, patch='propeller'):
    """Return one row per wall face: centroid xyz, area, depth, validity."""
    mesh = Path(mesh)
    npoints, text = _read_list(mesh/'points')
    points = np.fromstring(text.replace('(', ' ').replace(')', ' '), sep=' ')
    if points.size != 3*npoints:
        raise ValueError('Point count does not match the mesh header.')
    points = points.reshape(-1, 3)
    nfaces, text = _read_list(mesh/'faces')
    sizes = np.array([int(v) for v in re.findall(r'(\d+)\s*\(', text)], dtype=np.int64)
    vertices = np.fromstring(re.sub(r'\d+\s*\(', ' ', text).replace(')', ' '), sep=' ', dtype=np.int64)
    if len(sizes) != nfaces or sizes.sum() != vertices.size or np.any(sizes < 3):
        raise ValueError('Unsupported or malformed face list.')
    if np.any(vertices < 0) or np.any(vertices >= npoints):
        raise ValueError('Face references a point outside the mesh.')
    offsets = np.concatenate(([0], np.cumsum(sizes)))
    nowner, text = _read_list(mesh/'owner')
    owner = np.fromstring(text.strip(), sep=' ', dtype=np.int64)
    nneighbour, text = _read_list(mesh/'neighbour')
    neighbour = np.fromstring(text.strip(), sep=' ', dtype=np.int64)
    if len(owner) != nfaces or nowner != nfaces or len(neighbour) != nneighbour or nneighbour > nfaces:
        raise ValueError('Face addressing count does not match the mesh header.')
    if np.any(owner < 0) or np.any(neighbour < 0) or not np.isfinite(points).all():
        raise ValueError('Invalid cell addressing or nonfinite mesh points.')
    _, boundary = _read_list(mesh/'boundary')
    match = re.search(r'(?<!\w)"?' + re.escape(patch) + r'"?\s*\{([^{}]*)\}', boundary)
    if not match:
        raise ValueError(f'Patch {patch!r} is absent from the initial mesh.')
    start_entry = re.search(r'\bstartFace\s+(\d+)', match[1])
    count_entry = re.search(r'\bnFaces\s+(\d+)', match[1])
    if not start_entry or not count_entry:
        raise ValueError('Wall patch is missing face-count or start-face addressing.')
    start, count = int(start_entry[1]), int(count_entry[1])
    if not count or start < nneighbour or start+count > nfaces:
        raise ValueError('Selected wall patch has no faces or invalid addressing.')
    cells = np.unique(owner[start:start+count])
    owner_faces = np.flatnonzero(np.isin(owner, cells))
    neighbour_faces = np.flatnonzero(np.isin(neighbour, cells))
    cell_faces = {int(c): [] for c in cells}
    for i in owner_faces:
        cell_faces[int(owner[i])].append(int(i))
    for i in neighbour_faces:
        cell_faces[int(neighbour[i])].append(int(i))

    def face(i):
        return vertices[offsets[i]:offsets[i+1]]

    def geometry(i):
        p = points[face(i)]
        ref = p.mean(axis=0)
        triangles = np.cross(p-ref, np.roll(p, -1, axis=0)-ref)/2
        weights = np.linalg.norm(triangles, axis=1)
        if not weights.sum():
            return ref, np.zeros(3)
        centre = np.average((p+np.roll(p, -1, axis=0)+ref)/3, weights=weights, axis=0)
        return centre, triangles.sum(axis=0)

    geometry_cache = {int(i): geometry(i) for i in np.union1d(owner_faces, neighbour_faces)}
    rows = []
    for i in range(start, start+count):
        centre, normal = geometry_cache[i]
        area = float(np.linalg.norm(normal))
        wall = set(face(i))
        opposite = [j for j in cell_faces[int(owner[i])] if not wall.intersection(face(j))]
        depth = float('nan')
        if len(opposite) == 1 and area > 0:
            depth = float(np.dot(geometry_cache[opposite[0]][0]-centre, -normal/area))*1000
        rows.append([*centre, area, depth])
    return np.asarray(rows)


def thickness_statistics(rows, target=None):
    area = rows[:, 3]
    height = rows[:, 4]
    valid = np.isfinite(height) & (height > 0) & np.isfinite(area) & (area > 0)
    total_area = area[np.isfinite(area) & (area > 0)].sum()
    if not np.any(valid) or total_area <= 0:
        raise ValueError('No positive first-cell depths with unambiguous opposite faces.')
    h, w = height[valid], area[valid]
    result = dict(min_mm=float(h.min()), mean_mm=float(np.average(h, weights=w)), max_mm=float(h.max()),
                  total_faces=len(rows), measured_faces=int(valid.sum()), excluded_faces=int((~valid).sum()),
                  measured_area_fraction=float(w.sum()/total_area), area_m2=float(w.sum()),
                  mean_weighting='wall-face area', target_thickness_mm=target)
    result['area_fraction_in_target'] = (float(w[(h>=target[0]) & (h<=target[1])].sum()/w.sum()) if target else None)
    return result, valid


def create_thickness_report_data(case_path):
    """Recompute for every report; never reuse stale measurements or plots."""
    case = Path(case_path)
    report = case/'report'
    report.mkdir(parents=True, exist_ok=True)
    image = report/'mesh_first_layer_thickness.png'
    csv = report/'mesh_first_layer_thickness.csv'
    for path in (image, csv):
        path.unlink(missing_ok=True)
    result = {'status': 'unavailable', 'image': None,
              'method': 'Full first-cell wall-normal depth from wall-face to opposite-face centroids; initial mesh.'}
    try:
        config_path = case/'Parameters/meshReport.json'
        settings = json.loads(config_path.read_text()) if config_path.is_file() else {}
        if not isinstance(settings, dict) or set(settings)-{'patch', 'target_thickness_mm'}:
            raise ValueError('meshReport.json accepts patch and target_thickness_mm only.')
        patch = settings.get('patch', 'propeller')
        if not isinstance(patch, str) or not re.fullmatch(r'[\w.-]+', patch):
            raise ValueError('Invalid thickness-report patch name.')
        target = settings.get('target_thickness_mm')
        if target is not None:
            if (not isinstance(target, list) or len(target) != 2 or
                    any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) for v in target) or
                    not 0 < target[0] < target[1]):
                raise ValueError('target_thickness_mm must be null or two increasing positive numbers.')
        # Prefer the actual assembled initial mesh; use the rotor before assembly.
        mesh = case/'constant/polyMesh'
        if not mesh.is_dir():
            mesh = case/'cfmesh/rotor/constant/polyMesh'
        rows = measure_first_cells(mesh, patch)
        stats, valid = thickness_statistics(rows, target)
        result.update(stats, status='complete' if stats['excluded_faces'] == 0 else 'partial',
                      mesh_path=str(mesh), patch=patch, scope=f'Entire {patch} patch (including hub, if present)')
        np.savetxt(csv, rows, delimiter=',', header='x_m,y_m,z_m,wall_face_area_m2,normal_depth_mm', comments='')
        fig, ax = plt.subplots(figsize=(8, 4.8))
        try:
            h, w = rows[valid, 4], rows[valid, 3]
            upper = max(float(h.max())*1.05, target[1]*1.25 if target else 0, .001)
            ax.hist(h, bins=np.linspace(0, upper, 61), weights=100*w/w.sum(), color='#4a8195', zorder=2)
            if target:
                ax.axvspan(*target, color='#70c7ae', alpha=.28, zorder=1)
                ax.set_title(f'Green: requested {target[0]:g}-{target[1]:g} mm range')
            else:
                ax.set_title('First-cell thickness distribution')
            ax.set(xlabel='First-cell wall-normal height [mm]', ylabel='Measured wall surface area [%]', xlim=(0, upper))
            fig.tight_layout()
            fig.savefig(image, dpi=180)
        finally:
            plt.close(fig)
        result['image'] = str(image)
        result['csv'] = str(csv)
    except (OSError, ValueError, UnicodeError) as exc:
        result.update(status='unavailable', reason=str(exc))
    (report/'mesh_first_layer_thickness.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


def append_thickness_report(c, result):
    """Always append a page, including an explicit reason when unavailable."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph
    from xml.sax.saxutils import escape
    width, height = A4
    c.showPage()
    c.setPageSize(A4)
    c.setFillColor(colors.HexColor('#142f43'))
    c.setFont('Helvetica-Bold', 18)
    c.drawString(50, height-52, 'First-Cell Thickness')
    c.setFillColor(colors.black)
    style = ParagraphStyle('first-cell', fontName='Helvetica', fontSize=10, leading=14)
    y = height-83

    def paragraph(text):
        nonlocal y
        p = Paragraph(escape(text), style)
        _, h = p.wrap(width-100, height)
        p.drawOn(c, 50, y-h)
        y -= h+10

    if result['status'] == 'unavailable':
        paragraph('Measurement unavailable: '+result['reason'])
        paragraph('No thickness values or histogram have been inferred from meshing settings.')
        return
    paragraph(result['scope']+'. Initial mesh; full first-cell height.')
    c.setFont('Helvetica-Bold', 12)
    for x, label, key in zip((50, 215, 380), ('Minimum', 'Mean (area-weighted)', 'Maximum'), ('min_mm', 'mean_mm', 'max_mm')):
        c.setFont('Helvetica', 10)
        c.drawString(x, y-10, label)
        c.setFont('Helvetica-Bold', 15)
        c.drawString(x, y-32, f'{result[key]:.4g} mm')
    y -= 55
    plot_height = (width-100)*4.8/8
    c.drawImage(result['image'], 50, y-plot_height, width=width-100, height=plot_height)
    y -= plot_height+18
    paragraph(f"Measured {result['measured_faces']:,} / {result['total_faces']:,} wall faces; "
              f"{100*result['measured_area_fraction']:.2f}% of patch area. Excluded faces: {result['excluded_faces']:,}.")
    if result['target_thickness_mm']:
        paragraph(f"Within target: {100*result['area_fraction_in_target']:.2f}% of measured area. "
                  'The green band is a report target, not a mesher constraint.')
    paragraph('Height is the wall-to-opposite-face centroid separation projected onto the wall normal. '
              'Unmeasurable or nonpositive depths are excluded; the histogram is normalized to measured area.')
    paragraph('This is not total layer-stack thickness or the wall-to-cell-centre distance used for y+. '
              'A flow solution is needed to assess y+.')
    c.setFont('Helvetica', 8)
    c.drawString(50, 32, 'Measurements and per-face data: report/mesh_first_layer_thickness.json / .csv')
