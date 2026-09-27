"""Case-local configuration overview; no OpenFOAM execution or template fallback."""
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def read_dictionary(path, case, sources, warnings, stack=()):
    """Read local includes and ordinary dictionary entries without executing code.

    Expressions/directives are preserved, not evaluated. This deliberately is
    not a replacement for foamDictionary's runtime-dependent expansion.
    """
    path, case = Path(path).resolve(), Path(case).resolve()
    if not path.is_relative_to(case):
        raise ValueError('Include outside the saved case: ' + str(path))
    if path in stack:
        raise ValueError('Cyclic include: ' + str(path.relative_to(case)))
    raw = path.read_bytes()
    sources[str(path.relative_to(case))] = hashlib.sha256(raw).hexdigest()
    text = raw.decode('utf-8')
    text = re.sub(r'"(?:\\.|[^"\\])*"|/\*.*?\*/|//[^\n]*',
                  lambda m: m[0] if m[0].startswith('"') else ' ', text, flags=re.S)

    def include(match):
        target = path.parent / match[2]
        if match[1] == 'includeIfPresent' and not target.exists():
            return ''
        # Expand text in-place so includes obey dictionary order and scope.
        try:
            entries = read_dictionary(target, case, sources, warnings, (*stack, path))
            return serialize(entries)
        except (OSError, ValueError, UnicodeError) as exc:
            warnings.append(str(exc))
            return f'__unresolvedInclude "{match[2]}";'

    text = re.sub(r'#(include|includeIfPresent)\s+"([^"\n]+)"\s*;?', include, text)
    def external_include(match):
        warnings.append(f'{path.relative_to(case)}: external #includeEtc {match[1]} not resolved')
        return '__unresolvedExternalInclude "' + match[1] + '";'
    text = re.sub(r'#includeEtc\s+"([^"\n]+)"\s*;?', external_include, text)
    if re.search(r'#(?:include\w*|code\w*|\{)', text):
        raise ValueError('Unsupported external include or code directive in ' + path.name)
    tokens = re.findall(r'"(?:\\.|[^"\\])*"|[{};]|[^\s{};]+', text)
    index = 0

    def block(nested=False):
        nonlocal index
        result = {}
        while index < len(tokens):
            key = tokens[index]
            index += 1
            if key == '}':
                if not nested:
                    raise ValueError('Unexpected closing brace in ' + path.name)
                return result
            if key == ';':
                continue
            values = []
            while index < len(tokens) and tokens[index] not in (';', '{', '}'):
                values.append(tokens[index])
                index += 1
            if index < len(tokens) and tokens[index] == '{':
                index += 1
                result[key.strip('"')] = block(True)
            else:
                key = key.strip('"')
                value = ' '.join(values)
                # Common template pattern: include deltaT, then deltaT $deltaT.
                previous = result.get(key)
                if isinstance(previous, str):
                    value = re.sub(r'\$' + re.escape(key) + r'\b', lambda _: previous, value)
                result[key] = value
                if index < len(tokens) and tokens[index] == ';':
                    index += 1
        if nested:
            raise ValueError('Unclosed dictionary in ' + path.name)
        return result

    return block()


def serialize(entries):
    return '\n'.join(f'"{k}" {{ {serialize(v)} }}' if isinstance(v, dict)
                     else f'"{k}" {v};' for k, v in entries.items())


def resolve_dictionary(root):
    """Resolve ordinary scalar and dictionary substitutions, flag the rest."""
    def lookup(name, scopes):
        for scope in reversed(scopes):
            if name in scope:
                return scope[name]
            for key, value in scope.items():
                try:
                    if not key.startswith('$') and re.fullmatch(key, name):
                        return value
                except re.error:
                    pass
        return None

    def expand(node, scopes, trail=()):
        result = {}
        for key, value in node.items():
            if key == 'FoamFile':
                continue
            if key.startswith('$'):
                name = key[1:]
                inherited = lookup(name, scopes)
                if isinstance(inherited, dict) and name not in trail:
                    result.update(expand(inherited, scopes, (*trail, name)))
                else:
                    result[key] = 'unresolved inheritance'
            elif isinstance(value, dict):
                result[key] = expand(value, (*scopes, value), trail)
            else:
                def scalar(text, visited=()):
                    def substitute(match):
                        name = match[1]
                        found = lookup(name, scopes)
                        if isinstance(found, str) and name not in visited:
                            return scalar(found, (*visited, name))
                        return match[0]
                    return re.sub(r'\$([A-Za-z_]\w*)', substitute, text)
                value = scalar(value)
                result[key] = value + (' [unresolved expression/reference]' if '$' in value or '#' in value else '')
        return result

    return expand(root, (root,))


def compact(value):
    if isinstance(value, dict):
        return '; '.join(f'{key}: {compact(item)}' for key, item in value.items()) or '(empty)'
    if isinstance(value, list):
        return '(' + ', '.join(map(str, value)) + ')'
    return str(value)


def create_settings_report_data(case_path, rpm, mode, turbulence_model, mesh_only=False):
    case = Path(case_path)
    sources, warnings, documents, sections = {}, [], {}, []

    def read(relative):
        path = case / relative
        if not path.is_file():
            return {}
        try:
            if path.suffix == '.json':
                raw = path.read_bytes()
                sources[relative] = hashlib.sha256(raw).hexdigest()
                data = json.loads(raw)
                if not isinstance(data, dict):
                    raise ValueError(relative + ' is not a JSON object')
            else:
                data = resolve_dictionary(read_dictionary(path, case, sources, warnings))
            documents[relative] = data
            return data
        except (OSError, ValueError, UnicodeError) as exc:
            warnings.append(f'{relative}: {exc}')
            return {}

    def section(title, source, rows):
        sections.append(dict(title=title, source=source, rows=[
            [label, compact(value) if value not in (None, {}, '') else 'Not recorded'] for label, value in rows]))

    section('Run context', 'Report arguments', [
        ('Mode / speed / model', f'{mode}; {rpm} RPM; {turbulence_model}'),
        ('Report scope', 'Mesh only; solver settings configured, not evidence of a solver run' if mesh_only else 'Simulation')])
    geometry = read('cfmesh/geometry.json')
    common = read('Parameters/cfmeshCommon.cpp')
    section('Mesh geometry and resolution', 'cfmesh/geometry.json; Parameters/cfmeshCommon.cpp', [
        ('Domain / rotor [m]', {k: geometry[k] for k in ('box_min', 'box_max', 'rotor_radius_m', 'rotor_half_length_m') if k in geometry}),
        ('Resolution', geometry.get('resolution') or {k: v for k, v in common.items() if 'Level' in k or 'CellSize' in k})])
    for role in ('rotor', 'stator'):
        source = f'cfmesh/{role}/system/meshDict'
        mesh = read(source)
        section(f'{role.title()} meshing', source, [
            ('Base / workflow', {k: mesh[k] for k in ('maxCellSize', 'minCellSize', 'workflowControls') if k in mesh}),
            ('Surface refinement', mesh.get('localRefinement')),
            ('Layers', mesh.get('boundaryLayers')),
            ('Volume / edge refinement', {k: mesh[k] for k in ('objectRefinements', 'edgeMeshRefinement') if k in mesh})])
    pipeline = read('cfmesh/pipeline-controls.json')
    section('Mesh improvement and acceptance', 'cfmesh/pipeline-controls.json', [('Controls', pipeline)])
    control = read('system/controlDict')
    functions = control.get('functions', {})
    section('Time stepping and output', 'system/controlDict (local includes resolved where supported)', [
        ('Time controls', {k: control[k] for k in ('application', 'startFrom', 'startTime', 'stopAt', 'endTime', 'deltaT', 'adjustTimeStep', 'maxCo', 'maxDeltaT') if k in control}),
        ('Output', {k: control[k] for k in ('writeControl', 'writeInterval', 'purgeWrite', 'writeFormat', 'writePrecision') if k in control}),
        ('Startup time-step cap', functions.get('startupDeltaTCap') if isinstance(functions, dict) else None)])
    solution = read('system/fvSolution')
    section('Solver coupling and convergence', 'system/fvSolution', [
        (key, solution.get(key)) for key in ('PIMPLE', 'SIMPLE', 'solvers', 'relaxationFactors')
        if key in solution or key == 'solvers'])
    schemes = read('system/fvSchemes')
    section('Numerical schemes', 'system/fvSchemes', [(k, schemes.get(k)) for k in (
        'ddtSchemes', 'gradSchemes', 'divSchemes', 'laplacianSchemes', 'interpolationSchemes', 'snGradSchemes')])
    physical = []
    for relative in ('constant/transportProperties', 'constant/physicalProperties',
                     'constant/turbulenceProperties', 'constant/momentumTransport',
                     'constant/MRFProperties', 'constant/dynamicMeshDict', 'system/decomposeParDict'):
        data = read(relative)
        if data:
            physical.append((relative, data))
    section('Fluid, turbulence, rotation and parallelism', 'Case constant/ and system/ dictionaries',
            physical or [('Settings', None)])
    boundaries = []
    for field in ('U', 'p', 'k', 'omega', 'epsilon', 'nut'):
        relative = '0/' + field
        path = case / relative
        # Large nonuniform initial fields are data, not a compact settings input.
        if path.is_file() and path.stat().st_size > 2_000_000:
            warnings.append(relative + ': initial field exceeds 2 MB; boundary overview unavailable')
            continue
        data = read(relative)
        patches = data.get('boundaryField', {})
        field_boundaries = {}
        if isinstance(patches, dict):
            for patch, values in patches.items():
                if isinstance(values, dict):
                    selected = {k: v for k, v in values.items()
                                if k in ('type', 'value', 'inletValue', 'freestreamValue')}
                    selected = {k: ('nonuniform field (see case file)' if isinstance(v, str) and v.startswith('nonuniform') else v)
                                for k, v in selected.items()}
                    field_boundaries[patch] = selected
        if field_boundaries:
            boundaries.append((field, field_boundaries))
    section('Initial boundary conditions', 'Saved 0/ fields', boundaries or [('Boundary conditions', None)])
    result = dict(captured_at=datetime.now(timezone.utc).isoformat(), sections=sections,
                  sources=sources, documents=documents, warnings=warnings,
                  scope='Saved case configuration at report time, not a historical record of runtime edits. '
                        'Recorded mesher dictionaries and snapshots are preferred. Missing entries are not inferred defaults.')
    report = case / 'report'
    report.mkdir(parents=True, exist_ok=True)
    (report / 'mesh_solver_settings.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


def append_settings_report(c, result):
    """Dedicated settings chapter with readable continuation pages, no truncation."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph
    from xml.sax.saxutils import escape

    width, height = A4
    style = ParagraphStyle('settings', fontName='Helvetica', fontSize=8, leading=11)
    y = 0
    page_number = 0

    def page():
        nonlocal y, page_number
        c.showPage()
        c.setPageSize(A4)
        page_number += 1
        c.setFillColor(colors.HexColor('#142f43'))
        c.setFont('Helvetica-Bold', 18)
        c.drawString(50, height-52, 'Mesh and Solver Settings' + (' (continued)' if page_number > 1 else ''))
        c.setFillColor(colors.black)
        c.setFont('Helvetica', 7)
        c.drawString(50, 30, 'Full snapshot and source SHA-256 hashes: report/mesh_solver_settings.json')
        y = height-76

    def paragraph(text, bold=False):
        nonlocal y
        p = Paragraph(('<b>' + escape(text) + '</b>') if bold else escape(text), style)
        while p:
            _, h = p.wrap(width-100, height)
            if h <= y-50:
                p.drawOn(c, 50, y-h)
                y -= h+6
                break
            parts = p.split(width-100, y-50)
            if parts:
                _, h = parts[0].wrap(width-100, height)
                parts[0].drawOn(c, 50, y-h)
                p = parts[1] if len(parts) > 1 else None
            page()

    page()
    paragraph(result['scope'])
    for section in result['sections']:
        if y < 120:
            page()
        paragraph(section['title'], True)
        paragraph('Source: ' + section['source'])
        for label, value in section['rows']:
            paragraph(f'{label}: {value}')
    if result['warnings']:
        paragraph('Unresolved sources', True)
        for warning in result['warnings']:
            paragraph(warning)
