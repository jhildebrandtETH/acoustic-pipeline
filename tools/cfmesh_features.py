"""Stage explicit OBJ feature curves for native cfMesh edge refinement."""

import hashlib
import math
from pathlib import Path
import re
import warnings

from tools.cfmesh_parameters import level_cell_size
from tools.cfmesh_pipeline import query, write


def scaled_obj(path, scale):
    """Read OBJ vertices/line elements and emit scaled, two-point segments.

    Preserve explicit curves only: face triangulation edges are not features.
    Negative OBJ indices refer to the vertices preceding the line element.
    """
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Feature scale must be positive and finite")
    vertices, edges = [], []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        fields = line.partition("#")[0].split()
        if not fields:
            continue
        try:
            if fields[0] == "v":
                if len(fields) != 4:
                    raise ValueError("expected a three-coordinate vertex")
                point = tuple(float(value) * scale for value in fields[1:])
                if not all(math.isfinite(value) for value in point):
                    raise ValueError("vertex coordinates must be finite")
                vertices.append(point)
            elif fields[0] == "l":
                if len(fields) < 3:
                    raise ValueError("line requires at least two vertex indices")
                indices = []
                for token in fields[1:]:
                    index = int(token.split("/", 1)[0])
                    if index == 0:
                        raise ValueError("OBJ indices cannot be zero")
                    indices.append(index - 1 if index > 0 else len(vertices) + index)
                    if indices[-1] < 0:
                        raise ValueError("vertex index out of range")
                edges.extend(zip(indices, indices[1:]))
        except ValueError as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
    if not edges:
        raise ValueError(f"{path}: no explicit OBJ line elements ('l'); export feature curves as lines")
    for first, second in edges:
        if max(first, second) >= len(vertices):
            raise ValueError(f"{path}: vertex index out of range")
        if vertices[first] == vertices[second]:
            raise ValueError(f"{path}: zero-length feature segment")
    text = "# Feature curves scaled to metres for cfMesh.\n"
    text += "".join("v " + " ".join(format(v, ".17g") for v in point) + "\n" for point in vertices)
    text += "".join(f"l {a + 1} {b + 1}\n" for a, b in edges)
    return text, len(vertices), len(edges)


def prepare_features(case, source, scale, resolution):
    """Match FEATURES/<STL stem>_<group>.obj and snapshot rotor-only inputs."""
    case, source = Path(case), Path(source)
    dictionary = case / "Parameters/cfmeshFeatureDict"
    generated = case / "Parameters/cfmeshFeatures.generated"
    entries, report = [], {}
    # Old case snapshots and orders without feature files retain their behaviour.
    if not dictionary.is_file():
        write(generated, "// No feature controls in this case snapshot.\n")
        return report
    enabled = query(dictionary, "enabled").lower()
    if enabled not in {"true", "yes", "on", "1", "false", "no", "off", "0"}:
        raise ValueError("cfmeshFeatureDict enabled must be a boolean")
    if enabled in {"false", "no", "off", "0"}:
        write(generated, "// Explicit feature refinements disabled.\n")
        return report
    groups = query(dictionary, "groups").strip("() \n\r\t").split()
    if len(groups) != len(set(groups)) or any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", group) for group in groups):
        raise ValueError("Feature groups must be unique names containing letters, digits or underscores")
    for group in groups:
        level = int(query(dictionary, group + "/level"))
        size = level_cell_size(resolution["base_cell_size_m"], level)
        thickness = float(query(dictionary, group + "/refinementThickness"))
        if not math.isfinite(thickness) or thickness < 0:
            raise ValueError(f"Feature {group} refinementThickness must be finite and nonnegative")
        feature = source.parent.parent / "FEATURES" / f"{source.stem}_{group}.obj"
        if not feature.is_file():
            warnings.warn(f"Skipping missing cfMesh feature file: {feature}", stacklevel=2)
            report[group] = dict(source=str(feature), status="missing")
            continue
        text, points, edges = scaled_obj(feature, scale)
        relative = f"constant/triSurface/features/{group}.obj"
        write(case / "cfmesh/rotor" / relative, text)
        # Keep an unmodified input snapshot for reproducibility and inspection.
        snapshot = case / "FEATURES" / feature.name
        write(snapshot, feature.read_text(encoding="utf-8-sig"))
        entries.append(f'''{group}
{{
    edgeFile "{relative}";
    additionalRefinementLevels {level};
    refinementThickness {thickness:.17g};
}}
''')
        report[group] = dict(source=str(feature), status="active", level=level,
                             cell_size_m=size, refinement_thickness_m=thickness,
                             vertices=points, edges=edges, edge_file=relative,
                             source_sha256=hashlib.sha256(feature.read_bytes()).hexdigest())
    write(generated, "// Generated explicit edge refinements; edit cfmeshFeatureDict.\n" + "".join(entries))
    return report
