"""Read editable pipeline controls, retaining compatibility with old case snapshots."""

import math
from pathlib import Path


DEFAULTS = {
    "improveMeshQuality": {
        "enabled": True, "nLoops": 2, "nIterations": 20, "nSurfaceIterations": 0,
    },
    "interfaceProjection": {
        "enabled": True, "movementLimitFactor": 1.5, "absoluteTolerance": 1e-10,
    },
    "acceptance": {
        "maxRelativeVolumeError": 0.005,
        "minimumFaceCoverage": 0.95, "minimumAverageCoverage": 0.999,
    },
}


def read_controls(parameters):
    from tools.cfmesh_pipeline import query_entries

    path = Path(parameters) / "cfmeshPipelineDict"
    controls = {section: dict(values) for section, values in DEFAULTS.items()}
    # Older resumable cases predate this dictionary. An existing file must be complete.
    if not path.is_file():
        return controls
    raw_values = query_entries(path, (
        f"{section}/{key}" for section, values in controls.items() for key in values
    ))
    for section, values in controls.items():
        for key, default in values.items():
            entry = f"{section}/{key}"
            raw = raw_values[entry].strip()
            try:
                if isinstance(default, bool):
                    if raw.lower() not in {"true", "yes", "on", "1", "false", "no", "off", "0"}:
                        raise ValueError("expected a boolean")
                    value = raw.lower() in {"true", "yes", "on", "1"}
                elif isinstance(default, int):
                    value = int(raw)
                    if value < 0:
                        raise ValueError("expected a nonnegative integer")
                else:
                    value = float(raw)
                    if not math.isfinite(value) or value < 0:
                        raise ValueError("expected a finite nonnegative number")
                    if section == "acceptance" and value > 1:
                        raise ValueError("expected a fraction between 0 and 1")
                values[key] = value
            except ValueError as exc:
                raise ValueError(f"{path}: invalid {entry}: {raw!r} ({exc})") from exc
    return controls


def improvement_command(controls):
    settings = controls["improveMeshQuality"]
    if not settings["enabled"]:
        return None
    command = ["improveMeshQuality"]
    for key in ("nLoops", "nIterations", "nSurfaceIterations"):
        command.extend(["-" + key, str(settings[key])])
    return command
