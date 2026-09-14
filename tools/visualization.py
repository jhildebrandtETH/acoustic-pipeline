"""Visualization helpers for the acoustic pipeline."""

import os
import gzip
import shutil
import subprocess
import numpy as np
import re
import traceback
from pathlib import Path
import json
import math
from datetime import datetime

def visualization_settings(case_path, rpm, acoustic_surface, overrides=None):
    """Resolve reproducible case-local settings, with explicit scientific units."""
    import math

    settings = {
        "enabled": True, "required": False, "executable": None,
        "image_resolution": [3000, 1800], "surface_phases": 12,
        "statistics_revolutions": 5.0, "volume_phases": 4,
        "wake_stations_D": [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
        "q_over_omega2": [0.1, 0.5, 1.0],
        "observer_m": [1.0, 0.0, 0.0], "diameter_m": None,
        "timeout_seconds": 21600, "threads": 2,
        "color_ranges": {},
        "log_fields": ["vorticity_magnitude", "wallShearStress", "dpdt_rms"],
        "report_max_views": 32,
    }
    settings_file = Path(case_path) / "visualization.json"
    supplied = json.loads(settings_file.read_text(encoding="utf-8")) if settings_file.is_file() else {}
    supplied.update(overrides or {})
    unknown = set(supplied) - set(settings)
    if unknown:
        raise ValueError(f"Unknown visualization settings: {sorted(unknown)}")
    settings.update(supplied)
    for key in ("enabled", "required"):
        if not isinstance(settings[key], bool):
            raise ValueError(f"{key} must be a boolean")
    for key in ("surface_phases", "volume_phases", "threads", "report_max_views"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("statistics_revolutions", "timeout_seconds"):
        if not math.isfinite(float(settings[key])) or float(settings[key]) <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if (len(settings["image_resolution"]) != 2 or
            any(type(v) is not int or v < 600 for v in settings["image_resolution"])):
        raise ValueError("image_resolution must contain two integers >= 600")
    if len(settings["observer_m"]) != 3 or not all(math.isfinite(float(v)) for v in settings["observer_m"]):
        raise ValueError("observer_m must contain three finite coordinates")
    for key in ("wake_stations_D", "q_over_omega2"):
        if not settings[key] or not all(math.isfinite(float(v)) for v in settings[key]):
            raise ValueError(f"{key} must contain finite values")
    if any(float(v) <= 0 for v in settings["q_over_omega2"]):
        raise ValueError("q_over_omega2 thresholds must be positive")
    range_keys = {"surface_p", "surface_speed", "surface_p_fluctuation", "p_mean", "p_rms", "dpdt_rms",
                  "volume_speed", "volume_axial_velocity", "volume_p", "volume_vorticity_magnitude",
                  "volume_k", "volume_Co", "volume_swirl_velocity", "p", "yPlus", "wallShearStress"}
    if not isinstance(settings["color_ranges"], dict) or set(settings["color_ranges"]) - range_keys:
        raise ValueError("color_ranges must map documented field keys to [minimum, maximum]")
    for key, limits in settings["color_ranges"].items():
        if len(limits) != 2 or not all(math.isfinite(float(v)) for v in limits) or limits[0] >= limits[1]:
            raise ValueError(f"Invalid color range for {key}")
    if not isinstance(settings["log_fields"], list) or any(
            key not in {"vorticity_magnitude", "wallShearStress", "dpdt_rms", "p_rms"}
            for key in settings["log_fields"]):
        raise ValueError("log_fields accepts nonnegative magnitude fields only")
    rpm = float(rpm)
    if not math.isfinite(rpm) or rpm <= 0:
        raise ValueError("Visualization RPM must be finite and positive")
    if acoustic_surface not in {None, "impermeable", "permeable"}:
        raise ValueError("Unknown acoustic surface type")
    if settings["diameter_m"] is None:
        # Same geometry naming convention as preprocessing.py.
        match = re.match(r"(\d+(?:\.\d+)?)x", Path(case_path).name)
        if match:
            settings["diameter_m"] = float(match[1]) * 0.0254
    if settings["diameter_m"] is not None:
        settings["diameter_m"] = float(settings["diameter_m"])
        if not math.isfinite(settings["diameter_m"]) or settings["diameter_m"] <= 0:
            raise ValueError("diameter_m must be finite and positive")
    settings.update(case_path=str(Path(case_path).resolve()), rpm=rpm, acoustic_surface=acoustic_surface)
    return settings


def find_paraview_executable(configured=None):
    """Use a native host ParaView Python; never import it into the pipeline env."""
    explicit = configured or os.environ.get("PARAVIEW_EXECUTABLE")
    if explicit:
        resolved = shutil.which(str(explicit))
        if resolved:
            return resolved
        raise FileNotFoundError(f"ParaView executable not found: {explicit}")
    for name in ("pvpython", "pvbatch"):
        candidate = shutil.which(name)
        if candidate:
            return candidate
    if os.name == "nt":
        for directory in sorted(Path(os.environ.get("ProgramFiles", "C:/Program Files")).glob("ParaView*"), reverse=True):
            candidate = directory / "bin" / "pvpython.exe"
            if candidate.is_file():
                return str(candidate)
    raise FileNotFoundError("Install a headless-capable ParaView on the simulation host and set PARAVIEW_EXECUTABLE to pvpython or pvbatch.")


def run_visualization_job(case_path, rpm, acoustic_surface, config=None, status_callback=None):
    """Serialize memory-heavy rendering, preserve diagnostics, and publish a manifest."""
    from tools.common import VISUALIZATION_LOCK
    from tools.common import _atomic_write_json
    from tools.common import emit_status
    import inspect
    import uuid

    settings = visualization_settings(case_path, rpm, acoustic_surface, config)
    root = Path(case_path).resolve() / "report" / "visuals"
    root.mkdir(parents=True, exist_ok=True)
    # Unique run folders ensure an unsuccessful rerun cannot reuse stale images.
    run_dir = root / (datetime.now().strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8])
    run_dir.mkdir()
    manifest_path = root / "manifest.json"
    manifest = {
        "schema_version": 1, "status": "running", "views": [], "warnings": [],
        "settings": settings, "run_directory": str(run_dir),
        "created_at": datetime.now().astimezone().isoformat(),
    }
    if not settings["enabled"]:
        manifest["status"] = "disabled"
        _atomic_write_json(manifest_path, manifest)
        return manifest
    with VISUALIZATION_LOCK:
        _atomic_write_json(manifest_path, manifest)
        try:
            executable = find_paraview_executable(settings["executable"])
            settings_path = run_dir / "settings.json"
            _atomic_write_json(settings_path, settings)
            # Only this named helper family enters ParaView's Python runtime.
            # No pipeline imports (pandas, torch, Docker, etc.) are required there.
            script = "import json, math, traceback, sys\nfrom pathlib import Path\nimport numpy as np\nfrom paraview import simple as pvs, servermanager\n\n"
            script += "\n\n".join(inspect.getsource(value) for name, value in sorted(globals().items())
                                     if name.startswith("_pvvis_") and inspect.isfunction(value))
            script += "\n\nif __name__ == '__main__':\n    _pvvis_main(sys.argv[1])\n"
            script_path = run_dir / "render_visuals.py"
            script_path.write_text(script, encoding="utf-8")
            env = os.environ.copy()
            env["VTK_SMP_MAX_THREADS"] = str(settings["threads"])
            env["OMP_NUM_THREADS"] = str(settings["threads"])
            emit_status(status_callback, stage="visualization", detail="rendering acoustic and flow diagnostics on server", progress=10.0)
            with (run_dir / "paraview.log").open("w", encoding="utf-8") as log:
                process = subprocess.run(
                    [executable, "--disable-registry", "--force-offscreen-rendering", str(script_path), str(settings_path)],
                    stdout=log, stderr=subprocess.STDOUT, env=env, cwd=run_dir,
                    timeout=float(settings["timeout_seconds"]), check=False,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
            result_path = run_dir / "result.json"
            if result_path.is_file():
                manifest.update(json.loads(result_path.read_text(encoding="utf-8")))
            reader_errors = [line.strip() for line in (run_dir / "paraview.log").read_text(
                encoding="utf-8", errors="replace").splitlines() if " ERR|" in line]
            if reader_errors:
                manifest["warnings"].append(
                    f"ParaView reported {len(reader_errors)} reader/rendering errors; inspect paraview.log before interpreting the affected fields. "
                    + reader_errors[0][-240:])
                if manifest["status"] == "complete":
                    manifest["status"] = "partial"
            if process.returncode:
                raise RuntimeError(f"ParaView exited with code {process.returncode}; see {run_dir / 'paraview.log'}")
            if not result_path.is_file():
                raise RuntimeError("ParaView returned no result manifest")
            if not manifest["views"]:
                raise RuntimeError("No scientific views were generated; see the recorded warnings")
            validate_visualization_images(root, manifest["views"], settings["image_resolution"])
        except Exception as exc:
            # Recover completed views even after a timeout/crash, but never call
            # such a run complete. The PDF includes all coverage failures.
            result_path = run_dir / "result.json"
            if result_path.is_file():
                try:
                    manifest.update(json.loads(result_path.read_text(encoding="utf-8")))
                except (OSError, ValueError):
                    pass
            manifest["status"] = "failed"
            manifest["warnings"].append(str(exc))
        _atomic_write_json(manifest_path, manifest)
    if settings["required"] and manifest["status"] != "complete":
        raise RuntimeError(f"Required visual atlas is {manifest['status']}; inspect {manifest_path}")
    return manifest


def validate_visualization_images(root, views, resolution):
    """Require every advertised PNG to exist at the requested native resolution."""
    import struct

    root = Path(root).resolve()
    for view in views:
        path = (root / view["image"]).resolve()
        path.relative_to(root)
        with path.open("rb") as handle:
            header = handle.read(24)
        if header[:8] != b"\x89PNG\r\n\x1a\n" or len(header) != 24:
            raise ValueError(f"Invalid rendered PNG: {path}")
        if list(struct.unpack(">II", header[16:24])) != resolution:
            raise ValueError(f"Unexpected rendered image dimensions: {path}")


def select_report_views(views, maximum=32):
    """Balance figure families; keep all originals in the visual archive."""
    if len(views) <= maximum:
        return list(views)
    groups = {}
    # Group related quantities, not hundreds of near-identical phases/cameras.
    for item in views:
        title = item["title"]
        family = title.split(" (")[0]
        if title.startswith("Vortex structures"):
            family = "Vortex structures"
        if title.startswith("Permeable FW-H surface - pressure"):
            family = "Surface pressure"
        groups.setdefault(family, []).append(item)
    selected = []
    # Start with the latest view of each quantity, then add complementary views.
    pools = [sorted(items, key=lambda item: item.get("time_s", 0), reverse=True)
             for items in groups.values()]
    while pools and len(selected) < maximum:
        for pool in list(pools):
            selected.append(pool.pop(0))
            if not pool:
                pools.remove(pool)
            if len(selected) == maximum:
                break
    identifiers = {id(item) for item in selected}
    return [item for item in views if id(item) in identifiers]


def append_visualization_report(c, case_path):
    """Append a coverage page and one large, annotated landscape page per view."""
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph
    from xml.sax.saxutils import escape

    root = Path(case_path).resolve() / "report" / "visuals"
    path = root / "manifest.json"
    if not path.is_file():
        return {"status": "not_run", "views": 0}
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        manifest = {"status": "failed", "views": [], "warnings": [f"Unreadable visual manifest: {exc}"]}
    w, h = landscape(A4)
    style = ParagraphStyle("atlas", fontName="Helvetica", fontSize=9, leading=12, textColor="#263746")

    def paragraph(text, y, size=9):
        style.fontSize, style.leading = size, size + 3
        block = Paragraph(escape(str(text)), style)
        _, height = block.wrap(w - 84, h)
        block.drawOn(c, 42, y - height)
        return y - height - 9

    def new_page(title):
        c.showPage()
        c.setPageSize((w, h))
        c.setFillColorRGB(0.08, 0.16, 0.23)
        title_size = 16
        while c.stringWidth(title, "Helvetica-Bold", title_size) > w - 84 and title_size > 10:
            title_size -= 1
        c.setFont("Helvetica-Bold", title_size)
        c.drawString(42, h - 36, title)
        c.setStrokeColorRGB(0.65, 0.72, 0.77)
        c.line(42, h - 47, w - 42, h - 47)
        c.setFont("Helvetica", 8)
        c.drawRightString(w - 42, 19, f"Visual atlas | Page {c.getPageNumber()}")

    new_page("Scientific Visual Atlas - Coverage and Interpretation")
    y = paragraph(f"Status: {manifest['status']} | Generated views: {len(manifest['views'])} | Report selection: up to {manifest.get('settings', {}).get('report_max_views', 32)} representative views", h - 63, 11)
    for note in [
        "These figures show saved CFD fields and acoustic-surface diagnostics. Hydrodynamic pressure, its fluctuations and vortex structures are not radiated sound or SPL. Use the FW-H observer spectrum for the acoustic prediction. Steady loading on rotating panels can still radiate tonal noise; low rotating-frame RMS does not imply low sound.",
        "Rotation axis: +y through the origin; lengths in metres. Rotor phase is omega*t modulo 360 degrees, relative to solver t=0. Signed y/D stations are shown on both sides of the rotor; identify the downstream side using axial velocity.",
        "Surface statistics use time-weighted samples over the recorded window. Blade statistics follow verified rotating panels; permeable-surface statistics use verified stationary panels. They are window statistics, not a claim of statistical convergence. No volume averaging across a moving mesh is performed.",
        "Slice color ranges cover the displayed region and are fixed across the selected phases at each station. Different stations may use different ranges; compare their legends. Full displayed-region extrema remain in metadata. Labelled logarithmic magnitude scales clamp zero to the lowest displayed color without changing the underlying data. Cross-case comparison requires matching color_ranges in visualization.json. Derivative maps are limited by saved sample cadence.",
        "The PDF contains a representative selection; all generated views remain in the image archive. The manifest, rendering script, settings, extracted surface statistics and original-resolution PNGs are retained under report/visuals. No CFD fields or postprocessing data are deleted. A visualization recipe alone cannot recreate views after source data are removed.",
    ]:
        y = paragraph(note, y)
    for warning in manifest.get("warnings", []):
        # Paginate long warning lists rather than clipping missing-view evidence.
        block = Paragraph(escape("Coverage note: " + str(warning)), style)
        _, needed = block.wrap(w - 84, h)
        if y - needed < 45:
            new_page("Scientific Visual Atlas - Coverage Notes")
            y = h - 64
        y = paragraph("Coverage note: " + str(warning), y)

    embedded = 0
    selected_views = select_report_views(manifest["views"], manifest.get("settings", {}).get("report_max_views", 32))
    for item in selected_views:
        new_page(item["title"])
        image_path = root / item["image"]
        # Manifest paths must stay in this case's visual archive.
        try:
            image_path.resolve().relative_to(root)
            c.drawImage(str(image_path), 30, 126, width=w - 60, height=h - 183,
                        preserveAspectRatio=True, anchor="c", mask="auto")
            embedded += 1
        except Exception as exc:
            paragraph(f"Image unavailable: {exc}", h - 85)
        y = paragraph(item["caption"], 117)
        metadata = [f"View: {item.get('camera', 'n/a')}"]
        if item.get("camera") == "blade section" and "parallel_scale_m" in item:
            metadata.append(f"Full viewport height: {2 * item['parallel_scale_m']:.5g} m")
        if "time_s" in item:
            metadata += [f"Time: {item['time_s']:.8g} s", f"Rotor phase: {item['phase_deg']:.3f} deg"]
        if "field" in item:
            metadata += [item["field"], "cell values" if item["association"] == "CELLS" else "point values",
                         "Color scale: " + " to ".join(f"{v:.8g}" for v in item["color_range"]),
                         "Scale: " + item.get("color_scale", "linear"),
                         "Data extrema: " + " to ".join(f"{v:.8g}" for v in item["data_range"])]
        paragraph(" | ".join(metadata), y, 8)
    return {"status": manifest["status"], "views": embedded, "manifest": str(path)}


def _pvvis_save(result, run_dir):
    path = run_dir / "result.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _pvvis_times(directory, filename=None):
    entries = []
    if directory.is_dir():
        for path in directory.iterdir():
            try:
                value = float(path.name)
            except ValueError:
                continue
            if path.is_dir() and math.isfinite(value) and (filename is None or (path / filename).is_file()):
                entries.append((value, str(path / filename) if filename else str(path)))
    entries.sort()
    if any(a[0] == b[0] for a, b in zip(entries, entries[1:])):
        raise ValueError(f"Duplicate physical times in {directory}")
    return entries


def _pvvis_select(entries, rpm, count):
    if not entries:
        return []
    if count == 1:
        return entries[-1:]
    end = entries[-1][0]
    start = max(entries[0][0], end - 60.0 / rpm)
    entries = [entry for entry in entries if entry[0] >= start]
    targets = np.linspace(start, end, count, endpoint=True)
    return [entries[i] for i in sorted({min(range(len(entries)), key=lambda i: abs(entries[i][0] - t)) for t in targets})]


def _pvvis_read_surface(path):
    from vtkmodules.vtkIOLegacy import vtkPolyDataReader

    reader = vtkPolyDataReader()
    reader.SetFileName(str(path))
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.ReadAllFieldsOn()
    reader.Update()
    data = reader.GetOutput()
    if not data or data.GetNumberOfCells() == 0:
        raise ValueError(f"Empty or unreadable VTK surface: {path}")
    return data


def _pvvis_array(data, name):
    from vtkmodules.util.numpy_support import vtk_to_numpy

    for association, attributes in (("CELLS", data.GetCellData()), ("POINTS", data.GetPointData())):
        array = attributes.GetArray(name)
        if array is not None:
            values = vtk_to_numpy(array)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Non-finite {name} on surface")
            return association, values
    raise ValueError(f"Surface field {name} is unavailable")


def _pvvis_pressure_units(case, field="p"):
    import gzip
    import re

    candidates = [Path(p) for _, p in reversed(_pvvis_times(case))]
    for directory in candidates:
        for suffix in (field, field + ".gz"):
            path = directory / suffix
            if path.is_file():
                opener = gzip.open if suffix.endswith(".gz") else open
                with opener(path, "rb") as handle:
                    header = handle.read(65536).decode("ascii", errors="ignore")
                match = re.search(r"dimensions\s*\[([^]]+)\]", header)
                if match:
                    dimensions = [float(x) for x in match[1].split()]
                    if dimensions == [0, 2, -2, 0, 0, 0, 0]:
                        return "m^2/s^2", "Kinematic pressure"
                    if dimensions == [1, -1, -2, 0, 0, 0, 0]:
                        return "Pa", "Pressure"
                    return "unknown dimensions", "Saved p"
    return "units unverified", "Saved p"


def _pvvis_limits(values, key, settings, symmetric=False):
    if key in settings["color_ranges"]:
        return list(settings["color_ranges"][key])
    lo, hi = float(values[0]), float(values[1])
    if not all(math.isfinite(v) for v in (lo, hi)) or hi < lo:
        raise ValueError(f"Invalid range for {key}")
    if symmetric:
        hi = max(abs(lo), abs(hi), 1e-12)
        lo = -hi
    if hi == lo:
        delta = max(abs(lo) * 1e-6, 1e-12)
        lo, hi = lo - delta, hi + delta
    return [lo, hi]


def _pvvis_render(source, result, settings, run_dir, view, title, caption,
                  camera="oblique", field=None, limits=None, center=None, span=None,
                  time_s=None, edges=False, overlays=(), diverging=False, opacity=1.0):
    source.UpdatePipeline(time_s) if time_s is not None else source.UpdatePipeline()
    info = source.GetDataInformation()
    if info.GetNumberOfCells() == 0:
        raise ValueError(f"Empty geometry for {title}")
    pvs.HideAll(view)
    for representation in view.Representations:
        if "ScalarBar" in representation.GetXMLName():
            representation.Visibility = 0
    display = pvs.Show(source, view)
    display.Representation = "Surface With Edges" if edges else "Surface"
    display.DiffuseColor = [0.7, 0.75, 0.79]
    display.EdgeColor = [0.15, 0.19, 0.23]
    display.LineWidth = 1.0
    display.Opacity = opacity
    display.Ambient = 1.0 if field else 0.35
    display.Diffuse = 0.0 if field else 0.65
    display.Specular = 0.0
    # Map cell scalars to colors on the CPU. Texture interpolation on offscreen
    # renderers can drop polygon interiors while leaving edges/legends visible.
    display.InterpolateScalarsBeforeMapping = 0
    display.MapScalars = 1
    bounds = list(info.GetBounds())
    center = list(center) if center is not None else [(bounds[2*i] + bounds[2*i+1]) / 2 for i in range(3)]
    directions = {"oblique": [1.3, 1.0, 1.6], "front": [0, 1, 0],
                  "back": [0, -1, 0], "xy": [0, 0, 1], "yz": [1, 0, 0]}
    direction = np.asarray(directions[camera] if isinstance(camera, str) else camera, dtype=float)
    direction /= np.linalg.norm(direction)
    up = np.array([0, 0, 1] if isinstance(camera, str) and camera in {"front", "back"} else [0, 1, 0], dtype=float)
    right = np.cross(up, direction)
    right /= np.linalg.norm(right)
    up = np.cross(direction, right)
    aspect = settings["image_resolution"][0] / settings["image_resolution"][1]
    if span is None:
        # Fit the actual projected geometry, not a domain-sized bounding sphere.
        corners = np.array([[x, y, z] for x in bounds[:2] for y in bounds[2:4] for z in bounds[4:]]) - center
        half_height = max(np.max(np.abs(corners @ up)), np.max(np.abs(corners @ right)) / aspect, 1e-8)
        span = max(bounds[2*i+1] - bounds[2*i] for i in range(3))
    else:
        half_height = max(span / 2, span / (2 * aspect), 1e-8)
    span = max(span, 1e-8)
    view.CameraParallelProjection = 1
    view.CameraFocalPoint = center
    view.CameraViewUp = up.tolist()
    view.CameraParallelScale = half_height * (1.38 if field else 1.26)
    # Lift geometry clear of the scalar bar without shrinking it.
    focal_point = np.asarray(center) - up * half_height * (0.15 if field else 0.)
    view.CameraFocalPoint = focal_point.tolist()
    view.CameraPosition = (focal_point + direction * span * 3).tolist()
    # Oblique blade-section cuts do not have a Cartesian horizontal axis.
    # Three projected axes on one line falsely suggest a simple x/z section.
    view.AxesGrid.Visibility = int(isinstance(camera, str))
    view.AxesGrid.UseCustomBounds = 1
    view.AxesGrid.AxesToLabel = sum(1 << i for i in range(3) if abs(direction[i]) < .95)
    view.AxesGrid.CustomBounds = [value for i in range(3) for value in
                                 (max(bounds[2*i], center[i] - span/2), min(bounds[2*i+1], center[i] + span/2))]
    for i, axis in enumerate("XYZ"):
        low, high = view.AxesGrid.CustomBounds[2*i:2*i+2]
        setattr(view.AxesGrid, axis + "AxisUseCustomLabels", 1)
        setattr(view.AxesGrid, axis + "AxisLabels", [float(f"{v:.4g}") for v in (low, high)] if high - low > span * 1e-8 else [float(low)])
    display.SetScalarBarVisibility(view, False)
    data_range = None
    if field:
        association, name, label = field
        array = (source.CellData if association == "CELLS" else source.PointData).GetArray(name)
        if array is None:
            raise ValueError(f"Missing {association} array {name} for {title}")
        data_range = list(array.GetRange(-1 if array.GetNumberOfComponents() > 1 else 0))
        if not all(math.isfinite(v) for v in data_range):
            raise ValueError(f"Non-finite range for {name}")
        limits = list(limits or _pvvis_limits(data_range, name, settings))
        logarithmic = name in settings.get("log_fields", []) and limits[1] > 0
        if logarithmic:
            # Zero stays in data/metadata; only its display color is clamped.
            limits[0] = max(limits[0], limits[1] * 1e-6)
            label += " (log scale)"
        pvs.ColorBy(display, (association, name))
        lut = pvs.GetColorTransferFunction(name)
        display.LookupTable = lut
        lut.EnableOpacityMapping = 0
        if array.GetNumberOfComponents() > 1:
            lut.VectorMode = "Magnitude"
        # Explicit perceptually ordered control colors avoid installation-
        # dependent preset names and rainbow-map false boundaries.
        colors = ([[0.23, 0.30, 0.75], [0.87, 0.87, 0.87], [0.71, 0.02, 0.15]] if diverging else
                  [[0.267, 0.005, 0.329], [0.230, 0.322, 0.546], [0.128, 0.567, 0.551],
                   [0.369, 0.789, 0.383], [0.993, 0.906, 0.144]])
        lut.ColorSpace = "Lab"
        lut.UseLogScale = int(logarithmic)
        lut.RGBPoints = [v for position, rgb in zip(np.geomspace(*limits, len(colors)) if logarithmic else np.linspace(*limits, len(colors)), colors) for v in [float(position), *rgb]]
        lut.NanColor = [1.0, 0.0, 1.0]
        lut.AutomaticRescaleRangeMode = "Never"
        display.SetScalarBarVisibility(view, True)
        bar = pvs.GetScalarBar(lut, view)
        bar.Title, bar.ComponentTitle = label, ""
        bar.WindowLocation = "Lower Center"
        bar.Orientation = "Horizontal"
        bar.ScalarBarLength = 0.6
        bar.TitleColor = bar.LabelColor = [0.08, 0.1, 0.13]
        bar.TitleFontSize = max(20, round(settings["image_resolution"][0] * .016))
        bar.LabelFontSize = max(18, round(settings["image_resolution"][0] * .013))
        bar.AutomaticLabelFormat = 0
        # Recent VTK uses std::format; older ParaView versions use printf.
        bar.LabelFormat = "{:.7g}" if "{" in str(bar.LabelFormat) else "%.7g"
        bar.RangeLabelFormat = "{:.7g}" if "{" in str(bar.RangeLabelFormat) else "%.7g"
        bar.UseCustomLabels = 1
        bar.CustomLabels = [float(v) for v in (np.geomspace(*limits, 5) if logarithmic else np.linspace(*limits, 5))]
        bar.AddRangeLabels = 0
        bar.WindowLocation = "Any Location"
        bar.Position = [0.2, 0.065]
    else:
        display.ColorArrayName = ["POINTS", ""]
    for overlay in overlays:
        od = pvs.Show(overlay, view)
        od.ColorArrayName = ["POINTS", ""]
        od.DiffuseColor = [0.35, 0.38, 0.42]
        od.Opacity = 1.0
    # Explicitly refresh clipping after moving between observer, rotor and
    # near-wall scales. This works with local offscreen RenderView backends.
    view.Update()
    renderer = view.GetClientSideObject().GetRenderer()
    renderer.ResetCameraClippingRange()
    pvs.Render(view)
    filename = f"view_{len(result['views']) + 1:04d}.png"
    pvs.SaveScreenshot(str(run_dir / filename), view, ImageResolution=settings["image_resolution"], TransparentBackground=0)
    if field:
        _pvvis_check_foreground(run_dir / filename)
    entry = {
        "title": title, "caption": caption, "image": f"{run_dir.name}/{filename}",
        "camera": camera if isinstance(camera, str) else "blade section", "bounds_m": bounds,
        "camera_position_m": list(view.CameraPosition), "camera_focal_point_m": list(view.CameraFocalPoint),
        "parallel_scale_m": float(view.CameraParallelScale),
    }
    if field:
        entry.update(field=label, association=field[0], color_range=list(limits), data_range=data_range,
                     color_scale="logarithmic" if logarithmic else "linear")
        display.SetScalarBarVisibility(view, False)
    if time_s is not None:
        entry.update(time_s=float(time_s), phase_deg=float((time_s * settings["rpm"] * 6) % 360))
    result["views"].append(entry)
    _pvvis_save(result, run_dir)


def _pvvis_check_foreground(path):
    """Reject blank plot areas even when axes and a scalar bar were rendered."""
    from vtkmodules.vtkIOImage import vtkPNGReader
    from vtkmodules.util.numpy_support import vtk_to_numpy

    reader = vtkPNGReader()
    reader.SetFileName(str(path))
    reader.Update()
    image = reader.GetOutput()
    width, height, _ = image.GetDimensions()
    pixels = vtk_to_numpy(image.GetPointData().GetScalars()).reshape(height, width, -1)
    # VTK image rows run bottom to top. Exclude legends and orientation axes.
    plot = pixels[int(height * .24):int(height * .90),
                  int(width * .20):int(width * .80), :3]
    foreground = np.any(plot < 225, axis=2)
    if float(np.mean(foreground)) < .003:
        raise ValueError("Rendered plot area is nearly blank; figure withheld. Check the offscreen renderer and source geometry.")


def _pvvis_attempt(result, run_dir, name, action):
    try:
        return action()
    except Exception as exc:
        message = f"{name}: {exc}"
        result["warnings"].append(message)
        print(message, flush=True)
        traceback.print_exc()
        _pvvis_save(result, run_dir)
        return None


def _pvvis_statistics(entries, settings, run_dir, pressure_unit):
    """Stream exact trapezoid-weighted panel moments; reject changing identities.

    No time decimation: pressure and its derivative use every saved surface in
    the requested window. Array memory scales with a few surfaces, not duration.
    """
    from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk
    from vtkmodules.vtkIOXML import vtkXMLPolyDataWriter
    from vtkmodules.vtkCommonDataModel import vtkPolyData

    end = entries[-1][0]
    start = end - settings["statistics_revolutions"] * 60 / settings["rpm"]
    selected = [(t, p) for t, p in entries if t >= start]
    if len(selected) < 3:
        raise ValueError("At least three surface samples are needed for temporal statistics")
    times = np.array([t for t, _ in selected], dtype=float)
    dt = np.diff(times)
    weights = np.r_[dt[0] / 2, (dt[:-1] + dt[1:]) / 2, dt[-1] / 2]
    first = _pvvis_read_surface(selected[0][1])
    reference = vtkPolyData()
    reference.DeepCopy(first)
    ref_points = vtk_to_numpy(first.GetPoints().GetData()).copy()
    ref_topology = vtk_to_numpy(first.GetPolys().GetData()).copy()
    association, initial = _pvvis_array(first, "p")
    if initial.ndim != 1:
        raise ValueError("Pressure must be scalar")
    mean = np.zeros(initial.shape, dtype=np.float64)
    m2 = np.zeros_like(mean)
    derivative_energy = np.zeros_like(mean)
    total_weight, previous = 0.0, None
    length = max(float(np.ptp(ref_points, axis=0).max()), 1e-8)
    omega = settings["rpm"] * 2 * math.pi / 60
    for index, ((t, path), weight) in enumerate(zip(selected, weights)):
        data = _pvvis_read_surface(path)
        assoc, values = _pvvis_array(data, "p")
        points = vtk_to_numpy(data.GetPoints().GetData())
        topology = vtk_to_numpy(data.GetPolys().GetData())
        if assoc != association or values.shape != mean.shape or points.shape != ref_points.shape or not np.array_equal(topology, ref_topology):
            raise ValueError(f"Surface topology/panel ordering changed at {t:g} s; temporal maps withheld")
        if settings["acoustic_surface"] == "impermeable":
            angle = omega * (t - times[0])
            ca, sa = math.cos(angle), math.sin(angle)
            rotation = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])
            expected = ref_points @ rotation.T
        else:
            expected = ref_points
        if not np.allclose(points, expected, rtol=0, atol=length * 2e-5):
            raise ValueError(f"Surface motion/point correspondence inconsistent at {t:g} s; temporal maps withheld")
        values = np.asarray(values, dtype=np.float64)
        new_weight = total_weight + weight
        delta = values - mean
        mean += delta * weight / new_weight
        m2 += weight * delta * (values - mean)
        total_weight = new_weight
        if previous is not None:
            derivative_energy += (values - previous) ** 2 / dt[index - 1]
        previous = values.copy()
    reference.GetCellData().Initialize()
    reference.GetPointData().Initialize()
    attributes = reference.GetCellData() if association == "CELLS" else reference.GetPointData()
    for name, values in (("p_mean", mean), ("p_rms", np.sqrt(np.maximum(m2 / total_weight, 0))),
                         ("dpdt_rms", np.sqrt(derivative_energy / (times[-1] - times[0])))):
        array = numpy_to_vtk(values, deep=True)
        array.SetName(name)
        attributes.AddArray(array)
    writer = vtkXMLPolyDataWriter()
    writer.SetFileName(str(run_dir / "surface_statistics.vtp"))
    writer.SetInputData(reference)
    if writer.Write() != 1:
        raise OSError("Could not write surface statistics")
    metadata = {
        "start_s": float(times[0]), "end_s": float(times[-1]), "samples": len(times),
        "revolutions": float((times[-1] - times[0]) * settings["rpm"] / 60),
        "min_dt_s": float(dt.min()), "max_dt_s": float(dt.max()),
        "association": association, "pressure_unit": pressure_unit,
        "frame": "rotating panels" if settings["acoustic_surface"] == "impermeable" else "stationary panels",
        "weighting": "trapezoidal physical-time weights; interval first differences for dp/dt RMS",
        "reference_geometry_time_s": float(times[0]),
        "source_directory": str(Path(selected[0][1]).parent.parent),
    }
    (run_dir / "surface_statistics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return reference, metadata


def _pvvis_surface(result, settings, run_dir, view, units):
    from vtkmodules.util.numpy_support import vtk_to_numpy

    case = Path(settings["case_path"])
    permeable = settings["acoustic_surface"] == "permeable"
    directory = case / "postProcessing" / ("writePermeableSurfaceFields" if permeable else "writePatchFields")
    filename = "permeableSurface.vtk" if permeable else "propeller.vtk"
    entries = _pvvis_times(directory, filename)
    if not entries:
        raise ValueError(f"No original acoustic surface samples at {directory}")
    selected = _pvvis_select(entries, settings["rpm"], settings["surface_phases"])
    result["surface_samples"] = {"available": len(entries), "rendered_times_s": [t for t, _ in selected], "source": str(directory)}
    if len(selected) < settings["surface_phases"]:
        result["warnings"].append(f"Only {len(selected)} distinct surface snapshots available for {settings['surface_phases']} requested phases")
    # Pre-scan selected samples for common ranges, without retaining their data.
    ranges = {"surface_p": [math.inf, -math.inf], "surface_speed": [math.inf, -math.inf]}
    for _, path in selected:
        data = _pvvis_read_surface(path)
        for field, key in (("p", "surface_p"), ("U", "surface_speed")):
            try:
                _, values = _pvvis_array(data, field)
                if values.ndim > 1:
                    values = np.linalg.norm(values, axis=1)
                ranges[key] = [min(ranges[key][0], float(values.min())), max(ranges[key][1], float(values.max()))]
            except ValueError:
                pass
    data = _pvvis_read_surface(selected[-1][1])
    points = vtk_to_numpy(data.GetPoints().GetData())
    if settings["diameter_m"] is None and not permeable:
        settings["diameter_m"] = float(2 * np.linalg.norm(points[:, [0, 2]], axis=1).max())
    # Rotation-invariant framing keeps tips visible at every sampled phase.
    span = max(float(2 * np.linalg.norm(points[:, [0, 2]], axis=1).max()), float(np.ptp(points[:, 1])))
    source = pvs.TrivialProducer()
    source.GetClientSideObject().SetOutput(data)
    source.UpdatePipeline()
    surface_name = "Permeable FW-H surface" if permeable else "Blade FW-H surface"
    try:
        for camera in ("oblique", "front"):
            _pvvis_render(source, result, settings, run_dir, view, f"{surface_name} - sampling mesh",
                          "Inspect surface coverage, panel distribution and geometric continuity. The rendered geometry is the original saved surface, in the laboratory frame.",
                          camera=camera, time_s=selected[-1][0], edges=True)
        blade_path = case / "constant" / "triSurface" / "propeller.stl"
        if permeable and blade_path.is_file():
            def enclosure_view():
                blade = pvs.STLReader(FileNames=[str(blade_path)])
                rotated = pvs.Transform(Input=blade)
                rotated.Transform.Rotate = [0, selected[-1][0] * settings["rpm"] * 6, 0]
                try:
                    _pvvis_render(source, result, settings, run_dir, view, "Permeable FW-H surface - blade enclosure",
                                  "Translucent integration surface with the blade geometry inside. Inspect placement relative to blade tips and wake; a visually closed enclosure alone does not establish FW-H accuracy or exclude hydrodynamic contamination.",
                                  overlays=[rotated], opacity=0.2, time_s=selected[-1][0])
                finally:
                    pvs.Delete(rotated)
                    pvs.Delete(blade)
            _pvvis_attempt(result, run_dir, "Permeable enclosure", enclosure_view)
        observer = pvs.Sphere(Center=settings["observer_m"], Radius=span * 0.025)
        connector = pvs.Line(Point1=[0, 0, 0], Point2=settings["observer_m"])
        group = pvs.GroupDatasets(Input=[source, observer, connector])
        _pvvis_attempt(result, run_dir, "Observer geometry", lambda: _pvvis_render(
            group, result, settings, run_dir, view, "Acoustic surface and observer location",
            f"Sphere marker: observer at {settings['observer_m']} m; line connects rotor origin and observer. Marker radius is illustrative. This is the observer used by the current acoustic stage unless explicitly overridden for this atlas.",
            camera="oblique"))
        pvs.Delete(group)
        pvs.Delete(connector)
        pvs.Delete(observer)
        for t, path in selected:
            data = _pvvis_read_surface(path)
            source.GetClientSideObject().SetOutput(data)
            source.MarkModified(source)
            source.UpdatePipeline()
            for camera in ("front", "back"):
                def pressure_view():
                    association, _ = _pvvis_array(data, "p")
                    _pvvis_render(source, result, settings, run_dir, view,
                                  f"{surface_name} - pressure ({camera})",
                                  "Compare loading patterns across saved rotor phases using the same color scale. Pressure includes the solver reference offset; this is a source-surface field, not observer acoustic pressure. Front/back mean views from +y/-y.",
                                  camera=camera, field=(association, "p", f"{units[1]} [{units[0]}]"),
                                  limits=_pvvis_limits(ranges["surface_p"], "surface_p", settings),
                                  center=[0, 0, 0], span=span, time_s=t)
                _pvvis_attempt(result, run_dir, f"Surface p at {t:g}, {camera}", pressure_view)
        def velocity_view():
            association, _ = _pvvis_array(data, "U")
            _pvvis_render(source, result, settings, run_dir, view, f"{surface_name} - fluid speed",
                          "Saved fluid velocity magnitude in the laboratory frame. On an impermeable moving wall, wall motion can dominate this value; it is not the fluid velocity relative to the blade.",
                          field=(association, "U", "Fluid speed [m/s]"),
                          limits=_pvvis_limits(ranges["surface_speed"], "surface_speed", settings), time_s=selected[-1][0])
        _pvvis_attempt(result, run_dir, "Surface velocity", velocity_view)
        statistics = _pvvis_attempt(result, run_dir, "Surface temporal statistics", lambda: _pvvis_statistics(entries, settings, run_dir, units[0]))
        if statistics:
            stats_data, metadata = statistics
            result["surface_statistics"] = metadata
            source.GetClientSideObject().SetOutput(stats_data)
            source.MarkModified(source)
            source.UpdatePipeline()
            window = (f"{metadata['samples']} samples, t={metadata['start_s']:.6g} to {metadata['end_s']:.6g} s "
                      f"({metadata['revolutions']:.3g} rev); dt={metadata['min_dt_s']:.3g} to {metadata['max_dt_s']:.3g} s. ")
            if metadata["revolutions"] < settings["statistics_revolutions"] * 0.95:
                result["warnings"].append("Surface statistics cover less than the requested number of revolutions; inspect the actual window")
            diagnostics = [
                ("p_mean", f"Mean p [{units[0]}]", "Time-mean surface pressure", "Mean loading on corresponding panels. "),
                ("p_rms", f"p fluctuation RMS [{units[0]}]", "Surface pressure fluctuation RMS", "Highlights unsteady panel loading: RMS of p minus each panel's temporal mean. Low RMS on rotating panels does not imply low radiated noise. "),
                ("dpdt_rms", f"dp/dt RMS [({units[0]})/s]", "Surface pressure-change RMS", "Highlights rapidly changing panel pressure; interval first differences follow the sampled panels. Sensitive to output cadence and numerical noise. "),
            ]
            for name, label, title, explanation in diagnostics:
                for camera in ("front", "back", "oblique"):
                    _pvvis_attempt(result, run_dir, f"{name}, {camera}", lambda: _pvvis_render(
                        source, result, settings, run_dir, view, title,
                        explanation + window + f"Frame: {metadata['frame']}; shown on geometry at t={metadata['reference_geometry_time_s']:.6g} s. These are diagnostics, not FW-H contributions or SPL.",
                        camera=camera, field=(metadata["association"], name, label)))
            # Signed instantaneous departures complement RMS, which loses sign.
            # Reuse only samples whose identity was verified in this window.
            from vtkmodules.util.numpy_support import numpy_to_vtk
            _, panel_mean = _pvvis_array(stats_data, "p_mean")
            fluctuation_samples = [(t, path) for t, path in selected if t >= metadata["start_s"]]
            extrema = [math.inf, -math.inf]
            for _, path in fluctuation_samples:
                _, values = _pvvis_array(_pvvis_read_surface(path), "p")
                fluctuation = values - panel_mean
                extrema = [min(extrema[0], float(fluctuation.min())), max(extrema[1], float(fluctuation.max()))]
            limits = _pvvis_limits(extrema, "surface_p_fluctuation", settings, symmetric=True)
            for t, path in fluctuation_samples:
                data = _pvvis_read_surface(path)
                association, values = _pvvis_array(data, "p")
                array = numpy_to_vtk(values - panel_mean, deep=True)
                array.SetName("p_fluctuation")
                (data.GetCellData() if association == "CELLS" else data.GetPointData()).AddArray(array)
                source.GetClientSideObject().SetOutput(data)
                source.MarkModified(source)
                source.UpdatePipeline()
                for camera in ("front", "back"):
                    _pvvis_attempt(result, run_dir, f"Pressure fluctuation at {t:g}, {camera}", lambda: _pvvis_render(
                        source, result, settings, run_dir, view, f"Surface pressure departure from panel mean ({camera})",
                        "Signed p minus each corresponding panel's temporal mean. Compare positive and negative loading departures across rotor phases. " + window + "This is not a separation of acoustic and hydrodynamic pressure.",
                        camera=camera, field=(association, "p_fluctuation", f"p - panel mean [{units[0]}]"),
                        limits=limits, center=[0, 0, 0], span=span, time_s=t, diverging=True))
    finally:
        pvs.Delete(source)


def _pvvis_calc(source, name, expression):
    calculation = pvs.Calculator(Input=source)
    calculation.AttributeType = "Cell Data"
    calculation.ResultArrayName = name
    calculation.Function = expression
    calculation.UpdatePipeline()
    return calculation


def _pvvis_crop(source, center, widths):
    """Physically crop to the plotted region so remote cells cannot set its legend."""
    cropped = pvs.Clip(Input=source)
    cropped.ClipType = "Box"
    cropped.ClipType.Position = [center[i] - widths[i] / 2 for i in range(3)]
    cropped.ClipType.Length = list(widths)
    cropped.Invert = 1
    cropped.Crinkleclip = 1  # Retain native cells; ordinary Clip invents triangle edges.
    return cropped


def _pvvis_blade_detail(base, wall, t, result, settings, run_dir, view, units):
    """Use the saved blade geometry to place chordwise cuts, including its actual phase."""
    from vtkmodules.util.numpy_support import vtk_to_numpy

    merged = pvs.MergeBlocks(Input=wall)
    surface = pvs.ExtractSurface(Input=merged)
    try:
        surface.UpdatePipeline(t)
        data = servermanager.Fetch(surface)  # Only the blade surface, never the volume.
        points = vtk_to_numpy(data.GetPoints().GetData())
        radial_points = points[:, [0, 2]]
        _, axes = np.linalg.eigh(radial_points.T @ radial_points)
        radial = np.array([axes[0, -1], 0., axes[1, -1]])
        tangent = np.cross([0., 1., 0.], radial)
        radius = np.linalg.norm(radial_points, axis=1).max()
        for fraction in (0.5, 0.75, 0.95, -0.95):
            origin = radial * radius * fraction
            outline = pvs.Slice(Input=surface)
            outline.Triangulatetheslice = 0
            outline.SliceType.Normal = radial.tolist()
            outline.SliceType.Origin = origin.tolist()
            sliced = pvs.Slice(Input=base)
            sliced.Triangulatetheslice = 0
            sliced.SliceType.Normal = radial.tolist()
            sliced.SliceType.Origin = origin.tolist()
            cropped = None
            try:
                outline.UpdatePipeline(t)
                if outline.GetDataInformation().GetNumberOfCells() == 0:
                    continue
                section = servermanager.Fetch(outline)
                xyz = vtk_to_numpy(section.GetPoints().GetData())
                center = (xyz.min(axis=0) + xyz.max(axis=0)) / 2
                span = max(float(np.ptp(xyz @ tangent)) * 1.6, radius * 0.12)
                cropped = _pvvis_crop(sliced, center, [span] * 3)
                cropped.UpdatePipeline(t)
                station = f"r/R={fraction:g}"
                caption = (f"Chordwise cut at signed {station}, normal to the blade's measured principal radial axis. "
                           "Blade position comes from the saved mesh. Velocity is relative to rigid rotation about +y. "
                           "Inspect near-wall gradients and the trailing-edge shear layer; resolved detail depends on the mesh and saved fields.")
                for name, label in (("relative_speed", "Blade-relative speed [m/s]"),
                                    ("vorticity_magnitude", "Vorticity magnitude [1/s]")):
                    _pvvis_attempt(result, run_dir, f"Blade section {name} {station}", lambda: _pvvis_render(
                        cropped, result, settings, run_dir, view, f"Blade flow section - {name.replace('_', ' ')} ({station})",
                        caption + " Local color scale.", camera=radial.tolist(), field=("CELLS", name, label),
                        time_s=t, overlays=[outline]))
                if cropped.CellData.GetArray("relative_velocity") is not None:
                    normal_vector = "(" + "+".join(f"{v:.16g}*{axis}Hat" for v, axis in zip(radial, "ijk")) + ")"
                    projected = _pvvis_calc(cropped, "section_velocity",
                                            f"relative_velocity-dot(relative_velocity,{normal_vector})*{normal_vector}")
                    centers = pvs.CellCenters(Input=projected)
                    arrows = pvs.Glyph(Input=centers, GlyphType="Arrow")
                    arrows.OrientationArray = ["POINTS", "section_velocity"]
                    arrows.ScaleArray = ["POINTS", "No scale array"]
                    arrows.ScaleFactor = span * .035
                    arrows.GlyphMode = "Uniform Spatial Distribution (Bounds Based)"
                    arrows.MaximumNumberOfSamplePoints = 200
                    try:
                        _pvvis_render(cropped, result, settings, run_dir, view, f"Blade-relative flow directions ({station})",
                                      caption + " Equal-length arrows show the in-plane direction; color gives full relative speed, including its out-of-plane component.",
                                      camera=radial.tolist(), field=("CELLS", "relative_speed", "Blade-relative speed [m/s]"),
                                      time_s=t, overlays=[outline, arrows])
                    finally:
                        pvs.Delete(arrows)
                        pvs.Delete(centers)
                        pvs.Delete(projected)
                _pvvis_render(cropped, result, settings, run_dir, view, f"Blade mesh section ({station})",
                              "Actual volume-cell intersections around the blade profile. Inspect layer continuity, growth and transitions; section widths are not wall-normal layer-thickness measurements.",
                              camera=radial.tolist(), edges=True, time_s=t, overlays=[outline])
                # Two extrema along the local chord: do not guess which is the leading edge.
                for side, index in (("chord end A", np.argmin(xyz @ tangent)), ("chord end B", np.argmax(xyz @ tangent))):
                    zoom_center = xyz[index]
                    zoom = _pvvis_crop(sliced, zoom_center, [span * .3] * 3)
                    try:
                        _pvvis_render(zoom, result, settings, run_dir, view, f"Blade layer mesh - {side} ({station})",
                                      "Magnified volume-cell section at a chord extremum. Inspect edge resolution, near-wall layers and refinement transitions. A/B label geometry, not an inferred leading/trailing edge.",
                                      camera=radial.tolist(), edges=True, time_s=t)
                        _pvvis_attempt(result, run_dir, f"Near-wall flow {station} {side}", lambda: _pvvis_render(
                            zoom, result, settings, run_dir, view, f"Near-wall flow - {side} ({station})",
                            caption + " Local color scale; cell edges expose the available resolution.",
                            camera=radial.tolist(), field=("CELLS", "relative_speed", "Blade-relative speed [m/s]"),
                            edges=True, time_s=t))
                    finally:
                        pvs.Delete(zoom)
            finally:
                if cropped is not None:
                    pvs.Delete(cropped)
                pvs.Delete(sliced)
                pvs.Delete(outline)
    finally:
        pvs.Delete(surface)
        pvs.Delete(merged)


def _pvvis_volume(result, settings, run_dir, view, units):
    case = Path(settings["case_path"])
    if not (case / "constant" / "polyMesh").is_dir():
        raise ValueError("Reconstructed constant/polyMesh unavailable; volume and blade-wall views withheld")
    # ParaView needs a case marker in the case root; preserve any existing one.
    markers = sorted(case.glob("*.foam"))
    marker = markers[0] if markers else case / "visualization.foam"
    if not marker.exists():
        marker.touch()
    reader = pvs.OpenFOAMReader(FileName=str(marker))
    reader.CaseType = "Reconstructed Case"
    reader.MeshRegions = ["internalMesh"]
    reader.SkipZeroTime = 1
    reader.Createcelltopointfiltereddata = 0
    reader.UpdatePipelineInformation()
    available = list(reader.CellArrays.Available)
    wanted = [name for name in ("p", "U", "k", "Q", "vorticity", "Co") if name in available]
    missing = [name for name in ("p", "U", "k", "Co") if name not in available]
    if missing:
        result["warnings"].append(f"Volume fields unavailable (corresponding views omitted): {', '.join(missing)}")
    reader.CellArrays = wanted
    times = [(float(t), None) for t in reader.TimestepValues if float(t) > 0]
    if not times:
        pvs.Delete(reader)
        raise ValueError("No saved nonzero volume times available")
    selected = _pvvis_select(times, settings["rpm"], settings["volume_phases"])
    result["volume_samples"] = {"available": len(times), "rendered_times_s": [t for t, _ in selected], "fields": wanted, "source": str(marker)}
    diameter = settings["diameter_m"]
    if diameter is None:
        pvs.Delete(reader)
        raise ValueError("Cannot determine propeller diameter; set diameter_m in visualization.json")
    if len(selected) < settings["volume_phases"]:
        result["warnings"].append(f"Only {len(selected)} distinct volume snapshots available for {settings['volume_phases']} requested phases")
    proxies = []
    base = reader
    if "U" in wanted:
        omega = settings["rpm"] * 2 * math.pi / 60
        relative = pvs.PythonCalculator(Input=base)
        relative.ArrayAssociation = "Cell Data"
        relative.ArrayName = "relative_velocity"
        relative.UseMultilineExpression = 1
        relative.MultilineExpression = f"""import numpy as np
from vtkmodules.vtkFiltersCore import vtkCellCenters
from vtkmodules.numpy_interface import dataset_adapter as dsa
def velocity(block):
    centers = vtkCellCenters()
    centers.SetInputData(block.VTKObject)
    centers.Update()
    xyz = dsa.WrapDataObject(centers.GetOutput()).Points
    return block.CellData['U'] - {omega:.16g} * np.cross([0., 1., 0.], xyz)
data = inputs[0]
if isinstance(data, dsa.CompositeDataSet):
    return dsa.VTKCompositeDataArray([velocity(block) for block in data], dataset=data)
return velocity(data)
"""
        base = relative
        proxies.append(base)
        for name, expression in (("speed", "mag(U)"), ("axial_velocity", "U_Y"),
                                 ("relative_speed", "mag(relative_velocity)"),
                                 ("swirl_velocity", "dot(U-relative_velocity,U)/(mag(U-relative_velocity)+1e-30)")):
            base = _pvvis_calc(base, name, expression)
            proxies.append(base)
        if "vorticity" not in wanted or "Q" not in wanted:
            gradient = pvs.Gradient(Input=base)
            gradient.ScalarArray = ["CELLS", "U"]
            gradient.ComputeGradient = 0
            gradient.ComputeVorticity = int("vorticity" not in wanted)
            gradient.ComputeQCriterion = int("Q" not in wanted)
            gradient.VorticityArrayName = "vorticity"
            gradient.QCriterionArrayName = "Q"
            base = gradient
            proxies.append(base)
        base = _pvvis_calc(base, "vorticity_magnitude", "mag(vorticity)")
        proxies.append(base)
    fields = []
    if "U" in wanted:
        fields += [("speed", "Speed [m/s]", False), ("axial_velocity", "Axial velocity U_y [m/s]", True),
                   ("swirl_velocity", "Tangential velocity [m/s]", True),
                   ("vorticity_magnitude", "Vorticity magnitude [1/s]", False)]
    if "p" in wanted:
        fields += [("p", f"{units[1]} [{units[0]}]", False)]
    if "k" in wanted:
        fields += [("k", "Turbulent kinetic energy k [m^2/s^2]", False)]
    if "Co" in wanted:
        fields += [("Co", "Cell Courant number [-]", False)]
    if not fields:
        raise ValueError("No supported volume fields found")
    ranges_by_station = {}
    # Broad wake context plus rotor close-ups; all legends use these actual crops.
    cuts = [("xy", [0, 0, 1], [0, 0, 0], "z=0, wake", [1.35*diameter, 2.5*diameter, 1.35*diameter]),
            ("yz", [1, 0, 0], [0, 0, 0], "x=0, wake", [1.35*diameter, 2.5*diameter, 1.35*diameter]),
            ("xy", [0, 0, 1], [0, 0, 0], "z=0, rotor", [1.15*diameter]*3),
            ("yz", [1, 0, 0], [0, 0, 0], "x=0, rotor", [1.15*diameter]*3)]
    cuts += [("front", [0, 1, 0], [0, station*diameter, 0], f"y/D={station:g}", [1.35*diameter]*3)
             for station in settings["wake_stations_D"]]
    try:
        for t, _ in selected:
            view.ViewTime = t
            base.UpdatePipeline(t)
            bounds = base.GetDataInformation().GetBounds()
            for camera, normal, origin, station, widths in cuts:
                if not all(bounds[2*i] <= origin[i] <= bounds[2*i+1] for i in range(3)):
                    result["warnings"].append(f"Slice {station} lies outside saved domain at {t:g} s")
                    continue
                sliced = pvs.Slice(Input=base)
                sliced.Triangulatetheslice = 0
                sliced.SliceType = "Plane"
                sliced.SliceType.Origin = origin
                sliced.SliceType.Normal = normal
                cropped = _pvvis_crop(sliced, origin, widths)
                pressure = None
                # Latest time gets the full diagnostic set; earlier times show
                # velocity/pressure/vorticity to expose transient wake changes.
                active = fields if t == selected[-1][0] else [f for f in fields if f[0] in {"speed", "p", "vorticity_magnitude"}]
                try:
                    if station not in ranges_by_station:
                        ranges = {name: [math.inf, -math.inf] for name, _, _ in fields}
                        for sample_time, _ in selected:
                            cropped.UpdatePipeline(sample_time)
                            if cropped.GetDataInformation().GetNumberOfCells() == 0:
                                continue
                            for name, _, _ in fields:
                                lo, hi = cropped.CellData.GetArray(name).GetRange()
                                ranges[name] = [min(ranges[name][0], lo), max(ranges[name][1], hi)]
                        ranges_by_station[station] = {
                            name: _pvvis_limits(ranges[name], f"volume_{name}", settings, symmetric)
                            for name, _, symmetric in fields}
                    ranges = ranges_by_station[station]
                    cropped.UpdatePipeline(t)
                    if "p" in wanted:
                        reference = sum(ranges["p"]) / 2
                        pressure = _pvvis_calc(cropped, "p_departure", f"p - {reference:.16g}")
                    for name, label, diverging in active:
                        limits = ranges[name]
                        display_source, display_name = cropped, name
                        note = ""
                        if name == "p":
                            display_source, display_name = pressure, "p_departure"
                            limits = [value - reference for value in limits]
                            label = f"p - reference [{units[0]}]"
                            diverging = True
                            note = f" Pressure reference = {reference:.10g} {units[0]} (fixed range midpoint, not freestream or acoustic pressure)."
                        _pvvis_attempt(result, run_dir, f"{name}, {station}, t={t:g}", lambda: _pvvis_render(
                            display_source, result, settings, run_dir, view, f"Flow slice - {name.replace('_', ' ')} ({station})",
                            f"Plane {station}, laboratory frame; D={diameter:.6g} m. Inspect wake asymmetry, shear layers and rotor loading. Color range covers this cropped station across all selected phases; compare legends between stations." + note,
                            camera=camera, field=("CELLS", display_name, label), limits=limits,
                            time_s=t, diverging=diverging))
                    if t == selected[-1][0]:
                        _pvvis_attempt(result, run_dir, f"Mesh slice {station}", lambda: _pvvis_render(
                            cropped, result, settings, run_dir, view, f"Mesh section ({station})",
                            "Inspect refinement transitions and rotor-region cell structure. Slice edges show sectioned cells; they do not quantify mesh quality or prove adequate boundary-layer resolution.",
                            camera=camera, edges=True, time_s=t))
                finally:
                    if pressure is not None:
                        pvs.Delete(pressure)
                    pvs.Delete(cropped)
                    pvs.Delete(sliced)
            if "U" in wanted:
                def vortex_views():
                    region = _pvvis_crop(base, [0, 0, 0], [1.35*diameter, 2.5*diameter, 1.35*diameter])
                    point_data = pvs.CellDatatoPointData(Input=region)
                    point_data.ProcessAllArrays = 0
                    point_data.CellDataArraytoprocess = ["Q", "vorticity_magnitude"]
                    contour = pvs.Contour(Input=point_data)
                    contour.ContourBy = ["POINTS", "Q"]
                    try:
                        color_range = [math.inf, -math.inf]
                        omega = settings["rpm"] * 2 * math.pi / 60
                        # Scale to the actual displayed isosurfaces, not unrelated
                        # high-gradient cells elsewhere in the volume crop.
                        for sample_time, _ in selected:
                            for normalized_q in settings["q_over_omega2"]:
                                contour.Isosurfaces = [normalized_q * omega ** 2]
                                contour.UpdatePipeline(sample_time)
                                if contour.GetDataInformation().GetNumberOfCells():
                                    lo, hi = contour.PointData.GetArray("vorticity_magnitude").GetRange()
                                    color_range = [min(color_range[0], lo), max(color_range[1], hi)]
                        color_range = _pvvis_limits(color_range, "volume_vorticity_magnitude", settings)
                        for normalized_q in settings["q_over_omega2"]:
                            threshold = normalized_q * omega ** 2
                            contour.Isosurfaces = [threshold]
                            contour.UpdatePipeline(t)
                            for camera in ("oblique", "xy"):
                                _pvvis_attempt(result, run_dir, f"Q={threshold:g}, {camera}, t={t:g}", lambda: _pvvis_render(
                                    contour, result, settings, run_dir, view, f"Vortex structures - Q/omega^2 = {normalized_q:g}",
                                    f"Q={threshold:.6g} s^-2, colored by vorticity magnitude. Camera fits the vortex geometry within the rotor/wake crop. Fixed Q/omega^2 thresholds expose threshold sensitivity; colors span the displayed isosurfaces and are fixed across phases and thresholds. Cells are interpolated to points for isosurfaces. Vortices are not acoustic source strength.",
                                    camera=camera, field=("POINTS", "vorticity_magnitude", "Vorticity magnitude [1/s]"), limits=color_range, time_s=t))
                    finally:
                        pvs.Delete(contour)
                        pvs.Delete(point_data)
                        pvs.Delete(region)
                _pvvis_attempt(result, run_dir, f"Vortex views t={t:g}", vortex_views)
        _pvvis_attempt(result, run_dir, "Blade wall and section details", lambda: _pvvis_wall(
            marker, selected[-1][0], result, settings, run_dir, view, units, base=base))
    finally:
        for proxy in reversed(proxies):
            pvs.Delete(proxy)
        pvs.Delete(reader)


def _pvvis_wall(marker, t, result, settings, run_dir, view, units, base=None):
    wall = pvs.OpenFOAMReader(FileName=str(marker))
    try:
        wall.UpdatePipelineInformation()
        regions = [name for name in wall.MeshRegions.Available if name == "propeller" or name.endswith("/propeller")]
        if not regions:
            raise ValueError("propeller patch unavailable in OpenFOAM reader")
        wall.MeshRegions = regions
        wall.Createcelltopointfiltereddata = 0
        wall.CellArrays = [name for name in ("p", "yPlus", "wallShearStress") if name in wall.CellArrays.Available]
        wall.UpdatePipeline(t)
        view.ViewTime = t
        wall_fields = [("p", f"{units[1]} [{units[0]}]"), ("yPlus", "Wall y+ [-]")]
        if "wallShearStress" in wall.CellArrays.Available:
            shear_unit = _pvvis_pressure_units(Path(settings["case_path"]), "wallShearStress")[0]
            wall_fields.append(("wallShearStress", f"Wall shear magnitude [{shear_unit}]"))
        else:
            result["warnings"].append("wallShearStress unavailable; blade shear maps omitted")
        for name, label in wall_fields:
            for camera in ("front", "back", "oblique"):
                _pvvis_attempt(result, run_dir, f"Blade {name}, {camera}", lambda: _pvvis_render(
                    wall, result, settings, run_dir, view, f"Blade wall - {name} ({camera})",
                    "Inspect local loading or wall-treatment coverage, especially blade tips and roots. Colors show native patch cell values; no surface smoothing is applied. Assess y+ against the chosen turbulence model and wall treatment.",
                    camera=camera, field=("CELLS", name, label), time_s=t))
        for camera in ("front", "oblique"):
            _pvvis_render(wall, result, settings, run_dir, view, f"Blade surface mesh ({camera})",
                          "Inspect blade surface panel density near edges, root and tip. This surface view cannot measure prism-layer thickness; consult mesh sections and checkMesh results.",
                          camera=camera, time_s=t, edges=True)
        if base is not None:
            _pvvis_attempt(result, run_dir, "Blade cross-sections", lambda: _pvvis_blade_detail(
                base, wall, t, result, settings, run_dir, view, units))
    finally:
        pvs.Delete(wall)


def _pvvis_main(settings_path):
    pvs._DisableFirstRenderCameraReset()
    settings_path = Path(settings_path)
    run_dir = settings_path.parent
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    result = {"status": "running", "views": [], "warnings": [], "paraview_version": str(pvs.GetParaViewVersion())}
    _pvvis_save(result, run_dir)
    view = pvs.CreateView("RenderView")
    view.ViewSize = settings["image_resolution"]
    view.UseColorPaletteForBackground = 0
    view.Background = [1.0, 1.0, 1.0]
    view.BackgroundColorMode = "Single Color"
    view.OrientationAxesVisibility = 1
    view.OrientationAxesLabelColor = [0.1, 0.1, 0.1]
    view.CenterAxesVisibility = 0
    view.AxesGrid.Visibility = 1
    view.AxesGrid.ShowGrid = 0
    view.AxesGrid.AxesToLabel = 7  # MIN_X | MIN_Y | MIN_Z: label one side only.
    view.AxesGrid.GridColor = [0.55, 0.59, 0.62]
    for axis in "XYZ":
        setattr(view.AxesGrid, axis + "Title", axis.lower() + " [m]")
        setattr(view.AxesGrid, axis + "TitleColor", [0.15, 0.18, 0.21])
        setattr(view.AxesGrid, axis + "LabelColor", [0.15, 0.18, 0.21])
        setattr(view.AxesGrid, axis + "TitleFontSize", round(settings["image_resolution"][0] * .014))
        setattr(view.AxesGrid, axis + "LabelFontSize", round(settings["image_resolution"][0] * .012))
        setattr(view.AxesGrid, axis + "AxisPrecision", 3)
    units = _pvvis_pressure_units(Path(settings["case_path"]))
    result["pressure_units"] = {"unit": units[0], "label": units[1]}
    if units[0] in {"units unverified", "unknown dimensions"}:
        result["warnings"].append("Pressure dimensions could not be verified; pressure figures explicitly retain unverified units")
    if settings["acoustic_surface"] is not None:
        _pvvis_attempt(result, run_dir, "Acoustic-surface atlas", lambda: _pvvis_surface(result, settings, run_dir, view, units))
    _pvvis_attempt(result, run_dir, "Volume atlas", lambda: _pvvis_volume(result, settings, run_dir, view, units))
    result["resolved_diameter_m"] = settings["diameter_m"]
    result["status"] = "partial" if result["warnings"] else "complete"
    if not result["views"]:
        result["status"] = "failed"
    _pvvis_save(result, run_dir)
    pvs.Delete(view)
