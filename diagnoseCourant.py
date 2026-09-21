"""Inspect saved OpenFOAM Courant hotspots without changing the simulation."""
import argparse
import csv
import gzip
import json
import math
import re
from datetime import datetime
from pathlib import Path


def log_history(path):
    """Keep independent sequences: solvers print Co before/after Time differently."""
    number = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
    steps, courant = [], []
    with Path(path).open(errors="replace") as stream:
        for line in stream:
            match = re.match(rf"\s*deltaT\s*=\s*({number})", line)
            if match and math.isfinite(float(match[1])) and float(match[1]) > 0:
                steps.append(float(match[1]))
            match = re.match(rf"\s*Courant Number mean:\s*{number}\s+max:\s*({number})", line)
            if match and math.isfinite(float(match[1])):
                courant.append(float(match[1]))
    tail = steps[-20:]
    return dict(delta_t_tail=tail, max_co_tail=courant[-20:],
                delta_t_plateau=(len(tail) == 20 and max(tail) / min(tail) <= 1.05),
                note="Independent log sequences, not paired timestamps. Plateau means last 20 deltaT values span <=5%; fixed-step logs may omit deltaT. Log may extend beyond selected snapshot.")


def choose_time(times, requested):
    times = sorted(t for t in times if math.isfinite(t) and t > 0)
    if not times:
        raise ValueError("No saved simulation times above zero. Write a complete snapshot first.")
    if requested == "latest":
        return times[-1]
    requested = float(requested)
    matches = [t for t in times if math.isclose(t, requested, rel_tol=1e-10, abs_tol=1e-14)]
    if not matches:
        raise ValueError(f"Time {requested} is not saved; available range: {times[0]}..{times[-1]}")
    return matches[0]


def leaves(block, prefix=""):
    if block.IsA("vtkMultiBlockDataSet"):
        import vtk
        for index in range(block.GetNumberOfBlocks()):
            child = block.GetBlock(index)
            if child is not None:
                name = block.GetMetaData(index).Get(vtk.vtkCompositeDataSet.NAME()) or str(index)
                yield from leaves(child, prefix + "/" + name)
    else:
        yield prefix, block


def read_ascii_internal(path, count, components):
    """Read original cell fields without parsing NCC virtual boundary faces."""
    import numpy as np
    if not path.exists() and path.with_suffix(path.suffix + ".gz").exists():
        path = path.with_suffix(path.suffix + ".gz")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", errors="strict") as stream:
        text = stream.read()
    if not re.search(r"\bformat\s+ascii\s*;", text):
        raise ValueError(f"{path}: diagnostic field reader requires ASCII fields (gzip supported). Export an ASCII copy with OpenFOAM first.")
    match = re.search(r"\binternalField\s+nonuniform\s+List<\w+>\s+(\d+)\s*\((.*?)\)\s*;", text, re.S)
    if match:
        values = np.fromstring(match[2].replace("(", " ").replace(")", " "), sep=" ")
        if int(match[1]) != count or values.size != count * components:
            raise ValueError(f"{path}: internal field length does not match mesh.")
        return values if components == 1 else values.reshape(count, components)
    match = re.search(r"\binternalField\s+uniform\s+([^;]+);", text)
    if not match:
        raise ValueError(f"{path}: cannot parse internalField.")
    value = np.fromstring(match[1].replace("(", " ").replace(")", " "), sep=" ")
    if value.size != components:
        raise ValueError(f"{path}: incorrect component count.")
    return np.full(count, value[0]) if components == 1 else np.tile(value, (count, 1))


def load_case(case, requested, patch_name="propeller", co_field=None):
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
    reader = vtk.vtkOpenFOAMReader()
    reader_errors = []
    reader.AddObserver(vtk.vtkCommand.ErrorEvent, lambda *_: reader_errors.append(True))
    # VTK uses this filename to locate the case; no marker file is created.
    reader.SetFileName(str(case / "diagnostics.foam"))
    reader.SetCreateCellToPoint(False)
    if hasattr(reader, "SetDecomposePolyhedra"):
        reader.SetDecomposePolyhedra(False)  # New VTK versions always preserve polyhedra.
    reader.SetReadZones(False)
    if hasattr(reader, "SetUse64BitFloats"):
        reader.SetUse64BitFloats(True)
    reader.SetListTimeStepsByControlDict(False)
    reader.UpdateInformation()
    if reader_errors:
        raise ValueError("VTK could not read the OpenFOAM case metadata; check format/version compatibility.")
    values = reader.GetTimeValues()
    selected = choose_time(vtk_to_numpy(values).tolist() if values else [], requested)
    # NCC patches may contain virtual faces unsupported by VTK. Only read the
    # internal mesh and requested physical surface; neither needs NCC patches.
    reader.DisableAllPatchArrays()
    for index in range(reader.GetNumberOfPatchArrays()):
        name = reader.GetPatchArrayName(index)
        if name == "internalMesh" or name.split("/")[-1] == patch_name:
            reader.SetPatchArrayStatus(name, 1)
    reader.DisableAllCellArrays()
    reader.UpdateTimeStep(selected)
    if reader_errors:
        raise ValueError("VTK reported a mesh/field read error; refusing partial results.")
    blocks = list(leaves(reader.GetOutput()))
    internal = [(name, obj) for name, obj in blocks if name.endswith("/internalMesh")]
    if len(internal) != 1:
        raise ValueError("Expected one reconstructed internalMesh; multi-region/decomposed cases must be reconstructed/selected separately.")
    mesh = vtk.vtkUnstructuredGrid()
    mesh.DeepCopy(internal[0][1])
    if not mesh.GetNumberOfCells():
        raise ValueError("The selected internal mesh is empty.")
    time_directory = next(p for p in case.iterdir() if p.is_dir() and re.fullmatch(r"\d+(?:\.\d*)?(?:[eE][-+]?\d+)?", p.name) and float(p.name) == selected)
    candidates = [co_field] if co_field else ["Co", "CourantNo", "CourantNumber"]
    field = next((n for n in candidates if (time_directory / n).exists() or (time_directory / (n + ".gz")).exists()), None)
    if not field:
        raise ValueError("No saved cell Courant field; enable CourantNo output before running.")
    for name, components in [(field, 1), ("U", 3)]:
        values = read_ascii_internal(time_directory / name, mesh.GetNumberOfCells(), components)
        array = numpy_to_vtk(values, deep=True)
        array.SetName(name)
        mesh.GetCellData().AddArray(array)
    return mesh, blocks, selected, reader


def analyse(mesh, blocks, args):
    import numpy as np
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
    data = mesh.GetCellData()
    names = [data.GetArrayName(i) for i in range(data.GetNumberOfArrays())]
    candidates = [args.co_field] if args.co_field else ["Co", "CourantNo", "CourantNumber"]
    co_name = next((n for n in candidates if n in names), None)
    if co_name is None:
        raise ValueError(f"No saved cell Courant field. Available: {names}. Enable CourantNo output before running, or use --co-field NAME. Co will not be guessed from U or the current controlDict deltaT.")
    co = vtk_to_numpy(data.GetArray(co_name))
    if co.ndim != 1 or len(co) != mesh.GetNumberOfCells() or not np.all(np.isfinite(co)) or np.any(co < 0):
        raise ValueError("Courant field must contain finite nonnegative scalars for every cell.")
    if data.GetArray("U") is None:
        raise ValueError("Saved cell velocity U is required.")
    velocity = vtk_to_numpy(data.GetArray("U"))
    if velocity.shape != (len(co), 3) or not np.all(np.isfinite(velocity)):
        raise ValueError("U must contain finite 3-component cell velocities.")
    speed = np.linalg.norm(velocity, axis=1)
    size = vtk.vtkCellSizeFilter()
    size.SetInputData(mesh)
    size.SetComputeArea(False)
    size.SetComputeLength(False)
    size.SetComputeVertexCount(False)
    size.SetComputeVolume(True)
    size.Update()
    volume = vtk_to_numpy(size.GetOutput().GetCellData().GetArray("Volume"))
    centres_filter = vtk.vtkCellCenters()
    centres_filter.SetInputData(mesh)
    centres_filter.Update()
    centres = vtk_to_numpy(centres_filter.GetOutput().GetPoints().GetData())
    patch = next((obj for name, obj in blocks if name.split("/")[-1] == args.patch), None)
    locator = None
    radius = args.tip_radius
    axis = "xyz".index(args.axis)
    origin = np.asarray(args.origin)
    radial_axes = [i for i in range(3) if i != axis]
    if patch is not None and patch.GetNumberOfCells():
        locator = vtk.vtkStaticCellLocator()
        locator.SetDataSet(patch)
        locator.BuildLocator()
        if radius is None:
            points = vtk_to_numpy(patch.GetPoints().GetData())
            radius = float(np.linalg.norm((points - origin)[:, radial_axes], axis=1).max())
    top = np.argsort(co, kind="stable")[-min(args.top, len(co)):][::-1]
    rows, selected = [], set(map(int, top))
    mesh.BuildLinks()
    for rank, cell_id in enumerate(top, 1):
        cell_id = int(cell_id)
        cell = mesh.GetCell(cell_id)
        lengths = []
        for e in range(cell.GetNumberOfEdges()):
            edge = cell.GetEdge(e)
            lengths.append(float(np.linalg.norm(np.asarray(edge.GetPoints().GetPoint(0)) - edge.GetPoints().GetPoint(1))))
        neighbours = set()
        for f in range(cell.GetNumberOfFaces()):
            ids = vtk.vtkIdList()
            mesh.GetCellNeighbors(cell_id, cell.GetFace(f).GetPointIds(), ids)
            neighbours.update(ids.GetId(i) for i in range(ids.GetNumberOfIds()))
        selected.update(neighbours)
        distance = None
        if locator:
            closest, cid, subid, dist2 = [0., 0., 0.], vtk.reference(0), vtk.reference(0), vtk.reference(0.)
            locator.FindClosestPoint(centres[cell_id], closest, cid, subid, dist2)
            distance = math.sqrt(float(dist2))
        min_edge, max_edge = min(lengths, default=0), max(lengths, default=0)
        neighbour_speed = float(np.median(speed[list(neighbours)])) if neighbours else None
        radial = float(np.linalg.norm((centres[cell_id] - origin)[radial_axes]))
        rows.append(dict(rank=rank, cell_id=cell_id, Co=float(co[cell_id]),
                         x=float(centres[cell_id, 0]), y=float(centres[cell_id, 1]), z=float(centres[cell_id, 2]),
                         volume_m3=float(volume[cell_id]), equivalent_size_m=float(np.cbrt(volume[cell_id])),
                         min_edge_m=min_edge, max_edge_m=max_edge,
                         edge_ratio=max_edge / min_edge if min_edge > 0 else None,
                         U_magnitude=float(speed[cell_id]),
                         neighbour_median_U=neighbour_speed,
                         U_to_neighbour_ratio=float(speed[cell_id]) / neighbour_speed if neighbour_speed and neighbour_speed > 0 else None,
                         propeller_distance_m=distance, radius_m=radial,
                         radius_over_tip=radial / radius if radius else None))
    def add(name, values):
        array = numpy_to_vtk(np.asarray(values), deep=True)
        array.SetName(name)
        data.AddArray(array)
    add("originalCellId", np.arange(len(co), dtype=np.int64))
    add("diagnosticVolume", volume)
    add("diagnosticSpeed", speed)
    rank_array = np.zeros(len(co), dtype=np.int32)
    rank_array[top] = np.arange(1, len(top) + 1)
    add("hotspotRank", rank_array)
    ids = vtk.vtkIdList()
    for index in sorted(selected):
        ids.InsertNextId(index)
    extract = vtk.vtkExtractCells()
    extract.SetInputData(mesh)
    extract.SetCellList(ids)
    extract.Update()
    summary = dict(cells=len(co), co_field=co_name, max_co=float(co.max()),
                   cells_above_threshold=int(np.count_nonzero(co >= args.threshold)), threshold=args.threshold,
                   co_percentiles=dict(zip(["p50", "p95", "p99", "p99.9"], map(float, np.percentile(co, [50, 95, 99, 99.9])))),
                   speed_percentiles=dict(zip(["p50", "p95", "p99"], map(float, np.percentile(speed, [50, 95, 99])))),
                   invalid_vtk_volume_count=int(np.count_nonzero(volume <= 0)),
                   tip_radius_m=radius, propeller_patch_found=locator is not None,
                   same_flux_timestep_multiplier=args.target_co / float(co.max()) if co.max() > 0 else None)
    return summary, rows, extract.GetOutput()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--time", default="latest", help="Exact saved time or latest")
    parser.add_argument("--co-field")
    parser.add_argument("--threshold", type=float, default=4.0)
    parser.add_argument("--target-co", type=float, default=1.0)
    parser.add_argument("--top", type=int, default=100)
    parser.add_argument("--patch", default="propeller")
    parser.add_argument("--axis", choices=list("xyz"), default="y")
    parser.add_argument("--origin", nargs=3, type=float, default=[0., 0., 0.])
    parser.add_argument("--tip-radius", type=float)
    parser.add_argument("--log", type=Path, help="Solver log for independent time-step history")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.top < 1 or any(not math.isfinite(v) or v <= 0 for v in [args.threshold, args.target_co] + ([args.tip_radius] if args.tip_radius is not None else [])) or not all(map(math.isfinite, args.origin)):
        parser.error("Counts, thresholds and radius must be positive; coordinates must be finite.")
    try:
        import vtk
        case = args.case.resolve()
        if not (case / "constant" / "polyMesh").is_dir():
            raise ValueError("Expected case/constant/polyMesh; supply the reconstructed case root.")
        numeric = [p for p in case.iterdir() if p.is_dir() and re.fullmatch(r"\d+(?:\.\d*)?(?:[eE][-+]?\d+)?", p.name)]
        if args.time == "latest" and (case / "processor0").is_dir():
            proc_times = [float(p.name) for p in (case / "processor0").iterdir() if p.is_dir() and re.fullmatch(r"\d+(?:\.\d*)?(?:[eE][-+]?\d+)?", p.name)]
            if proc_times and max(proc_times) > max([float(p.name) for p in numeric], default=0):
                raise ValueError("Processor output is newer than reconstructed output. Reconstruct the desired time first, or choose an explicit reconstructed --time.")
        print("Reading saved mesh and fields...", flush=True)
        mesh, blocks, time_value, reader = load_case(case, args.time, args.patch, args.co_field)
        print(f"Analysing {mesh.GetNumberOfCells():,} cells at time {time_value:g}...", flush=True)
        summary, rows, subset = analyse(mesh, blocks, args)
        summary.update(case=str(case), time=time_value, vtk_version=vtk.vtkVersion.GetVTKVersion())
        summary["saved_context"] = {}
        for relative in ["log.checkMesh", "Parameters/cfmeshCommon.cpp", "Parameters/cfmeshFeatureDict", "Parameters/cfmeshGlobalDict", "Parameters/controlDict.cpp"]:
            path = case / relative
            if path.is_file():
                summary["saved_context"][relative] = path.read_text(errors="replace")[:40000]
        if args.log:
            summary["solver_history"] = log_history(args.log)
        output = args.output or case / "postProcessing" / "courantDiagnostics" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        output.mkdir(parents=True, exist_ok=False)
        with (output / "hotspots.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(str(output / "hotspots.vtu"))
        writer.SetInputData(subset)
        if writer.Write() != 1:
            raise ValueError("Could not write hotspot mesh.")
        worst = rows[0]
        report = f"""# Courant diagnostics

Case: {case}
Saved time: {time_value:g}; field: {summary['co_field']}

- Maximum Co: {summary['max_co']:.6g}
- Cells at/above {args.threshold:g}: {summary['cells_above_threshold']:,} / {summary['cells']:,}
- Worst cell ID: {worst['cell_id']}; centre: ({worst['x']:.8g}, {worst['y']:.8g}, {worst['z']:.8g}) m
- Worst-cell equivalent cube size: {worst['equivalent_size_m']:.6g} m; shortest edge: {worst['min_edge_m']:.6g} m
- Worst-cell speed: {worst['U_magnitude']:.6g} m/s; neighbour median: {worst['neighbour_median_U']}
- Same-flux time-step multiplier for Co {args.target_co:g}: {summary['same_flux_timestep_multiplier']}

Open hotspots.vtu in ParaView: hotspotRank > 0 selects ranked cells; rank 0 marks their immediate face neighbours. originalCellId is the reconstructed OpenFOAM cell index (polyhedron decomposition disabled). hotspots.csv contains detailed geometry and neighbourhood measurements.

Interpretation limits: edge_ratio is an edge-length ratio, not OpenFOAM aspect ratio. VTK volumes and parametric centres can differ from OpenFOAM for distorted polyhedra; use checkMesh for signed-volume validity. U magnitude is the saved velocity, not mesh-relative flux. Thin cells are not automatically the Co cause. Large velocity-to-neighbour ratios are leads, not proof of divergence. Radius and propeller distance locate the tip vicinity but do not identify trailing-edge membership or boundary-layer membership. Compare these cells with the surface and feature curves. A plateau near maxCo may be normal adaptive stepping; the script does not stop the solver or certify a mesh.
"""
        if args.log:
            report += "\nRecent time-step history:\n```json\n" + json.dumps(summary["solver_history"], indent=2) + "\n```\n"
        (output / "report.md").write_text(report, encoding="utf-8")
        print(f"Report: {output / 'report.md'}")
    except (ImportError, ValueError, OSError) as error:
        parser.exit(2, f"Diagnostic failed: {error}\nDependencies: python -m pip install numpy vtk\n")


if __name__ == "__main__":
    main()
