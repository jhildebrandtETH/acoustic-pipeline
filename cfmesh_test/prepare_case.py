#!/usr/bin/env python3
"""Prepare an isolated external-flow case; standard-library Python only."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parent


def read_stl(path, scale):
    data = path.read_bytes()
    count = struct.unpack_from("<I", data, 80)[0] if len(data) >= 84 else 0
    if len(data) == 84 + 50 * count:
        triangles = (record[3:12] for record in struct.iter_unpack("<12fH", data[84:]))
    else:
        ascii_coords = []
        for line in data.decode("ascii").splitlines():
            fields = line.split()
            if fields and fields[0].lower() == "vertex":
                ascii_coords.extend(map(float, fields[1:]))
        if not ascii_coords or len(ascii_coords) % 9:
            raise ValueError("Invalid or empty STL")
        triangles = (ascii_coords[i:i+9] for i in range(0, len(ascii_coords), 9))
    vertices, faces, indices = [], [], {}
    edges = defaultdict(lambda: [0, 0])
    volume = 0.0
    for coords in triangles:
        face = []
        for i in (0, 3, 6):
            point = tuple(scale * x for x in coords[i:i+3])
            if len(point) != 3 or not all(map(math.isfinite, point)):
                raise ValueError("Invalid STL coordinate")
            if point not in indices:
                indices[point] = len(vertices)
                vertices.append(point)
            face.append(indices[point])
        a, b, c = (vertices[i] for i in face)
        cross = tuple((b[(i+1)%3]-a[(i+1)%3])*(c[(i+2)%3]-a[(i+2)%3])
                      -(b[(i+2)%3]-a[(i+2)%3])*(c[(i+1)%3]-a[(i+1)%3])
                      for i in range(3))
        if len(set(face)) != 3 or sum(x*x for x in cross) == 0:
            raise ValueError("STL contains a degenerate triangle; no repair attempted")
        volume += sum(a[i] * (b[(i+1)%3]*c[(i+2)%3]-b[(i+2)%3]*c[(i+1)%3])
                      for i in range(3)) / 6
        for u, v in zip(face, face[1:]+face[:1]):
            edge = edges[min(u,v), max(u,v)]
            edge[0] += 1
            edge[1] += 1 if u < v else -1
        faces.append(face)
    if not faces or any(n != 2 or winding != 0 for n, winding in edges.values()):
        raise ValueError("STL is not closed with consistent winding; no repair attempted")
    # The selected input is one solid. Reject ambiguous multi-shell inputs.
    neighbours = defaultdict(list)
    for u, v in edges:
        neighbours[u].append(v)
        neighbours[v].append(u)
    seen = {0}
    stack = [0]
    while stack:
        for v in neighbours[stack.pop()]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    if len(seen) != len(vertices) or abs(volume) < 1e-18:
        raise ValueError("Expected one closed solid of nonzero volume")
    return vertices, faces, volume, hashlib.sha256(data).hexdigest()


def header(name):
    return ('FoamFile\n{\n    version 2.0;\n    format ascii;\n'
            '    class dictionary;\n    object ' + name + ';\n}\n\n')


def prepare(case, stl=None, settings=None):
    case = case.resolve()
    # All generated output must stay within this duplicate's test folder.
    if ROOT not in case.parents or case.exists():
        raise ValueError("Case must be a NEW directory inside cfmesh_test")
    settings = settings or json.loads((ROOT / "settings.json").read_text())
    stl = stl or ROOT / "input/fused.stl"
    for key, value in settings.items():
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid positive setting: {key}")
    if settings["domain_span_in_propeller_diameters"] <= 1:
        raise ValueError("Domain span must exceed the propeller span")
    if settings["propeller_cell_size"] > settings["max_cell_size"]:
        raise ValueError("Propeller cells must not exceed background cells")
    vertices, faces, volume, sha = read_stl(stl, settings["scale"])
    low = [min(v[i] for v in vertices) for i in range(3)]
    high = [max(v[i] for v in vertices) for i in range(3)]
    diameter = max(b-a for a,b in zip(low,high))
    if not 0.02 < diameter < 2:
        raise ValueError(f"Unexpected propeller span {diameter:g} m; check scale")
    centre = [(a+b)/2 for a,b in zip(low,high)]
    span = settings["domain_span_in_propeller_diameters"] * diameter
    boxlow = [x-span/2 for x in centre]
    boxhigh = [x+span/2 for x in centre]
    nprop = len(vertices)
    # Corner index: x + 2*y + 4*z.
    vertices += [tuple((boxhigh[i] if j & (1 << i) else boxlow[i])
                       for i in range(3)) for j in range(8)]
    quads = [(0,4,6,2), (1,3,7,5), (0,1,5,4),
             (2,6,7,3), (0,2,3,1), (4,5,7,6)]
    names = ["farfield_xmin", "farfield_xmax", "farfield_ymin",
             "farfield_ymax", "farfield_zmin", "farfield_zmax"]
    surf = case / "constant/triSurface"
    surf.mkdir(parents=True)
    (case / "system").mkdir()
    (case / "0").mkdir()
    # FTR preserves patches and exact STL coordinates, without extra dependencies.
    # Inner-shell normals point OUT OF FLUID (into the propeller).
    with (surf / "domain.ftr").open("w") as out:
        out.write("7\n(\npropellerSurface wall\n")
        out.writelines(name+" patch\n" for name in names)
        out.write(")\n"+str(len(vertices))+"\n(\n")
        out.writelines("("+ " ".join(format(x, ".17g") for x in v)+")\n"
                       for v in vertices)
        out.write(")\n"+str(len(faces)+12)+"\n(\n")
        for face in faces:
            f = face[::-1] if volume > 0 else face
            out.write(f"(({f[0]} {f[1]} {f[2]}) 0)\n")
        for patch, q in enumerate(quads, 1):
            for tri in ((q[0],q[1],q[2]), (q[0],q[2],q[3])):
                out.write("(("+ " ".join(str(nprop+i) for i in tri)+f") {patch})\n")
        out.write(")\n")
    mesh_dict = header("meshDict") + f'''surfaceFile "constant/triSurface/domain.ftr";
maxCellSize {settings["max_cell_size"]:.12g};
localRefinement
{{
    propellerSurface
    {{
        cellSize {settings["propeller_cell_size"]:.12g};
        refinementThickness {settings["refinement_thickness"]:.12g};
    }}
}}
// This build always adds one base layer during boundaryLayerGeneration.
// Stop before that stage. nLayers 0 alone DOES NOT disable the base layer.
workflowControls
{{
    stopAfter edgeExtraction;
}}
'''
    (case / "system/meshDict").write_text(mesh_dict)
    (case / "system/controlDict").write_text(header("controlDict")+'''application cartesianMesh;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime 1;
deltaT 1;
writeControl timeStep;
writeInterval 1;
writeFormat ascii;
writePrecision 12;
writeCompression off;
runTimeModifiable false;
''')
    (case / "system/fvSchemes").write_text(header("fvSchemes")+'''ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
''')
    (case / "system/fvSolution").write_text(header("fvSolution")+"solvers {}\n")
    (case / "system/createPatchDict").write_text(header("createPatchDict")+'''pointSync false;
patches
(
    {
        name farfield;
        patchInfo { type patch; }
        constructFrom patches;
        patches ("farfield_.*");
    }
);
''')
    report = {
        "input": str(stl), "input_sha256": sha,
        "triangles": len(faces), "unique_vertices": nprop,
        "closed": True, "consistent_winding": True, "components": 1,
        "bounds_m": [low,high], "span_m": diameter,
        "solid_volume_m3": abs(volume), "domain_bounds_m": [boxlow,boxhigh],
        "expected_fluid_volume_m3": span**3-abs(volume),
        "input_winding_reversed_for_fluid": volume > 0, "settings": settings,
        "note": "Exact coordinate welding only; no healing, decimation, rotation or recentering."
    }
    (case / "geometry.json").write_text(json.dumps(report, indent=2)+"\n")
    (case / "cfmesh_test.foam").touch()
    print(json.dumps(report, indent=2), flush=True)
    print(f"Prepared: {case}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    args = parser.parse_args()
    prepare(args.case)
