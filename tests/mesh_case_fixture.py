"""Small field-free OpenFOAM mesh with a blade-shaped internal boundary."""
from pathlib import Path


def write_mesh_case(root):
    root = Path(root)
    mesh = root / "constant" / "polyMesh"
    mesh.mkdir(parents=True)
    system = root / "system"
    system.mkdir()

    def header(kind, name):
        return f"FoamFile\n{{ version 2.0; format ascii; class {kind}; object {name}; }}\n"

    (system / "controlDict").write_text(header("dictionary", "controlDict") +
        "application simpleFoam; startFrom startTime; startTime 0; stopAt endTime; "
        "endTime 1; deltaT 1; writeControl timeStep; writeInterval 1;\n")
    nx, ny, nz = 12, 8, 8
    points = [(2*i/nx-1, 1.2*j/ny-.6, k/nz-.5)
              for k in range(nz+1) for j in range(ny+1) for i in range(nx+1)]

    def vertex(i, j, k):
        return (k*(ny+1)+j)*(nx+1)+i

    faces = {}
    cell = 0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if 2 <= i < 10 and 3 <= j < 5 and 3 <= k < 5:
                    continue
                a,b,c,d,e,f,g,h = [vertex(i+di,j+dj,k+dk) for di,dj,dk in
                    ((0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1))]
                for face in ((a,d,c,b),(e,f,g,h),(a,b,f,e),(d,h,g,c),(a,e,h,d),(b,c,g,f)):
                    key = tuple(sorted(face))
                    if key in faces:
                        faces[key][2] = cell
                    else:
                        faces[key] = [face, cell, None]
                cell += 1
    internal, outer, blade = [], [], []
    for face, owner, neighbour in faces.values():
        entry = (face, owner, neighbour)
        if neighbour is not None:
            internal.append(entry)
        elif any(all(abs(points[p][axis]-edge) < 1e-9 for p in face)
                 for axis, ends in enumerate(((-1,1),(-.6,.6),(-.5,.5))) for edge in ends):
            outer.append(entry)
        else:
            blade.append(entry)
    ordered = internal + outer + blade

    def write_list(name, kind, values):
        (mesh / name).write_text(header(kind, name) + str(len(values)) + "\n(\n" + "\n".join(values) + "\n)\n")

    write_list("points", "vectorField", ["("+" ".join(map(str,p))+")" for p in points])
    write_list("faces", "faceList", ["4("+" ".join(map(str,f))+ ")" for f,_,_ in ordered])
    write_list("owner", "labelList", [str(o) for _,o,_ in ordered])
    write_list("neighbour", "labelList", [str(n) for _,_,n in internal])
    write_list("boundary", "polyBoundaryMesh", [
        f"domain {{ type patch; nFaces {len(outer)}; startFace {len(internal)}; }}",
        f"propeller {{ type wall; nFaces {len(blade)}; startFace {len(internal)+len(outer)}; }}"])
