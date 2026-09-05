"""Tiny analytic rotating-surface and OpenFOAM fixture for real ParaView QA."""
import math
import sys
from pathlib import Path
import numpy as np

case = Path(sys.argv[1])
mesh = case / "constant" / "polyMesh"
mesh.mkdir(parents=True, exist_ok=True)
(case / "constant" / "triSurface").mkdir(exist_ok=True)
(case / "constant" / "triSurface" / "propeller.stl").write_text('''solid test_blade
facet normal 0 1 0
outer loop
vertex -0.12 0 -0.018
vertex 0.12 0 -0.018
vertex 0.12 0 0.018
endloop
endfacet
facet normal 0 1 0
outer loop
vertex -0.12 0 -0.018
vertex 0.12 0 0.018
vertex -0.12 0 0.018
endloop
endfacet
endsolid test_blade
''')
(case / "system").mkdir(exist_ok=True)
(case / "system" / "controlDict").write_text('FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }\napplication foamRun; startFrom startTime; startTime 0; stopAt endTime; endTime 0.2; deltaT 0.025; writeControl timeStep; writeInterval 1;\n')

def header(name, klass):
    return f'FoamFile {{ version 2.0; format ascii; class {klass}; object {name}; }}\n'

n = 8
axis = np.linspace(-0.35, 0.35, n+1)
points = np.array([[x,y,z] for z in axis for y in axis for x in axis])
def pid(x,y,z): return x + (n+1)*(y+(n+1)*z)
faces, owners, neighbors, seen, centers = [], [], [], {}, []
for z in range(n):
    for y in range(n):
        for x in range(n):
            owner = len(centers)
            centers.append([(axis[q]+axis[q+1])/2 for q in (x,y,z)])
            a,b,c,d,e,f,g,h = [pid(*xyz) for xyz in [(x,y,z),(x+1,y,z),(x+1,y+1,z),(x,y+1,z),(x,y,z+1),(x+1,y,z+1),(x+1,y+1,z+1),(x,y+1,z+1)]]
            for face in [(a,d,c,b),(e,f,g,h),(a,b,f,e),(d,h,g,c),(a,e,h,d),(b,c,g,f)]:
                key = tuple(sorted(face))
                if key in seen:
                    neighbors[seen[key]] = owner
                else:
                    seen[key] = len(faces)
                    faces.append(face); owners.append(owner); neighbors.append(-1)
order = [i for i,neigh in enumerate(neighbors) if neigh>=0] + [i for i,neigh in enumerate(neighbors) if neigh<0]
internal = sum(v>=0 for v in neighbors)
(mesh/'points').write_text(header('points','vectorField')+f'{len(points)}\n(\n'+'\n'.join('('+' '.join(map(str,p))+')' for p in points)+'\n)\n')
(mesh/'faces').write_text(header('faces','faceList')+f'{len(faces)}\n(\n'+'\n'.join('4('+' '.join(map(str,faces[i]))+')' for i in order)+'\n)\n')
(mesh/'owner').write_text(header('owner','labelList')+f'{len(faces)}\n(\n'+'\n'.join(str(owners[i]) for i in order)+'\n)\n')
(mesh/'neighbour').write_text(header('neighbour','labelList')+f'{internal}\n(\n'+'\n'.join(str(neighbors[i]) for i in order[:internal])+'\n)\n')
(mesh/'boundary').write_text(header('boundary','polyBoundaryMesh')+f'1\n(\npropeller {{ type wall; nFaces {len(faces)-internal}; startFace {internal}; }}\n)\n')
centers = np.array(centers)
omega = 2*math.pi*10
for index,t in enumerate(np.linspace(0, .2, 9)):
    directory = case / f'{t:.6f}'
    directory.mkdir(exist_ok=True)
    x,y,z = centers.T
    arrays = {
        'p': (101325 + 4*x + math.sin(omega*t)*z, '[0 2 -2 0 0 0 0]'),
        'U': (np.column_stack((omega*z, 2+np.sin(y*10), -omega*x)), '[0 1 -1 0 0 0 0]'),
        'k': (0.1+np.abs(x), '[0 2 -2 0 0 0 0]'),
        'Co': (np.full(len(x), .2), '[0 0 0 0 0 0 0]'),
        'yPlus': (1+np.abs(x)*100, '[0 0 0 0 0 0 0]'),
        'wallShearStress': (np.column_stack((x*x,y*y,z*z)), '[0 2 -2 0 0 0 0]'),
    }
    for name,(values,dims) in arrays.items():
        vector = values.ndim>1
        vals = '\n'.join('('+' '.join(map(str,v))+')' if vector else str(v) for v in values)
        typ = 'vector' if vector else 'scalar'
        wall = '(0 0 0)' if vector else '10'
        (directory/name).write_text(header(name,'volVectorField' if vector else 'volScalarField')+f'dimensions {dims};\ninternalField nonuniform List<{typ}>\n{len(values)}\n(\n{vals}\n);\nboundaryField {{ propeller {{ type fixedValue; value uniform {wall}; }} }}\n')
    surf = case/'postProcessing'/'writePatchFields'/f'{t:.6f}'
    surf.mkdir(parents=True,exist_ok=True)
    base = np.array([[-.12,0,-.018],[-.025,0,-.018],[-.025,0,.018],[-.12,0,.018],[.025,0,-.018],[.12,0,-.018],[.12,0,.018],[.025,0,.018]])
    ca,sa = math.cos(omega*t),math.sin(omega*t)
    rot=np.array([[ca,0,sa],[0,1,0],[-sa,0,ca]])
    xyz=base@rot.T
    vtk='# vtk DataFile Version 3.0\nAnalytic rotating panels\nASCII\nDATASET POLYDATA\nPOINTS 8 double\n'+'\n'.join(' '.join(map(str,p)) for p in xyz)+'\nPOLYGONS 2 10\n4 0 1 2 3\n4 4 5 6 7\nCELL_DATA 2\nSCALARS p double 1\nLOOKUP_TABLE default\n'+f'{101325+2*t}\n{101325+4*t}\nVECTORS U double\n1 0 0\n0 2 0\n'
    (surf/'propeller.vtk').write_text(vtk)
    enclosure = case/'postProcessing'/'writePermeableSurfaceFields'/f'{t:.6f}'
    enclosure.mkdir(parents=True,exist_ok=True)
    (enclosure/'permeableSurface.vtk').write_text(
        '# vtk DataFile Version 3.0\nStationary octahedron\nASCII\nDATASET POLYDATA\n'
        'POINTS 6 double\n.3 0 0\n-.3 0 0\n0 .3 0\n0 -.3 0\n0 0 .3\n0 0 -.3\n'
        'POLYGONS 8 32\n3 0 2 4\n3 2 1 4\n3 1 3 4\n3 3 0 4\n3 2 0 5\n3 1 2 5\n3 3 1 5\n3 0 3 5\n'
        'CELL_DATA 8\nSCALARS p double 1\nLOOKUP_TABLE default\n'
        + '\n'.join(str(101325+(i+1)*t) for i in range(8))
        + '\nVECTORS U double\n' + '1 0 0\n'*8
    )
print(case.resolve())
