> For the full cfMesh rotation/solver/acoustics pipeline with `main.py --mesh-only`, see [README.md](README.md). This document describes the older independent standalone experiment.

# Standalone cfMesh propeller test — no boundary layers

This is the independent copy at
`C:\repos\acoustic-pipeline-cfmesh-standalone-test-20260911`.
The original `C:\repos\acoustic-pipeline` was only read/copied.
The copy includes the current working files, uncommitted changes, Git history,
and the populated acousticSolver submodule. The new workflow consists only of
`run_cfmesh_test.sh` and `cfmesh_test/`; it never imports or invokes the main pipeline.

## Run now in Ubuntu / WSL

```bash
cd /mnt/c/repos/acoustic-pipeline-cfmesh-standalone-test-20260911
bash run_cfmesh_test.sh
```

The script loads your installed OpenFOAM 2512 environment in its own process.
It uses that installation's cfMesh utilities and checkMesh together. There is
no installation step, conda dependency, Docker invocation, or MPI decomposition.
The default is four shared-memory threads unless OMP_NUM_THREADS is already set:

```bash
OMP_NUM_THREADS=4 bash run_cfmesh_test.sh
```

Every run creates a fresh `cfmesh_test/runs/<timestamp>-<pid>/` case.
It prints each command and streams the complete stdout/stderr of each command
to the terminal, a separate stage log, and a complete transcript alongside the
case (`<timestamp>-<pid>.log`). Previous cases are never removed or overwritten.
Ctrl+C interrupts the run; the partial case and logs remain available.

The latest successfully prepared case path is saved in
`cfmesh_test/latest-case.txt`. This means “latest prepared”, not “mesh passed”.
Look for the final PASS message and `Mesh OK.` in `log.checkMesh`.
Failures retain the mesh for inspection and make the runner exit nonzero,
including checkMesh quality failures that return a zero process exit status.

To prepare the geometry and dictionaries without meshing:

```bash
bash run_cfmesh_test.sh --prepare-only
```

## Geometry and initial resolution

The user-selected input is copied byte for byte from:

```text
C:\Users\jonas\Downloads\APC10x7E 1\APC10x7E\fused.stl
```

The runner reads only the duplicate's `cfmesh_test/input/fused.stl`.
SHA-256: `20d939291510d0f572fd8993e9e599db3a8414f8b81d4220ac3ba93123e22161`.

The STL has 520,368 triangles and 260,186 unique vertices, one connected closed
shell, and consistent winding. Its span is 0.254000008 m; it is already in metres.
It is not scaled, rotated, recentered, healed, or decimated. These checks do
not establish absence of self-intersections or sufficient mesh resolution.

The generated fluid domain is a cube enclosing the propeller, with side length
three times its measured maximum span: approximately 0.762 m. It is centred on
the STL bounding-box centre. Its bounds, in metres, are approximately:

| Axis | Minimum | Maximum |
|---|---:|---:|
| X | -0.381000 | 0.381000 |
| Y | -0.379380 | 0.382620 |
| Z | -0.381000 | 0.381000 |

Only the box and propeller are meshing boundaries. No rotating interface,
AMI/NCC, acoustic sphere, solver, or inherited case template is involved.

Edit `cfmesh_test/settings.json` for subsequent fresh tests:

| Setting | Initial value |
|---|---:|
| STL scale | 1 |
| Domain side / propeller span | 3 |
| Background maximum cell size | 0.020 m |
| Propeller surface cell size | 0.00125 m |
| Surface refinement thickness | 0.005 m |

These are initial geometric trial settings, not a production mesh or a
domain-independence result. Requested cell sizes are subject to cfMesh's
octree refinement. No boundary-layer thickness or y+ target is imposed.
Inspect the thin trailing edge and tips before increasing overall mesh density.

`prepare_case.py` uses only Python's standard library. It writes a multi-patch
`constant/triSurface/domain.ftr`: an indexed triangulated format supported by
cfMesh. Exactly coincident STL vertices are welded; propeller triangle winding
is reversed for outward fluid normals. Six separately named box faces preserve
sharp box edges during meshing. They are merged afterward into `farfield`
(type `patch`); the propeller stays `propellerSurface` (type `wall`).
Each case includes a `geometry.json` provenance and bounds report.

## Why the no-layer sequence is explicit

The [cfMesh user guide](https://cfmesh.com/wp-content/uploads/2015/09/User_Guide-cfMesh_v1.1.pdf)
describes the Cartesian workflow's default base layer. Inspection of the locally
available cfMesh source and installed OpenFOAM 2512 library confirmed that
`cartesianMeshGenerator::generateBoundaryLayers()` unconditionally calls
`addLayerForAllPatches()`. Setting `nLayers 0` alone does not switch that off.

The generated `system/meshDict` therefore contains:

```text
surfaceFile "constant/triSurface/domain.ftr";
maxCellSize 0.02;

localRefinement
{
    propellerSurface
    {
        cellSize 0.00125;
        refinementThickness 0.005;
    }
}

workflowControls
{
    stopAfter edgeExtraction;
}
```

The actual sequence is:

```bash
cartesianMesh
improveMeshQuality -nLoops 2 -nIterations 20 -nSurfaceIterations 0
createPatch -overwrite
checkMesh -allGeometry -allTopology
```

The first command must report `Stopping after step edgeExtraction`.
This is an intentional successful stop after surface fitting and before layer
insertion. The runner checks that message. cfMesh's separate volume smoother
then optimizes cells with zero surface-smoothing iterations; createPatch only
combines the six farfield faces. The normal post-layer cartesianMesh
optimization/refinement stages are not run. This is a deliberate no-layer
baseline, with a separate volume-smoothing pass.

Do not remove stopAfter or add layer settings for this first test.
A later layered experiment needs its own complete Cartesian workflow and
should be compared separately.

## View in ParaView

After the run, in Ubuntu/WSL:

```bash
cd /mnt/c/repos/acoustic-pipeline-cfmesh-standalone-test-20260911
CASE="$(cat cfmesh_test/latest-case.txt)"
cd "$CASE"
```

Your detected installation is **Windows ParaView 6.1.0**. Launch it directly:

```bash
"/mnt/c/Program Files/ParaView 6.1.0/bin/paraview.exe" "$(wslpath -w "$CASE/cfmesh_test.foam")"
```

Alternatively, display the Windows filename and use ParaView's File > Open:

```bash
touch cfmesh_test.foam
wslpath -w "$CASE/cfmesh_test.foam"
```

The empty .foam marker is already created automatically; ParaView reads the
mesh from the neighbouring OpenFOAM case directories.

If you have Linux ParaView installed in WSL, the OpenFOAM launcher is:

```bash
source /usr/lib/openfoam/openfoam2512/etc/bashrc
cd "$CASE"
paraFoam -builtin
```

Plain `paraFoam` is also possible if its OpenFOAM reader plugin is available.
The installed launcher accepts `-builtin` (alias `-vtk`).
At setup time no Linux `paraview` executable was on PATH, so use the Windows
command above on this machine.

In ParaView:

1. Open `cfmesh_test.foam`; choose **OpenFOAM Reader** if prompted.
2. Select **Skip Zero Time = off** if present. This is a mesh-only case at time 0.
3. In **Mesh Regions**, first select only **propellerSurface**, then **Apply**.
   If the name is grouped under patches, expand that group.
4. Set representation to **Surface With Edges**, colour by **Solid Color**,
   and press **Reset Camera**. Inspect the leading/trailing edges, blade tips,
   and blade-to-hub transition. No pressure or velocity fields are expected.
5. To inspect cells around the blade, enable **internalMesh**, disable
   **farfield**, and Apply. Use **Filters > Clip** or **Slice**, choose a plane
   through a blade section, and display the result as **Surface With Edges**.
   Move the plane through the root, midspan, and tip; looking at the outer
   box alone hides the propeller and its local refinement.
6. For a geometry overlay, also open
   `cfmesh_test/input/fused.stl` from the duplicate and use a contrasting colour
   or Wireframe. It is already in the same coordinates and units.

## Validation performed

- Supplied STL closure, winding, connectivity, scale, and input-copy hash checked.
- Actual propeller geometry/dictionaries generated using --prepare-only.
- Tiny external hollow-box mesh: 3,136 cells, one fluid region, expected volume
  0.026 m3, two final named patches, and **Mesh OK.**
- The complete runner was also exercised on that tiny geometry. All stage output
  reached both the terminal and saved logs; a failed checkMesh summary was
  correctly rejected.
- The supplied propeller's volume meshing has **not** been run. Run the command
  at the top to test its actual mesh quality.
- Repository file-copy comparison and test logs are under
  `cfmesh_test/validation/`.
