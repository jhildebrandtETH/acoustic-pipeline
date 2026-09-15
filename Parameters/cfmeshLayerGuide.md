# Boundary layers in the native cfMesh workflow

`cartesianMesh` creates, optimises and subdivides layers within its own workflow.
Layer settings come from `cfmeshRotorDict` / `cfmeshStatorDict` and the values in
`cfmeshCommon.cpp`. There is no separate layer-generation pipeline stage or CLI
layer preset. The stator stops before native layer generation by default.

## Why five layers can occupy the same thickness as three

The mesher first establishes a near-wall layer based on the local surface cell
size and geometry. Layer refinement then subdivides those existing edges.
`propellerLayerCount` controls subdivision; it does not independently prescribe
the total stack thickness. If an edge has thickness T and ratio r, the uncapped
first thickness is `T*(r-1)/(r^n-1)`, or `T/n` when r=1. Increasing n therefore
reduces the first thickness while retaining the outer edge of the stack.

## Controls to edit in cfmeshCommon.cpp

| Input | Effect |
| --- | --- |
| `propellerLevel` and `baseCellSize` | Set local surface cell size; this also influences the available native layer thickness and surface resolution |
| `propellerRefinementThickness` | Width of the fine-cell region around the blade; this is NOT the boundary-layer stack thickness |
| `propellerLayerCount` | Number of subdivisions in the native layer |
| `propellerLayerThicknessRatio` | Growth between successive subdivisions; use a value at least 1 |
| `propellerMaxFirstLayerThickness` | First-layer upper bound in metres; 1e30 is effectively uncapped for this domain |
| `propellerLayerAllowDiscontinuity` | Controls propagation of the requested layer count across connected patches; not a thickness-smoothing switch |
| `layerOptimise` | Enables optional native layer optimisation; set 1 for a comparison with the supplied 0 setting |
| `layerSmoothNormalsIterations` | Iterations for smoothing layer-edge directions |
| `layerMaxIterations` | Iterations of the layer optimisation procedure |
| `layerFeatureSizeFactor` | Limits thickness relative to estimated local feature size; valid range 0 <= value < 1 |
| `layerRecalculateNormals` | Recomputes normal directions during optimisation |
| `layerRelativeThicknessTolerance` | Limits neighbouring thickness variation; lower values enforce a smoother, potentially thinner layer; valid range 0 <= value < 1 |
| `layerUntangle` | Native layer untangling switch; keep enabled for normal runs |

The optimisation parameters apply when `layerOptimise` is enabled. They belong
to the rotor's `boundaryLayers/optimisationParameters` block and can also be
edited there directly. The native spelling is **`recalculateNormals`** (lowercase
`c`); the common parameter name maps to that spelling.

The first-layer cap is not an exact first-layer target. A low cap can leave a
large last subdivision because the original outer endpoint is retained. This
can produce a pronounced jump from the inner layers to the last layer.

This open-source cfMesh implementation does not provide independent, exact
snappy-style total-thickness control through these settings. Changing surface
resolution and native optimisation influences thickness but does not prescribe
it. Do not copy newer commercial CF-MESH+ total-thickness or outer-layer controls
into this setup and assume they will work.

## Uneven cells at the outside of the stack

In the supplied blade section, thin layers meet a Cartesian mesh with changes
in cell size. Potential contributors to the irregular outer row are:

- Nonuniform thickness/direction of the initial layer, especially near strong
  curvature and the thin trailing edge.
- The transition between the surface-aligned cells and Cartesian core cells,
  including nearby refinement-level transitions.
- A restrictive first-layer cap in the actual case snapshot. The current source
  setting is effectively uncapped, so this is not established as the cause here.
- A section cutting three-dimensional cells obliquely; a section polygon alone
  does not establish the original cell's quality.

For a controlled comparison, first enable `layerOptimise 1` with the supplied
5 normal-smoothing iterations, 5 optimisation iterations, feature-size factor
0.3 and relative thickness tolerance 0.1. Keep layer count and refinement fixed
so the effect is identifiable. Inspect the outer row, measured wall-normal
heights and the checkMesh quality results. Then test surface refinement or the
fine-region width separately if the transition remains abrupt. More layers alone
will not repair a poorly shaped initial layer. These are diagnostic comparisons,
not a claim that the photographed mesh has been repaired.

## Studies and no-layer comparisons

Use `--study-file cfmeshCommon --study-parameter layerOptimise --study-values '0...1'`
with your normal single-propeller, single-RPM mesh-only study command.
For a no-layer region, add `workflowControls { stopAfter edgeExtraction; }` to
its dictionary. For the full native workflow remove that stop. Do not use
`nLayers 0` to try to remove the mesher's initial automatic layer.

New cases snapshot these parameters. Inspect the case's expanded
`cfmesh/rotor/system/meshDict`, not just the current source parameter file.

## Implementation references

- [cfMesh user guide, boundary layers](https://cfmesh.com/wp-content/uploads/2015/09/User_Guide-cfMesh_v1.1.pdf), pages 16–17.
- [Native layer subdivision](https://develop.openfoam.com/Community/integration-cfmesh/-/blob/3ff8555514827646c34cacfe5f0f691e49cdbc96/meshLibrary/utilities/boundaryLayers/refineBoundaryLayers/refineBoundaryLayersFunctions.C): edge splitting, geometric progression and retained outer endpoint.
- [Native optimisation settings](https://develop.openfoam.com/Community/integration-cfmesh/-/blob/3ff8555514827646c34cacfe5f0f691e49cdbc96/meshLibrary/utilities/smoothers/geometry/meshOptimizer/boundaryLayerOptimisation/boundaryLayerOptimisation.C): keyword spelling, ranges and smoothing parameters.
