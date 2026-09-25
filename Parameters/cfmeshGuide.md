# Editing the cfMesh setup

Use these files like the snappyHexMesh dictionaries: `.cpp` files hold shared
values and the region dictionaries reference them with `$variableName`.
These are OpenFOAM dictionary files, not compiled C++.

## Where to edit

| File | Settings |
| --- | --- |
| `cfmeshCommon.cpp` | Base cell size and refinement levels; blade refinement thickness; blade layer count, growth ratio and discontinuity switch |
| `cfmeshSizes.cpp` | Derived native sizes; leave expressions intact and edit inputs in the common file |
| `cfmeshGlobalDict` | Additional native cfMesh entries shared by rotor and stator |
| `cfmeshRotorDict` | Rotor surface refinement, native boundary layers and workflow; custom volume refinements |
| `cfmeshStatorDict` | Stator surface refinement and workflow; custom volume refinements |
| `cfmeshDomainDict` | STL scale, sphere-relative box margins and inlet/wake split, rotor cylinder dimensions and segment count |
| `cfmeshRefinementDict` | Generated volume refinement geometry, on/off switch and sphere surface resolution |
| `cfmeshFeatureDict` | Explicit OBJ feature groups, refinement levels and thicknesses |
| `cfmeshPipelineDict` | Quality improvement iterations, cylinder projection and acceptance thresholds |

Lengths and cell sizes are in metres. Geometry factors multiply the measured
propeller span or the sphere radius as described in the geometry dictionaries.
The shaft remains the Y axis through the origin; patch names and rotor/stator
assembly are part of the pipeline geometry contract.

## Sphere-relative domain

`boxSizing sphereRelative` derives the bounds from the effective sphere radius
`R = 0.5 * sphereDiameterFactor * measuredPropellerSpan`. In permeable mode,
`--acoustic-sphere-diameter` overrides that factor for both the sphere and domain.
Without a permeable acoustic surface, the refinement sphere factor still sizes
the domain, even if generated refinements are disabled.

With the supplied `lateralMargin 0.1`, `inletMargin 0.1` and `inletFraction 0.30`:

```text
boxMin = (-1.1R, -2.5666667R, -1.1R)
boxMax = (+1.1R, +1.1R, +1.1R)
```

The 10% margin is measured against sphere radius, on each side. The propeller
plane and sphere centre stay at Y=0; +Y is the inlet, -Y is the outlet.
The total axial length is 3.6666667R: 30% upstream and 70% downstream, measured from
the propeller plane. The box centre is at Y=-0.7333333R. In general, inlet distance
is `R*(1+inletMargin)` and outlet distance is `inletDistance*(1-inletFraction)/inletFraction`.

For a 10-inch (0.254 m) measured span and factor 2.5, R=0.3175 m, giving
`boxMin (-0.34925 -0.8149167 -0.34925)` and
`boxMax (0.34925 0.34925 0.34925)` in metres.
The bounds grow with the propeller or sphere factor; the previous 308 mm
sphere-containment limit no longer applies. Rotor axial clearance is still
controlled independently by `rotorHalfLength`, and STL/rotor fit checks remain.

`wakeOffsetFactor` separately moves the refinement cylinders toward the outlet;
it does not move the acoustic sphere or set the box split. These cylinder lengths
have not been extended to fill the new downstream domain. The margins specify
geometric clearance, not a demonstrated boundary-independent acoustic solution.

Use `boxSizing absolute` with explicit `boxMin` and `boxMax` to opt out of scaling.
Old case snapshots without `boxSizing` keep that absolute interpretation.
The resolved bounds are saved in `Parameters/cfmeshDomain.generated` and
`cfmesh/geometry.json`; edit the source ratios rather than the generated bounds.

## Base resolution and refinement levels

Every requested cell size is `baseCellSize / 2^level`. Levels must be nonnegative
integers: raising a level by one halves the requested size in each direction.
cfMesh still receives native physical cell sizes, so actual local cells depend
on its refinement, surface fitting and layer operations. Overlapping refinement
requests can make a region finer than its individual requested level.

The supplied APC 10x7E configuration uses a 0.022 m base. The supervisor's
reference 4000 RPM AMI case contains 1,245,656 cells:

| Input | Level | Requested size |
| --- | --- | --- |
| Background | 0 | 22 mm |
| `propellerLevel` | 4 | 1.375 mm |
| `interfaceLevel` | 3 | 2.75 mm |
| `rotaryRegionLevel` | 3 | 2.75 mm |
| `innerCylinderLevel` | 3 | 2.75 mm |
| `outerCylinderLevel` | 2 | 5.5 mm |
| `acousticSphereLevel` | 0 | 22 mm |

Cell count is geometry- and domain-dependent; 1.25 million is a saved reference result,
not a general guarantee. Layer settings and the 8 mm blade refinement-band
width remain independent.
Change `baseCellSize` to scale every requested size, or a single level to refine
one region. Increasing propeller dimensions does not automatically coarsen the
base cell size. Old case snapshots containing absolute cell sizes remain readable.

Keep computed expressions in `cfmeshSizes.cpp` separate from study inputs in
`cfmeshCommon.cpp`; this prevents dictionary editing from freezing derived sizes.
For a blade resolution study use:

```text
--study --study-file cfmeshCommon --study-parameter propellerLevel --study-values '4...5...6'
```

Each geometry report records the base size, requested levels and physical sizes.

## Native mesher settings

### Explicit feature curves

The supplied preset uses `enabled false` to reproduce the supervisor's actual
empty feature dictionary: its four OBJ files were missing. Level-5 settings
remain available for an explicit feature study, but enabling them with existing
feature files changes the reference configuration.

Place a `FEATURES` folder next to your order's `STL` folder:

```text
my-order/
  STL/10x7E.stl
  FEATURES/10x7E_le.obj
  FEATURES/10x7E_te.obj
  FEATURES/10x7E_tip.obj
  FEATURES/10x7E_hub.obj
```

The prefix comes from the selected STL filename, so `12x6.stl` uses
`12x6_le.obj`, `12x6_te.obj`, `12x6_tip.obj`, and `12x6_hub.obj`.

Edit `Parameters/cfmeshFeatureDict` before starting a new order. `le/level`,
`te/level`, `tip/level`, and `hub/level` independently select the refinement; `refinementThickness`
in each group sets the surrounding refinement distance in metres. Level 6
with a 0.02 m base requests 0.3125 mm cells. Levels are relative to the background,
not added to `propellerLevel`; overlapping requests use the finer refinement.
Set `enabled false` to disable all feature refinement, or remove a group from
`groups` to disable that group. Add another suffix and matching dictionary block
to support additional feature curves beyond `groups (le te tip hub)`.

The OBJ files must contain explicit `v` vertices and `l` line/polyline elements.
Face-only OBJ exports are rejected. Export curves in the same coordinate system
and units as the STL: `cfmeshDomainDict/scale` is applied to both. Missing files
are warned about and skipped, so existing STL-only orders still work.

The pipeline snapshots source curves in each case's `FEATURES` folder and writes
scaled curves to `cfmesh/rotor/constant/triSurface/features`. Generated native
`edgeMeshRefinement` entries affect the rotor only and are independent of the
volume refinement switch. The geometry report records files, hashes, settings
and missing groups. Existing order/case parameter snapshots must be updated or
a fresh order created to use newly edited repository parameters.

For a tip refinement study use:

```text
--study --study-file cfmeshFeatureDict --study-parameter tip/level --study-values '5...6...7'
```

These controls use cfMesh's native edge refinement interface described in the
[cfMesh user guide, section 4.3](https://cfmesh.com/wp-content/uploads/2015/09/User_Guide-cfMesh_v1.1.pdf).

`cfmeshRotorDict` and `cfmeshStatorDict` are complete native `meshDict` inputs.
Additional entries are passed through without a Python whitelist. Put common
native controls in `cfmeshGlobalDict`, or region-specific controls directly in
the appropriate region dictionary. Use entries supported by the installed
cfMesh version; snappyHexMesh keywords are not interchangeable with cfMesh.
Omitted native controls still use cfMesh's own internal defaults: this setup
does not enumerate every version-specific mesher option.

For example, change `propellerLayerCount` in `cfmeshCommon.cpp` for a layer-count
study, or lower `propellerMaxFirstLayerThickness` there to set a first-layer cap. Layer thicknesses are generated by cfMesh; a requested layer
count or cap does not by itself establish the achieved wall resolution or y+.

The `objectRefinements` blocks include `cfmeshRegions.generated`. This file is
created from the common, domain and refinement dictionaries for each case.
Add your own uniquely named native refinement objects next to the include.
Set `enabled false` in `cfmeshRefinementDict` to disable the generated objects
while retaining your custom objects. Do not edit the generated file as a source.

## Workflow and precedence

See [the layer control guide](cfmeshLayerGuide.md) for thickness behaviour,
optimisation parameters and diagnosing an irregular outer layer.

Layer generation follows the rotor and stator dictionaries directly:

- The supplied rotor dictionary runs the full native mesher with blade layers.
- The supplied stator dictionary stops after `edgeExtraction`, avoiding layers.
- Remove the stator `workflowControls` block to run its full native workflow;
  add the native `boundaryLayers` settings you want for that region.
- A full run or `stopAfter edgeExtraction` are the supported pipeline workflows.
  Other stop stages cannot be assembled by this pipeline.

For a no-layer comparison add `workflowControls { stopAfter edgeExtraction; }`
to the rotor dictionary as well. Setting `nLayers 0` is not a substitute for
stopping before the automatic base-layer stage. Native cfMesh inserts an
automatic base layer during the full workflow, including on the rotor interface.
Legacy order metadata for layer presets is no longer applied; when remeshing
an old order, review its case dictionaries. Existing solved meshes are unchanged.

`cfmeshPipelineDict/improveMeshQuality` controls the additional improvement
command for both rotor and stator after `cartesianMesh` finishes. For a full
workflow this runs after boundary layers have been generated; it also runs
after an `edgeExtraction` stop. It does not control the full mesher's internal
optimization. The `rotor` and `stator` subdictionaries each have independent
`enabled`, `nLoops`, `nIterations`, and `nSurfaceIterations` settings. Set
`improveMeshQuality/rotor/enabled false` to skip only the rotor pass, or
`improveMeshQuality/stator/enabled false` to skip only the stator pass.
Both region blocks must be complete. Older case dictionaries with the four
settings directly inside `improveMeshQuality` remain supported and apply those
shared settings to both regions. If region blocks exist, they take precedence
over any old shared settings. Cases without this dictionary retain shared defaults.
Additional smoothing can change layer spacing: compare layer thickness and
spacing as well as mesh-quality results when evaluating this variant.

Cylinder projection is separately switchable. Its allowed movement is
`movementLimitFactor * radius * (1 - cos(pi / cylinderSegments)) + absoluteTolerance`.
Acceptance entries are fractions between 0 and 1. They determine whether a mesh
is accepted; changing them does not improve the mesh. Volume and coverage checks
still apply when projection is disabled.

The diagnostic `checkMesh -allGeometry -allTopology` output is saved in
`log.checkMesh.extended` and recorded as `extended_mesh_diagnostics_ok`. Acceptance
uses the normal solver-facing `log.checkMesh` before NCC and
`log.checkMesh.NCC` afterward. `--allow-bad-mesh` bypasses failed standard checks
only; it does not bypass volume or interface coverage checks.

## Reproducible thesis runs

Edit source parameters before creating a new case. Each case receives a copy of
`Parameters`; later source edits do not change that case. Use new orders for
comparisons. Existing old cases without `cfmeshPipelineDict` retain the previous
hard-coded defaults; a supplied dictionary must contain all its entries.

Example study options (add to your usual single-STL, single-RPM mesh-only run):

```text
--study --study-file cfmeshCommon
--study-parameter propellerLayerCount --study-values '3...5...7'
```

For the additional quality-improvement iterations, use `--study-file
cfmeshPipelineDict --study-parameter improveMeshQuality/rotor/nIterations`
(or `improveMeshQuality/stator/nIterations` for the stator).

Inspect each case's `cfmesh/rotor/system/meshDict` and
`cfmesh/stator/system/meshDict` for the expanded native inputs.
`cfmesh/pipeline-controls.json` records effective pipeline settings;
`cfmesh/geometry.json` records geometry and generated refinement sizes.
Check the meshing logs, `cfmesh/mesh-status.json`, `cfmesh/ncc-status.json` and
`cfmesh/interface-projection.json` (when enabled) alongside the actual mesh.
Expanded inputs record supplied settings, not cfMesh's unlisted internal defaults.
