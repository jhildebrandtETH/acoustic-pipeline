# Native cfMesh acoustic pipeline (experimental duplicate)

This branch is developed in `C:\repos\acoustic-pipeline`. Use an external simulation directory, such as `~/run/`, when running from this checkout: the existing original-repository output guard remains enabled, including resolved links and Windows path case variants. Foreign pipeline orders and the pipeline/filesystem root are also rejected. See the runtime setup commands below before the first run.

The familiar `main.py` scheduler now meshes separate rotor and stator fluid volumes with cfMesh, merges them, creates the `rotaryRegion` cell zone and Foundation OpenFOAM 13 NCC interface, and then runs the rotating solver, surface sampling, FW-H acoustics and PDF report. `--mode AMI` is the existing command-line name; the actual coupling is NCC. MRF is not implemented. No blockMesh/snappyHexMesh or hybrid layer grafting runs in this route.

## First command: mesh the supplied propeller

If a WSL terminal retains a stale working-directory handle after a folder is moved or recreated, the launcher now re-enters the shell's absolute `PWD` before resolving paths. If that location no longer exists, it exits with a short recovery instruction. You can refresh the terminal manually:

```bash
cd /
cd /mnt/c/repos/acoustic-pipeline
```

The launcher resolves its own Python modules explicitly, including with `PYTHONSAFEPATH`, `python -P`, or `python -I`. A `ModuleNotFoundError: No module named tools` at startup is unrelated to simulation-directory reuse. Run the `main.py` in this checkout; do not copy `main.py` alone into the order folder.

Start Docker Desktop with Ubuntu/WSL integration enabled. In Ubuntu:

```bash
cd /mnt/c/repos/acoustic-pipeline
source ~/miniforge3/etc/profile.d/conda.sh
conda activate of_pipeline_env
python main.py --sim-dir ~/run/cfmesh_layers3 --rpms 4000 --mode AMI \
  --turbulence kOmegaSST --total-cores 4 --mesh-only --live-output
```

An empty `--sim-dir` (or empty `STL/`) is seeded automatically as `STL/10x7E.stl` from the selected `Downloads/APC10x7E 1/APC10x7E/fused.stl`, in metres. No FEATURES directory is needed. The name is only a label; dimensions come from the actual STL. Geometry assumes the shaft is the Y axis through the origin.

`--mesh-only` performs volume meshing, zone/interface assembly and checks, then stops before flow/acoustics. By default a failed strict checkMesh stops before NCC/solver. The earlier no-layer propeller baseline had **six failed strict checks**; do not mistake the completed diagnostic run for an accepted production mesh. Inspect `log.checkMesh` and refine/adjust the mesh. `--allow-bad-mesh` explicitly permits diagnostic continuation, but cannot override wrong volume, missing regions/patches or inadequate initial interface overlap.

`--live-output` streams full command output; omit it for the usual dashboard. Logs are saved either way. Several concurrent cases can interleave terminal output, so each case's log is authoritative.

## Bounded full-pipeline check, including solving

The launcher now runs geometry preparation and cylindrical-interface projection in separate Python processes. This avoids the reproduced native NumPy `invalid pointer` abort in a scheduler worker. Native faults are captured in logs and reported as case failures. The normal dashboard and `--live-output` remain available.

Both cfMesh interface surfaces are projected onto the same exact circular cylinder before `checkMesh` and NCC creation. The correction is limited to the polygon-to-circle approximation error; it cannot repair an incorrectly meshed interface. For the supplied propeller, the maximum correction was about 12.34 micrometres, and minimum initial NCC coverage increased from about 64% to over 98%. The existing 95% minimum-face and 99.9% average-coverage requirements remain enforced.

Fresh cases also discard the template's saved `0/uniform/time`, which otherwise overrides the configured initial timestep. A fresh solver run that exits without advancing time is now a failure.

For a short diagnostic check of the complete launcher, with the current dictionaries:

```bash
cd /mnt/c/repos/acoustic-pipeline
python main.py --sim-dir ~/run/cfmesh_flow_check \
  --rpms 4000 --mode AMI --turbulence kOmegaSST --total-cores 8 \
  --acoustic-surface impermeable --end-on time 3.6e-8 \
  --allow-bad-mesh --study --study-file controlDict \
  --study-parameter writeInterval --study-values 3 --live-output
```

This uses the ordinary parameter-study mechanism to save the final field after three startup steps, without changing the root dictionaries. It is a startup check, **not a developed-flow or acoustic validation**. The actual propeller mesh still has failed strict geometry checks; `--allow-bad-mesh` explicitly permits this diagnostic run. The checks and reports retain those failures. Three layers remain configured on the propeller. A trial enabling additional layer smoothing worsened other quality measures and was not adopted.

The acoustic surface fields are sampled during short runs. If fewer than five rotations are available, postprocessing records `report/acoustic-status.json` as `insufficient_duration`, skips the spectrum, and continues the other reports. A production spectrum still needs five fully arrived rotations, including propagation delay, after suitable flow settling. Other acoustic errors remain failures.

For your usual longer run, edit `Parameters/controlDict.cpp`, omit the study options, and choose your intended `--end-on` condition. Keep or omit `--allow-bad-mesh` deliberately; without it, the current propeller mesh stops at the strict quality checks. Use `--resume` only to continue the stored order, whose mesh-only/full-run settings remain unchanged.

## Dictionaries to edit

Edit the root `Parameters/`, then run the normal command again. Each new case receives a snapshot. A new order starts with current dictionaries. If `simulation_order.json` already exists, use `--resume` or move that file aside before creating a new order.

| File | Controls |
| --- | --- |
| `Parameters/cfmeshCommon.cpp` | Background, propeller, interface, cylinder and sphere cell sizes (metres); propeller refinement thickness |
| `Parameters/cfmeshRefinementDict` | Sphere diameter factor and the original cylinder radius/height/wake-offset factors |
| `Parameters/cfmeshDomainDict` | STL scale, farfield box corners, rotor radius relative to measured propeller span, cylinder half-length and segments |
| `Parameters/cfmeshRotorDict` | Complete native rotor meshDict: local/object refinement, native boundary layers and workflow controls |
| `Parameters/cfmeshStatorDict` | Complete native stator meshDict: local/object refinement and workflow controls |
| Existing `Parameters/controlDict.cpp`, solver/turbulence dictionaries | Time controls, numerical settings, field and solver parameters |

The default rotor dictionary now enables **three native cfMesh boundary layers** at `propeller`, with `thicknessRatio 1.2`. Edit `boundaryLayers/patchBoundaryLayers/propeller` in `cfmeshRotorDict` to change these settings. The stator retains its no-layer stop. The full rotor workflow also inserts a base layer at its interface.

`allowDiscontinuity 1` allows local layer termination at difficult features; requesting three layers is not a guarantee of three intact layers everywhere. Inspect the propeller mesh and quality log, especially near the trailing edge and root. Initial layer thickness follows the native base-layer geometry; this configuration does not prescribe a target y+.

For a no-layer comparison, rerun with `--boundary-layers none`. cfMesh inserts an automatic base layer during its full workflow, so `nLayers 0` does not suppress that stage. The no-layer option stops at `edgeExtraction`, followed by `improveMeshQuality -nLoops 2 -nIterations 20 -nSurfaceIterations 0`.

`--boundary-layers dict` is the default and obeys the dictionaries. `--boundary-layers none` forces both no-layer stops. `--boundary-layers cfmesh` enables the complete rotor workflow with the dictionary's layer settings and keeps the stator without layers. Only complete workflows or the supported `edgeExtraction` stop are accepted.

Both mesh dictionaries include generated native `objectRefinements` for the rotor volume, inner cylinder, outer cylinder and sphere. They use the same physical coordinates on both sides of the interface. Additional custom objects can be added inside either `objectRefinements` block. Edit the source dictionaries rather than `cfmeshRegions.generated`, which is regenerated per case.

The shapes reproduce normal kOmegaSST-AMI preprocessing. With `D` the measured propeller span and `R = 0.5 × sphereDiameterFactor × D`:

| Region | Radius | Y extent |
| --- | --- | --- |
| Rotor | `0.6D` | `-0.025` to `+0.025` m |
| Inner cylinder | `0.6R` | `-0.38R` to `+0.14R` |
| Outer cylinder | `0.8R` | `-0.48R` to `+0.24R` |
| Acoustic/refinement sphere | `R` | Centred at the origin |

The two refinement cylinders share the original wake offset `-0.12R`. The sphere diameter factor defaults to 2.5; in permeable runs `--acoustic-sphere-diameter` controls the sampling sphere and these refinement dimensions together. Mesh-only and impermeable runs also have the refinement regions, with the factor read from `cfmeshRefinementDict`; the original preprocessing's undefined sphere-radius problem in impermeable mode is avoided. Reference STLs are written in `constant/triSurface` for viewing. The sampling sphere uses the original six subdivisions.

Current cfMesh targets preserve your edited 10 mm background, 1.25 mm propeller and 6.25 mm interface settings, and add 3.125 mm rotor/inner-cylinder targets and 6.25 mm outer-cylinder/sphere targets. Propeller distance refinement extends 12.5 mm, matching the original two equal-level distance bands. The finest overlapping request wins, so a volume region can refine an interface beyond its surface-only size. These preserve the region hierarchy; they are not an exact translation of snappy levels because the normal base mesh is anisotropic (`16 × 48 × 16`) while this cfMesh setup uses cubic refinement. All target sizes remain independently editable.

The three-layer baseline keeps `nLayers 3`, `thicknessRatio 1.2`, and `allowDiscontinuity 1`. An optional `maxFirstLayerThickness 0.0001` is commented out: testing the cap produced approximately 0.1 mm first layers but a much thicker last layer. Leave it disabled for the smoother initial three-layer test. This does not establish a target y+.

Full-pipeline patch names match the original solver: `propeller`, `inlet`, `outlet`, `walls`, `rotaryRegion`, `rotaryRegion_slave`. The older standalone test retains `propellerSurface`/`farfield`. The farfield box has the original inlet/outlet/walls boundary-condition types; changing these physical conditions requires changing the field templates too.

## Full rotation, solver and acoustics

Create a fresh order for a full run after inspecting mesh quality:

```bash
mkdir -p ~/run/cfmesh_full/STL
cp cfmesh_test/input/fused.stl ~/run/cfmesh_full/STL/10x7E.stl
python main.py --sim-dir ~/run/cfmesh_full --rpms 4000 --mode AMI \
  --turbulence kOmegaSST --total-cores 4 --acoustic-surface impermeable \
  --end-on rev 10 --live-output
```

Ten revolutions is a convenient finite run, not evidence of flow convergence. The existing acoustic spectrum uses five fully arrived rotations, so very short solver runs cannot produce that spectrum. Choose a sufficiently long settled sampling interval for thesis results. `--end-on convergence`, `force_convergence`, `residual_convergence`, or `time SECONDS` remain available. Turbulence choices are `kOmegaSST`, `kEpsilon`, and `DES`; `--cores` remains an alias for `--total-cores`.

For the fixed permeable sphere, replace the acoustic options with:

```bash
--acoustic-surface permeable --acoustic-sphere-diameter 2.5
```

The diameter is a multiple of the measured propeller span. The sphere must fit inside the box and enclose the propeller. Its FW-H geometry is stationary; the impermeable propeller uses rotating geometry. Surface sampling is selected automatically in each turbulence template.

For multiple RPMs use e.g. `--rpms 3000 4000 5000`. `--field-init on` preserves the existing sequential RPM chain and maps the previous fields before parallel decomposition; `off` allows independent cases.

## Retry and parameter studies

External order example (Ubuntu/WSL, with `of_pipeline_env` activated):

```bash
python /mnt/c/repos/acoustic-pipeline/main.py \
  --sim-dir /home/jonas/run/cfmesh_refined \
  --rpms 4000 --mode AMI --turbulence kOmegaSST \
  --total-cores 24 --mesh-only --live-output
```

A mounted Windows folder is also accepted, for example `--sim-dir /mnt/c/CFD/cfmesh_refined`. Quote paths containing spaces. An existing `STL/` is used as supplied; an empty order receives the previously selected propeller. Existing directories, including directories containing only `STL/` or leftover case folders, are accepted. Only an existing `simulation_order.json` requires `--resume`. Parameters and templates still come from this checkout.

```bash
python main.py --sim-dir ~/run/cfmesh_layers3 --resume --live-output
```

**New order (without `--resume`):** the same `--sim-dir` is accepted whenever it contains no `simulation_order.json`. It may already contain `STL/`, unrelated files or old case folders. A case with the same name is preserved as `<case>_PREVIOUS_<timestamp>` before its replacement is installed. Templates are copied under a temporary sibling name and then renamed into place; this avoids the WSL mounted-drive failure where a deleted case directory is invisible but creating its old name raises `FileExistsError`.

**Existing order:** any `simulation_order.json` (including an empty or failed order) prevents a new order from overwriting it. Use `--resume` to continue. To intentionally start again with new options in the same directory, move the order file aside first, for example:

```bash
mv ~/run/cfmesh_layers3/simulation_order.json \
  ~/run/cfmesh_layers3/simulation_order.previous.$(date +%Y%m%d_%H%M%S_%N).json
# Then run your normal command with the same --sim-dir.
```

**Continue (`--resume`):** restores stored mesh-only/layer/quality/core settings, retries failed stages and skips completed cases. Failed mesh attempts restart from current dictionaries; existing case directories are kept with `_PREVIOUS_<timestamp>`. Solver/postprocessing resume retains its case snapshot. Extra options with `--resume` do not replace stored settings. A lock prevents concurrent launchers from using the same order; a stale lock file alone does not block reuse.

To switch mesh-only to a full simulation, move the existing order JSON aside as above, then use the same normal command without `--mesh-only` and without `--resume`, adding acoustic/end-time options. This rebuilds the case and preserves the earlier mesh in a sibling folder. You can also use a separate order if you prefer.

For a mesh study (one STL and one RPM in a new order):

```bash
mkdir -p ~/run/cfmesh_study/STL
cp cfmesh_test/input/fused.stl ~/run/cfmesh_study/STL/10x7E.stl
python main.py --sim-dir ~/run/cfmesh_study --rpms 4000 --mode AMI \
  --turbulence kOmegaSST --total-cores 4 --mesh-only --study \
  --study-file cfmeshCommon --study-parameter propellerCellSize \
  --study-values '0.00125...0.000625' --live-output
```

Studies accept existing files in `Parameters`, including `cfmeshCommon`, `cfmeshRefinementDict`, `cfmeshDomainDict`, `cfmeshRotorDict`, `cfmeshStatorDict`, and the usual solver parameter files such as `controlDict`. The `.cpp` suffix is optional. Nested entries use foamDictionary paths such as `boundaryLayers/patchBoundaryLayers/propeller/nLayers`; their generated case-folder names are sanitized.

## Inspect in ParaView

For the first order, after meshing (including a run stopped on quality):

```bash
cd ~/run/cfmesh_layers3/10x7E_4000RPM_AMI
touch sim.foam
paraview sim.foam
```

`paraview` requires an accessible ParaView executable. This machine currently has Windows ParaView, so the direct alternative from WSL is:

```bash
"/mnt/c/Program Files/ParaView 6.1.0/bin/paraview.exe" --data="$(wslpath -w "$PWD/sim.foam")"
```

Or use Windows ParaView > File > Open and select the case's `sim.foam`. Choose the OpenFOAM reader if prompted, select `internalMesh` and desired boundary patches, then Apply. Use **Surface With Edges** for cells and a **Clip** or **Slice** through the shaft to see internal refinement/layers. Hide the outer box to inspect the propeller. After a solver run select the reconstructed case, refresh the time list, and select `U` or `p`. NCC may appear as separate coincident interface patches in the built-in reader.

## Logs, results and installed runtimes

- `pipeline-native-fault.log` in the order directory: unexpected native faults in the launcher.
- `<case>.preprocessing.log` in the order directory: preprocessing output and native fault traces.
- `case/cfmesh/log.interfaceProjection`, `interface-projection.json`: cylindrical correction and its displacement limit.
- `case/log.cfmesh`: aggregate native meshing log; `case/cfmesh/{rotor,stator}/log.*`: individual native-container commands.
- `case/log.checkMesh`, `log.checkMesh.NCC`, `log.createNonConformalCouples`, `log.pimpleFoam`: mesh/coupling/flow evidence.
- `case/cfmesh/geometry.json`, `mesh-status.json`, `ncc-status.json`: dimensions, acceptance and interface overlap.
- `case/postProcessing/`: forces, residuals and sampled acoustic VTK surfaces.
- `case/report/spl_spectrum.png` and `simulation_report.pdf`: acoustic spectrum/report after a full run.

Native `cartesianMesh`, `improveMeshQuality` and every `foamDictionary` query/edit run in Docker image `opencfd/openfoam-default:2512`. No host OpenFOAM v2512 installation or `CFTEST_FOAM_BASHRC` is needed. Repository and case paths are bind-mounted read/write, including external `--sim-dir` paths and study dictionaries. Linux containers use the invoking user's UID/GID. Host case paths containing spaces receive whitespace-free container aliases for OpenFOAM. Live output and stage logs are retained.

The standalone `generateBoundaryLayers` executable remains host-side: the existing `CFMESH_BIN` / PATH / `~/.local/cfmesh` resolver is unchanged. Run `bash setup_cfmesh.sh` once if that binary is not already installed; rerunning the script verifies an existing installation. The native rotor workflow still uses its existing built-in cfMesh layer settings. Assembly, NCC, solver and remaining Foundation utilities continue to use `microfluidica/openfoam:13`. Python uses the existing `of_pipeline_env`.

On the Linux server, from this branch's repository directory:

```bash
git switch cfmesh-standalone
git pull --ff-only
conda activate of_pipeline_env
docker pull opencfd/openfoam-default:2512
docker image inspect microfluidica/openfoam:13 >/dev/null
bash setup_cfmesh.sh
python cfmesh_runtime.py --smoke-test
CFMESH_DOCKER_TESTS=1 python test_cfmesh_runtime.py
python -c 'from tools import resolve_cfmesh_executable; print(resolve_cfmesh_executable())'
```

If the Foundation image is absent, run `docker pull microfluidica/openfoam:13`. The smoke test runs all three native utilities with `-help`, queries and edits an included dictionary, and verifies host-visible polyMesh writes in temporary repository-local and external cases. It launches no mesh. Pipeline preflight also checks Docker, both images, the three utilities and the host binary. Docker pulls v2512 automatically if absent.

Use an external order directory (the existing original-repository output guard is retained):

```bash
mkdir -p "$HOME/run/cfmesh_docker2512/STL"
cp cfmesh_test/input/fused.stl "$HOME/run/cfmesh_docker2512/STL/10x7E.stl"
python main.py --sim-dir "$HOME/run/cfmesh_docker2512" --rpms 4000 --mode AMI \
  --turbulence kOmegaSST --total-cores 4 --mesh-only --live-output
```

Use a fresh order directory for a new run, or `--resume` for an existing order. This is the full propeller mesh and may take time; the smoke test above is the lightweight runtime check.

The optional ParaView report atlas currently reports unavailable because headless `pvpython`/`pvbatch` is not installed on the WSL host; the acoustic spectrum and PDF still generate. Interactive Windows ParaView viewing works independently. Keep optional visualization settings in the existing visualization configuration, or configure a compatible renderer if atlas images are needed.

## Validation and limitations

The current refined propeller test has **2,791,476 cells**, three-cell stacks at all 10,404 propeller faces, and 9,903 stacks whose two growth ratios are within 0.02 of 1.2. It still fails four strict checks (face-tet quality, low determinant, concavity, interpolation weight). With the diagnostic quality override, NCC mean coverage is 0.999993 but local minima are 0.642939 / 0.915357, below the 0.95 local threshold. The actual propeller solver is therefore blocked. This is a meshing/interface limitation, not a successful production simulation. See `rework-20260913/final-propeller-results.json` under the validation directory. The original repository's 16,628 files were rechecked and unchanged.

The 13 September rework is recorded in `cfmesh_test/validation/rework-20260913/`. It checks empty/repeated order directories, preservation of previous results, resume, concurrent-order protection, native region definitions on both sides, and the final three-layer setup. A three-step parallel rotating-flow test with permeable sampling exercises the updated connection to the acoustic predictor. These short tests demonstrate integration, not physical convergence.

Earlier 11 September results below refer to the previous refinement configuration:

The default three-layer propeller run is saved at `cfmesh_test/validation/native-propeller-layers3/10x7E_4000RPM_AMI`. Open its `sim.foam` to inspect it immediately. Native generation and layer refinement completed. Independent opposite-face tracing found three-cell stacks at all 10,404 propeller boundary faces; 9,897 stacks have both spacing ratios within 0.02 of 1.2. The measurements are in `cfmesh_test/validation/propeller-layers3-stacks.json`.

This layered mesh fails four strict checks: 152 low-quality face-tet decompositions, 1,014 concave cells, 27 small interpolation weights, and 72 small adjacent-cell volume ratios. All cell volumes are positive and maximum skewness is 3.73076, but the strict quality gate correctly stops before NCC and solving. These results confirm generated layers, not an accepted solver mesh.

Validation artifacts are under `cfmesh_test/validation/`. The selected propeller completed native meshing and NCC with mean initial coverage 0.999996, but failed six strict pre-NCC quality checks and one basic post-NCC check. Native-layer assembly was also exercised with an explicit quality override.

A small artificial rotating case completed two-core flow through six revolutions, impermeable FW-H prediction, spectrum and PDF generation. Three-step serial kEpsilon and DES cases produced permeable VTK samples, which were also read by the stationary-surface FW-H predictor. Native dictionary studies, uppercase STL input, output isolation and the default strict quality stop were checked. These are integration checks, not aerodynamic/acoustic validation or proof of a production-quality propeller mesh. Quality overrides in these fixtures are intentional and not defaults.

The original standalone experiment remains available through `run_cfmesh_test.sh` and `README_CFMESH_TEST.md`. It is separate from the full pipeline described here. The original README and changed template snapshots are retained in `cfmesh_test/validation/pre-native-pipeline/`.
