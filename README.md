# Acoustic Pipeline

**From propeller geometry to aerodynamic results and aeroacoustic predictions.**

The pipeline prepares OpenFOAM cases, meshes the rotor and surrounding fluid with
cfMesh, runs the flow solver, and produces plots, ParaView views and a PDF report.
Acoustic runs additionally sample a FW-H surface and predict sound at an observer.
Use `--aerodynamics-only` when you only need the flow solution and aerodynamic loads.

```text
STL + case settings
        |
        v
Case preparation -> cfMesh -> mesh checks -> OpenFOAM flow solution
                                  |                    |
                           --mesh-only                 v
                                                 Flow results
                                                       |
                                     Optional FW-H acoustic prediction
                                                       |
                                             Figures + PDF report
```

[Installation](#installation) · [First run](#first-run) · [Model selection](#model-selection)
· [Run options](#run-options) · [Results](#results-and-reports) · [Troubleshooting](#troubleshooting)

## Installation

### 1. Prepare the host

Run the pipeline in **Linux or a Linux distribution under WSL 2**. Use a Linux
terminal for all commands below. Native Windows Python is not the supported full
pipeline runtime; order locking and the standalone cfMesh helper require Linux.

You need Git, Conda, Docker, and `curl` or `wget` plus `tar`. Allocate enough RAM and
disk space for the mesh, concurrent cases, saved timesteps and acoustic samples.
`--total-cores` limits CPU allocation; it does not impose a memory limit.
Specify `--cores-per-case` for every new order. With 100 total cores and 20 cores per case, up to 5 cases run concurrently and the remaining cases queue. Unused remainder cores stay idle; field-initialization dependencies may further reduce concurrency. Resume keeps the saved allocation.

| Host | Docker setup |
| --- | --- |
| Linux workstation/server | Install [Docker Engine](https://docs.docker.com/engine/install/) for your distribution and configure access for your simulation account. |
| Windows + WSL 2 | Install [Docker Desktop with the WSL 2 backend](https://docs.docker.com/desktop/features/wsl/), enable your distribution under **Settings > Resources > WSL Integration**, and use Linux containers. Keep Docker Desktop running. |

Verify access from the **same Linux terminal** that will run the pipeline:

```bash
docker info
docker run --rm hello-world
```

Both commands must succeed without changing to another account. The Python Docker
client also needs access to that same daemon. On WSL, keep cases in the Linux
filesystem (for example `~/simulations`) when possible to reduce file I/O overhead.

### 2. Clone the repository and acoustic solver

```bash
git clone --recurse-submodules https://github.com/jhildebrandtETH/acoustic-pipeline.git
cd acoustic-pipeline
```

For an existing checkout:

```bash
git submodule update --init --recursive
```

`acousticSolver/` is a Git submodule. It must be populated before creating the
Python environment because the environment installs it as an editable package.

### 3. Create the Python environment

Install [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install)
or another Conda distribution **inside Linux/WSL**, then run from the repository root:

```bash
conda env create -f of_pipeline_env.yml
conda activate of_pipeline_env
python -m pip check
python main.py --help
```

The [environment file](of_pipeline_env.yml) specifies Python 3.12, the Docker SDK,
geometry and reporting dependencies, CPU PyTorch, and the local acoustic package.
A GPU is not required. Keep the environment file and submodule revision together
when sharing a reproducible checkout. See [Conda environment creation](https://docs.conda.io/projects/conda/en/stable/commands/env/create.html).

### 4. Pull both OpenFOAM images

```bash
docker pull opencfd/openfoam-default:2512
docker pull microfluidica/openfoam:13
```

| Runtime | Purpose |
| --- | --- |
| `opencfd/openfoam-default:2512` | Native cfMesh utilities and dictionary operations |
| `microfluidica/openfoam:13` | Mesh assembly, non-conformal coupling (NCC), checks and flow solution |
| Host `generateBoundaryLayers` | Standalone layer-generation utility |

These images are used for different stages. Do not substitute one for the other.
The pipeline sources the appropriate OpenFOAM environment **inside each container**;
a host OpenFOAM installation is not required. Docker bind mounts expose the case
and its `Parameters/` directory to the tools.

### 5. Install the cfMesh helper

```bash
bash setup_cfmesh.sh
```

The installer downloads the standalone Linux cfMesh 1.2.0 binaries and checks
`generateBoundaryLayers`. The default installation is discovered automatically.
For an existing custom installation:

```bash
export CFMESH_BIN=/absolute/path/to/generateBoundaryLayers
```

Use the executable path, not its directory. The current preflight checks this
helper even when a particular mesh configuration will not add layers.

### 6. Enable report visuals

Install [ParaView](https://www.paraview.org/download/) on the simulation host,
including its `pvpython` or `pvbatch` executable. It runs in its own Python runtime;
do not install ParaView into the Conda environment with pip.

```bash
export PARAVIEW_EXECUTABLE=/absolute/path/to/paraview/bin/pvpython
"$PARAVIEW_EXECUTABLE" --version
```

The pipeline otherwise searches `PATH` for `pvpython`, then `pvbatch`. Server
rendering uses `--force-offscreen-rendering`; the ParaView build and host graphics
libraries must support offscreen rendering. A desktop GUI session is not required
with a suitable headless build. Rendering failures are recorded in
`report/visuals/manifest.json` and the associated `paraview.log`.

ParaView is optional for solving. Without a working renderer, the report records
missing visuals rather than inventing them. Set `"required": true` in a case's
`visualization.json` if missing/partial visuals should fail postprocessing.

### 7. Check the installation

```bash
python -m tools.cfmesh_runtime --smoke-test
CFMESH_DOCKER_TESTS=1 python -m unittest discover -s tests -v
```

The smoke test exercises container utilities, dictionary includes and mounted-file
writes without creating a mesh. The integration tests require the installed
pipeline environment and Docker. Passing these checks is not a CFD validation;
inspect a mesh before committing to a production flow run.

## First run

### Supply a propeller

Create a dedicated **order directory** and put one or more STL files in its `STL/`
subdirectory. One order holds its input geometry, saved settings and generated cases.

```bash
mkdir -p ~/simulations/propeller_mesh/STL
cp /path/to/propeller.stl ~/simulations/propeller_mesh/STL/
```

Provide a closed, watertight propeller surface. The rotation axis is **+y through
the origin**. Check orientation, origin and dimensions before meshing.
`Parameters/cfmeshDomainDict` controls STL scaling; its current `scale 1` expects
coordinates in metres. For geometry in millimetres, set the scale to `0.001`.

If `examples/STL/10x7E.stl` exists, an empty order can use it automatically. Some
checkouts do not contain this example; supplying your own STL always avoids that
dependency. Source, template and configuration directories cannot be used as outputs.

### Inspect the mesh

```bash
python main.py --sim-dir ~/simulations/propeller_mesh \
  --rpms 4000 --mode AMI --turbulence kOmegaSST --wall-functions no \
  --total-cores 4 --cores-per-case 4 --mesh-only --live-output
```

Open the generated case's `sim.foam` in ParaView and inspect the blade surface,
near-wall cells, rotor/stator interface and mesh-quality logs. `--mesh-only` stops
after meshing and checks; it does not solve the flow or run acoustics.

### Run aerodynamics

Use a new order so its settings and results stay separate from the mesh inspection:

```bash
mkdir -p ~/simulations/propeller_flow/STL
cp ~/simulations/propeller_mesh/STL/*.stl ~/simulations/propeller_flow/STL/
python main.py --sim-dir ~/simulations/propeller_flow \
  --rpms 4000 --mode MRF --turbulence kOmegaSST --wall-functions no \
  --total-cores 4 --cores-per-case 4 --aerodynamics-only --end-on rev 20 --live-output
```

### Run aeroacoustics

```bash
mkdir -p ~/simulations/propeller_acoustics/STL
cp ~/simulations/propeller_mesh/STL/*.stl ~/simulations/propeller_acoustics/STL/
python main.py --sim-dir ~/simulations/propeller_acoustics \
  --rpms 4000 --mode AMI --turbulence kOmegaSST --wall-functions no \
  --total-cores 4 --cores-per-case 4 --acoustic-surface impermeable --end-on rev 20 --live-output
```

The durations above are examples, not convergence guarantees. Allow the flow to
settle and retain a sufficiently long acoustic recording. The acoustic calculation
requires five fully arrived rotations, including propagation delay. Insufficient
recordings produce an `insufficient_duration` status rather than a valid spectrum.

## Model selection

New orders explicitly select `--mode`, `--turbulence` and `--wall-functions`.

| Turbulence selection | AMI, resolved walls | AMI, wall functions | MRF, resolved walls | MRF, wall functions |
| --- | --- | --- | --- | --- |
| `kOmegaSST` | Yes | Yes | Yes | Yes |
| `kEpsilon` | LaunderSharmaKE | Standard kEpsilon | LaunderSharmaKE | Standard kEpsilon |
| `DES` | Yes | Unsupported | Unsupported | Unsupported |

- **AMI** rotates the rotor mesh and couples it through NCC in the Foundation
  runtime. Use it for time-resolved rotor flow and aeroacoustics.
- **MRF** keeps the mesh stationary and applies a rotating reference frame in
  `rotaryRegion`. It uses the transient PIMPLE solver with a frozen rotor position.
  It does not reproduce sliding-mesh blade-passing interactions.
- **`--wall-functions no`** selects viscous-sublayer resolution. Design the mesh
  for near-wall resolution and inspect y+; selecting this flag does not refine it.
- **`--wall-functions yes`** selects wall-function boundary conditions. Choose
  compatible near-wall spacing rather than assuming a finer mesh is always sufficient.

Resolved kEpsilon uses the [Launder-Sharma low-Re formulation](https://doc.cfd.direct/notes/cfd-general-principles/low-re-k-epsilon-models).
Resolved SST/DES evaluates its viscous omega boundary condition using runtime-compiled
OpenFOAM code. See [CoreTemplates/README.md](CoreTemplates/README.md) for details.

`--boundary-layers` controls **mesh layer generation**, independently of wall treatment:

| Value | Behavior |
| --- | --- |
| `dict` (default) | Follow the rotor and stator mesh dictionaries |
| `cfmesh` | Enable the complete native rotor workflow, including dictionary-controlled layer subdivision |
| `none` | Disable layers in both regions |

## Run options

| Task | Options |
| --- | --- |
| Several speeds | `--rpms 3000 4000 5000` |
| Cores per case (required for new orders) | `--cores-per-case 20` (`--target-cores` is an alias) |
| Total CPU budget | `--total-cores 8` (`--cores` is an alias) |
| Initialize each RPM from the previous one | `--field-init on` with ascending RPMs |
| Independent cases | `--field-init off` (default) |
| Flow only | `--aerodynamics-only` |
| Solid blade FW-H surface | `--acoustic-surface impermeable` |
| Fixed enclosing FW-H sphere | `--acoustic-surface permeable --acoustic-sphere-diameter 2.5` |
| End after revolutions | `--end-on rev 20` |
| End at simulation time | `--end-on time 0.3` |
| Convergence stopping | `--end-on convergence` (default), `force_convergence`, or `residual_convergence` |
| Full live logs | `--live-output` (omit for the dashboard) |

Sphere diameter is a multiple of propeller diameter. Do not combine acoustic
surface options with `--aerodynamics-only`. Independent cases run concurrently
within the total core budget; field initialization sequences cases by geometry.

### Resume an order

```bash
python main.py --sim-dir ~/simulations/propeller_flow --resume --live-output
```

Resume restores `simulation_order.json`, skips completed stages and retries failed
ones. Stored model, wall treatment, acoustic settings and core allocation take
precedence over replacement options. To change a run configuration, create a new
order. Recreated case folders are preserved with a `_PREVIOUS_<timestamp>` suffix.
Older orders without explicit wall treatment use the preserved legacy templates.

### Parameter studies

Studies require one STL and one RPM. Values are separated by `...`:

```bash
python main.py --sim-dir ~/simulations/mesh_study \
  --rpms 4000 --mode AMI --turbulence kOmegaSST --wall-functions no \
  --total-cores 4 --cores-per-case 4 --mesh-only --study \
  --study-file cfmeshCommon --study-parameter propellerCellSize \
  --study-values '0.00125...0.000625' --live-output
```

Populate this order's `STL/` first. Study files are names from `Parameters/`;
`.cpp` is optional. Nested keys use paths such as
`boundaryLayers/patchBoundaryLayers/propeller/nLayers`. Field initialization is
not supported for studies.

## Case settings

Edit source settings **before creating a new case**. Each case receives a snapshot;
subsequent source edits do not change an existing run.

| Location | Controls |
| --- | --- |
| `Parameters/cfmeshDomainDict` | STL scale, fluid domain and rotor dimensions |
| `Parameters/cfmeshCommon.cpp` | Cell sizes and propeller refinement thickness |
| `Parameters/cfmeshRefinementDict` | Refinement cylinders and spheres |
| `Parameters/cfmeshRotorDict` | Rotor refinement, layers and workflow |
| `Parameters/cfmeshStatorDict` | Stator refinement and workflow |
| `Parameters/controlDict.cpp` | Time stepping, saved-field cadence and retention |
| `Parameters/rotational_parameters.cpp` | Angular velocity, set from the requested RPM |
| `CoreTemplates/<mode>/<model>/<treatment>/` | Initial fields, boundary conditions and solver dictionaries |
| `<case>/visualization.json` | Optional rendering settings |

## Results and reports

```text
<order>/
  STL/                         Input geometry
  simulation_order.json        Stored settings, case status and scheduler state
  <case>.preprocessing.log      Preparation output
  <case>/
    sim.foam                   Open in ParaView
    Parameters/                Case-specific parameter snapshot
    constant/ + system/        Mesh, models and solver configuration
    <time>/                    Saved flow fields and moving-mesh data
    postProcessing/            Loads, diagnostics and acoustic samples
    report/
      simulation_report.pdf
      acoustic-status.json
      spl_spectrum.png         When acoustic prediction succeeds
      visuals/manifest.json    Rendered views, units, ranges and coverage notes
```

For diagnostics, inspect `log.cfmesh`, `log.checkMesh`, `log.checkMesh.NCC`,
`log.pimpleFoam`, `cfmesh/mesh-status.json`, and `cfmesh/ncc-status.json`.
A finished report does not establish convergence or mesh independence: review its
force history, residuals, Courant numbers and wall-treatment statistics.

The report shows hydrodynamic pressure and flow structures separately from the
FW-H observer spectrum. In incompressible OpenFOAM cases, `p` is usually kinematic
pressure, with units m^2/s^2; it must not be labelled Pa without a density conversion.
Pressure RMS, vorticity and Q isosurfaces are not sound-pressure level.

### Configure visuals

For a shorter rendering run, create `<case>/visualization.json` before postprocessing:

```json
{
  "image_resolution": [2400, 1600],
  "surface_phases": 4,
  "volume_phases": 2,
  "statistics_revolutions": 5,
  "wake_stations_D": [-1, -0.5, 0, 0.5, 1],
  "q_over_omega2": [0.1, 0.5, 1],
  "report_max_views": 32,
  "log_fields": ["vorticity_magnitude", "wallShearStress", "dpdt_rms"],
  "required": false
}
```

The PDF selects up to `report_max_views` representative figures across quantities;
all rendered PNGs remain in the archive. Increase the limit to embed more views.
Magnitude fields listed in `log_fields` use a labelled logarithmic color scale;
set this list to `[]` for linear scales throughout.

Phase settings select saved snapshots; they do not create new CFD samples.
Surface statistics still use all samples in their configured window. Set
`diameter_m` for geometry whose filename does not encode a diameter like `10x7E`.
The renderer accepts fixed `color_ranges` for cross-case comparisons; the supported
keys and validation rules are in `tools/visualization.py`.

**Retain the source data to regenerate figures.** Keep the case dictionaries,
`constant/polyMesh`, relevant saved time directories (including moving-mesh files),
`postProcessing`, logs, and the visual manifest/settings. The PDF and screenshots
cannot recover deleted fields or acoustic surface recordings. `purgeWrite` in
`Parameters/controlDict.cpp` controls how many volume writes survive a run.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| Docker unavailable in WSL | Start Docker Desktop and enable integration for the distribution used by this terminal. Check `docker info`. |
| Docker permission/connection error | Ensure the CLI and Python SDK can access the same daemon as your simulation user. |
| Required image not found | Pull both tagged images listed in Installation. |
| Editable acoustic package cannot be installed | Populate submodules and run environment creation from the repository root. |
| `generateBoundaryLayers` not found | Run `bash setup_cfmesh.sh` or set `CFMESH_BIN` to the executable. |
| No STL / missing example input | Place your geometry in `<order>/STL/`; do not depend on the optional example. |
| Existing order rejected | Use `--resume`, or create a separate order for changed settings. |
| Mesh quality checks fail | Inspect the mesh and logs, then adjust geometry/refinement/layers. `--allow-bad-mesh` is a diagnostic override, not a quality fix. |
| Acoustic spectrum unavailable | Inspect `acoustic-status.json`; retain a longer settled recording with propagation delay. |
| Blank or missing visuals | Inspect the visual manifest and `paraview.log`; verify the host's offscreen ParaView build and saved fields. Nearly blank plot areas are rejected. |
| Visual atlas is partial | Read the coverage notes for missing fields, out-of-domain slices or rendering failures. |
| Terminal reports a stale working directory | Change to `/`, then back to the repository and retry. |

## Repository guide and maintenance

| File/folder | Responsibility |
| --- | --- |
| `main.py` | CLI entry point and simulation-order dispatch |
| `preprocessing.py` | Case creation and geometry preparation |
| `cfmesh.py` | Meshing-stage entry point |
| `openfoamSimulation.py` | Container execution, meshing, solving and reconstruction |
| `postprocessing.py` | Acoustic, visualization and report stages |
| `acoustic_propagation.py` | FW-H observer prediction |
| `visualization.py` | ParaView visualization entry point |
| `createSimulationReport.py` | PDF report assembly |
| `tools/` | Implementation helpers, scheduling and runtime management |
| `tests/` | Pipeline regressions and optional Docker integration tests |
| `acousticSolver/` | FoamAcoustics submodule |

Local checks that do not launch a flow simulation:

```bash
python -m unittest discover -s tests -v
PARAVIEW_RENDER_TESTS=1 python -m unittest discover -s tests -p test_visualization.py -v
```

Docker integration checks are opt-in as shown in Installation. Tests that read the
bundled example require `examples/STL/10x7E.stl`. When changing the renderer, also
render representative cell/point fields, slices and mesh views in the target
ParaView runtime and inspect the resulting images and PDF.
