# Acoustic Pipeline

End-to-end propeller CFD and aeroacoustic analysis: prepare OpenFOAM cases from
STL geometry, mesh with cfMesh, solve the flow, predict FW-H observer sound, and
export figures and a PDF report. Supports RPM sweeps, parameter studies, parallel
cases, and resumable simulation orders.

```text
STL + settings -> case preparation -> cfMesh + mesh checks -> flow solution
                                            |                    |
                                       mesh report        loads + flow plots
                                                                 |
                                                       optional FW-H acoustics
                                                                 |
                                                          figures + PDF
```

[Installation](#installation) · [Quick start](#quick-start) · [Run controls](#run-controls)
· [Configuration](#configuration) · [Results](#results) · [Troubleshooting](#troubleshooting)

## Installation

### 1. Prepare Linux or Windows with WSL 2

The full pipeline runs in **Linux**, including Linux under **WSL 2**. Native
Windows Python is not supported for launching orders because order locking uses
Linux system calls. A host OpenFOAM installation and a GPU are not required.

**Linux:** install Git, [Conda](https://www.anaconda.com/docs/getting-started/miniconda/install),
and [Docker Engine](https://docs.docker.com/engine/install/) for your distribution.
On Ubuntu, Git can be installed with `sudo apt update` followed by
`sudo apt install -y git`.

**Windows:** install Ubuntu through WSL from **PowerShell as administrator**:

```powershell
wsl --install -d Ubuntu
```

Restart when prompted, open Ubuntu, and create your Linux user. In PowerShell,
check `wsl --list --verbose`; Ubuntu must show version **2**. If needed, run
`wsl --set-version Ubuntu 2`. See the [Microsoft WSL installation guide](https://learn.microsoft.com/en-us/windows/wsl/install).

Install [Docker Desktop](https://docs.docker.com/desktop/features/wsl/) on Windows,
enable **Use the WSL 2 based engine** and **Resources > WSL Integration > Ubuntu**,
and keep Docker Desktop running in Linux container mode. Install Git and Conda
**inside Ubuntu**. For this Docker Desktop setup, avoid a second Docker Engine
installation inside Ubuntu.

**All remaining commands run in a Linux/Ubuntu terminal**, from the repository
root unless stated otherwise. Keep the repository and simulation orders in the
Linux filesystem, such as `~/acoustic-pipeline` and `~/simulations`, for efficient
Docker file access. Windows files are accessible through paths such as
`/mnt/c/Users/<name>/Downloads/propeller.stl`.

Verify Docker access as the same Linux user that will launch the pipeline:

```bash
docker info
docker run --rm hello-world
```

Allow enough CPU, RAM, and disk space for meshes, simultaneous cases, and saved
fields. The pipeline's core budget does not limit memory; on WSL, ensure the
resources available to WSL/Docker match the intended workload.

### 2. Clone and install the Python environment

```bash
cd ~
git clone --recurse-submodules https://github.com/jhildebrandtETH/acoustic-pipeline.git
cd acoustic-pipeline
conda env create -f of_pipeline_env.yml
conda activate of_pipeline_env
python -m pip check
python main.py --help
```

For an existing checkout, run `git submodule update --init --recursive` before
creating the environment. The [environment file](of_pipeline_env.yml) installs
Python 3.12, CPU PyTorch, and the pipeline dependencies, including the editable
[FoamAcoustics submodule](acousticSolver/README.md). Preserve the repository and
submodule revisions with the environment file for reproducibility.

### 3. Pull the two OpenFOAM runtimes

```bash
docker pull opencfd/openfoam-default:2512
docker pull microfluidica/openfoam:13
python -m tools.cfmesh_runtime --smoke-test
```

| Image | Purpose |
| --- | --- |
| `opencfd/openfoam-default:2512` | Native cfMesh utilities and dictionary operations |
| `microfluidica/openfoam:13` | Mesh assembly, non-conformal coupling (NCC), checks, and flow solution |

Both images are required. The pipeline configures OpenFOAM inside the containers
and mounts case files automatically. The smoke test checks runtime utilities and
mounted-file access without building a mesh.

### 4. Set up ParaView for report visuals

Install [ParaView](https://www.paraview.org/download/) with `pvpython` or `pvbatch`
on the simulation host. Use its bundled Python runtime, not a Conda/pip package.
If the executable is not on `PATH`, configure it before launching:

```bash
export PARAVIEW_EXECUTABLE=/absolute/path/to/paraview/bin/pvpython
"$PARAVIEW_EXECUTABLE" --version
```

Rendering requires a build and graphics libraries that support offscreen use.
On WSL, the pipeline also discovers Windows ParaView under
`/mnt/c/Program Files/ParaView*` and translates paths when Windows interop is
available. Linux ParaView is preferred. ParaView is optional for solving;
missing or failed visuals are recorded in the report and visual manifest.

## Quick start

### Prepare geometry and settings

An **order** is a dedicated output directory containing `STL/`, stored run
settings, and generated cases. Each input STL is run at every requested RPM.
Supply a closed, watertight propeller with its shaft on the **+Y axis through the
origin**. `Parameters/cfmeshDomainDict` uses `scale 1` for metres; use `0.001` for
STL coordinates in millimetres. Review mesh sizing, boundary conditions, and wall
settings in [Configuration](#configuration) before creating cases.

```bash
mkdir -p ~/simulations/propeller_mesh/STL
cp /path/to/propeller.stl ~/simulations/propeller_mesh/STL/
```

An empty order uses `examples/STL/10x7E.stl` only if that example exists in your
checkout. Supply your own geometry to avoid this dependency. Use dedicated output
folders; source, parameter, and template directories are protected.

### Inspect a mesh

```bash
python main.py --sim-dir ~/simulations/propeller_mesh \
  --rpms 4000 --mode AMI --turbulence kOmegaSST --wall-functions no \
  --total-cores 4 --cores-per-case 4 --mesh-only
```

This stops after mesh generation, checks, and a mesh PDF report. Open the case's
`sim.foam` in ParaView and inspect the blade surface, boundary layers,
rotor/stator interface, and mesh-quality results before running the flow.

### Run aerodynamics or aeroacoustics

Create a new order for each run configuration. For **aerodynamics only**:

```bash
mkdir -p ~/simulations/propeller_flow/STL
cp ~/simulations/propeller_mesh/STL/*.stl ~/simulations/propeller_flow/STL/
python main.py --sim-dir ~/simulations/propeller_flow \
  --rpms 4000 --mode MRF --turbulence kOmegaSST --wall-functions no \
  --total-cores 4 --cores-per-case 4 --aerodynamics-only --end-on rev 20
```

For **time-resolved aeroacoustics**:

```bash
mkdir -p ~/simulations/propeller_acoustics/STL
cp ~/simulations/propeller_mesh/STL/*.stl ~/simulations/propeller_acoustics/STL/
python main.py --sim-dir ~/simulations/propeller_acoustics \
  --rpms 4000 --mode AMI --turbulence kOmegaSST --wall-functions no \
  --total-cores 4 --cores-per-case 4 --acoustic-surface impermeable \
  --end-on rev 20 
```

These durations are examples, not convergence guarantees. Retain a settled
acoustic recording of at least five rotations plus propagation delay; inspect
`report/acoustic-status.json` for the actual outcome. The current acoustic wrapper
uses an observer at **(1, 0, 0) m** and a two-blade spectrum setting in
[acoustic_propagation.py](acoustic_propagation.py); adapt these for other setups.

## Run controls

New orders require `--sim-dir`, `--rpms`, `--mode`, `--turbulence`,
`--wall-functions`, `--total-cores`, and `--cores-per-case`. Flow runs additionally
require either `--aerodynamics-only` or `--acoustic-surface`.

| Purpose | Options |
| --- | --- |
| RPM sweep | `--rpms 3000 4000 5000` |
| CPU allocation | `--total-cores 8 --cores-per-case 4`: up to two concurrent cases |
| Initialize from preceding RPM | `--field-init on`; requires ascending RPMs, sequences each geometry |
| Independent cases | `--field-init off` (default) |
| Mesh inspection / flow only | `--mesh-only` / `--aerodynamics-only` |
| Solid-blade FW-H surface | `--acoustic-surface impermeable` |
| Fixed enclosing FW-H sphere | `--acoustic-surface permeable --acoustic-sphere-diameter 2.5` |
| Stop after revolutions / seconds | `--end-on rev 20` / `--end-on time 0.3` |
| Stop by convergence | `--end-on convergence` (default), `force_convergence`, or `residual_convergence` |
| Stream logs instead of dashboard | `--live-output` |

Sphere diameter is a multiple of propeller diameter. Acoustic surface options
cannot be combined with `--aerodynamics-only`. Unused remainder cores stay idle;
field-initialization dependencies may reduce concurrency. `--cores` and
`--target-cores` are aliases for total cores and cores per case, respectively.
Use `python main.py --help` for the full option reference.

### Resume

```bash
python main.py --sim-dir ~/simulations/propeller_flow --resume
```

Resume restores `simulation_order.json`, skips completed stages, and retries
failed stages after the underlying issue is fixed. Saved settings take precedence;
changing the core allocation is rejected. Use a new order to change the model,
run type, or settings, including progressing from a mesh-only order to a flow run.
Only one launcher may own an order at a time. Recreated case folders are preserved
with a `_PREVIOUS_<timestamp>` suffix. Legacy orders may require an explicit
total-core budget for migration and retain their legacy wall-treatment templates.

### Parameter studies

Populate `~/simulations/mesh_study/STL/` with **exactly one STL**, then run:

```bash
python main.py --sim-dir ~/simulations/mesh_study \
  --rpms 4000 --mode AMI --turbulence kOmegaSST --wall-functions no \
  --total-cores 4 --cores-per-case 4 --mesh-only --study \
  --study-file cfmeshCommon --study-parameter propellerLevel \
  --study-values '4...5...6'
```

Studies require one RPM and independent cases (`--field-init off`). File names
refer to `Parameters/`, with `.cpp` optional. Separate values with `...`; use
slash-separated paths for nested dictionary keys.

## Configuration

Edit source settings **before case creation**. Each case receives a snapshot;
source edits do not update prepared cases when resuming.

### Flow model and wall treatment

| Turbulence | AMI, resolved | AMI, wall functions | MRF, resolved | MRF, wall functions |
| --- | --- | --- | --- | --- |
| `kOmegaSST` | Supported | Supported | Supported | Supported |
| `kEpsilon` | LaunderSharmaKE | Standard kEpsilon | LaunderSharmaKE | Standard kEpsilon |
| `DES` | Supported | Unsupported | Unsupported | Unsupported |

**AMI** moves the rotor mesh with NCC coupling; use it for time-resolved flow and
acoustics. **MRF** uses a stationary mesh and rotating reference frame with transient
PIMPLE; it represents a frozen rotor position. MRF impermeable acoustics is a
frozen-loading approximation and does not capture sliding-mesh interactions.

`--wall-functions no` requires viscous-sublayer resolution; `yes` requires spacing
appropriate for wall functions. Neither option refines the mesh. Inspect y+.
For resolved SST/DES, set `omegaWallValue` in `Parameters/wallResolvedDict` for the
actual viscosity and wall-to-first-cell-centre distance: `60*nu/(0.075*y1^2)`.
The supplied value is an example, not automatically inferred from the mesh.
See [template and boundary-condition details](CoreTemplates/README.md).

### Settings reference

| Location | Controls |
| --- | --- |
| `Parameters/cfmeshDomainDict` | STL scale, domain margins, inlet/wake split, rotor dimensions |
| `Parameters/cfmeshCommon.cpp` | Base cell size, refinement levels, blade layers and optimisation |
| `Parameters/cfmeshGlobalDict` | Shared native cfMesh settings |
| `Parameters/cfmeshRefinementDict`, `cfmeshFeatureDict` | Volume refinements and explicit features |
| `Parameters/cfmeshRotorDict`, `cfmeshStatorDict` | Region refinement, layer generation, workflow |
| `Parameters/cfmeshPipelineDict` | Per-region quality improvement, interface projection, acceptance thresholds |
| `Parameters/controlDict.cpp` | Time steps, field writes, retention (`purgeWrite`) |
| `Parameters/wallResolvedDict` | Prescribed omega for resolved SST/DES |
| `Parameters/rotational_parameters.cpp` | Angular velocity, set from requested RPM |
| `Parameters/meshReport.json` | Optional first-layer report patch and target thickness |
| `CoreTemplates/<mode>/<model>/<treatment>/` | Initial fields, boundary conditions, transport and solver settings |
| `<case>/visualization.json` | Optional case-specific rendering settings |

Layer generation occurs inside native `cartesianMesh`, independently of solver
wall treatment. Optional `improveMeshQuality` runs afterward with separate rotor
and stator controls. For detailed mesh setup and studies, see the
[cfMesh editing guide](Parameters/cfmeshGuide.md) and
[boundary-layer guide](Parameters/cfmeshLayerGuide.md).

## Results

```text
<order>/
  STL/                          Input geometry
  simulation_order.json         Saved settings, case status, scheduler state
  <case>.preprocessing.log       Preparation log
  <case>/
    sim.foam                    ParaView entry point
    Parameters/                 Case-specific settings snapshot
    constant/ + system/         Mesh, models, solver dictionaries
    <time>/                     Saved fields and moving-mesh data
    postProcessing/             Loads, diagnostics, acoustic samples
    cfmesh/                     Geometry and mesh/NCC status
    report/
      simulation_report.pdf     Mesh, flow, and available acoustic results
      acoustic-status.json      Acoustic outcome (flow runs)
      spl_spectrum.png          When acoustic prediction succeeds
      visuals/manifest.json     Visual coverage, units, settings, failures
```

Inspect `log.cfmesh`, `log.checkMesh`, `log.checkMesh.NCC`, `log.pimpleFoam`,
`cfmesh/mesh-status.json`, and `cfmesh/ncc-status.json` for diagnostics. Reports
include mesh quality and first-layer thickness analysis where data is available.
Review force histories, residuals, Courant numbers, and y+; successful execution
alone does not establish convergence or mesh independence.

Flow pressure and FW-H sound are distinct quantities. In incompressible cases,
`p` is generally kinematic pressure (m²/s²), requiring density conversion to Pa.
Pressure RMS, vorticity, and Q isosurfaces are not sound-pressure level.

### Visuals and data retention

Create `<case>/visualization.json` before postprocessing to override defaults:

```json
{
  "image_resolution": [2400, 1600],
  "surface_phases": 4,
  "volume_phases": 2,
  "statistics_revolutions": 5,
  "report_max_views": 32,
  "required": false
}
```

Set `required` to `true` to fail postprocessing on missing/partial visuals, or
`enabled` to `false` to disable rendering. Set `diameter_m` when a geometry name
does not encode diameter (such as `10x7E`). `color_ranges` supports consistent
cross-case comparisons; `log_fields: []` selects linear scales. All accepted keys
and validation rules are in [tools/visualization.py](tools/visualization.py).
Phase counts select existing snapshots; they do not add CFD samples.

The PDF selects representative views; all rendered PNGs remain available.
**Keep source data to regenerate reports:** case dictionaries, `constant/polyMesh`,
relevant time directories including moving-mesh files, `postProcessing`, logs,
and visualization settings/manifest. `purgeWrite` limits retained volume writes;
figures and PDFs cannot recover deleted fields or acoustic recordings.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| Docker unavailable in WSL | Start Docker Desktop, enable the correct distribution's integration, and check `docker info`. |
| Docker permission/connection error | Ensure the CLI and Python Docker SDK can access the same daemon as the simulation user. |
| Runtime image missing | Pull both tagged images in Installation. |
| Acoustic package installation fails | Initialize submodules and create the environment from the repository root. |
| Missing STL/example | Supply geometry in `<order>/STL/`. |
| Existing order rejected | Use `--resume`, or a fresh order for changed settings. |
| Mesh checks fail | Inspect geometry, mesh, and logs; adjust refinement/layers. `--allow-bad-mesh` is a diagnostic override, not a fix. |
| Spectrum unavailable | Read `report/acoustic-status.json` and retain a longer settled recording with propagation delay. |
| Missing/partial visuals | Read `report/visuals/manifest.json` and the associated `paraview.log`; check offscreen rendering and saved fields. |
| Stale working directory | Run `cd /`, then re-enter the repository and retry. |

## Development and further documentation

The workflow is `main.py` → `preprocessing.py` → `openfoamSimulation.py` →
`postprocessing.py`. Meshing enters through `cfmesh.py`; acoustics, visuals, and
PDF creation are handled by `acoustic_propagation.py`, `visualization.py`, and
`createSimulationReport.py`. See the [helper-module guide](tools/README.md),
[Courant diagnostics guide](Parameters/courantDiagnosticsGuide.md), and
[FoamAcoustics documentation](acousticSolver/README.md).

Run checks in the activated Linux environment:

```bash
python -m unittest discover -s tests -v
# Optional integration checks, with Docker or ParaView available:
CFMESH_DOCKER_TESTS=1 python -m unittest discover -s tests -v
PARAVIEW_RENDER_TESTS=1 python -m unittest discover -s tests -p test_visualization.py -v
```

Some tests require `examples/STL/10x7E.stl`. These checks verify software behavior;
validate representative meshes, flow results, and rendered reports for your setup.
