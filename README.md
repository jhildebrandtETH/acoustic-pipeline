# UAV Propeller CFD & Aeroacoustic Pipeline

Automated Python pipeline for preparing, meshing, solving, monitoring, postprocessing, and reporting transient UAV-propeller CFD simulations with **OpenFOAM 13**.

The pipeline is designed for **high-throughput simulation orders**: the user provides a total CPU-core budget, and the scheduler automatically distributes those cores across as many independent cases as possible. Every CFD case runs in its own Docker container and its own case directory.

> **Current implementation status:** the active preprocessing workflow supports **AMI only**. `MRF` is still accepted by the CLI for legacy compatibility, but preprocessing currently rejects it. Use `--mode AMI` unless MRF support is reimplemented.

---

## 1. What the Pipeline Does

A simulation order can contain multiple propeller geometries, RPM values, or parameter-study cases. For every case, the pipeline can automatically perform:

1. case preprocessing and template creation,
2. `blockMesh`,
3. `surfaceFeatures`,
4. `decomposePar` when more than one core is assigned,
5. `snappyHexMesh` in serial or parallel,
6. `checkMesh` and mesh diagnostics,
7. AMI/NCC creation using `createNonConformalCouples`,
8. optional field initialization from a lower-RPM case using `mapFields`,
9. transient OpenFOAM solution using `foamRun -solver incompressibleFluid`,
10. reconstruction and safe processor-folder cleanup,
11. acoustic postprocessing,
12. force/residual/y+ merging,
13. SPL-spectrum generation, and
14. PDF simulation-report generation.

The scheduler runs several of these case workflows simultaneously whenever the available CPU budget allows it.

---

## 2. Requirements

### Software

- Python / Conda
- Docker Engine **running**
- Docker image:

```text
microfluidica/openfoam:13
```

- the local `acousticSolver` package/submodule required by `acoustic_propagation.py`

A typical environment setup is:

```bash
conda env create -f of_pipeline_env.yml
conda activate of_pipeline_env

git submodule update --init --recursive

docker info
```

`docker info` must succeed before the pipeline is started.

### Linux Docker permissions

The pipeline starts and removes Docker containers automatically. On Linux, the user therefore needs permission to access the Docker daemon. The OpenFOAM container is started with the host UID/GID so files written into bind-mounted case directories remain owned by the user.

---

## 3. Repository Structure

The pipeline source directory is expected to contain the Python scripts, OpenFOAM templates, shared parameter files, and acoustic solver.

```text
acoustic-pipeline/
├── main.py
├── tools.py
├── preprocessing.py
├── openfoamSimulation.py
├── postprocessing.py
├── acoustic_propagation.py
├── createSimulationReport.py
├── Parameters/
├── Core Template AMI - kOmegaSST/
├── Core Template AMI - kEpsilon/
├── Core Template DES - kOmegaSST/
├── acousticSolver/
└── of_pipeline_env.yml
```

### Main Python files

| File | Responsibility |
|---|---|
| `main.py` | CLI, simulation-order creation/resume, scheduler startup |
| `tools.py` | scheduler, status dashboard, JSON persistence, monitors, OpenFOAM helpers, resume logic, report helpers |
| `preprocessing.py` | prepares exactly one simulation case |
| `openfoamSimulation.py` | executes the OpenFOAM lifecycle of exactly one case |
| `postprocessing.py` | coordinates acoustic postprocessing, merged data, and report generation |
| `acoustic_propagation.py` | FW-H acoustic prediction and SPL spectrum |
| `createSimulationReport.py` | creates the final PDF report |

The individual workflow scripts intentionally contain only their main public function. Reusable helper logic is centralized in `tools.py`.

---

## 4. Simulation-Order Directory

`--sim-dir` is **not the source-code repository**. It is the directory for one simulation order.

Before starting a new order, create:

```text
my_simulation_order/
├── STL/
│   ├── 10x7E.stl
│   └── 11x7E.stl
└── FEATURES/
    ├── 10x7E_tip.obj
    ├── 10x7E_other.obj
    ├── 11x7E_tip.obj
    └── 11x7E_other.obj
```

The pipeline automatically uses **every `.stl` file inside `STL/`**. There is no separate `--geometries` CLI argument anymore.

For each geometry `<name>.stl`, the corresponding feature files must be:

```text
<name>_tip.obj
<name>_other.obj
```

### Geometry naming requirement

The current preprocessing code derives the propeller diameter from the part of the geometry name before `x`.

For example:

```text
10x7E.stl → diameter = 10 in
11x7E.stl → diameter = 11 in
```

Therefore geometry names used with the current automatic geometry setup must start with a numeric propeller diameter in inches.

### One order = one directory

When a new order starts, the pipeline creates:

```text
simulation_order.json
```

inside `--sim-dir`. A directory containing an existing order cannot be reused for another new order. Either create a new directory or use `--resume`.

---

## 5. Quick Start

Example with two geometries in `STL/`, three requested RPM values, and 72 total CPU cores:

```bash
python main.py \
  --sim-dir /path/to/my_simulation_order \
  --rpms 3000 5000 7000 \
  --mode AMI \
  --turbulence DES \
  --total-cores 72 \
  --field-init off \
  --end-on time \
  --acoustic-surface impermeable
```

With two STLs and three RPM values, this creates six CFD cases.

Because `--field-init off` makes all six cases independent, the scheduler can run all six simultaneously and distribute the 72-core budget between them.

---

# 6. Parallel Scheduler

## `--total-cores`

`--total-cores` defines the CPU budget for the **complete simulation order**, not for one individual case.

```bash
--total-cores 100
```

The scheduler determines:

- how many cases may run simultaneously,
- how many cores each case receives, and
- which queued case starts when another case finishes.

`--cores` is still accepted as a backward-compatible alias for `--total-cores`, but new commands should use `--total-cores` because its meaning is unambiguous.

### Examples

| Cases | Total cores | Initial execution |
|---:|---:|---|
| 1 | 100 | 1 case × 100 cores |
| 10 | 100 | 10 cases × 10 cores |
| 50 | 100 | 50 cases × 2 cores |
| 100 | 100 | 100 cases × 1 core |
| 200 | 100 | 100 cases × 1 core, 100 queued |

If 200 cases are requested with 100 cores, the scheduler does **not** wait for the first group of 100 to finish completely. As soon as one running case finishes, its slot is immediately reused by the next eligible queued case.

### Uneven allocation

If the total number of cores cannot be divided exactly, the scheduler distributes the remainder across the first slots.

Example:

```text
100 total cores / 3 parallel cases
→ 34 + 33 + 33 cores
```

The assigned core count is stored per case in `simulation_order.json` and is used consistently for `numberOfSubdomains`, meshing, AMI creation, and the solver.

### Serial and MPI cases

A case assigned one core runs the relevant OpenFOAM commands serially.

```text
1 core  → serial OpenFOAM
2+ cores → decomposePar + MPI/OpenFOAM -parallel
```

The pipeline therefore does not launch unnecessary `mpirun -np 1` jobs.

---

# 7. Field Initialization

## Default: `--field-init off`

```bash
--field-init off
```

This is the default and the recommended mode for **maximum throughput**.

Every geometry/RPM case is independent and may run as soon as a scheduler slot is available.

Example:

```text
10 geometries × 5 RPM values = 50 independent cases
100 total cores            = 50 simultaneous cases × 2 cores
```

## `--field-init on`

```bash
--field-init on
```

This creates a separate RPM dependency chain for each geometry.

Example:

```text
10x7E: 3000 → 4000 → 5000 → 6000 → 7000 RPM
11x7E: 3000 → 4000 → 5000 → 6000 → 7000 RPM
 9x9E: 3000 → 4000 → 5000 → 6000 → 7000 RPM
```

The first RPM case of each geometry may run immediately. A higher-RPM case becomes eligible as soon as the previous RPM case **of the same geometry** has completed successfully.

There is no global RPM barrier. For example, `10x7E @ 4000 RPM` can start as soon as `10x7E @ 3000 RPM` is complete even if `11x7E @ 3000 RPM` is still running.

The completed predecessor is copied into the new case's `init/` directory during preprocessing and then mapped with:

```text
mapFields /simulation/init/ -consistent -sourceTime latestTime
```

### Requirements for field initialization

- RPM values must be supplied in ascending order.
- Duplicate RPM values are rejected.
- `--field-init on` cannot be combined with `--study`.

### Failed initialization source

If a predecessor case fails, dependent cases are marked `BLOCKED` rather than silently starting without their requested initialization.

For example:

```text
10x7E 3000 RPM → FAILED
10x7E 4000 RPM → BLOCKED
10x7E 5000 RPM → BLOCKED
```

Other independent geometry chains continue normally.

---

# 8. Live Batch Monitor

The terminal is managed by one central dashboard rather than by individual simulation workers.

The dashboard is grouped into:

```text
RUNNING CASES

QUEUED / WAITING CASES

FAILED / BLOCKED CASES   # only shown when required
```

A typical row contains:

```text
Case | Cores | Stage | Progress | Detail
```

Stages can include:

```text
preprocessing
blockMesh
surfaceFeatures
decomposePar
snappyHexMesh
checkMesh
createNonConformalCouples
mapFields
solving
reconstructing
cleanup
acoustics
report
postprocessing
```

For `--end-on time`, solver progress can be displayed as a percentage of configured `endTime`.

For convergence-controlled simulations, the final stop time is unknown in advance, so the monitor reports the current convergence information instead of inventing a completion percentage.

### Interactive terminal vs. SLURM/log files

In an interactive terminal the dashboard refreshes in place. When stdout is redirected, for example into a SLURM log, ANSI screen clearing is disabled and periodic snapshots are printed instead.

Detailed OpenFOAM output remains available in each case's `log.*` files even though the main terminal shows only the compact batch overview.

---

# 9. CLI Reference

## Required for a new non-mesh-only order

```text
--sim-dir
--rpms
--mode
--turbulence
--total-cores
--acoustic-surface
```

## Main configuration options

| Argument | Values / type | Default | Meaning |
|---|---|---|---|
| `--sim-dir` | path | required | directory containing `STL/` and `FEATURES/`; case folders are created here |
| `--rpms` | one or more integers | required | RPM values to simulate |
| `--mode` | `AMI`, `MRF` | required | current preprocessing implementation supports **AMI only** |
| `--turbulence` | `kEpsilon`, `kOmegaSST`, `DES` | required | selects the corresponding OpenFOAM template |
| `--total-cores` | integer ≥ 1 | required | total CPU budget for the complete order |
| `--cores` | integer ≥ 1 | — | legacy alias for `--total-cores` |
| `--field-init` | `on`, `off` | `off` | enable/disable same-geometry RPM initialization chains |
| `--end-on` | see below | `convergence` | condition used to terminate the CFD solver |
| `--acoustic-surface` | `impermeable`, `permeable` | required for normal run | FW-H surface type |
| `--acoustic-sphere-diameter` | float | `2.5` in permeable mode | permeable sphere diameter divided by propeller diameter |

## Feature flags

| Flag | Meaning |
|---|---|
| `--resume` | resume an existing simulation order |
| `--mesh-only` | stop after mesh generation/reconstruction; skip solver and postprocessing |
| `--allow-bad-mesh` | allow a case to continue when `checkMesh` does not report `Mesh OK` |
| `--study` | create a parameter-study order |

---

# 10. Solver Termination: `--end-on`

Available modes:

```text
time
force_convergence
residual_convergence
convergence
```

### `time`

Runs until the configured OpenFOAM `endTime`.

This mode provides a straightforward time-progress percentage in the dashboard.

### `force_convergence`

Monitors the thrust-based convergence criterion and stops the solver when the configured force-stability condition is satisfied.

### `residual_convergence`

Monitors residual behavior over the latest propeller revolution and stops when the residual criterion is satisfied.

### `convergence`

Requires the force criterion first and then checks residual convergence.

The detailed numerical convergence settings and residual slope bounds are implemented in `tools.py`. These should be treated as part of the simulation methodology and reviewed before changing them.

---

# 11. Turbulence Models and Templates

The current AMI preprocessing maps turbulence selections to these templates:

```text
kOmegaSST → Core Template AMI - kOmegaSST
kEpsilon  → Core Template AMI - kEpsilon
DES       → Core Template DES - kOmegaSST
```

Wall treatment, turbulence boundary conditions, discretization schemes, and related OpenFOAM settings are defined by the selected template and shared `Parameters/` files.

For reproducible studies, changes to those templates should be version-controlled together with the pipeline.

---

# 12. Acoustic Surface Options

## Impermeable

```bash
--acoustic-surface impermeable
```

Uses the propeller surface data written by the OpenFOAM function objects.

Do **not** provide `--acoustic-sphere-diameter` in impermeable mode.

## Permeable

```bash
--acoustic-surface permeable
```

Creates an enclosing spherical acoustic surface.

If no sphere factor is supplied, the default is:

```text
sphere diameter = 2.5 × propeller diameter
```

A custom value can be provided with:

```bash
--acoustic-surface permeable \
--acoustic-sphere-diameter 3.0
```

The value represents the **sphere diameter divided by propeller diameter**.

### Current acoustic postprocessing defaults

The current `acoustic_propagation.py` uses:

```text
observer position: [1.0, 0.0, 0.0] m
SPL spectrum:      5 rotations
blade count:       2
```

The acoustic solver automatically uses CUDA when PyTorch detects a CUDA device; otherwise it runs on CPU.

These values are currently code-level settings rather than CLI arguments and should be checked when applying the pipeline to a different propeller or observer configuration.

---

# 13. Resume

Resume an existing order with:

```bash
python main.py \
  --sim-dir /path/to/my_simulation_order \
  --resume
```

The configuration is read from `simulation_order.json`; the original RPMs, turbulence model, mode, field-init setting, acoustic configuration, and core allocation do not need to be supplied again.

For current scheduler-schema orders, changing the total core budget during `--resume` is intentionally disabled because cases may already be decomposed using the stored allocation.

## What happens to an interrupted solver case

The resume path can:

1. locate the latest safe timestep,
2. reconstruct decomposed history up to that timestep,
3. verify reconstruction completeness,
4. verify required reconstructed fields,
5. preserve processor folders if validation fails,
6. remove old processor folders only after validation,
7. decompose again when the case is assigned more than one core, and
8. continue the OpenFOAM solution.

Failed and dependency-blocked cases are reactivated when `--resume` is used, allowing the scheduler to reevaluate the dependency chain after the underlying problem has been corrected.

### Legacy orders

Old `simulation_order.json` files used `cores` to mean **cores per case**. The new scheduler uses a **total** core budget, so this cannot be converted safely by guessing.

For the first resume of a legacy order, explicitly provide:

```bash
python main.py \
  --sim-dir /path/to/legacy_order \
  --resume \
  --total-cores 72
```

The order is then migrated to the new scheduler metadata.

---

# 14. Parameter Studies

Study mode creates independent cases in which one parameter is varied.

Requirements:

- exactly one STL in `STL/`,
- exactly one RPM value,
- `--field-init off`, and
- all three study arguments below.

```text
--study-file
--study-parameter
--study-values
```

Example:

```bash
python main.py \
  --sim-dir /path/to/refinement_study \
  --rpms 7000 \
  --mode AMI \
  --turbulence DES \
  --total-cores 72 \
  --field-init off \
  --end-on time \
  --acoustic-surface impermeable \
  --study \
  --study-file snappyHexMeshDict \
  --study-parameter propellerTipRegionLevel \
  --study-values 5...6...7
```

`--study-file snappyHexMeshDict` refers to:

```text
Parameters/snappyHexMeshDict.cpp
```

Do not include `.cpp` in the CLI value.

Study values are separated by literal `...`, for example:

```text
5...6...7
```

or:

```text
(8 24 8)...(16 48 16)...(32 96 32)
```

Generated case folders follow approximately:

```text
<geometry>_<rpm>RPM_<parameter>_<value>
```

Study cases are independent and are therefore scheduled for maximum throughput.

---

# 15. Mesh-Only Mode

Use:

```bash
--mesh-only
```

to perform meshing and mesh reconstruction without running the CFD solver or acoustic/report postprocessing.

Typical uses:

- refinement development,
- `checkMesh` evaluation,
- mesh-quality debugging,
- feature-refinement validation, and
- visualization before expensive CFD runs.

Example:

```bash
python main.py \
  --sim-dir /path/to/mesh_test \
  --rpms 7000 \
  --mode AMI \
  --turbulence kOmegaSST \
  --total-cores 24 \
  --mesh-only \
  --acoustic-surface impermeable
```

> **Current implementation note:** `main.py` does not formally require `--acoustic-surface` for `--mesh-only`, but `preprocessing.py` still configures the acoustic function-object switches. Until that is changed, provide an acoustic surface in mesh-only commands as shown above.

---

# 16. `--allow-bad-mesh`

Normally a case stops when `checkMesh` does not report:

```text
Mesh OK
```

For debugging or deliberately permissive studies, this check can be bypassed with:

```bash
--allow-bad-mesh
```

This only bypasses the pipeline's stop decision. It does **not** make a poor-quality mesh numerically safe.

---

# 17. What Is Created for Each Case

A case directory contains the prepared OpenFOAM case, logs, solver results, postprocessing output, and reports.

Typical structure:

```text
10x7E_7000RPM_AMI/
├── 0/
├── constant/
├── system/
├── Parameters/
├── postProcessing/
├── report/
│   ├── spl_spectrum.png
│   ├── simulation_report.pdf
│   └── additional report figures
├── log.blockMesh
├── log.surfaceFeatures
├── log.decomposePar
├── log.snappyHexMesh
├── log.checkMesh
├── log.createNonConformalCouples
├── log.pimpleFoam
├── log.reconstructPar
└── sim.foam
```

Some files are only present when the corresponding stage was executed.

### Main outputs

- `sim.foam` — convenient ParaView entry point
- `log.pimpleFoam` — complete CFD solver log
- `postProcessing/` — OpenFOAM function-object results
- `report/spl_spectrum.png` — acoustic SPL spectrum
- `report/simulation_report.pdf` — consolidated CFD/acoustic report

After successful reconstruction, `processor*` folders are deleted only when the pipeline's reconstruction-integrity checks pass. If validation fails, processor data is deliberately preserved for recovery and manual inspection.

---

# 18. `simulation_order.json`

`simulation_order.json` is the durable state of the complete batch.

It stores information such as:

```text
simulation configuration
case list
case status
field-initialization dependencies
total core budget
per-case core allocation
study settings
acoustic settings
resume information
errors / blocked dependencies
```

Typical persistent case states include:

```text
pending
preprocessing_done
solver_running
solver_done
postprocessing_done
failed
blocked
```

Do not manually edit statuses while a scheduler process is running.

The live dashboard contains additional temporary information such as current solver percentage or reconstruction progress. That information is intentionally kept in memory and reconstructed from files after a restart rather than continuously written to JSON.

---

# 19. Execution Model in One Diagram

```text
                         TOTAL CPU BUDGET
                               │
                               ▼
                     Python batch scheduler
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
           Case A           Case B           Case C
         N_A cores        N_B cores        N_C cores
              │                │                │
              ▼                ▼                ▼
          Docker A         Docker B         Docker C
              │                │                │
              ▼                ▼                ▼
          OpenFOAM A       OpenFOAM B       OpenFOAM C
              │                │                │
              └────────────────┼────────────────┘
                               │
                               ▼
                    central runtime registry
                               │
                               ▼
                     self-refreshing dashboard
```

One case owns one case directory and one Docker container. The scheduler decides **when** it may run and **how many cores** it receives; `openfoamSimulation.py` is responsible only for executing that one approved case correctly.

---

# 20. Common Errors

### `simulation_order.json already exists`

The directory already belongs to an existing order.

Use a new `--sim-dir` or:

```bash
--resume
```

### `Feature file not found`

Check that every STL has both files:

```text
FEATURES/<geometry>_tip.obj
FEATURES/<geometry>_other.obj
```

### Docker connection error

Check:

```bash
docker info
```

and verify Docker permissions.

### `--field-init on requires RPM values in ascending order`

Use for example:

```bash
--rpms 3000 4000 5000 6000 7000
```

not an arbitrary sequence.

### Cases are `WAITING_INIT`

This is normal in field-init mode. The case is waiting for its preceding RPM case of the same geometry.

### Cases are `BLOCKED`

A required initialization predecessor failed. Correct the parent-case problem and use `--resume`.

### `Unsupported OpenFOAM mode: MRF`

The CLI still exposes the legacy `MRF` choice, but the current preprocessing implementation supports AMI only.

### Mesh fails the pipeline check

Inspect:

```text
log.checkMesh
```

Use `--allow-bad-mesh` only when continuing despite that result is intentional.

---

# 21. Recommended First Test After Code Changes

Before launching a large server order, run a small test with one or two geometries and a small number of RPM cases.

Check that:

1. `simulation_order.json` is created correctly,
2. the expected core allocation is shown,
3. separate Docker containers start for independent cases,
4. the RUNNING and QUEUED tables behave correctly,
5. case-local `log.*` files are written,
6. completed cases release their scheduler slot,
7. field-init dependencies behave as expected when enabled, and
8. `--resume` can recover an intentionally interrupted test case.

Only then scale the same configuration to the full available core budget.

---

# 22. Development Notes

Python bytecode/cache files should not be committed.

Recommended `.gitignore` entries:

```gitignore
__pycache__/
*.py[cod]
```

Large simulation orders and generated OpenFOAM case data should normally live outside the source repository.

When modifying the pipeline, preserve the current separation of responsibilities:

```text
main.py                → user entry point / scheduler startup
tools.py               → reusable infrastructure and helpers
preprocessing.py       → prepare one case
openfoamSimulation.py  → run one OpenFOAM case
postprocessing.py      → postprocess one completed case
```

This keeps the execution scripts readable while the reusable scheduling, monitoring, resume, parsing, and reporting logic remains centralized.

---

## Minimal Command Templates

### Maximum-throughput run

```bash
python main.py \
  --sim-dir <order-directory> \
  --rpms <rpm1> <rpm2> ... \
  --mode AMI \
  --turbulence <kOmegaSST|kEpsilon|DES> \
  --total-cores <total-available-cores> \
  --field-init off \
  --end-on <time|force_convergence|residual_convergence|convergence> \
  --acoustic-surface <impermeable|permeable>
```

### Field-initialized run

```bash
python main.py \
  --sim-dir <order-directory> \
  --rpms 3000 4000 5000 6000 7000 \
  --mode AMI \
  --turbulence <kOmegaSST|kEpsilon|DES> \
  --total-cores <total-available-cores> \
  --field-init on \
  --end-on <time|force_convergence|residual_convergence|convergence> \
  --acoustic-surface <impermeable|permeable>
```

### Resume

```bash
python main.py \
  --sim-dir <existing-order-directory> \
  --resume
```

---

## Summary

The pipeline is built around three ideas:

1. **one simulation order describes all requested cases,**
2. **one scheduler distributes a total CPU budget for maximum throughput,** and
3. **one isolated Docker/OpenFOAM workflow executes each individual case.**

Use `--field-init off` when throughput is the priority. Use `--field-init on` when higher-RPM cases should inherit the converged field of the preceding RPM case of the same geometry.

For debugging, always start with the central dashboard for the batch overview and then inspect the affected case's `log.*` files for detailed OpenFOAM output.
