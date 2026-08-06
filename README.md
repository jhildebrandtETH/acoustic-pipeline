# UAV Propeller CFD Pipeline

## Overview

Automated pipeline for generating meshes and running OpenFOAM simulations of UAV propellers.

### Features

- MRF (steady) and AMI (transient) simulation modes 
- kEpsilon and kOmegaSST turbulence are supported
- Fully automated meshing (blockMesh + snappyHexMesh)  
- Convergence monitoring (thrust + residuals)  
- Postprocessing & PDF report generation  
- Parameter study support  
- Robust resume capability  

---

## Requirements

- Docker Engine **running**
  - Docker image: microfluidica/openfoam:13
- Conda environment:

```bash
conda env create -f of_pipeline_env.yml
conda activate of_pipeline_env
docker info
```

---

## CLI Design

The CLI is structured into:

### 1. Configuration Parameters (define the simulation setup)

```bash
--sim-dir        Output directory
--geometries     STL names (without .stl)
--rpms           Rotation speeds
--mode           AMI | MRF
--cores          Number of cores
--field-init     on | off (default: on)
--turbulence     kOmegaSST | kEpsilon
--end-on         time | force_convergence | residual_convergence | convergence
```

### 2. Feature Flags (activated if present)

```bash
--study          Enable parameter study
--resume         Resume existing simulation batch
--mesh-only      Stop after mesh generation
--allow-bad-mesh Neglects bad mesh checks before solving
```

---

## Basic Usage

```bash
python main.py   --sim-dir <path>   --geometries <list>   --rpms <list>   --mode <AMI|MRF> --end-on <time|convergence|force_convergence|residual_convergence>  --cores <int> --turbulence <kOmegaSST|kEpsilon>  --field-init <on|off>
```


---

## Turbulence Models & Wall Treatment

### kOmegaSST (Low-Re, no wall functions)

- Fully resolves the boundary layer
- Requires **y+ ≈ 1**
- Higher computational cost
- Used for accurate simulations

---

### kEpsilon (Wall function approach)

- Uses wall functions
- Does not resolve viscous sublayer
- Requires **y+ ≈ 30–100**
- Faster and more robust

---

## Important Concept

- Turbulence model and wall treatment are conceptually different
- In this pipeline:

```
kOmegaSST → no wall functions
kEpsilon  → wall functions
```

This is handled automatically via template selection.

---

## Template Selection

Templates follow:

```
Core Template <MODE> - <TURBULENCE>
```

Examples:

- Core Template AMI - kOmegaSST
- Core Template MRF - kEpsilon

---


## Configuration Options

### `--field-init`

Controls sequential initialization between RPM cases.

- `on` → initialize from previous RPM result  
- `off` → start each case independently  

---

## Feature Flags

### `--resume`

Resumes an interrupted simulation batch.

```bash
python main.py --sim-dir <path> --resume
```

### `--mesh-only`

Stops pipeline after mesh generation.

Use this for:
- Mesh tuning
- y⁺ validation
- Pre-solver checks

Solver must be started manually afterward.

---

## Resume Feature

The pipeline automatically resumes simulations:

### Behavior

- Detects latest valid timestep (ignores `0`)
- Verifies required fields (`U`, `p`)
- Skips corrupted/incomplete timesteps
- Continues from last valid state

### Important ⚠️

> **One simulation run = one directory**

Always use a new folder for a new run.

---

## Study Mode

Enable parameter studies with:

```bash
--study
```

### Required Arguments

```bash
--study-parameter <name>
--study-file <file>
--study-values <values>
```

### Value Format

Values are separated by `...`

Examples:

```bash
7...8...9
(10 12 10)...(12 14 12)
```

---

## Study Behavior

- Requires exactly **one geometry** and **one RPM**
- Creates one simulation per value
- Folder naming:

```bash
<geometry>_<rpm>RPM_<parameter>_<value>
```

---

## End-On Feature

- specifies what event shoud trigger each simulation to end:

- time : standard endTime defined in controlDict (no monitoring)
- force_convergence : only monitors thrust convergence
- residual_convergence : only monitors residual convergence
- convergence : monitors both force and residual convergence (first priority thurst convergence, then residual convergence)

```bash
<geometry>_<rpm>RPM_<parameter>_<value>
```

---

## Examples

### Standard Run

```bash
python main.py   --sim-dir /scratch/simulations   --geometries 10x7E   --rpms 7000   --mode AMI  --end-on convergence --cores 24 --turbulence kOmegaSST
```

---

## Examples

### High-fidelity (kOmegaSST & AMI)

```bash
python main.py \
  --sim-dir /scratch \
  --geometries 10x7E \
  --rpms 7000 \
  --mode AMI \
  --cores 24 \
  --turbulence kOmegaSST
```

---

### Fast (kEpsilon & MRF)

```bash
python main.py \
  --sim-dir /scratch \
  --geometries 10x7E \
  --rpms 7000 \
  --mode MRF \
  --cores 24 \
  --turbulence kEpsilon
```

---

### Resume Run

```bash
python main.py   --sim-dir /scratch/simulations   --resume
```

---

### Mesh-Only Run

```bash
python main.py   --sim-dir /scratch/simulations   --geometries 10x7E   --rpms 7000   --mode AMI --end-on force_convergence  --turbulence kOmegaSST --cores 24   --mesh-only
```

---

### Allow-Bad-Mesh Run

```bash
python main.py   --sim-dir /scratch/simulations   --geometries 10x7E   --rpms 7000   --mode AMI -end-on force_convergence --turbulence kOmegaSST --cores 24   --allow-bad-mesh
```

---

### Parameter Study

```bash
python main.py   --sim-dir /scratch/simulations   --geometries 10x7E   --rpms 7000   --mode AMI --turbulence kOmegaSST -end-on force_convergence  --cores 24   --study   --study-parameter refinementLevel   --study-file snappyHexMeshDict   --study-values 3...4...5
```

---

## Notes

- STL files must be located in `STLs/`
- Names must match exactly (e.g. `10x7E.stl`)
- RPM order matters if `--field-init on`
- Simulation parameters are defined in `Parameters/`

### Simulation Modes

- **AMI** → more accurate, transient, slower  
- **MRF** → faster, steady, good for initial studies  

---

## Summary

- Use **configuration options** to define the simulation  
- Use **flags** to activate pipeline features  
- Keep simulation runs isolated per directory  

---

## Future Extensions

Planned improvements:

- Automated y⁺ targeting