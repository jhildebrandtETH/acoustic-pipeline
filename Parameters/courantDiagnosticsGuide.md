# Diagnosing Courant-limited cells

`diagnoseCourant.py` analyses an existing, reconstructed simulation snapshot. It
does not run or stop the solver, change settings, or modify mesh/solution files.
Outputs go in a new `postProcessing/courantDiagnostics/<timestamp>` directory.

## Capture a useful snapshot

Run until the adaptive time step is persistently limited and save a complete
time directory. A maximum Co near the configured maxCo (e.g. 5) is expected under
adaptive stepping; reaching 5 alone is not evidence of divergence. Keep the solver
log, and ideally snapshots before and during the plateau. Finish writing or stop
cleanly before analysing. An unwritten peak cannot be recovered from the log alone.

The AMI templates already contain the CourantNo function object. Verify it is
also present in the actual simulation's system/controlDict functions dictionary:

```cpp
CourantNumber
{
    type CourantNo;
    libs ("libfieldFunctionObjects.so");
    executeControl timeStep;
    executeInterval 1;
    writeControl writeTime;
}
```

Keep the resulting cell field (normally `Co`) and `U`. If the run is parallel,
reconstruct the desired saved time with the matching OpenFOAM version, including
the moving mesh and Co/U fields. For example, in that OpenFOAM environment:

```sh
reconstructPar -case /home/jonas/run/apc/10x7E_4000RPM_AMI -latestTime
```

The script reads the mesh at the selected time, preserves polyhedral cell IDs,
and refuses silently falling back from newer processor results to older root
results. It supports the single assembled internal mesh, not separate multi-region
meshes or direct processor partitions. NCC/reader compatibility errors must be
resolved before trusting output; no geometry repair or format conversion is done.
Co and U are read directly from their ASCII internalField entries (gzip is
supported), bypassing NCC virtual boundary-field incompatibilities in VTK.
Binary solution fields must first be exported to an ASCII copy with OpenFOAM.

## Run

Install optional diagnostic dependencies in your Python environment:

```sh
python -m pip install numpy vtk
```

From the repository directory in WSL:

```sh
python diagnoseCourant.py /home/jonas/run/apc/10x7E_4000RPM_AMI --time latest --threshold 4 --top 100
```

Or from Windows with Python installed:

```powershell
python diagnoseCourant.py '\\wsl.localhost\Ubuntu\home\jonas\run\apc\10x7E_4000RPM_AMI' --time latest --threshold 4 --top 100
```

Add `--log /path/to/log.solver` to include time-step history. Supply the actual log
filename. `--time 0.015` selects an exact saved time. `--co-field NAME` handles
custom output names. `--target-co 1` estimates a time-step multiplier assuming
unchanged fluxes; no absolute deltaT is guessed from configuration or log timing.

The default rotation axis is Y through (0,0,0), matching this pipeline. Change
`--axis`, `--origin X Y Z`, or `--tip-radius METRES` for other configurations.
Otherwise tip radius is estimated from the saved propeller boundary.

## Results

- `report.md`: readable summary and limitations.
- `summary.json`: whole-mesh Courant/velocity statistics and optional log history.
- `hotspots.csv`: worst cells with centre, volume, edge lengths, velocity, neighbour
  comparison, propeller distance and radial location.
- `hotspots.vtu`: worst cells and immediate face neighbours for ParaView. Colour
  by Co; `hotspotRank > 0` identifies ranked cells. `originalCellId` retains IDs.

The threshold counts all cells above the limit; `--top` bounds the detailed
extraction independently. Reports contain observations, not an automatic claim
that layers or the trailing edge caused the restriction. Blade-relative trailing
edge membership is not inferred from fixed coordinates on a rotating blade.
VTK volumes and edge ratios are diagnostic estimates, not replacements for
OpenFOAM checkMesh quality fields. Compare the extracted cells with the blade and
feature curves, then decide whether geometry, layers, transition or velocity is
responsible. Provide the case/report path for follow-up analysis.
