# Core templates

Select a complete case with `--mode AMI|MRF`, `--turbulence kOmegaSST|kEpsilon|DES`,
and `--wall-functions yes|no`. Templates are arranged as
`<mode>/<turbulence>/wall-functions/` or `<mode>/<turbulence>/wall-resolved/`.

There are nine supported combinations: both wall treatments for SST and k-epsilon
in both rotation modes, plus wall-resolved DES in AMI only.

- AMI: moving rotor mesh, NCC coupling.
- MRF: stationary mesh, NCC coupling, rotaryRegion MRF zone, MRFnoSlip propeller.
- Wall functions: nutkWallFunction, kqRWallFunction and omegaWallFunction or epsilonWallFunction.
- Resolved SST/DES: zero wall k/nut; coded viscous omega condition using case viscosity and wall-normal cell distance.
- Resolved k-epsilon: LaunderSharmaKE low-Re formulation, zero wall k/modified epsilon/nut.

Mesh layer generation is controlled by `Parameters/cfmeshRotorDict` and
`Parameters/cfmeshStatorDict`, independently of the wall-treatment selection.
The mesh must resolve the viscous sublayer for the resolved templates; inspect y+.
Legacy templates are retained unchanged under `legacy/` for resuming older orders.
Do not use legacy templates to start new orders.

Model and boundary-condition references:
- https://doc.cfd.direct/notes/cfd-general-principles/low-re-k-epsilon-models
- https://cpp.openfoam.org/v13/codedFixedValueFvPatchField_8H_source.html
- https://cpp.openfoam.org/v13/MRFnoSlipFvPatchVectorField_8C_source.html

MRF uses a frozen rotor orientation. It does not capture sliding-mesh blade-passing
interactions. Impermeable acoustics retains the existing rotating-source reconstruction
from reference geometry; using MRF loads makes this a frozen-loading approximation.
Use AMI for time-resolved aeroacoustic analysis.
