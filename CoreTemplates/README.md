# Core templates

Select a complete case with `--mode AMI|MRF`, `--turbulence kOmegaSST|kEpsilon|DES`,
and `--wall-functions yes|no`. Templates are arranged as
`<mode>/<turbulence>/wall-functions/` or `<mode>/<turbulence>/wall-resolved/`.

There are nine supported combinations: both wall treatments for SST and k-epsilon
in both rotation modes, plus wall-resolved DES in AMI only.

- AMI: moving rotor mesh, NCC coupling.
- MRF: stationary mesh, NCC coupling, rotaryRegion MRF zone, MRFnoSlip propeller.
- Wall functions: nutkWallFunction, kqRWallFunction and omegaWallFunction or epsilonWallFunction.
- Resolved SST/DES: native fixedValue conditions, zero wall k/nut and prescribed wall omega.
- Resolved k-epsilon: LaunderSharmaKE low-Re formulation, zero wall k/modified epsilon/nut.

Mesh layer generation is controlled by `Parameters/cfmeshRotorDict` and
`Parameters/cfmeshStatorDict`, independently of the wall-treatment selection.
The mesh must resolve the viscous sublayer for the resolved templates; inspect y+.
Wall treatment changes only OpenFOAM dictionaries. Both choices use the same
preprocessing, coupling and solver workflow, with no custom boundary code or
runtime compilation.

For resolved SST/DES, edit `Parameters/wallResolvedDict`: `omegaWallValue` is a
complete OpenFOAM field value (`uniform ...` or `nonuniform List<scalar> ...`).
Choose it from `60*nu/(0.075*y1^2)`, where `y1` is the wall-to-first-cell-centre
distance. The supplied `1.92e7 1/s` is an example for air with `nu=1.5e-5 m2/s`
and `y1=25 micrometres`, not an automatic mesh-dependent value. A cfMesh
first-layer thickness cap does not guarantee that cell-centre distance.
For nonuniform spacing, prescribe face values in propeller patch order.
Changing viscosity or near-wall spacing requires updating this dictionary.

Existing generated cases retain their copied dictionaries. Regenerate the case
from these templates to replace the former `codedFixedValue` omega boundary;
resuming an already prepared case alone does not update its initial fields.
Legacy templates are retained unchanged under `legacy/` for resuming older orders.
Do not use legacy templates to start new orders.

Model and boundary-condition references:
- https://doc.cfd.direct/notes/cfd-general-principles/low-re-k-epsilon-models
- https://cpp.openfoam.org/v13/classFoam_1_1fixedValueFvPatchField.html
- https://cpp.openfoam.org/v13/MRFnoSlipFvPatchVectorField_8C_source.html

MRF uses a frozen rotor orientation. It does not capture sliding-mesh blade-passing
interactions. Impermeable acoustics retains the existing rotating-source reconstruction
from reference geometry; using MRF loads makes this a frozen-loading approximation.
Use AMI for time-resolved aeroacoustic analysis.
