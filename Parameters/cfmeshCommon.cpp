// Shared physical sizes. These are cfMesh controls, not snappy refinement levels.
backgroundCellSize 0.02; // 0.02
propellerCellSize 0.000625;
propellerRefinementThickness 0.002; // matches both original level-5 distance bands
interfaceCellSize 0.0125;

// Region cell sizes. Original level ordering: rotor/inner 3; outer/sphere 2.
// Retain the current 0.01 m background and existing propeller/interface sizes.
rotaryRegionCellSize 0.00625;
innerCylinderCellSize 0.00625;
outerCylinderCellSize 0.0125;
acousticSphereCellSize 0.025;// 0.0125
