castellatedMesh true;
snap true;
addLayers true;

maxLocalCells 100000;
maxGlobalCells 3000000;
minRefinementCells 0;
maxLoadUnbalance 0.10;
nCellsBetweenLevels 3;

rotaryRegionSurfaceRefinementLevel (3 3);

propellerSurfaceRefinementLevel (5 5); // Uniform propeller surface refinement.

propellerRefinementRegionMode distance;
propellerRefinementRegionLevel ((0.0045 5) (0.0125 4));

rotaryRegionRefinementRegionMode inside;
rotaryRegionRefinementRegionLevel ((1E15 3));

innerCylinderRefinementRegionMode inside;
innerCylinderRefinementRegionLevel ((1E15 3));

outerCylinderRefinementRegionMode inside;
outerCylinderRefinementRegionLevel ((1E15 2));

acousticSurfaceRefinementRegionMode inside;
acousticSurfaceRefinementRegionLevel ((1E15 2));

nSmoothPatch 3;
tolerance 1.0;
nSolveIter 30;
nRelaxIter 20;
nFeatureSnapIter 10;
implicitFeatureSnap false;
explicitFeatureSnap true;
multiRegionFeatureSnap false;

// Native snappyHexMesh layers with absolute dimensions in metres.
// Three layers with expansionRatio 1.2 give approximately:
// first layer = 0.153 mm, final layer = 0.220 mm,
// total layer stack = 0.556 mm where all three layers are retained.
relativeSizes false;
propellerTipSurfaceLayers 3;
expansionRatio 1.2;
finalLayerThickness 2.2e-4;
minThickness 7.0e-5;
nGrow 0;
featureAngle 60;
addLayersnRelaxIter 5;
nSmoothSurfaceNormals 3;
nSmoothNormals 3;
nSmoothThickness 10;
maxFaceThicknessRatio 0.5;
maxThicknessToMedialRatio 0.2;
minMedialAxisAngle 90;
nLayerIter 100;
nBufferCellsNoExtrude 0;

maxNonOrtho 65;
maxBoundarySkewness 4;
maxInternalSkewness 4;
maxConcave 80;
minVol 1e-13;
mergeTolerance 1e-6;
nSmoothScale 4;
