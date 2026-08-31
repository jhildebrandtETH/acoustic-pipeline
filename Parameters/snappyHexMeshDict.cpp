castellatedMesh true;
snap true;
addLayers false; // OBSERVE CFMESH CONFIG

maxLocalCells 100000;
maxGlobalCells 3000000;
minRefinementCells 0;
maxLoadUnbalance 0.10;
nCellsBetweenLevels 2;

rotaryRegionSurfaceRefinementLevel (3 3);

propellerSurfaceRefinementLevel (5 5);// y+ targeting variable


propellerRefinementRegionMode distance;
propellerRefinementRegionLevel ((0.003 5) (0.01 4));

rotaryRegionRefinementRegionMode inside;
rotaryRegionRefinementRegionLevel ((1E15 3));

innerCylinderRefinementRegionMode inside;
innerCylinderRefinementRegionLevel ((1E15 2));

outerCylinderRefinementRegionMode inside;
outerCylinderRefinementRegionLevel ((1E15 2));


nSmoothPatch 3;
tolerance 1.0;
nSolveIter 30;
nRelaxIter 20;
nFeatureSnapIter 10;
implicitFeatureSnap true;
explicitFeatureSnap true;
multiRegionFeatureSnap true;


relativeSizes false;
propellerTipSurfaceLayers 3;
expansionRatio 1.0;
firstLayerThickness 0.001; // y+ targting variable
minThickness 0.0002;
nGrow 0;
featureAngle 180;
addLayersnRelaxIter 5;
nSmoothSurfaceNormals 5; //1
nSmoothNormals 10; //3
nSmoothThickness 30; //20
maxFaceThicknessRatio 5.0; // 2.0
maxThicknessToMedialRatio 2.0; //1.0
minMedialAxisAngle 15; //30
nLayerIter 100; //50
nBufferCellsNoExtrude 0;

maxNonOrtho 65;
maxBoundarySkewness 4;
maxInternalSkewness 4;
maxConcave 80;
minVol 1e-13;
mergeTolerance 1e-6;
nSmoothScale 4;
