// User inputs: nominal cell size = baseCellSize / 2^level.
// Levels pass directly to cfMesh as additionalRefinementLevels.
// Nonnegative integer levels; level 0 is the background resolution.
// Derived native cell sizes are in cfmeshSizes.cpp; edit inputs here for studies.
baseCellSize 0.025;
propellerLevel 4;        // 0.000625 m
interfaceLevel 2;        // 0.01 m, used on both interface sides
rotaryRegionLevel 3;     // 0.005 m
innerCylinderLevel 2;    // 0.005 m
outerCylinderLevel 1;    // 0.01 m
acousticSphereLevel 0;   // 0.02 m
propellerRefinementThickness 0.0015; // matches both original level-5 distance bands

// Propeller layers: native cfMesh controls referenced by cfmeshRotorDict.
propellerLayerCount 5;
propellerLayerThicknessRatio 1.1;
propellerLayerAllowDiscontinuity 0;
// Metres. 1e30 is effectively uncapped for this geometry; reduce for a cap.
// This is an upper bound, not an exact first-layer height or total thickness.
propellerMaxFirstLayerThickness 1e30;

// Native layer optimisation (whole rotor region, before layer subdivision).
// Enable for a controlled comparison; does not prescribe total layer thickness.
layerOptimise 1;
layerUntangle 1;
layerSmoothNormalsIterations 5;
layerMaxIterations 15; // Quality trial: increased from 5; compare full checkMesh results.
layerFeatureSizeFactor 0.2; // Curvature-based thickness limit, 0 <= value < 1.
layerRecalculateNormals 1;
layerRelativeThicknessTolerance 0.08; // 0 <= value < 1; lower enforces smoother thickness.
