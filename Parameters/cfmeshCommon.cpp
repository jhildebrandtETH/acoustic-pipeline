// User inputs: nominal cell size = baseCellSize / 2^level.
// Levels pass directly to cfMesh as additionalRefinementLevels.
// Nonnegative integer levels; level 0 is the background resolution.
// Derived native cell sizes are in cfmeshSizes.cpp; edit inputs here for studies.
// 10x7E target: fine propeller/inner wake with a coarse acoustic far field.
// All nominal cell sizes scale with this base; see cfmeshLayerGuide.md.
baseCellSize 0.022;
propellerLevel 4;
interfaceLevel 3;
rotaryRegionLevel 3;
innerCylinderLevel 3;
outerCylinderLevel 2;
acousticSphereLevel 0;
propellerRefinementThickness 0.008; // Fine-region width, not first-cell height.

// Propeller layers: native cfMesh controls referenced by cfmeshRotorDict.
propellerLayerCount 3; // Robust cfMesh stack; inspect measured thickness and solved y+.
propellerLayerThicknessRatio 1.3;
propellerLayerAllowDiscontinuity 0;
// Metres. 1e30 is effectively uncapped for this geometry; reduce for a cap.
// This is an upper bound, not an exact first-layer height or total thickness.
propellerMaxFirstLayerThickness 1e30;

// Native layer optimisation (whole rotor region, before layer subdivision).
// Optional thickness optimisation shrank the layers in the measured trials.
// Keep untangling and the separate improveMeshQuality pass enabled.
layerOptimise 0;
layerUntangle 1;
layerSmoothNormalsIterations 10;
layerMaxIterations 15; // Quality trial: increased from 5; compare full checkMesh results.
layerFeatureSizeFactor 0.1; // Curvature-based thickness limit, 0 <= value < 1.
layerRecalculateNormals 1;
layerRelativeThicknessTolerance 0.08; // 0 <= value < 1; lower enforces smoother thickness.
