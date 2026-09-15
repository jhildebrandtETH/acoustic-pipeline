// Nominal sizes for inspection. Native refinements use additionalRefinementLevels
// directly; do not supply these cellSize values alongside native level entries.
// Change cfmeshCommon.cpp for resolution studies.
// Keeping expressions here lets foamDictionary edit the inputs without freezing
// calculated sizes into cfmeshCommon.cpp.
#include "cfmeshCommon.cpp"
backgroundCellSize $baseCellSize;
propellerCellSize #eval "$baseCellSize / pow(2, $propellerLevel)";
interfaceCellSize #eval "$baseCellSize / pow(2, $interfaceLevel)";
rotaryRegionCellSize #eval "$baseCellSize / pow(2, $rotaryRegionLevel)";
innerCylinderCellSize #eval "$baseCellSize / pow(2, $innerCylinderLevel)";
outerCylinderCellSize #eval "$baseCellSize / pow(2, $outerCylinderLevel)";
acousticSphereCellSize #eval "$baseCellSize / pow(2, $acousticSphereLevel)";
