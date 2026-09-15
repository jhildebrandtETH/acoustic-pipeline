"""Sphere containment, axial allocation and level-based resolution regressions."""

import math
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from tools import cfmesh_parameters as parameters
from tools.cfmesh_pipeline import ROOT, native_run, query


class ParameterTests(unittest.TestCase):
    def setUp(self):
        self.values = {
            "boxSizing": "sphereRelative", "lateralMargin": "0.02",
            "inletMargin": "0.02", "inletFraction": "0.3",
            "sphereDiameterFactor": "2.5", "baseCellSize": "0.02",
            **{region + "Level": "2" for region in parameters.REGIONS},
        }
        self.addCleanup(patch.stopall)
        patch.object(parameters, "query", side_effect=lambda path, entry: self.values[entry]).start()
        patch.object(parameters, "optional", side_effect=lambda path, entry: self.values.get(entry)).start()

    def test_scaled_box_contains_sphere_and_preserves_30_70_split(self):
        for diameter in (0.1, 0.254, 0.3048, 0.5, 1.5):
            for sphere_factor in (2.5, 3.0):
                with self.subTest(diameter=diameter, sphere_factor=sphere_factor):
                    radius = parameters.effective_sphere_radius("unused", diameter, "permeable", sphere_factor)
                    lower, upper, settings = parameters.domain_bounds("unused", radius)
                    self.assertTrue(all(v < -radius for v in lower))
                    self.assertTrue(all(v > radius for v in upper))
                    self.assertAlmostEqual(upper[1] / (upper[1] - lower[1]), 0.3)
                    self.assertAlmostEqual(upper[0] - radius, 0.02 * radius)
                    self.assertAlmostEqual(lower[1], -2.38 * radius)
                    self.assertAlmostEqual(settings["centre_y_m"], -0.68 * radius)

    def test_effective_sphere_uses_cli_only_for_permeable_mode(self):
        self.assertAlmostEqual(parameters.effective_sphere_radius("unused", 0.4, "permeable", 3), 0.6)
        self.assertAlmostEqual(parameters.effective_sphere_radius("unused", 0.4, "impermeable", 3), 0.5)
        self.assertAlmostEqual(parameters.effective_sphere_radius("unused", 0.4, "permeable", None), 0.5)
        for value in (0, -2, math.nan, math.inf):
            with self.assertRaises(ValueError):
                parameters.effective_sphere_radius("unused", 0.4, "permeable", value)

    def test_invalid_relative_boxes(self):
        for args in (
            (1, 0, 0.02, 0.3), (1, -0.01, 0.02, 0.3),
            (1, 0.02, 0, 0.3), (1, 0.02, 0.02, 0),
            (1, 0.02, 0.02, 1), (1, 0.02, 0.02, 0.9),
            (math.inf, 0.02, 0.02, 0.3),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                parameters.relative_box(*args)

    def test_legacy_fixed_box_and_absolute_sizes(self):
        self.values.pop("boxSizing")
        self.values.update(boxMin="(-0.385 -0.650 -0.385)", boxMax="(0.385 0.385 0.385)")
        lower, upper, settings = parameters.domain_bounds("unused", 0.3175)
        self.assertEqual(lower, [-0.385, -0.65, -0.385])
        self.assertEqual(upper, [0.385] * 3)
        self.assertEqual(settings["mode"], "absolute")
        self.values.pop("baseCellSize")
        self.values.update({region + "CellSize": "0.0125" for region in ("background", *parameters.REGIONS)})
        resolution = parameters.resolution_settings("unused")
        self.assertEqual(resolution["mode"], "absolute")
        self.assertEqual(resolution["cell_sizes_m"]["propeller"], 0.0125)

    def test_levels_and_base_changes_recompute_every_size(self):
        self.values["propellerLevel"] = "5"
        first = parameters.resolution_settings("unused")
        self.assertAlmostEqual(first["cell_sizes_m"]["propeller"], 0.000625)
        self.values["baseCellSize"] = "0.04"
        second = parameters.resolution_settings("unused")
        for region, size in first["cell_sizes_m"].items():
            self.assertAlmostEqual(second["cell_sizes_m"][region], 2 * size)
        self.values["propellerLevel"] = "6"
        self.assertAlmostEqual(parameters.resolution_settings("unused")["cell_sizes_m"]["propeller"], 0.000625)

    def test_invalid_levels_and_base_sizes(self):
        for level in ("-1", "2.5", "nan", "99999"):
            self.values["propellerLevel"] = level
            with self.assertRaisesRegex(ValueError, "propellerLevel"):
                parameters.resolution_settings("unused")
        self.values["propellerLevel"] = "5"
        for base in ("0", "-0.1", "nan", "inf"):
            self.values["baseCellSize"] = base
            with self.assertRaisesRegex(ValueError, "baseCellSize"):
                parameters.resolution_settings("unused")


@unittest.skipUnless(os.environ.get("CFMESH_DOCKER_TESTS") == "1", "requires OpenFOAM Docker runtime")
class NativeParameterTests(unittest.TestCase):
    def test_study_edits_recompute_native_surface_and_volume_sizes(self):
        with tempfile.TemporaryDirectory(prefix="cfmesh-levels-") as tmp:
            folder = Path(tmp) / "Parameters"
            shutil.copytree(ROOT / "Parameters", folder)
            common = folder / "cfmeshCommon.cpp"
            self.assertAlmostEqual(int(query(folder / "cfmeshRotorDict", "localRefinement/propeller/additionalRefinementLevels")), 5)
            for key, value in (("baseCellSize", "0.04"), ("propellerLevel", "6")):
                native_run(["foamDictionary", common, "-entry", key, "-set", value], cwd=folder, check=True, capture_output=True)
            # Editing the input file must not freeze the expressions in the derived file.
            self.assertAlmostEqual(float(query(folder / "cfmeshRotorDict", "maxCellSize")), 0.04)
            self.assertAlmostEqual(int(query(folder / "cfmeshRotorDict", "localRefinement/propeller/additionalRefinementLevels")), 6)
            self.assertAlmostEqual(float(query(folder / "cfmeshSizes.cpp", "innerCylinderCellSize")), 0.01)
            self.assertAlmostEqual(float(query(folder / "cfmeshSizes.cpp", "acousticSphereCellSize")), 0.04)
            self.assertEqual(query(folder / "cfmeshStatorDict", "workflowControls/stopAfter"), "edgeExtraction")


@unittest.skipUnless(os.environ.get("CFMESH_GEOMETRY_TESTS") == "1", "requires trimesh and OpenFOAM Docker runtime")
class GeometryParameterTests(unittest.TestCase):
    def test_large_propeller_preparation_uses_shared_sphere_bounds_and_study_levels(self):
        import trimesh
        from tools.cfmesh_pipeline import prepare_geometry

        with tempfile.TemporaryDirectory(prefix="cfmesh-geometry-") as tmp:
            case = Path(tmp) / "case"
            shutil.copytree(ROOT / "Parameters", case / "Parameters")
            # A closed synthetic blade wider than the former 308 mm domain limit.
            source = Path(tmp) / "blade.stl"
            trimesh.creation.box(extents=[0.4, 0.01, 0.01]).export(source)
            refinement_file = case / "Parameters/cfmeshRefinementDict"
            refinement_file.write_text(refinement_file.read_text().replace("sphereSubdivisions 6;", "sphereSubdivisions 1;"))
            prepare_geometry(case, source, "permeable", 3.0,
                             study=("cfmeshCommon", "propellerLevel", "6"))
            geometry = json.loads((case / "cfmesh/geometry.json").read_text())
            radius = 1.5 * geometry["diameter_m"]
            self.assertAlmostEqual(geometry["acoustic_sphere_radius_m"], radius)
            self.assertAlmostEqual(geometry["box_min"][1], -2.38 * radius)
            self.assertAlmostEqual(geometry["box_max"][0], 1.02 * radius)
            self.assertEqual(geometry["resolution"]["levels"]["propeller"], 6)
            self.assertAlmostEqual(geometry["refinement_regions"]["acousticSurface"]["radius"], radius)
            self.assertAlmostEqual(geometry["refinement_regions"]["innerCylinder"]["cellSize"], 0.005)
            generated = (case / "Parameters/cfmeshRegions.generated").read_text()
            self.assertNotIn("cellSize", generated)
            self.assertIn("additionalRefinementLevels 0;", generated)
            self.assertTrue((case / "Parameters/cfmeshDomain.generated").is_file())
            self.assertTrue((case / "constant/triSurface/permeableSurface.stl").is_file())
            for role in ("rotor", "stator"):
                mesh = case / "cfmesh" / role / "system/meshDict"
                self.assertAlmostEqual(float(query(mesh, "objectRefinements/acousticSurface/radius")), radius)
                self.assertAlmostEqual(int(query(mesh, "objectRefinements/innerCylinder/additionalRefinementLevels")), 2)
            self.assertAlmostEqual(int(query(case / "cfmesh/rotor/system/meshDict", "localRefinement/propeller/additionalRefinementLevels")), 6)


if __name__ == "__main__":
    unittest.main()
