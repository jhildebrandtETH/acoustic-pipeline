"""Targeted Docker command/path regressions; no daemon required."""

import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import tools.cfmesh_pipeline as pipeline
import tools.cfmesh_runtime as runtime


class RuntimeTests(unittest.TestCase):
    def test_external_case_includes_and_literal_dictionary_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp).resolve()
            (case / "Parameters").mkdir()
            region = case / "cfmesh/rotor"
            region.mkdir(parents=True)
            dictionary = region / "system/meshDict"
            value = "{ stopAfter edgeExtraction; }"
            argv = runtime.command(
                ["foamDictionary", dictionary, "-entry", "workflowControls", "-set", value],
                cwd=region, cores=3,
            )
            self.assertIn(f"type=bind,source={case},target={runtime.container_path(case)}", argv)
            self.assertIn(runtime.container_path(dictionary), argv)
            self.assertEqual(argv[-1], value)
            self.assertIn("OMP_NUM_THREADS=3", argv)
            self.assertEqual(argv[argv.index("--workdir") + 1], runtime.container_path(region))
            self.assertIn('set --; source ', argv[argv.index("-c") + 1])
            if hasattr(os, "getuid"):
                self.assertEqual(argv[argv.index("--user") + 1], f"{os.getuid()}:{os.getgid()}")

    def test_runtime_separation(self):
        native = runtime.command(["cartesianMesh", "-help"])
        foundation = runtime.command(["checkMesh", "-help"], image=runtime.FOUNDATION_IMAGE)
        self.assertIn(runtime.IMAGE, native)
        self.assertNotIn(runtime.FOUNDATION_IMAGE, native)
        self.assertIn(runtime.FOUNDATION_IMAGE, foundation)
        self.assertIn("/opt/openfoam13/etc/bashrc", foundation[foundation.index("-c") + 1])

    def test_invalid_core_count(self):
        with self.assertRaises(ValueError):
            runtime.command(["cartesianMesh"], cores=0)

    def test_safety_guards_retained(self):
        for path in (pipeline.ROOT, *pipeline.protected_source_paths()):
            with self.assertRaises(ValueError):
                pipeline.safe_path(path)
        with self.assertRaises(ValueError):
            pipeline.parameter_file(pipeline.ROOT / "Parameters", "../outside", read_only=True)
        self.assertEqual(
            pipeline.parameter_file(pipeline.ROOT / "Parameters", "cfmeshCommon", read_only=True),
            pipeline.ROOT / "Parameters/cfmeshCommon.cpp",
        )


@unittest.skipUnless(os.environ.get("CFMESH_DOCKER_TESTS") == "1", "requires Docker and pipeline environment")
class PipelineIntegrationTests(unittest.TestCase):
    def test_preflight_study_and_streamed_logs_without_meshing(self):
        with tempfile.TemporaryDirectory(prefix="cfmesh-preflight-") as tmp:
            case = Path(tmp)
            pipeline.preflight(SimpleNamespace(
                sim_dir=case, mesh_only=True, mode="AMI", study=True,
                study_file="cfmeshCommon", study_parameter="backgroundCellSize",
            ))
            for role, utility in (("rotor", "cartesianMesh"), ("stator", "improveMeshQuality")):
                region = case / "cfmesh" / role
                region.mkdir(parents=True)
                pipeline.native_mesh_run(case, role, [utility, "-help"], 2, None, True)
                self.assertIn(utility, (region / f"log.{utility}").read_text())
                self.assertIn(utility, (case / "log.cfmesh").read_text())


if __name__ == "__main__":
    unittest.main()
