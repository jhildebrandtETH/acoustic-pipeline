"""Targeted Docker command/path regressions; no daemon required."""

import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor
import subprocess
import importlib.util

import tools.cfmesh_pipeline as pipeline
import tools.cfmesh_runtime as runtime


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        pool = patch.object(runtime._POOL, "container", return_value="test-helper")
        self.container = pool.start()
        self.addCleanup(pool.stop)

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
            self.assertIn(f"type=bind,source={case},target={runtime.container_path(case)}",
                          self.container.call_args.args[1])
            self.assertIn(runtime.container_path(dictionary), argv)
            self.assertEqual(argv[-1], value)
            self.assertIn("OMP_NUM_THREADS=3", argv)
            self.assertEqual(argv[argv.index("--workdir") + 1], runtime.container_path(region))
            self.assertIn('set --;', argv[argv.index("-c") + 1])
            self.assertEqual(argv[:2], ["docker", "exec"])
            self.assertIn("test-helper", argv)

    def test_runtime_separation(self):
        native = runtime.command(["cartesianMesh", "-help"])
        self.assertEqual(self.container.call_args.args[0], runtime.IMAGE)
        foundation = runtime.command(["checkMesh", "-help"], image=runtime.FOUNDATION_IMAGE)
        self.assertEqual(self.container.call_args.args[0], runtime.FOUNDATION_IMAGE)
        self.assertIn("/usr/lib/openfoam/openfoam2512/etc/bashrc", native[native.index("-c") + 1])
        self.assertIn("/opt/openfoam13/etc/bashrc", foundation[foundation.index("-c") + 1])

    def test_invalid_core_count(self):
        with self.assertRaises(ValueError):
            runtime.command(["cartesianMesh"], cores=0)
        self.container.assert_not_called()

    def test_regions_and_thread_counts_share_mounts(self):
        with tempfile.TemporaryDirectory(prefix="cfmesh space ") as tmp:
            case = Path(tmp).resolve()
            (case / "Parameters").mkdir()
            rotor = case / "cfmesh/rotor"
            stator = case / "cfmesh/stator"
            rotor.mkdir(parents=True)
            stator.mkdir()
            first = runtime.command(["cartesianMesh"], cwd=rotor, cores=2)
            first_mounts = self.container.call_args
            second = runtime.command(["improveMeshQuality"], cwd=stator, cores=4)
            self.assertEqual(first_mounts, self.container.call_args)
            self.assertIn("OMP_NUM_THREADS=2", first)
            self.assertIn("OMP_NUM_THREADS=4", second)
            self.assertTrue(first[first.index("--workdir") + 1].endswith("/cfmesh/rotor"))

    def test_repository_case_with_spaces_keeps_its_alias_mount(self):
        with tempfile.TemporaryDirectory(prefix="cfmesh space ", dir=runtime.ROOT) as tmp:
            case = Path(tmp).resolve()
            (case / "Parameters").mkdir()
            argv = runtime.command(["foamDictionary", case / "Parameters/values"], cwd=case)
            workdir = argv[argv.index("--workdir") + 1]
            self.assertNotIn(" ", workdir)
            self.assertTrue(workdir.startswith("/cfmesh/mount"))
            self.assertEqual(argv[-1], workdir + "/Parameters/values")
            self.assertIn(f"type=bind,source={case},target={workdir}",
                          self.container.call_args.args[1])


class PoolTests(unittest.TestCase):
    def test_concurrent_utilities_create_one_named_helper_and_cleanup_once(self):
        pool = runtime.RuntimePool()
        with patch.object(runtime.subprocess, "run", return_value=SimpleNamespace(returncode=0, stderr="")) as run:
            with ThreadPoolExecutor(max_workers=8) as executor:
                names = list(executor.map(lambda _: pool.container(runtime.IMAGE, []), range(32)))
            self.assertEqual(len(set(names)), 1)
            self.assertTrue(names[0].startswith("acoustic-pipeline-native-"))
            run.assert_called_once()
            launch = run.call_args.args[0]
            self.assertEqual(launch[:3], ["docker", "run", "--detach"])
            self.assertIn("--name", launch)
            if hasattr(os, "getuid"):
                self.assertEqual(launch[launch.index("--user") + 1], f"{os.getuid()}:{os.getgid()}")
            pool.close()
            pool.close()
            self.assertEqual(run.call_count, 2)
            self.assertEqual(run.call_args.args[0], ["docker", "rm", "--force", names[0]])

    def test_mounts_images_and_sessions_are_isolated(self):
        pool = runtime.RuntimePool()
        other = runtime.RuntimePool()
        with patch.object(runtime.subprocess, "run", return_value=SimpleNamespace(returncode=0, stderr="")):
            names = {pool.container(runtime.IMAGE, []),
                     pool.container(runtime.FOUNDATION_IMAGE, []),
                     pool.container(runtime.IMAGE, ["--mount", "case"]),
                     other.container(runtime.IMAGE, [])}
            self.assertEqual(len(names), 4)
            pool.close()
            other.close()

    def test_failed_start_is_not_reused_or_allowed_to_remove_other_helpers(self):
        pool = runtime.RuntimePool()
        ok = SimpleNamespace(returncode=0, stderr="")
        with patch.object(runtime.subprocess, "run", return_value=ok) as run:
            existing = pool.container(runtime.IMAGE, [])
            run.side_effect = [subprocess.CalledProcessError(1, "docker"), ok]
            with self.assertRaises(subprocess.CalledProcessError):
                pool.container(runtime.FOUNDATION_IMAGE, [])
            self.assertNotEqual(run.call_args.args[0][-1], existing)
            run.side_effect = None
            self.assertEqual(pool.container(runtime.IMAGE, []), existing)
            pool.container(runtime.FOUNDATION_IMAGE, [])
            self.assertEqual(run.call_args.args[0][:3], ["docker", "run", "--detach"])
            pool.close()

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


class PhaseTests(unittest.TestCase):
    def test_failed_cleanup_blocks_phase_transition(self):
        with patch.object(runtime.subprocess, "run", side_effect=[
            SimpleNamespace(returncode=0, stderr=""),
            SimpleNamespace(returncode=1, stderr="daemon unavailable"),
        ]):
            with self.assertWarnsRegex(UserWarning, "daemon unavailable"):
                with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
                    with runtime.phase(runtime.ROOT, "mesh"):
                        runtime.command(["cartesianMesh", "-help"])
            self.assertIsNone(runtime._PHASE.get())

    def test_nested_region_parameters_use_one_helper_and_cleanup_on_error(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            runtime.subprocess, "run", return_value=SimpleNamespace(returncode=0, stderr="")
        ) as run:
            case = Path(tmp)
            for relative in ("Parameters", "cfmesh/rotor/Parameters", "cfmesh/stator/Parameters"):
                (case / relative).mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "phase failed"):
                with runtime.phase(case, "mesh") as pool:
                    for relative in ("Parameters", "cfmesh/rotor", "cfmesh/stator"):
                        runtime.command(["foamDictionary", "-help"], cwd=case / relative)
                    self.assertEqual(len(pool._containers), 1)
                    self.assertEqual(run.call_count, 1)
                    raise ValueError("phase failed")
            self.assertEqual(run.call_count, 2)
            self.assertEqual(run.call_args.args[0][:3], ["docker", "rm", "--force"])
            self.assertIsNone(runtime._PHASE.get())

    def test_parallel_phases_do_not_share_or_close_another_cases_helper(self):
        import threading
        barrier = threading.Barrier(2)
        def work(label):
            with runtime.phase(runtime.ROOT, label) as pool:
                runtime.command(["foamDictionary", "-help"])
                name = next(iter(pool._containers.values()))
                barrier.wait(timeout=5)
                self.assertIs(runtime._PHASE.get()[1], pool)
            return name
        with patch.object(runtime.subprocess, "run", return_value=SimpleNamespace(returncode=0, stderr="")) as run:
            with ThreadPoolExecutor(max_workers=2) as executor:
                names = list(executor.map(work, ("case1", "case2")))
            self.assertEqual(len(set(names)), 2)
            removed = [call.args[0][-1] for call in run.call_args_list
                       if call.args[0][:2] == ["docker", "rm"]]
            self.assertCountEqual(removed, names)

    def test_batch_reads_preserve_values_and_report_missing_entries(self):
        with patch.object(pipeline, "native_run", return_value=SimpleNamespace(
            returncode=0, stdout="(1 2 3)\n\0{ custom 4; }\n\0", stderr=""
        )) as run:
            result = pipeline.query_entries(runtime.ROOT / "Parameters/cfmeshCommon.cpp",
                                            ["vector", "nested"])
            self.assertEqual(result, {"vector": "(1 2 3)", "nested": "{ custom 4; }"})
            run.assert_called_once()
            run.return_value = SimpleNamespace(returncode=1, stderr="missing nested")
            with self.assertRaisesRegex(ValueError, "missing nested"):
                pipeline.query_entries(runtime.ROOT / "Parameters/cfmeshCommon.cpp", ["nested"])


@unittest.skipUnless(os.environ.get("CFMESH_DOCKER_TESTS") == "1", "requires Docker and pipeline environment")
class PipelineIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("docker"), "requires Python Docker SDK")
    def test_preflight_study_and_streamed_logs_without_meshing(self):
        with tempfile.TemporaryDirectory(prefix="cfmesh-preflight-") as tmp:
            case = Path(tmp)
            pipeline.preflight(SimpleNamespace(
                sim_dir=case, mesh_only=True, mode="AMI", study=True,
                study_file="cfmeshCommon", study_parameter="baseCellSize",
            ))
            with runtime.phase(case, "mesh"):
                for role, utility in (("rotor", "cartesianMesh"), ("stator", "improveMeshQuality")):
                    region = case / "cfmesh" / role
                    region.mkdir(parents=True)
                    pipeline.native_mesh_run(case, role, [utility, "-help"], 2, None, False)
                    self.assertIn(utility, (region / f"log.{utility}").read_text())
                    self.assertIn(utility, (case / "log.cfmesh").read_text())

    def test_real_phase_mounts_environment_batch_reads_and_cleanup(self):
        with tempfile.TemporaryDirectory(prefix="cfmesh phase ") as tmp:
            case = Path(tmp)
            for role in ("rotor", "stator"):
                (case / "cfmesh" / role / "Parameters").mkdir(parents=True)
            (case / "Parameters").mkdir()
            dictionary = case / "Parameters/values"
            dictionary.write_text("a 17; b (1 2 3);\n")
            with patch.object(runtime.subprocess, "run", wraps=subprocess.run) as calls:
                with runtime.phase(case, "mesh") as pool:
                    self.assertEqual(pipeline.query_entries(dictionary, ["a", "b"]),
                                     {"a": "17", "b": "( 1 2 3 )"})
                    for role, cores in (("rotor", 2), ("stator", 3)):
                        output = runtime.run(["bash", "-c", 'printf "%s\\n" "$OMP_NUM_THREADS"; pwd'],
                                             cwd=case / "cfmesh" / role, cores=cores,
                                             text=True, capture_output=True, check=True)
                        self.assertEqual(output.stdout.splitlines()[0], str(cores))
                        self.assertTrue(output.stdout.strip().endswith("/cfmesh/" + role))
                        utility = "cartesianMesh" if role == "rotor" else "improveMeshQuality"
                        pipeline.native_mesh_run(case, role, [utility, "-help"], cores, None, False)
                        self.assertIn(utility, (case / "cfmesh" / role / f"log.{utility}").read_text())
                    self.assertEqual(len(pool._containers), 1)
                    name = next(iter(pool._containers.values()))
                launches = [call for call in calls.call_args_list if call.args[0][:2] == ["docker", "run"]]
                self.assertEqual(len(launches), 1)
            result = subprocess.run(["docker", "container", "inspect", name], capture_output=True)
            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
