"""Docker runtime for native OpenCFD utilities; no host OpenFOAM needed."""

import argparse
import atexit
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
import os
import re
from pathlib import Path
import subprocess
import threading
import uuid
import warnings

ROOT = Path(__file__).resolve().parent.parent
IMAGE = "opencfd/openfoam-default:2512"
FOUNDATION_IMAGE = "microfluidica/openfoam:13"


class RuntimePool:
    """Reuse helpers within this Python process, including scheduler threads."""

    def __init__(self, label="utility"):
        self._containers = {}
        self._lock = threading.Lock()
        self._session = uuid.uuid4().hex[:12]
        self._label = re.sub(r"[^a-zA-Z0-9_.-]", "-", label)[:70]

    def container(self, image, mounts):
        key = (image, tuple(mounts))
        with self._lock:
            if key not in self._containers:
                flavor = "native" if image == IMAGE else "foundation"
                name = f"acoustic-pipeline-{flavor}-{self._label}-{self._session}-{len(self._containers) + 1}"
                launch = ["docker", "run", "--detach", "--rm", "--name", name,
                          "--label", "acoustic-pipeline-role=utility"]
                if hasattr(os, "getuid"):
                    launch += ["--user", f"{os.getuid()}:{os.getgid()}"]
                launch += [*mounts, "--entrypoint", "/bin/bash", image,
                           "-c", "exec sleep infinity"]
                # Register before launch so interruption/ambiguous CLI failures
                # still leave an owned name for exit cleanup. Never adopt others.
                self._containers[key] = name
                try:
                    subprocess.run(launch, check=True, stdout=subprocess.PIPE, text=True)
                except BaseException:
                    self._remove(key)
                    raise
            return self._containers[key]

    def close(self, strict=False):
        """Remove only helpers created by this process; safe to call repeatedly."""
        # Callers must finish their utility commands before closing the pool.
        with self._lock:
            failed = []
            for key in list(self._containers):
                name = self._containers[key]
                if not self._remove(key):
                    failed.append(name)
            if strict and failed:
                raise RuntimeError("Could not finish container phase; cleanup failed for " + ", ".join(failed))

    def _remove(self, key):
        name = self._containers.pop(key)
        try:
            result = subprocess.run(["docker", "rm", "--force", name],
                                    capture_output=True, text=True, timeout=15)
            if result.returncode and "No such container" not in result.stderr:
                warnings.warn(f"Could not remove utility container {name}: {result.stderr.strip()}")
                return False
        except (OSError, subprocess.SubprocessError) as error:
            warnings.warn(f"Could not remove utility container {name}: {error}")
            return False
        return True


_POOL = RuntimePool()
atexit.register(_POOL.close)
_PHASE = ContextVar("cfmesh_runtime_phase", default=None)


@contextmanager
def phase(case, name):
    """Own one native helper for this phase, isolated from other case threads."""
    case = Path(case).resolve()
    pool = RuntimePool(f"{case.name[:50]}-{name}")
    token = _PHASE.set((case, pool))
    try:
        yield pool
    finally:
        try:
            pool.close(strict=True)
        finally:
            _PHASE.reset(token)


def container_path(path):
    """Keep POSIX/WSL paths intact; give Windows drives a Linux mount target."""
    path = Path(path).expanduser().resolve()
    if path.drive:
        if len(path.drive) != 2 or path.drive[1] != ":":
            raise ValueError("Use a local drive or WSL path for Docker bind mounts")
        return "/host/" + path.drive[0].lower() + path.as_posix()[2:]
    return path.as_posix()


def command(argv, cwd=ROOT, cores=1, mounts=(), image=IMAGE):
    """Build an argv-safe exec command in a reusable, named helper container.

    Path arguments are translated to their bind-mount targets. String arguments
    remain literal (including dictionary values). Mounting the case ancestor
    retains relative dictionary includes, Parameters and generated polyMesh.
    """
    cwd = Path(cwd).expanduser().resolve(strict=True)
    if int(cores) < 1:
        raise ValueError("cores must be at least 1")
    active = _PHASE.get()
    # Region folders also contain Parameters. Use the phase's complete case
    # mount so rotor/stator queries cannot each allocate another container.
    case_root = active[0] if active else next(
        (p for p in (cwd, *cwd.parents) if (p / "Parameters").is_dir()), cwd)
    roots = [ROOT, case_root, *(Path(p).resolve(strict=True) for p in mounts)]
    roots = list(dict.fromkeys(roots))
    # A repository mount covers ordinary child cases. Keep separate mounts for
    # paths with whitespace: OpenFOAM requires their whitespace-free aliases.
    roots = sorted((root for root in roots if not any(
        other in root.parents and not any(c.isspace() for c in str(root.relative_to(other)))
        for other in roots
    )), key=str)
    targets = {
        root: (f"/cfmesh/mount{index}" if any(c.isspace() for c in str(root))
               else container_path(root))
        for index, root in enumerate(roots)
    }

    def mounted_path(path):
        path = Path(path).expanduser().resolve()
        for root in sorted(roots, key=lambda p: len(p.parts), reverse=True):
            if path == root or root in path.parents:
                relative = path.relative_to(root).as_posix()
                return targets[root] if relative == "." else targets[root] + "/" + relative
        raise ValueError(f"Path is outside Docker mounts: {path}")

    mount_options = []
    for root in roots:
        if not root.is_dir():
            raise ValueError(f"Docker mount must be an existing directory: {root}")
        # Docker --mount uses comma-delimited fields; reject ambiguous paths.
        if "," in str(root):
            raise ValueError(f"Docker bind-mount paths cannot contain commas: {root}")
        mount_options += ["--mount", f"type=bind,source={root},target={targets[root]}"]
    # Translate/validate arguments before starting any container.
    arguments = [mounted_path(arg) if isinstance(arg, Path) else str(arg) for arg in argv]
    name = (active[1] if active else _POOL).container(image, mount_options)
    result = ["docker", "exec", "--workdir", mounted_path(cwd),
              "--env", f"OMP_NUM_THREADS={int(cores)}", name, "/bin/bash", "-c"]
    bashrc = ("/usr/lib/openfoam/openfoam2512/etc/bashrc" if image == IMAGE
              else "/opt/openfoam13/etc/bashrc")
    # Source only inside the container, without passing utility options to bashrc.
    # Cache the exported OpenFOAM environment once per helper. Preserve each
    # exec's cwd and allocated OpenMP threads rather than caching those values.
    result += ['cfmesh_args=("$@"); cfmesh_threads=$OMP_NUM_THREADS; set --; '
               'cfmesh_env=/tmp/acoustic-pipeline-environment; '
               'if [[ -r "$cfmesh_env" ]]; then source "$cfmesh_env"; else '
               f'source {bashrc} >/dev/null || exit $?; '
               'export -p | sed -E \'/^declare -x (PWD|OLDPWD|SHLVL|OMP_NUM_THREADS)=/d\' '
               '> "$cfmesh_env.$$" && mv "$cfmesh_env.$$" "$cfmesh_env" || exit $?; fi; '
               'export OMP_NUM_THREADS=$cfmesh_threads; '
               'exec stdbuf -oL -eL "${cfmesh_args[@]}"', "cfmesh"]
    result += arguments
    return result


def run(argv, cwd=ROOT, cores=1, mounts=(), **kwargs):
    return subprocess.run(command(argv, cwd, cores, mounts), **kwargs)


@lru_cache(maxsize=1)
def preflight():
    """Check the daemon, installed Foundation image and usable native utilities."""
    subprocess.run(["docker", "info"], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["docker", "image", "inspect", FOUNDATION_IMAGE],
                   check=True, stdout=subprocess.DEVNULL)
    # docker run pulls the native image automatically if absent.
    for utility in ("cartesianMesh", "improveMeshQuality", "foamDictionary"):
        run([utility, "-help"], check=True, stdout=subprocess.DEVNULL)


def smoke_test():
    """Exercise includes, dictionary edits and host-visible writes without meshing."""
    import tempfile

    preflight()
    # Test repository-local and external cases, with whitespace on the host.
    for parent in (ROOT, None):
        with tempfile.TemporaryDirectory(prefix="cfmesh runtime ", dir=parent) as tmp:
            case = Path(tmp)
            parameters = case / "Parameters"
            system = case / "cfmesh/rotor/system"
            parameters.mkdir()
            system.mkdir(parents=True)
            (parameters / "values").write_text("testValue 17;\n")
            dictionary = system / "meshDict"
            dictionary.write_text('#include "../../../Parameters/values"\n')
            output = run(["foamDictionary", dictionary, "-entry", "testValue", "-value"],
                         cwd=system.parent, check=True, text=True, capture_output=True)
            assert output.stdout.strip() == "17", output.stdout
            run(["foamDictionary", parameters / "values", "-entry", "testValue",
                 "-set", "23"], cwd=case, check=True, stdout=subprocess.DEVNULL)
            output = run(["foamDictionary", dictionary, "-entry", "testValue", "-value"],
                         cwd=system.parent, check=True, text=True, capture_output=True)
            assert output.stdout.strip() == "23", output.stdout
            assert "23" in (parameters / "values").read_text()
            run(["foamDictionary", dictionary, "-entry", "workflowControls", "-set",
                 "{ stopAfter edgeExtraction; }"], cwd=case, check=True,
                stdout=subprocess.DEVNULL)
            output = run(["foamDictionary", dictionary, "-entry", "workflowControls/stopAfter",
                          "-value"], cwd=case, check=True, text=True, capture_output=True)
            assert output.stdout.strip() == "edgeExtraction", output.stdout
            run(["foamDictionary", dictionary, "-entry", "workflowControls", "-remove"],
                cwd=case, check=True, stdout=subprocess.DEVNULL)
            run(["bash", "-c", "mkdir -p constant/polyMesh; printf mounted > constant/polyMesh/probe"],
                cwd=system.parent, check=True)
            assert (system.parent / "constant/polyMesh/probe").read_text() == "mounted"
    print("PASS: native utility help, dictionary includes/edits and mounted case writes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--foundation", action="store_true")
    parser.add_argument("argv", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.smoke_test:
        try:
            smoke_test()
        except subprocess.CalledProcessError as error:
            print(error.stdout or "")
            print(error.stderr or "")
            raise
    else:
        argv = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
        if not argv:
            parser.error("Supply a utility after --, or use --smoke-test")
        raise SystemExit(subprocess.call(command(
            argv, args.cwd, os.environ.get("OMP_NUM_THREADS", "1"),
            image=FOUNDATION_IMAGE if args.foundation else IMAGE)))
