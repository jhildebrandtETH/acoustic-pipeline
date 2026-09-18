"""Generate the rotor/stator mesh and connect it for rotating flow."""
from tools.cfmesh_pipeline import run_mesh


def cfmesh(container, case, cores, allow_bad=False,
           callback=None, live=False):
    """Mesh both regions, assemble the rotating zone, and check NCC and quality.

    Edit Parameters/cfmesh*. The implementation and geometry helpers live in
    tools/cfmesh_pipeline.py; the native meshing commands are unchanged.
    Pass a container factory to start Foundation only after native meshing exits.
    """
    return run_mesh(container, case, cores, allow_bad, callback, live)
