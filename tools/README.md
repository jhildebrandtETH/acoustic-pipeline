# Pipeline helpers

Top-level stage files show the workflow. Put supporting functions here.

| Module | Contents |
| --- | --- |
| `orders.py` | New-order validation and restoration of saved options |
| `cli.py` | Command-line options and terminal recovery |
| `scheduler.py` | Case execution, concurrency, dashboard and resume states |
| `common.py` | Status, atomic writes, shared locks and process helpers |
| `openfoam.py` | Orders, dictionary edits, convergence and reconstruction |
| `preprocessing.py` | Safe case copying and isolated preprocessing process |
| `cfmesh_pipeline.py` | Native meshing, assembly and quality checks |
| `cfmesh_runtime.py` | Docker commands and runtime smoke check |
| `cfmesh_orders.py` | Order locking and example input selection |
| `cfmesh_refinement.py` | Refinement shapes |
| `cfmesh_interface.py` | Cylindrical interface projection in an isolated process |
| `cfmesh_layers.py` | Existing standalone boundary-layer adapter |
| `geometry.py` | STL reader and validation |
| `reporting.py` | Result summaries and report plots |
| `visualization.py` | ParaView configuration, rendering and atlas helpers |
| `terminal_propeller.py` | Dashboard animation |

`__init__.py` keeps familiar `from tools import ...` imports working and loads
helpers when needed. New helpers can be imported directly from their module.
Shared locks stay in `common.py`. Cross-module imports inside functions avoid
cycles between scheduler and stages. ParaView helpers remain together because
their source is collected into a standalone renderer script.
