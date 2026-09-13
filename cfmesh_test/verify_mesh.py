#!/usr/bin/env python3
"""Reject a failed checkMesh summary or the wrong retained fluid region."""
import json
from pathlib import Path
import re
import sys
case = Path(sys.argv[1]).resolve()
text = (case / "log.checkMesh").read_text(errors="replace")
if "Mesh OK." not in text or re.search(r"Failed\s+\d+\s+mesh checks|FOAM FATAL", text):
    raise SystemExit("Mesh-quality checks did not pass. Inspect log.checkMesh; mesh remains viewable.")
regions = re.search(r"Number of regions:\s*(\d+)", text)
if not regions or int(regions[1]) != 1:
    raise SystemExit("Expected exactly one connected fluid region.")
boundary = (case / "constant/polyMesh/boundary").read_text()
boundary = re.sub(r"/\*.*?\*/|//[^\n]*", "", boundary, flags=re.S)
patches = {}
for name, body in re.findall(r"(\w+)\s*\{([^{}]*)\}", boundary):
    count = re.search(r"nFaces\s+(\d+)\s*;", body)
    kind = re.search(r"type\s+(\w+)\s*;", body)
    if count and int(count[1]) > 0:
        patches[name] = (int(count[1]), kind[1] if kind else None)
if set(patches) != {"propellerSurface", "farfield"}:
    raise SystemExit(f"Unexpected nonempty patches: {patches}")
if patches["propellerSurface"][1] != "wall" or patches["farfield"][1] != "patch":
    raise SystemExit(f"Unexpected patch types: {patches}")
volume = re.search(r"Total volume\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)", text)
expected = json.loads((case / "geometry.json").read_text())["expected_fluid_volume_m3"]
if not volume or abs(float(volume[1])-expected)/expected > 0.02:
    raise SystemExit("Mesh volume does not match the surrounding fluid domain within 2%.")
print("PASS: checkMesh, one fluid region, expected volume, and both named patches.")
print(patches)
