"""Put both sides of the rotating interface on the same circular cylinder."""

import json
import math
from pathlib import Path
import re
import subprocess
import sys


from cfmesh_pipeline import safe_path, write


def project_rotating_interface(case):
    """Keep native NumPy geometry operations out of scheduler worker threads."""
    case = safe_path(case)
    log_path = case / "cfmesh/log.interfaceProjection"
    with log_path.open("w") as log:
        result = subprocess.run(
            [
                sys.executable,
                "-u",
                "-X",
                "faulthandler",
                str(Path(__file__).resolve()),
                str(case),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise RuntimeError(
            f"Interface projection failed ({result.returncode}); see {log_path}"
        )


def _project_rotating_interface(case):
    """Remove the polygonal surface approximation before constructing NCC."""
    import numpy as np

    case = safe_path(case)
    geometry = json.loads((case / "cfmesh/geometry.json").read_text())
    radius = geometry["rotor_radius_m"]
    half_length = geometry["rotor_half_length_m"]
    segments = geometry["cylinder_segments"]
    mesh_directory = case / "constant/polyMesh"
    points_path = mesh_directory / "points"
    points_text = points_path.read_text()
    point_list = re.search(r"\n(\d+)\s*\n\(", points_text)
    if point_list is None or "format      binary" in points_text:
        raise ValueError("Interface projection requires an ASCII cfMesh points file")
    points = np.fromstring(
        points_text[point_list.end() : points_text.rindex(")")]
        .replace("(", " ")
        .replace(")", " "),
        sep=" ",
    ).reshape(-1, 3)
    if len(points) != int(point_list[1]):
        raise ValueError("Unexpected point count in cfMesh mesh")

    boundary_text = (mesh_directory / "boundary").read_text()
    face_ranges = []
    for patch_name in ("rotaryRegion", "rotaryRegion_slave"):
        patch = re.search(r"\b" + patch_name + r"\s*\{([^{}]*)\}", boundary_text)
        if patch is None:
            raise ValueError(f"Missing rotating interface patch: {patch_name}")
        first_face = int(re.search(r"startFace\s+(\d+)", patch[1])[1])
        face_count = int(re.search(r"nFaces\s+(\d+)", patch[1])[1])
        face_ranges.append((first_face, first_face + face_count))

    side_points, cap_points = set(), set()
    with (mesh_directory / "faces").open() as faces_file:
        for line in faces_file:
            if re.fullmatch(r"\d+\s*", line):
                break
        next(faces_file)  # Opening parenthesis of the face list.
        for face_index, line in enumerate(faces_file):
            if not any(first <= face_index < last for first, last in face_ranges):
                continue
            face = list(map(int, re.search(r"\(([^()]*)\)", line)[1].split()))
            vertices = points[face]
            normal = np.cross(vertices, np.roll(vertices, -1, axis=0)).sum(axis=0)
            is_cap = abs(normal[1]) > 0.5 * np.linalg.norm(normal)
            (cap_points if is_cap else side_points).update(face)

    interface_points = np.array(sorted(side_points | cap_points), dtype=int)
    side = np.array(sorted(side_points), dtype=int)
    caps = np.array(sorted(cap_points), dtype=int)
    if not len(side) or not len(caps):
        raise ValueError("The rotating cylinder needs both side and cap faces")
    original = points[interface_points].copy()
    angles = np.arctan2(points[side, 2], points[side, 0])
    points[side, 0] = radius * np.cos(angles)
    points[side, 2] = radius * np.sin(angles)
    points[caps, 1] = np.where(points[caps, 1] > 0, half_length, -half_length)
    maximum_movement = float(
        np.linalg.norm(points[interface_points] - original, axis=1).max()
    )
    # Only correct the chord-to-circle error, not an incorrectly meshed cylinder.
    movement_limit = 1.5 * radius * (1 - math.cos(math.pi / segments)) + 1e-10
    if maximum_movement > movement_limit:
        raise ValueError(
            f"Interface needs {maximum_movement:g} m correction, exceeding the "
            f"polygon approximation limit {movement_limit:g} m"
        )

    temporary_path = points_path.with_suffix(".projected")
    with temporary_path.open("w") as points_file:
        points_file.write(points_text[: point_list.end()] + "\n")
        np.savetxt(points_file, points, fmt="(%.17g %.17g %.17g)")
        points_file.write(points_text[points_text.rindex(")") :])
    temporary_path.replace(points_path)
    write(
        case / "cfmesh/interface-projection.json",
        json.dumps(
            {
                "projected_points": len(interface_points),
                "maximum_movement_m": maximum_movement,
                "movement_limit_m": movement_limit,
                "radius_m": radius,
                "half_length_m": half_length,
            },
            indent=2,
        )
        + "\n",
    )


if __name__ == "__main__":
    _project_rotating_interface(Path(sys.argv[1]))
