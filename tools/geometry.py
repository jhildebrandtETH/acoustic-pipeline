"""Read and validate the input propeller without changing its geometry."""
from collections import defaultdict
import hashlib
import math
import struct

def read_stl(path, scale):
    data = path.read_bytes()
    count = struct.unpack_from("<I", data, 80)[0] if len(data) >= 84 else 0
    if len(data) == 84 + 50 * count:
        triangles = (record[3:12] for record in struct.iter_unpack("<12fH", data[84:]))
    else:
        ascii_coords = []
        for line in data.decode("ascii").splitlines():
            fields = line.split()
            if fields and fields[0].lower() == "vertex":
                ascii_coords.extend(map(float, fields[1:]))
        if not ascii_coords or len(ascii_coords) % 9:
            raise ValueError("Invalid or empty STL")
        triangles = (ascii_coords[i:i+9] for i in range(0, len(ascii_coords), 9))
    vertices, faces, indices = [], [], {}
    edges = defaultdict(lambda: [0, 0])
    volume = 0.0
    for coords in triangles:
        face = []
        for i in (0, 3, 6):
            point = tuple(scale * x for x in coords[i:i+3])
            if len(point) != 3 or not all(map(math.isfinite, point)):
                raise ValueError("Invalid STL coordinate")
            if point not in indices:
                indices[point] = len(vertices)
                vertices.append(point)
            face.append(indices[point])
        a, b, c = (vertices[i] for i in face)
        cross = tuple((b[(i+1)%3]-a[(i+1)%3])*(c[(i+2)%3]-a[(i+2)%3])
                      -(b[(i+2)%3]-a[(i+2)%3])*(c[(i+1)%3]-a[(i+1)%3])
                      for i in range(3))
        if len(set(face)) != 3 or sum(x*x for x in cross) == 0:
            raise ValueError("STL contains a degenerate triangle; no repair attempted")
        volume += sum(a[i] * (b[(i+1)%3]*c[(i+2)%3]-b[(i+2)%3]*c[(i+1)%3])
                      for i in range(3)) / 6
        for u, v in zip(face, face[1:]+face[:1]):
            edge = edges[min(u,v), max(u,v)]
            edge[0] += 1
            edge[1] += 1 if u < v else -1
        faces.append(face)
    if not faces or any(n != 2 or winding != 0 for n, winding in edges.values()):
        raise ValueError("STL is not closed with consistent winding; no repair attempted")
    # The selected input is one solid. Reject ambiguous multi-shell inputs.
    neighbours = defaultdict(list)
    for u, v in edges:
        neighbours[u].append(v)
        neighbours[v].append(u)
    seen = {0}
    stack = [0]
    while stack:
        for v in neighbours[stack.pop()]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    if len(seen) != len(vertices) or abs(volume) < 1e-18:
        raise ValueError("Expected one closed solid of nonzero volume")
    return vertices, faces, volume, hashlib.sha256(data).hexdigest()
