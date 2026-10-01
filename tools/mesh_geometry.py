"""Bounded, vectorized polygon geometry for mesh diagnostics."""
import numpy as np


def face_geometry(points, vertices, offsets, face_ids, batch_size=10000):
    """Area-weighted triangle-fan centroids and oriented area vectors.

    Uses the same polygon definition as the first-cell measurement, including
    nonplanar polygons and zero-area faces, without a NumPy call per face.
    """
    result = {}
    for start in range(0, len(face_ids), batch_size):
        ids = face_ids[start:start + batch_size]
        sizes = offsets[ids + 1] - offsets[ids]
        local = np.r_[0, np.cumsum(sizes)]
        groups = np.repeat(np.arange(len(ids)), sizes)
        indices = np.arange(local[-1]) + np.repeat(offsets[ids] - local[:-1], sizes)
        p = points[vertices[indices]]
        following = np.arange(len(p)) + 1
        following[local[1:] - 1] = local[:-1]
        q = p[following]
        reference = np.add.reduceat(p, local[:-1]) / sizes[:, None]
        a, b = p - reference[groups], q - reference[groups]
        triangles = np.column_stack((a[:, 1]*b[:, 2] - a[:, 2]*b[:, 1],
                                     a[:, 2]*b[:, 0] - a[:, 0]*b[:, 2],
                                     a[:, 0]*b[:, 1] - a[:, 1]*b[:, 0])) / 2
        weights = np.linalg.norm(triangles, axis=1)
        total = np.add.reduceat(weights, local[:-1])
        weighted = np.add.reduceat((p + q + reference[groups]) / 3 * weights[:, None], local[:-1])
        centres = reference.copy()
        np.divide(weighted, total[:, None], out=centres, where=total[:, None] > 0)
        normals = np.add.reduceat(triangles, local[:-1])
        result.update((int(i), (centre, normal)) for i, centre, normal in zip(ids, centres, normals))
    return result
