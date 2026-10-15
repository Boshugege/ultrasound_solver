"""Remesh closed surfaces for the BEM (placement uses physics.mesh_to_sdf._place_mesh)."""

import numpy as np


def remesh(surface, element_size):
    """Near-uniform triangulation with edge length ~ ``element_size`` (pyacvd clustering)."""
    import pyacvd

    target_points = int(surface.area / (0.866 * element_size**2))
    clustering = pyacvd.Clustering(surface)
    while clustering.mesh.n_points < 3 * target_points:
        clustering.subdivide(2)
    # Clustering occasionally leaves a non-manifold spot; a slightly different
    # cluster count usually avoids it.
    for factor in (1.0, 0.99, 1.01, 0.97, 1.03, 0.95, 1.05):
        clustering.cluster(int(target_points * factor))
        result = clustering.create_mesh().triangulate().clean()
        if not result.n_open_edges:
            return result
    raise ValueError("remeshed surface is not closed")


def oriented_arrays(surface):
    """Vertices and faces with every face normal pointing out of the closed body."""
    oriented = surface.compute_normals(cell_normals=True, point_normals=False,
                                       auto_orient_normals=True, consistent_normals=True)
    vertices = np.asarray(oriented.points, float)
    faces = np.asarray(oriented.faces).reshape(-1, 4)[:, 1:].astype(np.int64)
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    flip = (cross * np.asarray(oriented.cell_data["Normals"])).sum(1) < 0
    faces[flip] = faces[flip][:, ::-1]
    return vertices, faces
