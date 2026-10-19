"""Validated closed triangular meshes for the general neural solver."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import Surface


@dataclass
class MeshBody:
    triangles: np.ndarray
    interior_points: np.ndarray
    name: str = "body"

    def __post_init__(self):
        self.triangles = np.asarray(self.triangles, dtype=np.float64)
        self.interior_points = np.atleast_2d(np.asarray(self.interior_points, dtype=np.float64))
        if (self.triangles.ndim != 3 or self.triangles.shape[1:] != (3, 3)
                or not len(self.triangles) or not np.isfinite(self.triangles).all()):
            raise ValueError("triangles must be finite [N,3,3]")
        if (self.interior_points.ndim != 2 or self.interior_points.shape[1] != 3
                or not len(self.interior_points) or not np.isfinite(self.interior_points).all()):
            raise ValueError("At least one finite interior CHIEF point is required")
        cross = np.cross(self.triangles[:, 1]-self.triangles[:, 0],
                         self.triangles[:, 2]-self.triangles[:, 0])
        if np.any(np.linalg.norm(cross, axis=1) < 1e-12):
            raise ValueError("Mesh contains degenerate triangles")
        volume = np.einsum("ij,ij->i", self.triangles.mean(1), cross).sum()/6
        if volume < 0:
            self.triangles = self.triangles[:, [0, 2, 1]]

    @property
    def surface(self):
        return Surface(self.triangles)

    def transformed(self, translation=(0, 0, 0), quaternion=(0, 0, 0, 1)):
        rotation = Rotation.from_quat(quaternion).as_matrix()
        offset = np.asarray(translation, float)
        return MeshBody(self.triangles @ rotation.T + offset,
                        self.interior_points @ rotation.T + offset, self.name)

    @classmethod
    def combine(cls, bodies, name="scene"):
        bodies = tuple(bodies)
        if not bodies:
            raise ValueError("At least one body is required")
        return cls(np.concatenate([b.triangles for b in bodies]),
                   np.concatenate([b.interior_points for b in bodies]), name)

    @classmethod
    def from_npz(cls, path):
        with np.load(Path(path), allow_pickle=False) as data:
            return cls(data["triangles"], data["interior_points"],
                       str(data["name"]) if "name" in data else Path(path).stem)

    @classmethod
    def from_surface_file(cls, path, interior_points, center=None,
                          rotation_deg=(0, 0, 0), scale=1.0):
        """Load a watertight VTK/STL/PLY surface; CHIEF points are in final coordinates."""
        import pyvista as pv
        mesh = pv.read(Path(path))
        if isinstance(mesh, pv.MultiBlock):
            mesh = mesh.combine()
        mesh = mesh.extract_surface(algorithm="dataset_surface").triangulate().clean()
        if mesh.n_open_edges:
            raise ValueError(f"Mesh must be watertight; open edges={mesh.n_open_edges}")
        points = np.asarray(mesh.points, dtype=np.float64)
        points = (points-.5*(points.min(0)+points.max(0)))*float(scale)
        points = Rotation.from_euler("xyz", rotation_deg, degrees=True).apply(points)
        if center is not None:
            points += np.asarray(center,float)-.5*(points.min(0)+points.max(0))
        cells = np.asarray(mesh.faces).reshape(-1,4)
        if np.any(cells[:,0] != 3):
            raise ValueError("Triangulation failed")
        return cls(points[cells[:,1:]], interior_points, Path(path).stem)

    def save(self, path):
        np.savez_compressed(path, triangles=self.triangles,
                            interior_points=self.interior_points, name=self.name)


def box_body(size, center, quaternion=(0, 0, 0, 1), name="reflector"):
    """Closed rectangular reflector; size and center use metres."""
    size = np.asarray(size, float)
    if size.shape != (3,) or np.any(size <= 0):
        raise ValueError("size must contain three positive lengths")
    corners = np.array([[x, y, z] for x in (-.5, .5)
                        for y in (-.5, .5) for z in (-.5, .5)]) * size
    faces = np.array([
        [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
        [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
        [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
    ])
    rotation = Rotation.from_quat(quaternion).as_matrix()
    center = np.asarray(center, float)
    return MeshBody(corners[faces] @ rotation.T + center, center[None], name)


def plate_body(plate, subdivisions=10, thickness_layers=4):
    surface = plate.mesh(subdivisions, thickness_layers)
    return MeshBody(surface.triangles, plate.interior_points(), "triangle_plate")
