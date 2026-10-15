"""Sample a closed surface mesh as an SDF on the solver grid."""

from pathlib import Path
import warnings

import numpy as np
import pyvista as pv


def _as_vector3(value, name: str, dtype=np.float64) -> np.ndarray:
    vector = np.asarray(value, dtype=dtype)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain three finite values")
    return vector


def _load_closed_surface(mesh_path: Path) -> pv.PolyData:
    dataset = pv.read(mesh_path)
    if isinstance(dataset, pv.MultiBlock):
        dataset = dataset.combine()
    surface = dataset.extract_surface(
        algorithm="dataset_surface"
    ).triangulate().clean()
    if surface.n_points < 4 or surface.n_cells < 4:
        raise ValueError("Mesh does not contain a usable closed surface")
    if not np.isfinite(surface.points).all():
        raise ValueError("Mesh contains NaN or infinite vertex coordinates")

    non_manifold = surface.extract_feature_edges(
        boundary_edges=False,
        non_manifold_edges=True,
        feature_edges=False,
        manifold_edges=False,
    ).n_cells
    if surface.n_open_edges or non_manifold:
        raise ValueError(
            "Mesh must be watertight and manifold: "
            f"open edges={surface.n_open_edges}, "
            f"non-manifold edges={non_manifold}"
        )
    return surface


def _rotation_matrix_xyz(rotation_deg: np.ndarray) -> np.ndarray:
    rx, ry, rz = np.deg2rad(rotation_deg)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rotation_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]]
    )
    rotation_y = np.array(
        [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]]
    )
    rotation_z = np.array(
        [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]]
    )
    return rotation_z @ rotation_y @ rotation_x


def _place_mesh(
    mesh: pv.PolyData,
    center_m: np.ndarray,
    rotation_deg: np.ndarray,
    scale: float,
) -> pv.PolyData:
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be a positive finite number")

    points = np.asarray(mesh.points, dtype=np.float64)
    source_center = 0.5 * (points.min(axis=0) + points.max(axis=0))
    points = (points - source_center) * scale
    points = points @ _rotation_matrix_xyz(rotation_deg).T
    rotated_center = 0.5 * (points.min(axis=0) + points.max(axis=0))

    placed = mesh.copy(deep=True)
    placed.points = points + center_m - rotated_center
    return placed


def mesh_to_sdf(
    mesh_path,
    grid_size,
    box_size,
    center_m,
    rotation_deg=(0.0, 0.0, 0.0),
    scale=1.0,
    origin=(0.0, 0.0, 0.0),
) -> np.ndarray:
    """Return an ``(nx, ny, nz)`` float32 SDF sampled inside the domain.

    The mesh is uniformly scaled, rotated around X then Y then Z, and placed
    so its final axis-aligned bounding-box center equals ``center_m``. Mesh
    geometry outside the solver domain is not altered; it is simply not
    sampled. Distances are in metres and are negative inside the mesh.
    """
    path = Path(mesh_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Mesh not found: {path}")

    grid_size = _as_vector3(grid_size, "grid_size", dtype=np.float64)
    if not np.equal(grid_size, np.floor(grid_size)).all():
        raise ValueError("grid_size must contain integers")
    grid_size = grid_size.astype(np.int32)
    box_size = _as_vector3(box_size, "box_size")
    center_m = _as_vector3(center_m, "center_m")
    rotation_deg = _as_vector3(rotation_deg, "rotation_deg")
    origin = _as_vector3(origin, "origin")
    if np.any(grid_size < 2) or np.any(box_size <= 0.0):
        raise ValueError("Domain must have positive size and >=2 grid points")

    mesh = _place_mesh(
        _load_closed_surface(path), center_m, rotation_deg, float(scale)
    )
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    mesh_min = bounds[[0, 2, 4]]
    mesh_max = bounds[[1, 3, 5]]
    domain_max = origin + box_size
    if np.any(mesh_max < origin) or np.any(mesh_min > domain_max):
        warnings.warn(
            "Transformed mesh lies completely outside the solver domain; "
            "the returned SDF contains no obstacle interior samples.",
            RuntimeWarning,
            stacklevel=2,
        )

    oriented = mesh.compute_normals(
        cell_normals=True,
        point_normals=True,
        consistent_normals=True,
        auto_orient_normals=True,
        non_manifold_traversal=False,
        inplace=False,
    )
    grid = pv.ImageData(
        dimensions=tuple(int(value) for value in grid_size),
        spacing=tuple(float(value) for value in box_size / (grid_size - 1)),
        origin=tuple(float(value) for value in origin),
    )
    sampled = grid.compute_implicit_distance(oriented, inplace=False)
    distance = np.abs(np.asarray(sampled.point_data["implicit_distance"]))
    # The sign of vtkImplicitPolyDataDistance comes from the normal of the closest
    # facet and is wrong wherever the closest point lies on an edge or vertex
    # (24 % of the obstacle study's grid was marked as inside the hand). Take the
    # inside/outside decision from ray casting against the closed surface instead.
    enclosed = grid.select_enclosed_points(oriented, tolerance=1e-9, check_surface=True)
    inside = np.asarray(enclosed.point_data["SelectedPoints"]).astype(bool)
    sdf = np.where(inside, -distance, distance).reshape(tuple(grid_size), order="F")
    sdf = np.asarray(sdf, dtype=np.float32)
    if not np.isfinite(sdf).all():
        raise RuntimeError("Generated SDF contains NaN or infinite values")
    if not np.any(sdf < 0.0):
        warnings.warn(
            "The solver grid contains no samples inside the transformed mesh.",
            RuntimeWarning,
            stacklevel=2,
        )
    elif np.all(sdf <= 0.0):
        warnings.warn(
            "Every solver-grid sample lies inside the transformed mesh.",
            RuntimeWarning,
            stacklevel=2,
        )
    return sdf
