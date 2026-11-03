"""Finite rigid reflectors represented as closed BEM obstacle meshes."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def _unit(vector: Iterable[float], name: str) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite 3-vector")
    norm = np.linalg.norm(value)
    if norm <= 0.0:
        raise ValueError(f"{name} must be non-zero")
    return value / norm


def _frame(normal: Iterable[float], tangent: Iterable[float] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = _unit(normal, "normal")
    if tangent is None:
        reference = np.asarray([1.0, 0.0, 0.0])
        if abs(np.dot(reference, n)) > 0.9:
            reference = np.asarray([0.0, 1.0, 0.0])
        u = reference - n * np.dot(reference, n)
    else:
        t = _unit(tangent, "tangent")
        u = t - n * np.dot(t, n)
        if np.linalg.norm(u) <= 1e-12:
            raise ValueError("tangent must not be parallel to normal")
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v, n


def _face_grid(
    center: np.ndarray,
    axis_a: np.ndarray,
    axis_b: np.ndarray,
    half_a: float,
    half_b: float,
    max_edge: float,
) -> tuple[np.ndarray, np.ndarray]:
    count_a = max(1, int(math.ceil(2.0 * half_a / max_edge)))
    count_b = max(1, int(math.ceil(2.0 * half_b / max_edge)))
    values_a = np.linspace(-half_a, half_a, count_a + 1)
    values_b = np.linspace(-half_b, half_b, count_b + 1)

    vertices = np.empty(((count_a + 1) * (count_b + 1), 3), dtype=np.float64)
    for ia, a in enumerate(values_a):
        for ib, b in enumerate(values_b):
            vertices[ia * (count_b + 1) + ib] = center + a * axis_a + b * axis_b

    faces = []
    stride = count_b + 1
    for ia in range(count_a):
        for ib in range(count_b):
            p00 = ia * stride + ib
            p10 = (ia + 1) * stride + ib
            p11 = (ia + 1) * stride + ib + 1
            p01 = ia * stride + ib + 1
            faces.append((p00, p10, p11))
            faces.append((p00, p11, p01))
    return vertices, np.asarray(faces, dtype=np.int64)


def merge_surfaces(surfaces: Iterable[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Merge independent triangulated surfaces, offsetting face indices."""
    vertices, faces, offset = [], [], 0
    for v, f in surfaces:
        v = np.asarray(v, dtype=np.float64).reshape(-1, 3)
        f = np.asarray(f, dtype=np.int64).reshape(-1, 3)
        vertices.append(v)
        faces.append(f + offset)
        offset += len(v)
    if not vertices:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int64)
    return np.concatenate(vertices), np.concatenate(faces)


def _weld_vertices(
    vertices: np.ndarray,
    faces: np.ndarray,
    tolerance: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    scale = 1.0 / tolerance
    keys = np.round(vertices * scale).astype(np.int64)
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    welded = np.empty((len(unique_keys), 3), dtype=np.float64)
    for index in range(len(unique_keys)):
        welded[index] = vertices[inverse == index].mean(axis=0)
    return welded, inverse[faces]


def thin_box_reflector(
    center: Iterable[float],
    normal: Iterable[float],
    size: Iterable[float],
    thickness: float,
    max_edge: float,
    tangent: Iterable[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Closed rectangular rigid reflector as a thin box.

    The returned triangles are oriented outward from the solid, which is the
    normal convention expected by the rigid-obstacle BEM code.
    """
    center = np.asarray(center, dtype=np.float64)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("center must be a finite 3-vector")
    sx, sy = np.asarray(size, dtype=np.float64).reshape(2)
    thickness = float(thickness)
    max_edge = float(max_edge)
    if not (np.isfinite(sx) and np.isfinite(sy) and sx > 0.0 and sy > 0.0):
        raise ValueError("size must contain two positive finite lengths")
    if not np.isfinite(thickness) or thickness <= 0.0:
        raise ValueError("thickness must be positive")
    if not np.isfinite(max_edge) or max_edge <= 0.0:
        raise ValueError("max_edge must be positive")

    u, v, n = _frame(normal, tangent)
    hx, hy, hz = sx * 0.5, sy * 0.5, thickness * 0.5
    surfaces = [
        _face_grid(center + hz * n, u, v, hx, hy, max_edge),
        _face_grid(center - hz * n, u, v, hx, hy, max_edge),
        _face_grid(center + hx * u, v, n, hy, hz, max_edge),
        _face_grid(center - hx * u, v, n, hy, hz, max_edge),
        _face_grid(center + hy * v, u, n, hx, hz, max_edge),
        _face_grid(center - hy * v, u, n, hx, hz, max_edge),
    ]
    vertices, faces = merge_surfaces(surfaces)
    vertices, faces = _weld_vertices(vertices, faces)

    tri = vertices[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    outward = ((tri.mean(axis=1) - center) * normals).sum(axis=1) > 0.0
    faces[~outward] = faces[~outward][:, ::-1]
    return vertices, faces


def reflector_from_config(obstacle: dict, default_max_edge: float) -> tuple[np.ndarray, np.ndarray]:
    """Build a finite reflector mesh from a BEM obstacle config."""
    center = obstacle.get("center_m", obstacle.get("center"))
    normal = obstacle.get("normal", obstacle.get("normal_m"))
    size = obstacle.get("size_m", obstacle.get("size"))
    if center is None or normal is None or size is None:
        raise ValueError("reflector obstacle needs center_m, normal and size_m")
    return thin_box_reflector(
        center=center,
        normal=normal,
        size=size,
        thickness=float(obstacle.get("thickness_m", obstacle.get("thickness", 0.003))),
        max_edge=float(obstacle.get("element_size_m", default_max_edge)),
        tangent=obstacle.get("tangent"),
    )
