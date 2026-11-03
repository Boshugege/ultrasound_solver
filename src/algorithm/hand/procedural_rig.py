"""Procedural driver for the fixed watertight hand mesh.

This is a deterministic first rig for data-pipeline validation. It does not
replace a production anatomical rig with manually authored joints and skinning
weights; it provides a stable mapping from a small set of finger curl values to
a deformed closed mesh so the BEM and surrogate-data flow can be exercised now.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np


FINGER_NAMES = ("thumb", "index", "middle", "ring", "little")


@dataclass(frozen=True)
class FingerRegion:
    name: str
    center_xy: np.ndarray
    base_center: np.ndarray
    tip_center: np.ndarray
    base_z: float
    tip_z: float
    bend_direction: np.ndarray


@dataclass(frozen=True)
class HandGesture:
    """Low-dimensional procedural gesture parameters.

    ``curl`` is in radians. Positive values bend a finger toward the palm.
    """

    curl: Dict[str, float]

    @classmethod
    def open(cls) -> "HandGesture":
        return cls({name: 0.0 for name in FINGER_NAMES})

    @classmethod
    def ok(cls) -> "HandGesture":
        return cls({
            "thumb": np.deg2rad(82.0),
            "index": np.deg2rad(92.0),
            "middle": np.deg2rad(18.0),
            "ring": np.deg2rad(14.0),
            "little": np.deg2rad(12.0),
        })


def _faces_from_polydata(mesh) -> np.ndarray:
    faces = np.asarray(mesh.faces).reshape(-1, 4)
    if not np.all(faces[:, 0] == 3):
        raise ValueError("Hand mesh must be triangulated")
    return faces[:, 1:].astype(np.int64)


def _polydata_from_arrays(points: np.ndarray, faces: np.ndarray):
    import pyvista as pv

    flat = np.column_stack([np.full(len(faces), 3, dtype=np.int64), faces]).ravel()
    return pv.PolyData(points.astype(np.float64), flat)


def _normalize(v: np.ndarray, fallback: Iterable[float]) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-12:
        v = np.asarray(fallback, dtype=np.float64)
        n = np.linalg.norm(v)
    return v / n


def _rodrigues(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Vectorized Rodrigues rotation matrices for one axis and many angles."""
    axis = _normalize(axis, (1.0, 0.0, 0.0))
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    one = 1.0 - c
    out = np.empty((len(np.atleast_1d(angle)), 3, 3), dtype=np.float64)
    out[:, 0, 0] = c + x * x * one
    out[:, 0, 1] = x * y * one - z * s
    out[:, 0, 2] = x * z * one + y * s
    out[:, 1, 0] = y * x * one + z * s
    out[:, 1, 1] = c + y * y * one
    out[:, 1, 2] = y * z * one - x * s
    out[:, 2, 0] = z * x * one - y * s
    out[:, 2, 1] = z * y * one + x * s
    out[:, 2, 2] = c + z * z * one
    return out


def rotation_matrix_xyz(rotation_deg: Iterable[float]) -> np.ndarray:
    rx, ry, rz = np.deg2rad(np.asarray(rotation_deg, dtype=np.float64))
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cx * 0.0 + cy]])
    rot_z = np.array([[cz, -sx * 0.0 + -np.sin(rz), 0.0], [np.sin(rz), cz, 0.0], [0.0, 0.0, 1.0]])
    return rot_z @ rot_y @ rot_x


class ProceduralHandRig:
    """Automatic region-based rig for the fixed hand mesh."""

    def __init__(self, points: np.ndarray, faces: np.ndarray, regions: Tuple[FingerRegion, ...],
                 vertex_region: np.ndarray, palm_center: np.ndarray):
        self.points = np.asarray(points, dtype=np.float64)
        self.faces = np.asarray(faces, dtype=np.int64)
        self.regions = regions
        self.vertex_region = np.asarray(vertex_region, dtype=np.int64)
        self.palm_center = np.asarray(palm_center, dtype=np.float64)

    @classmethod
    def from_vtk(cls, path: str | Path, seed: int = 3) -> "ProceduralHandRig":
        import pyvista as pv
        from scipy.cluster.vq import kmeans2

        mesh = pv.read(path).extract_surface(algorithm="dataset_surface").triangulate().clean()
        points = np.asarray(mesh.points, dtype=np.float64)
        faces = _faces_from_polydata(mesh)

        z = points[:, 2]
        palm_z = float(np.quantile(z, 0.45))
        cluster_mask = z > np.quantile(z, 0.58)
        centers, labels = kmeans2(points[cluster_mask, :2], 5, minit="++", seed=seed)

        palm_points = points[z <= np.quantile(z, 0.55)]
        palm_center = np.median(palm_points, axis=0)
        palm_xy = palm_center[:2]

        # The thumb is usually the cluster that is farthest away from the other
        # four in the transverse plane. The remaining fingers are ordered left to
        # right, which is enough for a deterministic procedural gesture.
        distances = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
        thumb_idx = int(np.argmax(np.sort(distances, axis=1)[:, -2:].sum(axis=1)))
        rest = [i for i in range(5) if i != thumb_idx]
        rest = sorted(rest, key=lambda i: centers[i, 0])
        ordered = [thumb_idx] + rest

        high = z > palm_z
        nearest = np.argmin(np.linalg.norm(points[:, None, :2] - centers[None, :, :], axis=2), axis=1)
        vertex_region = np.full(len(points), -1, dtype=np.int64)
        for out_idx, cluster_idx in enumerate(ordered):
            vertex_region[(nearest == cluster_idx) & high] = out_idx

        regions = []
        for out_idx, cluster_idx in enumerate(ordered):
            mask = vertex_region == out_idx
            pts = points[mask]
            if len(pts) < 8:
                base_z = float(palm_z)
                tip_z = float(points[:, 2].max())
                base_center = np.array([centers[cluster_idx, 0], centers[cluster_idx, 1], base_z])
                tip_center = np.array([centers[cluster_idx, 0], centers[cluster_idx, 1], tip_z])
            else:
                base_z = float(np.quantile(pts[:, 2], 0.05))
                tip_z = float(np.quantile(pts[:, 2], 0.98))
                base_band = pts[pts[:, 2] <= np.quantile(pts[:, 2], 0.18)]
                tip_band = pts[pts[:, 2] >= np.quantile(pts[:, 2], 0.88)]
                base_center = np.median(base_band, axis=0)
                tip_center = np.median(tip_band, axis=0)
            direction_xy = palm_xy - base_center[:2]
            bend_direction = np.array([direction_xy[0], direction_xy[1], -0.15])
            bend_direction = _normalize(bend_direction, (0.0, -1.0, -0.2))
            regions.append(FingerRegion(
                name=FINGER_NAMES[out_idx],
                center_xy=np.asarray(centers[cluster_idx], dtype=np.float64),
                base_center=base_center,
                tip_center=tip_center,
                base_z=base_z,
                tip_z=tip_z,
                bend_direction=bend_direction,
            ))
        return cls(points, faces, tuple(regions), vertex_region, palm_center)

    def deform(self, gesture: HandGesture) -> np.ndarray:
        deformed = self.points.copy()
        finger_axis = np.array([0.0, 0.0, 1.0])
        centers = np.array([r.center_xy for r in self.regions])
        if len(centers) > 1:
            center_dist = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
            center_dist[center_dist == 0.0] = np.inf
            sigma = float(np.clip(0.42 * np.median(center_dist.min(axis=1)), 0.010, 0.020))
        else:
            sigma = 0.014
        for index, region in enumerate(self.regions):
            curl = float(gesture.curl.get(region.name, 0.0))
            if abs(curl) < 1e-9:
                continue
            pts = self.points
            span = max(region.tip_z - region.base_z, 1e-6)
            s = np.clip((pts[:, 2] - region.base_z) / span, 0.0, 1.0)
            smooth = s * s * (3.0 - 2.0 * s)
            xy_distance = np.linalg.norm(pts[:, :2] - region.center_xy[None, :], axis=1)
            influence = np.exp(-0.5 * (xy_distance / sigma)**2) * smooth
            influence = np.clip(influence, 0.0, 1.0)
            axis = np.cross(finger_axis, region.bend_direction)
            axis = _normalize(axis, (1.0, 0.0, 0.0))
            angles = curl * smooth
            rotations = _rodrigues(axis, angles)
            local = pts - region.base_center
            bent = region.base_center + np.einsum("nij,nj->ni", rotations, local)
            deformed += influence[:, None] * (bent - pts)

        # For the OK preset, explicitly bring the thumb and index fingertips
        # together with smooth distal weights. This keeps the mesh continuous
        # while making the intended gesture visually clear.
        if (gesture.curl.get("thumb", 0.0) > 0.8 and gesture.curl.get("index", 0.0) > 0.8
                and len(self.regions) >= 2):
            thumb, index = self.regions[0], self.regions[1]
            contact = 0.5 * (thumb.tip_center + index.tip_center)
            contact[2] -= 0.018
            for region in (thumb, index):
                pts = self.points
                span = max(region.tip_z - region.base_z, 1e-6)
                distal = np.clip((pts[:, 2] - (region.base_z + 0.52 * span)) / (0.48 * span), 0.0, 1.0)
                distal = distal * distal * (3.0 - 2.0 * distal)
                xy_distance = np.linalg.norm(pts[:, :2] - region.center_xy[None, :], axis=1)
                influence = np.exp(-0.5 * (xy_distance / (0.82 * sigma))**2) * distal**1.6
                influence = np.clip(0.85 * influence, 0.0, 0.95)
                deformed += influence[:, None] * (contact[None, :] - deformed)
        return deformed

    def mesh(self, points: np.ndarray | None = None):
        return _polydata_from_arrays(self.points if points is None else points, self.faces)

    def write_mesh(self, path: str | Path, points: np.ndarray | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.mesh(points).save(path)
        return path


def place_points(points: np.ndarray, center_m=(0.095, 0.080, 0.082), rotation_deg=(0.0, 90.0, 0.0),
                 scale=0.42, min_z=0.024) -> np.ndarray:
    """Scale, rotate and place hand points into the acoustic scene."""
    pts = np.asarray(points, dtype=np.float64)
    source_center = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
    placed = (pts - source_center) * float(scale)
    placed = placed @ rotation_matrix_xyz(rotation_deg).T
    target_center = np.asarray(center_m, dtype=np.float64)
    placed += target_center - 0.5 * (placed.min(axis=0) + placed.max(axis=0))
    dz = float(min_z) - float(placed[:, 2].min())
    if dz > 0:
        placed[:, 2] += dz
    return placed
