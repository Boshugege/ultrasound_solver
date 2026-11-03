"""MANO parameter support for fixed-hand BEM scenes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ManoParameters:
    """Single-frame MANO parameters.

    ``pose`` is 48D axis-angle: 3D global orientation + 45D articulated hand pose.
    ``shape`` is 10D MANO beta. ``trans`` is 3D translation in the dataset frame.
    """

    pose: np.ndarray
    shape: np.ndarray
    trans: np.ndarray
    hand: str = "right"

    @classmethod
    def from_mapping(cls, data: dict, hand: str = "right") -> "ManoParameters":
        entry = data.get(hand, data)
        if entry is None:
            raise ValueError(f"MANO entry for {hand!r} is missing")
        pose = np.asarray(entry["pose"], dtype=np.float32).reshape(-1)
        shape = np.asarray(entry.get("shape", np.zeros(10)), dtype=np.float32).reshape(-1)
        trans = np.asarray(entry.get("trans", np.zeros(3)), dtype=np.float32).reshape(-1)
        if pose.shape != (48,):
            raise ValueError(f"MANO pose must be 48D, got {pose.shape}")
        if shape.shape != (10,):
            raise ValueError(f"MANO shape must be 10D, got {shape.shape}")
        if trans.shape != (3,):
            raise ValueError(f"MANO trans must be 3D, got {trans.shape}")
        return cls(pose=pose, shape=shape, trans=trans, hand=hand)


def resolve_mano_model_dir(model_dir: str | Path) -> Path:
    """Return a directory suitable for ``smplx.create(..., model_type='mano')``.

    Accepted layouts:

    - ``<model_dir>/MANO_RIGHT.pkl``
    - ``<model_dir>/MANO_LEFT.pkl``
    - ``<model_dir>/mano/MANO_RIGHT.pkl``
    """

    root = Path(model_dir).expanduser().resolve()
    candidates = [
        root / "MANO_RIGHT.pkl",
        root / "MANO_LEFT.pkl",
        root / "mano" / "MANO_RIGHT.pkl",
        root / "mano" / "MANO_LEFT.pkl",
    ]
    if not any(path.is_file() for path in candidates):
        lines = "\n".join(f"  - {path}" for path in candidates)
        raise FileNotFoundError(
            "MANO model file was not found. Download MANO from https://mano.is.tue.mpg.de/ "
            "after accepting its license, then place MANO_RIGHT.pkl/MANO_LEFT.pkl in one of:\n"
            f"{lines}"
        )
    return root


class MANOHandModel:
    """Thin wrapper around ``smplx`` MANO for one-frame mesh generation."""

    def __init__(self, model_dir: str | Path, hand: str = "right", device: str = "cpu",
                 flat_hand_mean: bool = False):
        import inspect

        if not hasattr(inspect, "getargspec"):
            inspect.getargspec = inspect.getfullargspec
        legacy_numpy_names = {
            "bool": np.bool_,
            "int": int,
            "float": float,
            "complex": complex,
            "object": object,
            "unicode": str,
            "str": str,
        }
        for name, value in legacy_numpy_names.items():
            if name not in np.__dict__:
                setattr(np, name, value)
        import smplx
        import torch

        self.model_dir = resolve_mano_model_dir(model_dir)
        self.hand = hand.lower()
        if self.hand not in {"right", "left"}:
            raise ValueError("hand must be 'right' or 'left'")
        model_name = f"MANO_{self.hand.upper()}.pkl"
        direct_model = self.model_dir / model_name
        nested_model = self.model_dir / "mano" / model_name
        model_path = direct_model if direct_model.is_file() else nested_model
        if not model_path.is_file():
            raise FileNotFoundError(f"MANO model for {self.hand!r} was not found under {self.model_dir}")
        self.device = torch.device(device)
        self.torch = torch
        self.layer = smplx.create(
            str(model_path),
            model_type="mano",
            is_rhand=(self.hand == "right"),
            use_pca=False,
            num_pca_comps=45,
            flat_hand_mean=flat_hand_mean,
            batch_size=1,
        ).to(self.device)
        self.faces = np.asarray(self.layer.faces, dtype=np.int64)

    def mesh_from_parameters(self, params: ManoParameters, apply_translation: bool = False):
        torch = self.torch
        pose = np.asarray(params.pose, dtype=np.float32).reshape(1, 48)
        shape = np.asarray(params.shape, dtype=np.float32).reshape(1, 10)
        trans = np.asarray(params.trans, dtype=np.float32).reshape(1, 3)
        kwargs = {
            "global_orient": torch.tensor(pose[:, :3], dtype=torch.float32, device=self.device),
            "hand_pose": torch.tensor(pose[:, 3:], dtype=torch.float32, device=self.device),
            "betas": torch.tensor(shape, dtype=torch.float32, device=self.device),
        }
        if apply_translation:
            kwargs["transl"] = torch.tensor(trans, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            out = self.layer(**kwargs)
        vertices = out.vertices[0].detach().cpu().numpy().astype(np.float64)
        joints = out.joints[0].detach().cpu().numpy().astype(np.float64)
        return vertices, self.faces.copy(), joints

    def mesh_from_arrays(self, pose: Iterable[float], shape: Iterable[float] | None = None,
                         trans: Iterable[float] | None = None, apply_translation: bool = False):
        params = ManoParameters(
            pose=np.asarray(pose, dtype=np.float32),
            shape=np.zeros(10, dtype=np.float32) if shape is None else np.asarray(shape, dtype=np.float32),
            trans=np.zeros(3, dtype=np.float32) if trans is None else np.asarray(trans, dtype=np.float32),
            hand=self.hand,
        )
        return self.mesh_from_parameters(params, apply_translation=apply_translation)


def polydata_from_arrays(vertices: np.ndarray, faces: np.ndarray):
    import pyvista as pv

    cells = np.column_stack([np.full(len(faces), 3, dtype=np.int64), faces]).ravel()
    return pv.PolyData(np.asarray(vertices, dtype=np.float64), cells)


def cap_open_boundaries(vertices: np.ndarray, faces: np.ndarray, hole_size: float = 1.0):
    """Close MANO's wrist opening for rigid-obstacle BEM use."""
    mesh = polydata_from_arrays(vertices, faces).extract_surface(
        algorithm="dataset_surface"
    ).triangulate().clean()
    if mesh.n_open_edges:
        mesh = mesh.fill_holes(float(hole_size)).triangulate().clean()
    if mesh.n_open_edges:
        raise ValueError(f"Failed to close MANO mesh: open edges={mesh.n_open_edges}")
    capped_vertices = np.asarray(mesh.points, dtype=np.float64)
    capped_faces = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:].astype(np.int64)
    return capped_vertices, capped_faces


def write_vtk(path: str | Path, vertices: np.ndarray, faces: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    polydata_from_arrays(vertices, faces).save(path)
    return path


def place_mano_vertices(vertices: np.ndarray, center_m=(0.115, 0.115, 0.075),
                        rotation_deg=(0.0, 0.0, 0.0), scale: float = 1.0,
                        min_z: float = 0.024) -> np.ndarray:
    from algorithm.hand.procedural_rig import place_points

    return place_points(vertices, center_m=center_m, rotation_deg=rotation_deg, scale=scale, min_z=min_z)


def place_mano_mesh(vertices: np.ndarray, joints: np.ndarray,
                    center_m=(0.095, 0.100, 0.075),
                    rotation_deg=(0.0, 0.0, 0.0), scale: float = 1.0,
                    min_z: float = 0.024):
    """Place MANO vertices and joints with one shared rigid transform."""
    from algorithm.hand.procedural_rig import rotation_matrix_xyz

    vertices = np.asarray(vertices, dtype=np.float64)
    joints = np.asarray(joints, dtype=np.float64)
    source_center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    rotation = rotation_matrix_xyz(rotation_deg)
    scene_vertices = ((vertices - source_center) * float(scale)) @ rotation.T
    scene_joints = ((joints - source_center) * float(scale)) @ rotation.T
    rotated_center = 0.5 * (scene_vertices.min(axis=0) + scene_vertices.max(axis=0))
    translation = np.asarray(center_m, dtype=np.float64) - rotated_center
    scene_vertices += translation
    scene_joints += translation
    dz = float(min_z) - float(scene_vertices[:, 2].min())
    if dz > 0.0:
        scene_vertices[:, 2] += dz
        scene_joints[:, 2] += dz
    return scene_vertices, scene_joints
