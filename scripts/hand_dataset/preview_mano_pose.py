"""Preview and export a MANO hand pose as VTK/PNG.

Examples:

  python scripts/hand_dataset/preview_mano_pose.py ^
    --mano-model-dir models/mano ^
    --interhand-mano-json path/to/InterHand2.6M_train_MANO_NeuralAnnot.json ^
    --capture-id 0 --frame-idx 123 --hand right
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from algorithm.hand.mano_model import (  # noqa: E402
    MANOHandModel,
    ManoParameters,
    cap_open_boundaries,
    place_mano_mesh,
    write_vtk,
)


def _load_interhand_params(path: Path, capture_id: str, frame_idx: str, hand: str) -> ManoParameters:
    data = json.loads(path.read_text(encoding="utf-8"))
    capture_key = str(capture_id)
    frame_key = str(frame_idx)
    try:
        frame = data[capture_key][frame_key]
    except KeyError as exc:
        raise KeyError(f"Could not find capture={capture_key!r}, frame={frame_key!r} in {path}") from exc
    if frame.get(hand) is None:
        raise ValueError(f"Frame exists but MANO {hand!r} entry is null")
    return ManoParameters.from_mapping(frame, hand=hand)


def _load_params_json(path: Path, hand: str) -> ManoParameters:
    data = json.loads(path.read_text(encoding="utf-8"))
    return ManoParameters.from_mapping(data, hand=hand)


TIP_VERTEX_IDS = (744, 320, 443, 555, 672)


def _plot_preview(path: Path, vertices: np.ndarray, faces: np.ndarray, joints: np.ndarray,
                  gesture: str = "mano") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    center = 0.5 * (lo + hi)
    radius = float(np.max(hi - lo) * 0.58)
    tips = vertices[np.asarray(TIP_VERTEX_IDS)]

    fig = plt.figure(figsize=(15, 5.2), dpi=150)
    views = [
        ("Top", 90, -90),
        ("Side", 0, -90),
        ("Perspective", 24, -62),
    ]
    for panel, (title, elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 3, panel, projection="3d")
        mesh = Poly3DCollection(
            vertices[faces],
            facecolor=(0.68, 0.76, 0.82, 0.92),
            edgecolor=(0.38, 0.44, 0.48, 0.04),
            linewidth=0.2,
        )
        ax.add_collection3d(mesh)
        ax.scatter(tips[:, 0], tips[:, 1], tips[:, 2],
                   c=["#e63946", "#f4a261", "#2a9d8f", "#457b9d", "#7b2cbf"],
                   s=32, depthshade=False)
        if gesture == "ok":
            ax.plot(
                tips[:2, 0], tips[:2, 1], tips[:2, 2],
                color="#111111", linewidth=1.5, linestyle="--",
            )
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_xlabel("x / m")
        ax.set_ylabel("y / m")
        ax.set_zlabel("z / m")
        ax.set_title(title)
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((1, 1, 1))
        if panel < 3:
            ax.set_axis_off()
            ax.set_proj_type("ortho")
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color,
               markeredgecolor="none", label=name, markersize=7)
        for name, color in zip(
            ("Thumb", "Index", "Middle", "Ring", "Little"),
            ("#e63946", "#f4a261", "#2a9d8f", "#457b9d", "#7b2cbf"),
        )
    ]
    fig.legend(handles=legend, loc="lower center", ncol=5, frameon=False)
    title = f"MANO {gesture} pose preview"
    if gesture == "ok":
        title += ": thumb-index contact shown by dashed line"
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mano-model-dir", default="models/mano")
    parser.add_argument("--hand", choices=["right", "left"], default="right")
    parser.add_argument("--interhand-mano-json", default="")
    parser.add_argument("--capture-id", default="")
    parser.add_argument("--frame-idx", default="")
    parser.add_argument("--params-json", default="", help="JSON containing pose/shape/trans or right/left entries")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/mano_pose_preview"))
    parser.add_argument("--scene-center", nargs=3, type=float, default=[0.095, 0.100, 0.075])
    parser.add_argument("--scene-rotation", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument("--scene-scale", type=float, default=1.0)
    parser.add_argument("--min-z", type=float, default=0.024)
    parser.add_argument("--apply-dataset-translation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.interhand_mano_json:
        if not args.capture_id or not args.frame_idx:
            raise SystemExit("--capture-id and --frame-idx are required with --interhand-mano-json")
        params = _load_interhand_params(Path(args.interhand_mano_json), args.capture_id, args.frame_idx, args.hand)
        source = {
            "type": "interhand",
            "path": str(Path(args.interhand_mano_json).resolve()),
            "capture_id": str(args.capture_id),
            "frame_idx": str(args.frame_idx),
            "hand": args.hand,
        }
    elif args.params_json:
        params = _load_params_json(Path(args.params_json), args.hand)
        source = {"type": "params_json", "path": str(Path(args.params_json).resolve()), "hand": args.hand}
        params_data = json.loads(Path(args.params_json).read_text(encoding="utf-8"))
        source["gesture"] = params_data.get("source", {}).get("gesture", "mano")
    else:
        raise SystemExit("Provide --interhand-mano-json or --params-json")

    try:
        model = MANOHandModel(args.mano_model_dir, hand=args.hand)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    vertices, faces, joints = model.mesh_from_parameters(params, apply_translation=args.apply_dataset_translation)
    scene_vertices, scene_joints = place_mano_mesh(
        vertices, joints,
        center_m=args.scene_center,
        rotation_deg=args.scene_rotation,
        scale=args.scene_scale,
        min_z=args.min_z,
    )
    preview_vertices = scene_vertices.copy()
    preview_faces = faces.copy()
    scene_vertices, scene_faces = cap_open_boundaries(scene_vertices, faces)

    vtk_path = write_vtk(out / "mano_scene_hand.vtk", scene_vertices, scene_faces)
    np.savez(
        out / "mano_pose_preview.npz",
        pose=params.pose,
        shape=params.shape,
        trans=params.trans,
        vertices=vertices,
        faces=scene_faces,
        joints=joints,
        scene_vertices=scene_vertices,
        scene_joints=scene_joints,
        source=json.dumps(source),
    )
    _plot_preview(
        out / "mano_pose_preview.png",
        preview_vertices,
        preview_faces,
        scene_joints,
        gesture=source.get("gesture", "mano"),
    )
    (out / "source.json").write_text(json.dumps(source, indent=2), encoding="utf-8")
    print(json.dumps({
        "vtk": str(vtk_path),
        "preview_png": str(out / "mano_pose_preview.png"),
        "vertices": int(len(vertices)),
        "faces": int(len(scene_faces)),
        "source": source,
    }, indent=2))


if __name__ == "__main__":
    main()
