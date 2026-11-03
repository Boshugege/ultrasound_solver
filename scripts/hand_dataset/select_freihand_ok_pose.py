"""Select real MANO gesture poses from FreiHAND annotations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from algorithm.hand.mano_model import MANOHandModel  # noqa: E402


# Official FreiHAND/MANO fingertip vertices from utils/mano_utils.py.
TIP_VERTEX_IDS = {
    "thumb": 744,
    "index": 320,
    "middle": 443,
    "ring": 555,
    "little": 672,
}

# MANO's 16 regressed joints are ordered differently from FreiHAND's 21 points.
FINGER_JOINT_IDS = {
    "thumb": (13, 14, 15),
    "index": (1, 2, 3),
    "middle": (4, 5, 6),
    "ring": (10, 11, 12),
    "little": (7, 8, 9),
}


def finger_straightness(joints: np.ndarray, tip: np.ndarray, joint_ids: tuple[int, int, int]) -> np.ndarray:
    """Return endpoint distance divided by articulated chain length."""
    chain = np.concatenate([joints[:, joint_ids], tip[:, None, :]], axis=1)
    chain_length = np.linalg.norm(np.diff(chain, axis=1), axis=2).sum(axis=1)
    endpoint_distance = np.linalg.norm(chain[:, -1] - chain[:, 0], axis=1)
    return endpoint_distance / np.maximum(chain_length, 1.0e-9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default=str(ROOT / "data/freihand/training_mano.json"))
    parser.add_argument("--mano-model-dir", default=str(ROOT / "models/mano"))
    parser.add_argument("--gesture", choices=["ok", "open", "fist", "point"], default="ok")
    parser.add_argument("--output", default="")
    parser.add_argument("--report", default="")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def _classify_gesture(gesture: str, metrics: dict) -> tuple[bool, float, dict]:
    straight = metrics["straightness"]
    extended = np.array([straight[name] for name in ("index", "middle", "ring", "little")])
    curled = np.array([straight[name] for name in ("middle", "ring", "little")])

    if gesture == "ok":
        valid = (
            metrics["thumb_index_mm"] < 15.0
            and metrics["other_thumb_min_mm"] > 30.0
            and straight["index"] < 0.82
            and float(curled.mean()) > 0.88
        )
        score = (
            metrics["thumb_index_mm"]
            + 2.0 * max(0.0, 35.0 - metrics["other_thumb_min_mm"])
            + 100.0 * max(0.0, straight["index"] - 0.78)
            + 100.0 * max(0.0, 0.92 - float(curled.mean()))
        )
        criteria = {
            "thumb_index_mm_max": 15.0,
            "other_thumb_tip_mm_min": 30.0,
            "index_straightness_max": 0.82,
            "extended_fingers_straightness_min": 0.88,
        }
    elif gesture == "open":
        valid = (
            float(extended.min()) > 0.90
            and straight["thumb"] > 0.78
            and metrics["adjacent_extended_tip_min_mm"] > 10.0
            and metrics["index_little_tip_mm"] > 35.0
        )
        score = (
            200.0 * max(0.0, 0.96 - float(extended.mean()))
            + 100.0 * max(0.0, 0.86 - straight["thumb"])
            + max(0.0, 50.0 - metrics["index_little_tip_mm"])
        )
        criteria = {
            "four_finger_straightness_min": 0.90,
            "thumb_straightness_min": 0.78,
            "adjacent_extended_tip_mm_min": 10.0,
            "index_little_tip_mm_min": 35.0,
        }
    elif gesture == "fist":
        valid = (
            float(extended.max()) < 0.82
            and float(extended.mean()) < 0.70
            and metrics["extended_tip_wrist_mean_mm"] < 95.0
            and min(metrics["thumb_index_mm"], metrics["thumb_middle_mm"]) < 40.0
        )
        score = (
            120.0 * float(extended.mean())
            + 0.4 * metrics["extended_tip_wrist_mean_mm"]
            + 50.0 * max(0.0, float(extended.max()) - 0.75)
            + min(metrics["thumb_index_mm"], metrics["thumb_middle_mm"])
        )
        criteria = {
            "four_finger_straightness_max": 0.82,
            "four_finger_straightness_mean_max": 0.70,
            "extended_tip_wrist_mean_mm_max": 95.0,
            "thumb_to_index_or_middle_mm_max": 40.0,
        }
    else:
        valid = (
            straight["index"] > 0.92
            and float(curled.max()) < 0.84
            and float(curled.mean()) < 0.74
            and metrics["index_to_other_tip_min_mm"] > 32.0
        )
        score = (
            150.0 * max(0.0, 0.97 - straight["index"])
            + 120.0 * float(curled.mean())
            + 2.0 * max(0.0, 40.0 - metrics["index_to_other_tip_min_mm"])
        )
        criteria = {
            "index_straightness_min": 0.92,
            "other_finger_straightness_max": 0.84,
            "other_finger_straightness_mean_max": 0.74,
            "index_to_other_tip_mm_min": 32.0,
        }
    return bool(valid), float(score), criteria


def main() -> None:
    args = parse_args()
    annotations = json.loads(Path(args.annotations).read_text(encoding="utf-8"))
    values = np.asarray([sample[0] for sample in annotations], dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < 61:
        raise ValueError(f"Expected FreiHAND MANO rows with >=61 values, got {values.shape}")

    model = MANOHandModel(args.mano_model_dir, hand="right", device="cpu")
    all_metrics = []
    tip_ids = np.array(list(TIP_VERTEX_IDS.values()), dtype=np.int64)
    with torch.no_grad():
        for start in range(0, len(values), args.batch_size):
            batch = values[start:start + args.batch_size]
            output = model.layer(
                global_orient=torch.from_numpy(batch[:, :3]),
                hand_pose=torch.from_numpy(batch[:, 3:48]),
                betas=torch.from_numpy(batch[:, 48:58]),
            )
            vertices = output.vertices.detach().cpu().numpy()
            joints = output.joints.detach().cpu().numpy()
            tips = vertices[:, tip_ids]
            wrist = joints[:, 0]
            thumb = tips[:, 0]
            distances = np.linalg.norm(tips[:, 1:] - thumb[:, None, :], axis=2) * 1000.0
            straightness = {
                name: finger_straightness(joints, tips[:, tip_index], FINGER_JOINT_IDS[name])
                for tip_index, name in enumerate(TIP_VERTEX_IDS)
            }
            for local in range(len(batch)):
                thumb_index, thumb_middle, thumb_ring, thumb_little = distances[local]
                sample_tips = tips[local]
                adjacent = np.linalg.norm(np.diff(sample_tips[1:], axis=0), axis=1) * 1000.0
                tip_wrist = np.linalg.norm(sample_tips - wrist[local], axis=1) * 1000.0
                sample_metrics = {
                    "index": start + local,
                    "thumb_index_mm": float(thumb_index),
                    "thumb_middle_mm": float(thumb_middle),
                    "thumb_ring_mm": float(thumb_ring),
                    "thumb_little_mm": float(thumb_little),
                    "other_thumb_min_mm": float(min(thumb_middle, thumb_ring, thumb_little)),
                    "adjacent_extended_tip_min_mm": float(adjacent.min()),
                    "index_little_tip_mm": float(np.linalg.norm(sample_tips[1] - sample_tips[4]) * 1000.0),
                    "index_to_other_tip_min_mm": float(
                        np.linalg.norm(sample_tips[2:] - sample_tips[1], axis=1).min() * 1000.0
                    ),
                    "extended_tip_wrist_mean_mm": float(tip_wrist[1:].mean()),
                    "straightness": {
                        name: float(straightness[name][local])
                        for name in TIP_VERTEX_IDS
                    },
                }
                valid, score, criteria = _classify_gesture(args.gesture, sample_metrics)
                sample_metrics["valid"] = valid
                sample_metrics["score"] = score
                all_metrics.append(sample_metrics)

    valid = [item for item in all_metrics if item["valid"]]
    ranked = sorted(valid if valid else all_metrics, key=lambda item: item["score"])
    top = ranked[:args.top_k]
    best = top[0]
    row = values[best["index"]]
    pose = row[:48].astype(np.float64)
    source_global_orient = pose[:3].copy()
    pose[:3] = 0.0
    params = {
        "right": {
            "pose": pose.astype(float).tolist(),
            "shape": row[48:58].astype(float).tolist(),
            "trans": [0.0, 0.0, 0.0],
        },
        "source": {
            "dataset": "FreiHAND_pub_v2",
            "gesture": args.gesture,
            "sample_index": int(best["index"]),
            "original_global_orient": source_global_orient.astype(float).tolist(),
            "camera_proxy": {
                "uv_root": row[58:60].astype(float).tolist(),
                "scale": float(row[60]),
            },
            "selection_metrics": best,
        },
    }

    output_path = Path(args.output) if args.output else (
        ROOT / "configs" / "mano" / f"freihand_{args.gesture}_pose_right.json"
    )
    report_path = Path(args.report) if args.report else (
        ROOT / "outputs" / "mano_pose_selection" / f"freihand_{args.gesture}_candidates.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(params, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps({
        "annotation_count": len(values),
        "valid_candidate_count": len(valid),
        "gesture": args.gesture,
        "criteria": criteria,
        "top_candidates": top,
    }, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output_path),
        "report": str(report_path),
        "annotation_count": len(values),
        "valid_candidate_count": len(valid),
        "best": best,
    }, indent=2))


if __name__ == "__main__":
    main()
