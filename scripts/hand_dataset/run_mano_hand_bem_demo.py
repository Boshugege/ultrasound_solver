"""Run BEM focusing demo from a MANO-exported hand mesh.

First create ``mano_scene_hand.vtk`` using ``preview_mano_pose.py``. Then run:

  python scripts/hand_dataset/run_mano_hand_bem_demo.py ^
    --mano-scene-mesh outputs/mano_pose_preview/mano_scene_hand.vtk
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyvista as pv


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from algorithm.bem.scene_solver import BemSceneSolver  # noqa: E402
from algorithm.config import SimulationConfig  # noqa: E402

from run_ok_hand_bem_demo import (  # noqa: E402
    _build_config,
    _plot_result,
    _sample_level_set,
    _write_report,
    _write_yaml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mano-scene-mesh", default=str(ROOT / "outputs/mano_pose_preview/mano_scene_hand.vtk"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/mano_hand_bem_demo"))
    parser.add_argument("--target", nargs=3, type=float, default=[0.08, 0.08, 0.065])
    parser.add_argument("--level-threshold", type=float, default=0.40)
    parser.add_argument("--level-resolution", type=int, default=35)
    parser.add_argument("--level-half-width", type=float, default=0.026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mesh_file = Path(args.mano_scene_mesh).expanduser().resolve()
    if not mesh_file.is_file():
        raise FileNotFoundError(
            f"MANO scene mesh not found: {mesh_file}\n"
            "Run scripts\\hand_dataset\\preview_mano_pose.py first."
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh = pv.read(mesh_file).extract_surface(algorithm="dataset_surface").triangulate().clean()
    scene_mesh = output_dir / "mano_hand_scene.vtk"
    mesh.save(scene_mesh)
    points = np.asarray(mesh.points, dtype=np.float64)
    faces = np.asarray(mesh.faces).reshape(-1, 4)[:, 1:].astype(np.int64)
    mesh_center = 0.5 * (points.min(axis=0) + points.max(axis=0))
    target = np.asarray(args.target, dtype=np.float64)

    raw_cfg = _build_config(scene_mesh.resolve(), mesh_center, target, output_dir)
    raw_cfg["obstacles"][0]["remesh"] = False
    cfg_path = _write_yaml(raw_cfg, output_dir / "mano_hand_bem_scene.yaml")

    started = time.perf_counter()
    solver = BemSceneSolver(SimulationConfig.from_yaml(str(cfg_path)))
    centers = solver.transducers.centers.copy()
    response = solver.target_response(target[None, :])[0]
    response_s = time.perf_counter() - started
    phases = np.mod(-np.angle(response), 2.0 * np.pi)
    np.savez(output_dir / "mano_hand_bem_response.npz", response=response, phases=phases, target=target)

    axes, amplitude, verts, level_faces, peak, level = _sample_level_set(
        solver, phases, target, half_width=args.level_half_width,
        resolution=args.level_resolution, threshold=args.level_threshold,
        obstacle_mesh=mesh, surface_clearance=0.001,
    )
    np.savez(output_dir / "mano_hand_levelset.npz", axes=np.array(axes, dtype=object),
             amplitude=amplitude, vertices=verts, faces=level_faces, peak=peak, level=level)

    metrics = {
        "target": np.array2string(target, precision=4),
        "bem_elements": int(solver.bem.n),
        "response_s": float(response_s),
        "peak_amplitude": float(peak),
        "level_threshold": float(args.level_threshold),
        "level_value": float(level),
        "level_triangles": int(len(level_faces)),
        "surface_clearance_m": 0.001,
        "mesh_file": str(scene_mesh),
        "config_file": str(cfg_path),
        "source_mesh": str(mesh_file),
    }
    png_path = output_dir / "mano_hand_bem_visualization.png"
    _plot_result(
        png_path, points, faces, centers, phases, target, verts, level_faces, metrics,
        scene_name="MANO hand",
    )
    solver.close()
    _write_report(
        output_dir / "mano_hand_bem_report.html", png_path.name, metrics,
        scene_name="MANO 手",
    )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
