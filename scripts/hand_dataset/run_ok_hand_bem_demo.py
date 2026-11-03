"""Run an OK-gesture hand BEM focusing demo and export visual evidence.

This script validates the end-to-end flow:

    procedural fixed-hand gesture -> deformed rigid mesh -> BEM response
    -> focusing phases -> local amplitude level set -> visualization

The BEM settings here are intentionally demo-grade so the pipeline can run
interactively. They are not the final teacher settings for surrogate training.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from algorithm.bem.scene_solver import BemSceneSolver  # noqa: E402
from algorithm.config import SimulationConfig  # noqa: E402
from algorithm.hand import HandGesture, ProceduralHandRig  # noqa: E402
from algorithm.hand.procedural_rig import place_points  # noqa: E402


def _build_config(mesh_file: Path, mesh_center: np.ndarray, target: np.ndarray, output_dir: Path) -> dict:
    return {
        "mode": "phase_optimization",
        "physics": {
            "sound_speed": 343.0,
            "medium_density": 1.21,
        },
        "transducer_specs": {
            "frequency": 40000.0,
            "array_n": 16,
            "diameter": 0.0098,
            "pitch": 0.0105,
            "geometry": "sonicsurface_reference",
            "reference_anchor_m": [0.08, 0.08],
            "source_radial_order": 8,
            "source_angular_order": 32,
        },
        "domain": {
            "origin_m": [0.0, 0.0, 0.0],
            "box_size": [0.18, 0.18, 0.18],
            "grid_size": [37, 37, 37],
        },
        "boundary_conditions": {
            "-x": "open",
            "+x": "open",
            "-y": "open",
            "+y": "open",
            "-z": "reflecting",
            "+z": "open",
        },
        "targets": [{"point": target.tolist()}],
        "obstacles": [{
            "type": "mesh",
            "file": str(mesh_file),
            "center_m": mesh_center.tolist(),
            "rotation_deg": [0.0, 0.0, 0.0],
            "scale": 1.0,
            "remesh": False,
        }],
        "solver": {
            "backend": "bem",
            "bem_rtol": 1e-2,
            "bem_elements_per_wavelength": 2.0,
            "bem_source_orders": [4, 16],
            "bem_near_source_orders": [8, 32],
        },
        "training": {
            "phase_basis_file": str(output_dir / "demo_basis.npy"),
            "phase_basis_batch_size": 8,
            "voxel_chunk_size": 65536,
        },
        "io": {
            "output_file": str(output_dir / "demo_field.npz"),
            "auto_visualize": False,
        },
    }


def _write_yaml(raw: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def _solve_field_for_phases(solver: BemSceneSolver, phases: np.ndarray, points: np.ndarray,
                            point_chunk: int = 4096) -> np.ndarray:
    """Evaluate the BEM field for fixed drive phases at arbitrary points."""
    weights = solver.transducers.amplitudes * np.exp(1j * np.asarray(phases, dtype=np.float64))
    weights = weights[:, None]
    emitters = np.arange(len(weights))
    bem = solver.bem
    surface_state = None
    if bem.n:
        started = time.perf_counter()
        rhs = sum(
            solver._surface_rhs(int(emitter), complex(weight))
            for emitter, weight in zip(emitters, weights[:, 0])
            if weight != 0
        )
        surface_state, info = bem.solve(rhs, rtol=solver.rtol)
        if not info.converged:
            raise RuntimeError(f"BEM GMRES did not converge ({info.iterations} iterations)")
        print(f"surface solve: {time.perf_counter() - started:.2f}s, iterations={info.iterations}")

    out = np.empty(len(points), dtype=np.complex128)
    for start in range(0, len(points), point_chunk):
        block_points = points[start:start + point_chunk]
        if bem.n:
            import cupy as cp

            cp_points = cp.asarray(block_points)
            block = bem.scattered_many(cp_points, surface_state[:, None]).astype(cp.complex128)
            block += solver._incident(cp_points, emitters, weights)
            out[start:start + len(block_points)] = cp.asnumpy(block[:, 0])
        else:
            import cupy as cp

            block = solver._incident(cp.asarray(block_points), emitters, weights)
            out[start:start + len(block_points)] = cp.asnumpy(block[:, 0])
    return out


def _sample_level_set(solver: BemSceneSolver, phases: np.ndarray, target: np.ndarray,
                      half_width: float, resolution: int, threshold: float,
                      obstacle_mesh=None, surface_clearance: float = 0.001):
    from skimage.measure import marching_cubes

    lo = np.maximum(target - half_width, np.array([0.0, 0.0, 0.008]))
    hi = np.minimum(target + half_width, np.array([0.18, 0.18, 0.17]))
    axes = [np.linspace(lo[i], hi[i], resolution) for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    points = grid.reshape(-1, 3)
    field = _solve_field_for_phases(solver, phases, points)
    amplitude = np.abs(field).reshape((resolution, resolution, resolution))
    if obstacle_mesh is not None:
        import pyvista as pv

        distance = np.asarray(
            pv.PolyData(points).compute_implicit_distance(obstacle_mesh)["implicit_distance"],
            dtype=np.float64,
        ).reshape(amplitude.shape)
        amplitude[distance < float(surface_clearance)] = 0.0
    peak = float(amplitude.max())
    level = threshold * peak
    if not (amplitude.min() < level < amplitude.max()):
        return axes, amplitude, np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64), peak, level
    spacing = tuple(float(axis[1] - axis[0]) for axis in axes)
    verts, faces, _, _ = marching_cubes(amplitude, level=level, spacing=spacing)
    verts += np.array([axes[0][0], axes[1][0], axes[2][0]])
    return axes, amplitude, verts, faces.astype(np.int64), peak, float(level)


def _cyclic_phase_colors(phases: np.ndarray) -> np.ndarray:
    import matplotlib.pyplot as plt

    return plt.get_cmap("hsv")(np.mod(phases, 2.0 * np.pi) / (2.0 * np.pi))


def _plot_result(output_png: Path, hand_points: np.ndarray, hand_faces: np.ndarray, centers: np.ndarray,
                 phases: np.ndarray, target: np.ndarray, level_vertices: np.ndarray,
                 level_faces: np.ndarray, metrics: dict, scene_name: str = "OK gesture hand") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(14, 7), dpi=160)
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    phase_ax = fig.add_subplot(1, 2, 2)

    hand_mesh = Poly3DCollection(hand_points[hand_faces], facecolor=(0.72, 0.76, 0.80, 0.34),
                                 edgecolor=(0.45, 0.48, 0.52, 0.08), linewidth=0.2)
    ax.add_collection3d(hand_mesh)
    if len(level_faces):
        level_mesh = Poly3DCollection(level_vertices[level_faces], facecolor=(0.0, 0.78, 1.0, 0.28),
                                      edgecolor=(0.0, 0.52, 0.75, 0.15), linewidth=0.2)
        ax.add_collection3d(level_mesh)

    colors = _cyclic_phase_colors(phases)
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c=colors, s=18, depthshade=False)
    ax.scatter([target[0]], [target[1]], [target[2]], c="#ffd8d8", edgecolors="#d71920",
               linewidths=1.4, s=34, depthshade=False)
    ax.set_title(f"{scene_name} + {metrics['level_threshold']:.0%} amplitude level set")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_zlabel("z / m")
    ax.set_xlim(0.0, 0.18)
    ax.set_ylim(0.0, 0.18)
    ax.set_zlim(0.0, 0.17)
    ax.view_init(elev=24, azim=-58)

    phase_ax.scatter(centers[:, 0], centers[:, 1], c=colors, s=58, marker="s")
    phase_ax.scatter([target[0]], [target[1]], c="#ffd8d8", edgecolors="#d71920", linewidths=1.2, s=36)
    phase_ax.set_aspect("equal", adjustable="box")
    phase_ax.set_xlim(0.0, 0.18)
    phase_ax.set_ylim(0.0, 0.18)
    phase_ax.set_title("256-channel BEM focusing phase")
    phase_ax.set_xlabel("x / m")
    phase_ax.set_ylabel("y / m")
    text = (
        f"target = {np.array2string(target, precision=3)}\n"
        f"phase channels = {len(phases)}\n"
        f"peak amplitude = {metrics['peak_amplitude']:.3e}\n"
        f"level = {metrics['level_threshold']:.0%} peak\n"
        f"level triangles = {metrics['level_triangles']}\n"
        f"BEM response time = {metrics['response_s']:.2f}s"
    )
    phase_ax.text(0.01, 0.99, text, transform=phase_ax.transAxes, va="top", ha="left",
                  fontsize=9, family="monospace",
                  bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.85})

    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


def _write_report(path: Path, image_name: str, metrics: dict, scene_name: str = "OK 手势") -> None:
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>{scene_name} BEM 聚焦演示</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; color: #17202a; }}
    img {{ max-width: 100%; border: 1px solid #c9d1d9; }}
    table {{ border-collapse: collapse; margin-top: 16px; }}
    td, th {{ border: 1px solid #d0d7de; padding: 6px 10px; }}
    th {{ background: #f6f8fa; }}
    code {{ background: #f6f8fa; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>{scene_name} BEM 聚焦演示</h1>
  <p>该结果用于验证流程：固定手网格驱动、BEM 单点聚焦相位、局部振幅场 marching-cubes level set。</p>
  <img src="{image_name}" alt="OK gesture BEM visualization">
  <table>
    <tr><th>项目</th><th>数值</th></tr>
    <tr><td>目标点</td><td><code>{metrics['target']}</code></td></tr>
    <tr><td>BEM 面元数</td><td>{metrics['bem_elements']}</td></tr>
    <tr><td>目标响应耗时</td><td>{metrics['response_s']:.2f} s</td></tr>
    <tr><td>峰值振幅</td><td>{metrics['peak_amplitude']:.6e}</td></tr>
    <tr><td>等值面阈值</td><td>{metrics['level_threshold']:.0%} peak</td></tr>
    <tr><td>等值面三角形</td><td>{metrics['level_triangles']}</td></tr>
  </table>
  <p>注意：当前手部驱动是自动估计的过程性 rig，用于跑通数据链路；正式数据集应替换为人工校准的骨骼和蒙皮权重。</p>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def run(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rig = ProceduralHandRig.from_vtk(args.hand_mesh)
    rest = rig.points
    ok_points = rig.deform(HandGesture.ok())
    scene_points = place_points(ok_points, center_m=args.hand_center, rotation_deg=args.hand_rotation,
                                scale=args.hand_scale, min_z=args.min_hand_z)
    mesh_path = rig.write_mesh(output_dir / "ok_hand_scene.vtk", scene_points)
    mesh_center = 0.5 * (scene_points.min(axis=0) + scene_points.max(axis=0))
    np.savez(output_dir / "ok_hand_rig_debug.npz", rest_points=rest, ok_points=ok_points,
             scene_points=scene_points, faces=rig.faces, vertex_region=rig.vertex_region)

    target = np.asarray(args.target, dtype=np.float64)
    raw_cfg = _build_config(mesh_path.resolve(), mesh_center, target, output_dir)
    cfg_path = _write_yaml(raw_cfg, output_dir / "ok_hand_bem_scene.yaml")

    started = time.perf_counter()
    solver = BemSceneSolver(SimulationConfig.from_yaml(str(cfg_path)))
    centers = solver.transducers.centers.copy()
    response = solver.target_response(target[None, :])[0]
    response_s = time.perf_counter() - started
    phases = np.mod(-np.angle(response), 2.0 * np.pi)
    np.savez(output_dir / "ok_hand_bem_response.npz", response=response, phases=phases, target=target)

    axes, amplitude, verts, faces, peak, level = _sample_level_set(
        solver, phases, target, half_width=args.level_half_width,
        resolution=args.level_resolution, threshold=args.level_threshold,
    )
    np.savez(output_dir / "ok_hand_levelset.npz", axes=np.array(axes, dtype=object),
             amplitude=amplitude, vertices=verts, faces=faces, peak=peak, level=level)

    metrics = {
        "target": np.array2string(target, precision=4),
        "bem_elements": int(solver.bem.n),
        "response_s": float(response_s),
        "peak_amplitude": float(peak),
        "level_threshold": float(args.level_threshold),
        "level_value": float(level),
        "level_triangles": int(len(faces)),
        "mesh_file": str(mesh_path),
        "config_file": str(cfg_path),
    }
    png_path = output_dir / "ok_hand_bem_visualization.png"
    _plot_result(png_path, scene_points, rig.faces, centers, phases, target, verts, faces, metrics)
    solver.close()
    _write_report(output_dir / "ok_hand_bem_report.html", png_path.name, metrics)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand-mesh", default=str(ROOT / "obstacle_displacement_study/models/hand_watertight.vtk"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/hand_ok_bem_demo"))
    parser.add_argument("--target", nargs=3, type=float, default=[0.08, 0.08, 0.065])
    parser.add_argument("--hand-center", nargs=3, type=float, default=[0.115, 0.115, 0.075])
    parser.add_argument("--hand-rotation", nargs=3, type=float, default=[0.0, 90.0, 0.0])
    parser.add_argument("--hand-scale", type=float, default=0.42)
    parser.add_argument("--min-hand-z", type=float, default=0.024)
    parser.add_argument("--level-threshold", type=float, default=0.40)
    parser.add_argument("--level-resolution", type=int, default=35)
    parser.add_argument("--level-half-width", type=float, default=0.026)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result, indent=2))
