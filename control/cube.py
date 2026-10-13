"""Precompute and play a focal point along all twelve edges of a cube."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

if __package__:
    from .circle import (
        implementation_hash as solver_implementation_hash,
        optimize_target_points,
        play_phase_path,
    )
else:
    from circle import (
        implementation_hash as solver_implementation_hash,
        optimize_target_points,
        play_phase_path,
    )


# ======================================================================
# Demo parameters: a cube above the center of the transducer plane
# ======================================================================
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = (
    REPOSITORY_ROOT / "src" / "examples" / "pde_homogeneous_16x16.yaml"
)
PHASE_CACHE_FILE = (
    REPOSITORY_ROOT
    / "outputs"
    / "cube_edges_16x16"
    / "cube_edge_phases.npz"
)
SERVICE_URL = "http://127.0.0.1:8765"

CUBE_CENTER_X_M = 0.08
CUBE_CENTER_Y_M = 0.08
CUBE_CENTER_Z_M = 0.075
CUBE_EDGE_LENGTH_M = 0.02

# Includes both endpoints. Shared corners are solved only once.
POINTS_PER_EDGE = 6
TRAVERSAL_SECONDS = 2

OPTIMIZATION_ITERATIONS = 120
FORCE_RECOMPUTE = False

LOOP_PLAYBACK = True
ENVELOPE_FREQUENCY_HZ = None  # Direct phase replacement for smooth motion.
ENVELOPE_DUTY_CYCLE = 0.5

# Direct trajectory playback uploads one complete 258-byte phase frame.
BYTES_PER_POINT_CHANGE = 258
MAX_THEORETICAL_POINT_HZ = (
    230400.0 / (BYTES_PER_POINT_CHANGE * 10.0)
)
RECOMMENDED_MAX_POINT_HZ = 70.0


Vertex = tuple[int, int, int]
Edge = tuple[Vertex, Vertex]

# All 12 unique edges, arranged into the minimum four edge-disjoint trails.
# A cube has eight odd-degree vertices, so three jumps between trails are
# unavoidable if every edge must be traversed exactly once.
CUBE_EDGES: tuple[Edge, ...] = (
    ((0, 0, 0), (1, 0, 0)),
    ((1, 0, 0), (1, 1, 0)),
    ((1, 1, 0), (0, 1, 0)),
    ((0, 1, 0), (0, 0, 0)),
    ((0, 0, 0), (0, 0, 1)),
    ((0, 0, 1), (1, 0, 1)),
    ((1, 0, 1), (1, 0, 0)),
    ((0, 0, 1), (0, 1, 1)),
    ((0, 1, 1), (0, 1, 0)),
    ((0, 1, 1), (1, 1, 1)),
    ((1, 1, 1), (1, 0, 1)),
    ((1, 1, 0), (1, 1, 1)),
)


@dataclass(frozen=True)
class CubeEdgePath:
    """Twelve edges of an axis-aligned cube."""

    center_x_m: float
    center_y_m: float
    center_z_m: float
    edge_length_m: float
    points_per_edge: int

    def _vertices(self) -> dict[Vertex, np.ndarray]:
        center = np.asarray(
            [self.center_x_m, self.center_y_m, self.center_z_m],
            dtype=np.float64,
        )
        if not np.isfinite(center).all():
            raise ValueError("立方体中心坐标必须是有限数值")
        if not np.isfinite(self.edge_length_m) or self.edge_length_m <= 0.0:
            raise ValueError("立方体边长必须是正数")
        if self.points_per_edge < 2:
            raise ValueError("每条棱的离散点数至少为 2")
        half = self.edge_length_m / 2.0
        if self.center_z_m - half <= 0.0:
            raise ValueError("立方体最低点必须位于阵列平面上方")
        return {
            key: center + half * (2.0 * np.asarray(key, dtype=np.float64) - 1.0)
            for key in {
                vertex
                for edge in CUBE_EDGES
                for vertex in edge
            }
        }

    def discretize(self) -> tuple[np.ndarray, np.ndarray]:
        """Return unique solve points and their ordered playback indices."""
        vertices = self._vertices()
        fractions = np.linspace(0.0, 1.0, self.points_per_edge)
        unique_points: list[np.ndarray] = []
        point_indices: dict[tuple[float, float, float], int] = {}
        playback_indices: list[int] = []
        previous_end: Vertex | None = None

        for start_key, end_key in CUBE_EDGES:
            start = vertices[start_key]
            end = vertices[end_key]
            edge_points = (
                start[None, :] * (1.0 - fractions[:, None])
                + end[None, :] * fractions[:, None]
            )
            first = 1 if previous_end == start_key else 0
            for point in edge_points[first:]:
                key = tuple(float(value) for value in point)
                index = point_indices.get(key)
                if index is None:
                    index = len(unique_points)
                    point_indices[key] = index
                    unique_points.append(point)
                playback_indices.append(index)
            previous_end = end_key

        return (
            np.asarray(unique_points, dtype=np.float64),
            np.asarray(playback_indices, dtype=np.int64),
        )


def implementation_hash() -> str:
    digest = hashlib.sha256()
    digest.update(solver_implementation_hash().encode("ascii"))
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()


def _cache_metadata(path: CubeEdgePath, config_file: Path) -> dict:
    return {
        "schema_version": 1,
        "implementation_sha256": implementation_hash(),
        "config_file": str(config_file),
        "config_sha256": hashlib.sha256(config_file.read_bytes()).hexdigest(),
        "center_m": [
            path.center_x_m,
            path.center_y_m,
            path.center_z_m,
        ],
        "edge_length_m": path.edge_length_m,
        "points_per_edge": path.points_per_edge,
        "optimization_iterations": OPTIMIZATION_ITERATIONS,
        "phase_unit": "rad",
    }


def load_cube_cache(
    path: CubeEdgePath,
    config_file: Path = CONFIG_FILE,
    cache_file: Path = PHASE_CACHE_FILE,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not cache_file.is_file():
        return None
    expected = _cache_metadata(path, config_file)
    try:
        with np.load(cache_file, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            points = np.asarray(data["target_points_m"], dtype=np.float64)
            phases = np.asarray(data["phases_rad"], dtype=np.float64)
    except (KeyError, ValueError, OSError, json.JSONDecodeError):
        return None
    expected_points, _ = path.discretize()
    if metadata != expected:
        return None
    if points.shape != expected_points.shape or phases.shape != (
        expected_points.shape[0],
        256,
    ):
        return None
    if not np.isfinite(points).all() or not np.isfinite(phases).all():
        return None
    if not np.allclose(points, expected_points, rtol=0.0, atol=1e-12):
        return None
    return points, phases


def prepare_cube_phases(
    path: CubeEdgePath,
    config_file: Path = CONFIG_FILE,
    cache_file: Path = PHASE_CACHE_FILE,
    force_recompute: bool = FORCE_RECOMPUTE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, playback_indices = path.discretize()
    if not force_recompute:
        cached = load_cube_cache(path, config_file, cache_file)
        if cached is not None:
            print(f"[缓存] 使用已生成的立方体棱边相位：{cache_file}")
            return cached[0], cached[1], playback_indices

    points, playback_indices = path.discretize()
    print(
        f"[预计算] 立方体 12 条棱，每条棱 {path.points_per_edge} 点，"
        f"共 {points.shape[0]} 个唯一目标点"
    )
    phases, best_losses, evaluations = optimize_target_points(
        points,
        config_file,
        OPTIMIZATION_ITERATIONS,
    )
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_file,
        target_points_m=points,
        phases_rad=phases,
        best_losses=best_losses,
        field_evaluations=evaluations,
        metadata=json.dumps(
            _cache_metadata(path, config_file),
            ensure_ascii=False,
            sort_keys=True,
        ),
    )
    print(f"[预计算] 立方体棱边相位已保存：{cache_file}")
    return points, phases, playback_indices


def _playback_hz(playback_point_count: int, traversal_seconds: float) -> float:
    if not np.isfinite(traversal_seconds) or traversal_seconds <= 0.0:
        raise ValueError("单次遍历时间必须是正数")
    frequency = playback_point_count / traversal_seconds
    if frequency >= MAX_THEORETICAL_POINT_HZ:
        minimum = playback_point_count / MAX_THEORETICAL_POINT_HZ
        raise ValueError(
            f"单次遍历时间过短；当前串口协议理论上至少需要 {minimum:.2f} 秒"
        )
    if frequency > RECOMMENDED_MAX_POINT_HZ:
        print(
            f"[警告] 请求 {frequency:.1f} 点/秒接近串口上限，"
            "实际遍历时间可能略长于设定值"
        )
    return frequency


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PDE-optimized focus moving along all twelve cube edges"
    )
    parser.add_argument(
        "--points-per-edge",
        type=int,
        default=POINTS_PER_EDGE,
        help="每条棱包含的振幅点数（含两个端点）",
    )
    parser.add_argument(
        "--traversal-seconds",
        type=float,
        default=TRAVERSAL_SECONDS,
        help="完成一次十二棱遍历所需秒数",
    )
    parser.add_argument(
        "--edge-length",
        type=float,
        default=CUBE_EDGE_LENGTH_M,
        help="立方体边长，单位 m",
    )
    parser.add_argument(
        "--center-height",
        type=float,
        default=CUBE_CENTER_Z_M,
        help="立方体中心高度，单位 m",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="只遍历一次，默认循环播放",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="忽略已有相位缓存并重新计算",
    )
    parser.add_argument("--config", type=Path, default=CONFIG_FILE)
    return parser


def main() -> None:
    args = _parser().parse_args()
    path = CubeEdgePath(
        center_x_m=CUBE_CENTER_X_M,
        center_y_m=CUBE_CENTER_Y_M,
        center_z_m=args.center_height,
        edge_length_m=args.edge_length,
        points_per_edge=args.points_per_edge,
    )
    try:
        unique_points, unique_phases, playback_indices = prepare_cube_phases(
            path,
            config_file=args.config,
            force_recompute=args.force_recompute,
        )
        playback_hz = _playback_hz(
            playback_indices.size,
            args.traversal_seconds,
        )
        print(
            f"[播放] 每次遍历 {playback_indices.size} 个播放点，"
            f"目标时间 {args.traversal_seconds:.3f} 秒，"
            f"更新率 {playback_hz:.2f} 点/秒"
        )
        play_phase_path(
            unique_points[playback_indices],
            unique_phases[playback_indices],
            service_url=SERVICE_URL,
            playback_hz=playback_hz,
            loop=LOOP_PLAYBACK and not args.once,
            envelope_frequency_hz=ENVELOPE_FREQUENCY_HZ,
            envelope_duty_cycle=ENVELOPE_DUTY_CYCLE,
            source_name="cube-edges",
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"错误：{exc}") from exc


if __name__ == "__main__":
    main()
