"""Precompute and play a horizontal focal-point line trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

if __package__:
    from .circle import (
        implementation_hash,
        optimize_target_points,
        play_phase_path,
    )
    from .line import playback_hz_for_round_trip
else:
    from circle import (
        implementation_hash,
        optimize_target_points,
        play_phase_path,
    )
    from line import playback_hz_for_round_trip


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = (
    REPOSITORY_ROOT / "src" / "examples" / "pde_homogeneous_16x16.yaml"
)
PHASE_CACHE_FILE = (
    REPOSITORY_ROOT
    / "outputs"
    / "horizontal_line_16x16"
    / "horizontal_line_phases.npz"
)
SERVICE_URL = "http://127.0.0.1:8765"

START_X_M = 0.05
END_X_M = 0.11
TARGET_Y_M = 0.08
TARGET_HEIGHT_M = 0.06
PATH_POINT_COUNT = 24

OPTIMIZATION_ITERATIONS = 120
FORCE_RECOMPUTE = False

ROUND_TRIP_SECONDS = 3.0
LOOP_PLAYBACK = True
ENVELOPE_FREQUENCY_HZ = None  # Direct phase replacement for smooth motion.
ENVELOPE_DUTY_CYCLE = 0.5


@dataclass(frozen=True)
class HorizontalLinePath:
    """Line segment parallel to the array x axis."""

    start_x_m: float
    end_x_m: float
    y_m: float
    height_m: float
    point_count: int

    def discretize(self) -> np.ndarray:
        values = np.asarray(
            [self.start_x_m, self.end_x_m, self.y_m, self.height_m],
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise ValueError("横向路径坐标必须是有限数值")
        if self.point_count < 2:
            raise ValueError("横向路径离散点数至少为 2")
        if self.height_m <= 0.0:
            raise ValueError("目标高度必须大于 0")
        if self.start_x_m == self.end_x_m:
            raise ValueError("横向路径的起点和终点不能相同")
        x_values = np.linspace(
            self.start_x_m,
            self.end_x_m,
            self.point_count,
        )
        return np.column_stack(
            (
                x_values,
                np.full(self.point_count, self.y_m),
                np.full(self.point_count, self.height_m),
            )
        )

    def playback_indices(self) -> np.ndarray:
        """Forward and backward indices without duplicate endpoint pauses."""
        forward = np.arange(self.point_count, dtype=np.int64)
        backward = np.arange(self.point_count - 2, 0, -1, dtype=np.int64)
        return np.concatenate((forward, backward))


def _cache_metadata(path: HorizontalLinePath, config_file: Path) -> dict:
    return {
        "schema_version": 1,
        "implementation_sha256": implementation_hash(),
        "config_file": str(config_file),
        "config_sha256": hashlib.sha256(config_file.read_bytes()).hexdigest(),
        "start_x_m": path.start_x_m,
        "end_x_m": path.end_x_m,
        "y_m": path.y_m,
        "height_m": path.height_m,
        "point_count": path.point_count,
        "optimization_iterations": OPTIMIZATION_ITERATIONS,
        "phase_unit": "rad",
    }


def load_horizontal_line_cache(
    path: HorizontalLinePath,
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
    if metadata != expected:
        return None
    if points.shape != (path.point_count, 3):
        return None
    if phases.shape != (path.point_count, 256):
        return None
    if not np.isfinite(points).all() or not np.isfinite(phases).all():
        return None
    return points, phases


def prepare_horizontal_line_phases(
    path: HorizontalLinePath,
    config_file: Path = CONFIG_FILE,
    cache_file: Path = PHASE_CACHE_FILE,
    force_recompute: bool = FORCE_RECOMPUTE,
) -> tuple[np.ndarray, np.ndarray]:
    if not force_recompute:
        cached = load_horizontal_line_cache(path, config_file, cache_file)
        if cached is not None:
            print(f"[缓存] 使用已生成的横向直线相位：{cache_file}")
            return cached

    points = path.discretize()
    print(
        f"[预计算] 横向 {path.point_count} 个目标点，"
        f"x={path.start_x_m:.3f} → {path.end_x_m:.3f} m，"
        f"y={path.y_m:.3f} m，z={path.height_m:.3f} m"
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
    print(f"[预计算] 横向直线相位已保存：{cache_file}")
    return points, phases


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PDE-optimized horizontal focal-point line trajectory"
    )
    parser.add_argument(
        "--point-count",
        type=int,
        default=PATH_POINT_COUNT,
        help="横向线段上的唯一振幅点数（含两个端点）",
    )
    parser.add_argument(
        "--round-trip-seconds",
        type=float,
        default=ROUND_TRIP_SECONDS,
        help="起点到终点再返回起点所需秒数",
    )
    parser.add_argument("--start-x", type=float, default=START_X_M)
    parser.add_argument("--end-x", type=float, default=END_X_M)
    parser.add_argument(
        "--height",
        type=float,
        default=TARGET_HEIGHT_M,
        help="横向线段高度，单位 m",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="只往返一次，默认循环播放",
    )
    envelope = parser.add_mutually_exclusive_group()
    envelope.add_argument(
        "--envelope-frequency",
        type=float,
        default=ENVELOPE_FREQUENCY_HZ,
        help="触觉包络频率，单位 Hz",
    )
    envelope.add_argument(
        "--no-envelope",
        action="store_true",
        help="关闭包络，只连续切换聚焦相位",
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
    path = HorizontalLinePath(
        start_x_m=args.start_x,
        end_x_m=args.end_x,
        y_m=TARGET_Y_M,
        height_m=args.height,
        point_count=args.point_count,
    )
    try:
        unique_points, unique_phases = prepare_horizontal_line_phases(
            path,
            config_file=args.config,
            force_recompute=args.force_recompute,
        )
        indices = path.playback_indices()
        envelope_frequency_hz = (
            None if args.no_envelope else args.envelope_frequency
        )
        playback_hz = playback_hz_for_round_trip(
            indices.size,
            args.round_trip_seconds,
            envelope_frequency_hz,
        )
        print(
            f"[播放] 每次横向往返 {indices.size} 个播放点，"
            f"目标时间 {args.round_trip_seconds:.3f} 秒，"
            f"更新率 {playback_hz:.2f} 点/秒"
        )
        play_phase_path(
            unique_points[indices],
            unique_phases[indices],
            service_url=SERVICE_URL,
            playback_hz=playback_hz,
            loop=LOOP_PLAYBACK and not args.once,
            envelope_frequency_hz=envelope_frequency_hz,
            envelope_duty_cycle=ENVELOPE_DUTY_CYCLE,
            source_name="horizontal-line-pde",
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"错误：{exc}") from exc


if __name__ == "__main__":
    main()
