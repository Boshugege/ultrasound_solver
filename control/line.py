"""Precompute and play a front/back focal-point line trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

if __package__:
    from .circle import optimize_target_points, play_phase_path, implementation_hash
else:
    from circle import optimize_target_points, play_phase_path, implementation_hash


# ======================================================================
# Demo 参数：目标沿阵列中心法线往返运动
# ======================================================================
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = (
    REPOSITORY_ROOT / "src" / "examples" / "pde_homogeneous_16x16.yaml"
)
PHASE_CACHE_FILE = (
    REPOSITORY_ROOT
    / "outputs"
    / "line_front_16x16"
    / "line_phases.npz"
)
SERVICE_URL = "http://127.0.0.1:8765"

TARGET_X_M = 0.08
TARGET_Y_M = 0.08
START_HEIGHT_M = 0.06
END_HEIGHT_M = 0.03
PATH_POINT_COUNT = 16

OPTIMIZATION_ITERATIONS = 120
FORCE_RECOMPUTE = False

ROUND_TRIP_SECONDS = 6.1
LOOP_PLAYBACK = True
ENVELOPE_FREQUENCY_HZ = None  # 运动轨迹直接覆盖相位，不插入关闭帧
ENVELOPE_DUTY_CYCLE = 0.5

BYTES_PER_MODULATED_POINT_CHANGE = 3 * 258
BYTES_PER_STATIC_POINT_CHANGE = 258
SERIAL_BITS_PER_SECOND = 230400.0
BITS_PER_SERIAL_BYTE = 10.0


@dataclass(frozen=True)
class LinePath:
    """Line segment along the array normal with fixed x and y."""

    x_m: float
    y_m: float
    start_height_m: float
    end_height_m: float
    point_count: int

    def discretize(self) -> np.ndarray:
        values = np.asarray(
            [self.x_m, self.y_m, self.start_height_m, self.end_height_m],
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise ValueError("直线路径坐标必须是有限数值")
        if self.point_count < 2:
            raise ValueError("直线路径离散点数至少为 2")
        if self.start_height_m <= 0.0 or self.end_height_m <= 0.0:
            raise ValueError("目标高度必须大于 0")
        heights = np.linspace(
            self.start_height_m,
            self.end_height_m,
            self.point_count,
        )
        return np.column_stack(
            (
                np.full(self.point_count, self.x_m),
                np.full(self.point_count, self.y_m),
                heights,
            )
        )

    def playback_indices(self) -> np.ndarray:
        """Forward and backward indices without pausing twice at endpoints."""
        forward = np.arange(self.point_count, dtype=np.int64)
        backward = np.arange(self.point_count - 2, 0, -1, dtype=np.int64)
        return np.concatenate((forward, backward))


def _cache_metadata(path: LinePath, config_file: Path) -> dict:
    return {
        "schema_version": 2,
        "implementation_sha256": implementation_hash(),
        "config_file": str(config_file),
        "config_sha256": hashlib.sha256(config_file.read_bytes()).hexdigest(),
        "x_m": path.x_m,
        "y_m": path.y_m,
        "start_height_m": path.start_height_m,
        "end_height_m": path.end_height_m,
        "point_count": path.point_count,
        "optimization_iterations": OPTIMIZATION_ITERATIONS,
        "phase_unit": "rad",
    }


def load_line_cache(
    path: LinePath,
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


def prepare_line_phases(
    path: LinePath,
    config_file: Path = CONFIG_FILE,
    cache_file: Path = PHASE_CACHE_FILE,
    force_recompute: bool = FORCE_RECOMPUTE,
) -> tuple[np.ndarray, np.ndarray]:
    if not force_recompute:
        cached = load_line_cache(path, config_file, cache_file)
        if cached is not None:
            print(f"[缓存] 使用已生成的直线路径相位：{cache_file}")
            return cached

    points = path.discretize()
    print(
        f"[预计算] 中心轴 {path.point_count} 个目标点，"
        f"高度 {path.start_height_m:.3f} → {path.end_height_m:.3f} m"
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
    print(f"[预计算] 直线路径相位已保存：{cache_file}")
    return points, phases


def playback_hz_for_round_trip(
    playback_point_count: int,
    round_trip_seconds: float,
    envelope_frequency_hz: float | None = ENVELOPE_FREQUENCY_HZ,
) -> float:
    if not np.isfinite(round_trip_seconds) or round_trip_seconds <= 0.0:
        raise ValueError("单次往返时间必须是正数")
    frequency = playback_point_count / round_trip_seconds
    bytes_per_change = (
        BYTES_PER_STATIC_POINT_CHANGE
        if envelope_frequency_hz is None
        else BYTES_PER_MODULATED_POINT_CHANGE
    )
    theoretical_max_hz = (
        SERIAL_BITS_PER_SECOND
        / (bytes_per_change * BITS_PER_SERIAL_BYTE)
    )
    recommended_max_hz = 70.0 if envelope_frequency_hz is None else 24.0
    if frequency >= theoretical_max_hz:
        minimum = playback_point_count / theoretical_max_hz
        raise ValueError(
            f"单次往返时间过短；当前串口协议理论上至少需要 {minimum:.2f} 秒"
        )
    if frequency > recommended_max_hz:
        print(
            f"[警告] 请求 {frequency:.1f} 点/秒接近串口上限，"
            "实际往返时间可能略长于设定值"
        )
    return frequency


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PDE-optimized vertical focal-point line trajectory"
    )
    parser.add_argument(
        "--point-count",
        type=int,
        default=PATH_POINT_COUNT,
        help="竖直线段上的唯一振幅点数（含两个端点）",
    )
    parser.add_argument(
        "--round-trip-seconds",
        type=float,
        default=ROUND_TRIP_SECONDS,
        help="起点到终点再返回起点所需秒数",
    )
    parser.add_argument(
        "--start-height",
        type=float,
        default=START_HEIGHT_M,
        help="起点高度，单位 m",
    )
    parser.add_argument(
        "--end-height",
        type=float,
        default=END_HEIGHT_M,
        help="终点高度，单位 m",
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
    path = LinePath(
        x_m=TARGET_X_M,
        y_m=TARGET_Y_M,
        start_height_m=args.start_height,
        end_height_m=args.end_height,
        point_count=args.point_count,
    )
    try:
        unique_points, unique_phases = prepare_line_phases(
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
            f"[播放] 每次往返 {indices.size} 个播放点，"
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
            source_name="line-pde",
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"错误：{exc}") from exc


if __name__ == "__main__":
    main()
