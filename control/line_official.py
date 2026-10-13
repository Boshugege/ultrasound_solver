"""Play the vertical line trajectory with official SonicSurface focus phases."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

if __package__:
    from .circle import play_phase_path
    from .line import (
        END_HEIGHT_M,
        ENVELOPE_DUTY_CYCLE,
        ENVELOPE_FREQUENCY_HZ,
        LOOP_PLAYBACK,
        PATH_POINT_COUNT,
        ROUND_TRIP_SECONDS,
        SERVICE_URL,
        START_HEIGHT_M,
        TARGET_X_M,
        TARGET_Y_M,
        LinePath,
        playback_hz_for_round_trip,
    )
    from .sonicsurface.reference import official_focus_phases
else:
    from circle import play_phase_path
    from line import (
        END_HEIGHT_M,
        ENVELOPE_DUTY_CYCLE,
        ENVELOPE_FREQUENCY_HZ,
        LOOP_PLAYBACK,
        PATH_POINT_COUNT,
        ROUND_TRIP_SECONDS,
        SERVICE_URL,
        START_HEIGHT_M,
        TARGET_X_M,
        TARGET_Y_M,
        LinePath,
        playback_hz_for_round_trip,
    )
    from sonicsurface.reference import official_focus_phases


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = (
    REPOSITORY_ROOT
    / "outputs"
    / "line_front_16x16"
    / "line_phases_official.npz"
)


def prepare_official_line_phases(
    path: LinePath,
    output_file: Path = OUTPUT_FILE,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate line phases with the vendor focus implementation."""
    points = path.discretize()
    phases = np.asarray(
        [official_focus_phases(point) for point in points],
        dtype=np.float64,
    )
    if phases.shape != (path.point_count, 256) or not np.isfinite(phases).all():
        raise RuntimeError("官方算法返回了无效的相位数组")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_file,
        target_points_m=points,
        phases_rad=phases,
        playback_indices=path.playback_indices(),
        algorithm_source="official_SonicSurface.phasesForFocus",
        phase_unit="rad",
    )
    print(
        f"[官方算法] 已计算 {points.shape[0]} 个唯一目标点；"
        f"相位已保存：{output_file}"
    )
    return points, phases


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Official SonicSurface vertical focal-point line trajectory"
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
        unique_points, unique_phases = prepare_official_line_phases(path)
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
            f"[官方算法] 每次往返 {indices.size} 个播放点，"
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
            source_name="line-official",
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"错误：{exc}") from exc


if __name__ == "__main__":
    main()
