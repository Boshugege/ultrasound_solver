"""Play the cube-edge trajectory with the official SonicSurface focus phases."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

if __package__:
    from .cube import (
        CUBE_CENTER_X_M,
        CUBE_CENTER_Y_M,
        CUBE_CENTER_Z_M,
        CUBE_EDGE_LENGTH_M,
        ENVELOPE_DUTY_CYCLE,
        ENVELOPE_FREQUENCY_HZ,
        LOOP_PLAYBACK,
        POINTS_PER_EDGE,
        SERVICE_URL,
        TRAVERSAL_SECONDS,
        CubeEdgePath,
        _playback_hz,
    )
    from .circle import play_phase_path
    from .sonicsurface.reference import official_focus_phases
else:
    from cube import (
        CUBE_CENTER_X_M,
        CUBE_CENTER_Y_M,
        CUBE_CENTER_Z_M,
        CUBE_EDGE_LENGTH_M,
        ENVELOPE_DUTY_CYCLE,
        ENVELOPE_FREQUENCY_HZ,
        LOOP_PLAYBACK,
        POINTS_PER_EDGE,
        SERVICE_URL,
        TRAVERSAL_SECONDS,
        CubeEdgePath,
        _playback_hz,
    )
    from circle import play_phase_path
    from sonicsurface.reference import official_focus_phases


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = (
    REPOSITORY_ROOT
    / "outputs"
    / "cube_edges_16x16"
    / "cube_edge_phases_official.npz"
)


def prepare_official_phases(
    path: CubeEdgePath,
    output_file: Path = OUTPUT_FILE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate every unique point with the vendor focus implementation."""
    points, playback_indices = path.discretize()
    phases = np.asarray(
        [official_focus_phases(point) for point in points],
        dtype=np.float64,
    )
    if phases.shape != (points.shape[0], 256) or not np.isfinite(phases).all():
        raise RuntimeError("官方算法返回了无效的相位数组")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_file,
        target_points_m=points,
        phases_rad=phases,
        playback_indices=playback_indices,
        algorithm_source="official_SonicSurface.phasesForFocus",
        phase_unit="rad",
    )
    print(
        f"[官方算法] 已计算 {points.shape[0]} 个唯一目标点；"
        f"相位已保存：{output_file}"
    )
    return points, phases, playback_indices


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Official SonicSurface focus moving along all twelve cube edges"
        )
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
        unique_points, unique_phases, playback_indices = (
            prepare_official_phases(path)
        )
        playback_hz = _playback_hz(
            playback_indices.size,
            args.traversal_seconds,
        )
        print(
            f"[官方算法] 每次遍历 {playback_indices.size} 个播放点，"
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
            source_name="cube-edges-official",
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"错误：{exc}") from exc


if __name__ == "__main__":
    main()
