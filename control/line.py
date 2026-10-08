"""Precompute and play a front/back focal-point line trajectory."""

from __future__ import annotations

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
START_HEIGHT_M = 0.12
END_HEIGHT_M = 0.03
PATH_POINT_COUNT = 16

OPTIMIZATION_ITERATIONS = 120
FORCE_RECOMPUTE = False

PLAYBACK_HZ = 5.0
LOOP_PLAYBACK = True
ENVELOPE_FREQUENCY_HZ = 200.0  # None 表示不加包络
ENVELOPE_DUTY_CYCLE = 0.5


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


def main() -> None:
    path = LinePath(
        x_m=TARGET_X_M,
        y_m=TARGET_Y_M,
        start_height_m=START_HEIGHT_M,
        end_height_m=END_HEIGHT_M,
        point_count=PATH_POINT_COUNT,
    )
    try:
        unique_points, unique_phases = prepare_line_phases(path)
        indices = path.playback_indices()
        play_phase_path(
            unique_points[indices],
            unique_phases[indices],
            service_url=SERVICE_URL,
            playback_hz=PLAYBACK_HZ,
            loop=LOOP_PLAYBACK,
            envelope_frequency_hz=ENVELOPE_FREQUENCY_HZ,
            envelope_duty_cycle=ENVELOPE_DUTY_CYCLE,
            source_name="line",
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"错误：{exc}") from exc


if __name__ == "__main__":
    main()
