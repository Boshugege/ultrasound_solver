"""Precompute and play a circular focal-point trajectory."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

if __package__:
    from .sonicsurface.client import request_json
else:
    from sonicsurface.client import request_json


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = (
    REPOSITORY_ROOT / "src" / "examples" / "pde_homogeneous_16x16.yaml"
)
PHASE_CACHE_FILE = (
    REPOSITORY_ROOT
    / "outputs"
    / "circle_front_16x16"
    / "circle_phases.npz"
)
SERVICE_URL = "http://127.0.0.1:8765"

BOARD_SIDE_M = 0.16
CIRCLE_CENTER_X_M = BOARD_SIDE_M / 2.0
CIRCLE_CENTER_Y_M = BOARD_SIDE_M / 2.0
TARGET_HEIGHT_M = 0.06
CIRCLE_RADIUS_M = BOARD_SIDE_M / 20.0
PATH_POINT_COUNT = 16

OPTIMIZATION_ITERATIONS = 120
FORCE_RECOMPUTE = False

PLAYBACK_HZ = 1.0
LOOP_PLAYBACK = True
ENVELOPE_FREQUENCY_HZ = 200.0  # None 表示不加包络
ENVELOPE_DUTY_CYCLE = 0.5


@dataclass(frozen=True)
class CirclePath:
    """Circle parallel to the x-y transducer plane at a fixed z height."""

    center_x_m: float
    center_y_m: float
    height_m: float
    radius_m: float
    point_count: int

    def discretize(self) -> np.ndarray:
        if self.point_count < 3:
            raise ValueError("圆周离散点数至少为 3")
        if not np.isfinite(self.radius_m) or self.radius_m <= 0.0:
            raise ValueError("圆周半径必须是正数")
        values = np.asarray(
            [self.center_x_m, self.center_y_m, self.height_m],
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise ValueError("圆心和高度必须是有限数值")
        angles = np.linspace(
            0.0,
            2.0 * np.pi,
            self.point_count,
            endpoint=False,
        )
        return np.column_stack(
            (
                self.center_x_m + self.radius_m * np.cos(angles),
                self.center_y_m + self.radius_m * np.sin(angles),
                np.full(self.point_count, self.height_m),
            )
        )


def implementation_hash() -> str:
    """Invalidate phase caches when geometry, physics or algorithm code changes."""
    digest = hashlib.sha256()
    source_root = REPOSITORY_ROOT / "src" / "algorithm"
    paths = sorted(source_root.rglob("*.py")) + sorted(source_root.rglob("*.json"))
    paths += [Path(__file__), REPOSITORY_ROOT / "control/sonicsurface/protocol.py"]
    for path in paths:
        digest.update(str(path.relative_to(REPOSITORY_ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _cache_metadata(path: CirclePath, config_file: Path) -> dict:
    config_hash = hashlib.sha256(config_file.read_bytes()).hexdigest()
    return {
        "schema_version": 2,
        "implementation_sha256": implementation_hash(),
        "config_file": str(config_file),
        "config_sha256": config_hash,
        "center_m": [path.center_x_m, path.center_y_m, path.height_m],
        "radius_m": path.radius_m,
        "point_count": path.point_count,
        "optimization_iterations": OPTIMIZATION_ITERATIONS,
        "phase_unit": "rad",
    }


def load_phase_cache(
    cache_file: Path,
    path: CirclePath,
    config_file: Path,
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


def optimize_target_points(
    points: np.ndarray,
    config_file: Path,
    optimization_iterations: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    src_path = str(REPOSITORY_ROOT / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    from algorithm.config import SimulationConfig
    from algorithm.solvers import HelmholtzDirectSolver
    from algorithm.training.target_response_optimizer import optimize_focal_response

    base_config = SimulationConfig.from_yaml(str(config_file))
    if base_config.specs.array_n != 16:
        raise ValueError("轨迹 demo 要求 16x16 阵列")
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("目标点必须是有限的 (N, 3) 数组")
    point_count = points.shape[0]
    phases = np.empty((point_count, 256), dtype=np.float64)
    best_losses = np.empty(point_count, dtype=np.float64)
    evaluations = np.empty(point_count, dtype=np.int64)

    solver = HelmholtzDirectSolver(base_config)
    if (base_config.algorithm != "adjoint"
            or base_config.algorithm_options.get("optimizer", "adam") != "lbfgsb"
            or base_config.training.loss_type != "focal_pressure"
            or base_config.training.background_weight != 0
            or base_config.training.target_field_file
            or base_config.training.max_seconds != 0):
        solver.close()
        raise ValueError(
            "Control optimization requires adjoint/lbfgsb, focal_pressure, "
            "no target field/background penalty, and no time limit"
        )
    try:
        responses = solver.target_response(points)
        rng = np.random.default_rng(base_config.training.seed)
        previous_phases = None
        for index, (point, response) in enumerate(zip(points, responses, strict=True)):
            iterations = (base_config.training.iterations
                          if optimization_iterations is None else optimization_iterations)
            if previous_phases is not None:
                initial = previous_phases
            elif base_config.training.initial_phase == "random":
                initial = rng.uniform(0, 2*np.pi, 256)
            elif base_config.training.initial_phase == "zero":
                initial = np.zeros(256)
            elif base_config.training.initial_phase == "current":
                initial = solver.transducers.phases
            elif base_config.training.initial_phase == "response":
                initial = -np.angle(response)
            elif base_config.training.initial_phase == "geometric":
                distance = np.linalg.norm(solver.transducers.centers-point, axis=1)
                initial = -base_config.k0*distance
            else:
                raise ValueError("Unsupported initial_phase")
            print(
                f"\n[PDE 预计算] 点 {index + 1:02d}/{point_count}: "
                f"({point[0]:.4f}, {point[1]:.4f}, {point[2]:.4f}) m"
            )
            result = optimize_focal_response(
                response,
                initial,
                int(iterations),
                int(base_config.training.max_evaluations),
            )
            if not result.success and result.amplitude_efficiency < 1-1e-8:
                raise RuntimeError(
                    f"PDE target optimization failed: {result.message}; "
                    f"efficiency={result.amplitude_efficiency:.9f}"
                )
            phases[index] = result.phases
            previous_phases = result.phases
            best_losses[index] = result.loss*base_config.training.target_weight
            evaluations[index] = result.field_evaluations
            print(
                f"[PDE 预计算] loss={result.loss:.6f}, "
                f"振幅上限效率={result.amplitude_efficiency:.9%}, "
                f"场评估={result.field_evaluations}"
            )
    finally:
        solver.close()
    return phases, best_losses, evaluations


def precompute_phases(
    path: CirclePath,
    config_file: Path,
    cache_file: Path,
) -> tuple[np.ndarray, np.ndarray]:
    points = path.discretize()
    print(
        f"[预计算] {path.point_count} 个目标点，圆心="
        f"({path.center_x_m:.3f}, {path.center_y_m:.3f}, {path.height_m:.3f}) m，"
        f"半径={path.radius_m:.3f} m"
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
    print(f"[预计算] 相位路径已保存：{cache_file}")
    return points, phases


def prepare_phase_path(
    path: CirclePath,
    config_file: Path = CONFIG_FILE,
    cache_file: Path = PHASE_CACHE_FILE,
    force_recompute: bool = FORCE_RECOMPUTE,
) -> tuple[np.ndarray, np.ndarray]:
    if not force_recompute:
        cached = load_phase_cache(cache_file, path, config_file)
        if cached is not None:
            print(f"[缓存] 使用已生成的圆周相位：{cache_file}")
            return cached
    return precompute_phases(path, config_file, cache_file)


def play_phase_path(
    target_points_m: np.ndarray,
    phases_rad: np.ndarray,
    service_url: str = SERVICE_URL,
    playback_hz: float = PLAYBACK_HZ,
    loop: bool = LOOP_PLAYBACK,
    envelope_frequency_hz: float | None = ENVELOPE_FREQUENCY_HZ,
    envelope_duty_cycle: float = ENVELOPE_DUTY_CYCLE,
    source_name: str = "circle",
) -> None:
    if not np.isfinite(playback_hz) or playback_hz <= 0.0:
        raise ValueError("PLAYBACK_HZ 必须是正数")
    status = request_json(service_url, "GET", "/v1/status")
    if not status.get("live", False):
        print(f"[播放] 服务端处于 dry-run，未向硬件发送 {source_name} 路径。")
        return

    interval_s = 1.0 / playback_hz
    frame_count = phases_rad.shape[0]
    loop_index = 0
    print(
        f"[播放] {frame_count} 个离散点，{playback_hz:.1f} 帧/秒；"
        "按 Ctrl+C 停止并关闭阵列。"
    )
    if envelope_frequency_hz is None:
        print("[播放] 包络调制：关闭")
    else:
        print(
            f"[播放] 包络调制：{envelope_frequency_hz:.1f} Hz，"
            f"占空比 {envelope_duty_cycle:.0%}"
        )
    try:
        while True:
            for index, (point, phases) in enumerate(
                zip(target_points_m, phases_rad, strict=True)
            ):
                started = time.perf_counter()
                payload = {
                    "phases_rad": phases.tolist(),
                    "source": (
                        f"{source_name}:loop={loop_index}:point={index}:"
                        f"xyz={point.tolist()}"
                    ),
                }
                if envelope_frequency_hz is None:
                    path = "/v1/pattern"
                else:
                    path = "/v1/modulation"
                    payload.update(
                        {
                            "frequency_hz": envelope_frequency_hz,
                            "duty_cycle": envelope_duty_cycle,
                            "duration_s": None,
                        }
                    )
                request_json(service_url, "POST", path, payload)
                print(
                    f"\r当前目标点 {index + 1:02d}/{frame_count}: "
                    f"({point[0]:.4f}, {point[1]:.4f}, {point[2]:.4f}) m",
                    end="",
                    flush=True,
                )
                remaining = interval_s - (time.perf_counter() - started)
                if remaining > 0.0:
                    time.sleep(remaining)
            loop_index += 1
            if not loop:
                print()
                break
    except KeyboardInterrupt:
        print("\n[播放] 正在停止...")
    finally:
        request_json(
            service_url,
            "POST",
            "/v1/off",
            {"source": f"{source_name}:stop"},
        )
        print("[播放] 阵列已关闭。")


def main() -> None:
    path = CirclePath(
        center_x_m=CIRCLE_CENTER_X_M,
        center_y_m=CIRCLE_CENTER_Y_M,
        height_m=TARGET_HEIGHT_M,
        radius_m=CIRCLE_RADIUS_M,
        point_count=PATH_POINT_COUNT,
    )
    try:
        points, phases = prepare_phase_path(path)
        play_phase_path(points, phases)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"错误：{exc}") from exc


if __name__ == "__main__":
    main()
