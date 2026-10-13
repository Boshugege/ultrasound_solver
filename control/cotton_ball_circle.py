"""Hold a center focus, then transport a light ball around a circle."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

if __package__:
    from .circle import implementation_hash, optimize_target_points
    from .sonicsurface.client import request_json
else:
    from circle import implementation_hash, optimize_target_points
    from sonicsurface.client import request_json


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = (
    REPOSITORY_ROOT / "src" / "examples" / "pde_homogeneous_16x16.yaml"
)
PHASE_CACHE_FILE = (
    REPOSITORY_ROOT
    / "outputs"
    / "cotton_ball_circle_16x16"
    / "cotton_ball_circle_phases.npz"
)
SERVICE_URL = "http://127.0.0.1:8765"

CENTER_X_M = 0.08
CENTER_Y_M = 0.08
TARGET_HEIGHT_M = 0.06
CIRCLE_RADIUS_M = 0.01

TRANSFER_POINT_COUNT = 32
TRANSFER_SECONDS = 2.0
CIRCLE_POINT_COUNT = 96
CIRCLE_SECONDS = 8.0
LOOP_CIRCLE = True
LOCATOR_FREQUENCY_HZ = 200.0
LOCATOR_DUTY_CYCLE = 0.5
LOCATOR_SECONDS = 2.0

OPTIMIZATION_ITERATIONS = 120
FORCE_RECOMPUTE = False

# A direct pattern is 258 bytes at 230400 baud, 8N1.
MAX_THEORETICAL_UPDATE_HZ = 230400.0 / (258.0 * 10.0)
RECOMMENDED_MAX_UPDATE_HZ = 70.0


@dataclass(frozen=True)
class CottonBallCircle:
    """Center hold, radial transfer, then a circle parallel to the array."""

    center_x_m: float
    center_y_m: float
    height_m: float
    radius_m: float
    transfer_point_count: int
    circle_point_count: int

    def discretize(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = np.asarray(
            [
                self.center_x_m,
                self.center_y_m,
                self.height_m,
                self.radius_m,
            ],
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise ValueError("搬运轨迹参数必须是有限数值")
        if self.height_m <= 0.0:
            raise ValueError("目标高度必须大于 0")
        if self.radius_m <= 0.0:
            raise ValueError("圆周半径必须大于 0")
        if self.transfer_point_count < 2:
            raise ValueError("中心到圆周的过渡点数至少为 2")
        if self.circle_point_count < 3:
            raise ValueError("圆周点数至少为 3")

        center = np.asarray(
            [self.center_x_m, self.center_y_m, self.height_m],
            dtype=np.float64,
        )
        fractions = np.linspace(0.0, 1.0, self.transfer_point_count)
        transfer = np.column_stack(
            (
                self.center_x_m + self.radius_m * fractions,
                np.full(self.transfer_point_count, self.center_y_m),
                np.full(self.transfer_point_count, self.height_m),
            )
        )
        angles = np.linspace(
            0.0,
            2.0 * np.pi,
            self.circle_point_count,
            endpoint=False,
        )
        circle = np.column_stack(
            (
                self.center_x_m + self.radius_m * np.cos(angles),
                self.center_y_m + self.radius_m * np.sin(angles),
                np.full(self.circle_point_count, self.height_m),
            )
        )
        return center, transfer, circle

    def unique_points(self) -> np.ndarray:
        center, transfer, circle = self.discretize()
        # The transfer endpoints duplicate the center and the first circle point.
        return np.vstack((center[None, :], transfer[1:-1], circle))


def _cache_metadata(path: CottonBallCircle, config_file: Path) -> dict:
    return {
        "schema_version": 1,
        "implementation_sha256": implementation_hash(),
        "config_file": str(config_file),
        "config_sha256": hashlib.sha256(config_file.read_bytes()).hexdigest(),
        "center_m": [path.center_x_m, path.center_y_m, path.height_m],
        "radius_m": path.radius_m,
        "transfer_point_count": path.transfer_point_count,
        "circle_point_count": path.circle_point_count,
        "optimization_iterations": OPTIMIZATION_ITERATIONS,
        "phase_unit": "rad",
        "source_mode": "continuous_direct_pattern",
    }


def _split_phases(
    path: CottonBallCircle,
    phases: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center_phase = phases[0]
    interior_count = path.transfer_point_count - 2
    circle_phases = phases[1 + interior_count:]
    transfer_phases = np.vstack(
        (
            center_phase,
            phases[1:1 + interior_count],
            circle_phases[0],
        )
    )
    return center_phase, transfer_phases, circle_phases


def load_phase_cache(
    path: CottonBallCircle,
    config_file: Path = CONFIG_FILE,
    cache_file: Path = PHASE_CACHE_FILE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if not cache_file.is_file():
        return None
    try:
        with np.load(cache_file, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            points = np.asarray(data["target_points_m"], dtype=np.float64)
            phases = np.asarray(data["phases_rad"], dtype=np.float64)
    except (KeyError, ValueError, OSError, json.JSONDecodeError):
        return None
    expected_points = path.unique_points()
    if metadata != _cache_metadata(path, config_file):
        return None
    if points.shape != expected_points.shape:
        return None
    if phases.shape != (expected_points.shape[0], 256):
        return None
    if not np.isfinite(points).all() or not np.isfinite(phases).all():
        return None
    if not np.allclose(points, expected_points, rtol=0.0, atol=1e-12):
        return None
    return (points, *_split_phases(path, phases))


def prepare_phases(
    path: CottonBallCircle,
    config_file: Path = CONFIG_FILE,
    cache_file: Path = PHASE_CACHE_FILE,
    force_recompute: bool = FORCE_RECOMPUTE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not force_recompute:
        cached = load_phase_cache(path, config_file, cache_file)
        if cached is not None:
            print(f"[缓存] 使用已生成的棉球圆周搬运相位：{cache_file}")
            return cached

    points = path.unique_points()
    print(
        f"[预计算] 中心保持 + {path.transfer_point_count} 个过渡点 + "
        f"{path.circle_point_count} 个圆周点；"
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
    print(f"[预计算] 棉球圆周搬运相位已保存：{cache_file}")
    return (points, *_split_phases(path, phases))


def _validate_direct_rate(point_count: int, duration_s: float, label: str) -> None:
    if not np.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError(f"{label}时间必须是正数")
    intervals = max(point_count - 1, 1)
    update_hz = intervals / duration_s
    if update_hz >= MAX_THEORETICAL_UPDATE_HZ:
        minimum = intervals / MAX_THEORETICAL_UPDATE_HZ
        raise ValueError(
            f"{label}时间过短；直接相位上传理论上至少需要 {minimum:.2f} 秒"
        )
    if update_hz > RECOMMENDED_MAX_UPDATE_HZ:
        print(
            f"[警告] {label}请求 {update_hz:.1f} 次相位更新/秒，"
            "实际运动时间可能略长于设定值"
        )


def _send_direct_pattern(
    phases: np.ndarray,
    point: np.ndarray,
    source: str,
) -> None:
    result = request_json(
        SERVICE_URL,
        "POST",
        "/v1/pattern",
        {
            "phases_rad": phases.tolist(),
            "source": f"{source}:xyz={point.tolist()}",
        },
    )
    if not result.get("transmitted", False):
        raise RuntimeError("控制服务接受了相位，但没有写入串口")


def _play_timed_direct(
    points: np.ndarray,
    phases: np.ndarray,
    duration_s: float,
    source: str,
    skip_first: bool = False,
) -> None:
    if points.shape[0] != phases.shape[0]:
        raise ValueError("目标点与相位帧数量不一致")
    _validate_direct_rate(points.shape[0], duration_s, source)
    interval_s = duration_s / max(points.shape[0] - 1, 1)
    started = time.perf_counter()
    first_index = 1 if skip_first else 0
    for index in range(first_index, points.shape[0]):
        deadline = started + index * interval_s
        remaining = deadline - time.perf_counter()
        if remaining > 0.0:
            time.sleep(remaining)
        _send_direct_pattern(
            phases[index],
            points[index],
            f"{source}:point={index}",
        )
        print(
            f"\r[{source}] {index + 1:03d}/{points.shape[0]} "
            f"({points[index, 0]:.4f}, {points[index, 1]:.4f}, "
            f"{points[index, 2]:.4f}) m",
            end="",
            flush=True,
        )
    print()


def _run_tactile_locator(
    center_phase: np.ndarray,
    center: np.ndarray,
) -> None:
    print(
        f"[触觉定位] {LOCATOR_FREQUENCY_HZ:.0f} Hz 包络，"
        f"持续 {LOCATOR_SECONDS:.1f} 秒；此模式不能承托颗粒。"
    )
    result = request_json(
        SERVICE_URL,
        "POST",
        "/v1/modulation",
        {
            "phases_rad": center_phase.tolist(),
            "frequency_hz": LOCATOR_FREQUENCY_HZ,
            "duty_cycle": LOCATOR_DUTY_CYCLE,
            "duration_s": LOCATOR_SECONDS,
            "source": "cotton-ball:tactile-locator",
        },
    )
    if not result.get("transmitted", False):
        raise RuntimeError("触觉定位相位没有写入串口")
    time.sleep(LOCATOR_SECONDS + 0.1)
    _send_direct_pattern(
        center_phase,
        center,
        "cotton-ball:center-hold-after-locator",
    )
    print("[触觉定位] 已恢复连续中心场。")


def _wait_for_start(
    center_phase: np.ndarray,
    center: np.ndarray,
) -> None:
    print(
        "[中心保持] 按 L 临时触觉定位；放置颗粒后按数字键 0 开始搬运；"
        "Ctrl+C 结束。"
    )
    try:
        import msvcrt
    except ImportError:
        while True:
            key = input().strip().lower()
            if key == "0":
                return
            if key == "l":
                _run_tactile_locator(center_phase, center)
                continue
            print("请输入 L 并回车定位，或输入 0 并回车开始。")
        return

    while True:
        key = msvcrt.getwch().lower()
        if key == "0":
            print("0")
            return
        if key == "l":
            print("L")
            _run_tactile_locator(center_phase, center)
            print(
                "[中心保持] 放置颗粒后按数字键 0 开始搬运；"
                "可再次按 L 定位。"
            )
            continue
        if key == "\x03":
            raise KeyboardInterrupt
        if key in {"\x00", "\xe0"}:
            msvcrt.getwch()


def run_transport(
    path: CottonBallCircle,
    transfer_seconds: float,
    circle_seconds: float,
    loop_circle: bool,
    config_file: Path,
    force_recompute: bool,
) -> None:
    _, center_phase, transfer_phases, circle_phases = prepare_phases(
        path,
        config_file=config_file,
        force_recompute=force_recompute,
    )
    center, transfer_points, circle_points = path.discretize()
    _validate_direct_rate(
        transfer_points.shape[0],
        transfer_seconds,
        "中心到圆周过渡",
    )
    # A circle is periodic, so it has one interval per point.
    _validate_direct_rate(
        circle_points.shape[0] + 1,
        circle_seconds,
        "圆周运动",
    )

    status = request_json(SERVICE_URL, "GET", "/v1/status")
    if not status.get("live", False):
        raise RuntimeError("服务端不是 live 模式，未连接真实硬件")

    _send_direct_pattern(center_phase, center, "cotton-ball:center-hold")
    print(
        f"[中心保持] 焦点=({center[0]:.3f}, {center[1]:.3f}, "
        f"{center[2]:.3f}) m，连续 40 kHz 发射"
    )
    try:
        _wait_for_start(center_phase, center)
        print(
            f"[搬运] 用 {transfer_seconds:.2f} 秒从圆心平滑移动到圆周；"
            "相位直接覆盖，不插入关闭帧。"
        )
        _play_timed_direct(
            transfer_points,
            transfer_phases,
            transfer_seconds,
            "cotton-ball:transfer",
            skip_first=True,
        )

        loop_index = 0
        while True:
            print(
                f"[圆周] 第 {loop_index + 1} 圈，"
                f"{circle_seconds:.2f} 秒/圈"
            )
            # The transfer ends at circle point 0. Advance through points
            # 1..N-1 and return to point 0 exactly at circle_seconds.
            interval_s = circle_seconds / circle_points.shape[0]
            started = time.perf_counter()
            for step in range(1, circle_points.shape[0] + 1):
                index = step % circle_points.shape[0]
                point = circle_points[index]
                phases = circle_phases[index]
                deadline = started + step * interval_s
                remaining = deadline - time.perf_counter()
                if remaining > 0.0:
                    time.sleep(remaining)
                _send_direct_pattern(
                    phases,
                    point,
                    f"cotton-ball:circle={loop_index}:point={index}",
                )
                print(
                    f"\r[圆周] {step:03d}/{circle_points.shape[0]} "
                    f"({point[0]:.4f}, {point[1]:.4f}, "
                    f"{point[2]:.4f}) m",
                    end="",
                    flush=True,
                )
            print()
            loop_index += 1
            if not loop_circle:
                break
    except KeyboardInterrupt:
        print("\n[搬运] 收到停止请求。")
    finally:
        request_json(
            SERVICE_URL,
            "POST",
            "/v1/off",
            {"source": "cotton-ball:stop"},
        )
        print("[搬运] 阵列已关闭。")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hold and transport a light cotton ball around a circle"
    )
    parser.add_argument("--height", type=float, default=TARGET_HEIGHT_M)
    parser.add_argument("--radius", type=float, default=CIRCLE_RADIUS_M)
    parser.add_argument(
        "--transfer-points",
        type=int,
        default=TRANSFER_POINT_COUNT,
    )
    parser.add_argument(
        "--transfer-seconds",
        type=float,
        default=TRANSFER_SECONDS,
    )
    parser.add_argument(
        "--circle-points",
        type=int,
        default=CIRCLE_POINT_COUNT,
    )
    parser.add_argument(
        "--circle-seconds",
        type=float,
        default=CIRCLE_SECONDS,
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="只运行一圈，默认持续循环",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
    )
    parser.add_argument("--config", type=Path, default=CONFIG_FILE)
    return parser


def main() -> None:
    args = _parser().parse_args()
    path = CottonBallCircle(
        center_x_m=CENTER_X_M,
        center_y_m=CENTER_Y_M,
        height_m=args.height,
        radius_m=args.radius,
        transfer_point_count=args.transfer_points,
        circle_point_count=args.circle_points,
    )
    try:
        run_transport(
            path,
            transfer_seconds=args.transfer_seconds,
            circle_seconds=args.circle_seconds,
            loop_circle=LOOP_CIRCLE and not args.once,
            config_file=args.config,
            force_recompute=args.force_recompute,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"错误：{exc}") from exc


if __name__ == "__main__":
    main()
