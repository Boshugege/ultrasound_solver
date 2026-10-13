"""Optimize and transmit a fixed acoustic vortex with the corrected PDE."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

if __package__:
    from .circle import implementation_hash as solver_implementation_hash
    from .sonicsurface.client import request_json
    from .sonicsurface.protocol import PHASE_DIVISIONS, quantize_phases
else:
    from circle import implementation_hash as solver_implementation_hash
    from sonicsurface.client import request_json
    from sonicsurface.protocol import PHASE_DIVISIONS, quantize_phases


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = (
    REPOSITORY_ROOT / "src" / "examples" / "pde_homogeneous_16x16.yaml"
)
PHASE_CACHE_FILE = (
    REPOSITORY_ROOT
    / "outputs"
    / "vortex_fixed_16x16"
    / "vortex_phases.npz"
)
SERVICE_URL = "http://127.0.0.1:8765"

TARGET_X_M = 0.08
TARGET_Y_M = 0.08
TARGET_HEIGHT_M = 0.06
VORTEX_RING_RADIUS_M = 0.005
RING_SAMPLE_COUNT = 32
TOPOLOGICAL_CHARGE = 1
CENTER_SUPPRESSION_WEIGHT = 2.0

OPTIMIZATION_ITERATIONS = 300
MAX_EVALUATIONS = 1000
FORCE_RECOMPUTE = False
HOLD_SECONDS = None

MAX_CENTER_TO_RING_RATIO = 0.08
MIN_MODE_PURITY = 0.97
LOCATOR_FREQUENCY_HZ = 200.0
LOCATOR_DUTY_CYCLE = 0.5
LOCATOR_SECONDS = 2.0


@dataclass(frozen=True)
class VortexMetrics:
    center_to_ring_ratio: float
    mode_purity: float
    ring_amplitude_cv: float
    mean_ring_amplitude: float


def _target() -> np.ndarray:
    return np.asarray(
        [TARGET_X_M, TARGET_Y_M, TARGET_HEIGHT_M],
        dtype=np.float64,
    )


def _sample_points(
    target: np.ndarray,
    radius_m: float,
    sample_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(radius_m) or radius_m <= 0.0:
        raise ValueError("涡旋环半径必须是正数")
    if sample_count < 8:
        raise ValueError("涡旋环采样点数至少为 8")
    angles = np.linspace(0.0, 2.0 * np.pi, sample_count, endpoint=False)
    ring = target + np.column_stack(
        (
            radius_m * np.cos(angles),
            radius_m * np.sin(angles),
            np.zeros(sample_count),
        )
    )
    return np.vstack((target, ring)), angles


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    digest.update(solver_implementation_hash().encode("ascii"))
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()


def _cache_metadata(config_file: Path) -> dict:
    return {
        "schema_version": 1,
        "implementation_sha256": _implementation_hash(),
        "config_file": str(config_file),
        "config_sha256": hashlib.sha256(config_file.read_bytes()).hexdigest(),
        "target_m": _target().tolist(),
        "ring_radius_m": VORTEX_RING_RADIUS_M,
        "ring_sample_count": RING_SAMPLE_COUNT,
        "topological_charge": TOPOLOGICAL_CHARGE,
        "center_suppression_weight": CENTER_SUPPRESSION_WEIGHT,
        "optimization_iterations": OPTIMIZATION_ITERATIONS,
        "max_evaluations": MAX_EVALUATIONS,
        "phase_unit": "rad",
        "algorithm_source": "corrected_pde_vortex_mode",
    }


def vortex_loss_gradient(
    phases: np.ndarray,
    modal_response: np.ndarray,
    center_response: np.ndarray,
    center_weight: float,
) -> tuple[float, np.ndarray]:
    """Maximize a vortex mode while suppressing pressure at its center."""
    values = np.asarray(phases, dtype=np.float64)
    modal = np.asarray(modal_response, dtype=np.complex128)
    center = np.asarray(center_response, dtype=np.complex128)
    weights = np.exp(1j * values)
    modal_pressure = modal @ weights
    center_pressure = center @ weights
    modal_upper = max(float(np.sum(np.abs(modal))), np.finfo(float).tiny)
    center_upper = max(float(np.sum(np.abs(center))), np.finfo(float).tiny)

    loss = (
        -(abs(modal_pressure) / modal_upper) ** 2
        + center_weight * (abs(center_pressure) / center_upper) ** 2
    )
    gradient = (
        -2.0
        * np.real(
            np.conj(modal_pressure)
            * 1j
            * modal
            * weights
        )
        / modal_upper**2
        + 2.0
        * center_weight
        * np.real(
            np.conj(center_pressure)
            * 1j
            * center
            * weights
        )
        / center_upper**2
    )
    return float(loss), gradient


def _metrics(
    response: np.ndarray,
    phases: np.ndarray,
    desired_mode: np.ndarray,
) -> VortexMetrics:
    field = response @ np.exp(1j * phases)
    ring_field = field[1:]
    amplitudes = np.abs(ring_field)
    mean_amplitude = float(np.mean(amplitudes))
    purity = abs(np.vdot(desired_mode, ring_field)) / (
        np.linalg.norm(desired_mode) * np.linalg.norm(ring_field)
    )
    return VortexMetrics(
        center_to_ring_ratio=float(abs(field[0]) / mean_amplitude),
        mode_purity=float(purity),
        ring_amplitude_cv=float(np.std(amplitudes) / mean_amplitude),
        mean_ring_amplitude=mean_amplitude,
    )


def _quantized_phases(phases: np.ndarray) -> np.ndarray:
    bins = quantize_phases(phases)
    return bins.astype(np.float64) * (2.0 * np.pi / PHASE_DIVISIONS)


def _print_metrics(label: str, metrics: VortexMetrics) -> None:
    print(
        f"[{label}] 中心/环均值={metrics.center_to_ring_ratio:.3%}，"
        f"模式纯度={metrics.mode_purity:.3%}，"
        f"环振幅变异系数={metrics.ring_amplitude_cv:.3%}，"
        f"环平均振幅={metrics.mean_ring_amplitude:.6f} 1/m"
    )


def _validate_hardware_metrics(metrics: VortexMetrics) -> None:
    if metrics.center_to_ring_ratio > MAX_CENTER_TO_RING_RATIO:
        raise RuntimeError(
            "32 档量化后的涡旋中心残余压力过高："
            f"{metrics.center_to_ring_ratio:.3%}"
        )
    if metrics.mode_purity < MIN_MODE_PURITY:
        raise RuntimeError(
            "32 档量化后的涡旋模式纯度不足："
            f"{metrics.mode_purity:.3%}"
        )


def _send_pattern(phases: np.ndarray, source: str) -> None:
    result = request_json(
        SERVICE_URL,
        "POST",
        "/v1/pattern",
        {
            "phases_rad": phases.tolist(),
            "source": source,
        },
    )
    if not result.get("transmitted", False):
        raise RuntimeError("服务端接受了相位，但没有写入串口")


def _run_tactile_locator(
    focus_phases: np.ndarray,
    vortex_phases: np.ndarray,
) -> None:
    print(
        f"[触觉定位] 同坐标单焦点，{LOCATOR_FREQUENCY_HZ:.0f} Hz 包络，"
        f"持续 {LOCATOR_SECONDS:.1f} 秒；此时不是涡旋陷阱。"
    )
    result = request_json(
        SERVICE_URL,
        "POST",
        "/v1/modulation",
        {
            "phases_rad": focus_phases.tolist(),
            "frequency_hz": LOCATOR_FREQUENCY_HZ,
            "duty_cycle": LOCATOR_DUTY_CYCLE,
            "duration_s": LOCATOR_SECONDS,
            "source": "vortex:pde:tactile-locator",
        },
    )
    if not result.get("transmitted", False):
        raise RuntimeError("触觉定位相位没有写入串口")
    time.sleep(LOCATOR_SECONDS + 0.1)
    _send_pattern(vortex_phases, "vortex:pde:restore-after-locator")
    print("[触觉定位] 已恢复固定涡旋；可放置颗粒，或再次按 L 定位。")


def _hold_with_keyboard(
    vortex_phases: np.ndarray,
    focus_phases: np.ndarray,
    duration_s: float | None,
) -> None:
    try:
        import msvcrt
    except ImportError:
        if duration_s is None:
            while True:
                command = input().strip().lower()
                if command == "l":
                    _run_tactile_locator(focus_phases, vortex_phases)
        else:
            time.sleep(duration_s)
        return

    started = time.perf_counter()
    while duration_s is None or time.perf_counter() - started < duration_s:
        if not msvcrt.kbhit():
            time.sleep(0.05)
            continue
        key = msvcrt.getwch().lower()
        if key == "l":
            print("L")
            _run_tactile_locator(focus_phases, vortex_phases)
        elif key == "\x03":
            raise KeyboardInterrupt
        elif key in {"\x00", "\xe0"}:
            msvcrt.getwch()


def load_phase_cache(
    config_file: Path = CONFIG_FILE,
    cache_file: Path = PHASE_CACHE_FILE,
) -> tuple[np.ndarray, np.ndarray, VortexMetrics, VortexMetrics] | None:
    if not cache_file.is_file():
        return None
    try:
        with np.load(cache_file, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            points = np.asarray(data["sample_points_m"], dtype=np.float64)
            response = np.asarray(data["pde_response"], dtype=np.complex128)
            phases = np.asarray(data["phases_rad"], dtype=np.float64)
            continuous = VortexMetrics(
                *np.asarray(data["continuous_metrics"], dtype=np.float64)
            )
            quantized = VortexMetrics(
                *np.asarray(data["quantized_metrics"], dtype=np.float64)
            )
    except (KeyError, ValueError, OSError, json.JSONDecodeError):
        return None
    expected_points, _ = _sample_points(
        _target(),
        VORTEX_RING_RADIUS_M,
        RING_SAMPLE_COUNT,
    )
    if metadata != _cache_metadata(config_file):
        return None
    if points.shape != expected_points.shape or response.shape != (
        expected_points.shape[0],
        256,
    ):
        return None
    if phases.shape != (256,):
        return None
    if not np.isfinite(points).all() or not np.isfinite(response).all():
        return None
    if not np.isfinite(phases).all():
        return None
    return phases, response, continuous, quantized


def prepare_vortex_phases(
    config_file: Path = CONFIG_FILE,
    cache_file: Path = PHASE_CACHE_FILE,
    force_recompute: bool = FORCE_RECOMPUTE,
) -> tuple[np.ndarray, np.ndarray, VortexMetrics, VortexMetrics]:
    if not force_recompute:
        cached = load_phase_cache(config_file, cache_file)
        if cached is not None:
            print(f"[缓存] 使用已生成的 PDE 涡旋相位：{cache_file}")
            return cached

    src_path = str(REPOSITORY_ROOT / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from algorithm.config import SimulationConfig
    from algorithm.solvers import HelmholtzDirectSolver

    target = _target()
    points, angles = _sample_points(
        target,
        VORTEX_RING_RADIUS_M,
        RING_SAMPLE_COUNT,
    )
    desired_mode = np.exp(1j * TOPOLOGICAL_CHARGE * angles)
    cfg = SimulationConfig.from_yaml(config_file)
    solver = HelmholtzDirectSolver(cfg)
    try:
        response = solver.target_response(points)
        centers = np.asarray(solver.transducers.centers, dtype=np.float64)
    finally:
        solver.close()

    center_response = response[0]
    modal_response = np.conj(desired_mode) @ response[1:] / RING_SAMPLE_COUNT
    emitter_azimuth = np.arctan2(
        centers[:, 0] - target[0],
        centers[:, 1] - target[1],
    )
    geometric_initial = (
        -np.angle(center_response)
        + TOPOLOGICAL_CHARGE * emitter_azimuth
    )
    matched_initial = -np.angle(modal_response)
    candidates = []
    for initial in (geometric_initial, matched_initial):
        result = minimize(
            vortex_loss_gradient,
            initial,
            args=(
                modal_response,
                center_response,
                CENTER_SUPPRESSION_WEIGHT,
            ),
            jac=True,
            method="L-BFGS-B",
            options={
                "maxiter": OPTIMIZATION_ITERATIONS,
                "maxfun": MAX_EVALUATIONS,
                "ftol": 1e-14,
                "gtol": 1e-10,
                "maxls": 40,
            },
        )
        candidates.append(result)
    result = min(candidates, key=lambda item: float(item.fun))
    raw_phases = np.asarray(result.x, dtype=np.float64)
    modal_pressure = modal_response @ np.exp(1j * raw_phases)
    phases = np.mod(raw_phases - np.angle(modal_pressure), 2.0 * np.pi)

    continuous = _metrics(response, phases, desired_mode)
    quantized = _metrics(response, _quantized_phases(phases), desired_mode)
    _print_metrics("PDE 连续相位", continuous)
    _print_metrics("硬件 32 档", quantized)
    _validate_hardware_metrics(quantized)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_file,
        sample_points_m=points,
        pde_response=response,
        phases_rad=phases,
        quantized_phases_rad=_quantized_phases(phases),
        continuous_metrics=np.asarray(
            [
                continuous.center_to_ring_ratio,
                continuous.mode_purity,
                continuous.ring_amplitude_cv,
                continuous.mean_ring_amplitude,
            ]
        ),
        quantized_metrics=np.asarray(
            [
                quantized.center_to_ring_ratio,
                quantized.mode_purity,
                quantized.ring_amplitude_cv,
                quantized.mean_ring_amplitude,
            ]
        ),
        optimizer_loss=float(result.fun),
        optimizer_success=bool(result.success),
        optimizer_message=str(result.message),
        field_evaluations=int(result.nfev),
        metadata=json.dumps(
            _cache_metadata(config_file),
            ensure_ascii=False,
            sort_keys=True,
        ),
    )
    print(
        f"[预计算] PDE 涡旋相位已保存：{cache_file}；"
        f"场评估={result.nfev}"
    )
    return phases, response, continuous, quantized


def run_vortex(
    offline: bool,
    duration_s: float | None,
    config_file: Path,
    force_recompute: bool,
) -> None:
    phases, response, continuous, quantized = prepare_vortex_phases(
        config_file=config_file,
        force_recompute=force_recompute,
    )
    _print_metrics("PDE 连续相位", continuous)
    _print_metrics("硬件 32 档", quantized)
    _validate_hardware_metrics(quantized)
    if offline:
        print("[离线验证] 未连接控制服务。")
        return

    status = request_json(SERVICE_URL, "GET", "/v1/status")
    if not status.get("live", False):
        raise RuntimeError("服务端不是 live 模式，没有向硬件发送")
    focus_phases = np.mod(
        -np.angle(response[0]),
        2.0 * np.pi,
    )
    _send_pattern(
        phases,
        (
            f"vortex:pde:target={tuple(_target())}:"
            f"charge={TOPOLOGICAL_CHARGE}"
        ),
    )

    print(
        f"[固定涡旋] 中心=({TARGET_X_M:.3f}, {TARGET_Y_M:.3f}, "
        f"{TARGET_HEIGHT_M:.3f}) m，拓扑荷={TOPOLOGICAL_CHARGE}，"
        "连续 40 kHz 发射"
    )
    print(
        "[固定涡旋] 按 L 临时触觉定位，随后自动恢复涡旋；"
        "按 Ctrl+C 停止并关闭阵列。"
    )
    try:
        _hold_with_keyboard(phases, focus_phases, duration_s)
    except KeyboardInterrupt:
        print("\n[固定涡旋] 收到停止请求。")
    finally:
        request_json(
            SERVICE_URL,
            "POST",
            "/v1/off",
            {"source": "vortex:pde:stop"},
        )
        print("[固定涡旋] 阵列已关闭。")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PDE-optimized fixed acoustic vortex"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="只计算和验证涡旋，不连接控制服务",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=HOLD_SECONDS,
        help="发射秒数；省略时持续到 Ctrl+C",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="忽略相位缓存并重新计算",
    )
    parser.add_argument("--config", type=Path, default=CONFIG_FILE)
    args = parser.parse_args()
    try:
        run_vortex(
            offline=args.offline,
            duration_s=args.duration,
            config_file=args.config,
            force_recompute=args.force_recompute,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"错误：{exc}") from exc


if __name__ == "__main__":
    main()
