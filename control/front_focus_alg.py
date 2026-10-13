"""Optimize a static front focus with the configured PDE and optionally transmit it."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import yaml

if __package__:
    from .circle import optimize_target_points, implementation_hash
    from .sonicsurface.client import request_json
    from .sonicsurface.reference import official_focus_phases
else:
    from circle import optimize_target_points, implementation_hash
    from sonicsurface.client import request_json
    from sonicsurface.reference import official_focus_phases


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SERVICE_URL = "http://127.0.0.1:8765"
TARGET_X_M = 0.08
TARGET_Y_M = 0.08
TARGET_HEIGHT_M = 0.12
MODULATION_FREQUENCY_HZ = 120
DUTY_CYCLE = 0.5
HOLD_SECONDS = None
CONFIG_FILE = (
    REPOSITORY_ROOT / "src" / "examples" / "pde_homogeneous_16x16.yaml"
)
PHASE_CACHE_FILE = (
    REPOSITORY_ROOT
    / "outputs"
    / "pde_homogeneous_16x16"
    / "front_focus_alg_phases.npz"
)

OPTIMIZATION_ITERATIONS = None  # Use training.iterations from the selected YAML.
FORCE_RECOMPUTE = False


def _target() -> np.ndarray:
    return np.asarray(
        [TARGET_X_M, TARGET_Y_M, TARGET_HEIGHT_M],
        dtype=np.float64,
    )


def _cache_metadata(config_file: Path) -> dict:
    config_data = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    return {
        "schema_version": 4,
        "implementation_sha256": implementation_hash(),
        "config_file": str(config_file),
        "config_sha256": hashlib.sha256(config_file.read_bytes()).hexdigest(),
        "target_m": _target().tolist(),
        "optimization_iterations": OPTIMIZATION_ITERATIONS,
        "phase_unit": "rad",
        "algorithm_source": "corrected_pde_target_response",
        "linear_solver": config_data.get("solver", {}).get("backend"),
    }


def load_phase_cache(
    config_file: Path = CONFIG_FILE,
    cache_file: Path = PHASE_CACHE_FILE,
) -> tuple[np.ndarray, float, int] | None:
    if not cache_file.is_file():
        return None
    try:
        with np.load(cache_file, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            phases = np.asarray(data["phases_rad"], dtype=np.float64)
            best_loss = float(data["best_loss"])
            evaluations = int(data["field_evaluations"])
    except (KeyError, ValueError, OSError, json.JSONDecodeError):
        return None
    if metadata != _cache_metadata(config_file):
        return None
    if phases.shape != (256,) or not np.isfinite(phases).all():
        return None
    return phases, best_loss, evaluations


def prepare_algorithm_phases(
    config_file: Path = CONFIG_FILE,
    cache_file: Path = PHASE_CACHE_FILE,
    force_recompute: bool = FORCE_RECOMPUTE,
) -> tuple[np.ndarray, float, int]:
    if not force_recompute:
        cached = load_phase_cache(config_file, cache_file)
        if cached is not None:
            print(f"[缓存] 使用算法聚焦相位：{cache_file}")
            return cached

    target = _target()
    phases, best_losses, evaluations = optimize_target_points(
        target.reshape(1, 3),
        config_file,
        OPTIMIZATION_ITERATIONS,
    )
    selected = np.mod(phases[0], 2.0 * np.pi)
    best_loss = float(best_losses[0])
    field_evaluations = int(evaluations[0])

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_file,
        phases_rad=selected,
        best_loss=best_loss,
        field_evaluations=field_evaluations,
        metadata=json.dumps(
            _cache_metadata(config_file),
            ensure_ascii=False,
            sort_keys=True,
        ),
    )
    print(f"[预计算] 算法聚焦相位已保存：{cache_file}")
    return selected, best_loss, field_evaluations


def phase_difference_degrees(
    algorithm_phases: np.ndarray,
    reference_phases: np.ndarray,
) -> np.ndarray:
    """Return circular phase differences after removing global phase."""
    delta = np.angle(
        np.exp(
            1j
            * (
                np.asarray(algorithm_phases, dtype=np.float64)
                - np.asarray(reference_phases, dtype=np.float64)
            )
        )
    )
    global_offset = np.angle(np.mean(np.exp(1j * delta)))
    residual = np.angle(np.exp(1j * (delta - global_offset)))
    return np.rad2deg(residual)


def run_algorithm_front_focus(offline: bool = False, config_file: Path = CONFIG_FILE) -> None:
    phases, best_loss, evaluations = prepare_algorithm_phases(config_file=config_file)
    reference = official_focus_phases(_target())
    differences = phase_difference_degrees(phases, reference)
    target = tuple(_target())

    print(
        "[算法聚焦] "
        f"loss={best_loss:.6g}，场评估={evaluations}，"
        f"相对官方相位 MAE={np.mean(np.abs(differences)):.2f}°，"
        f"最大误差={np.max(np.abs(differences)):.2f}°"
    )

    if offline:
        import sys
        sys.path[:0] = [str(REPOSITORY_ROOT), str(REPOSITORY_ROOT / "src")]
        from algorithm.config import SimulationConfig
        from algorithm.solvers import HelmholtzDirectSolver

        cfg = SimulationConfig.from_yaml(config_file)
        cfg.targets = [_target().tolist()]
        solver = HelmholtzDirectSolver(cfg)
        try:
            target = _target()[None, :]
            response = solver.target_response(target)
            algorithm_field = solver.field_at(target, phases)
            reference_field = solver.field_at(target, reference)
            output = Path(cfg.io.output_file).with_name(
                "front_focus_target_comparison.npz"
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output,
                target_m=target,
                pde_response=response,
                algorithm_phases_rad=phases,
                official_phases_rad=reference,
                algorithm_field=algorithm_field,
                official_field=reference_field,
            )
        finally:
            solver.close()
        print(
            "[离线 PDE] "
            f"算法相位目标振幅={abs(algorithm_field[0]):.6f} 1/m，"
            f"官方相位目标振幅={abs(reference_field[0]):.6f} 1/m，"
            f"比值={abs(algorithm_field[0])/abs(reference_field[0]):.6%}；"
            f"已保存：{output}"
        )
        print("[离线验证] 未连接控制服务。")
        return

    status = request_json(SERVICE_URL, "GET", "/v1/status")
    if not status.get("live", False):
        raise RuntimeError("服务端不是 live 模式，没有向硬件发送")

    payload = {
        "phases_rad": phases.tolist(),
        "source": f"front-focus:algorithm:target={target}",
    }
    if MODULATION_FREQUENCY_HZ is None:
        endpoint = "/v1/pattern"
        mode = "连续 40 kHz，无触觉包络"
    else:
        endpoint = "/v1/modulation"
        payload.update(
            {
                "frequency_hz": MODULATION_FREQUENCY_HZ,
                "duty_cycle": DUTY_CYCLE,
                "duration_s": HOLD_SECONDS,
            }
        )
        mode = (
            f"{MODULATION_FREQUENCY_HZ:.1f} Hz 包络，"
            f"{DUTY_CYCLE:.0%} 占空比"
        )

    result = request_json(SERVICE_URL, "POST", endpoint, payload)
    if not result.get("transmitted", False):
        raise RuntimeError("服务端接受了请求，但没有写入串口")

    print(
        "[算法聚焦] "
        f"目标=({TARGET_X_M:.3f}, {TARGET_Y_M:.3f}, "
        f"{TARGET_HEIGHT_M:.3f}) m，{mode}"
    )
    print("[算法聚焦] 正在发射；按 Ctrl+C 停止并关闭阵列。")
    try:
        if HOLD_SECONDS is None:
            while True:
                time.sleep(1.0)
        else:
            deadline = time.perf_counter() + HOLD_SECONDS
            while time.perf_counter() < deadline:
                time.sleep(min(0.2, deadline - time.perf_counter()))
    except KeyboardInterrupt:
        print("\n[算法聚焦] 收到停止请求。")
    finally:
        request_json(
            SERVICE_URL,
            "POST",
            "/v1/off",
            {"source": "front-focus:algorithm:stop"},
        )
        print("[算法聚焦] 阵列已关闭。")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="PDE-optimized SonicSurface front focus")
    parser.add_argument("--offline", action="store_true", help="只生成相位与对拍，不连接控制服务")
    parser.add_argument("--config", type=Path, default=CONFIG_FILE)
    args = parser.parse_args()
    try:
        run_algorithm_front_focus(offline=args.offline, config_file=args.config)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"错误：{exc}") from exc


if __name__ == "__main__":
    main()
