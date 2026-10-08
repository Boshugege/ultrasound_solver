"""Run a periodic SonicSurface envelope-frequency sweep experiment."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from sonicsurface.client import request_json
from sonicsurface.phase_source import load_phase_vector


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SERVICE_URL = "http://127.0.0.1:8765"
OPTIMIZER_CONFIG = (
    REPOSITORY_ROOT / "src" / "examples" / "front_center_free_field_16x16.yaml"
)
PHASE_FILE = (
    REPOSITORY_ROOT
    / "outputs"
    / "front_center_free_field_16x16"
    / "result_best_phases_rad.npy"
)
NPZ_KEY = None
FORCE_OPTIMIZE = False

MIN_FREQUENCY_HZ = 20.0
MAX_FREQUENCY_HZ = 500.0
SWEEP_PERIOD_S = 100.0
DUTY_CYCLE = 0.5
DURATION_S = None  # None 持续运行，直到按 Ctrl+C
REPORT_INTERVAL_S = 0.2


def ensure_phase_file(
    phase_file: Path,
    optimizer_config: Path,
    force_optimize: bool = False,
) -> Path:
    phase_file = phase_file.expanduser()
    optimizer_config = optimizer_config.expanduser()
    if not phase_file.is_absolute():
        phase_file = REPOSITORY_ROOT / phase_file
    if not optimizer_config.is_absolute():
        optimizer_config = REPOSITORY_ROOT / optimizer_config

    if phase_file.is_file() and not force_optimize:
        print(f"[相位] 使用现有文件：{phase_file}")
        return phase_file
    if not optimizer_config.is_file():
        raise RuntimeError(f"优化配置不存在：{optimizer_config}")

    print(f"[相位] 未找到最佳相位，开始运行优化：{optimizer_config}")
    subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "src" / "main.py"),
            "--config",
            str(optimizer_config),
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
    )
    if not phase_file.is_file():
        raise RuntimeError(f"优化结束后仍未生成相位文件：{phase_file}")
    print(f"[相位] 优化完成：{phase_file}")
    return phase_file


def run_frequency_sweep(
    service_url: str,
    phase_file: Path,
    npz_key: str | None,
    min_frequency_hz: float,
    max_frequency_hz: float,
    sweep_period_s: float,
    duty_cycle: float,
    duration_s: float | None,
    report_interval_s: float,
) -> None:
    phases, source, key = load_phase_vector(phase_file, npz_key)
    source_label = str(source) if key is None else f"{source}#{key}"
    result = request_json(
        service_url,
        "POST",
        "/v1/frequency-sweep",
        {
            "phases_rad": phases.tolist(),
            "min_frequency_hz": min_frequency_hz,
            "max_frequency_hz": max_frequency_hz,
            "sweep_period_s": sweep_period_s,
            "duty_cycle": duty_cycle,
            "duration_s": duration_s,
            "source": source_label,
        },
    )
    print(
        "[扫描] "
        f"{result['min_frequency_hz']:.1f}..{result['max_frequency_hz']:.1f} Hz，"
        f"往返周期 {result['sweep_period_s']:.1f} s，"
        f"占空比 {result['duty_cycle']:.0%}"
    )
    if not result["transmitted"]:
        print("[扫描] 服务端处于 dry-run，未启动真实频率扫描。")
        return

    print("[扫描] 正在运行，按 Ctrl+C 停止并关闭阵列。")
    try:
        while True:
            status = request_json(service_url, "GET", "/v1/status")
            modulation = status["modulation"]
            current = modulation.get("current_frequency_hz")
            cycles = modulation.get("cycles", 0)
            if current is not None:
                print(
                    f"\r当前包络频率：{current:7.2f} Hz"
                    f" | 占空比：{modulation['duty_cycle']:.0%}"
                    f" | 周期数：{cycles}",
                    end="",
                    flush=True,
                )
            if not modulation.get("running", False):
                print("\n[扫描] 已结束，阵列已关闭。")
                break
            time.sleep(report_interval_s)
    except KeyboardInterrupt:
        print("\n[扫描] 正在停止并关闭阵列...")
        request_json(
            service_url,
            "POST",
            "/v1/off",
            {"source": "frequency-sweep:keyboard-interrupt"},
        )
        print("[扫描] 阵列已关闭。")


def main() -> None:
    if REPORT_INTERVAL_S <= 0:
        raise SystemExit("错误：REPORT_INTERVAL_S 必须大于 0")
    try:
        phase_file = ensure_phase_file(
            PHASE_FILE,
            OPTIMIZER_CONFIG,
            force_optimize=FORCE_OPTIMIZE,
        )
        run_frequency_sweep(
            service_url=SERVICE_URL,
            phase_file=phase_file,
            npz_key=NPZ_KEY,
            min_frequency_hz=MIN_FREQUENCY_HZ,
            max_frequency_hz=MAX_FREQUENCY_HZ,
            sweep_period_s=SWEEP_PERIOD_S,
            duty_cycle=DUTY_CYCLE,
            duration_s=DURATION_S,
            report_interval_s=REPORT_INTERVAL_S,
        )
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"错误：{exc}") from exc


if __name__ == "__main__":
    main()
