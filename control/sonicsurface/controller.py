"""Thread-safe SonicSurface controller with dry-run support."""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from contextlib import suppress
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np

from .config import SonicSurfaceConfig
from .protocol import (
    CHANNELS,
    PHASE_DIVISIONS,
    START_PHASES,
    compile_all_off,
    compile_pattern,
    compile_swap,
)


SerialFactory = Callable[..., Any]
LOGGER = logging.getLogger("sonicsurface.controller")
MIN_MODULATION_HZ = 1.0
MAX_MODULATION_HZ = 500.0
MIN_DUTY_CYCLE = 0.05
MAX_DUTY_CYCLE = 0.95
MAX_DURATION_S = 3600.0
SERIAL_WRITE_TIMEOUT_S = 3.0
SERIAL_REOPEN_SETTLE_S = 2.0


def triangle_sweep_frequency(
    elapsed_s: float,
    min_frequency_hz: float,
    max_frequency_hz: float,
    sweep_period_s: float,
) -> float:
    phase = (elapsed_s % sweep_period_s) / sweep_period_s
    ratio = 2.0 * phase if phase <= 0.5 else 2.0 * (1.0 - phase)
    return min_frequency_hz + (max_frequency_hz - min_frequency_hz) * ratio


class SonicSurfaceController:
    def __init__(
        self,
        config: SonicSurfaceConfig,
        live: bool = False,
        serial_factory: SerialFactory | None = None,
    ):
        if live and not config.serial_port:
            raise ValueError("真实发送要求在配置中设置 serial.port")
        if live and not config.calibration_verified:
            raise ValueError(
                "真实发送要求 calibration.verified: true；"
                "请先确认通道映射和相位标定"
            )

        self.config = config
        self.live = live
        self._serial_factory = serial_factory
        self._serial_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._serial = None
        self._serial_recoveries = 0
        self._last_serial_error: str | None = None
        self._last_submission: dict[str, Any] | None = None
        self._modulation_stop = threading.Event()
        self._modulation_thread: threading.Thread | None = None
        self._modulation: dict[str, Any] = {
            "running": False,
            "mode": None,
            "frequency_hz": None,
            "current_frequency_hz": None,
            "min_frequency_hz": None,
            "max_frequency_hz": None,
            "sweep_period_s": None,
            "duty_cycle": None,
            "duration_s": None,
            "source": None,
            "cycles": 0,
            "last_error": None,
        }

        if live:
            if self._serial_factory is None:
                try:
                    import serial
                except ImportError as exc:
                    raise RuntimeError("缺少 pyserial，无法打开串口") from exc
                self._serial_factory = serial.Serial
            self._serial = self._open_serial()

    def _open_serial(self) -> Any:
        assert self._serial_factory is not None
        return self._serial_factory(
            self.config.serial_port,
            self.config.baudrate,
            timeout=1,
            write_timeout=SERIAL_WRITE_TIMEOUT_S,
        )

    def status(self) -> dict[str, Any]:
        with self._state_lock:
            modulation = dict(self._modulation)
        return {
            "service": "sonicsurface-control",
            "config": self.config.name,
            "channels": CHANNELS,
            "phase_divisions": PHASE_DIVISIONS,
            "baudrate": self.config.baudrate,
            "serial_port": self.config.serial_port,
            "channel_order": self.config.channel_order_name,
            "calibration_mode": self.config.calibration_mode,
            "calibration_verified": self.config.calibration_verified,
            "live": self.live,
            "serial_connected": self._serial is not None,
            "serial_recoveries": self._serial_recoveries,
            "last_serial_error": self._last_serial_error,
            "last_submission": self._last_submission,
            "modulation": modulation,
        }

    def _write_once_locked(self, data: bytes) -> None:
        assert self._serial is not None
        written = self._serial.write(data)
        self._serial.flush()
        if written is not None and written != len(data):
            raise IOError(f"串口仅写入 {written}/{len(data)} 字节")

    def _reopen_serial_locked(self) -> None:
        previous = self._serial
        self._serial = None
        if previous is not None:
            with suppress(Exception):
                previous.close()
        self._serial = self._open_serial()
        # Arduino-class USB serial bridges commonly reset when the port opens.
        # Wait for the bootloader/application before resending the full frame.
        time.sleep(SERIAL_REOPEN_SETTLE_S)
        self._serial_recoveries += 1

    def _send_bytes(self, data: bytes) -> bool:
        if not self.live:
            return False
        with self._serial_lock:
            try:
                self._write_once_locked(data)
                self._last_serial_error = None
            except (IOError, OSError) as exc:
                self._last_serial_error = str(exc)
                is_complete_frame = (
                    len(data) >= CHANNELS + 2
                    and data[0] == START_PHASES
                )
                if not is_complete_frame:
                    raise
                LOGGER.warning(
                    "完整相位帧写入失败，重开串口后重发一次：%s",
                    exc,
                )
                self._reopen_serial_locked()
                try:
                    # 0xFE resets the FPGA byte counter, so resending a
                    # complete frame safely replaces any partial frame.
                    self._write_once_locked(data)
                    self._last_serial_error = None
                except (IOError, OSError) as retry_exc:
                    self._last_serial_error = str(retry_exc)
                    raise
        return True

    def _result(self, frame: bytes, transmitted: bool, source: str) -> dict[str, Any]:
        result = {
            "accepted": True,
            "transmitted": transmitted,
            "frame_bytes": len(frame),
            "payload_sha256": hashlib.sha256(frame).hexdigest(),
            "source": source,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        self._last_submission = result
        return result

    def submit(self, phases_rad: np.ndarray, source: str = "client") -> dict[str, Any]:
        self._stop_modulation()
        corrections = (
            self.config.phase_corrections_bins
            if self.config.calibration_mode == "host"
            else None
        )
        frame = compile_pattern(
            phases_rad,
            channel_order=self.config.channel_order,
            phase_corrections_bins=corrections,
        )
        return self._result(frame, self._send_bytes(frame), source)

    def off(self, source: str = "client") -> dict[str, Any]:
        self._stop_modulation()
        frame = compile_all_off()
        return self._result(frame, self._send_bytes(frame), source)

    @staticmethod
    def _validate_modulation(
        frequency_hz: float,
        duty_cycle: float,
        duration_s: float | None,
    ) -> tuple[float, float, float | None]:
        try:
            frequency = float(frequency_hz)
            duty = float(duty_cycle)
            duration = None if duration_s is None else float(duration_s)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "frequency_hz、duty_cycle 和 duration_s 必须是数值"
            ) from exc
        if not np.isfinite(frequency) or not (
            MIN_MODULATION_HZ <= frequency <= MAX_MODULATION_HZ
        ):
            raise ValueError(
                f"frequency_hz 必须处于 {MIN_MODULATION_HZ:g}.."
                f"{MAX_MODULATION_HZ:g} Hz"
            )
        if not np.isfinite(duty) or not (
            MIN_DUTY_CYCLE <= duty <= MAX_DUTY_CYCLE
        ):
            raise ValueError(
                f"duty_cycle 必须处于 {MIN_DUTY_CYCLE:g}.."
                f"{MAX_DUTY_CYCLE:g}"
            )
        if duration is not None and (
            not np.isfinite(duration) or not (0.0 < duration <= MAX_DURATION_S)
        ):
            raise ValueError(f"duration_s 必须大于 0 且不超过 {MAX_DURATION_S:g}")
        return frequency, duty, duration

    def start_modulation(
        self,
        phases_rad: np.ndarray,
        frequency_hz: float,
        duty_cycle: float = 0.5,
        duration_s: float | None = None,
        source: str = "client",
    ) -> dict[str, Any]:
        """Start on/off envelope modulation while keeping the 40 kHz carrier."""
        frequency, duty, duration = self._validate_modulation(
            frequency_hz, duty_cycle, duration_s
        )
        return self._start_modulation_task(
            phases_rad=phases_rad,
            initial_frequency_hz=frequency,
            duty_cycle=duty,
            duration_s=duration,
            source=source,
            mode="fixed",
        )

    def start_frequency_sweep(
        self,
        phases_rad: np.ndarray,
        min_frequency_hz: float,
        max_frequency_hz: float,
        sweep_period_s: float,
        duty_cycle: float = 0.5,
        duration_s: float | None = None,
        source: str = "client",
    ) -> dict[str, Any]:
        """Sweep the envelope frequency up and down with a triangle waveform."""
        minimum, duty, duration = self._validate_modulation(
            min_frequency_hz, duty_cycle, duration_s
        )
        maximum, _, _ = self._validate_modulation(
            max_frequency_hz, duty_cycle, duration_s
        )
        try:
            sweep_period = float(sweep_period_s)
        except (TypeError, ValueError) as exc:
            raise ValueError("sweep_period_s 必须是数值") from exc
        if maximum <= minimum:
            raise ValueError("max_frequency_hz 必须大于 min_frequency_hz")
        if not np.isfinite(sweep_period) or not (0.1 <= sweep_period <= 3600.0):
            raise ValueError("sweep_period_s 必须处于 0.1..3600 秒")
        return self._start_modulation_task(
            phases_rad=phases_rad,
            initial_frequency_hz=minimum,
            duty_cycle=duty,
            duration_s=duration,
            source=source,
            mode="sweep",
            min_frequency_hz=minimum,
            max_frequency_hz=maximum,
            sweep_period_s=sweep_period,
        )

    def _start_modulation_task(
        self,
        phases_rad: np.ndarray,
        initial_frequency_hz: float,
        duty_cycle: float,
        duration_s: float | None,
        source: str,
        mode: str,
        min_frequency_hz: float | None = None,
        max_frequency_hz: float | None = None,
        sweep_period_s: float | None = None,
    ) -> dict[str, Any]:
        corrections = (
            self.config.phase_corrections_bins
            if self.config.calibration_mode == "host"
            else None
        )
        pattern_frame = compile_pattern(
            phases_rad,
            channel_order=self.config.channel_order,
            phase_corrections_bins=corrections,
        )
        off_frame = compile_all_off()

        self._stop_modulation()
        pattern_sent = self._send_bytes(pattern_frame)
        off_sent = self._send_bytes(off_frame)
        transmitted = pattern_sent and off_sent

        with self._state_lock:
            self._modulation = {
                "running": self.live,
                "mode": mode,
                "frequency_hz": (
                    initial_frequency_hz if mode == "fixed" else None
                ),
                "current_frequency_hz": initial_frequency_hz,
                "min_frequency_hz": min_frequency_hz,
                "max_frequency_hz": max_frequency_hz,
                "sweep_period_s": sweep_period_s,
                "duty_cycle": duty_cycle,
                "duration_s": duration_s,
                "source": source,
                "cycles": 0,
                "last_error": None,
            }

        if self.live:
            self._modulation_stop = threading.Event()
            self._modulation_thread = threading.Thread(
                target=self._run_modulation,
                args=(
                    initial_frequency_hz,
                    duty_cycle,
                    duration_s,
                    min_frequency_hz,
                    max_frequency_hz,
                    sweep_period_s,
                ),
                name="sonicsurface-modulation",
                daemon=True,
            )
            self._modulation_thread.start()

        result = self._result(
            pattern_frame + off_frame,
            transmitted,
            source,
        )
        result.update(
            {
                "modulation_running": self.live,
                "modulation_mode": mode,
                "frequency_hz": (
                    initial_frequency_hz if mode == "fixed" else None
                ),
                "current_frequency_hz": initial_frequency_hz,
                "min_frequency_hz": min_frequency_hz,
                "max_frequency_hz": max_frequency_hz,
                "sweep_period_s": sweep_period_s,
                "duty_cycle": duty_cycle,
                "duration_s": duration_s,
                "carrier_frequency_hz": 40000,
            }
        )
        return result

    def _run_modulation(
        self,
        frequency_hz: float,
        duty_cycle: float,
        duration_s: float | None,
        min_frequency_hz: float | None,
        max_frequency_hz: float | None,
        sweep_period_s: float | None,
    ) -> None:
        started = time.perf_counter()
        cycles = 0
        error: str | None = None
        try:
            while not self._modulation_stop.is_set():
                elapsed = time.perf_counter() - started
                if duration_s is not None and elapsed >= duration_s:
                    break
                current_frequency_hz = frequency_hz
                if (
                    min_frequency_hz is not None
                    and max_frequency_hz is not None
                    and sweep_period_s is not None
                ):
                    current_frequency_hz = triangle_sweep_frequency(
                        elapsed,
                        min_frequency_hz,
                        max_frequency_hz,
                        sweep_period_s,
                    )
                period_s = 1.0 / current_frequency_hz
                on_s = period_s * duty_cycle
                with self._state_lock:
                    self._modulation["current_frequency_hz"] = current_frequency_hz
                cycle_started = time.perf_counter()
                self._send_bytes(compile_swap())
                if self._modulation_stop.wait(on_s):
                    break
                self._send_bytes(compile_swap())
                cycles += 1
                with self._state_lock:
                    self._modulation["cycles"] = cycles
                remaining = period_s - (time.perf_counter() - cycle_started)
                if remaining > 0.0:
                    self._modulation_stop.wait(remaining)
        except (IOError, OSError) as exc:
            error = str(exc)
            LOGGER.exception("波群调制串口发送失败")
        finally:
            try:
                self._send_bytes(compile_all_off())
            except (IOError, OSError) as exc:
                error = str(exc)
                LOGGER.exception("波群调制结束时关闭阵列失败")
            with self._state_lock:
                self._modulation["running"] = False
                self._modulation["cycles"] = cycles
                self._modulation["last_error"] = error

    def _stop_modulation(self) -> None:
        thread = self._modulation_thread
        if thread is None or not thread.is_alive():
            self._modulation_thread = None
            return
        self._modulation_stop.set()
        if thread is not threading.current_thread():
            thread.join(timeout=2.0)
        if thread.is_alive():
            raise RuntimeError("波群调制线程未能在 2 秒内停止")
        self._modulation_thread = None

    def close(self) -> None:
        self._stop_modulation()
        if self._serial is not None:
            self._serial.close()
            self._serial = None

    def __enter__(self) -> "SonicSurfaceController":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
