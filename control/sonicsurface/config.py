"""Configuration loading for the SonicSurface controller."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .protocol import (
    CHANNELS,
    DEFAULT_BAUDRATE,
    SONICSURFACE_EMITTERS_ORDER,
    validate_permutation,
)


@dataclass(frozen=True)
class SonicSurfaceConfig:
    name: str
    serial_port: str | None
    baudrate: int
    channel_order_name: str
    channel_order: np.ndarray
    calibration_mode: str
    phase_corrections_bins: np.ndarray
    calibration_verified: bool


def _load_channel_order(raw: object) -> tuple[str, np.ndarray]:
    if raw == "identity":
        return "identity", np.arange(CHANNELS, dtype=np.int64)
    if raw == "sonicsurface":
        return "sonicsurface", SONICSURFACE_EMITTERS_ORDER.copy()
    return "custom", validate_permutation(np.asarray(raw, dtype=np.int64))


def _load_corrections(raw: object) -> np.ndarray:
    if raw == "zeros":
        return np.zeros(CHANNELS, dtype=np.int64)
    values = np.asarray(raw, dtype=np.int64)
    if values.shape != (CHANNELS,):
        raise ValueError("calibration.phase_corrections_bins 必须包含 256 项")
    if ((values < 0) | (values >= 32)).any():
        raise ValueError("相位校正档位必须处于 0..31")
    return values


def load_config(path: str | Path) -> SonicSurfaceConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("不支持的 SonicSurface 配置格式")

    serial = raw.get("serial", {})
    calibration = raw.get("calibration", {})
    if not isinstance(serial, dict) or not isinstance(calibration, dict):
        raise ValueError("serial 和 calibration 必须是 YAML 映射")

    baudrate = int(serial.get("baudrate", DEFAULT_BAUDRATE))
    if baudrate <= 0:
        raise ValueError("serial.baudrate 必须为正数")
    serial_port = serial.get("port")

    order_name, order = _load_channel_order(raw.get("channel_order", "sonicsurface"))
    mode = str(calibration.get("mode", "firmware")).lower()
    if mode not in {"firmware", "host"}:
        raise ValueError("calibration.mode 只能是 firmware 或 host")
    corrections = _load_corrections(
        calibration.get("phase_corrections_bins", "zeros")
    )
    if mode == "firmware" and np.any(corrections):
        raise ValueError(
            "firmware 模式下不能再配置主机相位校正，否则会重复补偿"
        )

    return SonicSurfaceConfig(
        name=str(raw.get("name", config_path.stem)),
        serial_port=None if serial_port is None else str(serial_port),
        baudrate=baudrate,
        channel_order_name=order_name,
        channel_order=order,
        calibration_mode=mode,
        phase_corrections_bins=corrections,
        calibration_verified=bool(calibration.get("verified", False)),
    )
