"""Load phase vectors produced by the inverse solver."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .protocol import CHANNELS


PREFERRED_FILES = (
    "result_best_phases_rad.npy",
    "result_final_phases_rad.npy",
    "result.npz",
)
PREFERRED_NPZ_KEYS = ("best_phases", "phases")


def resolve_phase_source(path: str | Path) -> Path:
    source = Path(path)
    if not source.exists():
        raise ValueError(f"相位来源不存在：{source}")
    if source.is_file():
        return source

    for name in PREFERRED_FILES:
        candidate = source / name
        if candidate.is_file():
            return candidate

    matches = [
        candidate
        for name in PREFERRED_FILES
        for candidate in source.rglob(name)
    ]
    if not matches:
        raise ValueError(f"目录中未找到标准相位输出：{source}")
    if len(matches) > 1:
        options = "\n".join(f"  - {item}" for item in sorted(matches))
        raise ValueError(f"目录中存在多个相位输出，请指定具体文件：\n{options}")
    return matches[0]


def _select_npz_key(data: np.lib.npyio.NpzFile, requested_key: str | None) -> str:
    if requested_key is not None:
        if requested_key not in data.files:
            raise ValueError(
                f"NPZ 中没有键 {requested_key!r}；可用键：{', '.join(data.files)}"
            )
        return requested_key
    for key in PREFERRED_NPZ_KEYS:
        if key in data.files:
            return key
    raise ValueError(
        "NPZ 中未找到 best_phases 或 phases；"
        f"可用键：{', '.join(data.files)}"
    )


def validate_phase_vector(values: np.ndarray, source: Path | None = None) -> np.ndarray:
    phases = np.asarray(values, dtype=np.float64)
    label = "" if source is None else f"（{source}）"
    if phases.shape != (CHANNELS,):
        raise ValueError(
            f"SonicSurface 需要 {CHANNELS} 个相位，实际形状为 {phases.shape}{label}"
        )
    if not np.isfinite(phases).all():
        raise ValueError(f"优化相位必须全部为有限数值{label}")
    return phases


def load_phase_vector(
    path: str | Path,
    npz_key: str | None = None,
) -> tuple[np.ndarray, Path, str | None]:
    source = resolve_phase_source(path)
    suffix = source.suffix.lower()
    if suffix == ".npy":
        phases = np.load(source, allow_pickle=False)
        key = None
    elif suffix == ".npz":
        with np.load(source, allow_pickle=False) as data:
            key = _select_npz_key(data, npz_key)
            phases = np.asarray(data[key])
    else:
        raise ValueError("仅支持 .npy、.npz 或包含标准结果文件的目录")
    return validate_phase_vector(phases, source), source, key
