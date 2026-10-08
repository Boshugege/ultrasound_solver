"""SonicSurface UART frame construction.

Protocol source:
control/thirdparty/SonicSurface/ControlSoftware/Python/SonicSurface.py
"""

from __future__ import annotations

import numpy as np


CHANNELS = 256
PHASE_DIVISIONS = 32
PHASE_OFF = 32
START_PHASES = 254
SWAP_BUFFERS = 253
DEFAULT_BAUDRATE = 230400

# Ideal 16x16 row-major emitter index -> FPGA channel index. This is copied
# from SonicSurface.sendPattern(..., permuteToFPGAOrder=True).
SONICSURFACE_EMITTERS_ORDER = np.asarray(
    [
        0, 7, 1, 2, 64, 71, 65, 66, 128, 135, 129, 130, 192, 199, 193, 194,
        4, 3, 6, 5, 68, 67, 70, 69, 132, 131, 134, 133, 196, 195, 198, 197,
        8, 15, 9, 10, 72, 79, 73, 74, 136, 143, 137, 138, 200, 207, 201, 202,
        12, 11, 14, 13, 76, 75, 78, 77, 140, 139, 142, 141, 204, 203, 206, 205,
        16, 23, 17, 18, 80, 87, 81, 82, 144, 151, 145, 146, 208, 215, 209, 210,
        20, 19, 22, 21, 84, 83, 86, 85, 148, 147, 150, 149, 212, 211, 214, 213,
        24, 31, 25, 26, 88, 95, 89, 90, 152, 159, 153, 154, 216, 223, 217, 218,
        28, 27, 30, 29, 92, 91, 94, 93, 156, 155, 158, 157, 220, 219, 222, 221,
        32, 39, 33, 34, 96, 103, 97, 98, 160, 167, 161, 162, 224, 231, 225, 226,
        36, 35, 38, 37, 100, 99, 102, 101, 164, 163, 166, 165, 228, 227, 230, 229,
        40, 47, 41, 42, 104, 111, 105, 106, 168, 175, 169, 170, 232, 239, 233, 234,
        44, 43, 46, 45, 108, 107, 110, 109, 172, 171, 174, 173, 236, 235, 238, 237,
        48, 55, 49, 50, 112, 119, 113, 114, 176, 183, 177, 178, 240, 247, 241, 242,
        52, 51, 54, 53, 116, 115, 118, 117, 180, 179, 182, 181, 244, 243, 246, 245,
        56, 63, 57, 58, 120, 127, 121, 122, 184, 191, 185, 186, 248, 255, 249, 250,
        60, 59, 62, 61, 124, 123, 126, 125, 188, 187, 190, 189, 252, 251, 254, 253,
    ],
    dtype=np.int64,
)


def validate_permutation(order: np.ndarray) -> np.ndarray:
    values = np.asarray(order, dtype=np.int64)
    if values.shape != (CHANNELS,) or not np.array_equal(
        np.sort(values), np.arange(CHANNELS)
    ):
        raise ValueError("channel_order 必须是 0..255 的完整排列")
    return values


def quantize_phases(
    phases_rad: np.ndarray,
    phase_corrections_bins: np.ndarray | None = None,
) -> np.ndarray:
    """Match vendor truncation to 32 bins; NaN marks a disabled channel."""
    phases = np.asarray(phases_rad, dtype=np.float64)
    if phases.shape != (CHANNELS,):
        raise ValueError(f"需要 {CHANNELS} 个相位，实际形状为 {phases.shape}")
    if np.isinf(phases).any():
        raise ValueError("相位不能包含正负无穷")

    corrections = (
        np.zeros(CHANNELS, dtype=np.int64)
        if phase_corrections_bins is None
        else np.asarray(phase_corrections_bins, dtype=np.int64)
    )
    if corrections.shape != (CHANNELS,):
        raise ValueError("phase_corrections_bins 必须包含 256 项")

    disabled = np.isnan(phases)
    safe_phases = np.where(disabled, 0.0, phases)
    bins = np.floor(
        np.mod(safe_phases, 2.0 * np.pi)
        * PHASE_DIVISIONS
        / 2.0 / np.pi
    )
    bins = bins.astype(np.int64)
    # Preserve the vendor's floating-point edge behavior (a remainder can
    # round to 2*pi and encode 32). Apply modular wrapping only to corrections.
    corrected = corrections != 0
    bins[corrected] = (bins[corrected] + corrections[corrected]) % PHASE_DIVISIONS
    bins[disabled] = PHASE_OFF
    return bins.astype(np.uint8)


def compile_pattern(
    phases_rad: np.ndarray,
    channel_order: np.ndarray,
    phase_corrections_bins: np.ndarray | None = None,
) -> bytes:
    """Build one complete 258-byte SonicSurface phase frame."""
    bins = quantize_phases(phases_rad, phase_corrections_bins)
    order = validate_permutation(channel_order)
    device_bins = np.full(CHANNELS, PHASE_OFF, dtype=np.uint8)
    device_bins[order] = bins
    return bytes((START_PHASES,)) + device_bins.tobytes() + bytes((SWAP_BUFFERS,))


def compile_all_off() -> bytes:
    return (
        bytes((START_PHASES,))
        + bytes((PHASE_OFF,)) * CHANNELS
        + bytes((SWAP_BUFFERS,))
    )


def compile_swap() -> bytes:
    """Swap the two FPGA phase buffers without uploading another pattern."""
    return bytes((SWAP_BUFFERS,))
