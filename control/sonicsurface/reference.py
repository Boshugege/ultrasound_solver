"""Read-only SonicSurface reference phase calculation for validation."""
from pathlib import Path
import sys

import numpy as np

from .protocol import SONICSURFACE_EMITTERS_ORDER


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ARRAY_CENTER_X_M = 0.08
ARRAY_CENTER_Y_M = 0.08


def official_focus_phases(target) -> np.ndarray:
    """Return official focus phases in the project's logical channel order."""
    thirdparty = (
        REPOSITORY_ROOT
        / "control"
        / "thirdparty"
        / "SonicSurface"
        / "ControlSoftware"
        / "Python"
    )
    if str(thirdparty) not in sys.path:
        sys.path.insert(0, str(thirdparty))
    from SonicSurface import SonicSurface

    point = np.asarray(target, dtype=np.float64)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("target must contain three finite coordinates")
    vendor_target = np.asarray([
        point[0]-ARRAY_CENTER_X_M,
        point[2],
        point[1]-ARRAY_CENTER_Y_M,
    ])
    phases = SonicSurface().phasesForFocus(vendor_target)
    return np.mod(phases[SONICSURFACE_EMITTERS_ORDER], 2*np.pi)
