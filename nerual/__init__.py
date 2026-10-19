"""GPU triangular-plate acoustic solver; no hardware output."""
import os
from pathlib import Path

_root = Path(__file__).resolve().parent
os.environ.setdefault("CUPY_CACHE_DIR", str(_root / ".cache" / "cupy"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(_root / ".cache" / "torch"))
_cuda = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6")
if _cuda.is_dir():
    os.environ.setdefault("CUDA_PATH", str(_cuda))
    if hasattr(os, "add_dll_directory"):
        _cuda_dll_directory = os.add_dll_directory(str(_cuda / "bin"))

from .geometry import Plate

__all__ = ["Plate"]
