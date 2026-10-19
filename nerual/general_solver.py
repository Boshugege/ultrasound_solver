"""Public neural solver for arbitrary closed rigid meshes and reflectors."""
from dataclasses import dataclass
from pathlib import Path
import time
import numpy as np
import torch
from .aperture import aperture_points, sinc_aperture_points
from .bem import green
from .generic_bem import MeshBEM
from .geometry import array_centers
from .model import BoundaryNet, CapturedBoundaryNet


@dataclass
class MeshSolution:
    response: np.ndarray
    phases_rad: np.ndarray
    phase_codes_32: np.ndarray
    method: str
    neural_residual: float | None
    final_residual: float
    setup_ms: float
    solve_ms: float


class GeneralNeuralSolver:
    """40 kHz rigid-mesh solver; static reflectors are merged closed bodies."""

    def __init__(self, checkpoint=None, receivers=None, diameter=.009,
                 device="cuda", residual_tolerance=2e-2, max_direct_panels=2200,
                 aperture_profile="uniform", frequency=40000.,
                 sound_speed=346.):
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA-enabled PyTorch is required")
        self.device = torch.device(device)
        self.receivers = array_centers() if receivers is None else np.asarray(receivers, float)
        if (self.receivers.ndim != 2 or self.receivers.shape[1] != 3
                or not np.isfinite(self.receivers).all()
                or not np.allclose(self.receivers[:, 2], 0, atol=1e-12)):
            raise ValueError("receivers must be finite [C,3] points on z=0")
        if aperture_profile == "sinc":
            self.aperture, self.weights = sinc_aperture_points(self.receivers, diameter)
        elif aperture_profile == "uniform":
            self.aperture, self.weights = aperture_points(self.receivers, diameter)
        else:
            raise ValueError("aperture_profile must be sinc or uniform")
        self.tolerance = float(residual_tolerance)
        self.max_direct_panels = int(max_direct_panels)
        self.frequency = float(frequency)
        self.sound_speed = float(sound_speed)
        path = Path(checkpoint) if checkpoint else Path(__file__).with_name("artifacts")/"boundary_net.pt"
        self.model = None
        self._graphs = {}
        if path.is_file():
            saved = torch.load(path, map_location=self.device, weights_only=True)
            self.model = BoundaryNet(saved["layers"]).to(self.device)
            self.model.load_state_dict(saved["state_dict"])
            self.model.eval()

    def _tensor(self, value):
        if self.device.type == "cuda":
            return torch.from_dlpack(value).to(torch.complex64)
        return torch.as_tensor(value, dtype=torch.complex64)

    def set_scene(self, dynamic_body, reflectors=()):
        from .mesh_scene import MeshBody
        body = MeshBody.combine((dynamic_body, *tuple(reflectors)))
        started = time.perf_counter()
        self.bem = MeshBEM(
            body, frequency=self.frequency, sound_speed=self.sound_speed,
            backend="cupy" if self.device.type=="cuda" else "numpy")
        self.a = self._tensor(self.bem.A)
        self.ah = self.a.mH
        self.gram = None
        self.chol = None
        propagation = self._tensor(self.bem.propagation(self.aperture.reshape(-1, 3)))
        w = torch.as_tensor(self.weights, device=self.device, dtype=torch.float32)
        self.d = (propagation.reshape(len(self.receivers), len(w), -1)*w[None, :, None]).sum(1)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.setup_ms = (time.perf_counter()-started)*1000
        return self

    def _direct(self, b, x0=None):
        if self.a.shape[1] > self.max_direct_panels:
            raise RuntimeError(
                f"Neural residual failed but direct fallback is disabled above "
                f"{self.max_direct_panels} panels (scene has {self.a.shape[1]}).")
        rhs = self.ah@b
        if self.gram is None:
            self.gram = self.ah@self.a
            self.chol, info = torch.linalg.cholesky_ex(self.gram)
            self.chol = None if info.max().item() else self.chol
        if self.chol is not None:
            x = torch.cholesky_solve(rhs, self.chol)
            x += torch.cholesky_solve(self.ah@(b-self.a@x), self.chol)
            return x
        return torch.linalg.lstsq(self.a, b).solution

    def _residual(self, x, b):
        return (torch.linalg.vector_norm(self.ah@(self.a@x-b), dim=0)
                / torch.linalg.vector_norm(self.ah@b, dim=0).clamp_min(1e-8)).max()

    @torch.inference_mode()
    def solve(self, targets, method="auto"):
        if not hasattr(self, "bem"):
            raise RuntimeError("Call set_scene() first")
        if method not in {"auto", "neural", "direct"}:
            raise ValueError("method must be auto, neural, or direct")
        targets = np.atleast_2d(np.asarray(targets, float))
        started = time.perf_counter()
        b = self._tensor(self.bem.incident(targets))
        candidate = None
        x = None
        used = "direct"
        if method in {"auto", "neural"}:
            if self.model is None:
                if method == "neural":
                    raise RuntimeError("No trained boundary_net.pt checkpoint")
            else:
                if self.device.type == "cuda":
                    key = (tuple(self.a.shape), tuple(b.shape))
                    if key not in self._graphs:
                        self._graphs[key] = CapturedBoundaryNet(self.model, self.a, b)
                    x = self._graphs[key](self.a, b)
                else:
                    x = self.model(self.a, b)
                candidate = float(self._residual(x, b))
                used = "neural"
                if method == "auto" and (not np.isfinite(candidate) or candidate > self.tolerance):
                    x, used = None, "direct_fallback"
        if x is None:
            x = self._direct(b)
        residual = float(self._residual(x, b))
        direct = green(self.aperture.reshape(-1, 3), targets, self.bem.k).reshape(
            len(self.receivers), len(self.weights), len(targets))
        direct = np.einsum("cqt,q->ct", direct, self.weights)/2
        direct = torch.as_tensor(direct, device=self.device, dtype=torch.complex64)
        response = (direct+self.d@x/2).T.cpu().numpy()
        phases = np.mod(-np.angle(response), 2*np.pi)
        codes = (np.floor(phases*32/(2*np.pi)+.5).astype(np.int32)%32).astype(np.uint8)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return MeshSolution(response, phases, codes, used, candidate, residual,
                            self.setup_ms, (time.perf_counter()-started)*1000)
