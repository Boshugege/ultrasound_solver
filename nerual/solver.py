"""Public inference API. No serial, TCP, HTTP, or hardware side effects."""
from dataclasses import dataclass
from pathlib import Path
import time
import numpy as np
import torch
from .bem import BEMTeacher, green
from .geometry import Plate, array_centers
from .model import BoundaryNet, CapturedBoundaryNet, relative_residual
from .dataset import COVERAGE
from .aperture import aperture_points


@dataclass
class Solution:
    response: np.ndarray
    phases_rad: np.ndarray
    phase_codes_32: np.ndarray
    method: str
    normal_residual: float
    boundary_fit_residual: float
    in_training_domain: bool
    setup_ms: float
    solve_ms: float
    neural_candidate_residual: float | None


class TriangleSolver:
    """Response and independent single-focus phases for a rigid triangular plate.

    set_plate() recomputes the geometry operator; solve() accepts new targets.
    auto checks the learned iterate and falls back to numerical least squares.
    A small linear residual does not certify physical/discretization accuracy.
    """
    def __init__(self, checkpoint=None, device="cuda", subdivisions=None,
                 thickness_layers=None, residual_tolerance=1e-2, receivers=None,
                 diameter=.009):
        if device not in ("cuda","cpu"):
            raise ValueError("device must be cuda or cpu")
        if device=="cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use the CUDA-enabled conda srt environment")
        if residual_tolerance <= 0 or not np.isfinite(residual_tolerance):
            raise ValueError("residual_tolerance must be positive")
        self.device = device
        self.subdivisions = subdivisions
        self.thickness_layers = thickness_layers
        self.tolerance = residual_tolerance
        self.receivers = array_centers() if receivers is None else np.asarray(receivers,float)
        if (self.receivers.ndim != 2 or self.receivers.shape[1] != 3
                or not np.isfinite(self.receivers).all()
                or not np.allclose(self.receivers[:,2],0,atol=1e-12)):
            raise ValueError("receivers must be finite Nx3 array centers on z=0")
        self.aperture_points,self.aperture_weights = aperture_points(self.receivers,diameter)
        path = Path(checkpoint) if checkpoint else Path(__file__).with_name("artifacts")/"boundary_net.pt"
        self.model = None
        if path.exists():
            saved = torch.load(path,map_location=device,weights_only=True)
            self.model = BoundaryNet(saved["layers"]).to(device)
            self.model.load_state_dict(saved["state_dict"])
            self.model.eval()
        elif checkpoint:
            raise FileNotFoundError(path)
        self.teacher = None
        self._chol = None
        self._graphs = {}

    def _sync(self):
        if self.device=="cuda":
            torch.cuda.synchronize()

    def _tensor(self,array):
        if self.device=="cuda":
            return torch.from_dlpack(array).to(torch.complex64)
        return torch.as_tensor(array,dtype=torch.complex64)

    def set_plate(self,plate: Plate):
        self._sync()
        started = time.perf_counter()
        n = self.subdivisions
        if n is None:
            n = max(3,int(np.ceil(plate.side/(346/40000/3))))
        m = self.thickness_layers or max(1,int(np.ceil(n*plate.thickness/plate.side)))
        if 2*n*n+6*n*m > 2200:
            raise ValueError("Requested geometry needs >2200 panels; increase the supported mesh budget explicitly in a validated configuration")
        self.teacher = BEMTeacher(plate,n,backend="cupy" if self.device=="cuda" else "numpy",
                                  thickness_layers=m)
        self.a = self._tensor(self.teacher.A)
        self.gram = self.a.mH@self.a
        abs_a = self.a.abs()
        self.bound = abs_a.sum(-1).max()*abs_a.sum(-2).max()
        self.normal = self.gram/self.bound
        propagation = self._tensor(self.teacher.propagation(self.aperture_points.reshape(-1,3)))
        weights = torch.as_tensor(self.aperture_weights,device=self.device,dtype=torch.float32)
        self.d = (propagation.reshape(len(self.receivers),len(weights),-1)*weights[None,:,None]).sum(1)
        self._chol = None
        self._sync()
        self.setup_ms = (time.perf_counter()-started)*1000
        return self

    def _in_domain(self,targets):
        p = self.teacher.plate
        pairs = [(p.side,COVERAGE["side_m"]),(p.thickness,COVERAGE["thickness_m"])]
        pairs += list(zip(p.center,([.045,.115],[.045,.115],[.035,.075])))
        return (all(lo<=v<=hi for v,(lo,hi) in pairs)
                and p.vertices()[:,2].min() >= .005
                and np.all(targets >= [.045,.045,.04])
                and np.all(targets <= [.115,.115,.13])
                and not p.contains(targets,margin=.005).any())

    def _direct(self,b):
        # CHIEF makes the system rectangular. Cholesky is reused for target motion.
        ah = self.a.mH
        if self._chol is None:
            self._chol,info = torch.linalg.cholesky_ex(self.gram)
            if info.item() != 0:
                self._chol = None
                return torch.linalg.lstsq(self.a,b).solution
        x = torch.cholesky_solve(ah@b,self._chol)
        # One iterative refinement improves single precision stationarity.
        x += torch.cholesky_solve(ah@(b-self.a@x),self._chol)
        return x

    @torch.inference_mode()
    def solve(self,targets,method="auto"):
        if self.teacher is None:
            raise RuntimeError("Call set_plate(Plate(...)) before solve()")
        if method not in ("auto","neural","direct"):
            raise ValueError("method must be auto, neural, or direct")
        targets = np.atleast_2d(np.asarray(targets,float))
        self._sync()
        started = time.perf_counter()
        b = self._tensor(self.teacher.incident(targets))
        in_domain = bool(self._in_domain(targets))
        candidate = None
        used = "direct"
        x = None
        if method=="neural" and self.model is None:
            raise RuntimeError("No trained checkpoint available")
        if method=="neural" or (method=="auto" and self.model is not None and in_domain):
            if self.device=="cuda":
                rhs = (self.a.mH@b)/self.bound
                key = (tuple(self.normal.shape),tuple(rhs.shape))
                if key not in self._graphs:
                    if len(self._graphs)>=4:
                        self._graphs.pop(next(iter(self._graphs)))
                    self._graphs[key] = CapturedBoundaryNet(self.model.forward_normal,self.normal,rhs)
                x = self._graphs[key](self.normal,rhs)
            else:
                x = self.model(self.a,b)
            candidate = float(relative_residual(self.a,x,b).max())
            used = "neural"
            if method=="auto" and (not np.isfinite(candidate) or candidate>self.tolerance):
                x = None
                used = "direct_fallback"
        if x is None:
            x = self._direct(b)
        residual = float(relative_residual(self.a,x,b).max())
        if not np.isfinite(residual) or (used!="neural" and residual>1e-4):
            raise RuntimeError(f"Numerical solve failed stationarity check: {residual:g}")
        fit = float((torch.linalg.vector_norm(self.a@x-b,dim=0)
                     /torch.linalg.vector_norm(b,dim=0)).max())
        direct = green(self.aperture_points.reshape(-1,3),targets,self.teacher.k).reshape(
            len(self.receivers),len(self.aperture_weights),len(targets))
        direct = np.einsum("cqt,q->ct",direct,self.aperture_weights)/2
        direct = torch.as_tensor(direct,
                                 device=self.device,dtype=torch.complex64)
        h = (direct+self.d@x/2).T.cpu().numpy()
        phases = np.mod(-np.angle(h),2*np.pi)
        codes = (np.floor(phases*(32/(2*np.pi))+.5).astype(np.int32)%32).astype(np.uint8)
        self._sync()
        elapsed = (time.perf_counter()-started)*1000
        return Solution(h,phases,codes,used,residual,fit,in_domain,self.setup_ms,elapsed,candidate)
