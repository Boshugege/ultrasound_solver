"""End-to-end differentiable rigid-acoustic BEM in PyTorch.

This path is for inversion/training. The fused CuPy path remains the realtime
inference implementation. Time convention: exp(-i*omega*t), outgoing exp(ikr).
"""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from numpy.polynomial.legendre import leggauss
from .geometry import _unit_triangles, array_centers
from .aperture import aperture_points, sinc_aperture_points
from .model import BoundaryNet


def _real_dtype(complex_dtype):
    return torch.float64 if complex_dtype == torch.complex128 else torch.float32


def triangle_rule(order, device, dtype):
    x, w = leggauss(order)
    u, v = np.meshgrid((x+1)/2, (x+1)/2, indexing="ij")
    bary = np.stack((1-u, u*(1-v), u*v), -1).reshape(-1, 3)
    weights = (w[:, None]*w[None, :]/2*u).ravel()
    return (torch.as_tensor(bary, device=device, dtype=dtype),
            torch.as_tensor(weights, device=device, dtype=dtype))


def surface_geometry(triangles):
    edges1 = triangles[:, 1]-triangles[:, 0]
    edges2 = triangles[:, 2]-triangles[:, 0]
    cross = torch.linalg.cross(edges1, edges2)
    twice_area = torch.linalg.vector_norm(cross, dim=-1).clamp_min(1e-12)
    return {
        "triangles": triangles,
        "centers": triangles.mean(1),
        "normals": cross/twice_area[:, None],
        "areas": twice_area/2,
    }


def green(points, sources, k, image=True):
    """Baffled monopole exp(ikr)/r. All arguments remain in the autograd graph."""
    delta = points[:, None, :]-sources[None, :, :]
    distance = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1e-9)
    value = torch.exp(1j*k*distance)/distance
    if image:
        sign = points.new_tensor((1., 1., -1.))
        delta = points[:, None, :]-sources[None, :, :]*sign
        distance = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1e-9)
        value = value+torch.exp(1j*k*distance)/distance
    return value


def double_layer(points, triangles, k, order=6, image=True, point_chunk=32):
    """Fixed-quadrature differentiable integral of dG/dn_y."""
    geo = surface_geometry(triangles)
    bary, weights = triangle_rule(order, triangles.device, triangles.dtype)
    samples = torch.einsum("qa,nad->nqd", bary, triangles)
    normals, areas = geo["normals"], geo["areas"]
    sign = triangles.new_tensor((1., 1., -1.))
    chunks = []

    def kernel(delta, normal):
        distance = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1e-9)
        dot = (delta*normal).sum(-1)
        # Direct self panel is a planar principal value and therefore zero.
        return (torch.exp(1j*k*distance)*(1-1j*k*distance)*dot
                /(4*np.pi*distance**3))

    for start in range(0, len(points), point_chunk):
        p = points[start:start+point_chunk]
        value = kernel(p[:, None, None]-samples[None],
                       normals[None, :, None])
        if image:
            value = value+kernel(p[:, None, None]-samples[None]*sign,
                                 normals[None, :, None]*sign)
        chunks.append((value*weights).sum(-1)*areas)
    return torch.cat(chunks)


def quaternion_matrix(quaternion):
    q = quaternion/torch.linalg.vector_norm(quaternion).clamp_min(1e-12)
    x, y, z, w = q.unbind()
    return torch.stack((
        1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y),
    )).reshape(3, 3)


class DifferentiablePlate(nn.Module):
    """Positive side/thickness plus unconstrained center and xyzw quaternion."""

    def __init__(self, side=.03, thickness=.004, center=(.08, .08, .055),
                 quaternion=(0, 0, 0, 1), subdivisions=10, thickness_layers=4,
                 chief_points=24, dtype=torch.float64, device="cuda"):
        super().__init__()
        self.log_side = nn.Parameter(torch.tensor(np.log(side), dtype=dtype, device=device))
        self.log_thickness = nn.Parameter(torch.tensor(np.log(thickness), dtype=dtype, device=device))
        self.center = nn.Parameter(torch.tensor(center, dtype=dtype, device=device))
        self.quaternion = nn.Parameter(torch.tensor(quaternion, dtype=dtype, device=device))
        unit = _unit_triangles(subdivisions, thickness_layers)
        self.register_buffer("unit_triangles", torch.tensor(unit, dtype=dtype, device=device))
        # Interior coordinates use the same normalized local convention.
        from .geometry import Plate
        p = Plate(1., 1., (0., 0., 1.)).interior_points(chief_points)
        self.register_buffer("unit_chief", torch.tensor(p-(0., 0., 1.),
                                                        dtype=dtype, device=device))

    def geometry(self):
        scale = torch.stack((self.log_side.exp(), self.log_side.exp(),
                             self.log_thickness.exp()))
        rotation = quaternion_matrix(self.quaternion)
        triangles = (self.unit_triangles*scale)@rotation.mT+self.center
        chief = (self.unit_chief*scale)@rotation.mT+self.center
        return triangles, chief


class DifferentiableBEM(nn.Module):
    """Differentiable geometry -> complex channel response."""

    def __init__(self, receivers=None, diameter=.009, frequency=40000.,
                 sound_speed=346., aperture_profile="uniform", quadrature_order=6,
                 checkpoint=None, dtype=torch.float64, device="cuda"):
        super().__init__()
        self.k = 2*np.pi*frequency/sound_speed
        self.order = int(quadrature_order)
        self.complex_dtype = (torch.complex128 if dtype == torch.float64
                              else torch.complex64)
        centers = array_centers() if receivers is None else np.asarray(receivers, float)
        if aperture_profile == "sinc":
            points, weights = sinc_aperture_points(centers, diameter)
        elif aperture_profile == "uniform":
            points, weights = aperture_points(centers, diameter)
        else:
            raise ValueError("aperture_profile must be sinc or uniform")
        self.channels, self.aperture_count = points.shape[:2]
        self.register_buffer("aperture", torch.tensor(points.reshape(-1, 3),
                                                       dtype=dtype, device=device))
        self.register_buffer("weights", torch.tensor(weights, dtype=dtype, device=device))
        path = Path(checkpoint) if checkpoint else Path(__file__).with_name(
            "artifacts")/"boundary_net.pt"
        self.network = None
        if path.is_file():
            saved = torch.load(path, map_location=device, weights_only=True)
            self.network = BoundaryNet(saved["layers"]).to(device)
            self.network.load_state_dict(saved["state_dict"])

    def assemble(self, triangles, chief):
        geo = surface_geometry(triangles)
        K = double_layer(geo["centers"], triangles, self.k, self.order)
        square = .5*torch.eye(len(triangles), dtype=K.dtype,
                             device=triangles.device)-K
        chief_rows = double_layer(chief, triangles, self.k, self.order)
        return torch.cat((square, chief_rows)), geo

    def solve_density(self, a, b, method):
        if method == "neural":
            if self.network is None:
                raise RuntimeError("Neural checkpoint is unavailable")
            return self.network(a, b)
        if method != "direct":
            raise ValueError("method must be neural or direct")
        ah = a.mH
        normal = ah@a
        regularizer = torch.finfo(normal.real.dtype).eps*normal.abs().amax()
        return torch.linalg.solve(
            normal+regularizer*torch.eye(len(normal), dtype=normal.dtype,
                                          device=normal.device), ah@b)

    def forward(self, triangles, chief, targets, method="direct"):
        targets = torch.as_tensor(targets, dtype=triangles.dtype,
                                  device=triangles.device)
        a, geo = self.assemble(triangles, chief)
        b = torch.cat((green(geo["centers"], targets, self.k),
                       -green(chief, targets, self.k)))
        density = self.solve_density(a, b, method)
        propagation = double_layer(self.aperture, triangles, self.k, self.order)
        propagation = (propagation.reshape(self.channels, self.aperture_count, -1)
                       *self.weights[None, :, None]).sum(1)
        direct = green(self.aperture, targets, self.k).reshape(
            self.channels, self.aperture_count, len(targets))
        direct = (direct*self.weights[None, :, None]).sum(1)/2
        response = (direct+propagation@density/2).mT
        residual = (torch.linalg.vector_norm(a.mH@(a@density-b), dim=0)
                    /torch.linalg.vector_norm(a.mH@b, dim=0).clamp_min(1e-12))
        return response, residual

    def plate_response(self, plate, targets, method="direct"):
        return self(*plate.geometry(), targets, method)


def scale_invariant_complex_loss(predicted, observed):
    """Complex L2 after eliminating one unknown global complex calibration."""
    scale = (predicted.conj()*observed).sum()/predicted.abs().square().sum().clamp_min(1e-12)
    return ((scale*predicted-observed).abs().square().sum()
            /observed.abs().square().sum().clamp_min(1e-12))
