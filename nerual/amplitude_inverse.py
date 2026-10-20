"""Amplitude-only inversion of a rigid triangular prism with a differentiable BEM.

Setting of AMPLITUDE_INVERSE_VISUAL_REPORT.md: a 3 x 3 array (pitch 10.5 mm,
diameter 9 mm, centred at (24, 24, 0) mm in a rigid baffle) emits known patterns,
and only the pressure amplitude is observed at 16 x 16 points on each of the faces
x = 6, 42 mm, y = 6, 42 mm and z = 42 mm of the region x, y in 6..42 mm,
z in 0..42 mm. Unknowns: prism centre (3), rotation vector (3), side and thickness.

Forward model, complex128 and differentiable in the vertices (PyTorch):
- incident field of the sinc Neumann apertures, sum_q w_q exp(ikr)/r per unit
  drive, with one right-hand side per emission pattern (not one per observation
  point as in DifferentiableBEM, which places the sources at the targets);
- rigid prism: (1/2 - K) u = u_inc at panel centroids with the half-space double
  layer. The prism is thinner than half a wavelength, so its interior has no
  Dirichlet resonance at 40 kHz and this equation alone is well posed (a thick
  body needs Burton-Miller or CHIEF);
- observed field p = u_inc + D u.
Element integrals use one, seven or near_order^2 points by distance, as in the
CUDA kernels of algorithm.bem.
"""

from dataclasses import dataclass
import time

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
import torch

from .aperture import sinc_aperture_points
from .differentiable import triangle_rule
from .geometry import _unit_triangles

FREQUENCY = 40000.0
SOUND_SPEED = 346.0
DIAMETER = 0.009
PITCH = 0.0105
ARRAY_CENTER = (0.024, 0.024)
REGION = ((0.006, 0.042), (0.006, 0.042), (0.0, 0.042))
FACE_POINTS = 16
FACES = ("-x", "+x", "-y", "+y", "+z")
# Optimiser variables are params / SCALE: centre in mm, rotation vector in 0.1 rad,
# log side and log thickness in steps of 5 %.
SCALE = np.array([1e-3, 1e-3, 1e-3, 0.1, 0.1, 0.1, 0.05, 0.05])
# Prism symmetries in its own frame (D3): turns about z and half turns about the
# three axes through a vertex and the opposite edge midpoint.
SYMMETRIES = Rotation.from_rotvec(np.vstack(
    [[0, 0, a] for a in (0.0, 2 * np.pi / 3, 4 * np.pi / 3)]
    + [np.pi * np.array([np.cos(a), np.sin(a), 0.0]) for a in (0.0, 2 * np.pi / 3, 4 * np.pi / 3)]))


def wavenumber(frequency=FREQUENCY, sound_speed=SOUND_SPEED):
    return 2 * np.pi * frequency / sound_speed


def array_centers():
    """The 3 x 3 sub-array of the report, x index outer."""
    offsets = (np.arange(3) - 1) * PITCH
    x, y = np.meshgrid(ARRAY_CENTER[0] + offsets, ARRAY_CENTER[1] + offsets, indexing="ij")
    return np.c_[x.ravel(), y.ravel(), np.zeros(9)]


def observation_points(n=FACE_POINTS):
    """Cell centres of an n x n grid on each face, in the order of FACES."""
    (x0, x1), (y0, y1), (z0, z1) = REGION

    def centres(a, b):
        return a + (np.arange(n) + 0.5) * (b - a) / n

    xs, ys, zs = centres(x0, x1), centres(y0, y1), centres(z0, z1)
    faces = []
    for x in (x0, x1):
        y, z = np.meshgrid(ys, zs, indexing="ij")
        faces.append(np.c_[np.full(y.size, x), y.ravel(), z.ravel()])
    for y in (y0, y1):
        x, z = np.meshgrid(xs, zs, indexing="ij")
        faces.append(np.c_[x.ravel(), np.full(x.size, y), z.ravel()])
    x, y = np.meshgrid(xs, ys, indexing="ij")
    faces.append(np.c_[x.ravel(), y.ravel(), np.full(x.size, z1)])
    return np.concatenate(faces)


def focus_patterns(points, k=None):
    """Drives (P, 9) that focus the 3 x 3 array at each point (free-field delays);
    the first pattern of the report is all ones (every phase 0)."""
    k = wavenumber() if k is None else k
    distance = np.linalg.norm(np.asarray(points, float)[:, None] - array_centers()[None], axis=-1)
    return np.exp(-1j * k * distance)


@dataclass(frozen=True)
class Prism:
    """Equilateral triangular prism: centre and side/thickness in metres, rotation
    vector in radians (the prism of nerual.geometry.Plate before rotation)."""

    center: tuple
    rotvec: tuple
    side: float
    thickness: float

    @classmethod
    def from_euler(cls, center_mm, euler_xyz_deg, side_mm, thickness_mm):
        """Extrinsic xyz Euler angles about the centroid, as in the report."""
        rotvec = Rotation.from_euler("xyz", euler_xyz_deg, degrees=True).as_rotvec()
        return cls(tuple(np.asarray(center_mm, float) * 1e-3), tuple(rotvec),
                   side_mm * 1e-3, thickness_mm * 1e-3)

    @classmethod
    def from_params(cls, params):
        p = np.asarray(params, float)
        return cls(tuple(p[:3]), tuple(p[3:6]), float(np.exp(p[6])), float(np.exp(p[7])))

    def params(self):
        return np.r_[self.center, self.rotvec, np.log(self.side), np.log(self.thickness)]

    @property
    def rotation(self):
        return Rotation.from_rotvec(self.rotvec)

    def triangles(self, n, m):
        local = _unit_triangles(n, m) * np.array([self.side, self.side, self.thickness])
        return local @ self.rotation.as_matrix().T + np.asarray(self.center)

    def contains(self, points):
        local = (np.asarray(points) - np.asarray(self.center)) @ self.rotation.as_matrix()
        angles = np.arange(3) * 2 * np.pi / 3 + np.pi / 3
        outward = np.c_[np.cos(angles), np.sin(angles)]
        return ((local[..., :2] @ outward.T <= self.side / (2 * np.sqrt(3))).all(-1)
                & (np.abs(local[..., 2]) <= self.thickness / 2))


def compare(true, estimate, voxel=0.25e-3):
    """Errors of an estimate: centre (m), side, thickness (m), rotation up to the
    prism's symmetry (deg) and volume IoU sampled on ``voxel`` cubes."""
    corners = np.vstack([true.triangles(1, 1).reshape(-1, 3), estimate.triangles(1, 1).reshape(-1, 3)])
    axes = [np.arange(lo, hi + voxel, voxel) for lo, hi in zip(corners.min(0), corners.max(0))]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    a, b = true.contains(grid), estimate.contains(grid)
    relative = true.rotation.inv() * estimate.rotation
    rotation = min((relative * g).magnitude() for g in SYMMETRIES)
    return {
        "center_error": float(np.linalg.norm(np.subtract(estimate.center, true.center))),
        "side_error": estimate.side - true.side,
        "thickness_error": estimate.thickness - true.thickness,
        "rotation_error_deg": float(np.degrees(rotation)),
        "iou": float((a & b).sum() / max((a | b).sum(), 1)),
    }


def _skew(v):
    zero = v.new_zeros(())
    return torch.stack((torch.stack((zero, -v[2], v[1])),
                        torch.stack((v[2], zero, -v[0])),
                        torch.stack((-v[1], v[0], zero))))


def prism_triangles(params, template):
    """World triangles (N, 3, 3) of a unit-prism template for params (8,)."""
    scale = torch.stack((params[6].exp(), params[6].exp(), params[7].exp()))
    rotation = torch.linalg.matrix_exp(_skew(params[3:6]))
    return (template * scale) @ rotation.mT + params[:3]


def _double_layer_kernel(delta, normal, k):
    """dG/dn_y for G = exp(ikr) / (4 pi r) and delta = x - y."""
    r = torch.linalg.vector_norm(delta, dim=-1)
    dot = (delta * normal).sum(-1)
    return torch.exp(1j * k * r) * (1 - 1j * k * r) * dot / (4 * np.pi * r**3)


def _half_space(points, samples, normals, k):
    """Kernel with the image in the baffle z = 0; points (..., 3), samples (..., Q, 3)."""
    flip = samples.new_tensor((1.0, 1.0, -1.0))
    return (_double_layer_kernel(points[..., None, :] - samples, normals[..., None, :], k)
            + _double_layer_kernel(points[..., None, :] - samples * flip, (normals * flip)[..., None, :], k))


# Dunavant's 7-point degree-5 rule (barycentric points, weights summing to one).
SEVEN_POINT = (
    np.array([[1 / 3, 1 / 3, 1 / 3],
              [0.0597158717, 0.4701420641, 0.4701420641], [0.4701420641, 0.0597158717, 0.4701420641],
              [0.4701420641, 0.4701420641, 0.0597158717], [0.7974269853, 0.1012865073, 0.1012865073],
              [0.1012865073, 0.7974269853, 0.1012865073], [0.1012865073, 0.1012865073, 0.7974269853]]),
    np.array([0.225] + [0.1323941527] * 3 + [0.1259391805] * 3),
)


def _rule_values(points, triangles, normals, areas, rows, cols, rule, k):
    bary, weights = rule
    samples = torch.einsum("qa,nad->nqd", bary, triangles[cols])
    return (_half_space(points[rows], samples, normals[cols], k) * weights).sum(-1) * areas[cols]


def double_layer(points, triangles, k, near_order=12, near_factor=2.0, middle_factor=4.0, skip_self=False):
    """(M, N) integrals of dG_h/dn_y over flat panels, with the image in the baffle.

    The rule depends on the distance d from the point to the panel centroid in
    panel diameters: one point at the centroid for d >= middle_factor + 1, the
    7-point rule from near_factor up to middle_factor, a near_order^2 Gauss-Duffy
    rule below near_factor (as in the CUDA kernels of algorithm.bem). Between
    middle_factor and middle_factor + 1 the one- and 7-point values are blended
    linearly in d, which keeps the loss continuous for the line search.
    ``skip_self`` leaves the flat panel's own principal value, which is zero, out.
    """
    edges1 = triangles[:, 1] - triangles[:, 0]
    cross = torch.linalg.cross(edges1, triangles[:, 2] - triangles[:, 0])
    areas = torch.linalg.vector_norm(cross, dim=-1) / 2
    normals = cross / (2 * areas[:, None])
    diameters = torch.stack([torch.linalg.vector_norm(triangles[:, i] - triangles[:, j], dim=-1)
                             for i, j in ((0, 1), (1, 2), (2, 0))]).amax(0)
    centroids = triangles.mean(1)
    real = dict(dtype=triangles.dtype, device=triangles.device)
    with torch.no_grad():
        ratio = torch.cdist(points, centroids) / diameters
        if skip_self:
            ratio.fill_diagonal_(-1.0)
    one_point = (torch.full((1, 3), 1 / 3, **real), torch.ones(1, **real))
    seven_point = tuple(torch.as_tensor(a, **real) for a in SEVEN_POINT)
    near_rule = triangle_rule(near_order, triangles.device, triangles.dtype)
    matrix = torch.zeros(ratio.shape, dtype=torch.complex128 if triangles.dtype == torch.float64
                         else torch.complex64, device=triangles.device)
    for mask, rule in ((ratio >= middle_factor + 1, one_point),
                       ((ratio >= near_factor) & (ratio < middle_factor + 1), seven_point),
                       ((ratio >= 0) & (ratio < near_factor), near_rule)):
        rows, cols = torch.nonzero(mask, as_tuple=True)
        if len(rows):
            matrix = matrix.index_put((rows, cols), _rule_values(points, triangles, normals, areas,
                                                                 rows, cols, rule, k))
    rows, cols = torch.nonzero((ratio >= middle_factor) & (ratio < middle_factor + 1), as_tuple=True)
    if len(rows):
        blend = (torch.linalg.vector_norm(points[rows] - centroids[cols], dim=-1) / diameters[cols]
                 - middle_factor)
        far = _rule_values(points, triangles, normals, areas, rows, cols, one_point, k)
        matrix = matrix.index_put((rows, cols), (1 - blend) * matrix[rows, cols] + blend * far)
    return matrix


def incident_columns(points, k, source_orders, chunk=256):
    """(M, 9) incident field of each aperture at unit drive, sum_q w_q exp(ikr)/r."""
    apertures, weights = sinc_aperture_points(array_centers(), DIAMETER, *source_orders)
    sources = points.new_tensor(apertures.reshape(-1, 3))
    w = torch.zeros(len(sources), 9, dtype=points.dtype, device=points.device)
    per = apertures.shape[1]
    for j in range(9):
        w[j * per:(j + 1) * per, j] = points.new_tensor(weights)
    out = []
    for start in range(0, len(points), chunk):
        r = torch.linalg.vector_norm(points[start:start + chunk, None] - sources[None], dim=-1)
        out.append((torch.exp(1j * k * r) / r) @ w.to(torch.complex128))
    return torch.cat(out)


@dataclass
class InverseResult:
    prism: Prism
    loss: float
    iterations: int
    evaluations: int
    seconds: float
    history: list


class PrismInverse:
    """Fit a prism to observed amplitudes (P patterns x 1280 points)."""

    def __init__(self, observed, patterns, panels_per_side=12, thickness_layers=2, near_order=12,
                 near_factor=2.0, middle_factor=4.0, source_orders=(8, 32),
                 observation_source_orders=(32, 128), device="cuda"):
        self.device = torch.device(device)
        self.k = wavenumber()
        self.rules = dict(near_order=near_order, near_factor=near_factor, middle_factor=middle_factor)
        self.source_orders = source_orders
        real = dict(dtype=torch.float64, device=self.device)
        self.template = torch.as_tensor(_unit_triangles(panels_per_side, thickness_layers).copy(), **real)
        self.points = torch.as_tensor(observation_points(), **real)
        self.patterns = torch.as_tensor(np.atleast_2d(patterns), dtype=torch.complex128, device=self.device)
        with torch.no_grad():
            incident = incident_columns(self.points, self.k, observation_source_orders)
            self.incident = incident @ self.patterns.T  # (M, P)
        observed = torch.as_tensor(np.atleast_2d(observed), **real)  # (P, M)
        self.observed = observed.reshape(len(observed), len(FACES), -1)
        self.norms = (self.observed**2).sum(-1)

    def field(self, params):
        """Complex pressure (P, M) at the observation points."""
        triangles = prism_triangles(params, self.template)
        centroids = triangles.mean(1)
        K = double_layer(centroids, triangles, self.k, skip_self=True, **self.rules)
        A = 0.5 * torch.eye(len(K), dtype=K.dtype, device=K.device) - K
        b = incident_columns(centroids, self.k, self.source_orders) @ self.patterns.T
        u = torch.linalg.solve(A, b)
        D = double_layer(self.points, triangles, self.k, **self.rules)
        return (self.incident + D @ u).T

    def loss(self, params):
        """Mean over patterns and faces of sum (|p| - a)^2 / sum a^2 (the report's L)."""
        amplitude = self.field(params).abs().reshape(self.observed.shape)
        return (((amplitude - self.observed)**2).sum(-1) / self.norms).mean()

    def value_and_grad(self, z):
        params = torch.as_tensor(z * SCALE, dtype=torch.float64, device=self.device).requires_grad_()
        loss = self.loss(params)
        loss.backward()
        return float(loss), params.grad.cpu().numpy() * SCALE

    def invert(self, start, maxiter=60, bounds=None):
        """L-BFGS-B from ``start`` (Prism); bounds (8 pairs, in params units) optional."""
        history = []
        started = time.perf_counter()

        def fun(z):
            value, grad = self.value_and_grad(z)
            history.append(value)
            return value, grad

        scaled = None if bounds is None else [
            tuple(None if v is None else v / s for v in pair) for pair, s in zip(bounds, SCALE)]
        result = minimize(fun, start.params() / SCALE, jac=True, method="L-BFGS-B", bounds=scaled,
                          options={"maxiter": maxiter})
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return InverseResult(Prism.from_params(result.x * SCALE), float(result.fun), int(result.nit),
                             int(result.nfev), time.perf_counter() - started, history)
