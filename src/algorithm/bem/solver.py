"""Rigid-scatterer BEM above a rigid baffle (z = 0), Burton-Miller formulation.

Total pressure u = u_inc + u_scat outside a rigid body with surface S and unit
normal n pointing into the air. The representation formula

    u(x) = u_inc(x) + int_S u(y) dG_h(x, y)/dn_y dS_y

uses the half-space Green's function G_h(x, y) = G(x, y) + G(x, y*) (y* = y
mirrored across z = 0), which already satisfies the baffle condition, so only
the obstacle surface is discretised. On S the field satisfies

    (1/2 - K) u = u_inc          and          H u = -du_inc/dn,

and Burton-Miller solves (1/2 - K - alpha H) u = u_inc + alpha du_inc/dn with
alpha = i / k, which removes the spurious interior resonances that make the
first equation alone ill-posed at the dense eigenfrequencies of a hand-sized body.

Sources are upstream's sinc Neumann apertures (``aperture_sources``) or point
sources (reciprocity). Obstacles are rigid: at 40 kHz in air the default
obstacle material (rho 1250, c 2200) differs from a rigid body by ~0.1 %.
"""

from dataclasses import dataclass
import time

import cupy as cp
import numpy as np

from algorithm.bem import kernels
from algorithm.bem.krylov import gmres

def geometry_arrays(vertices, faces, xp=np) -> dict:
    """Per-triangle data; faces must be oriented so normals point into the air."""
    tri = xp.asarray(vertices, dtype=xp.float64)[xp.asarray(faces)]  # (n, 3, 3)
    cross = xp.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = xp.linalg.norm(cross, axis=1)
    edges = xp.stack([tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 1], tri[:, 0] - tri[:, 2]], 1)
    return {
        "n": int(tri.shape[0]),
        "triangles": tri,
        "centroids": tri.mean(axis=1),
        "normals": cross / norm[:, None],
        "areas": 0.5 * norm,
        "hsize": xp.linalg.norm(edges, axis=2).max(axis=1),
    }


def self_hypersingular(tri, k: float, angular=16, radial=8, xp=np):
    """Finite-part integral of d2G/dn_x dn_y over each flat element at its centroid.

    In the element plane the kernel is exp(ikr)(1 - ikr) / (4 pi r^3). In polar
    coordinates around the centroid, FP int_0^R dr / r^2 = -1/R, and the
    remainder (exp(ikr)(1 - ikr) - 1) / r^2 is smooth (-> k^2 / 2).
    The angle integral is taken edge by edge: dtheta = dist * L / R(t)^2 dt.
    Depends only on each element's shape, so it is invariant under rigid motion.
    """
    x = tri.mean(axis=1)
    t, wt = np.polynomial.legendre.leggauss(angular)
    t, wt = xp.asarray(0.5 * (t + 1)), xp.asarray(0.5 * wt)
    s, ws = np.polynomial.legendre.leggauss(radial)
    s, ws = xp.asarray(0.5 * (s + 1)), xp.asarray(0.5 * ws)
    total = xp.zeros(tri.shape[0], dtype=complex)
    for a, b in ((0, 1), (1, 2), (2, 0)):
        va, vb = tri[:, a], tri[:, b]
        edge = vb - va
        length = xp.linalg.norm(edge, axis=1)
        points = va[:, None] + t[None, :, None] * edge[:, None]  # (n, angular, 3)
        R = xp.linalg.norm(points - x[:, None], axis=2)
        dist = xp.linalg.norm(xp.cross(edge, x - va), axis=1) / length
        dtheta = (dist * length)[:, None] / R**2
        r = R[..., None] * s  # (n, angular, radial)
        kr = k * r
        remainder = (xp.exp(1j * kr) * (1 - 1j * kr) - 1) / r**2
        inner = (remainder * ws).sum(axis=2) * R
        total += ((-1.0 / R + inner) * dtheta * wt).sum(axis=1)
    return total / (4.0 * np.pi)


HALF_SCALE = 1024.0  # matrix entries are O(1); keeps small entries out of float16 subnormals


@dataclass
class SolveInfo:
    iterations: int
    seconds: float
    converged: bool


def image_group(planes):
    """Reflections across axis-aligned rigid planes [(axis, position), ...], at most one
    per axis (perpendicular planes keep the image method exact). Returns (signs, offsets)
    for every combination, identity first: y -> sign * y + offset."""
    axes = [axis for axis, _ in planes]
    if len(set(axes)) != len(axes):
        raise ValueError("at most one reflecting plane per axis (parallel walls need an "
                         "infinite image series)")
    signs, offsets = [], []
    for mask in range(2 ** len(planes)):
        sign, offset = np.ones(3), np.zeros(3)
        for bit, (axis, position) in enumerate(planes):
            if mask >> bit & 1:
                sign[axis], offset[axis] = -1.0, 2.0 * position
        signs.append(sign)
        offsets.append(offset)
    return np.array(signs), np.array(offsets)


def spatial_blocks(points, size):
    """Index sets of at most ``size`` neighbouring points (recursive median bisection)."""
    blocks = []

    def split(index):
        if len(index) <= size:
            blocks.append(index)
            return
        coordinates = points[index]
        order = index[np.argsort(coordinates[:, np.argmax(np.ptp(coordinates, axis=0))])]
        split(order[: len(order) // 2])
        split(order[len(order) // 2:])

    split(np.arange(len(points)))
    return blocks


class BlockJacobi:
    """Inverse of the near-field diagonal blocks of a dense matrix (right preconditioner).

    Elements are grouped into spatially compact blocks; ``solve`` applies the inverse
    of the block diagonal, ``apply`` the block diagonal itself (to map a warm start
    into the preconditioned variable)."""

    def __init__(self, A, centroids, size=128):
        blocks = spatial_blocks(np.asarray(centroids), size)
        width = max(len(b) for b in blocks)
        n = A.shape[0]
        index = np.full((len(blocks), width), n, dtype=np.int64)  # n = padding slot
        for row, b in enumerate(blocks):
            index[row, :len(b)] = b
        self.index = cp.asarray(index)
        valid = self.index < n
        rows = cp.where(valid, self.index, 0)
        D = A[rows[:, :, None], rows[:, None, :]]
        mask = valid[:, :, None] & valid[:, None, :]
        D = cp.where(mask, D, 0)
        D[:, cp.arange(width), cp.arange(width)] += cp.where(valid, 0, 1).astype(D.dtype)
        self.blocks = D.astype(cp.complex64)
        self.inverses = cp.linalg.inv(D).astype(cp.complex64)
        self.n = n

    def _multiply(self, matrices, v):
        padded = cp.concatenate([v, cp.zeros(1, dtype=v.dtype)])
        out = cp.zeros(self.n + 1, dtype=v.dtype)
        out[self.index] = cp.matmul(matrices, padded[self.index][..., None])[..., 0]
        return out[: self.n]

    def solve(self, v):
        return self._multiply(self.inverses, v)

    def apply(self, v):
        return self._multiply(self.blocks, v)


class RigidBEM:
    """Rigid bodies (one closed surface or several) above/beside rigid planes.

    ``planes`` defaults to the transducer baffle z = 0. ``planes=()`` is free space.
    """

    def __init__(self, vertices, faces, k, planes=((2, 0.0),), alpha=None, block_size=None,
                 half_precision=False):
        self.k = float(k)
        self.block_size = block_size  # near-field block-Jacobi preconditioner, None = off
        self.preconditioner = None
        # Keep the matrix only as float16 real/imaginary parts (scaled by HALF_SCALE) for
        # GMRES: half the memory traffic per product (for rtol >= 1e-3).
        self.half_precision = half_precision
        self._half = None
        self._assembled = False
        self.planes = tuple(planes)
        self.images = image_group(self.planes)
        self.alpha = 1j / self.k if alpha is None else complex(alpha)
        self.update_geometry(vertices, faces)

    def transform_points(self, points):
        """All image copies of host points (Q, 3) -> (images * Q, 3)."""
        signs, offsets = self.images
        points = np.atleast_2d(np.asarray(points, float))
        return np.concatenate([points * sign + offset for sign, offset in zip(signs, offsets)])

    def update_geometry(self, vertices, faces, reuse_shape=False):
        """Upload a new surface. ``reuse_shape=True`` means the surface moved rigidly:
        the self terms and the obstacle-only matrix are kept (they are invariant)."""
        cp.cuda.Stream.null.synchronize()
        started = time.perf_counter()
        faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
        geo = geometry_arrays(cp.asarray(np.asarray(vertices, float).reshape(-1, 3)),
                              cp.asarray(faces), xp=cp)
        self.geo = geo
        f32 = lambda a: cp.ascontiguousarray(a.astype(cp.float32).ravel())  # noqa: E731
        self.device = {
            "n": geo["n"], "centroids": f32(geo["centroids"]), "normals": f32(geo["normals"]),
            "vertices": f32(geo["triangles"]), "areas": f32(geo["areas"]),
            "hsize": f32(geo["hsize"]),
        }
        if not reuse_shape:
            self.hself = self_hypersingular(geo["triangles"], self.k, xp=cp)
            self.A_obstacle = None
            self.preconditioner = None
        self.A = None
        self._assembled = False
        cp.cuda.Stream.null.synchronize()
        self.geometry_seconds = time.perf_counter() - started

    @property
    def n(self):
        return self.geo["n"]

    def assemble(self):
        """Matrix; for a rigidly moved surface only the image part is recomputed (and
        written on top of the body's own block in one pass)."""
        cp.cuda.Stream.null.synchronize()
        started = time.perf_counter()
        half = HALF_SCALE if self.half_precision else None
        previous = self._half if half else getattr(self, "_buffer", None)
        if self.n == 0:
            self.A, self._half = cp.zeros((0, 0), dtype=cp.complex64), None
        elif len(self.images[0]) == 1:
            self.A = kernels.assemble(self.device, self.k, self.alpha, self.hself, self.images)
            if half:  # free space: keep A (preconditioner) and add the float16 copy
                self._half = ((self.A.real * half).astype(cp.float16), (self.A.imag * half).astype(cp.float16))
        else:
            if self.A_obstacle is None:
                self.A_obstacle = kernels.assemble(self.device, self.k, self.alpha, self.hself,
                                                   self.images, 0, 1)
            shape = None if previous is None else (previous[0] if half else previous).shape
            reuse = previous if shape == self.A_obstacle.shape else None
            if half:
                self.A = None
                self._half = kernels.assemble(self.device, self.k, self.alpha, self.hself, self.images, 1,
                                              base=self.A_obstacle, out=reuse, half_scale=half)
            else:
                self.A = self._buffer = kernels.assemble(self.device, self.k, self.alpha, self.hself,
                                                         self.images, 1, base=self.A_obstacle, out=reuse)
        if self.block_size and self.preconditioner is None and self.n:
            # the body's own blocks are invariant under rigid motion
            own = self.A if self.A_obstacle is None else self.A_obstacle
            self.preconditioner = BlockJacobi(own, cp.asnumpy(self.geo["centroids"]), self.block_size)
        self._assembled = True
        cp.cuda.Stream.null.synchronize()
        self.assemble_seconds = time.perf_counter() - started
        return self.A

    def apply(self, v):
        """Product with the Burton-Miller matrix (float16 storage if enabled)."""
        if self._half is None:
            return self.A @ v
        real, imag = self._half
        X = cp.stack([v.real, v.imag], axis=1).astype(cp.float16)
        P, Q = real @ X, imag @ X
        return ((P[:, 0].astype(cp.float32) - Q[:, 1]) + 1j * (P[:, 1].astype(cp.float32) + Q[:, 0])) \
            .astype(cp.complex64) / HALF_SCALE

    # ---- incident fields -------------------------------------------------
    def _green_and_gradient(self, x, sources, weights):
        """sum_q w_q G(x, s_q) and its gradient in x; x (P,3), sources (Q,3)."""
        return kernels.monopoles(x, sources, weights, self.k)

    def point_source(self, x, source):
        """Monopole at ``source`` plus its images in the rigid planes."""
        sources = cp.asarray(self.transform_points(source))
        return self._green_and_gradient(x, sources, cp.ones(len(sources), dtype=cp.complex128))

    def point_field(self, x, source, chunk=1 << 20):
        """Value only of the monopole at ``source`` plus its images, for many points."""
        sources = cp.asarray(self.transform_points(source))
        x = cp.asarray(x)
        out = cp.empty(len(x), dtype=cp.complex128)
        for start in range(0, len(x), chunk):
            r = cp.linalg.norm(x[start:start + chunk, None, :] - sources[None], axis=2)
            out[start:start + chunk] = (cp.exp(1j * self.k * r) / (4 * np.pi * r)).sum(axis=1)
        return out

    def sources(self, x, points, weights):
        """Weighted monopoles at ``points`` plus their images, value and gradient at x
        (a point in the baffle coincides with its baffle image, giving 2 G there)."""
        count = len(self.images[0])
        sources = cp.asarray(self.transform_points(cp.asnumpy(cp.asarray(points))))
        return self._green_and_gradient(x, sources, cp.tile(cp.asarray(weights), count))

    def rhs(self, value, gradient):
        normals = self.geo["normals"]
        return (value + self.alpha * (gradient * normals).sum(axis=1)).astype(cp.complex64)

    def incident_on_surface(self, fn):
        return self.rhs(*fn(self.geo["centroids"]))

    # ---- solve and evaluate ---------------------------------------------
    def solve(self, b, x0=None, rtol=1e-5, restart=100, maxiter=500, check_every=1):
        if not self._assembled:
            self.assemble()
        product = self.apply
        cp.cuda.Stream.null.synchronize()
        started = time.perf_counter()
        P = self.preconditioner
        if P is None:
            result = gmres(product, b, x0=x0, rtol=rtol, restart=restart,
                           maxiter=maxiter, xp=cp, check_every=check_every)
            solution = result.solution
        else:  # right preconditioning: A P^-1 y = b, x = P^-1 y
            result = gmres(lambda y: product(P.solve(y)), b, x0=None if x0 is None else P.apply(x0),
                           rtol=rtol, restart=restart, maxiter=maxiter, xp=cp, check_every=check_every)
            solution = P.solve(result.solution)
        cp.cuda.Stream.null.synchronize()
        return solution, SolveInfo(result.iterations, time.perf_counter() - started, result.converged)

    def scattered(self, points, u):
        """Scattered field at points for one surface density ``u``."""
        if self.n == 0:
            return cp.zeros(len(points), dtype=cp.complex128)
        return kernels.evaluate(self.device, points, u, self.k, self.images)

    def scattered_many(self, points, U, chunk=8192):
        """Scattered fields at points for several densities U (n, m): chunked D @ U."""
        points = cp.asarray(points)
        if self.n == 0:
            return cp.zeros((len(points), U.shape[1]), dtype=cp.complex64)
        out = []
        for start in range(0, len(points), chunk):
            D = kernels.evaluation_matrix(self.device, points[start:start + chunk], self.k,
                                          self.images)
            out.append(D @ U)
        return cp.concatenate(out)


def icosphere(radius=1.0, subdivisions=3, center=(0.0, 0.0, 0.0)):
    """Outward-oriented triangulated sphere with 20 * 4^s faces."""
    phi = (1 + 5**0.5) / 2
    verts = [(-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0), (0, -1, phi), (0, 1, phi),
             (0, -1, -phi), (0, 1, -phi), (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1)]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4),
             (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8),
             (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    verts = [np.array(v, float) / np.linalg.norm(v) for v in verts]
    for _ in range(subdivisions):
        cache, new_faces = {}, []

        def midpoint(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = verts[a] + verts[b]
                verts.append(m / np.linalg.norm(m))
                cache[key] = len(verts) - 1
            return cache[key]

        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new_faces
    vertices = np.array(verts) * radius + np.asarray(center, float)
    faces = np.array(faces, dtype=np.int64)
    tri = vertices[faces]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    outward = (normal * (tri.mean(1) - np.asarray(center))).sum(1) > 0
    faces[~outward] = faces[~outward][:, ::-1]
    return vertices, faces


def rigid_sphere_point_source(points, source, center, radius, k, terms=None):
    """Exact total field of a monopole G(x, source) scattered by a rigid sphere (free space)."""
    from scipy.special import eval_legendre, spherical_jn, spherical_yn

    points = np.atleast_2d(points) - np.asarray(center)
    s = np.asarray(source, float) - np.asarray(center)
    r, r0 = np.linalg.norm(points, axis=1), np.linalg.norm(s)
    cos = points @ s / (r * r0)
    ka = k * radius
    terms = terms or int(ka + 10 * ka ** (1 / 3) + 20)
    total = np.zeros(len(points), dtype=complex)
    for order in range(terms):
        hn = lambda z: spherical_jn(order, z) + 1j * spherical_yn(order, z)  # noqa: E731
        dj = spherical_jn(order, ka, derivative=True)
        dh = dj + 1j * spherical_yn(order, ka, derivative=True)
        rmin, rmax = np.minimum(r, r0), np.maximum(r, r0)
        incident = spherical_jn(order, k * rmin) * hn(k * rmax)
        scattered = -(dj / dh) * hn(k * r0) * hn(k * r)
        total += (2 * order + 1) * (incident + scattered) * eval_legendre(order, cos)
    return 1j * k / (4 * np.pi) * total


def aperture_sources(centers, diameter, radial_order, angular_order):
    """Upstream's sinc Neumann apertures as weighted monopoles in the baffle.

    ``physics/sinc_aperture.py`` prescribes the outward flux (1/rho) dp/dn =
    (2 pi / rho) f_a on each aperture, f_a = 1 / (2 pi a sqrt(a^2 - r^2)), which
    radiates p(x) = int f_a(s) exp(ik|x - s|) / |x - s| dS per unit drive (1/m).
    On the baffle G_h = 2 G = exp(ikR) / (2 pi R), so p = sum_q (2 pi w_q) G_h(x, s_q)
    over upstream's aperture quadrature (unit total weight, rim singularity removed
    by mu = cos(theta)). Returns points (M*Q, 3), weights (M*Q,) and Q.
    """
    from algorithm.physics.sinc_aperture import aperture_quadrature

    offsets, weights = aperture_quadrature(diameter, radial_order, angular_order)
    centers = np.asarray(centers, float)
    points = (centers[:, None, :] + offsets[None]).reshape(-1, 3)
    return points, np.tile(2.0 * np.pi * weights, len(centers)), len(weights)
