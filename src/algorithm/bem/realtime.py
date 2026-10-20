"""Per-frame focusing phases for moving rigid obstacles (reciprocity + warm starts).

Reciprocity for the sinc Neumann source: with q the field of a unit monopole at
the target (baffle, apertures and obstacle all rigid),

    p_r(target) = 2 pi int f_a(s) q(s) dS      over aperture r,

so one BEM solve per target and frame gives all responses G[target, :] and the
response-alignment phases -arg(G). q is the monopole with its images (evaluated
with a fine aperture quadrature, cheap and accurate for near targets) plus the
obstacle's scattered field, which is smooth on the apertures and uses a coarse
quadrature. Rigid motion of the same mesh keeps the body's own matrix block; the
next solve for the same focus slot (the target may move) starts from the previous
surface field corrected by the change of the incident field at each element.
"""

import time

import cupy as cp
import numpy as np

from algorithm.bem.solver import RigidBEM, aperture_sources


class Apertures:
    """Quadrature of every aperture: points (M*Q, 3) and weights 2 pi w_q on the GPU."""

    def __init__(self, centers, diameter, orders):
        points, weights, self.per_aperture = aperture_sources(centers, diameter, *orders)
        self.count = len(centers)
        self.points = cp.asarray(points)
        self.weights = cp.asarray(weights)

    def integrate(self, values):
        return (values * self.weights).reshape(self.count, self.per_aperture).sum(axis=1)


def aperture_response(bem, target, direct, scattered, rtol, x0=None, incident=None, timings=None):
    """Response of ``target`` to every aperture at unit drive (field unit 1/m).

    ``incident`` optionally passes the monopole's (value, gradient) on the surface.
    Returns (row, surface density u or None, GMRES info or None)."""
    target = np.asarray(target, float)
    clock = _Clock(timings)
    row = direct.integrate(bem.point_field(direct.points, target))
    clock("direct")
    if bem.n == 0:
        return row, None, None
    incident, gradient = bem.point_source(bem.geo["centroids"], target) if incident is None else incident
    b = bem.rhs(incident, gradient)
    clock("incident")
    u, info = bem.solve(b, x0=x0, rtol=rtol)
    clock("solve")
    row = row + scattered.integrate(bem.scattered(scattered.points, u))
    clock("scattered")
    return row, u, info


class _Clock:
    """Accumulates synchronised GPU wall time per stage into ``timings`` (if given)."""

    def __init__(self, timings):
        self.timings = timings
        self.last = self._now() if timings is not None else None

    @staticmethod
    def _now():
        cp.cuda.Stream.null.synchronize()
        return time.perf_counter()

    def __call__(self, name):
        if self.timings is not None:
            now = self._now()
            self.timings[name] = self.timings.get(name, 0.0) + now - self.last
            self.last = now


class FocusSession:
    """Real-time response rows for a moving obstacle.

    Defaults are the measured real-time settings (``bem_study/speed_profile.py``, hand
    at lambda/4): GMRES rtol 1e-2, 3 x 8 aperture quadrature for the scattered term,
    512-element block-Jacobi preconditioner, float16 matrix storage; the phases reach
    99.999 % of the optimal focus of a tight solve.
    """

    def __init__(self, transducer_centers, diameter, k, planes=((2, 0.0),), rtol=1e-2,
                 direct_orders=(32, 128), scattered_orders=(3, 8), block_size=512, half_precision=True):
        self.k = float(k)
        self.block_size = block_size
        self.half_precision = half_precision
        self.planes = tuple(planes)
        self.rtol = float(rtol)
        self.centers = np.asarray(transducer_centers, float)
        self.direct = Apertures(self.centers, diameter, direct_orders)
        self.scattered = Apertures(self.centers, diameter, scattered_orders)
        self.bem = None
        self._previous = {}
        self.timings = {}

    def set_surface(self, vertices, faces, rigid_motion=False):
        """New obstacle surface. ``rigid_motion=True`` if it is the previous mesh moved
        rigidly (same faces): the body's own matrix block is reused."""
        if self.bem is None:
            self.bem = RigidBEM(vertices, faces, k=self.k, planes=self.planes, block_size=self.block_size,
                                half_precision=self.half_precision)
            rigid_motion = False
        else:
            same = rigid_motion and self.bem.n == len(faces)
            self.bem.update_geometry(vertices, faces, reuse_shape=same)
            if not same:
                self._previous.clear()
        cp.cuda.Stream.null.synchronize()
        started = time.perf_counter()
        self.bem.assemble()
        self.timings = {"geometry": self.bem.geometry_seconds,
                        "assemble": time.perf_counter() - started}

    def response_row(self, target, slot=0):
        """Complex response of the target point to each transducer (unit drive, 1/m).
        ``slot`` identifies the focus across frames for the warm start."""
        bem, target = self.bem, np.asarray(target, float)
        timings = {}
        clock = _Clock(timings)
        x0, incident = None, None
        if bem.n:
            incident = bem.point_source(bem.geo["centroids"], target)
            if slot in self._previous and self._previous[slot][0].shape == incident[0].shape:
                u_old, value_old = self._previous[slot]
                x0 = (u_old * (incident[0] / value_old)).astype(cp.complex64)
        clock("incident")
        row, u, info = aperture_response(bem, target, self.direct, self.scattered, self.rtol, x0,
                                         incident, timings)
        if u is not None:
            self._previous[slot] = (u, incident[0])
        self.timings.update(timings, iterations=0 if info is None else info.iterations)
        return cp.asnumpy(row)

    def phases(self, targets):
        """Response-alignment phases (rad) for one or several targets."""
        targets = np.atleast_2d(np.asarray(targets, float))
        combined = sum(np.conj(self.response_row(target, slot))
                       for slot, target in enumerate(targets))
        return np.mod(np.angle(combined), 2 * np.pi)
