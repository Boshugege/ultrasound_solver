"""Memory-bounded forward solve of the existing homogeneous FV discretization.

This is an internal exact strategy for the homogeneous case. It diagonalizes
x/y, eliminates z and solves the corrected integrated sinc Neumann flux through
a surface Schur complement. No analytic propagator is used.
"""
import hashlib
from collections import OrderedDict

import numpy as np
import scipy.linalg as la

from algorithm.physics.conservative_helmholtz import ConservativeHelmholtz, FACES, boundary_type


_EIGEN_CACHE = OrderedDict()
_CUDA_Z_TRANSFER_KERNEL = None


def _cached_eigendecomposition(matrix):
    """Reuse identical x/y operators and repeated control-layer validation."""
    contiguous = np.ascontiguousarray(matrix)
    key = (contiguous.shape, contiguous.dtype.str,
           hashlib.sha256(contiguous.view(np.uint8)).digest())
    cached = _EIGEN_CACHE.get(key)
    if cached is not None:
        _EIGEN_CACHE.move_to_end(key)
        return cached
    eigenvalues, eigenvectors = la.eig(matrix)
    inverse = la.inv(eigenvectors)
    # The inverse is already required for the modal transform. Its 1-norm
    # condition number avoids a second cubic-cost SVD during control startup.
    condition = float(la.norm(eigenvectors, 1)*la.norm(inverse, 1))
    result = (eigenvalues, eigenvectors, inverse, condition)
    _EIGEN_CACHE[key] = result
    while len(_EIGEN_CACHE) > 4:
        _EIGEN_CACHE.popitem(last=False)
    return result


def _cuda_z_transfer(shift, z):
    """Run all z recurrences in one kernel instead of hundreds of launches."""
    global _CUDA_Z_TRANSFER_KERNEL
    import cupy as cp
    if _CUDA_Z_TRANSFER_KERNEL is None:
        _CUDA_Z_TRANSFER_KERNEL = cp.RawKernel(r"""
        #include <cuComplex.h>
        extern "C" __global__
        void z_transfer(const cuDoubleComplex* shift,
                        const cuDoubleComplex* diagonal,
                        const cuDoubleComplex* upper,
                        const cuDoubleComplex* lower,
                        cuDoubleComplex* history,
                        const int modes,
                        const int nz) {
            const int index = blockDim.x * blockIdx.x + threadIdx.x;
            if (index >= modes) return;
            cuDoubleComplex ratio = make_cuDoubleComplex(0.0, 0.0);
            for (int j = nz - 1; j >= 1; --j) {
                cuDoubleComplex value = cuCadd(shift[index], diagonal[j]);
                if (j < nz - 1) value = cuCadd(value, cuCmul(upper[j], ratio));
                cuDoubleComplex numerator = lower[j - 1];
                numerator = make_cuDoubleComplex(-cuCreal(numerator), -cuCimag(numerator));
                ratio = cuCdiv(numerator, value);
                history[(j - 1) * modes + index] = ratio;
            }
            for (int j = 1; j < nz - 1; ++j) {
                history[j * modes + index] = cuCmul(
                    history[j * modes + index],
                    history[(j - 1) * modes + index]);
            }
        }
        """, "z_transfer")
    modes = int(shift.size)
    history = cp.empty((len(z)-1, *shift.shape), dtype=cp.complex128)
    diagonal = cp.asarray(np.diag(z))
    upper = cp.asarray(np.diag(z, 1))
    lower = cp.asarray(np.diag(z, -1))
    threads = 128
    _CUDA_Z_TRANSFER_KERNEL(
        ((modes+threads-1)//threads,),
        (threads,),
        (shift, diagonal, upper, lower, history, modes, len(z)),
    )
    return history


def axis_matrix(n, dx, omega, density, sound_speed, faces, pml, axis):
    """Reuse production PML and face validation on a small thin grid."""
    translated = {"-x": faces.get(FACES[2*axis], "reflecting"),
                  "+x": faces.get(FACES[2*axis+1], "reflecting")}
    if any(boundary_type(v) not in {"open", "reflecting", "impedance"}
           for v in translated.values()):
        raise ValueError("Surface diagnostic supports open/reflecting/impedance faces")
    original_faces = pml.get("faces", [f for f in FACES
                                     if boundary_type(faces.get(f, "reflecting")) == "open"])
    local_pml = {**pml, "faces": [("-x", "+x")[i] for i in range(2)
                                if FACES[2*axis+i] in original_faces]}
    op = ConservativeHelmholtz(np.full((n, 2, 2), density),
                              np.full((n, 2, 2), sound_speed), dx, omega,
                              translated, pml=local_pml)
    w, s = op.weights[0].ravel(), op.stretch[0].ravel()
    edge = 1/(.5*(s[:-1]+s[1:])*dx*dx)
    diagonal = np.zeros(n, complex)
    diagonal[:-1] -= edge
    diagonal[1:] -= edge
    matrix = np.diag(diagonal)+np.diag(edge, 1)+np.diag(edge, -1)
    matrix /= (w*s)[:, None]
    for i, face in ((0, "-x"), (-1, "+x")):
        kind = boundary_type(translated[face])
        if kind in {"open", "impedance"}:
            impedance = density*sound_speed if kind == "open" else complex(*translated[face]["impedance"])
            matrix[i, i] += 1j*omega*density/(impedance*dx*w[i]*s[i])
    return matrix, s


class HomogeneousSurfaceSolver:
    def __init__(self, shape, dx, frequency, sound_speed, density, boundaries,
                 pml=None, device="cpu"):
        if not all(np.ndim(v) == 0 and np.isfinite(v) and v > 0
                   for v in (dx, frequency, sound_speed, density)):
            raise ValueError("This diagnostic requires positive scalar homogeneous materials")
        if set(boundaries)-set(FACES):
            raise ValueError("Unknown boundary face")
        self.shape = tuple(shape)
        self.dx = dx
        self.xp = np
        if device == "cuda":
            import cupy
            self.xp = cupy
        elif device != "cpu":
            raise ValueError("device must be cpu or cuda")
        xp = self.xp
        self.boundaries = boundaries.copy()
        omega = 2*np.pi*frequency
        axes = [axis_matrix(n, dx, omega, density, sound_speed, boundaries,
                            pml or {}, axis) for axis, n in enumerate(shape)]
        self.axes = [a[0] for a in axes]
        self.surface_pml = ((axes[0][1].imag[:, None] + axes[1][1].imag[None, :]
                            + axes[2][1].imag[0]) > 0)
        self.surface_rhs = None
        ex, vx, ix, condition_x = _cached_eigendecomposition(self.axes[0])
        ey, vy, iy, condition_y = _cached_eigendecomposition(self.axes[1])
        self.eigen_condition = [condition_x, condition_y]
        self.vx, self.vy = xp.asarray(vx), xp.asarray(vy)
        self.ix, self.iy = xp.asarray(ix), xp.asarray(iy)
        shift = xp.asarray(ex[:, None]+ey[None, :]+(omega/sound_speed)**2)
        z = self.axes[2]
        # H[j] = u[j+1]/u[0], determined by a backwards continued fraction.
        if xp is np:
            self.h = xp.empty((shape[2]-1, *shape[:2]), dtype=xp.complex128)
            ratio = xp.zeros(shape[:2], dtype=xp.complex128)
            for j in range(shape[2]-1, 0, -1):
                diagonal = shift+z[j, j]
                if j < shape[2]-1:
                    diagonal = diagonal+z[j, j+1]*ratio
                ratio = -z[j, j-1]/diagonal
                self.h[j-1] = ratio
            for j in range(1, shape[2]-1):
                self.h[j] *= self.h[j-1]
        else:
            self.h = _cuda_z_transfer(shift, z)
        self.schur = shift+z[0, 0]+z[0, 1]*self.h[0]

    def _numpy(self, value):
        return np.asarray(value) if self.xp is np else self.xp.asnumpy(value)

    def transform(self, plane):
        return self.ix @ plane @ self.iy.T

    def inverse_transform(self, plane):
        return self.vx @ plane @ self.vy.T

    def _validate_flux_masses(self, masses):
        if boundary_type(self.boundaries.get("-z", "reflecting")) != "reflecting":
            raise ValueError("Sinc Neumann source requires a reflecting/baffled bottom")
        if (masses.shape[0] != np.prod(self.shape[:2]) or masses.shape[1] == 0
                or not np.isfinite(masses.data).all()):
            raise ValueError("Invalid source masses")
        active = np.asarray(abs(masses).sum(axis=1)).ravel().reshape(self.shape[:2], order="F") > 0
        if np.any(active & self.surface_pml) or any(np.any(a) for a in
                (active[0], active[-1], active[:, 0], active[:, -1])):
            raise ValueError("Flux source overlaps PML or side boundaries")

    def flux_target_response(self, masses, points):
        """Return point x channel complex responses of the discrete FV PDE.

        Transpose the surface-to-observation linear map, including trilinear
        interpolation. Ordinary transposes are intentional: p = response @ w,
        without complex conjugation. No analytic field or reference phases enter.
        """
        self._validate_flux_masses(masses)
        low, frac = self._sample_coordinates(points)
        response = np.empty((len(low), masses.shape[1]), complex)
        for index, ((x, y, z), (fx, fy, fz)) in enumerate(zip(low, frac)):
            vx = (1-fx)*self.vx[x]+fx*self.vx[x+1]
            vy = (1-fy)*self.vy[y]+fy*self.vy[y+1]
            hz = (1-fz)*(1 if z == 0 else self.h[z-1])+fz*self.h[z]
            observation = vx[:, None]*vy[None, :]*hz/self.schur
            kernel = self._numpy(self.ix.T @ observation @ self.iy)
            response[index] = -4*np.pi/self.dx**3*(masses.T @ kernel.ravel(order="F"))
        return response

    def solve_flux(self, masses, phases, amplitudes=None):
        """Solve a normalized baffled Neumann aperture, in the same 1/m units
        as the official propagator. masses is the integrated source distribution;
        no pressure is pinned and no target or analytic field enters this solve.
        """
        self._validate_flux_masses(masses)
        phases = np.asarray(phases, dtype=float)
        amplitudes = np.ones_like(phases) if amplitudes is None else np.asarray(amplitudes)
        if (phases.ndim != 1 or amplitudes.shape != phases.shape
                or masses.shape != (np.prod(self.shape[:2]), len(phases))
                or not np.isfinite(phases).all() or not np.isfinite(amplitudes).all()
                or not np.isfinite(masses.data).all()):
            raise ValueError("Invalid source masses or phases")
        self.surface_rhs = (-4*np.pi/self.dx**3 *
                            (masses @ (amplitudes*np.exp(1j*phases)))).reshape(self.shape[:2], order="F")
        self.surface_modes = self.transform(self.xp.asarray(self.surface_rhs))/self.schur
        self.surface = self.inverse_transform(self.surface_modes)
        self.diagnostics = {"iterations": 0, "eigenvector_condition": self.eigen_condition,
                            "source_integral_min_max": [float(masses.sum(axis=0).min()),
                                                       float(masses.sum(axis=0).max())]}
        return self.diagnostics

    def plane(self, index):
        if not 0 <= index < self.shape[2]:
            raise ValueError("z index outside grid")
        if index == 0:
            return self._numpy(self.surface)
        return self._numpy(self.inverse_transform(self.h[index-1]*self.surface_modes))

    def _sample_coordinates(self, points):
        points = np.asarray(points, float)
        coordinates = points/self.dx
        if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
                or np.any(coordinates < -1e-9)
                or np.any(coordinates > np.asarray(self.shape)-1+1e-9)):
            raise ValueError("Points must lie inside the local grid")
        coordinates = np.clip(coordinates, 0, np.asarray(self.shape)-1)
        low = np.minimum(np.floor(coordinates).astype(int), np.asarray(self.shape)-2)
        frac = coordinates-low
        return low, frac

    def sample(self, points):
        """Trilinear physical-grid interpolation, with no target snapping."""
        low, frac = self._sample_coordinates(points)
        result = np.zeros(len(low), complex)
        for z in np.unique(np.concatenate((low[:, 2], low[:, 2]+1))):
            plane = self.plane(z)
            for dz in (0, 1):
                selected = np.flatnonzero(low[:, 2]+dz == z)
                for a in (0, 1):
                    for b in (0, 1):
                        weight = ((frac[selected, 0] if a else 1-frac[selected, 0])
                                  * (frac[selected, 1] if b else 1-frac[selected, 1])
                                  * (frac[selected, 2] if dz else 1-frac[selected, 2]))
                        result[selected] += weight*plane[low[selected, 0]+a, low[selected, 1]+b]
        return result
