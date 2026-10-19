"""Memory-bounded solves of the unchanged heterogeneous FV Helmholtz matrix.

Homogeneous tensor eigenmodes are only a preconditioner. Every solve is checked
against the original assembled complex128 matrix, including the adjoint.
"""
import time
from types import SimpleNamespace
import numpy as np
import cupy as cp
from cupyx.scipy.sparse import csr_matrix
from cupyx.scipy.sparse.linalg import LinearOperator, gmres
from algorithm.solvers.helmholtz_solver import HelmholtzDirectSolver
from algorithm.solvers.homogeneous_surface import axis_matrix, _cached_eigendecomposition


class VolumeInverse:
    def __init__(self, cfg):
        self.shape = tuple(cfg.domain.grid_size)
        self.forward, self.backward, eigenvalues = [], [], []
        self.conditions = []
        for axis, n in enumerate(self.shape):
            matrix, _ = axis_matrix(
                n, cfg.domain.dx, cfg.omega, cfg.physics.medium_density,
                cfg.physics.sound_speed, cfg.boundary_conditions,
                cfg.solver.get("pml", {}), axis)
            ev, vectors, inverse, condition = _cached_eigendecomposition(matrix)
            self.forward.append(cp.asarray(inverse))
            self.backward.append(cp.asarray(vectors))
            eigenvalues.append(cp.asarray(ev))
            self.conditions.append(condition)
        self.denominator = (eigenvalues[0][:, None, None]
                            + eigenvalues[1][None, :, None]
                            + eigenvalues[2][None, None, :] + cfg.k0**2)

    @staticmethod
    def transform(value, matrix, axis):
        value = cp.moveaxis(value, axis, 0)
        shape = value.shape
        value = (matrix @ value.reshape(shape[0], -1)).reshape(shape)
        return cp.moveaxis(value, 0, axis)

    def __call__(self, rhs):
        value = rhs.reshape(self.shape, order="F")
        for axis, matrix in enumerate(self.forward):
            value = self.transform(value, matrix, axis)
        value = value/self.denominator
        for axis, matrix in enumerate(self.backward):
            value = self.transform(value, matrix, axis)
        return value.ravel(order="F")


class SeparableIterativeSolver(HelmholtzDirectSolver):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.volume_inverse = VolumeInverse(cfg)
        self.iterative_state = None
        self.solve_log = []
        self.max_iterations = int(cfg.solver.get("volume_maxiter", 800))
        if self.max_iterations <= 0:
            raise ValueError("volume_maxiter must be positive")

    def invalidate_factorization(self):
        super().invalidate_factorization()
        self.iterative_state = None

    def prepare(self):
        if self.iterative_state is not None:
            return
        op = self.warp_engine.operator
        if op.fixed.any():
            raise ValueError("Volume iteration requires no Dirichlet rows")
        matrix = csr_matrix(self.A_cached_csr_cpu)
        scale = cp.asarray(np.sqrt(op.rho).ravel(order="F"))
        weight = cp.asarray(op.mass_weight.ravel(order="F"))
        # C = W^-1 S A S. In uniform material C equals the modal operator.
        def action(v):
            return self.volume_inverse(scale*(matrix @ (scale*v))/weight)
        operator = LinearOperator(matrix.shape, matvec=action, dtype=cp.complex128)
        self.iterative_state = matrix, scale, weight, operator

    def _linear_solve(self, rhs, adjoint=False):
        self.prepare()
        matrix, scale, weight, operator = self.iterative_state
        original = cp.asarray(np.asarray(rhs).ravel(order="F"))
        b = original.conj() if adjoint else original
        transformed = self.volume_inverse(scale*b/weight)
        history = []
        started = time.perf_counter()
        value, info = gmres(operator, transformed, rtol=1e-10, restart=40,
                            maxiter=self.max_iterations, callback=lambda r: history.append(float(r)),
                            callback_type="pr_norm")
        field = scale*value
        residual = float(cp.linalg.norm(matrix@field-b)/cp.linalg.norm(b))
        # A is complex symmetric (validated by the tests). A^H = conjugate(A).
        if adjoint:
            field = field.conj()
        row = dict(adjoint=adjoint, residual=residual, info=int(info),
                   restart_cycles=len(history), seconds=time.perf_counter()-started)
        self.solve_log.append(row)
        print(f"VOLUME_SOLVE {row}", flush=True)
        if info or not np.isfinite(residual) or residual > 1e-7:
            raise RuntimeError(f"Volume iterative solve failed: {row}")
        if self.condensed_solver is None:
            self.condensed_solver = SimpleNamespace(close=lambda: None)
        name = "last_adjoint_relative_residual" if adjoint else "last_relative_residual"
        setattr(self.condensed_solver, name, residual)
        return cp.asnumpy(field)

    def solve_rhs(self, rhs, verbose=True):
        return self._linear_solve(rhs)

    def solve_adjoint(self, adjoint_rhs):
        return self._linear_solve(adjoint_rhs, adjoint=True)
