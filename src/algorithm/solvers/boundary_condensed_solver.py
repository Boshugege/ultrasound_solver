import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from algorithm.solvers.cudss_hybrid_solver import CuDSSHybridSymmetricSolver


class BoundaryCondensedDirectSolver:
    """
    Exact direct solver after eliminating prescribed pressure rows.

    The retained finite-difference rows form a complex-symmetric system. This
    allows cuDSS to use a symmetric factorization while the eliminated boundary
    values are recovered exactly through block back-substitution.
    """

    def __init__(
        self,
        matrix: sp.spmatrix,
        dims: tuple[int, int, int],
        boundary_conditions: dict,
        fixed_dofs: np.ndarray,
        num_threads: int = 1,
        multithreading_lib: str = None,
        execution_mode: str = "auto",
        device_memory_limit_gb: float = 5.6,
        device_memory_reserve_gb: float = 0.75,
        register_cuda_memory: bool = False,
        symmetry_tolerance: float = 1.0e-12,
    ):
        self.matrix = matrix.tocsr()
        self.n = int(self.matrix.shape[0])
        self.dims = tuple(int(value) for value in dims)
        self.boundary_conditions = boundary_conditions
        self.fixed_dofs = np.asarray(fixed_dofs, dtype=np.int64)
        self.symmetry_tolerance = float(symmetry_tolerance)

        if self.matrix.shape != (self.n, self.n):
            raise ValueError("matrix must be square")
        if np.prod(self.dims, dtype=np.int64) != self.n:
            raise ValueError(
                f"dims={self.dims} do not match matrix size {self.n}"
            )
        if np.any(self.fixed_dofs < 0) or np.any(self.fixed_dofs >= self.n):
            raise ValueError("fixed_dofs contains an out-of-range index")

        self.condensation_time = 0.0
        self.boundary_factor_time = 0.0
        self.boundary_solve_time = 0.0
        self.solve_time = 0.0
        self.adjoint_solve_time = 0.0
        self.solve_count = 0
        self.adjoint_solve_count = 0
        self.last_relative_residual = None
        self.last_adjoint_relative_residual = None

        start = time.perf_counter()
        self._build_condensed_system()
        self.condensation_time = time.perf_counter() - start

        start = time.perf_counter()
        self.boundary_factor = (
            spla.splu(self.a_ee.tocsc()) if self.a_ee.shape[0] else None
        )
        self.boundary_factor_time = time.perf_counter() - start

        self.reduced_solver = CuDSSHybridSymmetricSolver(
            self.reduced_matrix_lower,
            num_threads=num_threads,
            multithreading_lib=multithreading_lib,
            execution_mode=execution_mode,
            device_memory_limit_gb=device_memory_limit_gb,
            device_memory_reserve_gb=device_memory_reserve_gb,
            register_cuda_memory=register_cuda_memory,
        )

    def _build_condensed_system(self):
        eliminated_mask, _ = self._eliminated_mask()
        self.eliminated_indices = np.flatnonzero(eliminated_mask)
        self.retained_indices = np.flatnonzero(~eliminated_mask)
        if self.retained_indices.size == 0:
            raise ValueError(
                "boundary condensation requires at least one free row"
            )

        # Prescribed rows are identity equations, so their solution has no
        # dependency on the retained unknowns (zero transfer).
        self.boundary_transfer = sp.csr_matrix(
            (
                self.eliminated_indices.size,
                self.retained_indices.size,
            ),
            dtype=np.complex128,
        )

        retained = self.retained_indices
        eliminated = self.eliminated_indices
        a_rr = self.matrix[retained, :][:, retained].tocsr()
        self.a_re = self.matrix[retained, :][:, eliminated].tocsr()
        self.a_er = self.matrix[eliminated, :][:, retained].tocsr()
        self.a_ee = self.matrix[eliminated, :][:, eliminated].tocsr()
        reduced_matrix = a_rr
        reduced_matrix.sum_duplicates()
        reduced_matrix.eliminate_zeros()
        reduced_matrix.sort_indices()

        difference = (
            reduced_matrix - reduced_matrix.transpose()
        ).tocsr()
        max_difference = (
            float(np.max(np.abs(difference.data)))
            if difference.nnz
            else 0.0
        )
        max_value = float(np.max(np.abs(reduced_matrix.data)))
        self.relative_symmetry_error = (
            max_difference / max_value if max_value > 0.0 else 0.0
        )
        if self.relative_symmetry_error > self.symmetry_tolerance:
            raise ValueError(
                "condensed matrix is not complex symmetric: "
                f"relative error={self.relative_symmetry_error:.3e}"
            )

        self.reduced_nnz = int(reduced_matrix.nnz)
        self.reduced_matrix_lower = sp.tril(
            reduced_matrix,
            format="csr",
        )

    def _eliminated_mask(self) -> tuple[np.ndarray, np.ndarray]:
        # Robin/Neumann rows now retain the conservative PDE. Eliminating
        # them as legacy two-entry boundary equations would be incorrect.
        fixed_mask = np.zeros(self.n, dtype=bool)
        fixed_mask[self.fixed_dofs] = True
        prescribed = self.matrix[self.fixed_dofs]
        if (np.any(prescribed.getnnz(axis=1) != 1)
                or np.any(self.matrix.diagonal()[self.fixed_dofs] != 1)):
            raise ValueError("fixed_dofs must identify identity Dirichlet rows")
        return fixed_mask, fixed_mask

    def _solve_boundary(self, rhs, trans="N"):
        if self.boundary_factor is None:
            return np.empty_like(rhs)
        return self.boundary_factor.solve(rhs, trans=trans)

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        rhs_matrix, squeeze = self._as_rhs_matrix(rhs)
        start = time.perf_counter()

        boundary_start = time.perf_counter()
        boundary_part = self._solve_boundary(
            rhs_matrix[self.eliminated_indices, :]
        )
        reduced_rhs = (
            rhs_matrix[self.retained_indices, :]
            - self.a_re @ boundary_part
        )
        self.boundary_solve_time += (
            time.perf_counter() - boundary_start
        )

        retained_solution = self.reduced_solver.solve(reduced_rhs)
        eliminated_solution = (
            boundary_part
            + self.boundary_transfer @ retained_solution
        )
        result = np.empty_like(rhs_matrix, dtype=np.complex128)
        result[self.retained_indices, :] = retained_solution
        result[self.eliminated_indices, :] = eliminated_solution

        self.solve_time += time.perf_counter() - start
        self.solve_count += rhs_matrix.shape[1]
        self.last_relative_residual = self.relative_residual(
            result,
            rhs_matrix,
        )
        return result[:, 0] if squeeze else result

    def solve_adjoint(self, rhs: np.ndarray) -> np.ndarray:
        rhs_matrix, squeeze = self._as_rhs_matrix(rhs)
        start = time.perf_counter()

        boundary_start = time.perf_counter()
        boundary_part = self._solve_boundary(
            rhs_matrix[self.eliminated_indices, :],
            trans="H",
        )
        reduced_rhs = (
            rhs_matrix[self.retained_indices, :]
            - self.a_er.conjugate().transpose() @ boundary_part
        )
        self.boundary_solve_time += (
            time.perf_counter() - boundary_start
        )

        retained_solution = np.conjugate(
            self.reduced_solver.solve(np.conjugate(reduced_rhs))
        )
        eliminated_rhs = (
            rhs_matrix[self.eliminated_indices, :]
            - self.a_re.conjugate().transpose() @ retained_solution
        )
        eliminated_solution = self._solve_boundary(
            eliminated_rhs,
            trans="H",
        )

        result = np.empty_like(rhs_matrix, dtype=np.complex128)
        result[self.retained_indices, :] = retained_solution
        result[self.eliminated_indices, :] = eliminated_solution

        self.adjoint_solve_time += time.perf_counter() - start
        self.adjoint_solve_count += rhs_matrix.shape[1]
        self.last_adjoint_relative_residual = (
            self.adjoint_relative_residual(result, rhs_matrix)
        )
        return result[:, 0] if squeeze else result

    def relative_residual(
        self,
        solution: np.ndarray,
        rhs: np.ndarray,
    ) -> float:
        residual = self.matrix @ solution - rhs
        rhs_norm = np.linalg.norm(rhs)
        return (
            float(np.linalg.norm(residual) / rhs_norm)
            if rhs_norm > 0.0
            else 0.0
        )

    def adjoint_relative_residual(
        self,
        solution: np.ndarray,
        rhs: np.ndarray,
    ) -> float:
        residual = self.matrix.conjugate().transpose() @ solution - rhs
        rhs_norm = np.linalg.norm(rhs)
        return (
            float(np.linalg.norm(residual) / rhs_norm)
            if rhs_norm > 0.0
            else 0.0
        )

    def stats(self) -> dict:
        stats = self.reduced_solver.stats()
        stats.update({
            "dofs": self.n,
            "nnz": int(self.matrix.nnz),
            "reduced_dofs": int(self.retained_indices.size),
            "eliminated_boundary_dofs": int(
                self.eliminated_indices.size
            ),
            "reduced_nnz": self.reduced_nnz,
            "condensation_seconds": self.condensation_time,
            "boundary_factor_seconds": self.boundary_factor_time,
            "boundary_solve_seconds": self.boundary_solve_time,
            "solve_seconds": self.solve_time,
            "solve_count": self.solve_count,
            "adjoint_solve_seconds": self.adjoint_solve_time,
            "adjoint_solve_count": self.adjoint_solve_count,
            "last_relative_residual": self.last_relative_residual,
            "last_adjoint_relative_residual": (
                self.last_adjoint_relative_residual
            ),
            "relative_symmetry_error": self.relative_symmetry_error,
            "matrix_type": "boundary_condensed_symmetric",
        })
        return stats

    def close(self):
        self.reduced_solver.close()

    def _as_rhs_matrix(
        self,
        rhs: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        rhs_array = np.asarray(rhs, dtype=np.complex128)
        if rhs_array.ndim == 1:
            if rhs_array.size != self.n:
                raise ValueError(
                    f"rhs has {rhs_array.size} entries, expected {self.n}"
                )
            return np.asfortranarray(
                rhs_array.reshape((-1, 1))
            ), True
        if rhs_array.ndim == 2 and rhs_array.shape[0] == self.n:
            return np.asfortranarray(rhs_array), False
        raise ValueError(
            f"rhs must have shape ({self.n},) or ({self.n}, nrhs), "
            f"got {rhs_array.shape}"
        )
