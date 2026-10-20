"""Restarted GMRES for dense GPU (CuPy) or NumPy operators.

The Arnoldi process stays on the device: the Hessenberg matrix is copied to the
host (one synchronisation) only every ``check_every`` steps to estimate the
residual. Launch and synchronisation latency, not the matrix-vector product,
otherwise dominates an iteration for matrices of ~10^4 unknowns.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class KrylovResult:
    solution: object
    converged: bool
    iterations: int


def _least_squares(H, beta):
    """min ||beta e1 - H y|| for a small (k + 1) x k Hessenberg matrix; returns y, residual."""
    rhs = np.zeros(H.shape[0], dtype=complex)
    rhs[0] = beta
    y = np.linalg.lstsq(H, rhs, rcond=None)[0]
    return y, float(np.linalg.norm(rhs - H @ y))


def gmres(apply_a, b, x0=None, rtol=1e-5, restart=100, maxiter=500, xp=np, check_every=1):
    """Solve ``A x = b`` to ``||b - A x|| <= rtol ||b||`` (true residual checked at
    every restart); ``iterations`` counts matrix-vector products in the Arnoldi loop."""
    b_norm = float(xp.linalg.norm(b))
    if b_norm == 0.0:
        return KrylovResult(xp.zeros_like(b), True, 0)
    x = xp.zeros_like(b) if x0 is None else x0.copy()
    total = 0
    while True:
        r = b - apply_a(x)
        beta = float(xp.linalg.norm(r))
        if beta <= rtol * b_norm or total >= maxiter:
            return KrylovResult(x, beta <= rtol * b_norm, total)
        m = min(restart, maxiter - total)
        V = xp.zeros((m + 1, b.size), dtype=b.dtype)
        H = xp.zeros((m + 1, m), dtype=complex)
        V[0] = r / beta
        steps, y = 0, None
        for j in range(m):
            w = apply_a(V[j])
            # Classical Gram-Schmidt applied twice: stable and GPU friendly.
            basis = V[: j + 1]
            h = (w.conj() @ basis.T).conj()  # V^H w without materialising conj(V)
            w = w - basis.T @ h
            correction = (w.conj() @ basis.T).conj()
            w = w - basis.T @ correction
            norm = xp.linalg.norm(w)
            H[: j + 1, j] = h + correction
            H[j + 1, j] = norm
            V[j + 1] = w / xp.where(norm > 0, norm, 1)
            steps, total = j + 1, total + 1
            if steps % check_every == 0 or steps == m or total >= maxiter:
                Hk = H[: steps + 1, :steps]
                Hk = Hk.get() if hasattr(Hk, "get") else np.asarray(Hk)
                y, estimate = _least_squares(Hk, beta)
                if estimate <= rtol * b_norm or total >= maxiter or Hk[steps, steps - 1] == 0:
                    break
        x = x + V[:steps].T @ xp.asarray(y, dtype=b.dtype)
