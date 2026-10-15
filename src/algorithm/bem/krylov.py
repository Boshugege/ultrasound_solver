"""Restarted GMRES for dense GPU (CuPy) or NumPy operators."""

from dataclasses import dataclass

import numpy as np


@dataclass
class KrylovResult:
    solution: object
    converged: bool
    iterations: int


def _givens(a: complex, b: complex):
    """Complex rotation G with G @ [a, b] = [r, 0]."""
    if a == 0:
        return 0.0, 1.0 + 0j
    norm = np.hypot(abs(a), abs(b))
    return abs(a) / norm, (a / abs(a)) * np.conj(b) / norm


def gmres(apply_a, b, x0=None, rtol=1e-5, restart=100, maxiter=500, xp=np):
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
        H = np.zeros((m + 1, m), dtype=complex)
        cs = np.zeros(m)
        sn = np.zeros(m, dtype=complex)
        g = np.zeros(m + 1, dtype=complex)
        g[0] = beta
        V[0] = r / beta
        steps = 0
        for j in range(m):
            w = apply_a(V[j])
            # Classical Gram-Schmidt applied twice: stable and GPU friendly.
            basis = V[: j + 1]
            h = (w.conj() @ basis.T).conj()  # V^H w without materialising conj(V)
            w = w - basis.T @ h
            correction = (w.conj() @ basis.T).conj()
            w = w - basis.T @ correction
            h = h + correction
            H[: j + 1, j] = np.asarray(h.get() if hasattr(h, "get") else h)
            H[j + 1, j] = float(xp.linalg.norm(w))
            if H[j + 1, j] > 0:
                V[j + 1] = w / H[j + 1, j]
            for i in range(j):
                upper = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                H[i + 1, j] = -np.conj(sn[i]) * H[i, j] + cs[i] * H[i + 1, j]
                H[i, j] = upper
            cs[j], sn[j] = _givens(H[j, j], H[j + 1, j])
            H[j, j] = cs[j] * H[j, j] + sn[j] * H[j + 1, j]
            H[j + 1, j] = 0.0
            g[j + 1] = -np.conj(sn[j]) * g[j]
            g[j] = cs[j] * g[j]
            steps = j + 1
            total += 1
            if abs(g[j + 1]) <= rtol * b_norm or total >= maxiter or H[j, j] == 0:
                break
        y = np.linalg.solve(np.triu(H[:steps, :steps]), g[:steps])
        x = x + V[:steps].T @ xp.asarray(y, dtype=b.dtype)
