"""The BEM's source is upstream's sinc Neumann aperture: free-field checks.

1. Forward aperture field (``aperture_sources`` + baffle image) against upstream's
   own reference integral ``sinc_aperture.aperture_field`` at high order, per
   emitter, from 5 mm above the baffle to the far field.
2. Reciprocity (``FocusSession.response_row``, monopole at the target) against
   the forward field: the same numbers, computed the other way round.
3. Response rows against upstream's PDE (``HelmholtzDirectSolver.target_response``,
   src/examples/pde_homogeneous_16x16.yaml, 1/3 mm grid) at the eight targets of
   docs/PDE_PHASE_ALIGNMENT.md: amplitude, phase (common offset removed) and the
   focus efficiency of the BEM phases inside the PDE model.
"""

from pathlib import Path
import sys
import time

import cupy as cp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from algorithm.bem.realtime import FocusSession  # noqa: E402
from algorithm.bem.solver import RigidBEM, aperture_sources  # noqa: E402
from algorithm.config import SimulationConfig  # noqa: E402
from algorithm.physics.sinc_aperture import aperture_field  # noqa: E402
from algorithm.physics.transducer_array import TransducerArray  # noqa: E402

CONFIG = ROOT / "src" / "examples" / "pde_homogeneous_16x16.yaml"
TARGETS_MM = [(80, 80, 120), (115, 55, 100), (80, 80, 70), (80, 80, 135),
              (120, 80, 110), (120, 120, 110), (45, 115, 100), (140, 40, 80)]


def forward_rows(bem, centers, diameter, points, orders):
    """rows[p, j]: field at points[p] of emitter j driven alone at unit drive."""
    quad, weights, per = aperture_sources(centers, diameter, *orders)
    rows = np.empty((len(points), len(centers)), dtype=complex)
    for j in range(len(centers)):
        value, _ = bem.sources(cp.asarray(points), quad[j * per:(j + 1) * per], weights[j * per:(j + 1) * per])
        rows[:, j] = cp.asnumpy(value)
    return rows


def main():
    cfg = SimulationConfig.from_yaml(CONFIG)
    k, diameter = cfg.k0, cfg.specs.diameter
    centers = TransducerArray(cfg).centers
    empty = RigidBEM(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64), k=k)
    print(f"k = {k:.3f} 1/m, aperture {diameter * 1e3:.1f} mm, {len(centers)} emitters")

    # 1. forward field vs upstream's reference integral
    emitters = [0, 37, 128, 255]
    rng = np.random.default_rng(0)
    for height in (0.005, 0.02, 0.08):
        points = np.column_stack([rng.uniform(-0.04, 0.2, 20), rng.uniform(-0.04, 0.2, 20), np.full(20, height)])
        reference = np.column_stack([
            aperture_field(points, centers[j:j + 1], np.zeros(1), k, diameter, 64, 256, device="cuda")
            for j in emitters])
        for orders in ((8, 32), (32, 128)):
            rows = forward_rows(empty, centers[emitters], diameter, points, orders)
            error = np.abs(rows - reference).max() / np.abs(reference).max()
            print(f"  forward vs aperture_field, z = {height * 1e3:3.0f} mm, orders {orders}: "
                  f"max relative error {error:.1e}", flush=True)

    # 2. reciprocity vs forward, 3. vs upstream PDE
    targets = np.array(TARGETS_MM, float) * 1e-3
    session = FocusSession(centers, diameter, k=k)
    session.set_surface(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
    reciprocal = np.array([session.response_row(t) for t in targets])
    forward = forward_rows(empty, centers, diameter, targets, (32, 128))
    print(f"reciprocity vs forward: max relative difference "
          f"{np.abs(reciprocal - forward).max() / np.abs(forward).max():.1e}")

    from algorithm.solvers import HelmholtzDirectSolver

    solver = HelmholtzDirectSolver(cfg)
    started = time.perf_counter()
    pde = solver.target_response(targets)
    solver.close()
    print(f"upstream PDE target responses ({cfg.domain.grid_size}, dx = {cfg.domain.dx * 1e3:.3f} mm): "
          f"{time.perf_counter() - started:.1f} s")
    print("target (mm)       |BEM|/|PDE| optimum   phase MAE  max   BEM phases in PDE   PDE phases in BEM")
    for t, b, p in zip(TARGETS_MM, reciprocal, pde):
        ratio = np.abs(b).sum() / np.abs(p).sum()
        d = np.exp(1j * (np.angle(b) - np.angle(p)))
        error = np.degrees(np.abs(np.angle(d * np.conj(d.mean() / abs(d.mean())))))
        in_pde = abs(np.sum(p * np.exp(-1j * np.angle(b)))) / np.abs(p).sum()
        in_bem = abs(np.sum(b * np.exp(-1j * np.angle(p)))) / np.abs(b).sum()
        print(f"{str(t):17s} {ratio:10.5f}        {error.mean():6.2f}  {error.max():5.1f}   "
              f"{in_pde:.5%}          {in_bem:.5%}", flush=True)


if __name__ == "__main__":
    main()
