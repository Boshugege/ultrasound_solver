"""Rigid-obstacle BEM against upstream's PDE (cudss_hybrid_direct) with an obstacle.

Small scene that the PDE can solve at several resolutions: 2 x 2 ideal-grid
array (pitch 10.5 mm, aperture 9 mm) in a 50 mm cube, -z reflecting baffle, the
other faces open with a 12 mm PML, and a sphere of radius 8 mm at
(25, 25, 22) mm with upstream's default obstacle material (rho 1250, c 2200).
Per emitter, fields at 300 points outside the PML and >= 2.5 mm from the sphere
are compared (relative L2 over all emitters and points):

* free field (no sphere): PDE grid + PML error alone;
* with the sphere: the same plus the obstacle model (BEM: rigid surface, exact
  geometry; PDE: smoothed material inclusion);
* scattered field (with - without): obstacle physics, PML error largely cancels.
Usage: check_obstacle_vs_pde.py NODES_PER_AXIS [...]   (e.g. 47 70 93)
"""

from pathlib import Path
import sys
import time

import cupy as cp
import numpy as np
import yaml
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from algorithm.bem.solver import RigidBEM, aperture_sources, icosphere  # noqa: E402
from algorithm.config import SimulationConfig  # noqa: E402
from algorithm.physics.transducer_array import TransducerArray  # noqa: E402

BOX, PML, CENTER, RADIUS = 0.05, 0.012, np.array([0.025, 0.025, 0.022]), 0.008
WORK = ROOT / "outputs" / "bem_study" / "obstacle_vs_pde"


def config(n, sphere):
    raw = {
        "mode": "same_phase", "same_phase_rad": 0.0,
        "physics": {"sound_speed": 346.0, "medium_density": 1.21},
        "transducer_specs": {"frequency": 40000.0, "array_n": 2, "diameter": 0.009, "pitch": 0.0105,
                             "geometry": "ideal_grid"},
        "domain": {"box_size": [BOX] * 3, "grid_size": [n] * 3},
        "boundary_conditions": {"-x": "open", "+x": "open", "-y": "open", "+y": "open",
                                "-z": "reflecting", "+z": "open"},
        "targets": [], "obstacles": [],
        "solver": {"backend": "cudss_hybrid_direct", "direct_residual_tol": 1e-6,
                   "hybrid_device_memory_limit_gb": 20.0, "hybrid_device_memory_reserve_gb": 1.0,
                   "pml": {"cells": int(round(PML / (BOX / (n - 1)))), "strength": 2.0, "order": 3,
                           "faces": ["-x", "+x", "-y", "+y", "+z"]}},
        "io": {"output_file": str(WORK / f"unused_{n}.npz"), "auto_visualize": False},
    }
    if sphere:
        raw["obstacles"] = [{"type": "sphere", "center": CENTER.tolist(), "radius": RADIUS,
                             "material": {"density": 1250.0, "sound_speed": 2200.0}}]
    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / f"scene_{n}_{'sphere' if sphere else 'free'}.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return SimulationConfig.from_yaml(path)


def sample_points():
    rng = np.random.default_rng(1)
    points = []
    while len(points) < 300:
        p = rng.uniform([PML + 0.001, PML + 0.001, 0.003], [BOX - PML - 0.001, BOX - PML - 0.001, BOX - PML - 0.001])
        if np.linalg.norm(p - CENTER) > RADIUS + 0.0025:
            points.append(p)
    return np.array(points)


def pde_fields(n, sphere, points):
    from algorithm.solvers import HelmholtzDirectSolver

    cfg = config(n, sphere)
    solver = HelmholtzDirectSolver(cfg)
    try:
        started = time.perf_counter()
        solver.assemble_matrix()
        solver.factorize()
        columns = np.asfortranarray(solver.warp_engine.source_columns.toarray(), dtype=np.complex128)
        fields = solver.solve_rhs(columns, verbose=False)
        seconds = time.perf_counter() - started
    finally:
        solver.close()
    axes = [np.arange(n) * cfg.domain.dx] * 3
    out = np.empty((len(points), fields.shape[1]), dtype=complex)
    for j in range(fields.shape[1]):
        grid = fields[:, j].reshape((n, n, n), order="F")
        out[:, j] = RegularGridInterpolator(axes, grid)(points)
    return out, seconds, cfg


def bem_fields(cfg, points, sphere):
    k = cfg.k0
    centers = TransducerArray(cfg).centers
    quad, weights, per = aperture_sources(centers, cfg.specs.diameter, 16, 64)
    if sphere:
        subdivisions = 0
        while RADIUS * 1.0515 / 2 ** subdivisions > cfg.wavelength / 8:
            subdivisions += 1
        bem = RigidBEM(*icosphere(RADIUS, subdivisions, center=CENTER), k=k)
        bem.assemble()
    else:
        bem = RigidBEM(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64), k=k)
    out = np.empty((len(points), len(centers)), dtype=complex)
    for j in range(len(centers)):
        q, w = quad[j * per:(j + 1) * per], weights[j * per:(j + 1) * per]
        value, _ = bem.sources(cp.asarray(points), q, w)
        if bem.n:
            u, info = bem.solve(bem.incident_on_surface(lambda x: bem.sources(x, q, w)), rtol=1e-6)
            value = value + bem.scattered(cp.asarray(points), u)
        out[:, j] = cp.asnumpy(value)
    return out, bem.n


def relative(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def main():
    points = sample_points()
    reference = None
    for n in [int(v) for v in sys.argv[1:]]:
        free, t_free, cfg = pde_fields(n, False, points)
        with_sphere, t_sphere, _ = pde_fields(n, True, points)
        if reference is None:
            bem_free, _ = bem_fields(cfg, points, False)
            bem_sphere, elements = bem_fields(cfg, points, True)
            reference = (bem_free, bem_sphere)
            print(f"BEM sphere: {elements} elements (lambda/8)", flush=True)
        bem_free, bem_sphere = reference
        print(f"PDE {n}^3 ({cfg.wavelength / cfg.domain.dx:4.1f} points/wavelength, PML "
              f"{cfg.solver['pml']['cells']} cells, {t_free + t_sphere:.0f} s): free field "
              f"{relative(free, bem_free):.2e} | with sphere {relative(with_sphere, bem_sphere):.2e} | "
              f"scattered {relative(with_sphere - free, bem_sphere - bem_free):.2e} "
              f"(scattered / total {np.linalg.norm(bem_sphere - bem_free) / np.linalg.norm(bem_sphere):.2f})",
              flush=True)


if __name__ == "__main__":
    main()
