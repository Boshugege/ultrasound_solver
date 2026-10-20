"""With an obstacle: how good are the PDE's focusing phases, judged by the BEM?

4 x 4 ideal-grid array (pitch 10.5 mm, aperture 9 mm) in a 70 mm cube, -z baffle,
12 mm PML on the open faces, sphere of radius 10 mm at (35, 35, 24) mm with the
default obstacle material. For focus points in the sphere's shadow, beside it and
below it, the single-focus phases -arg(h) of each model are evaluated in the BEM
model (rigid sphere, exact geometry): fraction of the optimal focus pressure.
"Free-field phases" ignore the sphere (exact aperture integral). The PDE responses
are trilinear interpolations of its fields (all emitters, one factorisation).
Usage: compare_obstacle_focus.py NODES_PER_AXIS [...]   (66 ~ 8, 98 ~ 12 points/wavelength)
"""

from pathlib import Path
import sys
import time

import numpy as np
import yaml
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from algorithm.bem.scene_solver import BemSceneSolver  # noqa: E402
from algorithm.config import SimulationConfig  # noqa: E402

BOX, PML = 0.07, 0.012
SPHERE = {"type": "sphere", "center": [0.035, 0.035, 0.024], "radius": 0.010,
          "material": {"density": 1250.0, "sound_speed": 2200.0}}
TARGETS_MM = {"shadow, above the sphere": (35, 35, 46), "shadow, off axis": (40, 33, 50),
              "beside the sphere": (20, 35, 26), "beside, low": (50, 40, 14), "below the sphere": (35, 30, 8)}
WORK = ROOT / "outputs" / "bem_study" / "obstacle_focus"


def config(n, backend, sphere=True):
    raw = {
        "mode": "same_phase", "same_phase_rad": 0.0,
        "physics": {"sound_speed": 346.0, "medium_density": 1.21},
        "transducer_specs": {"frequency": 40000.0, "array_n": 4, "diameter": 0.009, "pitch": 0.0105,
                             "geometry": "ideal_grid"},
        "domain": {"box_size": [BOX] * 3, "grid_size": [n] * 3},
        "boundary_conditions": {"-x": "open", "+x": "open", "-y": "open", "+y": "open",
                                "-z": "reflecting", "+z": "open"},
        "targets": [], "obstacles": [SPHERE] if sphere else [],
        "solver": {"backend": backend, "direct_residual_tol": 1e-6,
                   "hybrid_device_memory_limit_gb": 20.0, "hybrid_device_memory_reserve_gb": 1.0,
                   "pml": {"cells": int(round(PML / (BOX / (n - 1)))), "strength": 2.0, "order": 3,
                           "faces": ["-x", "+x", "-y", "+y", "+z"]}},
        "io": {"output_file": str(WORK / "unused.npz"), "auto_visualize": False},
    }
    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / f"scene_{backend}_{n}_{int(sphere)}.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return SimulationConfig.from_yaml(path)


def pde_responses(n, targets):
    from algorithm.solvers import HelmholtzDirectSolver

    cfg = config(n, "cudss_hybrid_direct")
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
    h = np.column_stack([RegularGridInterpolator(axes, fields[:, j].reshape((n,) * 3, order="F"))(targets)
                         for j in range(fields.shape[1])])
    return h, seconds, cfg


def efficiency(h_true, phases):
    return abs(np.sum(h_true * np.exp(1j * phases))) / np.abs(h_true).sum()


def main():
    targets = np.array(list(TARGETS_MM.values()), float) * 1e-3
    started = time.perf_counter()
    bem = BemSceneSolver(config(66, "bem"))
    h_bem = bem.target_response(targets)
    bem_seconds = time.perf_counter() - started
    h_free = BemSceneSolver(config(66, "bem", sphere=False)).target_response(targets)
    print(f"BEM: {bem.bem.n} elements, setup + {len(targets)} targets {bem_seconds:.2f} s", flush=True)
    columns = {"free-field phases": h_free}
    for n in [int(v) for v in sys.argv[1:]]:
        h_pde, seconds, cfg = pde_responses(n, targets)
        columns[f"PDE {cfg.wavelength / cfg.domain.dx:.0f} ppw ({seconds:.0f} s)"] = h_pde
    print("focus pressure in the BEM model with the sphere, fraction of its optimum:")
    print(f"{'target (mm)':38s}" + "".join(f"{name:>24s}" for name in columns))
    for (name, mm), truth, row in zip(TARGETS_MM.items(), h_bem, range(len(targets))):
        print(f"{name + ' ' + str(mm):38s}" + "".join(f"{efficiency(truth, -np.angle(h[row])):24.4%}"
                                                     for h in columns.values()), flush=True)


if __name__ == "__main__":
    main()
