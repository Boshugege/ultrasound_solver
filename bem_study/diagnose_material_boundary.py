"""Where is the effective surface of upstream's smoothed material obstacle?

Upstream blends rho and c linearly across +-1.2 cells around the SDF zero level.
With rho_obstacle / rho_air ~ 1000, the blend is already acoustically rigid well
outside the zero level, so the PDE sphere should behave like a larger rigid sphere.
For each PDE resolution of ``check_obstacle_vs_pde.py`` the scattered field is
compared with BEM rigid spheres of radius r + delta.
Usage: diagnose_material_boundary.py NODES_PER_AXIS [...]
"""

from pathlib import Path
import sys

import cupy as cp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_obstacle_vs_pde as scene  # noqa: E402
from algorithm.bem.solver import RigidBEM, aperture_sources, icosphere  # noqa: E402
from algorithm.physics.transducer_array import TransducerArray  # noqa: E402

OFFSETS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)  # in grid cells


def bem_scattered(cfg, points, radius):
    centers = TransducerArray(cfg).centers
    quad, weights, per = aperture_sources(centers, cfg.specs.diameter, 16, 64)
    subdivisions = 0
    while radius * 1.0515 / 2 ** subdivisions > cfg.wavelength / 8:
        subdivisions += 1
    bem = RigidBEM(*icosphere(radius, subdivisions, center=scene.CENTER), k=cfg.k0)
    bem.assemble()
    out = np.empty((len(points), len(centers)), dtype=complex)
    for j in range(len(centers)):
        q, w = quad[j * per:(j + 1) * per], weights[j * per:(j + 1) * per]
        u, _ = bem.solve(bem.incident_on_surface(lambda x: bem.sources(x, q, w)), rtol=1e-6)
        out[:, j] = cp.asnumpy(bem.scattered(cp.asarray(points), u))
    return out


def main():
    points = scene.sample_points()
    for n in [int(v) for v in sys.argv[1:]]:
        free, _, cfg = scene.pde_fields(n, False, points)
        with_sphere, _, _ = scene.pde_fields(n, True, points)
        pde = with_sphere - free
        errors = []
        for cells in OFFSETS:
            bem = bem_scattered(cfg, points, scene.RADIUS + cells * cfg.domain.dx)
            errors.append(np.linalg.norm(pde - bem) / np.linalg.norm(bem))
        best = OFFSETS[int(np.argmin(errors))]
        print(f"PDE {n}^3 (dx = {cfg.domain.dx * 1e3:.3f} mm): scattered-field difference to a rigid sphere of "
              f"radius r + delta: " + ", ".join(f"{c:.2f} cells {e:.3f}" for c, e in zip(OFFSETS, errors))
              + f"  -> best delta = {best} cells = {best * cfg.domain.dx * 1e3:.2f} mm", flush=True)


if __name__ == "__main__":
    main()
