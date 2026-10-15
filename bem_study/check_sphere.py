"""BEM accuracy against an exact solution: rigid sphere in free space, point source.

Radius 2 cm (k a = 14.5 at 40 kHz, c = 346 m/s), monopole 6 cm from the centre;
total field at three points around the sphere vs the modal series. Checks the
element integrals, the Burton-Miller combination and the singular self terms.
"""

from pathlib import Path
import sys
import time

import cupy as cp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from algorithm.bem.solver import RigidBEM, icosphere, rigid_sphere_point_source  # noqa: E402

K = 2 * np.pi * 40000.0 / 346.0
RADIUS, SOURCE = 0.02, np.array([0.0, 0.0, 0.06])
PROBES = np.array([[0.03, 0.0, 0.0], [0.0, -0.025, 0.01], [0.0, 0.0, -0.03]])


def main():
    exact = rigid_sphere_point_source(PROBES, SOURCE, (0, 0, 0), RADIUS, K)
    wavelength = 2 * np.pi / K
    print(f"rigid sphere a = {RADIUS * 100:.0f} cm, k a = {K * RADIUS:.1f}, source at 3 a")
    for subdivisions in (3, 4, 5):
        bem = RigidBEM(*icosphere(RADIUS, subdivisions), k=K, planes=())
        started = time.perf_counter()
        bem.assemble()
        u, info = bem.solve(bem.incident_on_surface(lambda x: bem.point_source(x, SOURCE)), rtol=1e-6)
        incident, _ = bem.point_source(cp.asarray(PROBES), SOURCE)
        total = cp.asnumpy(incident + bem.scattered(cp.asarray(PROBES), u))
        cp.cuda.Stream.null.synchronize()
        edge = float(cp.asnumpy(bem.geo["hsize"]).mean())
        print(f"{bem.n:6d} elements, edge {edge * 1e3:4.2f} mm (lambda/{wavelength / edge:4.1f}): "
              f"error {np.linalg.norm(total - exact) / np.linalg.norm(exact):.2e}, "
              f"GMRES {info.iterations} it, {time.perf_counter() - started:.2f} s", flush=True)


if __name__ == "__main__":
    main()
