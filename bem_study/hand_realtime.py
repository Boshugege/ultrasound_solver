"""Hand above the SonicSurface array (scene of src/examples/bem_hand_16x16.yaml).

1. Mesh convergence of the response row at the focus (surface elements of
   lambda/3 ... lambda/6, GMRES rtol 1e-6) against lambda/6 (29k elements, the
   largest dense matrix that leaves room on a 24 GB GPU).
2. Real-time frames at lambda/4 and lambda/5 with FocusSession's defaults: one cold
   frame, then frames after the hand moves by (2, 1, 1) mm while the focus stays put
   (rigid motion: the hand's own matrix block is reused, the solve starts from the
   previous surface field). Moving hand and focus together is a pure
   translation along the baffle and would make the warm start exact.
3. What the hand changes: focus pressure (fraction of the optimum in the BEM model
   with the hand) for phases computed without the hand vs with the hand.
"""

from pathlib import Path
import sys
import time

import cupy as cp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from algorithm.bem.mesh import oriented_arrays, remesh  # noqa: E402
from algorithm.bem.realtime import FocusSession  # noqa: E402
from algorithm.config import SimulationConfig  # noqa: E402
from algorithm.physics.mesh_to_sdf import _load_closed_surface, _place_mesh  # noqa: E402
from algorithm.physics.transducer_array import TransducerArray  # noqa: E402

CONFIG = ROOT / "src" / "examples" / "bem_hand_16x16.yaml"


def efficiency(row, phases):
    return abs(np.sum(row * np.exp(1j * phases))) / np.abs(row).sum()


def main():
    cfg = SimulationConfig.from_yaml(CONFIG)
    k, wavelength = cfg.k0, cfg.wavelength
    centers = TransducerArray(cfg).centers
    target = np.asarray(cfg.targets[0], float)
    obstacle = cfg.obstacles[0]
    placed = _place_mesh(_load_closed_surface(Path(obstacle["file"])), np.asarray(obstacle["center_m"]),
                         np.asarray(obstacle["rotation_deg"], float), float(obstacle["scale"]))
    bounds = np.asarray(placed.bounds)
    print(f"hand: {np.ptp(bounds.reshape(3, 2), axis=1).round(4).tolist()} m, lowest point z = "
          f"{bounds[4] * 1e3:.1f} mm, target {target.tolist()}", flush=True)

    rows, meshes = {}, {}
    for per in (3, 4, 5, 6):
        vertices, faces = oriented_arrays(remesh(placed, wavelength / per))
        meshes[per] = (vertices, faces)
        session = FocusSession(centers, cfg.specs.diameter, k=k, rtol=1e-6, scattered_orders=(6, 16),
                               block_size=None, half_precision=False)
        session.set_surface(vertices, faces)
        rows[per] = session.response_row(target)
        print(f"lambda/{per}: {len(faces)} elements, {session.timings['iterations']} GMRES iterations", flush=True)
        del session
        cp.get_default_memory_pool().free_all_blocks()
    finest = rows[6]
    for per in (3, 4, 5):
        print(f"  lambda/{per} vs lambda/6: |row| difference {np.linalg.norm(rows[per] - finest) / np.linalg.norm(finest):.2e}, "
              f"focus with its phases {efficiency(finest, -np.angle(rows[per])):.5%}")

    for per in (4, 5):
        vertices, faces = meshes[per]
        session = FocusSession(centers, cfg.specs.diameter, k=k)

        def frame(shift, rigid):
            cp.cuda.Stream.null.synchronize()
            started = time.perf_counter()
            session.set_surface(vertices + shift, faces, rigid_motion=rigid)
            row = session.response_row(target)
            total = time.perf_counter() - started
            t = session.timings
            print(f"  geometry {t['geometry'] * 1e3:4.0f} + assemble {t['assemble'] * 1e3:4.0f} + incident "
                  f"{t['incident'] * 1e3:3.0f} + solve {t['solve'] * 1e3:4.0f} ({t['iterations']:2d} it) + aperture "
                  f"integrals {(t['direct'] + t['scattered']) * 1e3:4.0f} = {total * 1e3:5.0f} ms", flush=True)
            return row

        frame(np.zeros(3), False)  # compiles the kernels
        session._previous.clear()
        print(f"real-time frames, lambda/{per} ({len(faces)} elements), FocusSession defaults:")
        print(" cold frame:")
        fast = frame(np.zeros(3), False)
        print(" hand moves (2, 1, 1) mm per frame, focus fixed (rigid motion, warm start):")
        for step in range(1, 6):
            frame(np.array([0.002, 0.001, 0.001]) * step, True)
        print(f"  real-time vs tight solve: focus with its phases {efficiency(rows[per], -np.angle(fast)):.5%}, "
              f"max phase difference {np.degrees(np.abs(np.angle(fast / rows[per]))).max():.2f} deg", flush=True)
        del session
        cp.get_default_memory_pool().free_all_blocks()

    empty = FocusSession(centers, cfg.specs.diameter, k=k)
    empty.set_surface(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
    free = empty.response_row(target)
    print("focus pressure in the model with the hand, as a fraction of its optimum:")
    print(f"  phases computed without the hand: {efficiency(finest, -np.angle(free)):.4f}")
    for per in (4, 5):
        print(f"  phases computed with the hand (lambda/{per}): {efficiency(finest, -np.angle(rows[per])):.4f}")
    print(f"  optimum |p| with hand / without hand: {np.abs(finest).sum() / np.abs(free).sum():.4f}")


if __name__ == "__main__":
    main()
