"""Where does a real-time frame spend its time, and what would a better start buy?

Hand of src/examples/bem_hand_16x16.yaml at lambda/4, focus fixed, hand moving
(2, 1, 1) mm per frame for 10 frames (rigid motion). Per frame: timings of each
step, GMRES iterations and the relative residual of the warm start, and the
accuracy of the row against a reference (GMRES rtol 1e-6, fine aperture
quadrature): focus efficiency of its phases and the largest weighted phase error.
Options select the speed-ups under test.
Usage: speed_profile.py [--scattered-orders RxA] [--rtol X] [--frames N]
"""

import argparse
from pathlib import Path
import sys
import time

import cupy as cp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from algorithm.bem.mesh import oriented_arrays, remesh  # noqa: E402
from algorithm.bem.realtime import Apertures, FocusSession, aperture_response  # noqa: E402
from algorithm.config import SimulationConfig  # noqa: E402
from algorithm.physics.mesh_to_sdf import _load_closed_surface, _place_mesh  # noqa: E402
from algorithm.physics.transducer_array import TransducerArray  # noqa: E402

CONFIG = ROOT / "src" / "examples" / "bem_hand_16x16.yaml"


def hand(cfg, per_wavelength):
    obstacle = cfg.obstacles[0]
    placed = _place_mesh(_load_closed_surface(Path(obstacle["file"])), np.asarray(obstacle["center_m"]),
                         np.asarray(obstacle["rotation_deg"], float), float(obstacle["scale"]))
    return oriented_arrays(remesh(placed, cfg.wavelength / per_wavelength))


def synced():
    cp.cuda.Stream.null.synchronize()
    return time.perf_counter()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scattered-orders", default="3x8", help="radial x angular, e.g. 6x16")
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--per-wavelength", type=float, default=4.0)
    parser.add_argument("--block-size", type=int, default=512, help="block-Jacobi preconditioner, 0 = off")
    parser.add_argument("--single", action="store_true", help="complex64 instead of float16 matrix storage")
    args = parser.parse_args()
    orders = tuple(int(v) for v in args.scattered_orders.lower().split("x"))
    cfg = SimulationConfig.from_yaml(CONFIG)
    centers, k, target = TransducerArray(cfg).centers, cfg.k0, np.asarray(cfg.targets[0], float)
    vertices, faces = hand(cfg, args.per_wavelength)
    session = FocusSession(centers, cfg.specs.diameter, k=k, rtol=args.rtol, scattered_orders=orders,
                           block_size=args.block_size or None, half_precision=not args.single)
    reference_quad = Apertures(centers, cfg.specs.diameter, (16, 64))
    print(f"hand lambda/{args.per_wavelength:g}: {len(faces)} elements; scattered aperture orders {orders}, "
          f"GMRES rtol {args.rtol:g}, block-Jacobi {args.block_size or 'off'}, "
          f"fp16 {not args.single}",
          flush=True)

    session.set_surface(vertices, faces)  # compile kernels
    session.response_row(target)
    session._previous.clear()
    rows = []
    for frame in range(args.frames + 1):
        shift = np.array([0.002, 0.001, 0.001]) * frame
        moved = vertices + shift
        # warm-start quality, measured outside the timed frame
        start_residual = 1.0
        t0 = synced()
        session.set_surface(moved, faces, rigid_motion=frame > 0)
        bem = session.bem
        if 0 in session._previous:
            t_probe = synced()
            incident, gradient = bem.point_source(bem.geo["centroids"], target)
            u_old, value_old = session._previous[0]
            x0 = (u_old * (incident / value_old)).astype(cp.complex64)
            b = bem.rhs(incident, gradient)
            start_residual = float(cp.linalg.norm(b - bem.apply(x0)) / cp.linalg.norm(b))
            t0 += synced() - t_probe  # exclude the probe from the frame time
        row = session.response_row(target)
        total = synced() - t0
        t = session.timings
        v = cp.ones(bem.n, dtype=cp.complex64)
        t6 = synced()
        for _ in range(10):
            v = bem.apply(v)
        matvec = (synced() - t6) / 10
        reference, _, _ = aperture_response(bem, target, session.direct, reference_quad, 1e-6)
        reference = cp.asnumpy(reference)
        weights = np.abs(reference) / np.abs(reference).sum()
        phase_error = np.degrees(np.abs(np.angle(row / reference)))
        efficiency = abs(np.sum(reference * np.exp(-1j * np.angle(row)))) / np.abs(reference).sum()
        parts = {"surface": t["geometry"] + t["assemble"], "incident": t["incident"], "solve": t["solve"],
                 "aperture integrals": t["direct"] + t["scattered"]}
        label = "cold" if frame == 0 else f"move {frame}"
        print(f"{label:7s} " + " + ".join(f"{name} {value * 1e3:5.1f}" for name, value in parts.items())
              + f" -> {total * 1e3:5.1f} ms | {t['iterations']:2d} it, start residual {start_residual:.2f}, "
              f"{t['solve'] / max(t['iterations'], 1) * 1e3:.2f} ms/it (matvec {matvec * 1e3:.2f}) | "
              f"focus {efficiency:.5%}, weighted phase error {np.sum(weights * phase_error):.3f} deg, "
              f"max {phase_error.max():.2f} deg", flush=True)
        if frame:
            rows.append((total, t["iterations"], efficiency))
    total, iterations, efficiency = np.array(rows).T
    print(f"moving frames: mean {total.mean() * 1e3:.1f} ms, {iterations.mean():.1f} iterations, "
          f"focus min {efficiency.min():.5%}")


if __name__ == "__main__":
    main()
