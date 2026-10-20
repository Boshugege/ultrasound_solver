"""Amplitude-only inversion of a triangular prism with the BEM (nerual.amplitude_inverse).

Setting of nerual/AMPLITUDE_INVERSE_VISUAL_REPORT.md: 3 x 3 array, amplitudes on
five faces (1280 points), unknown prism centre, rotation, side and thickness.
Observations are synthesised with the CUDA Burton-Miller BEM (algorithm.bem) on a
fine mesh (``--reference-panels`` per side, 0.5 mm by default) and inverted with the
PyTorch double-layer model on ~1.1 mm panels (``--panels`` per side), so the
inversion never sees its own forward model.

validate   reference convergence, model error at the true prism, timing, gradient check
report     the report's three scenes, three fixed starts each, 60 L-BFGS-B iterations
starts     success over random starts, the report's single pattern vs several patterns
scan       coarse scan of centre and in-plane turn, then L-BFGS-B from the best poses
Usage: plate_inverse.py validate|report|starts|scan [--starts N] [--seed S] [--keep K]
                       [--patterns one five single] [--noise 0.02]
"""

import argparse
from pathlib import Path
import sys
import time

import cupy as cp
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from algorithm.bem.solver import RigidBEM, aperture_sources  # noqa: E402
from nerual.amplitude_inverse import (  # noqa: E402
    DIAMETER, SCALE, Prism, PrismInverse, array_centers, compare, focus_patterns,
    observation_points, wavenumber)

SCENES = {
    "flat": Prism.from_euler((25, 23, 23.5), (0, 0, 0), 18, 3),
    "rotated": Prism.from_euler((25, 23, 23.5), (12, -8, 15), 18, 3),
    "shifted": Prism.from_euler((27, 26, 21.5), (0, 0, 0), 18, 3),
}
FIXED_STARTS = (
    Prism.from_euler((24, 24, 20), (0, 0, 0), 15, 4),
    Prism.from_euler((20, 27, 26), (10, 10, 0), 21, 2.5),
    Prism.from_euler((28, 20, 18), (-10, 5, 40), 16, 3.5),
)
# Bounds in params units: centre z, log side, log thickness (others free).
BOUNDS = [(None, None)] * 2 + [(5e-3, 40e-3)] + [(None, None)] * 3 + [
    (np.log(8e-3), np.log(30e-3)), (np.log(1e-3), np.log(8e-3))]
ONE_PATTERN = np.ones((1, 9))
FIVE_PATTERNS = np.vstack([ONE_PATTERN, focus_patterns(
    np.array([[16, 16, 25], [32, 16, 25], [16, 32, 25], [32, 32, 25]]) * 1e-3)])
PATTERN_SETS = {
    "one": ("1 pattern (all phases 0)", ONE_PATTERN),
    "five": ("5 patterns (all 0 + 4 foci)", FIVE_PATTERNS),
    "single": ("9 patterns (one element each)", np.eye(9)),
}


def soup_to_mesh(triangles):
    points = triangles.reshape(-1, 3)
    _, first, inverse = np.unique(np.round(points / 1e-9).astype(np.int64), axis=0,
                                  return_index=True, return_inverse=True)
    return points[first], inverse.reshape(-1, 3)


def reference_field(prism, patterns, panels_per_side):
    """Complex field (P, 1280) from the CUDA Burton-Miller BEM, dense solve."""
    layers = max(1, int(np.ceil(panels_per_side * prism.thickness / prism.side)))
    vertices, faces = soup_to_mesh(prism.triangles(panels_per_side, layers))
    bem = RigidBEM(vertices, faces, k=wavenumber())
    A = bem.assemble()
    points, weights, per = aperture_sources(array_centers(), DIAMETER, 32, 128)
    observed = cp.asarray(observation_points())
    fields = []
    for drive in np.atleast_2d(patterns):
        w = weights * np.repeat(drive, per)
        u = cp.linalg.solve(A, bem.rhs(*bem.sources(bem.geo["centroids"], points, w)))
        fields.append(cp.asnumpy(bem.sources(observed, points, w)[0] + bem.scattered(observed, u)))
    del bem, A
    cp.get_default_memory_pool().free_all_blocks()  # leave the GPU to PyTorch
    return np.array(fields)


def rms(loss):
    return 100 * np.sqrt(loss)


def amplitude_rms(field, reference):
    """The report's amplitude RMS (%) of ``field`` against ``reference``, (P, 1280) each."""
    a, b = (np.abs(f).reshape(len(f), 5, -1) for f in (field, reference))
    return rms(np.mean(((a - b)**2).sum(-1) / (b**2).sum(-1)))


def validate(args):
    for name, prism in SCENES.items():
        fine = reference_field(prism, FIVE_PATTERNS, args.reference_panels)
        coarse = reference_field(prism, FIVE_PATTERNS, args.reference_panels * 2 // 3)
        truth = torch.as_tensor(prism.params(), dtype=torch.float64, device="cuda")
        print(f"{name}: reference {args.reference_panels} vs {args.reference_panels * 2 // 3} panels per side: "
              f"amplitude RMS {amplitude_rms(coarse, fine):.3f} % (5 patterns)")
        for panels in (8, 12, 16, 20):
            model = PrismInverse(np.abs(fine), FIVE_PATTERNS, panels_per_side=panels,
                                 thickness_layers=max(1, round(panels * 3 / 18)))
            with torch.no_grad():
                error = rms(float(model.loss(truth)))
            z = prism.params() / SCALE
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model.value_and_grad(z)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 2**30
            started = time.perf_counter()
            for _ in range(5):
                model.value_and_grad(z)
            torch.cuda.synchronize()
            seconds = (time.perf_counter() - started) / 5
            print(f"  inversion mesh {panels} panels per side ({len(model.template)} panels): amplitude RMS at the "
                  f"truth {error:.3f} % (5 patterns); loss + gradient {seconds * 1e3:.1f} ms, peak {peak:.1f} GiB")
        z = prism.params() / SCALE + np.array([1.0, -0.5, 0.7, 0.3, -0.2, 0.4, 0.5, -0.4])
        model = PrismInverse(np.abs(fine), FIVE_PATTERNS)
        _, grad = model.value_and_grad(z)
        numeric = []
        for i in range(8):
            step = np.zeros(8)
            step[i] = 1e-4
            numeric.append((model.value_and_grad(z + step)[0] - model.value_and_grad(z - step)[0]) / 2e-4)
        numeric = np.array(numeric)
        print(f"  gradient vs central differences: max relative difference "
              f"{np.max(np.abs(grad - numeric)) / np.max(np.abs(numeric)):.2e}")


def measured(prism, patterns, args, rng):
    """Amplitudes of the reference field with relative Gaussian noise ``args.noise``."""
    amplitude = np.abs(reference_field(prism, patterns, args.reference_panels))
    return amplitude * (1 + args.noise * rng.standard_normal(amplitude.shape))


PANELS = 16  # inversion mesh: panels per prism side (set by --panels)


def run(prism, observed, patterns, start, maxiter):
    layers = max(1, round(PANELS * start.thickness / start.side))
    model = PrismInverse(np.abs(observed), patterns, panels_per_side=PANELS, thickness_layers=layers)
    result = model.invert(start, maxiter=maxiter, bounds=BOUNDS)
    del model
    torch.cuda.empty_cache()
    errors = compare(prism, result.prism)
    # 0.25 mm voxels (as in the report) quantise a 3 mm thickness; also sample at 0.05 mm
    errors["iou_fine"] = compare(prism, result.prism, voxel=0.05e-3)["iou"]
    return result, errors


def describe(result, errors):
    return (f"RMS {rms(result.loss):8.4f} %  centre {errors['center_error'] * 1e3:7.3f} mm  "
            f"side {errors['side_error'] * 1e3:+7.3f}  thick {errors['thickness_error'] * 1e3:+6.3f} mm  "
            f"rot {errors['rotation_error_deg']:6.2f} deg  IoU {errors['iou']:6.2%} "
            f"(0.05 mm: {errors['iou_fine']:6.2%})  "
            f"{result.iterations:2d} it / {result.evaluations:3d} ev  {result.seconds:5.1f} s")


def report(args):
    for name, prism in SCENES.items():
        observed = reference_field(prism, ONE_PATTERN, args.reference_panels)
        print(f"{name}: one pattern (all phases 0), 3 fixed starts")
        for index, start in enumerate(FIXED_STARTS):
            result, errors = run(prism, observed, ONE_PATTERN, start, 60)
            print(f"  start {index}: {describe(result, errors)}")


def random_start(rng):
    axis = rng.normal(size=3)
    rotvec = axis / np.linalg.norm(axis) * np.radians(rng.uniform(0, 25))
    return Prism((rng.uniform(18e-3, 30e-3), rng.uniform(18e-3, 30e-3), rng.uniform(16e-3, 30e-3)),
                 tuple(rotvec), rng.uniform(14e-3, 22e-3), rng.uniform(2e-3, 4.5e-3))


def starts(args):
    rng = np.random.default_rng(args.seed)
    draws = [random_start(rng) for _ in range(args.starts)]
    for name, prism in SCENES.items():
        for label, patterns in (PATTERN_SETS[key] for key in args.patterns):
            observed = measured(prism, patterns, args, rng)
            rows = [run(prism, observed, patterns, start, 60) for start in draws]
            losses = np.array([r.loss for r, _ in rows])
            iou = np.array([e["iou"] for _, e in rows])
            centre = np.array([e["center_error"] for _, e in rows]) * 1e3
            seconds = np.array([r.seconds for r, _ in rows])
            success = iou >= 0.9
            best = []  # chance that the lowest-loss start among K is a success
            for k in (1, 3, 10):
                picks = rng.integers(0, len(rows), size=(2000, k))
                chosen = picks[np.arange(2000), np.argmin(losses[picks], axis=1)]
                best.append(success[chosen].mean())
            ok = success.nonzero()[0]
            print(f"{name}, {label}, noise {args.noise:.0%}: {success.mean():.0%} of {len(rows)} random starts "
                  f"reach IoU >= 90 % "
                  f"(median centre error of those {np.median(centre[ok]) if len(ok) else np.nan:.3f} mm); "
                  f"best-of-K by loss succeeds with K=1/3/10: {best[0]:.0%}/{best[1]:.0%}/{best[2]:.0%}; "
                  f"median {np.median(seconds):.1f} s per start")


def scan_candidates(model, keep, side=17e-3, thickness=3.5e-3):
    """Lowest-loss prisms on a grid of centres (1.5 mm) and in-plane turns (30 deg),
    flat and of a nominal size that is not the true one."""
    xs, zs = np.arange(18, 30.01, 1.5) * 1e-3, np.arange(16, 30.01, 1.5) * 1e-3
    found = []
    with torch.no_grad():
        for angle in (0, 30, 60, 90):
            for x in xs:
                for y in xs:
                    for z in zs:
                        prism = Prism((x, y, z), (0.0, 0.0, np.radians(angle)), side, thickness)
                        params = torch.as_tensor(prism.params(), dtype=torch.float64, device="cuda")
                        found.append((float(model.loss(params)), prism))
    found.sort(key=lambda item: item[0])
    return [prism for _, prism in found[:keep]], len(found)


def scan(args):
    rng = np.random.default_rng(args.seed)
    for name, prism in SCENES.items():
        for label, patterns in (PATTERN_SETS[key] for key in args.patterns):
            observed = measured(prism, patterns, args, rng)
            torch.cuda.synchronize()
            started = time.perf_counter()
            coarse = PrismInverse(np.abs(observed), patterns, panels_per_side=8, thickness_layers=1)
            candidates, count = scan_candidates(coarse, args.keep)
            del coarse
            torch.cuda.synchronize()
            scanned = time.perf_counter() - started
            rows = [run(prism, observed, patterns, start, 60) for start in candidates]
            result, errors = min(rows, key=lambda row: row[0].loss)
            refine = sum(r.seconds for r, _ in rows)
            hits = sum(e["iou"] >= 0.9 for _, e in rows)
            print(f"{name}, {label}, noise {args.noise:.0%}: scan of {count} poses {scanned:.1f} s + {len(rows)} refinements {refine:.1f} s "
                  f"({hits} of them reach IoU >= 90 %); lowest loss: {describe(result, errors)}")


def main():
    global PANELS
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "report", "starts", "scan"))
    parser.add_argument("--starts", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference-panels", type=int, default=36)
    parser.add_argument("--keep", type=int, default=5, help="scan: candidates refined with L-BFGS-B")
    parser.add_argument("--patterns", nargs="+", choices=tuple(PATTERN_SETS), default=["one", "five"])
    parser.add_argument("--noise", type=float, default=0.0, help="relative amplitude noise (std)")
    parser.add_argument("--panels", type=int, default=PANELS, help="inversion mesh, panels per prism side")
    args = parser.parse_args()
    PANELS = args.panels
    {"validate": validate, "report": report, "starts": starts, "scan": scan}[args.command](args)


if __name__ == "__main__":
    main()
