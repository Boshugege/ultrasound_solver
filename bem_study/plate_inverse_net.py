"""Learned starting pose for the amplitude inversion (nerual.inverse_start).

From a random start L-BFGS-B succeeds only 20-35 % of the time (plate_inverse.py
starts); the coarse scan (plate_inverse.py scan) is reliable but takes ~33 s. Here
PoseNet predicts the pose from the amplitudes and one refinement starts from it;
locate() falls back to the scan when the fit stays poor.

Training data: the PyTorch BEM model (12 panels per side) at random prisms of
nerual.inverse_start.PRIOR, for the report's pattern (all phases 0) and the nine
one-element patterns at once. Test data: the CUDA Burton-Miller BEM (24 panels per
side) at fresh prisms and at the report's three scenes; --wide draws test prisms
from a wider range, mostly outside the training prior.

generate   training samples into outputs/plate_inverse/training.npz (~9 min for 30000)
train      fit PoseNet for a pattern set (seconds)
evaluate   guess, refine once with 16 panels per side, scan if the fit stays poor
Usage: plate_inverse_net.py generate [--samples N] | train --patterns one|single
       | evaluate --patterns one|single [--tests N] [--noise 0.02] [--wide] [--accept-rms 0.025]
"""

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bem_study"))
from plate_inverse import BOUNDS, PANELS, SCENES, describe, reference_field  # noqa: E402
from nerual.amplitude_inverse import PrismInverse, compare  # noqa: E402
from nerual.inverse_start import (  # noqa: E402
    PRIOR, PoseGuess, locate, network_inputs, pose_features, sample_prism, train)

DATA = ROOT / "outputs" / "plate_inverse"
PATTERNS = {"one": np.ones((1, 9)), "single": np.eye(9)}
WIDE = {"x": (15e-3, 33e-3), "z": (14e-3, 33e-3), "tilt": 35.0, "side": (12e-3, 24e-3),
        "thickness": (1.5e-3, 5e-3)}


def inside(prism, prior=PRIOR):
    f = pose_features(prism)
    tilt = np.degrees(np.arcsin(min(1.0, np.linalg.norm(f[5:7]))))
    return (all(prior["x"][0] <= c <= prior["x"][1] for c in prism.center[:2])
            and prior["z"][0] <= prism.center[2] <= prior["z"][1] and tilt <= prior["tilt"]
            and prior["side"][0] <= prism.side <= prior["side"][1]
            and prior["thickness"][0] <= prism.thickness <= prior["thickness"][1])


def generate(args):
    DATA.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    model = PrismInverse(np.ones((9, 1280)), np.eye(9), panels_per_side=12, thickness_layers=2)
    incident = model.incident.T.cpu().numpy()  # (9, M): one-element patterns
    single = np.empty((args.samples, 9, 1280), np.float16)
    one = np.empty((args.samples, 1280), np.float16)
    targets = np.empty((args.samples, 9), np.float32)
    started = time.perf_counter()
    with torch.no_grad():
        for i in range(args.samples):
            prism = sample_prism(rng)
            field = model.field(torch.as_tensor(prism.params(), dtype=torch.float64, device="cuda")).cpu().numpy()
            single[i] = network_inputs(np.abs(field), incident)
            one[i] = network_inputs(np.abs(field.sum(0)), incident.sum(0))  # all drives 1: sum of fields
            targets[i] = pose_features(prism)
            if (i + 1) % 5000 == 0:
                print(f"{i + 1} samples, {(time.perf_counter() - started) / (i + 1) * 1e3:.1f} ms each", flush=True)
    np.savez(DATA / "training.npz", single=single, one=one, targets=targets)


def train_command(args):
    data = np.load(DATA / "training.npz")
    started = time.perf_counter()
    bundle, errors = train(data[args.patterns], data["targets"], epochs=args.epochs)
    bundle["patterns"] = PATTERNS[args.patterns]
    torch.save(bundle, DATA / f"posenet_{args.patterns}.pt")
    print(f"{args.patterns}: {len(data['targets'])} samples, {time.perf_counter() - started:.0f} s; validation "
          f"centre error median {np.median(errors):.2f} mm, p90 {np.percentile(errors, 90):.2f} mm, "
          f"max {errors.max():.2f} mm")


def evaluate(args):
    bundle = torch.load(DATA / f"posenet_{args.patterns}.pt", weights_only=False)
    patterns = PATTERNS[args.patterns]
    rng = np.random.default_rng(args.seed + 1000)
    tests = ([] if args.wide else list(SCENES.items())) + [
        (f"random {i}", sample_prism(rng, WIDE if args.wide else PRIOR)) for i in range(args.tests)]
    rows = []
    for name, prism in tests:
        amplitude = np.abs(reference_field(prism, patterns, 24))
        amplitude = amplitude * (1 + args.noise * rng.standard_normal(amplitude.shape))
        coarse = PrismInverse(amplitude, patterns, panels_per_side=8, thickness_layers=1)
        torch.cuda.synchronize()
        started = time.perf_counter()
        guess = PoseGuess(bundle, coarse.incident.T.cpu().numpy())(amplitude)
        seconds = time.perf_counter() - started
        layers = max(1, round(PANELS * guess.thickness / guess.side))
        model = PrismInverse(amplitude, patterns, panels_per_side=PANELS, thickness_layers=layers)
        torch.cuda.synchronize()
        started = time.perf_counter()
        result, method = locate(model, guess, coarse, accept_rms=args.accept_rms, bounds=BOUNDS)
        seconds += time.perf_counter() - started
        errors = compare(prism, result.prism)
        errors["iou_fine"] = compare(prism, result.prism, voxel=0.05e-3)["iou"]
        off = compare(prism, guess)["center_error"] * 1e3
        rows.append((off, errors["iou_fine"] >= 0.9, seconds, inside(prism), method == "scan"))
        print(f"{name:10s}{'' if inside(prism) else ' (outside the prior)'}: guess {off:.2f} mm off, {method}; "
              f"{describe(result, errors)}; {seconds:.1f} s in total", flush=True)
        del model, coarse
        torch.cuda.empty_cache()
    off, success, seconds, within, scanned = (np.array(v) for v in zip(*rows))
    for label, mask in (("inside", within), ("outside", ~within)):
        if mask.any():
            print(f"{args.patterns}, noise {args.noise:.0%}, {mask.sum()} test prisms {label} the training prior: "
                  f"guess within 2 mm for {np.mean(off[mask] < 2):.0%} (median {np.median(off[mask]):.2f} mm); "
                  f"success {success[mask].mean():.0%} (IoU >= 90 % at 0.05 mm); scan needed for "
                  f"{scanned[mask].mean():.0%}; time median {np.median(seconds[mask]):.1f} s, "
                  f"max {seconds[mask].max():.1f} s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("generate", "train", "evaluate"))
    parser.add_argument("--samples", type=int, default=30000)
    parser.add_argument("--patterns", choices=tuple(PATTERNS), default="one")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--tests", type=int, default=30)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--wide", action="store_true", help="evaluate: test prisms from a wider range")
    parser.add_argument("--accept-rms", type=float, default=0.025,
                        help="evaluate: amplitude RMS above which locate() falls back to the scan")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    {"generate": generate, "train": train_command, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
