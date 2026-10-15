"""BEM as the default field model for upstream's control demos: same answers, faster?

Every control demo (control/front_focus_alg.py, circle.py, line.py,
horizontal_line.py, cube.py) focuses one point at a time in the free field of
src/examples/pde_homogeneous_16x16.yaml, so it only needs the response vector h
of each target (optimal phases -arg h for a single focus). For all demo points
plus the eight targets of docs/PDE_PHASE_ALIGNMENT.md this compares upstream's PDE
(HelmholtzDirectSolver.target_response, 1/3 mm grid, fresh cache) with the BEM
(BemSceneSolver.target_response, the exact aperture integral of the same source):
time, optimum amplitude, phases (common offset removed), how much focus each
model's phases reach in the other model, and the 32-level hardware quantisation.
"""

from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "control")]
import circle  # noqa: E402
import cube  # noqa: E402
import horizontal_line  # noqa: E402
import line  # noqa: E402
from algorithm.bem.scene_solver import BemSceneSolver  # noqa: E402
from algorithm.config import SimulationConfig  # noqa: E402
from algorithm.solvers import HelmholtzDirectSolver  # noqa: E402

CONFIG = ROOT / "src" / "examples" / "pde_homogeneous_16x16.yaml"
CACHE = ROOT / "outputs" / "bem_study" / "compare_default" / "pde_target_responses.npz"
ALIGNMENT_MM = [(80, 80, 120), (115, 55, 100), (80, 80, 70), (80, 80, 135),
                (120, 80, 110), (120, 120, 110), (45, 115, 100), (140, 40, 80)]


def demo_points():
    groups = {"front focus": np.array([[0.08, 0.08, 0.12]])}
    groups["circle"] = circle.CirclePath(circle.CIRCLE_CENTER_X_M, circle.CIRCLE_CENTER_Y_M, circle.TARGET_HEIGHT_M,
                                         circle.CIRCLE_RADIUS_M, circle.PATH_POINT_COUNT).discretize()
    groups["line"] = line.LinePath(line.TARGET_X_M, line.TARGET_Y_M, line.START_HEIGHT_M, line.END_HEIGHT_M,
                                   line.PATH_POINT_COUNT).discretize()
    groups["horizontal line"] = horizontal_line.HorizontalLinePath(
        horizontal_line.START_X_M, horizontal_line.END_X_M, horizontal_line.TARGET_Y_M,
        horizontal_line.TARGET_HEIGHT_M, horizontal_line.PATH_POINT_COUNT).discretize()
    groups["cube"] = cube.CubeEdgePath(cube.CUBE_CENTER_X_M, cube.CUBE_CENTER_Y_M, cube.CUBE_CENTER_Z_M,
                                       cube.CUBE_EDGE_LENGTH_M, cube.POINTS_PER_EDGE).discretize()[0]
    groups["alignment targets"] = np.array(ALIGNMENT_MM, float) * 1e-3
    return groups


def quantise(phases):
    return np.floor(np.mod(phases, 2 * np.pi) * 32 / (2 * np.pi)).astype(int) % 32


def compare(h_bem, h_pde):
    """Per-point metrics for single-focus phases -arg(h)."""
    phi_b, phi_p = -np.angle(h_bem), -np.angle(h_pde)
    d = np.exp(1j * (phi_p - phi_b))
    offset = np.angle(d.mean())
    error = np.degrees(np.abs(np.angle(d * np.exp(-1j * offset))))
    q_b, q_p = quantise(phi_b), quantise(phi_p - offset)
    bins = np.minimum(np.abs(q_b - q_p), 32 - np.abs(q_b - q_p))
    return {
        "optimum BEM/PDE": np.abs(h_bem).sum() / np.abs(h_pde).sum(),
        "phase MAE": error.mean(), "phase max": error.max(),
        "PDE phases in exact model": abs(np.sum(h_bem * np.exp(1j * phi_p))) / np.abs(h_bem).sum(),
        "BEM phases in PDE model": abs(np.sum(h_pde * np.exp(1j * phi_b))) / np.abs(h_pde).sum(),
        "same 32-level bin": np.count_nonzero(bins == 0), "max bin difference": bins.max(),
        "quantised PDE phases in exact model": abs(np.sum(h_bem * np.exp(1j * (q_p + 0.5) * 2 * np.pi / 32)))
        / np.abs(h_bem).sum(),
        "quantised BEM phases in exact model": abs(np.sum(h_bem * np.exp(1j * (q_b + 0.5) * 2 * np.pi / 32)))
        / np.abs(h_bem).sum(),
    }


def main():
    groups = demo_points()
    points = np.vstack(list(groups.values()))
    print(f"{len(points)} target points: " + ", ".join(f"{k} {len(v)}" for k, v in groups.items()), flush=True)

    cfg = SimulationConfig.from_yaml(CONFIG)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.unlink(missing_ok=True)
    cfg.solver["target_response_cache"] = str(CACHE)
    started = time.perf_counter()
    pde_solver = HelmholtzDirectSolver(cfg)
    h_pde = pde_solver.target_response(points)
    pde_solver.close()
    pde_seconds = time.perf_counter() - started
    print(f"upstream PDE ({cfg.domain.grid_size}, dx = {cfg.domain.dx * 1e3:.3f} mm): {pde_seconds:.1f} s", flush=True)

    started = time.perf_counter()
    bem_solver = BemSceneSolver(cfg)
    setup = time.perf_counter() - started
    started = time.perf_counter()
    h_bem = bem_solver.target_response(points)
    bem_seconds = time.perf_counter() - started
    print(f"BEM: setup {setup:.2f} s, responses {bem_seconds:.2f} s "
          f"({bem_seconds / len(points) * 1e3:.1f} ms per point)", flush=True)

    metrics = [compare(b, p) for b, p in zip(h_bem, h_pde)]
    keys = list(metrics[0])
    print("\nper group (mean over points; max for 'phase max' and 'max bin difference'; min for efficiencies):")
    start = 0
    for name, group in list(groups.items()) + [("all points", points)]:
        rows = metrics[start:start + len(group)] if name != "all points" else metrics
        start += len(group) if name != "all points" else 0
        summary = []
        for key in keys:
            values = np.array([row[key] for row in rows])
            if key in ("phase max", "max bin difference"):
                summary.append(f"{key} {values.max():.4g}")
            elif "model" in key:
                summary.append(f"{key} {values.min():.5%}")
            elif key == "same 32-level bin":
                summary.append(f"{key} {values.mean():.0f}/256")
            else:
                summary.append(f"{key} {values.mean():.4g}")
        print(f"  {name} ({len(rows)}): " + "; ".join(summary), flush=True)


if __name__ == "__main__":
    main()
