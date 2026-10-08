"""Run a consistent multi-algorithm comparison for one phase-optimization YAML."""

import argparse
import copy
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithm.baselines import ALGORITHMS, run, validate_algorithm
from algorithm.config import SimulationConfig
from algorithm.baselines.loss_curve import (
    ComparisonLossDashboard,
    save_loss_dashboard,
)
from main import save_phase_optimization_result
from algorithm.solvers import HelmholtzDirectSolver


QUALITY_TARGETS = (
    ("response_alignment", -1220.675),
    ("lbfgsb_reference", -1593.498),
    ("high_quality", -1800.0),
    ("near_adam_quality", -1860.0),
)


PROFILES = {
    "geometric": {
        "algorithm_options": {},
        "training": {"iterations": 0, "max_evaluations": 8, "max_seconds": 60.0},
    },
    "response_alignment": {
        "algorithm_options": {},
        "training": {"iterations": 0, "max_evaluations": 8, "max_seconds": 60.0},
    },
    "adjoint": {
        "algorithm_options": {
            "optimizer": "adam",
            "learning_rate": 0.05,
            "gradient_check": False,
            "convergence_patience": 100,
            "convergence_relative_tolerance": 1.0e-5,
        },
        "training": {
            "iterations": 5000,
            "max_evaluations": 15000,
            "max_seconds": 60.0,
        },
    },
    "gabs": {
        "algorithm_options": {"phase_levels": 8},
        "training": {
            "iterations": 10,
            "max_evaluations": 15000,
            "max_seconds": 60.0,
            "loss_curve_update_interval": 16,
        },
    },
    "spsa": {
        "algorithm_options": {
            "learning_rate": 0.05,
            "perturbation": 0.10,
            "alpha": 0.602,
            "gamma": 0.101,
        },
        "training": {
            "iterations": 5000,
            "max_evaluations": 15000,
            "max_seconds": 60.0,
            "loss_curve_update_interval": 5,
        },
    },
    "cmaes": {
        "algorithm_options": {"sigma": 0.50, "population_size": 24},
        "training": {
            "iterations": 600,
            "max_evaluations": 15000,
            "max_seconds": 60.0,
            "loss_curve_update_interval": 12,
        },
    },
    "lshade": {
        "algorithm_options": {
            "population_size": 32,
            "min_population_size": 8,
            "memory_size": 6,
            "p_best_rate": 0.15,
            "convergence_patience": 25,
            "convergence_relative_tolerance": 2.0e-4,
        },
        "training": {
            "iterations": 2000,
            "max_evaluations": 15000,
            "max_seconds": 60.0,
            "loss_curve_update_interval": 24,
        },
    },
    "sac": {
        "algorithm_options": {
            "total_timesteps": 8000,
            "episode_steps": 32,
            "evaluation_steps": 32,
            "action_scale": 0.20,
            "reward_scale": 100.0,
            "learning_rate": 3.0e-4,
            "batch_size": 64,
            "learning_starts": 256,
            "buffer_size": 50000,
            "random_reset": True,
            "device": "cpu",
            "run_mode": "train",
        },
        "training": {
            "iterations": 1,
            "max_evaluations": 15000,
            "max_seconds": 60.0,
            "loss_curve_update_interval": 50,
        },
    },
    "ppo": {
        "algorithm_options": {
            "total_timesteps": 8192,
            "episode_steps": 32,
            "evaluation_steps": 32,
            "action_scale": 0.20,
            "reward_scale": 100.0,
            "learning_rate": 3.0e-4,
            "batch_size": 64,
            "n_steps": 256,
            "random_reset": True,
            "device": "cpu",
            "run_mode": "train",
        },
        "training": {
            "iterations": 1,
            "max_evaluations": 15000,
            "max_seconds": 60.0,
            "loss_curve_update_interval": 50,
        },
    },
}


def _copy_profile(
    base_cfg, algorithm, output_dir, show_loss_curve, max_seconds=None
):
    cfg = copy.deepcopy(base_cfg)
    profile = PROFILES[algorithm]
    cfg.algorithm = algorithm
    cfg.algorithm_options = copy.deepcopy(profile["algorithm_options"])
    for key, value in profile["training"].items():
        setattr(cfg.training, key, value)
    if max_seconds is not None:
        cfg.training.max_seconds = float(max_seconds)
    # Iterative methods use the same information-limited initial point.
    if algorithm not in {"geometric", "response_alignment"}:
        cfg.training.initial_phase = "geometric"
    cfg.training.compare_geometric = algorithm != "geometric"
    cfg.training.show_loss_curve = show_loss_curve
    cfg.io.auto_visualize = False
    method_dir = output_dir / algorithm
    cfg.io.output_file = str(method_dir / "result.npz")
    if algorithm in {"sac", "ppo"}:
        cfg.algorithm_options["checkpoint"] = str(method_dir / "policy.zip")
    return cfg


def _summary_row(algorithm, status, metadata=None, error=None):
    row = {"algorithm": algorithm, "status": status, "error": error or ""}
    if metadata:
        best = metadata["best_metrics"]
        row.update(
            best_loss=metadata["best_loss"],
            final_loss=metadata["final_loss"],
            target_mean=best["target_mean"],
            background_max=best["background_max"],
            contrast=best["contrast"],
            field_evaluations=metadata["field_evaluations"],
            vjp_evaluations=metadata["vjp_evaluations"],
            optimization_seconds=metadata["elapsed_seconds"],
            time_budget_seconds=metadata.get("time_budget_seconds"),
            termination_reason=metadata["termination_reason"],
        )
    return row


def _write_summary(output_dir, base_cfg, rows, basis_seconds, selected_algorithms):
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_config": str(base_cfg),
        "algorithms": selected_algorithms,
        "shared_basis_build_or_load_seconds": basis_seconds,
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    keys = sorted({key for row in rows for key in row})
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _quality_rows(histories):
    rows = []
    for algorithm, history in histories.items():
        for name, threshold in QUALITY_TARGETS:
            reached = next(
                (
                    row
                    for row in history
                    if row["best_loss"] <= threshold
                ),
                None,
            )
            rows.append(
                {
                    "algorithm": algorithm,
                    "quality_target": name,
                    "loss_threshold": threshold,
                    "reached": reached is not None,
                    "field_evaluations_to_target": (
                        reached["evaluation"] if reached is not None else None
                    ),
                    "seconds_to_target": (
                        reached["elapsed_seconds"] if reached is not None else None
                    ),
                }
            )
    return rows


def _write_quality_summary(output_dir, histories):
    rows = _quality_rows(histories)
    payload = {
        "quality_targets": [
            {"name": name, "loss_threshold": threshold}
            for name, threshold in QUALITY_TARGETS
        ],
        "rows": rows,
    }
    (output_dir / "quality_summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    with (output_dir / "quality_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "algorithm",
                "quality_target",
                "loss_threshold",
                "reached",
                "field_evaluations_to_target",
                "seconds_to_target",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def rebuild_quality_summary(output_dir):
    histories = {}
    for path in Path(output_dir).glob("*/result.npz"):
        with np.load(path) as result:
            histories[path.parent.name] = [
                {
                    "evaluation": int(evaluation),
                    "elapsed_seconds": float(elapsed_seconds),
                    "best_loss": float(best_loss),
                }
                for evaluation, elapsed_seconds, best_loss in zip(
                    result["evaluation_history"],
                    result["elapsed_seconds_history"],
                    result["best_loss_history"],
                )
            ]
    _write_quality_summary(Path(output_dir), histories)


def compare(config_path, output_root="outputs/algs", algorithms=None,
            show_loss_curve=True, basis_on_gpu=False, tensorboard_log_dir=None,
            max_seconds=None):
    base_cfg = SimulationConfig.from_yaml(config_path)
    if base_cfg.mode != "phase_optimization":
        raise ValueError("compare.py requires mode: phase_optimization")
    selected = algorithms or list(PROFILES)
    unknown = [name for name in selected if name not in PROFILES]
    if unknown:
        raise ValueError(f"Unknown comparison algorithms: {unknown}")
    output_dir = Path(output_root) / Path(config_path).stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # The response basis is physical-scene dependent and is intentionally shared.
    solver = HelmholtzDirectSolver(base_cfg)
    basis = None
    rows = []
    histories = {}
    tensorboard_dir = (
        Path(tensorboard_log_dir)
        if tensorboard_log_dir
        else output_dir / "tensorboard"
    )
    dashboard = ComparisonLossDashboard(
        selected,
        enabled=show_loss_curve,
        pause_seconds=base_cfg.training.loss_curve_pause_seconds,
        tensorboard_log_dir=tensorboard_dir,
    )
    started = time.perf_counter()
    try:
        basis = solver.build_phase_response_basis()
        if basis_on_gpu or base_cfg.training.load_basis_to_gpu:
            basis.to_gpu()
        basis_seconds = time.perf_counter() - started
        for algorithm in selected:
            cfg = _copy_profile(
                base_cfg, algorithm, output_dir, show_loss_curve, max_seconds
            )
            method_dir = output_dir / algorithm
            method_dir.mkdir(parents=True, exist_ok=True)
            (method_dir / "config.yaml").write_text(
                yaml.safe_dump(asdict(cfg), sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            print(f"\n{'=' * 72}\n[Compare] Running {algorithm}\n{'=' * 72}")
            try:
                validate_algorithm(cfg)
                curve = dashboard.curve(
                    algorithm, cfg.training.loss_curve_update_interval
                )
                problem, result = run(cfg, basis, loss_curve=curve)
                _, metadata = save_phase_optimization_result(
                    cfg, solver, basis, problem, result, 0.0,
                    extra_metadata={
                        "comparison_shared_basis_seconds": basis_seconds,
                        "comparison_profile": PROFILES[algorithm],
                    },
                )
                histories[algorithm] = result.history
                rows.append(_summary_row(algorithm, "completed", metadata))
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                rows.append(_summary_row(algorithm, "failed", error=error))
                (method_dir / "error.txt").write_text(
                    f"{error}\n\n{traceback.format_exc()}",
                    encoding="utf-8",
                )
                print(f"[Compare] {algorithm} failed: {error}")
            _write_summary(output_dir, config_path, rows, basis_seconds, selected)
    finally:
        save_loss_dashboard(histories, output_dir / "loss_dashboard.png")
        _write_quality_summary(output_dir, histories)
        dashboard.close()
        if basis is not None:
            basis.close()
        if solver.condensed_solver is not None:
            solver.condensed_solver.close()
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Compare all supported phase algorithms on one fixed scene."
    )
    parser.add_argument(
        "--config", default="src/examples/phase_oblique_reflecting_x_16x16.yaml"
    )
    parser.add_argument("--output-dir", default="outputs/algs")
    parser.add_argument(
        "--algorithms",
        default=",".join(PROFILES),
        help=f"Comma-separated subset of: {', '.join(PROFILES)}",
    )
    parser.add_argument(
        "--no-loss-window", action="store_true",
        help="Disable live loss windows for batch or headless execution.",
    )
    parser.add_argument(
        "--basis-on-gpu", action="store_true",
        help="Keep the shared response basis on GPU during the comparison.",
    )
    parser.add_argument(
        "--tensorboard-logdir",
        default=None,
        help="TensorBoard event directory; defaults to <output>/<scene>/tensorboard.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Shared per-algorithm wall-clock budget in seconds; defaults to 60.",
    )
    parser.add_argument(
        "--rebuild-quality-summary",
        action="store_true",
        help="Regenerate quality summary CSV/JSON from existing result files.",
    )
    args = parser.parse_args()
    if args.rebuild_quality_summary:
        output_dir = Path(args.output_dir) / Path(args.config).stem
        rebuild_quality_summary(output_dir)
        print(f"[Compare] rebuilt quality summary: {output_dir}")
        return
    selected = [name.strip() for name in args.algorithms.split(",") if name.strip()]
    rows = compare(
        args.config, args.output_dir, selected,
        show_loss_curve=not args.no_loss_window,
        basis_on_gpu=args.basis_on_gpu,
        tensorboard_log_dir=args.tensorboard_logdir,
        max_seconds=args.max_seconds,
    )
    completed = sum(row["status"] == "completed" for row in rows)
    print(f"[Compare] completed {completed}/{len(rows)} algorithms")
    if completed != len(rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
