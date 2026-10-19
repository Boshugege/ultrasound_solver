"""Run a configured phase algorithm or a fixed equal-phase forward experiment."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

from algorithm.config import SimulationConfig
from algorithm.baselines import ALGORITHMS, run, validate_algorithm


def scene_metadata(cfg, solver):
    output = {
        "source_positions": solver.transducers.centers,
        "target_points": np.asarray(cfg.targets).reshape(-1, 3),
        "boundary_conditions": json.dumps(cfg.boundary_conditions),
        "transducer_radius": cfg.specs.diameter * 0.5,
        "dx": cfg.domain.dx, "frequency": cfg.frequency,
        "c0": cfg.physics.sound_speed, "mode": cfg.mode,
        "propagation_model": solver.MODEL_VERSION,
        "field_unit": solver.field_unit,
        "time_convention": "Re(u exp(-i omega t))",
        "source_geometry": cfg.specs.geometry,
        "source_model": "sinc_neumann",
        "source_aperture_m": cfg.specs.diameter,
    }
    if solver.obstacle_sdf is not None:
        output["sdf"] = solver.obstacle_sdf
    elif cfg.obstacles and cfg.obstacles[0].get("type") == "sphere":
        obs = cfg.obstacles[0]
        axes = [
            np.linspace(origin, origin+length, count)
            for origin, length, count in zip(
                cfg.domain.origin_m,
                cfg.domain.box_size,
                cfg.domain.grid_size,
            )
        ]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        output["sdf"] = np.linalg.norm(grid - obs["center"], axis=-1) - obs["radius"]
    if hasattr(solver, "last_neural_stats"):
        output["neural_backend_stats"] = json.dumps(
            solver.last_neural_stats, allow_nan=False)
    return output


def _phase_artifact_paths(result_path):
    return {
        "recommended_phase_file": (
            f"{result_path.stem}_best_phases_rad.npy"
        ),
        "final_phase_file": f"{result_path.stem}_final_phases_rad.npy",
        "manifest_file": f"{result_path.stem}_phase_export.json",
    }


def save_phase_artifacts(result_path, cfg, result, metadata):
    """Save portable continuous phase vectors beside the full field result."""
    result_path = Path(result_path)
    artifacts = _phase_artifact_paths(result_path)
    best_phases = np.asarray(result.best.phases, dtype=np.float64)
    final_phases = np.asarray(result.final.phases, dtype=np.float64)
    if best_phases.ndim != 1 or final_phases.shape != best_phases.shape:
        raise ValueError("Phase artifacts require matching one-dimensional vectors")
    if not np.isfinite(best_phases).all() or not np.isfinite(final_phases).all():
        raise ValueError("Phase artifacts must be finite")

    np.save(result_path.with_name(artifacts["recommended_phase_file"]), best_phases)
    np.save(result_path.with_name(artifacts["final_phase_file"]), final_phases)
    manifest = {
        "schema_version": 1,
        "result_file": result_path.name,
        "recommended_phase_file": artifacts["recommended_phase_file"],
        "final_phase_file": artifacts["final_phase_file"],
        "recommended_phase_kind": "best_evaluated",
        "phase_unit": "rad",
        "phase_convention": "solver_source_boundary",
        "propagation_backend": cfg.solver.get("backend", "helmholtz"),
        "source_geometry": cfg.specs.geometry,
        "source_model": metadata.get("source_model", "sinc_neumann"),
        "source_aperture_m": metadata.get("source_aperture_m", cfg.specs.diameter),
        "source_positions_m": metadata.get("source_positions_m"),
        "channel_order": (
            "sonicsurface_logical" if cfg.specs.geometry == "sonicsurface_reference"
            else "ideal_x_major_y_minor"
        ),
        "hardware_ready": False,
        "hardware_note": (
            "Continuous solver phases require device mapping, calibration, "
            "and quantization before transmission."
        ),
        "num_transducers": int(best_phases.size),
        "array_shape": [cfg.specs.array_n, cfg.specs.array_n],
        "frequency_hz": cfg.frequency,
        "target_points_m": np.asarray(cfg.targets, dtype=np.float64).tolist(),
        "algorithm": cfg.algorithm,
        "algorithm_options": cfg.algorithm_options,
        "seed": cfg.training.seed,
        "termination_reason": result.termination_reason,
        "best_loss": float(result.best.loss),
        "final_loss": float(result.final.loss),
        "field_evaluations": metadata.get("field_evaluations"),
        "vjp_evaluations": metadata.get("vjp_evaluations"),
    }
    result_path.with_name(artifacts["manifest_file"]).write_text(
        json.dumps(manifest, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return artifacts


def save_phase_optimization_result(
    cfg,
    solver,
    basis,
    problem,
    result,
    basis_setup_seconds: float,
    extra_metadata: dict | None = None,
):
    reporting_started = time.perf_counter()
    initial_field = basis.field(problem.initial_phases, return_numpy=True)
    reporting_evaluations = 1
    output = scene_metadata(cfg, solver)
    output.update(
        algorithm=cfg.algorithm,
        phases=result.final.phases, u_complex=result.final.field,
        amplitude=np.abs(result.final.field), amp_sq=np.abs(result.final.field)**2,
        best_phases=result.best.phases, best_u_complex=result.best.field,
        best_amplitude=np.abs(result.best.field),
        initial_amplitude=np.abs(initial_field),
        target_amplitude=problem.target.target, target_weight=problem.target.weight,
        loss_history=np.array([row["loss"] for row in result.history]),
        best_loss_history=np.array([row["best_loss"] for row in result.history]),
        gradient_norm_history=np.array([
            np.nan if row["gradient_norm"] is None else row["gradient_norm"]
            for row in result.history]),
        evaluation_history=np.array([row["evaluation"] for row in result.history]),
        elapsed_seconds_history=np.array([
            row["elapsed_seconds"] for row in result.history]),
    )
    if cfg.training.compare_geometric:
        phases = solver.transducers.compute_geometric_phases()
        geometric_field = basis.field(phases, return_numpy=True)
        reporting_evaluations += 1
        output.update(
            geometric_phases=phases, geometric_u_complex=geometric_field,
            geometric_amplitude=np.abs(geometric_field),
            geometric_amp_sq=np.abs(geometric_field)**2,
            geometric_metrics=json.dumps(
                problem.metrics(geometric_field, problem.focal_field(phases)), allow_nan=False
            ),
        )
    path = Path(cfg.io.output_file)
    if path.suffix != ".npz":
        raise ValueError("io.output_file must end in .npz")
    loss_plot_path = path.with_name(f"{path.stem}_loss.png")
    metadata = {
        "source_positions_m": solver.transducers.centers.tolist(),
        "field_unit": solver.field_unit,
        "source_model": output["source_model"],
        "source_aperture_m": output["source_aperture_m"],
        "propagation_model": solver.MODEL_VERSION,
        "algorithm": cfg.algorithm, "algorithm_options": cfg.algorithm_options,
        "seed": cfg.training.seed, "termination_reason": result.termination_reason,
        "final_loss": result.final.loss, "best_loss": result.best.loss,
        "final_metrics": result.final.metrics, "best_metrics": result.best.metrics,
        "basis_build_or_load_seconds": basis_setup_seconds,
        "reporting_field_evaluations": reporting_evaluations,
        "reporting_seconds": time.perf_counter() - reporting_started,
        "loss_plot_file": str(loss_plot_path),
        **result.metadata,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    phase_artifacts = _phase_artifact_paths(path)
    metadata["phase_artifacts"] = phase_artifacts
    output["run_metadata"] = json.dumps(metadata, allow_nan=False)
    output["evaluation_log"] = json.dumps(result.history, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **output)
    save_phase_artifacts(path, cfg, result, metadata)
    from algorithm.baselines.loss_curve import save_loss_history
    save_loss_history(
        result.history,
        loss_plot_path,
        f"{cfg.algorithm}: current and best loss",
    )
    print(
        f"[Main] Saved {path}; recommended phases: "
        f"{phase_artifacts['recommended_phase_file']}"
    )
    return output, metadata


def execute(cfg):
    if cfg.mode == "sdf_inverse":
        raise NotImplementedError(
            "sdf_inverse is reserved for fixed equal-phase SDF reconstruction. "
            "Observation loading and differentiable geometry assembly are not implemented."
        )
    if cfg.mode == "phase_optimization":
        validate_algorithm(cfg)
    backend = str(cfg.solver.get("backend", "")).lower()
    if backend == "bem":
        from algorithm.bem.scene_solver import BemSceneSolver as Solver
    elif backend == "neural_bem":
        repository_root = str(Path(__file__).resolve().parents[1])
        if repository_root not in sys.path:
            sys.path.insert(0, repository_root)
        from nerual.backend import NeuralBemSceneSolver as Solver
    else:
        from algorithm.solvers import HelmholtzDirectSolver as Solver

    solver = Solver(cfg)
    basis = None
    started = time.perf_counter()
    try:
        if cfg.mode == "same_phase":
            solver.set_phases(np.full(solver.transducers.num_transducers, cfg.same_phase_rad))
            field = solver.solve()
            output = scene_metadata(cfg, solver)
            output.update(phases=solver.transducers.phases, u_complex=field,
                          amplitude=np.abs(field), amp_sq=np.abs(field)**2)
        else:
            basis = solver.build_phase_response_basis()
            if cfg.training.load_basis_to_gpu:
                basis.to_gpu()
            setup_seconds = time.perf_counter() - started
            problem, result = run(cfg, basis)
            output, _ = save_phase_optimization_result(
                cfg, solver, basis, problem, result, setup_seconds
            )
            print(f"[Result] {cfg.algorithm}: {result.termination_reason}; "
                  f"final={result.final.loss:.6e} best={result.best.loss:.6e}")
        path = Path(cfg.io.output_file)
        if path.suffix != ".npz":
            raise ValueError("io.output_file must end in .npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **output)
        print(f"[Main] Saved {path}")
    finally:
        if basis is not None:
            basis.close()
        if solver.condensed_solver is not None:
            solver.condensed_solver.close()
    if cfg.io.auto_visualize:
        from algorithm.visualizer import show_pyvista_scene
        show_pyvista_scene(
            cfg.io.output_file, field_name="amplitude",
            compare_methods=cfg.mode == "phase_optimization" and cfg.training.compare_geometric,
        )
    return output


def main():
    parser = argparse.ArgumentParser(description="Ultrasound phase optimization and SDF experiments")
    parser.add_argument("--config", default="src/examples/config.yaml")
    parser.add_argument("--list-algorithms", action="store_true")
    args = parser.parse_args()
    if args.list_algorithms:
        print("\n".join(ALGORITHMS))
        return
    execute(SimulationConfig.from_yaml(args.config))


if __name__ == "__main__":
    main()
