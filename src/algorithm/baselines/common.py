"""Shared objective, accounting and results for phase algorithms."""

from dataclasses import dataclass, field
import time

import cupy as cp
import numpy as np

from algorithm.training.phase_adjoint_optimizer import (
    AmplitudeTarget, AmplitudeFieldLoss, FocalPressureLoss, FocalContrastLoss,
    _build_background_mask, _nearest_grid_index, _validate_target_point,
)


class BudgetExhausted(RuntimeError):
    pass


class TimeBudgetExhausted(BudgetExhausted):
    pass


def positive_int(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def positive_float(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def wrap(phases):
    phases = np.asarray(phases, dtype=np.float64)
    if not np.isfinite(phases).all():
        raise ValueError("phases must be finite")
    return np.mod(phases, 2 * np.pi)


@dataclass
class Evaluation:
    phases: np.ndarray
    loss: float
    field: np.ndarray
    metrics: dict
    gradient: np.ndarray | None = None


@dataclass
class AlgorithmResult:
    final: Evaluation
    best: Evaluation
    history: list
    termination_reason: str
    metadata: dict = field(default_factory=dict)


class PhaseProblem:
    def __init__(self, cfg, basis, loss_curve=None):
        self.cfg, self.basis = cfg, basis
        self.size = basis.num_transducers
        self.target = AmplitudeTarget.from_config(cfg)
        for point in cfg.targets:
            _validate_target_point(cfg, np.asarray(point))
        if cfg.training.loss_type != "field_match" and not cfg.targets:
            raise ValueError("Focal losses require targets")
        factories = {
            "field_match": lambda: AmplitudeFieldLoss(self.target),
            "focal_pressure": lambda: FocalPressureLoss(cfg, self.target),
            "focal_contrast": lambda: FocalContrastLoss(cfg, self.target),
        }
        if cfg.training.loss_type not in factories:
            raise ValueError("Unknown training.loss_type")
        self.loss_model = factories[cfg.training.loss_type]()
        if (not np.isfinite(self.target.target).all()
                or not np.isfinite(self.target.weight).all()
                or np.any(self.target.target < 0) or np.any(self.target.weight < 0)):
            raise ValueError("Target amplitudes and weights must be finite and nonnegative")
        self.target_indices = [_nearest_grid_index(cfg, np.asarray(p)) for p in cfg.targets]
        self.target_response = (
            basis.target_response(np.asarray(cfg.targets))
            if cfg.targets and callable(getattr(basis, "target_response", None))
            else None
        )
        self.background_mask = _build_background_mask(
            cfg, np.asarray(cfg.targets), cfg.training.target_radius,
            cfg.training.source_exclusion_layers,
        )
        if not np.any(self.background_mask):
            raise ValueError("Evaluation background mask is empty")
        self.evaluation_budget = positive_int(
            cfg.training.max_evaluations, "max_evaluations"
        )
        self.time_budget_seconds = float(cfg.training.max_seconds)
        if (not np.isfinite(self.time_budget_seconds)
                or self.time_budget_seconds < 0):
            raise ValueError("max_seconds must be finite and nonnegative")
        self._reserved_evaluations = 0
        self.iterations = positive_int(cfg.training.iterations, "iterations", 0)
        self.rng = np.random.default_rng(cfg.training.seed)
        self.initial_phases = self._initial_phases()
        self.field_evaluations = 0
        self.vjp_evaluations = 0
        self.history = []
        self.best = None
        self.last = None
        if loss_curve is None:
            from algorithm.baselines.loss_curve import LiveLossCurve

            loss_curve = LiveLossCurve(
                cfg.training.show_loss_curve,
                cfg.training.loss_curve_update_interval,
                cfg.training.loss_curve_pause_seconds,
                cfg.algorithm,
            )
        self.loss_curve = loss_curve
        if (
            getattr(self.basis, "basis_gpu", None) is not None
            and cfg.algorithm == "adjoint"
        ):
            self._warm_gpu_pipeline()
        self.started = time.perf_counter()

    def _warm_gpu_pipeline(self):
        field = self.basis.field(self.initial_phases, return_numpy=False)
        focal = self.focal_field(self.initial_phases, device=True)
        loss, cotangent = self.loss_model.evaluate_device(field, **self._focal_kwargs(focal))
        self.basis.phase_vjp(cotangent, self.initial_phases, return_numpy=False)
        if self._focal_kwargs(focal):
            self._focal_gradient(self.initial_phases, focal, device=True)
        self.metrics_device(field, focal)
        if not bool(cp.isfinite(loss)):
            raise FloatingPointError("Non-finite GPU warm-up loss")
        cp.cuda.Stream.null.synchronize()

    def _initial_phases(self):
        mode = self.cfg.training.initial_phase
        if mode == "zero":
            return np.zeros(self.size)
        if mode == "current":
            return wrap(self.basis.solver.transducers.phases)
        if mode == "random":
            return self.rng.uniform(0, 2 * np.pi, self.size)
        if mode == "geometric":
            return wrap(self.basis.solver.transducers.compute_geometric_phases())
        if mode == "response":
            from algorithm.baselines.response_alignment import phases_for_targets
            return phases_for_targets(self)
        raise ValueError("initial_phase must be geometric, response, zero, current or random")

    @property
    def remaining(self):
        return self.evaluation_budget - self._reserved_evaluations - self.field_evaluations

    @property
    def elapsed_seconds(self):
        return time.perf_counter() - self.started

    @property
    def time_remaining_seconds(self):
        if self.time_budget_seconds == 0:
            return float("inf")
        return self.time_budget_seconds - self.elapsed_seconds

    def focal_field(self, phases, device=False):
        if self.target_response is None:
            return None
        xp = cp if device else np
        if device:
            if not hasattr(self, "_gpu_target_response"):
                self._gpu_target_response = cp.asarray(self.target_response)
            response = self._gpu_target_response
        else:
            response = self.target_response
        return response @ xp.exp(1j * xp.asarray(phases))

    def _focal_kwargs(self, focal):
        if focal is not None and self.cfg.training.loss_type != "field_match":
            return {"focal_field": focal}
        return {}

    def _focal_gradient(self, phases, focal, device=False):
        xp = cp if device else np
        response = self._gpu_target_response if device else self.target_response
        cotangent = (-self.loss_model.target_weight / len(focal)
                     * focal / (xp.abs(focal) + self.loss_model.eps))
        adjoint = response.conj().T @ cotangent
        return xp.real(adjoint.conj() * 1j * xp.exp(1j * xp.asarray(phases)))

    def metrics(self, field, focal=None):
        amplitude = np.abs(field)
        values = (np.abs(focal) if focal is not None else
                  np.array([amplitude[i] for i in self.target_indices]))
        if not values.size:
            values = amplitude[self.target.weight >= np.max(self.target.weight) * 0.5]
        mean, peak = float(np.mean(values)), float(np.max(values))
        background = float(np.max(amplitude[self.background_mask]))
        norm = float(np.linalg.norm(self.target.target))
        error = float(np.linalg.norm(amplitude - self.target.target))
        return {
            "target_mean": mean,
            "target_min": float(np.min(values)),
            "background_max": background,
            "contrast": mean / background if background > 0 else None,
            "uniformity": float(np.min(values)) / peak if peak > 0 else None,
            "target_cv": float(np.std(values)) / mean if mean > 0 else None,
            "field_relative_l2": error / norm if norm > 0 else None,
            "field_absolute_l2": error,
        }

    def metrics_device(self, field, focal=None):
        amplitude = cp.abs(field)
        if focal is not None:
            values = cp.abs(focal)
        elif self.target_indices:
            indices = tuple(np.asarray(self.target_indices, dtype=np.intp).T)
            values = amplitude[indices]
        else:
            target_mask = self._device_target_mask()
            values = amplitude[target_mask]
        background_mask = self._device_background_mask()
        target = self._device_target()
        mean = cp.mean(values)
        peak = cp.max(values)
        background = cp.max(amplitude[background_mask])
        error = cp.linalg.norm(amplitude - target)
        target_min = cp.min(values)
        target_std = cp.std(values)
        summary = cp.asnumpy(
            cp.asarray(
                [
                    mean, target_min, background, peak, target_std, error,
                ]
            )
        )
        mean, target_min, background, peak, target_std, error = (
            float(value) for value in summary
        )
        norm = float(np.linalg.norm(self.target.target))
        return {
            "target_mean": mean,
            "target_min": target_min,
            "background_max": background,
            "contrast": mean / background if background > 0 else None,
            "uniformity": target_min / peak if peak > 0 else None,
            "target_cv": target_std / mean if mean > 0 else None,
            "field_relative_l2": error / norm if norm > 0 else None,
            "field_absolute_l2": error,
        }

    def _device_target(self):
        if not hasattr(self, "_gpu_target"):
            self._gpu_target = cp.asarray(self.target.target)
        return self._gpu_target

    def _device_target_mask(self):
        if not hasattr(self, "_gpu_target_mask"):
            self._gpu_target_mask = cp.asarray(
                self.target.weight >= np.max(self.target.weight) * 0.5
            )
        return self._gpu_target_mask

    def _device_background_mask(self):
        if not hasattr(self, "_gpu_background_mask"):
            self._gpu_background_mask = cp.asarray(self.background_mask)
        return self._gpu_background_mask

    def evaluate(self, phases, gradient=False, stage="optimization"):
        # Every method must produce one valid initial candidate for reporting.
        if self.field_evaluations > 0 and self.time_remaining_seconds <= 0:
            raise TimeBudgetExhausted("wall-clock time budget exhausted")
        if self.remaining <= 0:
            raise BudgetExhausted("field evaluation budget exhausted")
        phases = wrap(phases)
        if phases.shape != (self.size,):
            raise ValueError(f"Expected {self.size} phases")
        self.field_evaluations += 1
        use_gpu = getattr(self.basis, "basis_gpu", None) is not None
        field_value = self.basis.field(phases, return_numpy=not use_gpu)
        focal = self.focal_field(phases, device=use_gpu)
        focal_kwargs = self._focal_kwargs(focal)
        if use_gpu:
            loss_value, cotangent = self.loss_model.evaluate_device(
                field_value,
                with_cotangent=gradient,
                **focal_kwargs,
            )
            loss = float(loss_value)
        else:
            loss, cotangent = self.loss_model.evaluate(
                field_value,
                with_cotangent=gradient,
                **focal_kwargs,
            )
        if not np.isfinite(loss) or (
            use_gpu and not bool(cp.isfinite(field_value).all())
        ):
            raise FloatingPointError("Non-finite field or loss")
        grad = None
        if gradient:
            self.vjp_evaluations += 1
            grad = self.basis.phase_vjp(
                cotangent,
                phases,
                return_numpy=True,
            )
            if focal_kwargs:
                extra = self._focal_gradient(phases, focal, device=use_gpu)
                grad += cp.asnumpy(extra) if use_gpu else extra
            if not np.isfinite(grad).all():
                raise FloatingPointError("Non-finite phase gradient")
        metrics = (self.metrics_device(field_value, focal) if use_gpu else
                   self.metrics(field_value, focal))
        evaluation = Evaluation(
            phases.copy(),
            loss,
            None if use_gpu else field_value,
            metrics,
            grad,
        )
        self.last = evaluation
        if self.best is None or loss < self.best.loss:
            self.best = evaluation
        row = {
            "evaluation": self.field_evaluations, "vjp_evaluations": self.vjp_evaluations,
            "elapsed_seconds": self.elapsed_seconds,
            "loss": loss, "best_loss": self.best.loss, "stage": stage,
            "gradient_norm": float(np.linalg.norm(grad)) if grad is not None else None,
            **evaluation.metrics,
        }
        self.history.append(row)
        self.loss_curve.update(row)
        print(f"[{self.cfg.algorithm}] eval={self.field_evaluations:05d} "
              f"stage={stage} loss={loss:.6e} best={self.best.loss:.6e} "
              f"target={row['target_mean']:.3f} background={row['background_max']:.3f}")
        return evaluation

    def reserve_evaluations(self, count):
        count = positive_int(count, "reserved evaluation count", 0)
        if self.remaining < count:
            raise ValueError(
                f"max_evaluations={self.evaluation_budget} cannot reserve {count} "
                "evaluations after work already completed"
            )
        self._reserved_evaluations += count

    def release_evaluations(self, count):
        count = positive_int(count, "reserved evaluation count", 0)
        if count > self._reserved_evaluations:
            raise ValueError("Cannot release more evaluations than reserved")
        self._reserved_evaluations -= count

    def result(self, final, reason, **metadata):
        if final is None or self.best is None:
            raise RuntimeError("Algorithm returned no evaluated candidate")
        # Amplitude objectives do not fix global phase. For an optimized single
        # exact target, choose a real-positive focus before export/quantization.
        # This rotates all channels equally; relative phases are untouched.
        if (self.target_response is not None and len(self.target_response) == 1
                and self.cfg.training.loss_type != "field_match"
                and self.cfg.algorithm in {"adjoint", "response_alignment"}):
            for value in {id(final): final, id(self.best): self.best}.values():
                offset = float(np.angle(self.focal_field(value.phases)[0]))
                value.phases = wrap(value.phases - offset)
                if value.field is not None:
                    value.field = value.field * np.exp(-1j * offset)
            metadata["global_phase_convention"] = "real_positive_exact_target"
        if final.field is None:
            final.field = self.basis.field(final.phases, return_numpy=True)
        if self.best.field is None:
            self.best.field = self.basis.field(self.best.phases, return_numpy=True)
        self.loss_curve.finalize(reason)
        return AlgorithmResult(
            final, self.best, list(self.history), reason,
            {"field_evaluations": self.field_evaluations,
             "vjp_evaluations": self.vjp_evaluations,
             "elapsed_seconds": self.elapsed_seconds,
             "time_budget_seconds": self.time_budget_seconds, **metadata},
        )
