"""Response-basis analytic VJP with several phase-update rules."""

import numpy as np

from algorithm.baselines.common import (
    BudgetExhausted,
    TimeBudgetExhausted,
    positive_float,
)


def _convergence_options(options):
    patience = int(options.get("convergence_patience", 100))
    tolerance = float(options.get("convergence_relative_tolerance", 1.0e-5))
    if patience < 1:
        raise ValueError("convergence_patience must be positive")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError(
            "convergence_relative_tolerance must be finite and positive"
        )
    return patience, tolerance


def _stagnated(problem, checkpoint_loss, step, patience, tolerance):
    if step % patience:
        return False, checkpoint_loss, None
    relative_improvement = (
        checkpoint_loss - problem.best.loss
    ) / max(1.0, abs(checkpoint_loss))
    if relative_improvement <= tolerance:
        return True, checkpoint_loss, relative_improvement
    return False, problem.best.loss, relative_improvement


def solve(problem, options):
    method = options.get("optimizer", "adam")
    if method not in {"lbfgsb", "adam", "adamw", "lion", "nonlinear_cg"}:
        raise ValueError(
            "adjoint optimizer must be lbfgsb, adam, adamw, lion or nonlinear_cg"
        )
    current = problem.evaluate(problem.initial_phases, gradient=True)
    checks = []
    try:
        if options.get("gradient_check", False):
            h = positive_float(options.get("gradient_check_step", 1e-3),
                               "gradient_check_step")
            for index in range(min(8, problem.size)):
                offset = np.zeros(problem.size)
                offset[index] = h
                plus = problem.evaluate(current.phases + offset, stage="gradient_check")
                minus = problem.evaluate(current.phases - offset, stage="gradient_check")
                numerical = (plus.loss - minus.loss) / (2 * h)
                exact = float(current.gradient[index])
                error = abs(exact - numerical) / max(abs(exact), abs(numerical), 1.0)
                checks.append({"index": index, "analytic": exact,
                               "finite_difference": numerical, "scaled_error": error})
                print(f"[GradCheck] emitter={index} analytic={exact:.6e} "
                      f"finite_diff={numerical:.6e} scaled_error={error:.3e}")
        if problem.iterations == 0:
            return problem.result(current, "iteration_limit", gradient_checks=checks)
        if method in {"adam", "adamw"}:
            rate = positive_float(options.get("learning_rate", 0.05),
                                  "learning_rate")
            decay = float(options.get("weight_decay", 0.0))
            patience, tolerance = _convergence_options(options)
            if not np.isfinite(decay) or decay < 0:
                raise ValueError("weight_decay must be finite and nonnegative")
            first = np.zeros(problem.size)
            second = np.zeros(problem.size)
            checkpoint_loss = problem.best.loss
            for step in range(1, problem.iterations + 1):
                first = 0.9 * first + 0.1 * current.gradient
                second = 0.999 * second + 0.001 * current.gradient**2
                update = rate * (first / (1 - 0.9**step)) / (
                    np.sqrt(second / (1 - 0.999**step)) + 1e-8)
                phases = current.phases - update
                # Decoupled weight decay is supplied for optimizer parity only.
                # Default zero is appropriate because raw phase magnitude is periodic.
                if method == "adamw" and decay:
                    phases *= 1.0 - rate * decay
                current = problem.evaluate(phases, gradient=True)
                stopped, checkpoint_loss, improvement = _stagnated(
                    problem, checkpoint_loss, step, patience, tolerance
                )
                if stopped:
                    return problem.result(
                        problem.best,
                        "converged",
                        convergence_patience=patience,
                        convergence_relative_tolerance=tolerance,
                        convergence_relative_improvement=improvement,
                    )
            return problem.result(
                current, "iteration_limit", gradient_checks=checks
            )

        if method == "lion":
            rate = positive_float(options.get("learning_rate", 0.01),
                                  "learning_rate")
            beta1 = float(options.get("beta1", 0.9))
            beta2 = float(options.get("beta2", 0.99))
            if not (0 <= beta1 < 1 and 0 <= beta2 < 1):
                raise ValueError("Lion beta1 and beta2 must be in [0, 1)")
            momentum = np.zeros(problem.size)
            patience, tolerance = _convergence_options(options)
            checkpoint_loss = problem.best.loss
            for step in range(1, problem.iterations + 1):
                direction = beta1 * momentum + (1.0 - beta1) * current.gradient
                momentum = beta2 * momentum + (1.0 - beta2) * current.gradient
                current = problem.evaluate(
                    current.phases - rate * np.sign(direction), gradient=True
                )
                stopped, checkpoint_loss, improvement = _stagnated(
                    problem, checkpoint_loss, step, patience, tolerance
                )
                if stopped:
                    return problem.result(
                        problem.best,
                        "converged",
                        convergence_patience=patience,
                        convergence_relative_tolerance=tolerance,
                        convergence_relative_improvement=improvement,
                    )
            return problem.result(
                current, "iteration_limit", gradient_checks=checks
            )

        if method == "nonlinear_cg":
            initial_step = positive_float(
                options.get("initial_step", 0.25), "initial_step"
            )
            armijo = float(options.get("armijo", 1e-4))
            shrink = float(options.get("line_search_shrink", 0.5))
            max_line_search = int(options.get("max_line_search", 12))
            if not 0 < armijo < 1 or not 0 < shrink < 1 or max_line_search < 1:
                raise ValueError("Invalid nonlinear_cg line-search options")
            direction = -current.gradient
            patience, tolerance = _convergence_options(options)
            checkpoint_loss = problem.best.loss
            for step_index in range(1, problem.iterations + 1):
                if np.dot(current.gradient, direction) >= 0:
                    direction = -current.gradient
                scale = max(float(np.max(np.abs(direction))), 1.0)
                direction = direction / scale
                directional_derivative = float(np.dot(current.gradient, direction))
                accepted = None
                step = initial_step
                for _ in range(max_line_search):
                    candidate = problem.evaluate(
                        current.phases + step * direction,
                        gradient=True,
                        stage="line_search",
                    )
                    if candidate.loss <= current.loss + armijo * step * directional_derivative:
                        accepted = candidate
                        break
                    step *= shrink
                if accepted is None:
                    return problem.result(
                        current, "line_search_stalled", gradient_checks=checks
                    )
                previous_gradient = current.gradient
                current = accepted
                denominator = float(np.dot(previous_gradient, previous_gradient))
                beta = max(
                    0.0,
                    float(
                        np.dot(
                            current.gradient,
                            current.gradient - previous_gradient,
                        )
                    ) / max(denominator, 1e-30),
                )
                direction = -current.gradient + beta * direction
                stopped, checkpoint_loss, improvement = _stagnated(
                    problem,
                    checkpoint_loss,
                    step_index,
                    patience,
                    tolerance,
                )
                if stopped:
                    return problem.result(
                        problem.best,
                        "converged",
                        convergence_patience=patience,
                        convergence_relative_tolerance=tolerance,
                        convergence_relative_improvement=improvement,
                    )
            return problem.result(
                current, "iteration_limit", gradient_checks=checks
            )

        from scipy.optimize import minimize

        def objective(phases):
            nonlocal current
            # Reuse the initial evaluation and repeated line-search requests.
            if not np.array_equal(np.mod(phases, 2 * np.pi), current.phases):
                current = problem.evaluate(phases, gradient=True)
            return current.loss, current.gradient

        result = minimize(
            objective, current.phases, jac=True, method="L-BFGS-B",
            # Phases live on a circle: box bounds create false boundary optima.
            # Keep the optimizer's coordinates unwrapped; evaluation is periodic.
            options={"maxiter": problem.iterations, "ftol": 1e-12,
                     "gtol": 1e-8, "maxls": 40},
        )
        if not np.array_equal(np.mod(result.x, 2 * np.pi), current.phases):
            current = problem.evaluate(result.x, gradient=True)
        return problem.result(current, "converged" if result.success else "optimizer_stopped",
                              optimizer_message=str(result.message), gradient_checks=checks)
    except BudgetExhausted as exc:
        reason = "time_budget" if isinstance(exc, TimeBudgetExhausted) else "evaluation_budget"
        return problem.result(current, reason, gradient_checks=checks)
