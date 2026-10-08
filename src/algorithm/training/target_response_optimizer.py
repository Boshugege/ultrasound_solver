"""Memory-bounded focal-pressure optimization from discrete PDE responses."""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class TargetResponseResult:
    phases: np.ndarray
    loss: float
    field_evaluations: int
    iterations: int
    success: bool
    message: str
    amplitude_efficiency: float


def focal_pressure_loss_gradient(phases, response):
    """Match FocalPressureLoss for one point: loss = -abs(response @ weights)."""
    response = np.asarray(response, dtype=np.complex128)
    phases = np.asarray(phases, dtype=np.float64)
    if (response.ndim != 1 or phases.shape != response.shape
            or not np.isfinite(response).all() or not np.isfinite(phases).all()
            or not np.any(abs(response))):
        raise ValueError("Finite nonzero response and matching phases are required")
    contributions = response*np.exp(1j*phases)
    pressure = contributions.sum()
    amplitude = abs(pressure)
    gradient = -np.real(
        pressure.conjugate()/max(amplitude, np.finfo(float).tiny)
        * 1j*contributions
    )
    return -float(amplitude), gradient


def optimize_focal_response(response, initial_phases, iterations, max_evaluations):
    """Optimize unbounded phase coordinates and fix only the final global gauge."""
    response = np.asarray(response, dtype=np.complex128)
    initial = np.asarray(initial_phases, dtype=np.float64)
    if not isinstance(iterations, (int, np.integer)) or iterations < 0:
        raise ValueError("iterations must be a nonnegative integer")
    if not isinstance(max_evaluations, (int, np.integer)) or max_evaluations < 1:
        raise ValueError("max_evaluations must be a positive integer")
    if iterations == 0:
        raw = initial
        loss, _ = focal_pressure_loss_gradient(raw, response)
        evaluations, completed, message = 1, True, "iteration limit is zero"
        nit = 0
    else:
        result = minimize(
            focal_pressure_loss_gradient,
            initial,
            args=(response,),
            jac=True,
            method="L-BFGS-B",
            options={
                "maxiter": int(iterations),
                "maxfun": int(max_evaluations),
                "ftol": 1e-12,
                "gtol": 1e-8,
                "maxls": 40,
            },
        )
        raw = result.x
        loss = float(result.fun)
        evaluations = int(result.nfev)
        completed = bool(result.success)
        message = str(result.message)
        nit = int(result.nit)
    pressure = response @ np.exp(1j*raw)
    phases = np.mod(raw-np.angle(pressure), 2*np.pi)
    amplitude = abs(response @ np.exp(1j*phases))
    upper = float(np.sum(abs(response)))
    return TargetResponseResult(
        phases=phases,
        loss=-float(amplitude),
        field_evaluations=evaluations,
        iterations=nit,
        success=completed,
        message=message,
        amplitude_efficiency=float(amplitude/upper),
    )
