"""Pipeline adapter for ``solver.backend: neural_bem``.

The accurate Burton-Miller BEM remains the compatibility and fallback backend.
The neural CHIEF-BEM path accelerates exact target-response queries only.
"""
import hashlib
from pathlib import Path
import time
import warnings
import numpy as np

from algorithm.bem.scene_solver import BemSceneSolver, obstacle_surface
from .general_solver import GeneralNeuralSolver
from .mesh_scene import MeshBody


class NeuralBemSceneSolver(BemSceneSolver):
    MODEL_VERSION = "neural_chief_bem_target_v1"

    def __init__(self, cfg):
        super().__init__(cfg)
        options = cfg.solver
        self.neural_max_panels = int(options.get("neural_max_panels", 2200))
        self.neural_tolerance = float(options.get("neural_residual_tolerance", 2e-2))
        checkpoint = str(options.get("neural_checkpoint", "")).strip() or None
        self.neural_checkpoint = (
            Path(checkpoint) if checkpoint else
            Path(__file__).with_name("artifacts")/"boundary_net.pt")
        self.neural_solver = None
        self.last_neural_stats = {
            "method": "accurate_bem", "reason": "not_initialized",
        }
        if not cfg.obstacles:
            self.last_neural_stats["reason"] = "no_obstacle"
            return
        bodies = []
        element_size = cfg.wavelength/float(
            options.get("neural_elements_per_wavelength", 3.0))
        for index, obstacle in enumerate(cfg.obstacles):
            vertices, faces = obstacle_surface(obstacle, element_size)
            kind = obstacle.get("type", "").lower()
            if "neural_chief_points_m" in obstacle:
                chief = np.asarray(obstacle["neural_chief_points_m"], float)
            elif kind == "sphere":
                chief = np.atleast_2d(np.asarray(obstacle["center"], float))
            elif "center_m" in obstacle:
                chief = np.atleast_2d(np.asarray(obstacle["center_m"], float))
            else:
                raise ValueError(
                    f"obstacle {index} requires neural_chief_points_m")
            bodies.append(MeshBody(vertices[faces], chief, f"obstacle_{index}"))
        panels = sum(len(body.triangles) for body in bodies)
        if panels > self.neural_max_panels:
            self.last_neural_stats = {
                "method": "accurate_bem", "reason": "panel_limit",
                "panels": panels,
            }
            return
        try:
            self.neural_solver = GeneralNeuralSolver(
                checkpoint=checkpoint,
                receivers=self.transducers.centers,
                diameter=cfg.specs.diameter,
                device=str(options.get("neural_device", "cuda")),
                residual_tolerance=self.neural_tolerance,
                max_direct_panels=0,
                aperture_profile="sinc",
                frequency=cfg.frequency,
                sound_speed=cfg.physics.sound_speed,
            )
            self.neural_solver.set_scene(bodies[0], bodies[1:])
            self.last_neural_stats = {
                "method": "ready", "panels": panels,
                "setup_ms": self.neural_solver.setup_ms,
            }
        except Exception as error:
            warnings.warn(
                f"neural_bem initialization failed; using accurate BEM: {error}",
                RuntimeWarning)
            self.neural_solver = None
            self.last_neural_stats = {
                "method": "accurate_bem", "reason": "initialization_failed",
                "error": str(error), "panels": panels,
            }

    def target_response(self, points):
        points = np.atleast_2d(np.asarray(points, float))
        if self.neural_solver is None:
            return super().target_response(points)
        started = time.perf_counter()
        try:
            result = self.neural_solver.solve(points, method="neural")
            if (not np.isfinite(result.final_residual)
                    or result.final_residual > self.neural_tolerance):
                self.last_neural_stats = {
                    "method": "accurate_bem_fallback",
                    "reason": "residual", "residual": result.final_residual,
                    "neural_ms": result.solve_ms,
                }
                return super().target_response(points)
            self.last_neural_stats = {
                "method": "neural", "residual": result.final_residual,
                "solve_ms": result.solve_ms,
                "total_ms": (time.perf_counter()-started)*1000,
            }
            return result.response*self.transducers.amplitudes[None, :]
        except Exception as error:
            self.last_neural_stats = {
                "method": "accurate_bem_fallback",
                "reason": "runtime_error", "error": str(error),
            }
            warnings.warn(
                f"neural_bem target solve failed; using accurate BEM: {error}",
                RuntimeWarning)
            return super().target_response(points)

    def field_at(self, points, phases=None):
        phases = self.transducers.phases if phases is None else np.asarray(phases, float)
        return self.target_response(points)@np.exp(1j*phases)

    def scene_fingerprint(self):
        digest = hashlib.sha256(super().scene_fingerprint().encode())
        payload = {
            "model": self.MODEL_VERSION,
            "neural_max_panels": self.neural_max_panels,
            "neural_tolerance": self.neural_tolerance,
        }
        digest.update(repr(sorted(payload.items())).encode())
        if self.neural_solver is not None:
            if self.neural_checkpoint.is_file():
                digest.update(self.neural_checkpoint.read_bytes())
        return digest.hexdigest()

    def close(self):
        self.neural_solver = None
        super().close()
