from importlib import util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

GPU = util.find_spec("cupy") is not None
K = 2 * np.pi * 40000.0 / 346.0


def small_config(obstacles=(), boundary=None, grid=26):
    """2 x 2 array in a 5 cm cube; the grid only samples the BEM field."""
    from algorithm.config import SimulationConfig
    import yaml

    directory = Path(tempfile.mkdtemp())
    raw = {
        "mode": "phase_optimization", "algorithm": "adjoint",
        "physics": {"sound_speed": 346.0, "medium_density": 1.21},
        "transducer_specs": {"frequency": 40000.0, "array_n": 2, "diameter": 0.009, "pitch": 0.0105,
                             "geometry": "ideal_grid"},
        "domain": {"box_size": [0.05] * 3, "grid_size": [grid] * 3},
        "boundary_conditions": boundary or {"-x": "open", "+x": "open", "-y": "open", "+y": "open",
                                            "-z": "reflecting", "+z": "open"},
        "targets": [{"point": [0.025, 0.025, 0.044]}],
        "obstacles": list(obstacles),
        "solver": {"backend": "bem", "bem_rtol": 1e-6},
        "training": {"phase_basis_file": str(directory / "basis.npy")},
        "io": {"output_file": str(directory / "result.npz"), "auto_visualize": False},
    }
    path = directory / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return SimulationConfig.from_yaml(path)


SPHERE = {"type": "sphere", "center": [0.025, 0.025, 0.022], "radius": 0.008}


class ImageGroupTests(unittest.TestCase):
    def test_perpendicular_planes_give_four_images(self):
        if not GPU:
            self.skipTest("cupy not installed")
        from algorithm.bem.solver import image_group

        signs, offsets = image_group([(2, 0.0), (0, 0.16)])
        point = np.array([0.1, 0.05, 0.02])
        images = {tuple(np.round(point * s + o, 6)) for s, o in zip(signs, offsets)}
        self.assertEqual(images, {(0.1, 0.05, 0.02), (0.1, 0.05, -0.02),
                                  (0.22, 0.05, 0.02), (0.22, 0.05, -0.02)})
        with self.assertRaises(ValueError):
            image_group([(0, 0.0), (0, 0.16)])


@unittest.skipUnless(GPU, "cupy not installed")
class AccuracyTests(unittest.TestCase):
    def test_rigid_sphere_matches_series(self):
        import cupy as cp
        from algorithm.bem.solver import RigidBEM, icosphere, rigid_sphere_point_source

        source = np.array([0.0, 0.0, 0.06])
        bem = RigidBEM(*icosphere(0.02, 4), k=K, planes=())
        u, info = bem.solve(bem.incident_on_surface(lambda x: bem.point_source(x, source)), rtol=1e-6)
        self.assertTrue(info.converged)
        probe = np.array([[0.03, 0.0, 0.0], [0.0, -0.025, 0.01], [0.0, 0.0, -0.03]])
        incident, _ = bem.point_source(cp.asarray(probe), source)
        total = cp.asnumpy(incident + bem.scattered(cp.asarray(probe), u))
        exact = rigid_sphere_point_source(probe, source, (0, 0, 0), 0.02, K)
        self.assertLess(np.linalg.norm(total - exact) / np.linalg.norm(exact), 0.02)

    def test_source_is_upstream_sinc_aperture(self):
        import cupy as cp
        from algorithm.bem.solver import RigidBEM, aperture_sources
        from algorithm.physics.sinc_aperture import aperture_field

        center = np.array([[0.02, 0.03, 0.0]])
        points = np.array([[0.02, 0.03, 0.01], [0.05, 0.01, 0.004], [0.0, 0.0, 0.09]])
        bem = RigidBEM(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64), k=K)
        quad, weights, _ = aperture_sources(center, 0.009, 32, 128)
        value, _ = bem.sources(cp.asarray(points), quad, weights)
        reference = aperture_field(points, center, np.zeros(1), K, 0.009, 32, 128)
        self.assertLess(np.abs(cp.asnumpy(value) - reference).max() / np.abs(reference).max(), 1e-10)


@unittest.skipUnless(GPU, "cupy not installed")
class SceneSolverTests(unittest.TestCase):
    def test_basis_direct_target_response_and_realtime_agree(self):
        from algorithm.bem.realtime import FocusSession
        from algorithm.bem.scene_solver import BemSceneSolver

        cfg = small_config([SPHERE])
        solver = BemSceneSolver(cfg)
        basis = solver.build_phase_response_basis()
        self.assertEqual(basis.basis.shape, (26**3, 4))
        phases = np.array([0.3, 1.2, -0.7, 2.0])
        solver.set_phases(phases)
        direct = solver.solve()
        combined = basis.field(phases, return_numpy=True)
        self.assertLess(np.linalg.norm(combined - direct) / np.linalg.norm(direct), 1e-6)
        inside = solver.obstacle_sdf < 0
        self.assertTrue(inside.any() and np.all(direct[inside] == 0))

        # Exact target response (reciprocity) agrees with the forward field at a node up
        # to the BEM discretisation error (the collocation operator is not symmetric).
        index = (13, 13, 20)
        point = np.array(index) * cfg.domain.dx
        row = basis.target_response(point[None])[0]
        reference = basis.basis[np.ravel_multi_index(index, (26, 26, 26), order="F")]
        self.assertLess(np.abs(row - reference).max() / np.abs(reference).max(), 5e-3)

        surface = (solver.bem.geo["triangles"].get().reshape(-1, 3), np.arange(3 * solver.bem.n).reshape(-1, 3))
        exact = FocusSession(solver.transducers.centers, cfg.specs.diameter, k=cfg.k0, planes=solver.planes,
                             rtol=1e-6, scattered_orders=(8, 32), block_size=None, half_precision=False)
        exact.set_surface(*surface)
        np.testing.assert_allclose(exact.response_row(point), row, rtol=0, atol=1e-5 * np.abs(row).max())

        # Real-time defaults (rtol 1e-2, coarse quadrature, block-Jacobi, float16 matrix).
        fast = FocusSession(solver.transducers.centers, cfg.specs.diameter, k=cfg.k0, planes=solver.planes)
        fast.set_surface(*surface)
        quick = fast.response_row(point)
        self.assertGreater(abs(np.sum(row * np.exp(-1j * np.angle(quick)))) / np.abs(row).sum(), 0.9999)
        basis.close()

    def test_unsupported_scenes_raise(self):
        from algorithm.bem.scene_solver import BemSceneSolver

        soft = {"-x": "soft", "+x": "open", "-y": "open", "+y": "open", "-z": "reflecting", "+z": "open"}
        with self.assertRaises(NotImplementedError):
            BemSceneSolver(small_config(boundary=soft))
        walls = {"-x": "reflecting", "+x": "reflecting", "-y": "open", "+y": "open",
                 "-z": "reflecting", "+z": "open"}
        with self.assertRaises(NotImplementedError):
            BemSceneSolver(small_config(boundary=walls))


if __name__ == "__main__":
    unittest.main()
