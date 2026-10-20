"""Run: python -m unittest nerual.test_amplitude_inverse -v"""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

from .amplitude_inverse import (
    SCALE, SYMMETRIES, Prism, PrismInverse, array_centers, compare, observation_points, wavenumber)

PRISM = Prism.from_euler((25, 23, 23.5), (12, -8, 15), 18, 3)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class AmplitudeInverseTests(unittest.TestCase):
    def model(self, observed=None, **kwargs):
        kwargs.setdefault("panels_per_side", 8)
        kwargs.setdefault("thickness_layers", 1)
        observed = np.ones((1, 1280)) if observed is None else observed
        return PrismInverse(observed, np.ones((1, 9)), **kwargs)

    def test_compare_respects_the_prism_symmetry(self):
        for g in SYMMETRIES:
            turned = Prism(PRISM.center, tuple((PRISM.rotation * g).as_rotvec()), PRISM.side, PRISM.thickness)
            errors = compare(PRISM, turned)
            self.assertLess(errors["rotation_error_deg"], 1e-6)
            self.assertGreater(errors["iou"], 0.999)
        shifted = Prism(tuple(np.add(PRISM.center, (1e-3, 0, 0))), PRISM.rotvec, PRISM.side, PRISM.thickness)
        self.assertAlmostEqual(compare(PRISM, shifted)["center_error"], 1e-3)
        self.assertLess(compare(PRISM, shifted)["iou"], 0.95)

    def test_gradient_matches_central_differences(self):
        model = self.model()
        z = PRISM.params() / SCALE
        _, grad = model.value_and_grad(z)
        for i in range(8):
            step = np.zeros(8)
            step[i] = 1e-4
            numeric = (model.value_and_grad(z + step)[0] - model.value_and_grad(z - step)[0]) / 2e-4
            self.assertLess(abs(grad[i] - numeric), 1e-4 * np.abs(grad).max() + 1e-12, i)

    def test_field_agrees_with_the_burton_miller_bem(self):
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "src"))
        import cupy as cp
        from algorithm.bem.solver import RigidBEM, aperture_sources

        triangles = PRISM.triangles(16, 3)
        points = triangles.reshape(-1, 3)
        _, first, inverse = np.unique(np.round(points / 1e-9).astype(np.int64), axis=0,
                                      return_index=True, return_inverse=True)
        bem = RigidBEM(points[first], inverse.reshape(-1, 3), k=wavenumber())
        A = bem.assemble()
        sources, weights, _ = aperture_sources(array_centers(), 0.009, 32, 128)
        u = cp.linalg.solve(A, bem.rhs(*bem.sources(bem.geo["centroids"], sources, weights)))
        observed = cp.asarray(observation_points())
        reference = cp.asnumpy(bem.sources(observed, sources, weights)[0] + bem.scattered(observed, u))
        scattered = reference - cp.asnumpy(bem.sources(observed, sources, weights)[0])
        model = self.model(panels_per_side=16, thickness_layers=3)
        params = torch.as_tensor(PRISM.params(), dtype=torch.float64, device="cuda")
        with torch.no_grad():
            field = model.field(params)[0].cpu().numpy()
        # same mesh, different formulation (double layer vs Burton-Miller) and quadrature
        error = np.linalg.norm(field - reference) / np.linalg.norm(scattered)
        self.assertLess(error, 0.05)

    def test_inversion_recovers_a_nearby_prism_from_its_own_data(self):
        params = torch.as_tensor(PRISM.params(), dtype=torch.float64, device="cuda")
        with torch.no_grad():
            observed = self.model().field(params).abs().cpu().numpy()
        start = Prism(tuple(np.add(PRISM.center, (1e-3, -0.5e-3, 0.5e-3))), PRISM.rotvec,
                      PRISM.side * 1.05, PRISM.thickness)
        result = self.model(observed).invert(start, maxiter=60)
        errors = compare(PRISM, result.prism)
        self.assertLess(errors["center_error"], 1e-5)
        self.assertGreater(errors["iou"], 0.99)


if __name__ == "__main__":
    unittest.main()
