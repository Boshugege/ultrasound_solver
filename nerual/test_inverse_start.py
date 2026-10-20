"""Run: python -m unittest nerual.test_inverse_start -v"""
import unittest

import numpy as np
import torch

from .amplitude_inverse import SYMMETRIES, InverseResult, Prism, compare
from .inverse_start import (
    PRIOR, PoseGuess, locate, pose_features, prism_from_features, sample_prism, train)


class PoseFeatureTests(unittest.TestCase):
    def test_features_round_trip_and_ignore_the_prism_symmetry(self):
        rng = np.random.default_rng(3)
        for _ in range(20):
            prism = sample_prism(rng)
            errors = compare(prism, prism_from_features(pose_features(prism)), voxel=1e-3)
            self.assertLess(errors["center_error"], 1e-12)
            self.assertLess(errors["rotation_error_deg"], 1e-6)
            self.assertAlmostEqual(errors["side_error"], 0, places=12)
            for g in SYMMETRIES:
                turned = Prism(prism.center, tuple((prism.rotation * g).as_rotvec()), prism.side, prism.thickness)
                np.testing.assert_allclose(pose_features(turned), pose_features(prism), atol=1e-9)

    def test_samples_stay_in_the_prior(self):
        rng = np.random.default_rng(4)
        for _ in range(50):
            prism = sample_prism(rng)
            self.assertTrue(PRIOR["x"][0] <= prism.center[0] <= PRIOR["x"][1])
            self.assertTrue(PRIOR["z"][0] <= prism.center[2] <= PRIOR["z"][1])
            tilt = np.degrees(np.arcsin(np.linalg.norm(pose_features(prism)[5:7])))
            self.assertLessEqual(tilt, PRIOR["tilt"] + 1e-9)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class GuessAndFallbackTests(unittest.TestCase):
    def test_trained_bundle_predicts_a_prism(self):
        rng = np.random.default_rng(5)
        prisms = [sample_prism(rng) for _ in range(64)]
        targets = np.array([pose_features(p) for p in prisms])
        inputs = targets[:, :3].repeat(4, axis=1) * 1e-3  # inputs that determine the centre
        bundle, errors = train(inputs, targets, epochs=3)
        incident = np.ones((1, inputs.shape[1]))
        guess = PoseGuess(bundle, incident)(1 + inputs[0][None])
        self.assertIsInstance(guess, Prism)
        self.assertEqual(len(errors), 6)
        with self.assertRaises(ValueError):
            PoseGuess(bundle, np.ones((1, 5)))(np.ones((1, 5)))

    def test_locate_scans_only_when_the_guess_fits_poorly(self):
        truth = Prism((0.024, 0.024, 0.022), (0.0, 0.0, 0.0), 0.017, 0.0035)

        class Model:
            device = torch.device("cuda")

            def invert(self, start, maxiter, bounds):
                d = np.linalg.norm(np.subtract(start.center, truth.center))
                return InverseResult(start, 1e-4 if d < 2e-3 else 0.09, 1, 1, 0.0, [])

            def loss(self, params):
                return torch.linalg.vector_norm(params[:3] - params.new_tensor(truth.center))

        good = Prism(tuple(np.add(truth.center, 1e-3)), truth.rotvec, truth.side, truth.thickness)
        bad = Prism(tuple(np.add(truth.center, 5e-3)), truth.rotvec, truth.side, truth.thickness)
        self.assertEqual(locate(Model(), good, Model())[1], "guess")
        result, method = locate(Model(), bad, Model())
        self.assertEqual(method, "scan")
        self.assertLess(result.loss, 1e-3)
        self.assertEqual(locate(Model(), bad, None)[1], "guess")  # no coarse model: no scan


if __name__ == "__main__":
    unittest.main()
