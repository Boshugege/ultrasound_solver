from importlib import util
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@unittest.skipUnless(util.find_spec("warp") is not None, "warp not installed")
class MaterialBlendTests(unittest.TestCase):
    def test_coefficients_blend_linearly_and_end_values_are_exact(self):
        from algorithm.physics.warp_utils import blend_obstacle_material

        h = np.linspace(0.0, 1.0, 11)
        rho, c = blend_obstacle_material(h, 1250.0, 2200.0, 1.21, 346.0)
        np.testing.assert_allclose([rho[0], c[0], rho[-1], c[-1]], [1250.0, 2200.0, 1.21, 346.0])
        # 1/rho and 1/(rho c^2) are linear in h, so the interface (half way in 1/rho)
        # sits at h = 1/2, i.e. on the SDF zero level.
        np.testing.assert_allclose(np.diff(1 / rho, 2), 0, atol=1e-12)
        np.testing.assert_allclose(np.diff(1 / (rho * c**2), 2), 0, atol=1e-18)
        inv = 1 / rho
        self.assertAlmostEqual(inv[5], 0.5 * (inv[0] + inv[-1]))


if __name__ == "__main__":
    unittest.main()
