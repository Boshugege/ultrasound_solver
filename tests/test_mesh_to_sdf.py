from importlib import util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
MODELS = ROOT / "obstacle_displacement_study" / "models"


@unittest.skipUnless(util.find_spec("pyvista") is not None, "pyvista not installed")
class MeshToSdfTests(unittest.TestCase):
    def check(self, name, grid, box, center, rotation, scale):
        from algorithm.physics.mesh_to_sdf import _load_closed_surface, _place_mesh, mesh_to_sdf

        sdf = mesh_to_sdf(MODELS / name, (grid,) * 3, (box,) * 3, center, rotation, scale)
        placed = _place_mesh(_load_closed_surface(MODELS / name), np.asarray(center, float),
                             np.asarray(rotation, float), scale)
        inside_volume = (sdf < 0).sum() * (box / (grid - 1)) ** 3
        self.assertAlmostEqual(inside_volume / placed.volume, 1.0, delta=0.1)

    def test_hand_interior_matches_mesh_volume(self):
        # The closest-facet sign rule used to mark 60x the hand's volume as inside.
        self.check("hand_watertight.vtk", 70, 0.1, (0.0485, 0.05, 0.048), (0, 0, 90), 0.18)

    def test_convex_shapes(self):
        self.check("plate_22x22x3mm.vtk", 70, 0.1, (0.0485, 0.05, 0.045), (0, 0, 0), 1.0)
        self.check("sphere_d20mm.vtk", 70, 0.1, (0.0485, 0.05, 0.045), (0, 0, 0), 1.0)


if __name__ == "__main__":
    unittest.main()
