import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import Plate
from .mesh_scene import MeshBody, box_body, plate_body
from .general_solver import GeneralNeuralSolver


class GeneralSolverTests(unittest.TestCase):
    def test_mesh_validation_and_rigid_transform(self):
        plate = plate_body(Plate(), 4, 2)
        moved = plate.transformed((.01, -.01, .005),
                                  tuple(Rotation.from_euler("z", 30, degrees=True).as_quat()))
        self.assertEqual(moved.triangles.shape, plate.triangles.shape)
        np.testing.assert_allclose(moved.surface.areas, plate.surface.areas)
        with self.assertRaises(ValueError):
            MeshBody(np.zeros((1,3,3)), [[0,0,0]])

    def test_reflector_combination(self):
        plate = plate_body(Plate(), 4, 2)
        reflector = box_body((.02,.03,.002),(.12,.08,.06))
        combined = MeshBody.combine((plate, reflector))
        self.assertEqual(len(combined.triangles), len(plate.triangles)+12)
        self.assertEqual(len(combined.interior_points), len(plate.interior_points)+1)

    def test_neural_api_and_phase_contract(self):
        solver = GeneralNeuralSolver(device="cuda")
        solver.set_scene(plate_body(Plate(), 6, 3))
        solver.solve([[.08,.08,.1]], "auto")
        result = solver.solve([[.08,.08,.1]], "auto")
        self.assertIn(result.method, {"neural","direct_fallback"})
        self.assertLess(result.final_residual, .02)
        self.assertEqual(result.response.shape, (1,256))
        self.assertEqual(result.phase_codes_32.dtype, np.uint8)
        optimal = np.abs(result.response).sum(1)
        achieved = np.abs((result.response*np.exp(1j*result.phases_rad)).sum(1))
        np.testing.assert_allclose(achieved, optimal, rtol=2e-6)


if __name__ == "__main__":
    unittest.main()
