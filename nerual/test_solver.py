"""Run: conda run -n srt python -m unittest nerual.test_solver -v"""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import Plate
from .bem import BEMTeacher,green,double_layer
from .aperture import aperture_points
from .solver import TriangleSolver


class PhysicalContractTests(unittest.TestCase):
    def test_rotated_prism_volume_and_containment(self):
        rotation = Rotation.from_euler("xyz",[79,-51,133],degrees=True)
        plate = Plate(.032,.006,(.07,.09,.06),tuple(rotation.as_quat()))
        mesh = plate.mesh(5)
        volume = np.einsum("ij,ij->i",mesh.centers,mesh.normals)@mesh.areas/3
        self.assertAlmostEqual(volume,np.sqrt(3)/4*plate.side**2*plate.thickness,places=12)
        self.assertTrue(plate.contains([plate.center])[0])
        outside = np.array(plate.center)+rotation.apply([0,0,.004])
        self.assertFalse(plate.contains([outside])[0])

    def test_input_rejection(self):
        for kwargs in ({"side":0},{"thickness":-1},{"quaternion":(0,0,0,0)},{"center":(0,0,0)}):
            with self.assertRaises(ValueError):
                Plate(**kwargs)
        teacher = BEMTeacher(Plate(),3)
        with self.assertRaises(ValueError):
            teacher.incident([[.08,.08,.055]])

    def test_reciprocity_independent_source_target_swap(self):
        teacher = BEMTeacher(Plate(),10)
        points = np.array([[.04,.065,.075],[.08,.08,.105]])
        p = teacher.solve(points)
        scatter = double_layer(points,teacher.surface,teacher.k)@p
        direct01 = green(points[:1],points[1:],teacher.k)[0,0]
        g01,g10 = direct01+scatter[0,1],direct01+scatter[1,0]
        self.assertLess(abs(g01-g10)/max(abs(g01),abs(g10)),.015)

    def test_phase_and_quantization_against_independent_response(self):
        solver = TriangleSolver(device="cpu",subdivisions=5,diameter=.009)
        solver.set_plate(Plate())
        result = solver.solve([[.08,.08,.1]],method="direct")
        optimal = np.abs(result.response).sum(1)
        achieved = np.abs((result.response*np.exp(1j*result.phases_rad)).sum(1))
        quantized = np.abs((result.response*np.exp(1j*result.phase_codes_32.astype(float)*2*np.pi/32)).sum(1))
        np.testing.assert_allclose(achieved,optimal,rtol=2e-6)
        self.assertGreaterEqual(float((quantized/optimal).min()),np.cos(np.pi/32)-1e-6)
        self.assertEqual(result.phase_codes_32.shape,(1,256))
        self.assertLess(result.normal_residual,1e-4)

    def test_aperture_moments(self):
        points,w = aperture_points([[0,0,0]],diameter=.009)
        r2 = (points[0,:,:2]**2).sum(1)
        self.assertAlmostEqual(float(w.sum()),1.)
        self.assertAlmostEqual(float(w@r2),(.009/2)**2/2,places=12)
        self.assertAlmostEqual(float(w@(r2*r2)),(.009/2)**4/3,places=14)


if __name__=="__main__":
    unittest.main()
