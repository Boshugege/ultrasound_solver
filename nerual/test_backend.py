from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

from algorithm.config import SimulationConfig
from algorithm.bem.scene_solver import BemSceneSolver
from .backend import NeuralBemSceneSolver


class NeuralBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=SimulationConfig.from_yaml(
            ROOT/"nerual"/"examples"/"neural_bem_triangle_2x2.yaml")
        cls.solver=NeuralBemSceneSolver(cls.cfg)

    @classmethod
    def tearDownClass(cls):
        cls.solver.close()

    def test_neural_target_response_and_residual_gate(self):
        target=np.asarray(self.cfg.targets,float)
        self.solver.target_response(target)
        neural=self.solver.target_response(target)
        self.assertEqual(self.solver.last_neural_stats["method"],"neural")
        self.assertLess(self.solver.last_neural_stats["residual"],
                        self.solver.neural_tolerance)
        accurate=BemSceneSolver.target_response(self.solver,target)
        scale=np.vdot(neural,accurate)/np.vdot(neural,neural)
        error=np.linalg.norm(scale*neural-accurate)/np.linalg.norm(accurate)
        focus=abs(np.sum(accurate*np.exp(-1j*np.angle(neural))))/np.abs(accurate).sum()
        self.assertLess(error,.08)
        self.assertGreater(focus,.99)

        old=self.solver.neural_tolerance
        self.solver.neural_tolerance=1e-12
        fallback=self.solver.target_response(target)
        self.solver.neural_tolerance=old
        self.assertEqual(self.solver.last_neural_stats["method"],
                         "accurate_bem_fallback")
        np.testing.assert_allclose(fallback,accurate,rtol=2e-5,atol=1e-7)


if __name__=="__main__":
    unittest.main()
