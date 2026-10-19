import unittest
import numpy as np
import torch
from .differentiable import DifferentiablePlate, DifferentiableBEM
from .geometry import array_centers


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class DifferentiableBEMTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.plate = DifferentiablePlate(
            side=.02, thickness=.003, center=(.025,.025,.025),
            subdivisions=2, thickness_layers=1, chief_points=4,
            dtype=torch.float64, device="cuda")
        self.solver = DifferentiableBEM(
            receivers=array_centers(2,.0105,(.025,.025)),
            quadrature_order=3, dtype=torch.float64, device="cuda")
        self.targets = torch.tensor([[.025,.025,.042]], device="cuda",
                                    dtype=torch.float64)

    def objective(self):
        response,_ = self.solver.plate_response(self.plate,self.targets,"direct")
        weights=torch.tensor([[1.,.8,.6,.4]],device="cuda",dtype=torch.float64)
        return (weights*response.abs().square()).sum()

    def finite_difference(self, parameter, index, step):
        with torch.no_grad():
            flat=parameter.reshape(-1)
            original=flat[index].item()
            flat[index]=original+step
            plus=self.objective().item()
            flat[index]=original-step
            minus=self.objective().item()
            flat[index]=original
        return (plus-minus)/(2*step)

    def test_parameter_gradients_match_finite_difference(self):
        loss=self.objective()
        loss.backward()
        checks=[
            (self.plate.log_side,0,1e-5),
            (self.plate.log_thickness,0,1e-5),
            (self.plate.center,0,1e-6),
            (self.plate.center,2,1e-6),
            (self.plate.quaternion,0,1e-5),
        ]
        for parameter,index,step in checks:
            analytic=parameter.grad.reshape(-1)[index].item()
            numeric=self.finite_difference(parameter,index,step)
            scale=max(abs(analytic),abs(numeric),1.)
            self.assertLess(abs(analytic-numeric)/scale,3e-3,
                            (analytic,numeric))

    def test_target_and_vertex_gradients_exist(self):
        triangles,chief=self.plate.geometry()
        triangles.retain_grad()
        targets=self.targets.clone().requires_grad_()
        response,residual=self.solver(triangles,chief,targets,"direct")
        (response.abs().square().mean()+residual.mean()).backward()
        self.assertTrue(torch.isfinite(targets.grad).all())
        self.assertTrue(torch.isfinite(triangles.grad).all())
        self.assertGreater(float(targets.grad.abs().max()),0)
        self.assertGreater(float(triangles.grad.abs().max()),0)

    def test_neural_path_retains_geometry_gradient(self):
        response,residual=self.solver.plate_response(
            self.plate,self.targets,"neural")
        response.abs().square().mean().backward()
        self.assertTrue(torch.isfinite(self.plate.center.grad).all())
        self.assertGreater(float(self.plate.center.grad.abs().max()),0)
        self.assertTrue(torch.isfinite(residual).all())


if __name__=="__main__":
    unittest.main()
