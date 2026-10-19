"""Trainable accelerated Landweber network for complex boundary operators.

Geometry enters through A, not through a finite list of sampled poses.
Each layer learns step length and momentum; all layers are differentiable.
No reference solution or factorization is used by forward().
"""
import torch
from torch import nn


class BoundaryNet(nn.Module):
    def __init__(self, layers=48):
        super().__init__()
        self.layers = layers
        self.raw_steps = nn.Parameter(torch.full((layers,), -0.7))
        self.raw_momentum = nn.Parameter(torch.full((layers,), 1.5))

    def forward(self, a, b):
        """a [...,M,N], b [...,M,T], complex64/128 -> boundary pressure."""
        # ||A||_2^2 <= ||A||_1 ||A||_infinity; independent of matrix dimension.
        aa = a.abs()
        bound = aa.sum(-1).amax(-1)*aa.sum(-2).amax(-1)
        bound = bound[...,None,None].clamp_min(1e-12)
        ah = a.mH
        x = b.new_zeros(*b.shape[:-2],a.shape[-1],b.shape[-1])
        previous = x
        # Upper bound 4 permits learned over-relaxation; checked by residual gate.
        steps = 4*torch.sigmoid(self.raw_steps)
        momenta = .999*torch.sigmoid(self.raw_momentum)
        for step,momentum in zip(steps.unbind(),momenta.unbind()):
            extrapolated = x+momentum*(x-previous)
            next_x = extrapolated+step*(ah@(b-a@extrapolated))/bound
            previous,x = x,next_x
        return x

    def forward_normal(self,normal,rhs):
        """Equivalent inference with A^H A and A^H b prepared once per scene."""
        x = torch.zeros_like(rhs)
        previous = x
        steps = 4*torch.sigmoid(self.raw_steps)
        momenta = .999*torch.sigmoid(self.raw_momentum)
        for step,momentum in zip(steps.unbind(),momenta.unbind()):
            extrapolated = x+momentum*(x-previous)
            next_x = extrapolated+step*(rhs-normal@extrapolated)
            previous,x = x,next_x
        return x


def relative_residual(a,x,b):
    """Stationarity residual of CHIEF least squares, not physical model error."""
    gradient = a.mH@(a@x-b)
    scale = torch.linalg.vector_norm(a.mH@b,dim=-2).clamp_min(1e-12)
    return torch.linalg.vector_norm(gradient,dim=-2)/scale


def relative_error(x,y):
    return torch.linalg.vector_norm(x-y,dim=-2)/torch.linalg.vector_norm(y,dim=-2).clamp_min(1e-12)


class CapturedBoundaryNet:
    """Reuse a CUDA execution graph for one operator/RHS shape."""
    def __init__(self,model,a,b):
        self.a,self.b = a.clone(),b.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream),torch.inference_mode():
            for _ in range(3):
                self.output = model(self.a,self.b)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph),torch.inference_mode():
            self.output = model(self.a,self.b)

    def __call__(self,a,b):
        self.a.copy_(a)
        self.b.copy_(b)
        self.graph.replay()
        return self.output
