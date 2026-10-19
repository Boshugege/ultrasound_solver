"""Normalized circular piston quadrature on z=0.

Radau in r^2: nodes 0, 2/3; weights 1/4, 3/4. Twelve azimuths.
No empirical directivity fit; average the half-space Green response.
"""
import numpy as np


def aperture_points(centers,diameter=.009):
    centers = np.asarray(centers,float)
    if not np.isfinite(diameter) or diameter<0:
        raise ValueError("diameter must be finite and nonnegative")
    if diameter==0:
        return centers[:,None,:],np.ones(1)
    angle = np.arange(12)*2*np.pi/12
    offsets = np.vstack([np.zeros((1,3)),
                         np.c_[np.cos(angle),np.sin(angle),np.zeros(12)]*diameter/2*np.sqrt(2/3)])
    return centers[:,None,:]+offsets[None,:,:],np.r_[.25,np.full(12,.75/12)]


def sinc_aperture_points(centers, diameter=.009, radial_order=12, angular_order=48):
    """Project sinc-Neumann aperture quadrature used by the PDE onto each center."""
    from numpy.polynomial.legendre import leggauss
    centers = np.asarray(centers, float)
    mu, weights = leggauss(radial_order)
    mu, weights = (mu+1)/2, weights/2
    radius = diameter/2*np.sqrt(1-mu*mu)
    phi = 2*np.pi*np.arange(angular_order)/angular_order
    offsets = np.column_stack(((radius[:,None]*np.cos(phi)).ravel(),
                               (radius[:,None]*np.sin(phi)).ravel(),
                               np.zeros(radial_order*angular_order)))
    return centers[:,None,:]+offsets[None,:,:], np.repeat(weights/angular_order,
                                                          angular_order)
