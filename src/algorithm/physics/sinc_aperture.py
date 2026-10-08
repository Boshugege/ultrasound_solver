"""Finite aperture whose far-field directivity is SonicSurface's sinc.

For a=diameter/2, surface density f(r)=1/(2*pi*a*sqrt(a*a-r*r))
has unit integral and 2D Fourier transform sinc(a*|q|/pi). With
exp(-i*omega*t), prescribing outward (1/rho)*dp/dn=2*pi*f/rho_ref
on a baffled plane gives p=int f(s)*exp(ik|x-s|)/|x-s| ds in a
homogeneous reference medium. Unit emitter weights give p in 1/m.

The official D(theta)*exp(ikr)/r is the far-field approximation of this
integral, not its exact near field. No target-dependent fitting is used.
"""
import numpy as np
import scipy.sparse as sp
from numpy.polynomial.legendre import leggauss

MODEL_VERSION = "sinc_neumann_projected_hemisphere_v1"


def aperture_quadrature(diameter, radial_order=32, angular_order=128):
    if not np.isfinite(diameter) or diameter <= 0:
        raise ValueError("Aperture diameter must be positive")
    if radial_order < 2 or angular_order < 8:
        raise ValueError("Aperture quadrature orders are too small")
    # mu=cos(theta) cancels the integrable rim singularity exactly.
    mu, weights = leggauss(radial_order)
    mu, weights = (mu+1)/2, weights/2
    radius = diameter/2*np.sqrt(1-mu*mu)
    phi = 2*np.pi*np.arange(angular_order)/angular_order
    offsets = np.column_stack(((radius[:, None]*np.cos(phi)).ravel(),
                               (radius[:, None]*np.sin(phi)).ravel(),
                               np.zeros(radial_order*angular_order)))
    weights = np.repeat(weights/angular_order, angular_order)
    return offsets, weights


def source_mass_matrix(shape_xy, dx, centers, diameter, radial_order=64, angular_order=256):
    """Sparse xy-node x emitter weights, each column has unit mass.

    Bilinear deposition preserves mass and the off-grid emitter centroid,
    avoids evaluating the singular rim, and converges as grid/quadrature refine.
    Support extends by at most one grid interval beyond the physical aperture.
    """
    shape_xy = tuple(shape_xy)
    centers = np.asarray(centers, float)
    if (len(shape_xy) != 2 or min(shape_xy) < 3 or not np.isfinite(dx) or dx <= 0
            or centers.ndim != 2 or centers.shape[1] != 3 or not len(centers)
            or not np.isfinite(centers).all() or np.any(centers[:, 2] != 0)):
        raise ValueError("Sources require finite centers on z=0 and a valid xy grid")
    offsets, weights = aperture_quadrature(diameter, radial_order, angular_order)
    extent = (np.asarray(shape_xy)-1)*dx
    # Keep all deposited mass away from side faces (and validate PML separately).
    if np.any(centers[:, :2]-diameter/2 < dx) or np.any(centers[:, :2]+diameter/2 > extent-dx):
        raise ValueError("Sinc aperture needs one grid cell clearance from side boundaries")
    rows, columns, values = [], [], []
    for channel, center in enumerate(centers):
        coordinates = (center[:2]+offsets[:, :2])/dx
        low = np.floor(coordinates).astype(int)
        fraction = coordinates-low
        local_rows, local_values = [], []
        for a, b in ((0, 0), (0, 1), (1, 0), (1, 1)):
            mass = weights*(fraction[:, 0] if a else 1-fraction[:, 0])*(fraction[:, 1] if b else 1-fraction[:, 1])
            local_rows.append(low[:, 0]+a+shape_xy[0]*(low[:, 1]+b))
            local_values.append(mass)
        unique, inverse = np.unique(np.concatenate(local_rows), return_inverse=True)
        rows.append(unique)
        columns.append(np.full(len(unique), channel))
        values.append(np.bincount(inverse, weights=np.concatenate(local_values)))
    matrix = sp.coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
                          shape=(np.prod(shape_xy), len(centers))).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return matrix


def flux_rhs_columns(masses, shape, dx, reference_density):
    """Integrated conservative RHS for fixed normalized velocity/flux.

    Reference density is held fixed as spatial material parameters vary, so
    db/d(rho,c)=0. This preserves the existing matrix-only material adjoint.
    """
    if not np.isfinite(reference_density) or reference_density <= 0:
        raise ValueError("reference_density must be positive")
    bottom = -2*np.pi/(reference_density*dx**3)*masses
    return sp.vstack((bottom, sp.csr_matrix((np.prod(shape)-masses.shape[0], masses.shape[1]))),
                     format="csr")


def aperture_field(points, centers, phases, k, diameter, radial_order=12,
                   angular_order=48, device="cpu"):
    """Independent Rayleigh integral using exact outgoing Green kernels.

    Only a verification reference. This function is never called by PDE solves.
    """
    xp = np
    if device == "cuda":
        import cupy as cp
        xp = cp
    elif device != "cpu":
        raise ValueError("device must be cpu or cuda")
    points, centers, phases = np.asarray(points), np.asarray(centers), np.asarray(phases)
    if (points.ndim != 2 or points.shape[1] != 3 or np.any(points[:, 2] <= 0)
            or centers.ndim != 2 or centers.shape != (len(phases), 3)
            or np.any(centers[:, 2] != 0) or not np.isfinite(points).all()
            or not np.isfinite(centers).all() or not np.isfinite(phases).all()
            or not np.isfinite(k) or k <= 0):
        raise ValueError("Finite upper-half-space points and bottom-plane sources required")
    offsets, weights = aperture_quadrature(diameter, radial_order, angular_order)
    emitters = xp.asarray((centers[:, None, :]+offsets[None, :, :]).reshape(-1, 3))
    strength = xp.asarray((np.exp(1j*phases[:, None])*weights).ravel())
    result = np.empty(len(points), complex)
    for start in range(0, len(points), 128):
        p = xp.asarray(points[start:start+128])
        value = xp.zeros(len(p), dtype=xp.complex128)
        for offset in range(0, len(emitters), 4096):
            s = emitters[offset:offset+4096]
            distance = xp.sqrt(sum((p[:, None, axis]-s[None, :, axis])**2 for axis in range(3)))
            value += (xp.exp(1j*k*distance)/distance) @ strength[offset:offset+4096]
        result[start:start+len(p)] = value if xp is np else xp.asnumpy(value)
    return result
