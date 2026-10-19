"""Constant-panel exterior Neumann BEM, exp(-i wt), reflecting z=0.

(.5 I - K) p = p_inc; p(x) = p_inc(x) + D(x) p.
Green kernels use 1/(4*pi*r); monopole incident fields use exp(ikr)/r.
This is a fluid sound-hard approximation, not an elastic plate solver.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss
from .geometry import array_centers


def triangle_rule(order=4):
    """Tensor Gauss-Duffy rule; weights sum to one (multiply by panel area)."""
    x,w = leggauss(order)
    u,v = np.meshgrid((x+1)/2, (x+1)/2, indexing="ij")
    bary = np.stack([1-u, u*(1-v), u*v], -1).reshape(-1,3)
    # Jacobian for these barycentrics is u, rather than (1-u).
    weights = (w[:,None]*w[None,:]/2*u).ravel()
    return bary, weights


def green(points, sources, k, image=True):
    p = np.asarray(points, float)
    s = np.asarray(sources, float)
    r = np.linalg.norm(p[:,None]-s[None,:], axis=-1)
    if np.any(r < 1e-10):
        raise ValueError("Green function evaluated at its source")
    result = np.exp(1j*k*r)/r
    if image:
        mirror = s*np.array([1,1,-1])
        r = np.linalg.norm(p[:,None]-mirror[None,:], axis=-1)
        if np.any(r < 1e-10):
            raise ValueError("Image Green function evaluated at its source")
        result += np.exp(1j*k*r)/r
    return result


def double_layer(points, surface, k, order=4, near_order=12, backend="numpy", image=True):
    """Integrate dG/dn_y. Refine close panel pairs with high-order quadrature.

    Direct self-panel principal value is zero for a planar constant panel.
    Image self-panel is retained. CPU/GPU assemble the same rule.
    """
    if backend == "cupy":
        from .gpu_kernel import integrate
        return integrate(points,surface,k,*triangle_rule(order),
                         *triangle_rule(max(order,near_order)),image=image)
    if backend != "numpy":
        raise ValueError("backend must be numpy or cupy")
    xp = np
    points = xp.asarray(points)
    triangles = xp.asarray(surface.triangles)
    normals = xp.asarray(surface.normals)
    areas = xp.asarray(surface.areas)
    bary,w = triangle_rule(order)
    samples = xp.einsum("qa,nad->nqd", xp.asarray(bary), triangles)
    weights = xp.asarray(w)
    output = xp.empty((len(points),len(triangles)), dtype=xp.complex128)

    def kernel(delta, normal):
        r = xp.linalg.norm(delta, axis=-1)
        dot = xp.sum(delta*normal, axis=-1)
        # A coplanar source and receiver have exactly zero direct contribution.
        dot = xp.where(xp.abs(dot) < 1e-14, 0., dot)
        r = xp.maximum(r, 1e-14)
        return xp.exp(1j*k*r)*(1-1j*k*r)*dot/(4*np.pi*r**3)

    reflect = xp.asarray([1,1,-1])
    for start in range(0,len(points),32):
        p = points[start:start+32]
        value = kernel(p[:,None,None,:]-samples[None,:,:,:], normals[None,:,None,:])
        if image:
            value += kernel(p[:,None,None,:]-samples[None,:,:,:]*reflect,
                            normals[None,:,None,:]*reflect)
        output[start:start+32] = xp.sum(value*weights, axis=-1)*areas
    if near_order > order:
        distances = xp.linalg.norm(points[:,None,:]-xp.asarray(surface.centers)[None,:,:], axis=-1)
        if image:
            mirror_distances = xp.linalg.norm(points[:,None,:]-xp.asarray(surface.centers)[None,:,:]*reflect, axis=-1)
            distances = xp.minimum(distances,mirror_distances)
        rows,cols = xp.where(distances < 1.8*xp.asarray(surface.diameters)[None,:])
        bary,w = triangle_rule(near_order)
        bary,w = xp.asarray(bary),xp.asarray(w)
        for start in range(0,len(rows),2048):
            r,c = rows[start:start+2048],cols[start:start+2048]
            q = xp.einsum("qa,nad->nqd", bary, triangles[c])
            delta = points[r,None,:]-q
            value = kernel(delta, normals[c,None,:])
            if image:
                value += kernel(points[r,None,:]-q*reflect, normals[c,None,:]*reflect)
            output[r,c] = xp.sum(value*w,axis=-1)*areas[c]
    return output


class BEMTeacher:
    def __init__(self, plate, subdivisions=10, frequency=40000., sound_speed=346.,
                 backend="numpy", order=4, near_order=12, chief_points=24,
                 thickness_layers=None):
        if not np.isfinite([frequency,sound_speed]).all() or frequency <= 0 or sound_speed <= 0:
            raise ValueError("frequency and sound_speed must be positive")
        self.plate = plate
        self.surface = plate.mesh(subdivisions,thickness_layers)
        self.k = 2*np.pi*frequency/sound_speed
        self.backend = backend
        self.order,self.near_order = order,near_order
        self.K = double_layer(self.surface.centers,self.surface,self.k,order,near_order,backend)
        if backend == "cupy":
            import cupy as xp
        else:
            xp = np
        self.xp = xp
        self.A = .5*xp.eye(len(self.K),dtype=self.K.dtype)-self.K
        self.chief = plate.interior_points(chief_points) if chief_points else np.empty((0,3))
        if len(self.chief):
            interior = double_layer(self.chief,self.surface,self.k,order,near_order,backend)
            self.A = xp.concatenate([self.A,interior])

    def incident(self, targets):
        targets = np.atleast_2d(np.asarray(targets, float))
        if (targets.ndim != 2 or targets.shape[1] != 3 or not len(targets)
                or not np.isfinite(targets).all() or (targets[:,2] <= 0).any()):
            raise ValueError("Targets must be finite Nx3 points with z>0")
        if self.plate.contains(targets,margin=1e-6).any():
            raise ValueError("A target lies in or on the plate")
        b = green(self.surface.centers,targets,self.k)
        if len(self.chief):
            b = np.concatenate([b,-green(self.chief,targets,self.k)])
        return self.xp.asarray(b,dtype=self.A.dtype)

    def solve(self, targets):
        b = self.incident(targets)
        if len(self.chief):
            return self.xp.linalg.lstsq(self.A,b,rcond=1e-7)[0]
        return self.xp.linalg.solve(self.A,b)

    def propagation(self, receivers=None):
        receivers = array_centers() if receivers is None else np.asarray(receivers,float)
        return double_layer(receivers,self.surface,self.k,self.order,self.near_order,self.backend)

    def response(self, targets, pressure=None, receivers=None):
        """targets x channels; reciprocal point sources on baffled plane.

        Divide by two for normalized e^(ikr)/r response on z=0.
        Finite transducer aperture is NOT included.
        """
        targets = np.atleast_2d(targets)
        receivers = array_centers() if receivers is None else np.asarray(receivers,float)
        if receivers.ndim != 2 or receivers.shape[1] != 3 or not np.isfinite(receivers).all():
            raise ValueError("receivers must be finite Nx3")
        if not np.allclose(receivers[:,2],0,atol=1e-12):
            raise ValueError("Array receivers must lie on z=0")
        if pressure is None:
            pressure = self.solve(targets)
        direct = self.xp.asarray(green(receivers,targets,self.k))/2
        return (direct+self.propagation(receivers)@pressure/2).T

    def to_numpy(self, x):
        return self.xp.asnumpy(x) if self.backend == "cupy" else np.asarray(x)
