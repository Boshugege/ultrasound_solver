"""Fast CHIEF-stabilized BEM for arbitrary closed rigid triangle meshes."""
import numpy as np
from .bem import double_layer, green


class MeshBEM:
    def __init__(self, body, frequency=40000., sound_speed=346.,
                 backend="cupy", order=4, near_order=12):
        self.body = body
        self.surface = body.surface
        self.k = 2*np.pi*frequency/sound_speed
        self.backend = backend
        self.order, self.near_order = order, near_order
        if backend == "cupy":
            import cupy as xp
        elif backend == "numpy":
            xp = np
        else:
            raise ValueError("backend must be cupy or numpy")
        self.xp = xp
        K = double_layer(self.surface.centers, self.surface, self.k,
                         order, near_order, backend)
        square = .5*xp.eye(len(K), dtype=K.dtype)-K
        chief = double_layer(body.interior_points, self.surface, self.k,
                             order, near_order, backend)
        self.A = xp.concatenate((square, chief))

    def incident(self, targets):
        targets = np.atleast_2d(np.asarray(targets, float))
        if targets.ndim != 2 or targets.shape[1] != 3 or not np.isfinite(targets).all():
            raise ValueError("targets must be finite [T,3]")
        b = np.concatenate((green(self.surface.centers, targets, self.k),
                            -green(self.body.interior_points, targets, self.k)))
        return self.xp.asarray(b, dtype=self.A.dtype)

    def propagation(self, points):
        return double_layer(np.asarray(points, float), self.surface, self.k,
                            self.order, self.near_order, self.backend)

    def direct(self, b):
        return self.xp.linalg.lstsq(self.A, b, rcond=1e-6)[0]

    def normal_residual(self, x, b):
        ah = self.A.conj().T
        return self.xp.linalg.norm(ah@(self.A@x-b))/self.xp.linalg.norm(ah@b)
