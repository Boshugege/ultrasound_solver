"""Equilateral triangular prism. SI units; active xyzw quaternion."""
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class Plate:
    side: float = 0.03
    thickness: float = 0.004
    center: tuple = (0.08, 0.08, 0.055)
    quaternion: tuple = (0., 0., 0., 1.)

    def __post_init__(self):
        values = np.r_[self.side, self.thickness, self.center, self.quaternion]
        if values.shape != (9,) or not np.isfinite(values).all():
            raise ValueError("Expected finite side, thickness, center[3], quaternion[4]")
        if min(self.side, self.thickness) <= 0:
            raise ValueError("side and thickness must be positive (metres)")
        if not np.isclose(np.linalg.norm(self.quaternion), 1., atol=1e-6):
            raise ValueError("quaternion must have unit norm, in xyzw order")
        if self.vertices()[:, 2].min() <= 0:
            raise ValueError("The entire plate must lie above the z=0 baffle")

    @property
    def rotation(self):
        return Rotation.from_quat(self.quaternion).as_matrix()

    def vertices(self):
        r = self.side / np.sqrt(3)
        angles = np.arange(3) * 2*np.pi/3
        xy = np.c_[r*np.cos(angles), r*np.sin(angles)]
        local = np.vstack([np.c_[xy, np.full(3, z)]
                           for z in (-self.thickness/2, self.thickness/2)])
        return local @ self.rotation.T + np.asarray(self.center)

    def contains(self, points, margin=0.):
        p = (np.asarray(points)-self.center) @ self.rotation
        angles = np.arange(3)*2*np.pi/3 + np.pi/3
        outward = np.c_[np.cos(angles), np.sin(angles)]
        return ((p[..., :2] @ outward.T <= self.side/(2*np.sqrt(3))+margin).all(-1)
                & (np.abs(p[..., 2]) <= self.thickness/2+margin))

    def mesh(self, subdivisions=10, thickness_layers=None):
        if not isinstance(subdivisions,int) or not 1 <= subdivisions <= 64:
            raise ValueError("subdivisions must be an integer in [1,64]")
        m = max(1,int(np.ceil(subdivisions*self.thickness/self.side))) if thickness_layers is None else thickness_layers
        if not isinstance(m,int) or not 1 <= m <= 64:
            raise ValueError("thickness_layers must be an integer in [1,64]")
        local = _unit_triangles(subdivisions,m)*np.array([self.side,self.side,self.thickness])
        return Surface(local@self.rotation.T+np.asarray(self.center))

    def _mesh_uncached(self, subdivisions=10, thickness_layers=None):
        if not isinstance(subdivisions, int) or not 1 <= subdivisions <= 64:
            raise ValueError("subdivisions must be an integer in [1,64]")
        v = self.vertices()
        faces = []
        n = subdivisions
        m = max(1,int(np.ceil(n*self.thickness/self.side))) if thickness_layers is None else thickness_layers
        if not isinstance(m,int) or not 1 <= m <= 64:
            raise ValueError("thickness_layers must be an integer in [1,64]")
        # All facets are oriented outward below, independent of vertex ordering.
        for ids in ((0, 1, 2), (3, 4, 5)):
            a, b, c = v[list(ids)]
            point = lambda i, j: a+(b-a)*i/n+(c-a)*j/n
            for i in range(n):
                for j in range(n-i):
                    faces.append([point(i,j), point(i+1,j), point(i,j+1)])
                    if i+j < n-1:
                        faces.append([point(i+1,j), point(i+1,j+1), point(i,j+1)])
        for i in range(3):
            j = (i+1) % 3
            for k in range(n):
                for layer in range(m):
                    d = (v[i+3]-v[i])/m
                    a = v[i]+(v[j]-v[i])*k/n+layer*d
                    b = v[i]+(v[j]-v[i])*(k+1)/n+layer*d
                    faces.extend([[a,b,b+d], [a,b+d,a+d]])
        triangles = np.asarray(faces)
        cross = np.cross(triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0])
        inward = np.einsum("ij,ij->i", cross, triangles.mean(1)-self.center) < 0
        triangles[inward] = triangles[inward][:, [0,2,1]]
        return Surface(triangles)

    def interior_points(self, count=24):
        """Deterministic CHIEF points strictly inside; no test-dependent tuning."""
        from scipy.stats import qmc
        u = qmc.Halton(3,scramble=False).random(count+1)[1:]
        r = np.sqrt(u[:,0])
        bary = np.c_[1-r,r*(1-u[:,1]),r*u[:,1]]
        # Keep all points away from panel singularities.
        bary = .7*bary+.1
        v = self.vertices()
        base = (v[:3]+v[3:])/2
        return bary@base+((u[:,2]-.5)*.7*self.thickness)[:,None]*self.rotation[:,2]


@lru_cache(maxsize=32)
def _unit_triangles(n,m):
    unit = Plate(1.,1.,(0.,0.,1.))
    triangles = unit._mesh_uncached(n,m).triangles-np.array([0.,0.,1.])
    triangles.setflags(write=False)
    return triangles


@dataclass
class Surface:
    triangles: np.ndarray

    def __post_init__(self):
        self.centers = self.triangles.mean(1)
        cross = np.cross(self.triangles[:,1]-self.triangles[:,0],
                         self.triangles[:,2]-self.triangles[:,0])
        self.areas = np.linalg.norm(cross, axis=1)/2
        self.normals = cross/(2*self.areas[:,None])
        self.diameters = np.max([np.linalg.norm(self.triangles[:,i]-self.triangles[:,j], axis=1)
                                 for i,j in ((0,1),(1,2),(2,0))], axis=0)


def array_centers(n=16, pitch=0.0105, center=(0.08, 0.08)):
    xy = (np.arange(n)-(n-1)/2)*pitch
    x,y = np.meshgrid(xy+center[0], xy+center[1], indexing="ij")
    return np.c_[x.ravel(), y.ravel(), np.zeros(n*n)]
