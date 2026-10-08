"""Node-centred finite volumes for div(rho^-1 grad p) + omega²/(rho c²)p.

Time convention: exp(-i omega t). Rows are integrated over nodal control
volumes and divided by dx³, including half volumes at the six physical faces.
CPU and CUDA use the same float64/complex128 assembly expressions.
"""
import numpy as np
import scipy.sparse as sp

FACES = ("-x", "+x", "-y", "+y", "-z", "+z")


def boundary_type(value):
    return value.get("type", "reflecting") if isinstance(value, dict) else value


class ConservativeHelmholtz:
    def __init__(self, rho, c, dx, omega, boundaries, fixed_dofs=(),
                 pml=None, device="cpu"):
        self.rho = np.asarray(rho, dtype=np.float64)
        self.c = np.asarray(c, dtype=np.float64)
        self.shape = self.rho.shape
        if (len(self.shape) != 3 or min(self.shape) < 2
                or self.c.shape != self.shape):
            raise ValueError("rho and c must have the same 3D grid shape, each dimension >= 2")
        if not all(np.isfinite(a).all() and (a > 0).all()
                   for a in (self.rho, self.c)):
            raise ValueError("density and sound speed must be finite and positive")
        if not np.isfinite(dx) or dx <= 0 or not np.isfinite(omega) or omega <= 0:
            raise ValueError("dx and omega must be finite and positive")
        if set(boundaries) - set(FACES):
            raise ValueError("Unknown boundary face")
        self.dx, self.omega = float(dx), float(omega)
        self.boundaries = boundaries
        self.device = device
        self.weights = []
        self.stretch = []
        pml = pml or {}
        cells = int(pml.get("cells", 0))
        strength = float(pml.get("strength", 2.0))
        order = int(pml.get("order", 3))
        if cells < 0 or strength < 0 or not np.isfinite(strength) or order < 1:
            raise ValueError("Invalid PML cells, strength or order")
        pml_faces = pml.get("faces", [f for f in FACES
                                    if boundary_type(boundaries.get(f, "reflecting")) == "open"])
        if set(pml_faces) - set(FACES):
            raise ValueError("Unknown PML face")
        for axis, n in enumerate(self.shape):
            w = np.ones(n)
            w[[0, -1]] = .5
            dims = [1, 1, 1]
            dims[axis] = n
            self.weights.append(w.reshape(dims))
            sigma = np.zeros(n)
            active = [f for f in FACES[2*axis:2*axis+2] if f in pml_faces]
            if cells and active:
                if cells * len(active) >= n - 1:
                    raise ValueError("PML leaves no physical interior on an axis")
                for face in active:
                    if boundary_type(boundaries.get(face, "reflecting")) != "open":
                        raise ValueError("PML may only be placed on open faces")
                    distance = np.arange(n) if face[0] == "-" else np.arange(n)[::-1]
                    sigma += strength * np.maximum(1 - distance/cells, 0)**order
            self.stretch.append((1 + 1j*sigma).reshape(dims))
        self.volume = self.weights[0]*self.weights[1]*self.weights[2]
        self.mass_weight = self.volume*self.stretch[0]*self.stretch[1]*self.stretch[2]
        self.fixed = np.zeros(self.shape, dtype=bool, order="F")
        self.fixed.ravel(order="F")[np.asarray(fixed_dofs, dtype=int)] = True
        source_mask = self.fixed.copy()
        pml_mask = np.broadcast_to(
            (self.stretch[0].imag + self.stretch[1].imag + self.stretch[2].imag) > 0,
            self.shape)
        if np.any(self.fixed & pml_mask):
            raise ValueError("PML overlaps a source; enlarge the domain or reduce PML thickness")
        self.face_terms = []
        for index, face in enumerate(FACES):
            axis = index//2
            selection = [slice(None)]*3
            selection[axis] = 0 if index % 2 == 0 else -1
            selection = tuple(selection)
            value = boundaries.get(face, "reflecting")
            kind = boundary_type(value)
            if kind not in {"reflecting", "open", "soft", "impedance"}:
                raise ValueError(f"Unsupported boundary {face}: {value!r}")
            if kind == "soft":
                if np.any(source_mask[selection]):
                    raise ValueError("A pressure-release face conflicts with a source")
                self.fixed[selection] = True
            if kind in {"open", "impedance"}:
                others = [a for a in range(3) if a != axis]
                area = (self.weights[others[0]]*self.weights[others[1]]
                        *self.stretch[others[0]]*self.stretch[others[1]])
                area = np.broadcast_to(area, self.shape)[selection]/dx
                if kind == "open":
                    admittance = 1/(self.rho[selection]*self.c[selection])
                else:
                    if not isinstance(value, dict):
                        raise ValueError("impedance requires {type: impedance, impedance: [real, imag]}")
                    z = np.asarray(value.get("impedance"), dtype=float)
                    if z.shape != (2,) or not np.isfinite(z).all() or z[0] < 0 or not np.any(z):
                        raise ValueError("impedance must be finite [real>=0, imag], nonzero, in Pa s/m")
                    admittance = 1/complex(*z)
                self.face_terms.append((selection, kind, 1j*omega*area*admittance))

    def edges(self):
        for axis in range(3):
            low, high = [slice(None)]*3, [slice(None)]*3
            low[axis], high[axis] = slice(None, -1), slice(1, None)
            low, high = tuple(low), tuple(high)
            others = [a for a in range(3) if a != axis]
            # Stretch evaluated at the face midpoint; symmetric for both nodes.
            stretch = self.stretch[axis]
            midpoint = .5*(stretch[low]+stretch[high])
            factor = (self.weights[others[0]]*self.weights[others[1]]
                      *self.stretch[others[0]]*self.stretch[others[1]]
                      /midpoint/self.dx**2)
            yield low, high, factor

    def assemble(self):
        xp = np
        if self.device == "cuda":
            import cupy as cp
            xp = cp
        rho, c = xp.asarray(self.rho), xp.asarray(self.c)
        ids = xp.arange(rho.size).reshape(self.shape, order="F")
        fixed = xp.asarray(self.fixed)
        diagonal = xp.asarray(self.mass_weight)*self.omega**2/(rho*c*c)
        rows, cols, values = [], [], []
        for low, high, factor in self.edges():
            coefficient = xp.asarray(factor)*2/(rho[low]+rho[high])
            diagonal[low] -= coefficient
            diagonal[high] -= coefficient
            for src, dst in ((low, high), (high, low)):
                mask = ~fixed[src]
                rows.append(ids[src][mask])
                cols.append(ids[dst][mask])
                values.append(coefficient[mask])
        for selection, _, term in self.face_terms:
            diagonal[selection] += xp.asarray(term)
        diagonal[fixed] = 1
        rows.append(ids.ravel())
        cols.append(ids.ravel())
        values.append(diagonal.ravel())
        rows, cols, values = map(xp.concatenate, (rows, cols, values))
        if xp is not np:
            rows, cols, values = map(xp.asnumpy, (rows, cols, values))
        matrix = sp.coo_matrix((values, (rows, cols)), shape=(rho.size, rho.size)).tocsr()
        matrix.eliminate_zeros()
        return matrix

    def material_vjp(self, field, adjoint):
        """dL = -Re(lambda^H dA p), holding prescribed pressure RHS fixed.

        Includes all density flux terms, mass terms, and local open impedance.
        PML stretch and explicitly prescribed wall impedance are held fixed.
        """
        u = np.asarray(field).reshape(self.shape, order="F")
        lam = np.asarray(adjoint).reshape(self.shape, order="F").copy()
        lam[self.fixed] = 0  # Dirichlet rows have no material dependence.
        mass = self.mass_weight*self.omega**2/(self.rho*self.c**2)
        product = np.conj(lam)*u*mass
        grad_rho = np.real(product/self.rho)
        grad_c = np.real(2*product/self.c)
        for low, high, factor in self.edges():
            dr = -2*factor/(self.rho[low]+self.rho[high])**2
            contribution = -np.real(
                (np.conj(lam[low])-np.conj(lam[high]))*(u[high]-u[low])*dr)
            grad_rho[low] += contribution
            grad_rho[high] += contribution
        for selection, kind, term in self.face_terms:
            if kind == "open":
                contribution = np.real(np.conj(lam[selection])*u[selection]*term)
                grad_rho[selection] += contribution/self.rho[selection]
                grad_c[selection] += contribution/self.c[selection]
        return grad_rho, grad_c
