"""Material/source preparation and shared CPU/CUDA conservative assembly."""
import numpy as np
import warp as wp

from algorithm.physics.conservative_helmholtz import ConservativeHelmholtz, boundary_type
from algorithm.physics.sinc_aperture import source_mass_matrix, flux_rhs_columns

wp.init()


def blend_obstacle_material(h, rho_obstacle, c_obstacle, rho_medium, c_medium):
    """Density and sound speed across a smoothed obstacle interface.

    ``h`` runs from 0 inside the obstacle to 1 in the medium. The coefficients that
    enter the equation, 1/rho (flux) and 1/(rho c^2) (mass), are blended linearly,
    so the transition is centred on the SDF zero level. Blending rho itself makes a
    dense obstacle acoustically rigid well outside the zero level: a rho = 1250
    sphere in air then scatters like a rigid sphere ~1 cell larger.
    """
    inv_rho = 1/rho_obstacle+(1/rho_medium-1/rho_obstacle)*h
    compressibility = (1/(rho_obstacle*c_obstacle**2)
                       +(1/(rho_medium*c_medium**2)-1/(rho_obstacle*c_obstacle**2))*h)
    return 1/inv_rho, np.sqrt(inv_rho/compressibility)


class WarpAssemblyEngine:
    def __init__(self, nx: int, ny: int, nz: int, device: str = "cuda"):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.total_dofs = nx*ny*nz
        self.device = device if wp.is_cuda_available() and device == "cuda" else "cpu"
        self.operator = None

    def assemble_system_gpu(
        self, dx: float, omega: float, k0_bg: float,
        rho0: float, c0: float, obs_cfg: dict,
        trans_phases_np: np.ndarray, trans_amps_np: np.ndarray,
        array_n: int, trans_radius: float, trans_pitch: float,
        bcs_dict: dict, obs_sdf: np.ndarray | None = None,
        trans_centers_np: np.ndarray | None = None,
        material_fields: tuple | None = None, pml: dict | None = None,
        source_radial_order: int = 64, source_angular_order: int = 256,
    ) -> tuple:
        """Assemble in float64/complex128; k0_bg retained for API compatibility."""
        shape = (self.nx, self.ny, self.nz)
        if material_fields is not None:
            if obs_cfg:
                raise ValueError("Explicit material fields and obstacle materials are mutually exclusive")
            rho, c = (np.asarray(a, dtype=np.float64) for a in material_fields)
            if rho.shape != shape or c.shape != shape:
                raise ValueError(f"Material fields must match grid {shape}")
        elif obs_cfg:
            obs_type = obs_cfg.get("type", "sphere").lower()
            if obs_type == "mesh":
                if obs_sdf is None:
                    raise ValueError("mesh obstacle requires a precomputed SDF (obs_sdf)")
                sdf = np.asarray(obs_sdf, dtype=np.float64)
            elif obs_type == "sphere":
                coords = np.ogrid[:self.nx, :self.ny, :self.nz]
                center = np.asarray(obs_cfg.get("center", [.04, .04, .02]), dtype=float)
                sdf = np.sqrt(sum((coords[a]*dx-center[a])**2 for a in range(3)))
                sdf -= float(obs_cfg.get("radius", .012))
            else:
                raise ValueError(f"Unsupported obstacle type: {obs_type!r}")
            if sdf.shape != shape or not np.isfinite(sdf).all():
                raise ValueError("SDF must be finite and match the grid")
            width = float(obs_cfg.get("smoothing_width_cells", 1.2))*dx
            if not np.isfinite(width) or width < 0:
                raise ValueError("smoothing_width_cells must be finite and >= 0")
            if width:
                t = np.clip(sdf/width, -1, 1)
                h = .5*(1+t+np.sin(np.pi*t)/np.pi)
            else:
                h = (sdf > 0).astype(float)
            material = obs_cfg.get("material", {})
            rho_obs = float(material.get("density", 1250.))
            c_obs = float(material.get("sound_speed", 2200.))
            rho, c = blend_obstacle_material(h, rho_obs, c_obs, rho0, c0)
        else:
            rho, c = np.full(shape, rho0), np.full(shape, c0)
        self.rho_field = wp.from_numpy(np.asarray(rho, dtype=np.float64), dtype=wp.float64, device=self.device)
        self.c_field = wp.from_numpy(np.asarray(c, dtype=np.float64), dtype=wp.float64, device=self.device)

        if trans_centers_np is None:
            span = (array_n-1)*trans_pitch*.5
            xs = (self.nx-1)*dx*.5-span+np.arange(array_n)*trans_pitch
            ys = (self.ny-1)*dx*.5-span+np.arange(array_n)*trans_pitch
            trans_centers_np = np.array([[x, y, 0] for x in xs for y in ys])
        centers = np.asarray(trans_centers_np, dtype=float)
        if (centers.shape != (len(trans_phases_np), 3)
                or len(trans_amps_np) != len(centers)
                or not np.isfinite(centers).all() or np.any(centers[:, 2] != 0)):
            raise ValueError("PDE sources require finite bottom-plane centers matching phases")
        if boundary_type(bcs_dict.get("-z", "reflecting")) != "reflecting":
            raise ValueError("The sinc Neumann source requires a reflecting baffled bottom plane")
        masses = source_mass_matrix(
            shape[:2], dx, centers, trans_radius*2,
            source_radial_order, source_angular_order,
        )
        self.source_columns = flux_rhs_columns(masses, shape, dx, rho0)
        fixed = []
        self.operator = ConservativeHelmholtz(
            rho, c, dx, omega, bcs_dict, fixed,
            pml=pml, device=self.device)
        active = np.asarray(abs(self.source_columns).sum(axis=1)).ravel() > 0
        stretch = self.operator.stretch
        pml_mask = np.broadcast_to(
            stretch[0].imag+stretch[1].imag+stretch[2].imag,
            shape,
        )
        if np.any(active & (pml_mask.ravel(order="F") > 0)):
            raise ValueError("PML overlaps the deposited sinc aperture")
        if np.any(active & self.operator.fixed.ravel(order="F")):
            raise ValueError("Sinc flux conflicts with a pressure-release boundary")
        matrix = self.operator.assemble()
        rhs = self.assemble_rhs(trans_phases_np, trans_amps_np)
        return matrix, rhs

    def assemble_rhs(self, trans_phases_np, trans_amps_np):
        """Apply the integrated sinc Neumann source columns."""
        return self.source_columns @ (np.asarray(trans_amps_np)*np.exp(1j*np.asarray(trans_phases_np)))

    def fixed_dofs(self):
        if self.operator is None:
            raise RuntimeError("assemble_system_gpu() must be called first")
        return np.flatnonzero(self.operator.fixed.ravel(order="F"))
