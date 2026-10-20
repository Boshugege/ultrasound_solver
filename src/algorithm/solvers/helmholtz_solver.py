import time
import os
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("CUPY_CACHE_IN_MEMORY", "1")
import cupy as cp
import cupyx.scipy.sparse as csp
import numpy as np
import warp as wp
from nvmath.sparse.advanced import DirectSolver

from algorithm.config import SimulationConfig
from algorithm.physics.conservative_helmholtz import FACES, boundary_type
from algorithm.physics.sinc_aperture import source_mass_matrix
from algorithm.physics.transducer_array import TransducerArray
from algorithm.physics.warp_utils import WarpAssemblyEngine
from algorithm.solvers.boundary_condensed_solver import BoundaryCondensedDirectSolver
from algorithm.solvers.homogeneous_surface import HomogeneousSurfaceSolver
from algorithm.solvers.iterative_solver import GpuIterativeSolver


class HelmholtzDirectSolver:
    """Assemble and solve the discrete Helmholtz system."""

    # v5: obstacle materials blend 1/rho and 1/(rho c^2) across the SDF interface.
    MODEL_VERSION = "helmholtz_conservative_sources_v5"
    field_unit = "Pa"

    def __init__(self, config: SimulationConfig):
        self.cfg = config
        self.transducers = TransducerArray(config)
        self.field_unit = "normalized_1_per_m"
        self.origin = np.asarray(config.domain.origin_m, dtype=np.float64)
        if not np.isclose(self.origin[2], 0.0):
            raise ValueError("The transducer plane requires domain.origin_m[2] == 0")
        self.total_dofs = (
            self.cfg.domain.nx
            * self.cfg.domain.ny
            * self.cfg.domain.nz
        )
        self.warp_engine = WarpAssemblyEngine(
            nx=self.cfg.domain.nx,
            ny=self.cfg.domain.ny,
            nz=self.cfg.domain.nz,
            device="cuda",
        )

        self.A_cached_csr_cpu = None
        self.A_cached_csr_gpu = None
        self.A_cached_H_csr_gpu = None
        self.rhs_cached = None
        self._obs_sdf = None
        self._scene_key = None
        self._material_fields = None
        self._grid_definition = asdict(config.domain)
        self._source_definition = asdict(config.specs)
        self.condensed_solver = None
        self._target_surface_solver = None
        self._target_source_masses = None
        self.iterative_solver = GpuIterativeSolver(
            restart=int(self.cfg.solver.get("gmres_restart", 30)),
            maxiter=int(self.cfg.solver.get("gmres_maxiter", 3000)),
            rtol=float(self.cfg.solver.get("gmres_rtol", 1.0e-6)),
            backward_error_tol=float(
                self.cfg.solver.get(
                    "gmres_backward_error_tol",
                    1.0e-4,
                )
            ),
        )

    def _ensure_target_surface_solver(self):
        if self.cfg.obstacles or self.cfg.physics.material_field_file or self._material_fields is not None:
            raise ValueError("Surface condensation requires a homogeneous medium")
        if boundary_type(self.cfg.boundary_conditions.get("-z", "reflecting")) != "reflecting":
            raise ValueError("Surface condensation requires a reflecting baffled -z face")
        if self._target_surface_solver is not None:
            return
        centers = self.transducers.centers-self.origin
        self._target_source_masses = source_mass_matrix(
            self.cfg.domain.grid_size[:2],
            self.cfg.domain.dx,
            centers,
            self.cfg.specs.diameter,
            self.cfg.specs.source_radial_order,
            self.cfg.specs.source_angular_order,
        )
        self._target_surface_solver = HomogeneousSurfaceSolver(
            tuple(self.cfg.domain.grid_size),
            self.cfg.domain.dx,
            self.cfg.frequency,
            self.cfg.physics.sound_speed,
            self.cfg.physics.medium_density,
            self.cfg.boundary_conditions,
            pml=self.cfg.solver.get("pml", {}),
            device=str(self.cfg.solver.get("device", "cuda")).lower(),
        )

    def _local_target_points(self, points):
        points = np.asarray(points, dtype=float)
        local = points-self.origin
        extent = np.asarray(self.cfg.domain.box_size, dtype=float)
        if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
                or np.any(local < -1e-12) or np.any(local > extent+1e-12)):
            raise ValueError("Targets must be finite physical coordinates inside the PDE domain")
        pml = self.cfg.solver.get("pml", {})
        cells = int(pml.get("cells", 0))
        if cells:
            faces = pml.get("faces", [
                face for face in FACES
                if boundary_type(self.cfg.boundary_conditions[face]) == "open"
            ])
            thickness = cells*self.cfg.domain.dx
            for face in faces:
                axis = "xyz".index(face[1])
                distance = local[:, axis] if face[0] == "-" else extent[axis]-local[:, axis]
                if np.any(distance < thickness-1e-12):
                    raise ValueError("Target lies in PML; enlarge the physical interior")
        return local

    def _target_response_cache_fingerprint(self):
        digest = hashlib.sha256()
        for path in (
            Path(__file__),
            Path(__file__).with_name("homogeneous_surface.py"),
            Path(__file__).parents[1]/"physics"/"sinc_aperture.py",
            Path(__file__).parents[1]/"physics"/"conservative_helmholtz.py",
        ):
            digest.update(path.read_bytes())
        payload = {
            "model": self.MODEL_VERSION,
            "scene": self.scene_fingerprint(),
            "origin_m": self.origin.tolist(),
            "implementation_sha256": digest.hexdigest(),
        }
        return hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()

    def _read_target_response_cache(self):
        filename = str(self.cfg.solver.get("target_response_cache", "")).strip()
        path = Path(filename) if filename else None
        if path is None or not path.is_file():
            return path, None
        try:
            with np.load(path, allow_pickle=False) as data:
                if str(data["fingerprint"]) != self._target_response_cache_fingerprint():
                    return path, None
                points = np.asarray(data["points_m"], dtype=float)
                response = np.asarray(data["response"], dtype=complex)
        except (KeyError, OSError, ValueError):
            return path, None
        if (points.ndim != 2 or points.shape[1] != 3
                or response.shape != (len(points), self.transducers.num_transducers)
                or not np.isfinite(points).all() or not np.isfinite(response).all()):
            return path, None
        return path, (points, response)

    def _write_target_response_cache(self, path, points, response):
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            fingerprint=self._target_response_cache_fingerprint(),
            points_m=points,
            response=response,
        )
        temporary.replace(path)

    def target_response(self, points):
        """Return target x channel responses of the corrected discrete PDE."""
        points = np.asarray(points, dtype=float)
        local = self._local_target_points(points)
        path, cached = self._read_target_response_cache()
        channels = self.transducers.num_transducers
        result = np.empty((len(points), channels), complex)
        missing = []
        if cached is None:
            cached_points = np.empty((0, 3), float)
            cached_response = np.empty((0, channels), complex)
        else:
            cached_points, cached_response = cached
        for index, point in enumerate(points):
            matches = np.flatnonzero(np.all(np.isclose(
                cached_points, point, rtol=0, atol=1e-12
            ), axis=1))
            if matches.size:
                result[index] = cached_response[matches[0]]
            else:
                missing.append(index)
        if missing:
            self._ensure_target_surface_solver()
            computed = self._target_surface_solver.flux_target_response(
                self._target_source_masses,
                local[missing],
            )
            result[missing] = computed
            self._write_target_response_cache(
                path,
                np.vstack((cached_points, points[missing])),
                np.vstack((cached_response, computed)),
            )
        return result

    def field_at(self, points, phases=None):
        values = self.transducers.phases if phases is None else np.asarray(phases, dtype=float)
        if values.shape != (self.transducers.num_transducers,) or not np.isfinite(values).all():
            raise ValueError("phases must match the transducer array and be finite")
        return self.target_response(points) @ np.exp(1j*values)

    def field_at_direct(self, points, phases=None):
        values = self.transducers.phases if phases is None else np.asarray(phases, dtype=float)
        if values.shape != (self.transducers.num_transducers,) or not np.isfinite(values).all():
            raise ValueError("phases must match the transducer array and be finite")
        self._ensure_target_surface_solver()
        self._target_surface_solver.solve_flux(self._target_source_masses, values)
        return self._target_surface_solver.sample(self._local_target_points(points))

    def close(self):
        self._target_surface_solver = None
        self._target_source_masses = None
        self.invalidate_factorization()

    @property
    def obstacle_sdf(self) -> np.ndarray | None:
        """(nx, ny, nz) SDF of the active mesh obstacle, or None."""
        return self._obs_sdf

    def set_material_fields(self, density, sound_speed):
        """Set spatial material arrays (nx,ny,nz); copied to isolate caller edits."""
        shape = tuple(self.cfg.domain.grid_size)
        arrays = tuple(np.array(a, dtype=np.float64, copy=True) for a in (density, sound_speed))
        if any(a.shape != shape or not np.isfinite(a).all() or np.any(a <= 0) for a in arrays):
            raise ValueError("Material fields must match the grid and be finite and positive")
        if self.cfg.obstacles or self.cfg.physics.material_field_file:
            raise ValueError("Choose one material input: arrays, NPZ file, or obstacle")
        self._material_fields = arrays
        self.invalidate_factorization()

    def scene_fingerprint(self):
        """Hash matrix-defining inputs, including current external file contents."""
        if asdict(self.cfg.domain) != self._grid_definition or asdict(self.cfg.specs) != self._source_definition:
            raise ValueError("Grid or source specification changed; construct a new solver")
        files = [o["file"] for o in self.cfg.obstacles if o.get("type", "").lower() == "mesh"]
        if self.cfg.physics.material_field_file:
            files.append(self.cfg.physics.material_field_file)
        digest = hashlib.sha256(json.dumps({
            "model": self.MODEL_VERSION, "physics": asdict(self.cfg.physics),
            "domain": asdict(self.cfg.domain), "specs": asdict(self.cfg.specs),
            "origin_m": self.origin.tolist(),
            "centers": (self.transducers.centers-self.origin).tolist(),
            "boundaries": self.cfg.boundary_conditions, "obstacles": self.cfg.obstacles,
            "pml": self.cfg.solver.get("pml", {}),
        }, sort_keys=True).encode())
        for filename in files:
            with Path(filename).open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024*1024), b""):
                    digest.update(chunk)
        if self._material_fields is not None:
            for array in self._material_fields:
                digest.update(array.tobytes())
        return digest.hexdigest()

    def assemble_matrix(self, force: bool = False) -> np.ndarray:
        key = self.scene_fingerprint()
        if self.A_cached_csr_cpu is not None and not force and key == self._scene_key:
            return self.update_rhs()
        if force or key != self._scene_key:
            self.invalidate_factorization()
            self._obs_sdf = None

        import warnings
        diagnostics = self.source_diagnostics()
        if diagnostics["points_per_wavelength"] < 6:
            raise ValueError(
                "PDE grid has fewer than 6 points per background wavelength; "
                "refine grid_size before solving"
            )
        if diagnostics["points_per_wavelength"] < 10:
            warnings.warn(
                f"PDE resolution: {diagnostics['points_per_wavelength']:.2f} points/wavelength; "
                "small algebraic residual does not establish phase accuracy.",
                RuntimeWarning,
            )
        if diagnostics["clipped_emitters"]:
            warnings.warn(
                f"Source discs clipped by the domain: {diagnostics['clipped_emitters']}",
                RuntimeWarning,
            )
        if len(self.cfg.obstacles) > 1:
            raise NotImplementedError(
                "The solver currently supports one obstacle at a time"
            )
        obstacle = dict(self.cfg.obstacles[0]) if self.cfg.obstacles else {}
        if obstacle.get("type", "").lower() == "sphere":
            obstacle["center"] = (
                np.asarray(obstacle.get("center", [.04, .04, .02]), dtype=float)
                - self.origin
            ).tolist()
        obs_sdf = None
        if obstacle.get("type", "").lower() == "mesh":
            if self._obs_sdf is None:
                from algorithm.physics.mesh_to_sdf import mesh_to_sdf

                self._obs_sdf = mesh_to_sdf(
                    mesh_path=obstacle["file"],
                    grid_size=(
                        self.cfg.domain.nx,
                        self.cfg.domain.ny,
                        self.cfg.domain.nz,
                    ),
                    box_size=(
                        self.cfg.domain.lx,
                        self.cfg.domain.ly,
                        self.cfg.domain.lz,
                    ),
                    center_m=obstacle["center_m"],
                    rotation_deg=obstacle.get(
                        "rotation_deg", (0.0, 0.0, 0.0)
                    ),
                    origin=tuple(self.origin),
                    scale=float(obstacle.get("scale", 1.0)),
                )
            obs_sdf = self._obs_sdf

        materials = self._material_fields
        if self.cfg.physics.material_field_file:
            if materials is not None or obstacle:
                raise ValueError("Material NPZ is mutually exclusive with arrays or obstacles")
            with np.load(self.cfg.physics.material_field_file, allow_pickle=False) as data:
                materials = (data["density"], data["sound_speed"])
        matrix, _ = self.warp_engine.assemble_system_gpu(
            dx=self.cfg.domain.dx,
            omega=self.cfg.omega,
            k0_bg=self.cfg.k0,
            rho0=self.cfg.physics.medium_density,
            c0=self.cfg.physics.sound_speed,
            obs_cfg=obstacle,
            trans_phases_np=self.transducers.phases,
            trans_amps_np=self.transducers.amplitudes,
            array_n=self.cfg.specs.array_n,
            trans_radius=self.cfg.specs.diameter * 0.5,
            trans_pitch=self.cfg.specs.pitch,
            bcs_dict=self.cfg.boundary_conditions,
            obs_sdf=obs_sdf,
            trans_centers_np=self.transducers.centers-self.origin,
            material_fields=materials,
            pml=self.cfg.solver.get("pml"),
            source_radial_order=self.cfg.specs.source_radial_order,
            source_angular_order=self.cfg.specs.source_angular_order,
        )
        local_c = self.warp_engine.operator.c.min()
        if local_c/self.cfg.frequency/self.cfg.domain.dx < 6:
            raise ValueError(
                "PDE grid has fewer than 6 points per shortest local wavelength; "
                "refine grid_size before solving (10+ points and convergence checks recommended)"
            )
        self.A_cached_csr_cpu = matrix.tocsr()
        self._scene_key = key
        return self.update_rhs()

    def source_diagnostics(self):
        centers = (self.transducers.centers-self.origin)[:, :2]
        radius = self.cfg.specs.diameter * 0.5
        clipped = np.any(
            (centers - radius < 0)
            | (centers + radius > np.asarray([self.cfg.lx, self.cfg.ly])), axis=1
        )
        return {
            "points_per_wavelength": self.cfg.wavelength / self.cfg.domain.dx,
            "kdx": self.cfg.k0 * self.cfg.domain.dx,
            "clipped_emitters": np.flatnonzero(clipped).tolist(),
        }

    def update_rhs(self) -> np.ndarray:
        self.rhs_cached = self.warp_engine.assemble_rhs(
            trans_phases_np=self.transducers.phases,
            trans_amps_np=self.transducers.amplitudes,
        )
        return self.rhs_cached

    def set_phases(self, phases: np.ndarray):
        self.transducers.set_phases(phases)

    def invalidate_factorization(self):
        if self.condensed_solver is not None:
            self.condensed_solver.close()
        self.condensed_solver = None
        self.A_cached_csr_cpu = None
        self.A_cached_csr_gpu = None
        self.A_cached_H_csr_gpu = None
        self.rhs_cached = None

    def factorize(self):
        self.assemble_matrix()
        if self._requested_backend() == "cudss_hybrid_direct":
            self._ensure_condensed_solver()

    def build_phase_response_basis(self):
        from algorithm.training.phase_response_basis import PhaseResponseBasis

        training = self.cfg.training
        return PhaseResponseBasis(
            solver=self,
            storage_path=training.phase_basis_file,
            rhs_batch_size=training.phase_basis_batch_size,
            voxel_chunk_size=training.voxel_chunk_size,
            release_factor_after_build=(
                training.release_factor_after_basis
            ),
        ).build()

    def solve(self) -> np.ndarray:
        print("\n" + "=" * 65)
        print(
            "[Smart Solver] 启动智能混合线性求解 "
            f"(DOFs: {self.total_dofs:,})"
        )
        print("=" * 65)

        start = time.perf_counter()
        reuse_matrix = bool(self.cfg.solver.get("reuse_matrix", True))
        rhs = self.assemble_matrix(force=not reuse_matrix)
        stage = "物理场矩阵检查与 RHS 更新"
        assembly_time = time.perf_counter() - start
        print(
            f"[OK] 阶段 1/2: {stage}完成 | "
            f"NNZ: {self.A_cached_csr_cpu.nnz:,} | "
            f"耗时: {assembly_time * 1000.0:.2f} ms"
        )

        start = time.perf_counter()
        solution = self.solve_rhs(rhs)
        print(
            "[OK] 阶段 2/2: 线性系统求解完成! "
            f"耗时: {time.perf_counter() - start:.2f} 秒"
        )
        print("=" * 65)
        return self._reshape_solution(solution)

    def solve_rhs(
        self,
        rhs: np.ndarray,
        verbose: bool = True,
    ) -> np.ndarray:
        if self.A_cached_csr_cpu is None:
            raise RuntimeError("矩阵尚未装配，请先调用 assemble_matrix()")

        self.assemble_matrix()
        rhs_matrix, squeeze = self._as_rhs_matrix(rhs)
        backend = self._requested_backend()

        if backend == "cudss_hybrid_direct":
            if verbose:
                print(
                    "[RUN] 阶段 2/2: "
                    "调用凝聚复对称 cuDSS 直接求解..."
                )
            self._ensure_condensed_solver()
            solution = self.condensed_solver.solve(rhs_matrix)
            self._validate_direct_residual(
                self.condensed_solver.last_relative_residual
            )
            if verbose:
                stats = self.condensed_solver.stats()
                print(
                    "  --> cuDSS: "
                    f"mode={stats['execution_mode']} | "
                    f"plan={stats['plan_seconds']:.3f}s | "
                    f"factor={stats['factor_seconds']:.3f}s | "
                    f"residual={stats['last_relative_residual']:.3e}"
                )

        elif backend == "gpu_direct":
            if verbose:
                print(
                    "[RUN] 阶段 2/2: "
                    "调用 NVIDIA cuDSS GPU 直接求解..."
                )
            self._ensure_gpu_matrix()
            gpu_rhs = (
                cp.asarray(rhs_matrix[:, 0])
                if squeeze
                else cp.asfortranarray(cp.asarray(rhs_matrix))
            )
            solution = cp.asnumpy(
                self._solve_gpu_direct(
                    self.A_cached_csr_gpu,
                    gpu_rhs,
                )
            )

        else:
            solution = self._solve_iterative(
                rhs_matrix,
                verbose=verbose,
            )

        solution = np.asarray(solution)
        if squeeze and solution.ndim == 2:
            return solution[:, 0]
        return solution

    def solve_adjoint(self, adjoint_rhs: np.ndarray) -> np.ndarray:
        self.assemble_matrix()
        rhs_matrix, squeeze = self._as_adjoint_rhs_matrix(adjoint_rhs)
        backend = self._requested_backend()

        if backend == "cudss_hybrid_direct":
            self._ensure_condensed_solver()
            solution = self.condensed_solver.solve_adjoint(rhs_matrix)
            self._validate_direct_residual(
                self.condensed_solver.last_adjoint_relative_residual
            )
        else:
            solution = self._solve_gpu_adjoint(rhs_matrix, backend)

        solution = np.asarray(solution)
        if squeeze:
            solution = solution[:, 0]
        return self._reshape_solution(solution)

    def material_vjp(self, field, field_cotangent):
        """Return voxel density/speed gradients using one A^H adjoint solve.

        The field must be the solution of the current scene. Prescribed source
        amplitudes, geometry, frequency, impedance and PML stretch stay fixed.
        """
        adjoint = self.solve_adjoint(field_cotangent)
        return self.warp_engine.operator.material_vjp(field, adjoint)

    def _ensure_condensed_solver(self):
        if self.condensed_solver is not None:
            return
        fixed_dofs = self.warp_engine.fixed_dofs()
        self.condensed_solver = BoundaryCondensedDirectSolver(
            matrix=self.A_cached_csr_cpu,
            dims=(
                self.cfg.domain.nx,
                self.cfg.domain.ny,
                self.cfg.domain.nz,
            ),
            boundary_conditions=self.cfg.boundary_conditions,
            fixed_dofs=fixed_dofs,
            num_threads=int(
                self.cfg.solver.get("hybrid_num_threads", 1)
            ),
            multithreading_lib=self.cfg.solver.get(
                "hybrid_multithreading_lib"
            ),
            execution_mode=str(
                self.cfg.solver.get("hybrid_execution_mode", "auto")
            ),
            device_memory_limit_gb=float(
                self.cfg.solver.get(
                    "hybrid_device_memory_limit_gb",
                    5.6,
                )
            ),
            device_memory_reserve_gb=float(
                self.cfg.solver.get(
                    "hybrid_device_memory_reserve_gb",
                    0.75,
                )
            ),
            register_cuda_memory=bool(
                self.cfg.solver.get(
                    "hybrid_register_cuda_memory",
                    False,
                )
            ),
            symmetry_tolerance=float(
                self.cfg.solver.get(
                    "condensed_symmetry_tolerance",
                    1.0e-12,
                )
            ),
        )

    def _solve_iterative(
        self,
        rhs_matrix: np.ndarray,
        verbose: bool = True,
    ) -> np.ndarray:
        if rhs_matrix.shape[1] != 1:
            raise ValueError("gpu_iterative supports one RHS at a time")
        self._ensure_gpu_matrix()
        rhs_gpu = cp.asarray(rhs_matrix[:, 0])
        if verbose:
            print("[RUN] 阶段 2/2: 调用 GPU Jacobi-GMRES...")
        solution = self.iterative_solver.solve(
            self.A_cached_csr_gpu,
            rhs_gpu,
        )
        return cp.asnumpy(solution).reshape((-1, 1))

    def _solve_gpu_adjoint(
        self,
        rhs_matrix: np.ndarray,
        backend: str,
    ) -> np.ndarray:
        if rhs_matrix.shape[1] != 1:
            raise ValueError(f"{backend} supports one adjoint RHS")
        if self.A_cached_H_csr_gpu is None:
            matrix_h = (
                self.A_cached_csr_cpu
                .conjugate()
                .transpose()
                .tocsr()
            )
            self.A_cached_H_csr_gpu = csp.csr_matrix(matrix_h)
        rhs_gpu = cp.asarray(rhs_matrix[:, 0])
        if backend == "gpu_direct":
            result = self._solve_gpu_direct(
                self.A_cached_H_csr_gpu,
                rhs_gpu,
            )
        else:
            result = self.iterative_solver.solve(
                self.A_cached_H_csr_gpu,
                rhs_gpu,
            )
        return cp.asnumpy(result).reshape((-1, 1))

    def _requested_backend(self) -> str:
        backend = str(
            self.cfg.solver.get("backend", "gpu_iterative")
        ).lower()
        valid = {
            "gpu_iterative",
            "cudss_hybrid_direct",
            "gpu_direct",
        }
        if backend not in valid:
            raise ValueError(
                f"未知 solver.backend={backend!r}，可选值为 "
                f"{sorted(valid)}"
            )
        return backend

    def _ensure_gpu_matrix(self):
        if self.A_cached_csr_gpu is None:
            self.A_cached_csr_gpu = csp.csr_matrix(
                self.A_cached_csr_cpu
            )

    def _validate_direct_residual(self, residual: float):
        tolerance = float(
            self.cfg.solver.get("direct_residual_tol", 1.0e-4)
        )
        if not np.isfinite(residual) or residual > tolerance:
            raise RuntimeError(
                "Direct solve failed validation: "
                f"relative residual={residual:.3e}, "
                f"tolerance={tolerance:.3e}"
            )

    def _as_rhs_matrix(
        self,
        rhs: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        array = np.asarray(rhs, dtype=np.complex128)
        if array.ndim == 1 and array.size == self.total_dofs:
            return np.asfortranarray(array.reshape((-1, 1))), True
        if array.ndim == 2 and array.shape[0] == self.total_dofs:
            return np.asfortranarray(array), False
        raise ValueError(
            f"rhs must have shape ({self.total_dofs},) or "
            f"({self.total_dofs}, nrhs), got {array.shape}"
        )

    def _as_adjoint_rhs_matrix(
        self,
        rhs: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        array = np.asarray(rhs, dtype=np.complex128)
        spatial_shape = (
            self.cfg.domain.nx,
            self.cfg.domain.ny,
            self.cfg.domain.nz,
        )
        if array.shape == spatial_shape:
            return np.asfortranarray(
                array.reshape((-1, 1), order="F")
            ), True
        if array.ndim == 4 and array.shape[:3] == spatial_shape:
            return np.asfortranarray(
                array.reshape((self.total_dofs, -1), order="F")
            ), False
        return self._as_rhs_matrix(array)

    def _reshape_solution(self, solution: np.ndarray) -> np.ndarray:
        array = np.asarray(solution)
        spatial_shape = (
            self.cfg.domain.nx,
            self.cfg.domain.ny,
            self.cfg.domain.nz,
        )
        if array.ndim == 1:
            return array.reshape(spatial_shape, order="F")
        return array.reshape(
            spatial_shape + (array.shape[1],),
            order="F",
        )

    @staticmethod
    def _solve_gpu_direct(matrix, rhs):
        with DirectSolver(matrix, rhs) as solver:
            solver.plan()
            solver.factorize()
            solution = solver.solve()
        residual = cp.linalg.norm(
            matrix @ solution - rhs
        ) / cp.linalg.norm(rhs)
        residual = float(residual.get())
        if (
            not bool(cp.isfinite(solution).all().get())
            or residual > 1.0e-4
        ):
            raise RuntimeError(
                "GPU direct solve failed validation: "
                f"relative residual={residual:.3e}"
            )
        return solution

    def to_warp_array(self, np_field: np.ndarray) -> wp.array:
        return wp.from_numpy(
            np_field,
            dtype=wp.complex64,
            device=self.warp_engine.device,
        )
