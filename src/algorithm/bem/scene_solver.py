"""``solver.backend: bem`` — rigid-obstacle BEM as a field model for the main pipeline.

Same scene description and outputs as ``HelmholtzDirectSolver`` (full-grid response
basis in Fortran voxel order, exact target responses, field unit 1/m), so phase
optimisation, losses, gradients, result files and the visualiser are unchanged.

Physics relative to the PDE:
* source: upstream's sinc Neumann apertures (``physics/sinc_aperture.py``), unchanged;
* ``-z`` reflecting baffle and other reflecting faces: exact image planes (at most
  one per axis); ``open`` faces: true free space, so ``solver.pml`` is not needed
  and ignored; ``soft``/``impedance`` faces and material fields are not supported;
* obstacles: rigid surfaces of exactly the configured geometry (sphere, mesh; any
  number). The ``material`` block is ignored: the default material (rho 1250,
  c 2200) differs from a rigid body by ~0.1 %. A warning is raised below 100x the
  air impedance;
* the grid only samples the field: voxels inside an obstacle are set to zero and
  the baffle layer z = 0 is sampled half a cell above the baffle (the aperture
  integral is singular on the aperture itself).
"""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import time
import warnings

import cupy as cp
import numpy as np

from algorithm.bem.realtime import Apertures, aperture_response
from algorithm.bem.solver import RigidBEM, aperture_sources, icosphere
from algorithm.physics.conservative_helmholtz import FACES, boundary_type
from algorithm.physics.transducer_array import TransducerArray
from algorithm.training.phase_response_basis import PhaseResponseBasis


def planes_from_config(cfg):
    """Image planes (axis, position) for the reflecting faces; -z must be one of them."""
    origin = np.asarray(cfg.domain.origin_m, float)
    sizes = np.asarray(cfg.domain.box_size, float)
    planes = []
    for index, face in enumerate(FACES):
        axis, kind = index // 2, boundary_type(cfg.boundary_conditions.get(face, "reflecting"))
        if kind == "reflecting":
            planes.append((axis, float(origin[axis] + (0.0 if face[0] == "-" else sizes[axis]))))
        elif kind != "open":
            raise NotImplementedError(f"bem backend: {face} is {kind!r}; only open and reflecting "
                                      "faces are supported")
    if (2, 0.0) not in planes:
        raise ValueError("bem backend needs the -z face at z = 0 to be the reflecting baffle")
    if len({axis for axis, _ in planes}) != len(planes):
        raise NotImplementedError("bem backend: opposite reflecting faces form a waveguide "
                                  "(infinite images); use cudss_hybrid_direct for that scene")
    return planes


def obstacle_surface(obstacle, element_size):
    """Closed triangulation (normals into the air) of one configured obstacle."""
    kind = obstacle.get("type", "").lower()
    if kind == "sphere":
        radius = float(obstacle["radius"])
        subdivisions = 0
        while radius * 1.0515 / 2**subdivisions > element_size:  # icosphere edge length
            subdivisions += 1
        return icosphere(radius, subdivisions, center=obstacle["center"])
    if kind == "mesh":
        from algorithm.bem.mesh import oriented_arrays, remesh
        from algorithm.physics.mesh_to_sdf import _load_closed_surface, _place_mesh

        surface = _place_mesh(
            _load_closed_surface(Path(obstacle["file"])),
            np.asarray(obstacle["center_m"], float),
            np.asarray(obstacle.get("rotation_deg", (0.0, 0.0, 0.0)), float),
            float(obstacle.get("scale", 1.0)),
        )
        return oriented_arrays(remesh(surface, element_size))
    raise ValueError(f"bem backend supports sphere and mesh obstacles, got {kind!r}")


class BemSceneSolver:
    """Field model with the interface the pipeline uses from HelmholtzDirectSolver."""

    MODEL_VERSION = "bem_rigid_sinc_neumann_v1"

    def __init__(self, cfg):
        self.cfg = cfg
        options = cfg.solver
        self.field_unit = "normalized_1_per_m"
        self.condensed_solver = None
        self.transducers = TransducerArray(cfg)
        self.origin = np.asarray(cfg.domain.origin_m, float)
        if not np.isclose(self.origin[2], 0.0) or np.any(self.transducers.centers[:, 2] != 0):
            raise ValueError("bem backend: the transducers and the baffle must lie at z = 0")
        if cfg.physics.material_field_file:
            raise NotImplementedError("bem backend: material fields need cudss_hybrid_direct")
        self.k = cfg.k0
        self.total_dofs = int(np.prod(cfg.domain.grid_size))
        self.rtol = float(options.get("bem_rtol", 1e-5))
        self.elements_per_wavelength = float(options.get("bem_elements_per_wavelength", 5.0))
        self.source_orders = tuple(options.get("bem_source_orders", (8, 32)))
        self.near_source_orders = tuple(options.get("bem_near_source_orders", (32, 128)))
        self.planes = planes_from_config(cfg)
        self._options = {"rtol": self.rtol, "elements_per_wavelength": self.elements_per_wavelength,
                         "source_orders": self.source_orders,
                         "near_source_orders": self.near_source_orders}

        vertices, faces, offset = [], [], 0
        for obstacle in cfg.obstacles:
            material = obstacle.get("material", {})
            impedance = (float(material.get("density", 1250.0)) * float(material.get("sound_speed", 2200.0))
                         / (cfg.physics.medium_density * cfg.physics.sound_speed))
            if impedance < 100:
                warnings.warn(f"bem backend treats obstacles as rigid, but this material's impedance is "
                              f"only {impedance:.0f}x the medium's", RuntimeWarning)
            v, f = obstacle_surface(obstacle, cfg.wavelength / self.elements_per_wavelength)
            vertices.append(v)
            faces.append(f + offset)
            offset += len(v)
        self.bem = RigidBEM(np.concatenate(vertices) if vertices else np.zeros((0, 3)),
                            np.concatenate(faces) if faces else np.zeros((0, 3), dtype=np.int64),
                            k=self.k, planes=self.planes)
        self.bem.assemble()
        centers, diameter = self.transducers.centers, cfg.specs.diameter
        self._far = aperture_sources(centers, diameter, *self.source_orders)
        self._near = aperture_sources(centers, diameter, *self.near_source_orders)
        self._direct = Apertures(centers, diameter, self.near_source_orders)
        self._scattered = Apertures(centers, diameter, self.source_orders)
        self._obs_sdf = None
        self._grid = None  # sample points and SDF are built on first use: target responses need neither

    # ---- pipeline interface -------------------------------------------------
    @property
    def obstacle_sdf(self):
        if self._obs_sdf is None and self.cfg.obstacles:
            self._obs_sdf = self._sdf()
        return self._obs_sdf

    def set_phases(self, phases):
        self.transducers.set_phases(phases)

    def solve(self):
        weights = self.transducers.amplitudes * np.exp(1j * self.transducers.phases)
        field = self._fields(weights[:, None], np.arange(len(weights)))[:, 0]
        return field.reshape(self._shape, order="F")

    def target_response(self, points):
        """Exact responses (points x transducers) at the configured amplitudes, by
        reciprocity: one BEM solve per point (none without obstacles)."""
        points = np.atleast_2d(np.asarray(points, float))
        rows = [cp.asnumpy(aperture_response(self.bem, p, self._direct, self._scattered, self.rtol)[0])
                for p in points]
        return np.array(rows) * self.transducers.amplitudes[None, :]

    def field_at(self, points, phases=None):
        phases = self.transducers.phases if phases is None else np.asarray(phases, float)
        return self.target_response(points) @ np.exp(1j * phases)

    def build_phase_response_basis(self):
        training = self.cfg.training
        return BemResponseBasis(
            solver=self,
            storage_path=training.phase_basis_file,
            rhs_batch_size=training.phase_basis_batch_size,
            voxel_chunk_size=training.voxel_chunk_size,
            release_factor_after_build=False,
        ).build()

    def scene_fingerprint(self):
        digest = hashlib.sha256(json.dumps({
            "model": self.MODEL_VERSION, "physics": asdict(self.cfg.physics),
            "domain": asdict(self.cfg.domain), "specs": asdict(self.cfg.specs),
            "centers": self.transducers.centers.tolist(), "boundaries": self.cfg.boundary_conditions,
            "obstacles": self.cfg.obstacles, "bem": self._options,
        }, sort_keys=True).encode())
        for obstacle in self.cfg.obstacles:
            if obstacle.get("type", "").lower() == "mesh":
                digest.update(Path(obstacle["file"]).read_bytes())
        return digest.hexdigest()

    def close(self):
        self.bem = None

    # ---- fields -------------------------------------------------------------
    @property
    def _shape(self):
        return tuple(self.cfg.domain.grid_size)

    def _axes(self):
        return [self.origin[a] + np.arange(n) * self.cfg.domain.dx for a, n in enumerate(self._shape)]

    def _grid_points(self):
        axes = self._axes()
        axes[2] = axes[2].copy()
        axes[2][0] = 0.5 * self.cfg.domain.dx
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        return grid.reshape(-1, 3, order="F")

    def _sdf(self):
        if not self.cfg.obstacles:
            return None
        from algorithm.physics.mesh_to_sdf import mesh_to_sdf

        X, Y, Z = np.meshgrid(*self._axes(), indexing="ij")
        sdf = np.full(self._shape, np.inf)
        for obstacle in self.cfg.obstacles:
            if obstacle.get("type", "").lower() == "sphere":
                center = np.asarray(obstacle["center"], float)
                distance = np.sqrt((X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2)
                sdf = np.minimum(sdf, distance - float(obstacle["radius"]))
            else:
                sdf = np.minimum(sdf, mesh_to_sdf(
                    obstacle["file"], self._shape, self.cfg.domain.box_size, obstacle["center_m"],
                    obstacle.get("rotation_deg", (0.0, 0.0, 0.0)), float(obstacle.get("scale", 1.0)),
                    origin=tuple(self.origin)))
        return sdf.astype(np.float32)

    def _incident(self, points, emitters, weights):
        """Aperture fields at ``points`` for columns of emitter weights (len(emitters) x m);
        points closer to the baffle than one aperture diameter use the fine quadrature."""
        out = cp.zeros((len(points), weights.shape[1]), dtype=cp.complex128)
        near = points[:, 2] < self.cfg.specs.diameter
        for mask, (quad, quad_w, per) in ((~near, self._far), (near, self._near)):
            if not bool(mask.any()):
                continue
            x = points[mask]
            index = np.asarray(emitters)
            q = cp.asarray(quad.reshape(-1, per, 3)[index].reshape(-1, 3))
            w = cp.asarray(quad_w.reshape(-1, per)[index])
            block = cp.zeros((len(x), len(index)), dtype=cp.complex128)
            for sign, offset in zip(*self.bem.images):
                r = cp.linalg.norm(x[:, None, :] - (q * cp.asarray(sign) + cp.asarray(offset))[None], axis=2)
                g = (cp.exp(1j * self.k * r) / (4 * np.pi * r)).reshape(len(x), len(index), per)
                block += (g * w[None]).sum(axis=2)
            out[mask] = block @ cp.asarray(weights)
        return out

    def _surface_rhs(self, emitter, weight):
        quad, quad_w, per = self._far
        q, w = quad[emitter * per:(emitter + 1) * per], quad_w[emitter * per:(emitter + 1) * per] * weight
        return self.bem.incident_on_surface(lambda x: self.bem.sources(x, q, w))

    def _fields(self, weight_columns, emitters, point_chunk=2048, emitter_chunk=16):
        """Full-grid fields for columns of emitter weights (len(emitters) x m): all
        surface solves first, then one pass over the grid (the grid <- surface
        matrix is built once per chunk of points for every column)."""
        bem, weight_columns = self.bem, np.asarray(weight_columns, complex)
        emitters = np.asarray(emitters)
        U = cp.zeros((bem.n, weight_columns.shape[1]), dtype=cp.complex64)
        if bem.n:
            for column in range(weight_columns.shape[1]):
                b = sum(self._surface_rhs(e, w) for e, w in zip(emitters, weight_columns[:, column]) if w != 0)
                U[:, column], info = bem.solve(b, rtol=self.rtol)
                if not info.converged:
                    raise RuntimeError(f"BEM GMRES did not converge ({info.iterations} iterations)")
        if self._grid is None:
            self._grid = self._grid_points()
        fields = np.empty((self.total_dofs, weight_columns.shape[1]), dtype=np.complex128)
        for start in range(0, self.total_dofs, point_chunk):
            points = cp.asarray(self._grid[start:start + point_chunk])
            block = bem.scattered_many(points, U).astype(cp.complex128) if bem.n else \
                cp.zeros((len(points), weight_columns.shape[1]), dtype=cp.complex128)
            for first in range(0, len(emitters), emitter_chunk):
                rows = slice(first, first + emitter_chunk)
                block += self._incident(points, emitters[rows], weight_columns[rows])
            fields[start:start + point_chunk] = cp.asnumpy(block)
        if self.obstacle_sdf is not None:
            fields[self.obstacle_sdf.reshape(-1, order="F") < 0] = 0.0
        return fields


class BemResponseBasis(PhaseResponseBasis):
    """Response basis filled by the BEM; same storage, cache and online interface."""

    def build(self):
        self._open_or_create_storage()
        if self.is_complete:
            return self
        started = time.perf_counter()
        solver = self.solver
        emitters = np.arange(self.num_transducers)
        self.basis[:] = solver._fields(np.diag(solver.transducers.amplitudes).astype(complex), emitters)
        self.basis.flush()
        self.completed_columns = self.num_transducers
        self._write_metadata()
        print(f"[BEM] response basis {self.shape} built in {time.perf_counter() - started:.1f} s "
              f"({solver.bem.n} surface elements)")
        return self

    def target_response(self, points):
        """Exact (not interpolated) responses at physical points."""
        return self.solver.target_response(points)
