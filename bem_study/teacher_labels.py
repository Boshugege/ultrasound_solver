"""Teacher labels for the hand surrogate dataset (scripts/hand_dataset): speed and accuracy.

generate_mano_bem_dataset.py labels each hand geometry with BemSceneSolver.target_response
at 64-256 targets, on the native MANO mesh (1552 faces, remesh off), with the 'pilot'
(GMRES rtol 1e-2) or 'high' (rtol 1e-4, finer aperture quadrature) settings. The MANO
files need a licence, so the hand here is the repository's 17 cm hand
(src/examples/bem_hand_16x16.yaml) remeshed into regular triangles with MANO's face count
and placed like the dataset's hands; --mano N takes the dataset's own first N hands instead
(MANO with FreiHAND poses, needs models/mano and data/freihand/training_mano.json), --mesh
any closed surface in scene coordinates.

speed       one geometry: one surface solve per target point (target_response before
            the change, kept as ``aperture_response``) against target_response (one
            factorization and one aperture matrix for all points), for both teacher
            settings, on the native mesh and on the mesh subdivided 4x and 16x
resolution  labels on the native mesh and on the same polyhedron with 4x and 16x the
            elements (linear subdivision: same geometry, smaller elements), compared with
            a reference (16x, tight solve, fine quadrature): relative error of the
            response rows and the focus efficiency of each label's phases in the
            reference, for targets anywhere in the workspace and 5-15 mm from the hand;
            with --uniform N..., also near-uniform remeshes at lambda / N (pyacvd), the
            finest of which is then the reference
Usage: teacher_labels.py speed|resolution [--targets 64] [--geometries 3 | --mano 3 | --mesh file.vtk]
                         [--uniform 3 4 5]
"""

import argparse
import copy
from pathlib import Path
import sys
import time

import cupy as cp
import numpy as np
import pyvista as pv
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from algorithm.bem.mesh import oriented_arrays, remesh  # noqa: E402
from algorithm.bem.realtime import Apertures, aperture_response, aperture_responses  # noqa: E402
from algorithm.bem.scene_solver import BemSceneSolver  # noqa: E402
from algorithm.bem.solver import RigidBEM  # noqa: E402
from algorithm.config import SimulationConfig  # noqa: E402
from algorithm.physics.mesh_to_sdf import _load_closed_surface, _place_mesh  # noqa: E402
from algorithm.physics.transducer_array import TransducerArray  # noqa: E402

CONFIG = ROOT / "src" / "examples" / "bem_hand_16x16.yaml"
OUT = ROOT / "outputs" / "bem_study" / "teacher_labels"
MANO_FACES = 1552  # MANO's 1538 faces plus the wrist cap (docs/HAND_OK_BEM_DEMO.md)
TEACHERS = {  # generate_mano_bem_dataset.py (_teacher_config) and run_ok_hand_bem_demo.py
    "pilot": {"bem_rtol": 1e-2, "bem_source_orders": [4, 16], "bem_near_source_orders": [8, 32]},
    "high": {"bem_rtol": 1e-4, "bem_source_orders": [8, 32], "bem_near_source_orders": [16, 64]},
}
WORKSPACE = (np.array([0.020, 0.020, 0.025]), np.array([0.160, 0.160, 0.145]))  # dataset targets
CLEARANCE = 0.005
WAVELENGTH = 346.0 / 40000.0


def synced():
    cp.cuda.Stream.null.synchronize()
    return time.perf_counter()


def release():
    cp.get_default_memory_pool().free_all_blocks()


def polydata(vertices, faces):
    return pv.PolyData(vertices, np.hstack([np.full((len(faces), 1), 3), faces]).ravel())


def mano_hands(count):
    """The first ``count`` hand surfaces of generate_mano_bem_dataset.py (its default seed and
    10 shapes): MANO with FreiHAND poses, placed and wrist-capped exactly as the dataset."""
    sys.path.insert(0, str(ROOT / "scripts" / "hand_dataset"))
    import generate_mano_bem_dataset as dataset
    from algorithm.hand.mano_model import MANOHandModel

    values = dataset._load_freihand(ROOT / "data" / "freihand" / "training_mano.json")
    options = argparse.Namespace(seed=20261102, num_geometries=count, targets_per_geometry=64, num_shapes=10,
                                 teacher_profile="high", target_clearance_m=0.005)
    manifest = dataset._manifest(options, values, OUT / "mano")
    model = MANOHandModel(ROOT / "models" / "mano", hand="right", device="cpu")
    return [dataset._make_geometry(model, values, manifest, i)["mesh"] for i in range(count)]


def hands(cfg, args, rng):
    """Closed hand surfaces in scene coordinates, one per geometry."""
    if args.mesh:
        return [_load_closed_surface(Path(args.mesh))]
    if args.mano:
        return mano_hands(args.mano)
    obstacle = cfg.obstacles[0]
    source = _load_closed_surface((CONFIG.parent / obstacle["file"]).resolve())
    out = []
    for _ in range(args.geometries):  # generate_mano_bem_dataset.py: centre and tilt ranges
        center = rng.uniform([0.085, 0.075, 0.065], [0.115, 0.125, 0.095])
        rotation = np.asarray(obstacle["rotation_deg"], float) + rng.uniform([-12, -18, -12], [12, 18, 12])
        placed = _place_mesh(source, center, rotation, float(obstacle["scale"]))
        lift = max(0.0, 0.024 - placed.points[:, 2].min())  # dataset: lowest point >= 24 mm
        placed.points = placed.points + np.array([0.0, 0.0, lift])
        # regular triangles with MANO's face count (remesh makes ~2 faces per cluster point)
        out.append(remesh(placed, np.sqrt(2.0 * placed.area / (0.866 * MANO_FACES))))
    return out


def densified(surface, element_size):
    """Same surface with every edge shorter than element_size / 2 (adaptive linear
    subdivision). pyacvd needs evenly spread input points: on the native MANO mesh (edges
    up to 48 mm at the wrist cap) it often returns a surface that is not closed."""
    return surface.subdivide_adaptive(max_edge_len=0.5 * element_size).triangulate().clean()


def refined(surface, levels):
    """Same polyhedron with 4^levels times the elements; oriented (vertices, faces)."""
    return oriented_arrays(surface.subdivide(levels, subfilter="linear") if levels else surface)


def describe(vertices, faces):
    tri = vertices[faces]
    edges = np.linalg.norm(tri - np.roll(tri, 1, axis=1), axis=2)
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum()
    return f"{len(faces)} elements, area {area * 1e4:.0f} cm^2, mean edge {edges.mean() * 1e3:.1f} mm"


def targets(surface, rng, count):
    """Half anywhere in the dataset workspace, half 5-15 mm from the hand; all outside it
    and at least 5 mm from its surface. Returns points and a 'near' mask."""
    vertices, faces = oriented_arrays(surface)
    closed = polydata(vertices, faces).compute_normals(cell_normals=True, point_normals=True)
    bounds = np.asarray(closed.bounds).reshape(3, 2)
    lo, hi = np.maximum(WORKSPACE[0], bounds[:, 0] - 0.015), np.minimum(WORKSPACE[1], bounds[:, 1] + 0.015)
    chosen = {False: [], True: []}
    for near in (False, True):
        while len(chosen[near]) < count // 2:
            candidates = rng.uniform(*((lo, hi) if near else WORKSPACE), size=(4096, 3))
            signed = np.asarray(pv.PolyData(candidates).compute_implicit_distance(closed)["implicit_distance"])
            keep = signed >= CLEARANCE if not near else (signed >= CLEARANCE) & (signed <= 0.015)
            chosen[near] += list(candidates[keep][: count // 2 - len(chosen[near])])
    return np.array(chosen[False] + chosen[True]), np.arange(2 * (count // 2)) >= count // 2


def scene_solver(cfg_raw, vertices, faces, teacher, name):
    """BemSceneSolver for a mesh in scene coordinates, configured like the dataset."""
    OUT.mkdir(parents=True, exist_ok=True)
    mesh_file = OUT / f"{name}.vtk"
    polydata(vertices, faces).save(mesh_file)
    raw = copy.deepcopy(cfg_raw)
    raw["obstacles"] = [{"type": "mesh", "file": str(mesh_file), "remesh": False, "scale": 1.0,
                         "center_m": (0.5 * (vertices.min(0) + vertices.max(0))).tolist(),
                         "rotation_deg": [0.0, 0.0, 0.0]}]
    raw["solver"] = {"backend": "bem", **TEACHERS[teacher]}
    path = OUT / f"{name}.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return BemSceneSolver(SimulationConfig.from_yaml(str(path)))


def per_point(solver, points):
    """target_response before batching: one GMRES solve and one aperture integral per point."""
    return np.array([cp.asnumpy(aperture_response(solver.bem, p, solver._direct, solver._scattered,
                                                  solver.rtol)[0]) for p in points]) * solver.transducers.amplitudes


def speed(args):
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    rng = np.random.default_rng(args.seed)
    surface = hands(SimulationConfig.from_yaml(str(CONFIG)), args, rng)[0]
    points, _ = targets(surface, rng, args.targets)
    warm = scene_solver(raw, *refined(surface, 0), "pilot", "warmup")  # compile the kernels
    warm.target_response(points)  # many points: also initialises the LU solver
    per_point(warm, points[:2])
    del warm
    for levels in (0, 1, 2):
        vertices, faces = refined(surface, levels)
        print(f"hand: {describe(vertices, faces)}; {len(points)} targets", flush=True)
        for teacher in ("pilot", "high"):
            release()
            started = synced()
            solver = scene_solver(raw, vertices, faces, teacher, f"speed_{levels}_{teacher}")
            build = synced() - started
            started = synced()
            rows = solver.target_response(points)
            batched = synced() - started
            count = min(len(points), (len(points), 16, 8)[levels])
            started = synced()
            old = per_point(solver, points[:count])
            single = (synced() - started) / count
            difference = np.abs(rows[:count] - old).max(axis=1) / np.abs(old).max(axis=1)
            print(f"  {teacher:5s} (rtol {solver.rtol:g}): build {build:.2f} s; one solve per point "
                  f"{single * 1e3:.0f} ms/point = {single * len(points):.1f} s"
                  f"{'' if count == len(points) else f' (timed on {count})'}; target_response "
                  f"{batched:.2f} s ({batched / len(points) * 1e3:.1f} ms/point) -> "
                  f"{single * len(points) / batched:.0f}x; largest row difference "
                  f"{difference.max():.1e}", flush=True)
            solver.close()
            del solver


def efficiency(reference, label):
    """Focus amplitude of the phases -arg(label) in the reference, relative to its optimum."""
    return np.abs((reference * np.exp(-1j * np.angle(label))).sum(1)) / np.abs(reference).sum(1)


def resolution(args):
    cfg = SimulationConfig.from_yaml(str(CONFIG))
    centers, diameter, k = TransducerArray(cfg).centers, cfg.specs.diameter, cfg.k0
    rng = np.random.default_rng(args.seed)
    results = {}
    for g, surface in enumerate(hands(cfg, args, rng)):
        points, near = targets(surface, rng, args.targets)
        meshes = {levels: refined(surface, levels) for levels in (0, 1, 2)}
        print(f"geometry {g}: hand {describe(*meshes[0])}; lowest point {surface.points[:, 2].min() * 1e3:.0f} mm; "
              f"{(~near).sum()} workspace + {near.sum()} near targets", flush=True)

        def labels(mesh, direct, scattered, rtol, single=False):
            release()
            started = synced()
            bem = RigidBEM(*mesh, k=k)
            bem.assemble()
            direct, scattered = Apertures(centers, diameter, direct), Apertures(centers, diameter, scattered)
            if single:  # the dataset's per-point GMRES (pilot labels)
                rows = cp.stack([aperture_response(bem, p, direct, scattered, rtol)[0] for p in points])
            else:  # the aperture matrix is built in small pieces: the reference matrix is large
                rows, _ = aperture_responses(bem, points, direct, scattered, rtol, matrix_chunk=2048)
            rows = cp.asnumpy(rows)
            print(f"    {describe(*mesh)}: {synced() - started:.1f} s", flush=True)
            del bem
            return rows

        uniform = {}
        for n in args.uniform:  # near-uniform triangles with edges ~ lambda / n (pyacvd)
            started = time.perf_counter()
            uniform[n] = oriented_arrays(remesh(densified(surface, WAVELENGTH / n), WAVELENGTH / n))
            print(f"  remeshed at lambda/{n:g}: {len(uniform[n][1])} elements, {time.perf_counter() - started:.1f} s",
                  flush=True)
        if uniform:
            finest = max(uniform)
            print(f"  reference: lambda/{finest:g} remesh", flush=True)
            reference = labels(uniform.pop(finest), (32, 128), (8, 32), 1e-5)
        else:
            print("  reference: 16x elements", flush=True)
            reference = labels(meshes[2], (32, 128), (8, 32), 1e-6)
        empty = RigidBEM(np.zeros((0, 3)), np.zeros((0, 3), np.int64), k=k)
        free = cp.asnumpy(aperture_responses(empty, points, Apertures(centers, diameter, (32, 128)),
                                             None, 1e-6)[0])
        variants = {"no hand (direct field only)": free}
        if not args.uniform:
            variants["native, pilot (GMRES 1e-2)"] = labels(meshes[0], (8, 32), (4, 16), 1e-2, single=True)
        variants["native, high"] = labels(meshes[0], (16, 64), (8, 32), 1e-4)
        variants["4x elements, high"] = labels(meshes[1], (16, 64), (8, 32), 1e-4)
        if args.uniform:
            variants["16x elements, high"] = labels(meshes[2], (16, 64), (8, 32), 1e-4)
        for n, mesh in sorted(uniform.items()):
            variants[f"lambda/{n:g} remesh, high"] = labels(mesh, (16, 64), (8, 32), 1e-4)
        for name, rows in variants.items():
            error = np.linalg.norm(rows - reference, axis=1) / np.linalg.norm(reference, axis=1)
            eta = efficiency(reference, rows)
            results.setdefault(name, []).append((error, eta, near))
            print(f"  {name:28s} " + "; ".join(
                f"{label}: error median {np.median(error[m]):.1%} max {error[m].max():.1%}, focus median "
                f"{np.median(eta[m]):.2%} min {eta[m].min():.2%}" for label, m in (("workspace", ~near), ("near", near))),
                flush=True)
    if len(results) and (args.mano or args.geometries) > 1 and not args.mesh:
        print("all geometries:")
        for name, parts in results.items():
            error, eta, near = (np.concatenate(v) for v in zip(*parts))
            print(f"  {name:28s} " + "; ".join(
                f"{label}: error median {np.median(error[m]):.1%} p95 {np.percentile(error[m], 95):.1%}, focus "
                f"median {np.median(eta[m]):.2%} p5 {np.percentile(eta[m], 5):.2%} min {eta[m].min():.2%}"
                for label, m in (("workspace", ~near), ("near", near))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("speed", "resolution"))
    parser.add_argument("--targets", type=int, default=64)
    parser.add_argument("--geometries", type=int, default=3)
    parser.add_argument("--mesh", help="closed surface in scene coordinates instead of the remeshed hand")
    parser.add_argument("--mano", type=int, default=0,
                        help="the first N hands of generate_mano_bem_dataset.py (needs models/mano and "
                             "data/freihand/training_mano.json)")
    parser.add_argument("--uniform", type=float, nargs="*", default=[],
                        help="resolution: also remesh at lambda / N for each N; the finest is the reference")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    {"speed": speed, "resolution": resolution}[args.command](args)


if __name__ == "__main__":
    main()
