"""Generate a resumable MANO-to-BEM complex-response dataset.

Each completed geometry is stored independently so interrupted runs can resume
without rebuilding existing labels.
"""

from __future__ import annotations

import argparse
import atexit
import ctypes
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyvista as pv


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from algorithm.bem.scene_solver import BemSceneSolver  # noqa: E402
from algorithm.config import SimulationConfig  # noqa: E402
from algorithm.hand.mano_model import (  # noqa: E402
    MANOHandModel,
    cap_open_boundaries,
    place_mano_mesh,
    polydata_from_arrays,
    write_vtk,
)
from algorithm.hand.procedural_rig import rotation_matrix_xyz  # noqa: E402
from run_ok_hand_bem_demo import _build_config, _write_yaml  # noqa: E402


SCHEMA_VERSION = 1
TARGET_CATEGORIES = {"uniform": 0, "near_hand": 1, "trajectory": 2}
WORKSPACE_LO = np.array([0.020, 0.020, 0.025], dtype=np.float64)
WORKSPACE_HI = np.array([0.160, 0.160, 0.145], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default=str(ROOT / "data/freihand/training_mano.json"))
    parser.add_argument("--mano-model-dir", default=str(ROOT / "models/mano"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/hand_phase_surrogate_pilot_v1"))
    parser.add_argument("--num-geometries", type=int, default=500)
    parser.add_argument("--targets-per-geometry", type=int, default=64)
    parser.add_argument("--num-shapes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--teacher-profile", choices=["pilot", "high"], default="pilot")
    parser.add_argument("--target-clearance-m", type=float, default=0.005)
    parser.add_argument("--max-new", type=int, default=0,
                        help="Stop after this many new geometries; 0 processes all missing geometries")
    parser.add_argument("--stage", choices=["all", "geometry", "labels"], default="all",
                        help=argparse.SUPPRESS)
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def _atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            process_query_limited_information, False, int(pid)
        )
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _acquire_dataset_lock(output_dir: Path) -> Path:
    lock_path = output_dir / ".generation.lock"
    for _ in range(2):
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump({"pid": os.getpid(), "created_unix_s": time.time()}, stream)
            atexit.register(lambda: lock_path.unlink(missing_ok=True))
            return lock_path
        except FileExistsError:
            try:
                lock_data = json.loads(lock_path.read_text(encoding="utf-8"))
                owner_pid = int(lock_data.get("pid", 0))
            except Exception:
                owner_pid = 0
            if _pid_is_running(owner_pid):
                raise RuntimeError(
                    f"Dataset generation is already running for {output_dir} (PID {owner_pid})"
                )
            lock_path.unlink(missing_ok=True)
    raise RuntimeError(f"Could not acquire dataset lock: {lock_path}")


def _load_freihand(path: Path) -> np.ndarray:
    annotations = json.loads(path.read_text(encoding="utf-8"))
    rows = [sample[0] if len(sample) == 1 and isinstance(sample[0], list) else sample
            for sample in annotations]
    values = np.asarray(rows, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < 61:
        raise ValueError(f"Expected FreiHAND rows with at least 61 values, got {values.shape}")
    if not np.isfinite(values[:, :61]).all():
        raise ValueError("FreiHAND annotations contain NaN or Inf")
    return values[:, :61]


def _shape_splits(num_shapes: int, rng: np.random.Generator) -> list[str]:
    if num_shapes < 3:
        return ["train"] * num_shapes
    order = rng.permutation(num_shapes)
    n_validation = max(1, int(round(0.1 * num_shapes)))
    n_test = max(1, int(round(0.1 * num_shapes)))
    result = ["train"] * num_shapes
    for index in order[:n_validation]:
        result[int(index)] = "validation"
    for index in order[n_validation:n_validation + n_test]:
        result[int(index)] = "test_id"
    return result


def _manifest(args: argparse.Namespace, values: np.ndarray, output_dir: Path) -> dict:
    path = output_dir / "dataset.json"
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "schema_version": SCHEMA_VERSION,
            "seed": args.seed,
            "targets_per_geometry": args.targets_per_geometry,
            "num_shapes": args.num_shapes,
            "teacher_profile": args.teacher_profile,
            "target_clearance_m": args.target_clearance_m,
        }
        mismatches = {
            key: (data.get(key), value)
            for key, value in expected.items()
            if data.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Existing dataset configuration differs: {mismatches}")
        if args.num_geometries > int(data.get("num_geometries", 0)):
            data["num_geometries"] = args.num_geometries
            _atomic_json(path, data)
        return data

    eligible = np.flatnonzero(np.max(np.abs(values[:, 48:58]), axis=1) <= 3.0)
    if len(eligible) < args.num_shapes:
        raise ValueError(f"Only {len(eligible)} bounded shapes are available")
    rng = np.random.default_rng(args.seed)
    shape_indices = rng.choice(eligible, size=args.num_shapes, replace=False).astype(int)
    data = {
        "schema_version": SCHEMA_VERSION,
        "created_unix_s": time.time(),
        "seed": args.seed,
        "num_geometries": args.num_geometries,
        "targets_per_geometry": args.targets_per_geometry,
        "num_shapes": args.num_shapes,
        "shape_source_indices": shape_indices.tolist(),
        "shape_splits": _shape_splits(args.num_shapes, rng),
        "teacher_profile": args.teacher_profile,
        "target_clearance_m": args.target_clearance_m,
        "target_categories": TARGET_CATEGORIES,
        "input_schema": {
            "shape_beta": [10],
            "hand_pose": [45],
            "root_rotation": [3],
            "root_translation_m": [3],
            "target_world_m": ["N", 3],
        },
        "label_schema": {
            "response_real": ["N", 256],
            "response_imag": ["N", 256],
        },
    }
    _atomic_json(path, data)
    return data


def _matrix_to_axis_angle(matrix: np.ndarray) -> np.ndarray:
    cosine = np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0)
    angle = float(np.arccos(cosine))
    if angle < 1.0e-8:
        return np.zeros(3, dtype=np.float64)
    axis = np.array([
        matrix[2, 1] - matrix[1, 2],
        matrix[0, 2] - matrix[2, 0],
        matrix[1, 0] - matrix[0, 1],
    ], dtype=np.float64)
    axis /= 2.0 * np.sin(angle)
    return axis * angle


def _make_geometry(
    model: MANOHandModel,
    values: np.ndarray,
    manifest: dict,
    geometry_index: int,
) -> dict:
    rng = np.random.default_rng(int(manifest["seed"]) + 104729 * geometry_index)
    shape_slot = geometry_index % int(manifest["num_shapes"])
    shape_source_index = int(manifest["shape_source_indices"][shape_slot])
    pose_source_index = int(rng.integers(0, len(values)))
    shape = values[shape_source_index, 48:58].astype(np.float64)
    hand_pose = values[pose_source_index, 3:48].astype(np.float64)
    pose = np.concatenate([np.zeros(3, dtype=np.float64), hand_pose])

    vertices, faces, joints = model.mesh_from_arrays(pose, shape)
    center_m = rng.uniform([0.085, 0.075, 0.065], [0.115, 0.125, 0.095])
    rotation_deg = rng.uniform([-12.0, -18.0, -12.0], [12.0, 18.0, 12.0])
    scene_vertices, scene_joints = place_mano_mesh(
        vertices,
        joints,
        center_m=center_m,
        rotation_deg=rotation_deg,
        scale=1.0,
        min_z=0.024,
    )
    scene_vertices, scene_faces = cap_open_boundaries(scene_vertices, faces)
    mesh = polydata_from_arrays(scene_vertices, scene_faces).triangulate().clean()
    mesh = mesh.compute_normals(
        cell_normals=True,
        point_normals=True,
        consistent_normals=True,
        auto_orient_normals=True,
        inplace=False,
    )
    if mesh.n_open_edges:
        raise ValueError(f"Generated MANO mesh has {mesh.n_open_edges} open edges")
    if not np.isfinite(mesh.points).all() or mesh.n_cells < 1500:
        raise ValueError("Generated MANO mesh is invalid")

    rotation = rotation_matrix_xyz(rotation_deg)
    return {
        "mesh": mesh,
        "shape_beta": shape,
        "hand_pose": hand_pose,
        "root_rotation": _matrix_to_axis_angle(rotation),
        "root_translation_m": np.asarray(scene_joints[0], dtype=np.float64),
        "scene_center_m": center_m,
        "scene_rotation_deg": rotation_deg,
        "shape_slot": shape_slot,
        "shape_source_index": shape_source_index,
        "pose_source_index": pose_source_index,
        "split": manifest["shape_splits"][shape_slot],
        "rng": rng,
    }


def _outside_with_clearance(
    mesh: pv.PolyData,
    candidates: np.ndarray,
    clearance_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    candidates = np.asarray(candidates, dtype=np.float64).reshape(-1, 3)
    cloud = pv.PolyData(candidates)
    selected = cloud.select_interior_points(
        mesh,
        method="signed_distance",
        check_surface=True,
    )
    inside = np.asarray(selected["selected_points"], dtype=bool)
    distance_data = cloud.compute_implicit_distance(mesh)
    distance = np.abs(np.asarray(distance_data["implicit_distance"], dtype=np.float64))
    valid = (~inside) & (distance >= clearance_m)
    return candidates[valid], distance[valid]


def _collect_targets(
    mesh: pv.PolyData,
    rng: np.random.Generator,
    count: int,
    clearance_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    category_counts = {
        "uniform": int(round(0.50 * count)),
        "near_hand": int(round(0.30 * count)),
    }
    category_counts["trajectory"] = count - sum(category_counts.values())
    accepted_points: list[np.ndarray] = []
    accepted_categories: list[np.ndarray] = []

    for category, needed in category_counts.items():
        points: list[np.ndarray] = []
        for _ in range(100):
            remaining = needed - sum(len(batch) for batch in points)
            if remaining <= 0:
                break
            batch_size = max(64, 8 * remaining)
            if category == "uniform":
                candidates = rng.uniform(WORKSPACE_LO, WORKSPACE_HI, size=(batch_size, 3))
            elif category == "near_hand":
                bounds = np.asarray(mesh.bounds, dtype=np.float64).reshape(3, 2)
                lo = np.maximum(WORKSPACE_LO, bounds[:, 0] - 0.040)
                hi = np.minimum(WORKSPACE_HI, bounds[:, 1] + 0.040)
                candidates = rng.uniform(lo, hi, size=(batch_size, 3))
            else:
                angles = rng.uniform(0.0, 2.0 * np.pi, size=batch_size)
                radii = rng.uniform(0.015, 0.060, size=batch_size)
                center_xy = rng.uniform([0.065, 0.065], [0.095, 0.095])
                z0 = float(rng.uniform(0.045, 0.115))
                candidates = np.column_stack([
                    center_xy[0] + radii * np.cos(angles),
                    center_xy[1] + radii * np.sin(angles),
                    z0 + 0.006 * np.sin(2.0 * angles),
                ])
                candidates = np.clip(candidates, WORKSPACE_LO, WORKSPACE_HI)

            valid, distance = _outside_with_clearance(mesh, candidates, clearance_m)
            if category == "near_hand":
                valid = valid[distance <= 0.040]
            if len(valid):
                points.append(valid[:remaining])
        if sum(len(batch) for batch in points) < needed:
            raise RuntimeError(f"Could not sample {needed} valid {category} targets")
        category_points = np.concatenate(points, axis=0)[:needed]
        accepted_points.append(category_points)
        accepted_categories.append(np.full(needed, TARGET_CATEGORIES[category], dtype=np.uint8))

    targets = np.concatenate(accepted_points, axis=0)
    categories = np.concatenate(accepted_categories, axis=0)
    order = rng.permutation(len(targets))
    return targets[order], categories[order]


def _teacher_config(
    mesh_file: Path,
    mesh: pv.PolyData,
    first_target: np.ndarray,
    geometry_dir: Path,
    profile: str,
) -> Path:
    points = np.asarray(mesh.points, dtype=np.float64)
    mesh_center = 0.5 * (points.min(axis=0) + points.max(axis=0))
    raw = _build_config(mesh_file.resolve(), mesh_center, first_target, geometry_dir)
    raw["obstacles"][0]["remesh"] = False
    if profile == "high":
        raw["solver"].update({
            "bem_rtol": 1.0e-4,
            "bem_source_orders": [8, 32],
            "bem_near_source_orders": [16, 64],
        })
    return _write_yaml(raw, geometry_dir / "scene.yaml")


def _release_solver(solver: BemSceneSolver | None) -> None:
    if solver is not None:
        solver.close()
    gc.collect()
    try:
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def _sample_path(output_dir: Path, split: str, geometry_index: int) -> Path:
    return output_dir / "samples" / split / f"geometry_{geometry_index:06d}.npz"


def _geometry_input_path(output_dir: Path, geometry_index: int) -> Path:
    return output_dir / "geometry" / f"{geometry_index:06d}" / "inputs.npz"


def _prepare_geometry_stage(
    args: argparse.Namespace,
    values: np.ndarray,
    manifest: dict,
    output_dir: Path,
) -> int:
    pending = []
    for geometry_index in range(args.num_geometries):
        shape_slot = geometry_index % int(manifest["num_shapes"])
        split = manifest["shape_splits"][shape_slot]
        if _sample_path(output_dir, split, geometry_index).is_file():
            continue
        if _geometry_input_path(output_dir, geometry_index).is_file():
            continue
        pending.append(geometry_index)
        if args.max_new and len(pending) >= args.max_new:
            break
    if not pending:
        return 0

    model = MANOHandModel(args.mano_model_dir, hand="right", device="cpu")
    prepared = 0
    for geometry_index in pending:
        shape_slot = geometry_index % int(manifest["num_shapes"])
        split = manifest["shape_splits"][shape_slot]
        input_path = _geometry_input_path(output_dir, geometry_index)

        geometry_started = time.perf_counter()
        try:
            geometry = _make_geometry(model, values, manifest, geometry_index)
            targets, target_category = _collect_targets(
                geometry["mesh"],
                geometry["rng"],
                args.targets_per_geometry,
                args.target_clearance_m,
            )
            geometry_dir = input_path.parent
            write_vtk(
                geometry_dir / "hand.vtk",
                np.asarray(geometry["mesh"].points),
                np.asarray(geometry["mesh"].faces).reshape(-1, 4)[:, 1:],
            )
            _atomic_npz(
                input_path,
                geometry_id=np.int64(geometry_index),
                split=np.asarray(geometry["split"]),
                shape_slot=np.int32(geometry["shape_slot"]),
                shape_source_index=np.int64(geometry["shape_source_index"]),
                pose_source_index=np.int64(geometry["pose_source_index"]),
                shape_beta=np.asarray(geometry["shape_beta"], dtype=np.float32),
                hand_pose=np.asarray(geometry["hand_pose"], dtype=np.float32),
                root_rotation=np.asarray(geometry["root_rotation"], dtype=np.float32),
                root_translation_m=np.asarray(geometry["root_translation_m"], dtype=np.float32),
                scene_center_m=np.asarray(geometry["scene_center_m"], dtype=np.float32),
                scene_rotation_deg=np.asarray(geometry["scene_rotation_deg"], dtype=np.float32),
                target_world_m=np.asarray(targets, dtype=np.float32),
                target_category=np.asarray(target_category, dtype=np.uint8),
                preparation_seconds=np.float32(time.perf_counter() - geometry_started),
            )
            prepared += 1
            print(json.dumps({
                "stage": "geometry",
                "geometry": geometry_index,
                "split": split,
                "targets": len(targets),
                "seconds": round(time.perf_counter() - geometry_started, 2),
            }))
        except Exception as exc:
            failure = {
                "stage": "geometry",
                "geometry": geometry_index,
                "error": f"{type(exc).__name__}: {exc}",
                "time_unix_s": time.time(),
            }
            with (output_dir / "failures.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(failure) + "\n")
            print(json.dumps(failure), file=sys.stderr)
            if args.stop_on_error:
                raise
    return prepared


def _run_geometry_subprocess() -> None:
    forwarded: list[str] = []
    skip_next = False
    for argument in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if argument == "--stage":
            skip_next = True
            continue
        forwarded.append(argument)
    command = [sys.executable, str(Path(__file__).resolve()), *forwarded, "--stage", "geometry"]
    completed = subprocess.run(command, cwd=str(ROOT), check=False)
    if completed.returncode:
        raise RuntimeError(f"MANO geometry subprocess failed with exit code {completed.returncode}")


def _write_progress(output_dir: Path, manifest: dict, started: float, failures: int) -> dict:
    completed = sum(1 for _ in (output_dir / "samples").glob("**/geometry_*.npz"))
    progress = {
        "completed_geometries": completed,
        "requested_geometries": int(manifest["num_geometries"]),
        "completed_labels": completed * int(manifest["targets_per_geometry"]),
        "failures": failures,
        "elapsed_s": time.perf_counter() - started,
        "updated_unix_s": time.time(),
    }
    _atomic_json(output_dir / "progress.json", progress)
    return progress


def main() -> None:
    args = parse_args()
    if args.num_geometries <= 0 or args.targets_per_geometry <= 0 or args.num_shapes <= 0:
        raise SystemExit("Geometry, target and shape counts must be positive")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage != "geometry":
        _acquire_dataset_lock(output_dir)
    values = _load_freihand(Path(args.annotations))
    manifest = _manifest(args, values, output_dir)
    if args.stage == "geometry":
        prepared = _prepare_geometry_stage(args, values, manifest, output_dir)
        print(json.dumps({"prepared_geometries": prepared}, indent=2))
        return
    if args.stage == "all":
        _run_geometry_subprocess()

    started = time.perf_counter()
    generated = 0
    failures = 0

    for geometry_index in range(args.num_geometries):
        input_path = _geometry_input_path(output_dir, geometry_index)
        if not input_path.is_file():
            continue
        with np.load(input_path, allow_pickle=False) as cached:
            split = str(cached["split"].item())
        sample_path = _sample_path(output_dir, split, geometry_index)
        if sample_path.is_file():
            continue
        if args.max_new and generated >= args.max_new:
            break

        geometry_dir = output_dir / "geometry" / f"{geometry_index:06d}"
        solver = None
        geometry_started = time.perf_counter()
        try:
            with np.load(input_path, allow_pickle=False) as cached:
                inputs = {name: cached[name] for name in cached.files}
            targets = np.asarray(inputs["target_world_m"], dtype=np.float64)
            target_category = np.asarray(inputs["target_category"], dtype=np.uint8)
            mesh_file = geometry_dir / "hand.vtk"
            mesh = pv.read(mesh_file).extract_surface(
                algorithm="dataset_surface"
            ).triangulate().clean()
            cfg_path = _teacher_config(
                mesh_file,
                mesh,
                targets[0],
                geometry_dir,
                args.teacher_profile,
            )
            solver = BemSceneSolver(SimulationConfig.from_yaml(str(cfg_path)))
            label_started = time.perf_counter()
            response = solver.target_response(targets)
            label_s = time.perf_counter() - label_started
            if response.shape != (args.targets_per_geometry, 256):
                raise ValueError(f"Unexpected response shape {response.shape}")
            if not np.isfinite(response.real).all() or not np.isfinite(response.imag).all():
                raise ValueError("BEM response contains NaN or Inf")

            if not (output_dir / "array_metadata.npz").is_file():
                _atomic_npz(
                    output_dir / "array_metadata.npz",
                    centers_m=np.asarray(solver.transducers.centers, dtype=np.float32),
                    amplitudes=np.asarray(solver.transducers.amplitudes, dtype=np.float32),
                )
            _atomic_npz(
                sample_path,
                geometry_id=np.int64(geometry_index),
                split=np.asarray(split),
                shape_slot=np.int32(inputs["shape_slot"]),
                shape_source_index=np.int64(inputs["shape_source_index"]),
                pose_source_index=np.int64(inputs["pose_source_index"]),
                shape_beta=np.asarray(inputs["shape_beta"], dtype=np.float32),
                hand_pose=np.asarray(inputs["hand_pose"], dtype=np.float32),
                root_rotation=np.asarray(inputs["root_rotation"], dtype=np.float32),
                root_translation_m=np.asarray(inputs["root_translation_m"], dtype=np.float32),
                scene_center_m=np.asarray(inputs["scene_center_m"], dtype=np.float32),
                scene_rotation_deg=np.asarray(inputs["scene_rotation_deg"], dtype=np.float32),
                target_world_m=np.asarray(targets, dtype=np.float32),
                target_category=np.asarray(target_category, dtype=np.uint8),
                response_real=np.asarray(response.real, dtype=np.float32),
                response_imag=np.asarray(response.imag, dtype=np.float32),
                bem_elements=np.int32(solver.bem.n),
                preparation_seconds=np.float32(inputs["preparation_seconds"]),
                label_seconds=np.float32(label_s),
                geometry_seconds=np.float32(time.perf_counter() - geometry_started),
            )
            generated += 1
            progress = _write_progress(output_dir, manifest, started, failures)
            print(json.dumps({
                "geometry": geometry_index,
                "split": split,
                "targets": len(targets),
                "bem_elements": int(solver.bem.n),
                "seconds": round(time.perf_counter() - geometry_started, 2),
                "completed": progress["completed_geometries"],
            }))
        except Exception as exc:
            failures += 1
            failure = {
                "geometry": geometry_index,
                "error": f"{type(exc).__name__}: {exc}",
                "time_unix_s": time.time(),
            }
            with (output_dir / "failures.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(failure) + "\n")
            print(json.dumps(failure), file=sys.stderr)
            if args.stop_on_error:
                raise
        finally:
            _release_solver(solver)

    progress = _write_progress(output_dir, manifest, started, failures)
    print(json.dumps(progress, indent=2))


if __name__ == "__main__":
    main()
