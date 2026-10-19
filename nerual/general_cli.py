"""CLI for the arbitrary closed-mesh neural solver."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from .geometry import Plate
from .mesh_scene import MeshBody, box_body, plate_body
from .general_solver import GeneralNeuralSolver


def body_from_spec(spec):
    kind = spec["type"]
    if kind == "npz":
        return MeshBody.from_npz(spec["file"])
    if kind == "mesh":
        return MeshBody.from_surface_file(
            spec["file"], spec["interior_points"], spec.get("center"),
            spec.get("rotation_deg", (0,0,0)), spec.get("scale", 1.0))
    if kind == "triangle_plate":
        plate = Plate(spec["side"], spec["thickness"], tuple(spec["center"]),
                      tuple(spec.get("quaternion", (0,0,0,1))))
        return plate_body(plate, int(spec.get("subdivisions", 10)),
                          int(spec.get("thickness_layers", 4)))
    if kind == "box":
        return box_body(spec["size"], spec["center"],
                        tuple(spec.get("quaternion", (0,0,0,1))), spec.get("name","reflector"))
    raise ValueError(f"Unknown body type: {kind}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path,
                        default=Path(__file__).with_name("general_example.json"))
    parser.add_argument("--method", choices=("auto","neural","direct"), default="auto")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("artifacts")/"general_solution.npz")
    args = parser.parse_args()
    scene = json.loads(args.scene.read_text(encoding="utf-8"))
    dynamic = body_from_spec(scene["dynamic_body"])
    reflectors = [body_from_spec(v) for v in scene.get("reflectors", ())]
    solver = GeneralNeuralSolver(diameter=scene.get("diameter_m", .009))
    solver.set_scene(dynamic, reflectors)
    solver.solve(scene["targets_m"], args.method)
    result = solver.solve(scene["targets_m"], args.method)
    args.output.parent.mkdir(exist_ok=True)
    np.savez(args.output, response=result.response, phases_rad=result.phases_rad,
             phase_codes_32=result.phase_codes_32)
    metrics = {k:v for k,v in asdict(result).items() if not isinstance(v,np.ndarray)}
    args.output.with_suffix(".json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    print(json.dumps(metrics,indent=2))


if __name__ == "__main__":
    main()
