"""Usage: conda run -n srt python -m nerual --scene nerual/example.json"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from .geometry import Plate
from .solver import TriangleSolver


def main():
    parser = argparse.ArgumentParser(description="Rigid triangular plate acoustic phase solver")
    parser.add_argument("--scene",type=Path,default=Path(__file__).with_name("example.json"))
    parser.add_argument("--method",choices=["auto","neural","direct"],default="auto")
    parser.add_argument("--device",choices=["cuda","cpu"],default="cuda")
    parser.add_argument("--subdivisions",type=int,default=None)
    parser.add_argument("--output",type=Path,default=Path(__file__).with_name("artifacts")/"solution.npz")
    args = parser.parse_args()
    scene = json.loads(args.scene.read_text(encoding="utf-8"))
    plate = Plate(**scene["plate"])
    solver = TriangleSolver(device=args.device,subdivisions=args.subdivisions,
                            receivers=scene.get("array_centers_m"),diameter=scene.get("diameter_m",.009))
    solver.set_plate(plate)
    # Warm-up is excluded from the second solve, but geometry setup is reported.
    solver.solve(scene["targets_m"],method=args.method)
    result = solver.solve(scene["targets_m"],method=args.method)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez(args.output,response=result.response,phases_rad=result.phases_rad,
             phase_codes_32=result.phase_codes_32,array_centers_m=solver.receivers)
    metrics = {key:value for key,value in asdict(result).items() if not isinstance(value,np.ndarray)}
    metrics.update({"output":str(args.output),"units":"normalized 1/m",
                    "channel_order":"x index outer, y index inner; custom centers preserve supplied order",
                    "targets":"each row is an independent single-focus phase solution"})
    args.output.with_suffix(".json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    print(json.dumps(metrics,indent=2),flush=True)


if __name__ == "__main__":
    main()
