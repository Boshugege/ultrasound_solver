"""What does an obstacle scene cost with upstream's PDE solvers?

Scene: src/examples/pde_homogeneous_16x16.yaml (SonicSurface array, 240 x 240 x
160 mm domain, 20 mm PML) plus a sphere of radius 20 mm at (80, 80, 60) mm with
the default obstacle material, at dx = 160 mm / (NZ - 1).

* ``direct``: cudss_hybrid_direct (assembly, boundary condensation, factorisation,
  one and eight right-hand sides; a 256-column response basis is ~32 x eight);
* ``gmres``: gpu_iterative (row-scaled Jacobi + GMRES(30)) with upstream defaults.
Usage: upstream_obstacle_cost.py {direct|gmres} NZ    (NZ = 113 gives 6 points/wavelength)
"""

import json
from pathlib import Path
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from algorithm.config import SimulationConfig  # noqa: E402
from algorithm.solvers import HelmholtzDirectSolver  # noqa: E402

WORK = ROOT / "outputs" / "bem_study" / "upstream_obstacle_cost"


def config(nz, backend):
    raw = yaml.safe_load((ROOT / "src/examples/pde_homogeneous_16x16.yaml").read_text(encoding="utf-8"))
    dx = 0.16 / (nz - 1)
    nxy = int(round(0.24 / dx)) + 1
    raw["mode"], raw["same_phase_rad"] = "same_phase", 0.0
    raw["domain"]["grid_size"] = [nxy, nxy, nz]
    raw["solver"]["backend"] = backend
    raw["solver"]["pml"]["cells"] = int(round(0.02 / dx))
    raw["solver"].pop("target_response_cache", None)
    raw["solver"].update(hybrid_device_memory_limit_gb=20.0, hybrid_device_memory_reserve_gb=1.0)
    raw["obstacles"] = [{"type": "sphere", "center": [0.08, 0.08, 0.06], "radius": 0.02,
                         "material": {"density": 1250.0, "sound_speed": 2200.0}}]
    raw["io"]["output_file"] = str(WORK / f"field_{backend}_{nz}.npz")
    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / f"sphere_{backend}_{nz}.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return SimulationConfig.from_yaml(path)


def main():
    mode, nz = sys.argv[1], int(sys.argv[2])
    cfg = config(nz, "cudss_hybrid_direct" if mode == "direct" else "gpu_iterative")
    row = {"mode": mode, "grid": list(cfg.domain.grid_size), "nodes": int(np.prod(cfg.domain.grid_size)),
           "points_per_wavelength": round(cfg.wavelength / cfg.domain.dx, 2)}
    solver = HelmholtzDirectSolver(cfg)
    started = time.perf_counter()
    try:
        solver.assemble_matrix()
        row["assemble_s"] = time.perf_counter() - started
        if mode == "direct":
            started = time.perf_counter()
            solver.factorize()
            row["condense_and_factor_s"] = time.perf_counter() - started
            columns = solver.warp_engine.source_columns
            for batch in (1, 8):
                started = time.perf_counter()
                solver.solve_rhs(np.asfortranarray(columns[:, :batch].toarray(), dtype=np.complex128),
                                 verbose=False)
                row[f"solve_{batch}_rhs_s"] = time.perf_counter() - started
        else:
            started = time.perf_counter()
            solver.solve()
            row["gmres_s"] = time.perf_counter() - started
        row["status"] = "ok"
    except (MemoryError, RuntimeError) as error:
        row["status"] = f"{type(error).__name__}: {error}"
        row["seconds_until_failure"] = time.perf_counter() - started
    finally:
        solver.close()
    print("RESULT " + json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
