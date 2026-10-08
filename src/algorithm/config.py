import yaml
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Union


@dataclass
class PhysicsConfig:
    sound_speed: float = 343.0
    medium_density: float = 1.21
    material_field_file: str = ""


@dataclass
class TransducerSpecsConfig:
    frequency: float = 40000.0     # Hz
    diameter: float = 0.0098       # m
    pitch: float = 0.0105          # m
    array_n: int = 6
    geometry: str = "ideal_grid"
    reference_anchor_m: List[float] = field(default_factory=lambda: [0.08, 0.08])
    source_radial_order: int = 64
    source_angular_order: int = 256

    def __post_init__(self):
        if (not isinstance(self.source_radial_order, int) or self.source_radial_order < 2
                or not isinstance(self.source_angular_order, int) or self.source_angular_order < 8):
            raise ValueError("Invalid source quadrature orders")
        if self.geometry not in {"ideal_grid", "sonicsurface_reference"}:
            raise ValueError("geometry must be ideal_grid or sonicsurface_reference")
        if self.geometry == "sonicsurface_reference" and self.array_n != 16:
            raise ValueError("SonicSurface reference geometry requires array_n=16")
        anchor = np.asarray(self.reference_anchor_m, dtype=float)
        if anchor.shape != (2,) or not np.isfinite(anchor).all():
            raise ValueError("reference_anchor_m must contain two finite coordinates")


@dataclass
class DomainConfig:
    # 物理总边长 [Lx, Ly, Lz] (m)
    box_size: List[float] = field(default_factory=lambda: [0.08, 0.08, 0.08])
    # 离散网格点数 [Nx, Ny, Nz]
    grid_size: List[int] = field(default_factory=lambda: [81, 81, 81])
    # 物理坐标系中的网格原点
    origin_m: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def __post_init__(self):
        origin = np.asarray(self.origin_m, dtype=float)
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError("domain.origin_m must contain three finite coordinates")

    @property
    def nx(self) -> int: return self.grid_size[0]
    @property
    def ny(self) -> int: return self.grid_size[1]
    @property
    def nz(self) -> int: return self.grid_size[2]

    @property
    def lx(self) -> float: return self.box_size[0]
    @property
    def ly(self) -> float: return self.box_size[1]
    @property
    def lz(self) -> float: return self.box_size[2]

    @property
    def dx(self) -> float:
        """ 
        dx = Lx / (Nx - 1)
        """
        calc_dx = self.lx / (self.nx - 1)
        calc_dy = self.ly / (self.ny - 1)
        calc_dz = self.lz / (self.nz - 1)

        # 是否为正方各向同性网格
        if not (np.isclose(calc_dx, calc_dy) and np.isclose(calc_dx, calc_dz)):
            raise ValueError(
                f"当前求解器要求各向同性等步长网格 (dx=dy=dz)，但计算得到: "
                f"dx={calc_dx*1000:.3f}mm, dy={calc_dy*1000:.3f}mm, dz={calc_dz*1000:.3f}mm。"
                f"请检查 box_size 与 grid_size 的比例是否一致！"
            )
        return float(calc_dx)


@dataclass
class IOConfig:
    output_file: str = "ultrasound_field.npz"
    auto_visualize: bool = True
    colormap: str = "plasma"


@dataclass
class TrainingConfig:
    loss_type: str = "field_match"
    initial_phase: str = "geometric"
    compare_geometric: bool = False
    phase_basis_file: str = "phase_response_basis.npy"
    phase_basis_batch_size: int = 8
    voxel_chunk_size: int = 262144
    load_basis_to_gpu: bool = True
    release_factor_after_basis: bool = True
    target_field_file: str = ""
    target_peak_pressure: float = 600.0
    target_sigma: float = 0.006
    background_pressure: float = 0.0
    target_weight: float = 1.0
    background_weight: float = 0.05
    target_radius: float = 0.006
    source_exclusion_layers: int = 4
    sidelobe_temperature: float = 25.0
    iterations: int = 100
    seed: int = 0
    max_evaluations: int = 10000
    max_seconds: float = 0.0
    show_loss_curve: bool = True
    loss_curve_update_interval: int = 1
    loss_curve_pause_seconds: float = 0.01


@dataclass
class SimulationConfig:
    mode: str
    physics: PhysicsConfig
    specs: TransducerSpecsConfig
    domain: DomainConfig
    boundary_conditions: Dict[str, Union[str, dict]]
    targets: List[List[float]]
    obstacles: List[Dict]
    io: IOConfig
    training: TrainingConfig = field(default_factory=TrainingConfig)
    solver: Dict[str, Union[str, int, float, bool]] = field(default_factory=dict)
    algorithm: str = "adjoint"
    algorithm_options: dict = field(default_factory=dict)
    same_phase_rad: float = 0.0

    def __post_init__(self):
        if self.mode not in {"phase_optimization", "same_phase", "sdf_inverse"}:
            raise ValueError("mode must be phase_optimization, same_phase or sdf_inverse")
        if not np.isfinite(self.same_phase_rad):
            raise ValueError("same_phase_rad must be finite")
        if not isinstance(self.algorithm_options, dict):
            raise ValueError("algorithm_options must be a mapping")

    @property
    def frequency(self) -> float:
        return self.specs.frequency

    @property
    def omega(self) -> float:
        return 2.0 * np.pi * self.specs.frequency

    @property
    def k0(self) -> float:
        return self.omega / self.physics.sound_speed

    @property
    def wavelength(self) -> float:
        return self.physics.sound_speed / self.specs.frequency

    @property
    def lx(self) -> float: return self.domain.lx
    @property
    def ly(self) -> float: return self.domain.ly
    @property
    def lz(self) -> float: return self.domain.lz

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "SimulationConfig":
        yaml_path = Path(yaml_path).expanduser().resolve()
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        training = dict(raw.get("training", {}))
        mode = str(raw.get("mode", "phase_optimization")).lower()
        algorithm = str(raw.get("algorithm", "adjoint")).lower()
        targets_list = [t["point"] for t in raw.get("targets", [])]
        obstacles = []
        for item in raw.get("obstacles", []):
            obstacle = dict(item)
            if obstacle.get("type", "").lower() == "mesh":
                if "file" not in obstacle:
                    raise ValueError("mesh obstacle requires a 'file' path")
                if "center_m" not in obstacle:
                    raise ValueError("mesh obstacle requires 'center_m'")
                mesh_path = Path(obstacle["file"]).expanduser()
                if not mesh_path.is_absolute():
                    mesh_path = yaml_path.parent / mesh_path
                obstacle["file"] = str(mesh_path.resolve())
            obstacles.append(obstacle)

        physics = dict(raw["physics"])
        if physics.get("material_field_file"):
            path = Path(physics["material_field_file"]).expanduser()
            if not path.is_absolute():
                path = yaml_path.parent / path
            physics["material_field_file"] = str(path.resolve())

        return cls(
            mode=mode,
            physics=PhysicsConfig(**physics),
            specs=TransducerSpecsConfig(**raw["transducer_specs"]),
            domain=DomainConfig(**raw["domain"]),
            boundary_conditions=raw["boundary_conditions"],
            targets=targets_list,
            obstacles=obstacles,
            io=IOConfig(**raw.get("io", {})),
            training=TrainingConfig(**training),
            solver=raw.get("solver", {}),
            algorithm=algorithm,
            algorithm_options=raw.get("algorithm_options", {}),
            same_phase_rad=float(raw.get("same_phase_rad", 0.0)),
        )
