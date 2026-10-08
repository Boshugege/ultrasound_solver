import numpy as np
from algorithm.config import SimulationConfig


class TransducerArray:
    def __init__(self, config: SimulationConfig):
        self.cfg = config
        self.centers = self._compute_transducer_centers()
        self.num_transducers = len(self.centers)

        # The corrected sinc Neumann source has unit normalization (field 1/m).
        self.surface_p0 = 1.0
        self.amplitudes = np.ones(self.num_transducers, dtype=np.float64)
        
        # 初始发射相位
        self.phases = np.zeros(self.num_transducers, dtype=np.float64)
        if self.cfg.mode in {"same_phase", "sdf_inverse"}:
            self.phases.fill(self.cfg.same_phase_rad)

    def _compute_transducer_centers(self) -> np.ndarray:
        """计算 N*N 阵元在底面 (z = 0) 的中心物理坐标 [x, y, 0]"""
        if self.cfg.specs.geometry == "sonicsurface_reference":
            import json
            from pathlib import Path
            data = json.loads(
                Path(__file__).with_name("sonicsurface_reference.json").read_text(encoding="utf-8")
            )
            positions = np.asarray(data["positions_vendor_xyz_m"], dtype=np.float64)
            centers = positions[data["logical_to_device"]][:, [0, 2, 1]].copy()
            centers[:, :2] += self.cfg.specs.reference_anchor_m
            return centers
        n = self.cfg.specs.array_n
        pitch = self.cfg.specs.pitch
        origin = np.asarray(self.cfg.domain.origin_m)
        cx, cy = origin[:2] + np.asarray([self.cfg.lx, self.cfg.ly]) * 0.5
        half_span = (n - 1) * pitch * 0.5

        centers = []
        for i in range(n):
            x = cx - half_span + i * pitch
            for j in range(n):
                y = cy - half_span + j * pitch
                centers.append([x, y, 0.0])
        return np.array(centers, dtype=np.float64)

    def compute_geometric_phases(self) -> np.ndarray:
        """
        多目标几何相位的加权复数场叠加:
        U_i = sum_m (1 / d_im) * exp(-i * k0 * d_im)
        phi_i = arg(U_i)
        """
        print("Computing geometric phases...")
        targets = self.cfg.targets
        if not targets:
            return np.zeros(self.num_transducers, dtype=np.float64)

        if len(targets) == 1:
            focal_pos = np.array(targets[0], dtype=np.float64)
            dists = np.linalg.norm(self.centers - focal_pos, axis=-1)
            cycles = dists / self.cfg.wavelength
            return 2.0 * np.pi * (np.ceil(cycles) - cycles)

        # 多目标时间反转波前叠加
        superposed_field = np.zeros(self.num_transducers, dtype=np.complex128)
        for target_pos in targets:
            pos = np.array(target_pos, dtype=np.float64)
            dists = np.linalg.norm(self.centers - pos, axis=-1)
            superposed_field += (1.0 / dists) * np.exp(-1j * self.cfg.k0 * dists)

        return np.angle(superposed_field)

    def set_phases(self, phases: np.ndarray):
        """供逆向优化器直接更新相位的接口"""
        phases = np.asarray(phases, dtype=np.float64)
        if phases.shape != (self.num_transducers,) or not np.isfinite(phases).all():
            raise ValueError("phases must be a finite vector matching the array")
        self.phases = phases.copy()
