from pathlib import Path
import os

os.environ.setdefault("CUPY_CACHE_IN_MEMORY", "1")
import cupy as cp
import numpy as np


class AmplitudeTarget:
    def __init__(self, target: np.ndarray, weight: np.ndarray):
        if target.shape != weight.shape:
            raise ValueError("target and weight must have the same shape")
        self.target = np.asarray(target, dtype=np.float64)
        self.weight = np.asarray(weight, dtype=np.float64)

    @classmethod
    def from_config(cls, cfg):
        training = cfg.training
        if training.target_field_file:
            return cls._from_file(cfg, Path(training.target_field_file))
        return cls._from_targets(cfg)

    @classmethod
    def _from_file(cls, cfg, path: Path):
        if not path.is_absolute():
            path = Path.cwd() / path
        if path.suffix.lower() == ".npy":
            target = np.load(path)
            weight = np.ones_like(target, dtype=np.float64)
        elif path.suffix.lower() == ".npz":
            data = np.load(path)
            target = np.asarray(data["target"], dtype=np.float64)
            if "weight" in data:
                weight = np.asarray(data["weight"], dtype=np.float64)
            else:
                weight = np.ones_like(target, dtype=np.float64)
            data.close()
        else:
            raise ValueError("target_field_file must be .npy or .npz")
        cls._validate_shape(cfg, target)
        return cls(target, weight)

    @classmethod
    def _from_targets(cls, cfg):
        if not cfg.targets:
            raise ValueError(
                "phase_optimization requires targets or "
                "training.target_field_file"
            )
        target, weight = build_gaussian_target_fields(
            cfg=cfg,
            target_points=np.asarray(cfg.targets, dtype=np.float64),
            peak_pressure=float(cfg.training.target_peak_pressure),
            sigma=float(cfg.training.target_sigma),
            background_pressure=float(cfg.training.background_pressure),
            target_weight=float(cfg.training.target_weight),
            background_weight=float(cfg.training.background_weight),
        )
        return cls(target, weight)

    @staticmethod
    def _validate_shape(cfg, target: np.ndarray):
        expected = (
            cfg.domain.nx,
            cfg.domain.ny,
            cfg.domain.nz,
        )
        if target.shape != expected:
            raise ValueError(
                f"target field shape must be {expected}, got {target.shape}"
            )


def build_gaussian_target_fields(
    cfg,
    target_points: np.ndarray,
    peak_pressure: float,
    sigma: float,
    background_pressure: float = 0.0,
    target_weight: float = 1.0,
    background_weight: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build full-resolution target amplitude and weight fields.

    Each requested target point creates one Gaussian lobe. The lobe is
    normalized at the nearest grid sample to keep all target samples at the
    same peak amplitude. Multiple lobes are combined by maximum so nearby
    targets do not raise each other's requested peak pressure.
    """
    points = np.asarray(target_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("target_points must have shape (num_targets, 3)")
    if points.shape[0] == 0:
        raise ValueError("at least one target point is required")
    if sigma <= 0.0:
        raise ValueError("target sigma must be positive")
    if peak_pressure < background_pressure:
        raise ValueError("peak_pressure must be >= background_pressure")
    shape = (
        cfg.domain.nx,
        cfg.domain.ny,
        cfg.domain.nz,
    )
    target = np.full(shape, float(background_pressure), dtype=np.float64)
    weight = np.full(shape, float(background_weight), dtype=np.float64)

    origin = np.asarray(cfg.domain.origin_m, dtype=float)
    xs = np.linspace(origin[0], origin[0]+cfg.domain.lx, cfg.domain.nx)
    ys = np.linspace(origin[1], origin[1]+cfg.domain.ly, cfg.domain.ny)
    zs = np.linspace(origin[2], origin[2]+cfg.domain.lz, cfg.domain.nz)
    x, y, z = np.meshgrid(xs, ys, zs, indexing="ij")
    sigma_sq = sigma * sigma
    for point in points:
        _validate_target_point(cfg, point)
        center = np.asarray(point, dtype=np.float64)
        radius_sq = (
            (x - center[0]) ** 2
            + (y - center[1]) ** 2
            + (z - center[2]) ** 2
        )
        gaussian = np.exp(-0.5 * radius_sq / sigma_sq)
        index = _nearest_grid_index(cfg, center)
        gaussian = gaussian / gaussian[index]
        target = np.maximum(
            target,
            background_pressure
            + (peak_pressure - background_pressure) * gaussian,
        )
        weight = np.maximum(
            weight,
            background_weight
            + (target_weight - background_weight) * gaussian,
        )
    return target, weight


def _nearest_grid_index(cfg, point: np.ndarray) -> tuple[int, int, int]:
    dx = cfg.domain.dx
    return tuple(np.rint((point-np.asarray(cfg.domain.origin_m))/dx).astype(int))


def _validate_target_point(cfg, point: np.ndarray) -> None:
    lower = np.asarray(cfg.domain.origin_m, dtype=np.float64)
    upper = lower+np.array([cfg.domain.lx, cfg.domain.ly, cfg.domain.lz])
    if np.any(point < lower) or np.any(point > upper):
        raise ValueError(
            "target point is outside the domain: "
            f"{point.tolist()} not in [{lower.tolist()}, {upper.tolist()}]"
        )


class AmplitudeFieldLoss:
    def __init__(self, amplitude_target: AmplitudeTarget, eps: float = 1.0e-12):
        self.target = amplitude_target.target
        self.weight = amplitude_target.weight
        self.eps = float(eps)

    def evaluate(self, field: np.ndarray, with_cotangent: bool = True):
        amplitude = np.abs(field)
        residual = amplitude - self.target
        weighted = self.weight * residual
        loss = 0.5 * float(np.vdot(weighted, weighted).real)
        if not with_cotangent:
            return loss, None
        cotangent = (
            self.weight
            * self.weight
            * residual
            * field
            / (amplitude + self.eps)
        )
        return loss, cotangent.astype(np.complex128, copy=False)

    def evaluate_device(self, field, with_cotangent: bool = True):
        target = self._device_array("target", self.target)
        weight = self._device_array("weight", self.weight)
        amplitude = cp.abs(field)
        residual = amplitude - target
        weighted = weight * residual
        loss = 0.5 * cp.real(cp.vdot(weighted, weighted))
        if not with_cotangent:
            return loss, None
        return loss, weight * weight * residual * field / (amplitude + self.eps)

    def _device_array(self, name, value):
        attribute = f"_gpu_{name}"
        cached = getattr(self, attribute, None)
        if cached is None:
            cached = cp.asarray(value)
            setattr(self, attribute, cached)
        return cached

    def stats(self, field: np.ndarray) -> tuple[float, float]:
        amplitude = np.abs(field)
        target_mask = self.weight >= 0.5 * float(np.max(self.weight))
        background_mask = ~target_mask
        target_mean = float(np.mean(amplitude[target_mask]))
        if np.any(background_mask):
            background_mean = float(np.mean(amplitude[background_mask]))
        else:
            background_mean = 0.0
        return target_mean, background_mean


def _focal_term(field, indices, weight, eps, with_cotangent, focal_field, xp):
    """External exact samples have their own VJP, separate from the grid VJP."""
    index_arrays = tuple(np.asarray(indices, dtype=np.intp).T)
    values = field[index_arrays] if focal_field is None else focal_field
    amplitude = xp.abs(values)
    loss = -weight * xp.mean(amplitude)
    cotangent = xp.zeros_like(field, dtype=xp.complex128) if with_cotangent else None
    if with_cotangent and focal_field is None:
        # Repeated target indices must accumulate (also on GPU).
        contribution = -weight / len(indices) * values / (amplitude + eps)
        # CuPy scatter-add supports real types, not complex128.
        xp.add.at(cotangent.real, index_arrays, contribution.real)
        xp.add.at(cotangent.imag, index_arrays, contribution.imag)
    return loss, cotangent


class FocalPressureLoss:
    def __init__(
        self,
        cfg,
        amplitude_target: AmplitudeTarget,
        eps: float = 1.0e-12,
    ):
        self.cfg = cfg
        self.target = amplitude_target.target
        self.weight = amplitude_target.weight
        self.eps = float(eps)
        self.target_indices = [
            _nearest_grid_index(cfg, np.asarray(point, dtype=np.float64))
            for point in cfg.targets
        ]
        self.target_weight = float(cfg.training.target_weight)
        self.background_weight = float(cfg.training.background_weight)

    def evaluate(self, field: np.ndarray, with_cotangent: bool = True, focal_field=None):
        amplitude = np.abs(field)
        focal_loss, cotangent = _focal_term(
            field, self.target_indices, self.target_weight, self.eps,
            with_cotangent, focal_field, np,
        )

        background_loss = 0.0
        if self.background_weight > 0.0:
            background_loss = 0.5 * self.background_weight * float(
                np.mean(amplitude * amplitude)
            )
            if with_cotangent:
                cotangent += self.background_weight / field.size * field
        return focal_loss + background_loss, cotangent

    def evaluate_device(self, field, with_cotangent: bool = True, focal_field=None):
        amplitude = cp.abs(field)
        loss, cotangent = _focal_term(
            field, self.target_indices, self.target_weight, self.eps,
            with_cotangent, focal_field, cp,
        )
        if self.background_weight > 0.0:
            loss += 0.5 * self.background_weight * cp.mean(amplitude * amplitude)
            if with_cotangent:
                cotangent += self.background_weight / field.size * field
        return loss, cotangent

    def stats(self, field: np.ndarray) -> tuple[float, float]:
        amplitude = np.abs(field)
        target_values = [
            amplitude[index]
            for index in self.target_indices
        ]
        target_mean = float(np.mean(target_values))
        background_mean = float(np.mean(amplitude))
        return target_mean, background_mean


class FocalContrastLoss:
    def __init__(
        self,
        cfg,
        amplitude_target: AmplitudeTarget,
        eps: float = 1.0e-12,
    ):
        self.cfg = cfg
        self.target = amplitude_target.target
        self.weight = amplitude_target.weight
        self.eps = float(eps)
        self.target_indices = [
            _nearest_grid_index(cfg, np.asarray(point, dtype=np.float64))
            for point in cfg.targets
        ]
        self.target_weight = float(cfg.training.target_weight)
        self.sidelobe_weight = float(cfg.training.background_weight)
        self.temperature = float(cfg.training.sidelobe_temperature)
        if self.temperature <= 0.0:
            raise ValueError("training.sidelobe_temperature must be positive")
        self.background_mask = _build_background_mask(
            cfg,
            np.asarray(cfg.targets, dtype=np.float64),
            float(cfg.training.target_radius),
            int(cfg.training.source_exclusion_layers),
        )
        if not np.any(self.background_mask):
            raise ValueError("focal_contrast background mask is empty")

    def evaluate(self, field: np.ndarray, with_cotangent: bool = True, focal_field=None):
        amplitude = np.abs(field)
        loss, cotangent = _focal_term(
            field, self.target_indices, self.target_weight, self.eps,
            with_cotangent, focal_field, np,
        )

        background_values = amplitude[self.background_mask]
        maximum = float(np.max(background_values))
        shifted = (background_values - maximum) / self.temperature
        weights = np.exp(shifted)
        probabilities = weights / np.sum(weights)
        smooth_max = maximum + self.temperature * (
            np.log(np.sum(weights)) - np.log(background_values.size)
        )
        loss += self.sidelobe_weight * float(smooth_max)
        if not with_cotangent:
            return loss, None

        background_cotangent = (
            self.sidelobe_weight
            * probabilities
            * field[self.background_mask]
            / (background_values + self.eps)
        )
        cotangent[self.background_mask] += background_cotangent
        return loss, cotangent

    def evaluate_device(self, field, with_cotangent: bool = True, focal_field=None):
        amplitude = cp.abs(field)
        loss, cotangent = _focal_term(
            field, self.target_indices, self.target_weight, self.eps,
            with_cotangent, focal_field, cp,
        )
        background_mask = self._device_array("background_mask", self.background_mask)
        background_values = amplitude[background_mask]
        maximum = cp.max(background_values)
        weights = cp.exp((background_values - maximum) / self.temperature)
        probabilities = weights / cp.sum(weights)
        loss += self.sidelobe_weight * (
            maximum + self.temperature * (
                cp.log(cp.sum(weights)) - np.log(background_values.size)
            )
        )
        if not with_cotangent:
            return loss, None
        cotangent[background_mask] += (
            self.sidelobe_weight
            * probabilities
            * field[background_mask]
            / (background_values + self.eps)
        )
        return loss, cotangent

    def _device_array(self, name, value):
        attribute = f"_gpu_{name}"
        cached = getattr(self, attribute, None)
        if cached is None:
            cached = cp.asarray(value)
            setattr(self, attribute, cached)
        return cached

    def stats(self, field: np.ndarray) -> tuple[float, float]:
        amplitude = np.abs(field)
        target_values = [
            amplitude[index]
            for index in self.target_indices
        ]
        background_max = float(np.max(amplitude[self.background_mask]))
        return float(np.mean(target_values)), background_max


def _build_background_mask(
    cfg,
    target_points: np.ndarray,
    target_radius: float,
    source_exclusion_layers: int,
) -> np.ndarray:
    if target_radius <= 0.0:
        raise ValueError("training.target_radius must be positive")
    shape = (
        cfg.domain.nx,
        cfg.domain.ny,
        cfg.domain.nz,
    )
    xs = np.linspace(0.0, cfg.domain.lx, cfg.domain.nx)
    ys = np.linspace(0.0, cfg.domain.ly, cfg.domain.ny)
    zs = np.linspace(0.0, cfg.domain.lz, cfg.domain.nz)
    x, y, z = np.meshgrid(xs, ys, zs, indexing="ij")
    mask = np.ones(shape, dtype=bool)
    layers = max(int(source_exclusion_layers), 0)
    if layers > 0:
        mask[:, :, :layers] = False
    radius_sq_limit = target_radius * target_radius
    for point in target_points:
        center = np.asarray(point, dtype=np.float64)
        radius_sq = (
            (x - center[0]) ** 2
            + (y - center[1]) ** 2
            + (z - center[2]) ** 2
        )
        mask[radius_sq <= radius_sq_limit] = False
    return mask
