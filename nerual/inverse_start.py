"""Starting pose for the amplitude inversion: a learned guess, with a scan as fallback.

The amplitude-only loss of nerual.amplitude_inverse has a false minimum about every
half wavelength of plate displacement (the fringes between incident and scattered
waves, and the standing wave between plate and baffle repeat), so L-BFGS-B only
converges from within ~2.5-3 mm of the true centre. PoseNet maps the observed
amplitudes to the prism's centre, size and orientation. Trained on simulated data
of a prior (PRIOR), it puts the centre within ~0.3 mm (median) of the truth, and
one refinement from there succeeds. locate() refines from the guess and, if the fit
stays poor (typically a prism outside the prior), scans coarsely instead.
"""

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch import nn

from .amplitude_inverse import Prism

# Prior of the training data (metres, degrees): centre, tilt of the plate normal,
# side and thickness; the in-plane turn is uniform.
PRIOR = {"x": (18e-3, 30e-3), "z": (16e-3, 30e-3), "tilt": 25.0, "side": (14e-3, 22e-3),
         "thickness": (2e-3, 4.5e-3)}


def tilt_rotation(normal):
    """Smallest rotation taking z to ``normal``."""
    axis = np.cross([0.0, 0.0, 1.0], normal)
    sine = np.linalg.norm(axis)
    if sine < 1e-12:
        return np.eye(3)
    return Rotation.from_rotvec(axis / sine * np.arctan2(sine, normal[2])).as_matrix()


def sample_prism(rng, prior=PRIOR):
    tilt, azimuth = np.radians(rng.uniform(0, prior["tilt"])), rng.uniform(0, 2 * np.pi)
    normal = np.array([np.sin(tilt) * np.cos(azimuth), np.sin(tilt) * np.sin(azimuth), np.cos(tilt)])
    rotation = tilt_rotation(normal) @ Rotation.from_rotvec([0, 0, rng.uniform(0, 2 * np.pi / 3)]).as_matrix()
    return Prism((rng.uniform(*prior["x"]), rng.uniform(*prior["x"]), rng.uniform(*prior["z"])),
                 tuple(Rotation.from_matrix(rotation).as_rotvec()), rng.uniform(*prior["side"]),
                 rng.uniform(*prior["thickness"]))


def pose_features(prism):
    """Centre (mm), log side and log thickness (mm), plate normal (x, y; z >= 0) and
    in-plane turn psi as (cos 3 psi, sin 3 psi): the same for all the prism's
    symmetric poses."""
    rotation = prism.rotation.as_matrix()
    normal, x_axis = rotation[:, 2], rotation[:, 0]
    if normal[2] < 0:  # half turn about the local x axis, a symmetry of the prism
        normal = -normal
    x_local = tilt_rotation(normal).T @ x_axis
    psi = np.arctan2(x_local[1], x_local[0])
    return np.r_[np.asarray(prism.center) * 1e3, np.log(prism.side * 1e3), np.log(prism.thickness * 1e3),
                 normal[:2], np.cos(3 * psi), np.sin(3 * psi)]


def prism_from_features(f):
    normal_xy = np.asarray(f[5:7], float)
    if np.linalg.norm(normal_xy) > 0.95:
        normal_xy = normal_xy / np.linalg.norm(normal_xy) * 0.95
    normal = np.r_[normal_xy, np.sqrt(1 - normal_xy @ normal_xy)]
    psi = np.arctan2(f[8], f[7]) / 3
    rotation = tilt_rotation(normal) @ Rotation.from_rotvec([0, 0, psi]).as_matrix()
    return Prism(tuple(np.asarray(f[:3], float) * 1e-3), tuple(Rotation.from_matrix(rotation).as_rotvec()),
                 float(np.exp(f[3])) * 1e-3, float(np.exp(f[4])) * 1e-3)


def network_inputs(amplitude, incident):
    """Amplitudes (..., P, M) relative to the known incident amplitude, minus one."""
    return amplitude / np.abs(incident) - 1


class PoseNet(nn.Module):
    def __init__(self, features, hidden=512):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(features, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 9))

    def forward(self, x):
        return self.layers(x)


def train(inputs, targets, epochs=60, augment_noise=0.03, validation_share=0.1, seed=0, device="cuda"):
    """Fit PoseNet to network inputs (N, P*M) and pose_features targets (N, 9).
    Training adds relative amplitude noise ``augment_noise``. Returns a bundle
    (weights and normalisation) for PoseGuess and the validation centre errors (mm)."""
    torch.manual_seed(seed)
    x = np.asarray(inputs, np.float32).reshape(len(inputs), -1)
    y = np.asarray(targets, np.float32)
    order = np.random.default_rng(seed).permutation(len(x))
    held_out, fitted = order[:int(len(x) * validation_share)], order[int(len(x) * validation_share):]
    bundle = {"x_mean": x[fitted].mean(0), "x_std": x[fitted].std(0) + 1e-6,
              "y_mean": y[fitted].mean(0), "y_std": y[fitted].std(0) + 1e-6, "features": x.shape[1]}
    X = torch.as_tensor(x, device=device)
    Y = torch.as_tensor((y - bundle["y_mean"]) / bundle["y_std"], device=device)
    xm = torch.as_tensor(bundle["x_mean"], device=device)
    xs = torch.as_tensor(bundle["x_std"], device=device)
    net = PoseNet(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    fitted_index = torch.as_tensor(fitted, device=device)
    for _ in range(epochs):
        net.train()
        for batch in fitted_index[torch.randperm(len(fitted_index), device=device)].split(256):
            xb = X[batch]
            xb = (1 + xb) * (1 + augment_noise * torch.randn_like(xb)) - 1  # a (1 + e)
            loss = ((net((xb - xm) / xs) - Y[batch])**2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    net.eval()
    with torch.no_grad():
        predicted = net((X[held_out] - xm) / xs).cpu().numpy() * bundle["y_std"] + bundle["y_mean"]
    bundle["state"] = {key: value.cpu() for key, value in net.state_dict().items()}
    return bundle, np.linalg.norm(predicted[:, :3] - y[held_out, :3], axis=1)


class PoseGuess:
    """Predict a Prism from observed amplitudes (P, M) with a trained bundle;
    ``incident`` (P, M) is the field of the same patterns without the prism."""

    def __init__(self, bundle, incident, device="cuda"):
        self.bundle, self.incident, self.device = bundle, np.asarray(incident), device
        self.net = PoseNet(bundle["features"]).to(device)
        self.net.load_state_dict(bundle["state"])
        self.net.eval()

    def __call__(self, amplitude):
        x = network_inputs(np.asarray(amplitude), self.incident).reshape(1, -1).astype(np.float32)
        if x.shape[1] != self.bundle["features"]:
            raise ValueError("amplitudes do not match the patterns the network was trained for")
        with torch.no_grad():
            f = self.net(torch.as_tensor((x - self.bundle["x_mean"]) / self.bundle["x_std"], device=self.device))
        return prism_from_features(f.cpu().numpy()[0] * self.bundle["y_std"] + self.bundle["y_mean"])


def scan_candidates(model, keep, side=17e-3, thickness=3.5e-3):
    """Lowest-loss prisms on a grid of centres (1.5 mm) and in-plane turns (30 deg),
    flat and of a nominal size; returns (prisms, number of poses scanned)."""
    xs, zs = np.arange(18, 30.01, 1.5) * 1e-3, np.arange(16, 30.01, 1.5) * 1e-3
    found = []
    with torch.no_grad():
        for angle in (0, 30, 60, 90):
            for x in xs:
                for y in xs:
                    for z in zs:
                        prism = Prism((x, y, z), (0.0, 0.0, np.radians(angle)), side, thickness)
                        params = torch.as_tensor(prism.params(), dtype=torch.float64, device=model.device)
                        found.append((float(model.loss(params)), prism))
    found.sort(key=lambda item: item[0])
    return [prism for _, prism in found[:keep]], len(found)


def locate(model, guess, coarse=None, accept_rms=0.025, keep=5, maxiter=60, bounds=None):
    """Invert ``model``'s observations from the learned guess (a Prism). If the
    amplitude RMS (square root of the loss) stays above ``accept_rms`` and a coarse
    PrismInverse is given, scan with it and refine its ``keep`` best poses. Successful
    fits are within the model error plus the measurement noise (~1-2 % without noise),
    false minima around 20-30 %; raise accept_rms with noisy data.
    Returns (InverseResult, "guess" or "scan")."""
    result = model.invert(guess, maxiter, bounds)
    if np.sqrt(result.loss) <= accept_rms or coarse is None:
        return result, "guess"
    candidates, _ = scan_candidates(coarse, keep)
    results = [result] + [model.invert(start, maxiter, bounds) for start in candidates]
    return min(results, key=lambda r: r.loss), "scan"
