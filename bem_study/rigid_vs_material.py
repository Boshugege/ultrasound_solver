"""How close is a rigid sphere to upstream's penetrable obstacle material?

Exact modal series (pressure and normal velocity continuous at r = a) for a
fluid sphere in air vs the rigid sphere, monopole source at 3 radii, upstream
air parameters (c = 346 m/s, rho = 1.21 kg/m^3, 40 kHz). Relative L2 difference
of the total pressure on 400 random points at 1.5 a, plus lit/shadow axis points.
"""

import numpy as np
from scipy.special import eval_legendre, spherical_jn, spherical_yn

C0, RHO0, FREQUENCY = 346.0, 1.21, 40000.0
K = 2 * np.pi * FREQUENCY / C0


def total_field(points, source, radius, rho1=None, c1=None):
    r = np.linalg.norm(points, axis=1)
    r0 = np.linalg.norm(source)
    cos = points @ source / (r * r0)
    ka = K * radius
    total = np.zeros(len(points), dtype=complex)
    for n in range(int(ka + 10 * ka ** (1 / 3) + 20)):
        h = lambda z: spherical_jn(n, z) + 1j * spherical_yn(n, z)  # noqa: E731
        dh = lambda z: spherical_jn(n, z, True) + 1j * spherical_yn(n, z, True)  # noqa: E731
        c = 1j * K / (4 * np.pi) * (2 * n + 1) * h(K * r0)  # incident coefficient for r < r0
        if rho1 is None:
            A = -c * spherical_jn(n, ka, True) / dh(ka)
        else:
            k1a = 2 * np.pi * FREQUENCY / c1 * radius
            M = np.array([[h(ka), -spherical_jn(n, k1a)],
                          [K / RHO0 * dh(ka), -(k1a / radius) / rho1 * spherical_jn(n, k1a, True)]])
            A = np.linalg.solve(M, -c * np.array([spherical_jn(n, ka), K / RHO0 * spherical_jn(n, ka, True)]))[0]
        incident = c * spherical_jn(n, K * r)  # all probe points lie inside r0
        total += (incident + A * h(K * r)) * eval_legendre(n, cos)
    return total


def main():
    radius = 0.02
    source = np.array([0.0, 0.0, 3 * radius])
    probe = np.random.default_rng(0).standard_normal((400, 3))
    probe = 1.5 * radius * probe / np.linalg.norm(probe, axis=1, keepdims=True)
    axis = np.array([[0, 0, 1.5 * radius], [0, 0, -1.5 * radius]])
    rigid, rigid_axis = total_field(probe, source, radius), total_field(axis, source, radius)
    print(f"sphere radius {radius * 100:.0f} cm, k a = {K * radius:.1f}, monopole at 3 a, c0 = {C0} m/s")
    for name, rho1, c1 in [("upstream default obstacle material", 1250.0, 2200.0),
                           ("skin / soft tissue", 1100.0, 1550.0), ("water", 1000.0, 1480.0),
                           ("PMMA (acrylic)", 1190.0, 2700.0),
                           ("impedance 100 x air", RHO0 * 100 * C0 / 1000, 1000.0),
                           ("impedance 10 x air", RHO0 * 10 * C0 / 1000, 1000.0)]:
        field, field_axis = total_field(probe, source, radius, rho1, c1), total_field(axis, source, radius, rho1, c1)
        lit, shadow = np.abs(field_axis / rigid_axis - 1)
        print(f"{name:36s} Z/Z_air = {rho1 * c1 / (RHO0 * C0):6.0f}  relative difference to rigid "
              f"{np.linalg.norm(field - rigid) / np.linalg.norm(rigid):.2e} (lit {lit:.1e}, shadow {shadow:.1e})")


if __name__ == "__main__":
    main()
