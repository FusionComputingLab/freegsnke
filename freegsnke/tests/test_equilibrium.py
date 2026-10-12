from types import SimpleNamespace

import numpy as np

from freegsnke import boundary, equilibrium, jtor, picard
from freegsnke.gradshafranov import mu0


def test_inoutseparatrix():

    eq = equilibrium.Equilibrium(
        Rmin=0.1, Rmax=2.0, Zmin=-1.0, Zmax=1.0, nx=65, ny=65
    )

    # Two O-points, one X-point half way between them
    psi = np.exp((-((eq.R - 1.0) ** 2) - eq.Z**2) * 3) + np.exp(
        (-((eq.R - 1.0) ** 2) - (eq.Z + 1) ** 2) * 3
    )

    eq._updatePlasmaPsi(psi)

    Rin, Rout = eq.innerOuterSeparatrix()

    assert Rin >= eq.Rmin and Rout >= eq.Rmin
    assert Rin <= eq.Rmax and Rout <= eq.Rmax


def test_fixed_boundary_psi():
    # This is adapted from example 5

    profiles = jtor.ConstrainPaxisIp(
        1e3,
        1e5,
        1.0,  # Plasma pressure on axis [Pascals]  # Plasma current [Amps]
    )  # fvac = R*Bt

    eq = equilibrium.Equilibrium(
        Rmin=0.1,
        Rmax=2.0,
        Zmin=-1.0,
        Zmax=1.0,
        nx=65,
        ny=65,
        boundary=boundary.fixedBoundary,
    )
    # Nonlinear solve
    picard.solve(eq, profiles)

    psi = eq.psi()
    assert psi[0, 0] == 0.0  # Boundary is fixed
    assert psi[32, 32] != 0.0  # Solution is not all zero

    assert eq.psi_bndry == 0.0
    assert eq.poloidalBeta() > 0.0


def test_poloidal_beta3_has_dimensionless_normalisation():
    """Definition 3 includes 2*mu0 when normalised by Bpol squared."""

    eq = object.__new__(equilibrium.Equilibrium)
    eq.R, eq.Z = np.meshgrid([1.0, 2.0], [-0.5, 0.5], indexing="ij")
    eq.dR = 1.0
    eq.dZ = 1.0
    eq._profiles = SimpleNamespace(limiter_core_mask=np.ones_like(eq.R))

    pressure = 3.0
    boundary_field = 2.0
    separatrix = np.array(
        [[1.0, -0.5], [2.0, -0.5], [2.0, 0.5], [1.0, 0.5], [1.0, -0.5]]
    )
    volume = np.sum(2.0 * np.pi * eq.R * eq.dR * eq.dZ)

    def constant_psi(R, Z):
        return np.zeros_like(R)

    def constant_pressure(psi):
        return np.full_like(psi, pressure)

    def constant_boundary_field(R, Z):
        return np.full_like(R, boundary_field)

    def plasma_volume():
        return volume

    def separatrix_curve():
        return separatrix

    def boundary_length():
        return 4.0

    eq.psiNRZ = constant_psi
    eq.pressure = constant_pressure
    eq.plasmaVolume = plasma_volume
    eq.separatrix = separatrix_curve
    eq.separatrix_length = boundary_length
    eq.Bpol = constant_boundary_field

    expected = 2.0 * mu0 * pressure / boundary_field**2
    assert np.isclose(eq.poloidalBeta3(), expected)
