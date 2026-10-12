import io

import pytest
from numpy import allclose

import freegsnke


def test_readwrite():
    """Test HDF5 reading/writing independently of the legacy Picard solver."""

    for tokamak in [
        freegsnke.machine.TestTokamak(),
        freegsnke.machine.MAST_sym(),
    ]:

        eq = freegsnke.Equilibrium(
            tokamak=tokamak,
            Rmin=0.1,
            Rmax=2.0,
            Zmin=-1.0,
            Zmax=1.0,
            nx=17,
            ny=17,
            boundary=freegsnke.boundary.freeBoundaryHagenow,
        )
        memory_file = io.BytesIO()

        with freegsnke.OutputFile(memory_file, "w") as f:
            f.write_equilibrium(eq)

        with freegsnke.OutputFile(memory_file, "r") as f:
            read_eq = f.read_equilibrium()

        assert tokamak == read_eq.tokamak
        assert allclose(eq.psi(), read_eq.psi())


def test_original_readwrite_solve_setup_is_unsupported():
    """Record the unsupported standalone solve formerly hidden by CI."""
    eq = freegsnke.Equilibrium(
        tokamak=freegsnke.machine.TestTokamak(),
        Rmin=0.1,
        Rmax=2.0,
        Zmin=-1.0,
        Zmax=1.0,
        nx=17,
        ny=17,
        boundary=freegsnke.boundary.freeBoundaryHagenow,
    )
    profiles = freegsnke.jtor.ConstrainPaxisIp(1e4, 1e6, 2.0)
    constrain = freegsnke.control.constrain(
        xpoints=[(1.1, -0.6), (1.1, 0.8)],
        isoflux=[(1.1, -0.6, 1.1, 0.6)],
    )

    with pytest.raises(ValueError, match="No opoints found"):
        freegsnke.solve(eq, profiles, constrain, maxits=25, atol=1e-3, rtol=1e-1)
