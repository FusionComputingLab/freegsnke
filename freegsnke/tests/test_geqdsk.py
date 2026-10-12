from io import StringIO

import numpy

from freegsnke import _geqdsk, geqdsk, machine


class SyntheticGeqdskEquilibrium:
    """Small seeded diverted equilibrium used for G-EQDSK round-trip tests."""

    def __init__(self, seed=4):
        rng = numpy.random.default_rng(seed)
        self.tokamak = machine.TestTokamak()
        self.Rmin, self.Rmax = 0.4, 1.6
        self.Zmin, self.Zmax = -0.8, 0.8
        self.R_1D = numpy.linspace(self.Rmin, self.Rmax, 33)
        self.Z_1D = numpy.linspace(self.Zmin, self.Zmax, 33)
        self.R, self.Z = numpy.meshgrid(self.R_1D, self.Z_1D, indexing="ij")
        self.R0 = 1.0 + 0.02 * rng.standard_normal()
        self.lam = 1.45 + 0.05 * rng.random()
        self._psi = -((self.R - self.R0) ** 2 + self.Z**2 - self.lam * self.Z**3)
        self._fvac = 1.8 + 0.1 * rng.random()
        self._ip = 1.0e5 + 1.0e4 * rng.random()

    def psi(self):
        return self._psi

    def fvac(self):
        return self._fvac

    def plasmaCurrent(self):
        return self._ip

    def fpol(self, psinorm):
        psinorm = numpy.asarray(psinorm)
        return self._fvac + 0.08 * (1.0 - psinorm) ** 2

    def pressure(self, psinorm):
        psinorm = numpy.asarray(psinorm)
        return 1200.0 * (1.0 - psinorm) ** 2

    def ffprime(self, psinorm):
        psinorm = numpy.asarray(psinorm)
        return self.fpol(psinorm) * (-0.16 * (1.0 - psinorm)) / self.psirange()

    def pprime(self, psinorm):
        psinorm = numpy.asarray(psinorm)
        return -2400.0 * (1.0 - psinorm) / self.psirange()

    def q(self, psinorm):
        psinorm = numpy.asarray(psinorm)
        return 1.0 + 0.2 * psinorm

    def separatrix(self, ntheta=101):
        theta = numpy.linspace(0.0, 2.0 * numpy.pi, ntheta, endpoint=False)
        z_xpoint = 2.0 / (3.0 * self.lam)
        radius = 0.8 * z_xpoint
        return numpy.column_stack(
            [
                self.R0 + radius * numpy.cos(theta),
                radius * numpy.sin(theta),
            ]
        )

    def psirange(self):
        z_xpoint = 2.0 / (3.0 * self.lam)
        psi_xpoint = -(z_xpoint**2 - self.lam * z_xpoint**3)
        return psi_xpoint


def test_writeread():
    """
    Test that data can be written then read back
    """
    nx = 65
    ny = 65

    # Create a dummy dataset
    data = {
        "nx": nx,
        "ny": ny,
        "rdim": 2.0,
        "zdim": 1.5,
        "rcentr": 1.2,
        "bcentr": 2.42,
        "rleft": 0.5,
        "zmid": 0.1,
        "rmagx": 1.1,
        "zmagx": 0.2,
        "simagx": -2.3,
        "sibdry": 0.21,
        "cpasma": 1234521,
        "fpol": numpy.random.rand(nx),
        "pres": numpy.random.rand(nx),
        "qpsi": numpy.random.rand(nx),
        "psi": numpy.random.rand(nx, ny),
    }

    output = StringIO()

    # Write to string
    _geqdsk.write(data, output)

    # Move to the beginning of the buffer
    output.seek(0)

    # Read from string
    data2 = _geqdsk.read(output)

    # Check that data and data2 are the same
    for key in data:
        numpy.testing.assert_allclose(data2[key], data[key])


def test_cocos_flux_derivative_scaling():
    """Test that COCOS flux conversion also rescales psi derivatives."""

    nx = 5
    ny = 5
    data = {
        "nx": nx,
        "ny": ny,
        "rdim": 2.0,
        "zdim": 1.5,
        "rcentr": 1.2,
        "bcentr": 2.42,
        "rleft": 0.5,
        "zmid": 0.1,
        "rmagx": 1.1,
        "zmagx": 0.2,
        "simagx": -2.3,
        "sibdry": 0.21,
        "cpasma": 1234521,
        "fpol": numpy.linspace(1.0, 2.0, nx),
        "pres": numpy.linspace(10.0, 0.0, nx),
        "ffprime": numpy.linspace(0.2, 0.6, nx),
        "pprime": numpy.linspace(-4.0, -1.0, nx),
        "qpsi": numpy.linspace(1.0, 3.0, nx),
        "psi": numpy.arange(nx * ny).reshape(nx, ny),
    }

    output = StringIO()
    _geqdsk.write(data, output)
    output.seek(0)

    converted = _geqdsk.read(output, cocos=11)

    numpy.testing.assert_allclose(converted["psi"], data["psi"] / (2 * numpy.pi))
    numpy.testing.assert_allclose(converted["simagx"], data["simagx"] / (2 * numpy.pi))
    numpy.testing.assert_allclose(converted["sibdry"], data["sibdry"] / (2 * numpy.pi))
    numpy.testing.assert_allclose(converted["pprime"], data["pprime"] * (2 * numpy.pi))
    numpy.testing.assert_allclose(
        converted["ffprime"], data["ffprime"] * (2 * numpy.pi)
    )


def test_equilibrium_geqdsk_write_read():
    """Test that a seeded diverted equilibrium can be saved and loaded."""

    eq = SyntheticGeqdskEquilibrium()
    output = StringIO()

    geqdsk.write(eq, output, label="SYNTH")
    output.seek(0)

    raw_data = _geqdsk.read(output)
    psinorm = numpy.linspace(0.0, 1.0, raw_data["nx"], endpoint=True)
    numpy.testing.assert_allclose(raw_data["pres"], eq.pressure(psinorm))
    numpy.testing.assert_allclose(raw_data["fpol"], eq.fpol(psinorm))
    numpy.testing.assert_allclose(raw_data["pprime"], eq.pprime(psinorm))
    numpy.testing.assert_allclose(raw_data["ffprime"], eq.ffprime(psinorm))

    output.seek(0)
    loaded = geqdsk.read(output, machine.TestTokamak(), rtol=1e2, maxits=1)

    assert numpy.all(numpy.isfinite(loaded.psi()))
    assert numpy.isfinite(loaded.plasmaCurrent())

    profile_points = numpy.linspace(0.0, 1.0, 9, endpoint=True)
    for loaded_profile, reference_profile in [
        (loaded.pressure, eq.pressure),
        (loaded.fpol, eq.fpol),
        (loaded.pprime, eq.pprime),
        (loaded.ffprime, eq.ffprime),
    ]:
        numpy.testing.assert_allclose(
            loaded_profile(profile_points),
            reference_profile(profile_points),
            rtol=1e-7,
            atol=1e-8,
        )
