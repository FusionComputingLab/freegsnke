"""Machine-independent native strike-point measurements."""

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from freegsnke import shape_descriptors as sd
from freegsnke.equilibrium_update import Equilibrium


def saddle_equilibrium():
    r, z = np.meshgrid(np.linspace(0, 2.2, 111), np.linspace(-2, 1, 151), indexing="ij")
    wall = np.array([[0, -1.2], [2, -1.6]])
    eq = SimpleNamespace(
        R=r,
        Z=z,
        psi=lambda: (r - 0.8) ** 2 - (z + 0.5) ** 2,
        psi_bndry=0.0,
        xpt=np.array([[0.8, -0.5, 0.0]]),
        tokamak=SimpleNamespace(wall=SimpleNamespace(R=wall[:, 0], Z=wall[:, 1])),
    )
    return eq, wall


def test_native_ss_rs_on_sloped_wall_and_reversed_arc():
    eq, wall = saddle_equilibrium()
    strike = sd.measure_strikepoint(eq)
    np.testing.assert_allclose(strike.point, [1.875, -1.575], atol=0.001)
    assert strike.Ss == pytest.approx(1.875 * np.sqrt(1.04), abs=0.001)
    reverse = sd.measure_strikepoint(eq, wall=wall[::-1])
    assert reverse.Rs == pytest.approx(strike.Rs)
    assert reverse.Ss + strike.Ss == pytest.approx(np.linalg.norm(wall[1] - wall[0]))
    eq.strikepoint = lambda **kw: Equilibrium.strikepoint(eq, **kw)
    assert Equilibrium.Ss(eq) == pytest.approx(strike.Ss)
    assert Equilibrium.Rs(eq) == pytest.approx(strike.Rs)
    np.testing.assert_allclose(
        sd.StrikePointDescriptor(names=("Rs", "Ss"))(eq), [strike.Rs, strike.Ss]
    )


def test_native_coordinates_work_for_upper_leg():
    eq, wall = saddle_equilibrium()
    eq.Z = -eq.Z
    eq.xpt[:, 1] *= -1
    contact = sd.measure_strikepoint(eq, wall=wall * [1, -1])
    np.testing.assert_allclose(contact.point, [1.875, 1.575], atol=0.001)


def test_both_coordinates_trace_once(monkeypatch):
    eq, wall = saddle_equilibrium()
    calls = []

    def find(*args, **kwargs):
        calls.append(1)
        return [1.875, -1.575]

    monkeypatch.setattr(sd, "find_leg_strikepoint", find)
    result = sd.StrikePointDescriptor(wall=wall, xpoint=lambda eq: eq.xpt[0, :2])(eq)
    assert len(calls) == 1
    np.testing.assert_allclose(result, [1.875 * np.sqrt(1.04), 1.875])


def test_no_contact_and_invalid_wall_fail_explicitly():
    eq, wall = saddle_equilibrium()
    with pytest.raises(RuntimeError, match="does not intersect"):
        sd.measure_strikepoint(eq, wall=[[3, -1], [3, -2]])
    with pytest.raises(ValueError, match="distinct"):
        sd.measure_strikepoint(eq, wall=[wall[0], wall[0], wall[1]])
    with pytest.raises(ValueError, match="finite"):
        sd.measure_strikepoint(eq, wall=[[0, np.nan], [1, 0]])
    eq.xpt = np.empty((0, 3))
    with pytest.raises(RuntimeError, match="X-point"):
        sd.measure_strikepoint(eq)


def test_contour_extraction_preserves_existing_figure():
    eq, _ = saddle_equilibrium()
    figure = plt.figure()
    try:
        sd.measure_strikepoint(eq)
        assert plt.fignum_exists(figure.number)
        assert plt.get_fignums() == [figure.number]
    finally:
        plt.close(figure)


def test_full_wall_excludes_core_direction_toward_opposite_divertor():
    eq, _ = saddle_equilibrium()
    eq.opt = np.array([[0.8, 0.1, -1.0]])
    # A full wall also intersects the branches on the core side of this
    # X-point. The measured lower leg must not wrap to an upper contact.
    full_wall = np.array(
        [[0.0, -1.2], [2.0, -1.6], [2.0, 1.0], [0.0, 1.0], [0.0, -1.2]]
    )
    contact = sd.measure_strikepoint(eq, wall=full_wall)
    np.testing.assert_allclose(contact.point, [1.875, -1.575], atol=0.001)
