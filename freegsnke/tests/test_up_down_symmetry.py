"""Tests for machine-description up-down symmetrisation."""

import numpy as np
import pytest

from freegsnke.up_down_symmetry import prepare_up_down_symmetric_machine


def _active_element(r, z):
    """Build a minimal active-coil data entry."""
    return {
        "R": np.asarray(r, dtype=float),
        "Z": np.asarray(z, dtype=float),
        "dR": 0.02,
        "dZ": 0.03,
        "resistivity": 1.5e-8,
        "polarity": 1,
        "multiplier": 1,
    }


def _passive_element(name, element, r, z):
    """Build a minimal passive polygon data entry."""
    return {
        "name": name,
        "element": element,
        "R": np.asarray(r, dtype=float),
        "Z": np.asarray(z, dtype=float),
        "resistivity": 7.4e-7,
    }


def test_pairing_symmetrisation_and_current_transforms():
    """Geometry assignment handles reordered children and transforms round-trip."""
    active = {
        "solenoid": _active_element([0.2, 0.2, 0.2, 0.2], [-0.8, -0.2, 0.19, 0.79]),
        "p_upper": {"1": _active_element([1.0, 1.1], [0.7, 0.8])},
        "p_lower": {"1": _active_element([1.0, 1.1], [-0.72, -0.82])},
    }
    passive = [
        _passive_element("case_upper_0", "case_upper", [1.0, 1.2, 1.2, 1.0], [0.4] * 4),
        _passive_element("case_upper_1", "case_upper", [1.0, 1.2, 1.2, 1.0], [0.8] * 4),
        _passive_element(
            "case_lower_0", "case_lower", [1.0, 1.2, 1.2, 1.0], [-0.82] * 4
        ),
        _passive_element(
            "case_lower_1", "case_lower", [1.0, 1.2, 1.2, 1.0], [-0.42] * 4
        ),
    ]

    prepared = prepare_up_down_symmetric_machine(
        active, passive, max_pair_mismatch=0.05
    )

    assert prepared.original_active_names == ("solenoid", "p_upper", "p_lower")
    assert prepared.even_active_names == ("solenoid", "p")
    assert prepared.active_odd_names == ("p",)
    assert len(prepared.passive_pairs) == 2
    assert prepared.maximum_geometry_discrepancy is not None
    assert prepared.largest_geometry_discrepancies()[0].reflected_rms >= (
        prepared.largest_geometry_discrepancies()[-1].reflected_rms
    )
    with pytest.raises(ValueError, match="Largest failures"):
        prepared.check_geometry_tolerance(0.0)
    assert {(pair.upper_name, pair.lower_name) for pair in prepared.passive_pairs} == {
        ("case_upper_0", "case_lower_1"),
        ("case_upper_1", "case_lower_0"),
    }

    symmetric_passive = {entry["name"]: entry for entry in prepared.passive_coils_data}
    for pair in prepared.passive_pairs:
        upper = symmetric_passive[pair.upper_name]
        lower = symmetric_passive[pair.lower_name]
        np.testing.assert_allclose(np.sort(upper["R"]), np.sort(lower["R"]))
        np.testing.assert_allclose(np.sort(upper["Z"]), np.sort(-lower["Z"]))

    reflection = prepared.reflection_operator
    np.testing.assert_allclose(reflection @ reflection, np.eye(7))

    original_currents = np.asarray([8.0, 5.0, 3.0])
    even, odd = prepared.split_active_currents(original_currents)
    np.testing.assert_allclose(even, [8.0, 4.0])
    np.testing.assert_allclose(odd, [1.0])
    np.testing.assert_allclose(
        prepared.combine_active_currents(even, odd), original_currents
    )
    np.testing.assert_allclose(prepared.combine_active_currents(even), [8.0, 4.0, 4.0])


def test_operator_and_green_symmetrisation():
    """Operator helpers produce matrices and fields commuting with reflection."""
    active = {
        "self": _active_element([0.2, 0.2], [-0.5, 0.5]),
        "a_upper": _active_element([1.0], [0.5]),
        "a_lower": _active_element([1.0], [-0.5]),
    }
    prepared = prepare_up_down_symmetric_machine(active)
    reflection = prepared.reflection_operator

    operator = np.arange(9, dtype=float).reshape(3, 3)
    symmetric_operator = prepared.symmetrise_square_operator(operator)
    np.testing.assert_allclose(
        reflection @ symmetric_operator @ reflection, symmetric_operator
    )

    greens = np.arange(3 * 2 * 5, dtype=float).reshape(3, 2, 5)
    symmetric_greens = prepared.symmetrise_greens(greens)
    np.testing.assert_allclose(
        np.einsum("ij,j...->i...", reflection, symmetric_greens[..., ::-1]),
        symmetric_greens,
    )


def test_automatic_recentering_and_boundary_symmetrisation():
    """A common source offset is fitted, removed, and recorded."""
    source_midplane = 0.12
    active = {
        "self": _active_element(
            [0.2, 0.2, 0.2, 0.2],
            source_midplane + np.asarray([-0.6, -0.2, 0.2, 0.6]),
        ),
        "a_upper": _active_element([1.0], [source_midplane + 0.5]),
        "a_lower": _active_element([1.0], [source_midplane - 0.5]),
    }
    limiter = [
        {"R": 0.5, "Z": source_midplane},
        {"R": 0.7, "Z": source_midplane + 0.6},
        {"R": 1.5, "Z": source_midplane + 0.5},
        {"R": 1.7, "Z": source_midplane},
        {"R": 1.5, "Z": source_midplane - 0.5},
        {"R": 0.7, "Z": source_midplane - 0.6},
    ]

    prepared = prepare_up_down_symmetric_machine(
        active,
        limiter_data=limiter,
        z_midplane="auto",
    )

    np.testing.assert_allclose(prepared.source_z_midplane, source_midplane)
    np.testing.assert_allclose(prepared.z_shift, -source_midplane)
    np.testing.assert_allclose(prepared.midplane_fit_rms, 0.0, atol=1e-15)
    points = np.asarray([[entry["R"], entry["Z"]] for entry in prepared.limiter_data])
    reflected = points.copy()
    reflected[:, 1] *= -1
    nearest_reflected_distance = np.min(
        np.linalg.norm(points[:, np.newaxis] - reflected[np.newaxis, :], axis=-1),
        axis=1,
    )
    np.testing.assert_allclose(nearest_reflected_distance, 0.0, atol=1e-15)

    even_reflection = prepared.even_machine_reflection_operator
    resistances = np.arange(1, len(even_reflection) + 1, dtype=float)
    symmetric_resistances = prepared.symmetrise_even_machine_resistances(resistances)
    np.testing.assert_allclose(
        even_reflection @ symmetric_resistances, symmetric_resistances
    )
    assert any(
        record.component == "limiter" for record in prepared.geometry_discrepancies
    )


def test_odd_series_circuit_requires_explicit_exclusion():
    """Opposite-polarity reflected bundles are reported and never retained."""
    upper = _active_element([1.0, 1.1], [0.6, 0.7])
    lower = _active_element([1.0, 1.1], [-0.6, -0.7])
    lower["polarity"] = -1
    active = {
        "even": _active_element([0.3, 0.3], [-0.4, 0.4]),
        "vertical_control": {"upper": upper, "lower": lower},
    }

    with pytest.raises(ValueError, match="vertical_control"):
        prepare_up_down_symmetric_machine(active)

    prepared = prepare_up_down_symmetric_machine(
        active,
        exclude_odd_active=True,
    )
    assert prepared.excluded_odd_active_names == ("vertical_control",)
    assert prepared.original_active_names == ("even",)
    assert prepared.even_active_names == ("even",)
    assert "vertical_control" not in prepared.active_coils_data
