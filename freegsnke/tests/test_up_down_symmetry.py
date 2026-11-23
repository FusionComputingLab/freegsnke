"""Tests for machine-description up-down symmetrisation."""

import os
import pickle
import warnings
from pathlib import Path
from unittest.mock import Mock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
from shapely.affinity import scale
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from freegsnke import build_machine, equilibrium_update
from freegsnke.build_machine import build_tokamak_components
from freegsnke.symmetry_reduced_solve import SymmetryReducedSolver
from freegsnke.up_down_symmetry import (
    prepare_up_down_symmetric_machine,
    symmetrise_boundary,
    symmetrise_limiter,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("lower_closed", [False, True])
def test_closed_passive_pairs_preserve_footprint(lower_closed):
    """Closing vertices and different ring starts cannot alter a conductor."""
    upper = np.array([[1, 0.4], [2, 0.4], [2, 0.6], [1, 0.6], [1, 0.4]])
    lower = np.array([[2, -0.6], [1, -0.6], [1, -0.4], [2, -0.4]])
    if lower_closed:
        lower = np.vstack((lower, lower[0]))
    passive = [
        _passive_element(name, "vessel", *p.T)
        for name, p in [("upper", upper), ("lower", lower)]
    ]
    original = pickle.dumps(passive)
    prepared = prepare_up_down_symmetric_machine({}, passive, max_pair_mismatch=1e-12)
    for output, source in zip(prepared.passive_coils_data, (upper, lower)):
        points = np.column_stack((output["R"], output["Z"]))
        assert Polygon(points).symmetric_difference(Polygon(source)).area < 1e-14
        assert np.array_equal(points[0], points[-1]) == np.array_equal(
            source[0], source[-1]
        )
    assert pickle.dumps(passive) == original


def test_mixed_passive_vertex_counts_find_compatible_pairs():
    """Mixed triangle/rectangle groups match after normalising closure."""
    triangle = np.array([[1, 0.4], [1.1, 0.4], [1.05, 0.5], [1, 0.4]])
    rectangle = np.array([[1.3, 0.4], [1.4, 0.4], [1.4, 0.5], [1.3, 0.5]])
    data = [
        _passive_element(name, "vessel", *points.T)
        for name, points in [
            ("tri_upper", triangle),
            ("rect_upper", rectangle),
            ("rect_lower", rectangle * [1, -1]),
            ("tri_lower", triangle[:-1] * [1, -1]),
        ]
    ]
    prepared = prepare_up_down_symmetric_machine({}, data)
    assert {(p.upper_name, p.lower_name) for p in prepared.passive_pairs} == {
        ("tri_upper", "tri_lower"),
        ("rect_upper", "rect_lower"),
    }


@pytest.mark.parametrize("self_first", [False, True])
def test_reduced_labels_cannot_overwrite_self_circuits(self_first):
    """A stripped pair name may also be a retained self-circuit name."""
    pairs = {
        "PF_upper": _active_element([1], [0.5]),
        "PF_lower": _active_element([1], [-0.5]),
    }
    centre = {"PF": _active_element([1, 1], [-0.3, 0.3])}
    data = {**centre, **pairs} if self_first else {**pairs, **centre}
    prepared = prepare_up_down_symmetric_machine(data)
    assert len(prepared.even_active_names) == len(set(prepared.even_active_names)) == 2
    assert set(prepared.even_active_names) == set(prepared.even_active_coils_data)
    currents = np.arange(3.0)
    even, odd = prepared.split_active_currents(currents)
    np.testing.assert_allclose(prepared.combine_active_currents(even, odd), currents)


@pytest.mark.parametrize("offset", [0.0, 1e-16, -1e-16])
def test_compound_midplane_classification_tolerates_roundoff(offset):
    """A middle bundle cannot also be classified as an unmatched upper bundle."""
    prepared = prepare_up_down_symmetric_machine(
        {"central": {"bundle": _active_element([1, 1], [-0.3, 0.3 + offset])}}
    )
    part = prepared.even_active_coils_data["central"]["bundle"]
    np.testing.assert_allclose(part["Z"], [-0.3, 0.3], atol=1e-15)


def test_scalar_passive_recentering_and_midplane_active_are_supported():
    """Standard point-coil coordinates remain usable with automatic recentring."""
    point = dict(name="point", R=1.2, Z=0.12, dR=0.02, dZ=0.02, resistivity=1e-6)
    prepared = prepare_up_down_symmetric_machine(
        {"central": _active_element([1], [0.12])}, [point], z_midplane="auto"
    )
    np.testing.assert_allclose(prepared.source_z_midplane, 0.12)
    np.testing.assert_allclose(prepared.passive_coils_data[0]["Z"], [0.0])
    assert prepared.even_active_names == ("central",)


@pytest.mark.parametrize("legacy_cache", [False, True])
def test_prepared_passive_fields_share_one_reflected_discretisation(legacy_cache):
    """Grid, named-cache and fresh point fields agree for paired/self passives."""
    upper = np.array([[1.2, 0.3], [1.4, 0.3], [1.4, 0.45], [1.2, 0.45]])
    centre = np.array([[1.5, -0.1], [1.6, -0.1], [1.6, 0.1], [1.5, 0.1]])
    data = [
        _passive_element(name, "vessel", *points.T)
        for name, points in [
            ("upper", upper),
            ("lower", upper * [1, -1]),
            ("central", centre),
        ]
    ]
    prepared = prepare_up_down_symmetric_machine(
        {"PF": _active_element([0.8], [0.0])}, data
    )
    if legacy_cache:
        for element in prepared.passive_coils_data:
            element.pop("_up_down_refinement", None)
    limiter = [
        dict(R=r, Z=z) for r, z in [(0.5, -0.6), (1.8, -0.6), (1.8, 0.6), (0.5, 0.6)]
    ]
    tokamak = build_machine.tokamak(
        active_coils_data=prepared.even_active_coils_data,
        passive_coils_data=prepared.passive_coils_data,
        limiter_data=limiter,
    )
    eq = equilibrium_update.Equilibrium(
        tokamak=tokamak, Rmin=0.4, Rmax=1.9, Zmin=-0.7, Zmax=0.7, nx=9, ny=9
    )
    static_solver = Mock()
    static_solver.forward_solve.side_effect = RuntimeError("Reached GS")
    with pytest.raises(RuntimeError, match="Reached GS"):
        SymmetryReducedSolver(
            object(),
            eq,
            static_solver,
            passive_reflection_operator=prepared.passive_reflection_operator,
        )
    counts = [
        len(coil.filaments) for _, coil in tokamak.coils if hasattr(coil, "filaments")
    ]
    pairs = [(p.upper_name, p.lower_name) for p in prepared.passive_pairs]
    pairs += [(name, name) for name in prepared.self_symmetric_passive_names]
    assert not build_machine._symmetrise_passive_refinement(
        tokamak.coils, tokamak.coils_dict, pairs
    )
    assert counts == [
        len(coil.filaments) for _, coil in tokamak.coils if hasattr(coil, "filaments")
    ]
    tokamak.set_all_coil_currents(np.array([10.0, 1000.0, 1000.0, 1000.0]))
    flux = tokamak.getPsitokamak(eq._vgreen)
    np.testing.assert_allclose(flux, flux[:, ::-1], atol=1e-14, rtol=0)
    np.testing.assert_allclose(tokamak.psi(eq.R, eq.Z), flux, atol=1e-14, rtol=0)
    np.testing.assert_allclose(
        tokamak.calcPsiFromGreens(eq._pgreen), flux, atol=1e-14, rtol=0
    )
    r, z = np.array([0.7, 1.0, 1.7]), np.array([0.2, 0.25, 0.3])
    np.testing.assert_allclose(
        tokamak.psi(r, z), tokamak.psi(r, -z), atol=1e-14, rtol=0
    )
    np.testing.assert_allclose(tokamak.Br(r, z), -tokamak.Br(r, -z), atol=1e-14, rtol=0)
    np.testing.assert_allclose(tokamak.Bz(r, z), tokamak.Bz(r, -z), atol=1e-14, rtol=0)


@pytest.fixture(autouse=True)
def close_geometry_plots():
    """Release diagnostic figures after each geometry test."""
    yield
    plt.close("all")


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


@pytest.mark.parametrize("n_bundles", [1, 2])
def test_centred_compound_coils_match_flat_description(n_bundles):
    """Packaging centred filaments into bundles must preserve the same machine."""
    parts = {
        f"bundle_{index}": _active_element(
            [
                0.2 + 0.5 * index,
                0.21 + 0.5 * index,
                0.19 + 0.5 * index,
                0.2 + 0.5 * index,
            ],
            [-0.8, -0.2, 0.2, 0.8],
        )
        for index in range(n_bundles)
    }
    source = pickle.dumps(parts)
    flat = _active_element(
        np.concatenate([part["R"] for part in parts.values()]),
        np.concatenate([part["Z"] for part in parts.values()]),
    )
    direct = prepare_up_down_symmetric_machine({"solenoid": flat})
    compound = prepare_up_down_symmetric_machine({"solenoid": parts})
    assert compound.even_active_names == ("solenoid",)
    assert not compound.active_pairs
    assert len(compound.even_active_coils_data["solenoid"]) == n_bundles
    limiter = [{"R": r, "Z": z} for r, z in [(0.1, -1), (1.2, -1), (1.2, 1), (0.1, 1)]]
    components = [
        build_tokamak_components(
            active_coils_data=prepared.even_active_coils_data, limiter_data=limiter
        )
        for prepared in (direct, compound)
    ]
    for field in ("coords", "polarity", "multiplier"):
        np.testing.assert_allclose(
            components[0]["coils_dict"]["solenoid"][field],
            components[1]["coils_dict"]["solenoid"][field],
        )
    assert pickle.dumps(parts) == source


@pytest.mark.parametrize("layout", ["paired", "series"])
@pytest.mark.parametrize("midplane", [0.0, "auto"])
def test_mixed_size_active_bundles_match_compatible_partners(layout, midplane):
    """Two- and three-filament bundles match despite unrelated keys and ordering."""
    offset = 0.12 if midplane == "auto" else 0.0
    upper = {
        "two": _active_element([1.0, 1.1], offset + np.array([0.4, 0.5])),
        "three": _active_element([1.5, 1.6, 1.7], offset + np.array([0.6, 0.7, 0.8])),
    }
    lower = {
        "other_three": _active_element(
            [1.7, 1.6, 1.5], offset - np.array([0.8, 0.7, 0.6])
        ),
        "other_two": _active_element([1.1, 1.0], offset - np.array([0.5, 0.4])),
    }
    active = (
        {"p_upper": upper, "p_lower": lower}
        if layout == "paired"
        else {"p": {**upper, **lower}}
    )
    original = pickle.dumps(active)
    prepared = prepare_up_down_symmetric_machine(active, z_midplane=midplane)
    assert len(prepared.even_active_names) == 1
    parts = next(iter(prepared.even_active_coils_data.values()))
    assert sorted(np.size(part["R"]) for part in parts.values()) == [2, 2, 3, 3]
    points = np.concatenate(
        [np.column_stack((part["R"], part["Z"])) for part in parts.values()]
    )
    reflected = points * np.array([1, -1])
    assert (
        np.min(
            np.linalg.norm(points[:, None] - reflected[None, :], axis=-1), axis=1
        ).max()
        < 1e-14
    )
    np.testing.assert_allclose(prepared.source_z_midplane, offset, atol=1e-14)
    assert pickle.dumps(active) == original


@pytest.mark.parametrize("layout", ["paired", "series"])
def test_active_bundle_matching_rejects_incompatible_filament_counts(layout):
    """A missing compatible partner still fails with a specific error."""
    upper = {
        "a": _active_element([1, 1], [0.4, 0.5]),
        "b": _active_element([2, 2], [0.6, 0.7]),
    }
    lower = {
        "c": _active_element([1, 1], [-0.4, -0.5]),
        "d": _active_element([2, 2, 2], [-0.6, -0.7, -0.8]),
    }
    active = (
        {"p_upper": upper, "p_lower": lower}
        if layout == "paired"
        else {"p": {**upper, **lower}}
    )
    with pytest.raises(ValueError, match="equal filament counts"):
        prepare_up_down_symmetric_machine(active)


@pytest.mark.parametrize("kind", ["active", "passive"])
@pytest.mark.parametrize(
    "pairs",
    [
        [("a", "b"), ("a", "c")],
        [("a", "c"), ("b", "c")],
        [("a", "b"), ("b", "a")],
        [("a", "b"), ("a", "b")],
        [("a", "a")],
    ],
)
def test_explicit_pairs_reject_reused_elements_before_symmetrisation(kind, pairs):
    """Each element may appear in only one pair and cannot pair with itself."""
    if kind == "active":
        active = {
            name: _active_element([1.0], [z])
            for name, z in [("a", 0.5), ("b", -0.5), ("c", -0.7)]
        }
        passive = None
        options = {"active_pairs": pairs}
    else:
        active = {}
        passive = [
            _passive_element(name, "case", [1.0], [z])
            for name, z in [("a", 0.5), ("b", -0.5), ("c", -0.7)]
        ]
        options = {"passive_pairs": pairs}
    source = pickle.dumps((active, passive))
    with patch("freegsnke.up_down_symmetry._symmetrise_element_pair") as symmetrise:
        with pytest.raises(
            ValueError, match=f"{kind} pair element names must be unique"
        ):
            prepare_up_down_symmetric_machine(active, passive, **options)
        symmetrise.assert_not_called()
    assert pickle.dumps((active, passive)) == source


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


def test_geometry_mismatch_is_rms_point_distance():
    """A displacement in both coordinates uses Euclidean point distances."""
    active = {
        "a_upper": _active_element([1.0], [1.0]),
        "a_lower": _active_element([0.97], [-0.96]),
    }

    prepared = prepare_up_down_symmetric_machine(active)

    np.testing.assert_allclose(
        prepared.maximum_geometry_discrepancy.reflected_rms,
        0.05,
    )


def test_unknown_explicit_passive_pair_has_clear_error():
    """Invalid explicit labels fail before geometry lookup."""
    passive = [
        _passive_element("upper", "case", [1.0], [0.5]),
        _passive_element("lower", "case", [1.0], [-0.5]),
    ]

    with pytest.raises(ValueError, match="missing"):
        prepare_up_down_symmetric_machine(
            {},
            passive,
            passive_pairs=(("upper", "missing"),),
        )


def test_boundary_tolerates_roundoff_at_midplane_vertices():
    """Near-zero trigonometric coordinates count as vertices, not crossings."""
    theta = np.linspace(0.0, 2 * np.pi, 96, endpoint=False)
    limiter = [
        {"R": 1.0 + 0.4 * np.cos(angle), "Z": 0.4 * np.sin(angle)} for angle in theta
    ]
    active = {"self": _active_element([0.2, 0.2], [-0.5, 0.5])}

    prepared = prepare_up_down_symmetric_machine(active, limiter_data=limiter)

    points = np.asarray([[entry["R"], entry["Z"]] for entry in prepared.limiter_data])
    reflected = points.copy()
    reflected[:, 1] *= -1
    distances = np.linalg.norm(
        points[:, np.newaxis] - reflected[np.newaxis, :], axis=-1
    )
    np.testing.assert_allclose(np.min(distances, axis=1), 0.0, atol=1e-14)


def test_unpaired_active_coil_raises():
    active = {
        "solenoid": _active_element(
            [0.2, 0.2],
            [-0.8, 0.8],
        ),
        "p_upper": {"1": _active_element([1.0, 1.1], [0.7, 0.8])},
        # Missing p_lower
    }

    with pytest.raises(ValueError, match="Active"):
        prepare_up_down_symmetric_machine(active)


def test_unpaired_passive_element_raises():
    active = {
        "solenoid": _active_element(
            [0.2, 0.2],
            [-0.8, 0.8],
        ),
    }

    passive = [
        _passive_element(
            "case_upper_0",
            "case_upper",
            [1.0, 1.2, 1.2, 1.0],
            [0.4] * 4,
        ),
        # No matching lower element
    ]

    with pytest.raises(ValueError, match="Passive"):
        prepare_up_down_symmetric_machine(active, passive)


def test_fails_when_negative_tolerance():
    active = {
        "solenoid": _active_element(
            [0.2, 0.2, 0.2, 0.2],
            [-0.8, -0.2, 0.19, 0.79],
        ),
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

    with pytest.raises(ValueError, match="non-negative"):
        prepare_up_down_symmetric_machine(active, passive, max_pair_mismatch=-0.05)


def test_passive_pairing_fails_when_geometry_mismatch_too_large():
    active = {
        "solenoid": _active_element(
            [0.2, 0.2],
            [-0.8, 0.8],
        ),
    }

    passive = [
        _passive_element(
            "case_upper_0",
            "case_upper",
            [1.0, 1.2, 1.2, 1.0],
            [0.4] * 4,
        ),
        _passive_element(
            "case_lower_0",
            "case_lower",
            [1.0, 1.3, 1.2, 1.0],  # deliberately wrong R
            [-0.4] * 4,
        ),
    ]

    with pytest.raises(ValueError, match="Largest failures"):
        prepare_up_down_symmetric_machine(
            active,
            passive,
            max_pair_mismatch=0.05,
        )


def test_passive_pairing_with_multiple_identical_blocks_is_stable():
    passive = [
        _passive_element("case_upper_0", "case", [1.0, 1.2, 1.2, 1.0], [0.2] * 4),
        _passive_element("case_upper_1", "case", [1.0, 1.2, 1.2, 1.0], [0.6] * 4),
        _passive_element("case_lower_0", "case", [1.0, 1.2, 1.2, 1.0], [-0.2] * 4),
        _passive_element("case_lower_1", "case", [1.0, 1.2, 1.2, 1.0], [-0.6] * 4),
    ]

    prepared = prepare_up_down_symmetric_machine({}, passive)

    assert {(p.upper_name, p.lower_name) for p in prepared.passive_pairs} == {
        ("case_upper_0", "case_lower_0"),
        ("case_upper_1", "case_lower_1"),
    }


def test_identical_passive_elements_are_paired_one_to_one():
    passive = [
        _passive_element("case_upper_0", "case", [1.0, 1.2, 1.2, 1.0], [0.2] * 4),
        _passive_element("case_upper_1", "case", [1.0, 1.2, 1.2, 1.0], [0.2] * 4),
        _passive_element("case_lower_0", "case", [1.0, 1.2, 1.2, 1.0], [-0.2] * 4),
        _passive_element("case_lower_1", "case", [1.0, 1.2, 1.2, 1.0], [-0.8] * 4),
    ]
    prepared = prepare_up_down_symmetric_machine({}, passive)

    assert len(prepared.passive_pairs) == 2
    assert {p.upper_name for p in prepared.passive_pairs} == {
        "case_upper_0",
        "case_upper_1",
    }
    assert {p.lower_name for p in prepared.passive_pairs} == {
        "case_lower_0",
        "case_lower_1",
    }
    np.testing.assert_allclose(
        sorted(p.reflected_rms for p in prepared.passive_pairs), [0.0, 0.6]
    )


def test_identify_misspelled_active_pairs():
    active = {
        "solenoid": _active_element(
            [0.2, 0.2],
            [-0.8, 0.8],
        ),
        "p_uper": {"1": _active_element([1.0, 1.1], [0.7, 0.8])},
        "p_lwer": {"1": _active_element([1.0, 1.1], [-0.7, -0.8])},
        "ps_uper": {"1": _active_element([1.2, 1.1], [0.8, 0.8])},
    }

    with pytest.raises(ValueError, match="active"):
        prepare_up_down_symmetric_machine(active)


def test_misspelled_passive_pairs():
    active = {
        "solenoid": _active_element([0.2, 0.2, 0.2, 0.2], [-0.8, -0.2, 0.19, 0.79]),
        "p_upper": {"1": _active_element([1.0, 1.1], [0.7, 0.8])},
        "p_lower": {"1": _active_element([1.0, 1.1], [-0.72, -0.82])},
    }
    passive = [
        _passive_element("case_upper_0", "case_upper", [1.0, 1.2, 1.2, 1.0], [0.4] * 4),
        _passive_element("case_upper_1", "case_upper", [1.0, 1.2, 1.2, 1.0], [0.8] * 4),
        _passive_element(
            "case_upper_2", "case_lower", [1.0, 1.2, 1.2, 1.0], [-0.82] * 4
        ),
        _passive_element(
            "case_lower_1", "case_lower", [1.0, 1.2, 1.2, 1.0], [-0.42] * 4
        ),
    ]

    prepared = prepare_up_down_symmetric_machine(
        active, passive, max_pair_mismatch=0.05
    )

    assert {(pair.upper_name, pair.lower_name) for pair in prepared.passive_pairs} == {
        ("case_upper_0", "case_lower_1"),
        ("case_upper_1", "case_upper_2"),
    }


def test_sparc_machine_can_be_symmetrised_and_built():
    machine_config_dir = os.path.join(REPO_ROOT, "machine_configs", "SPARC")

    machine_data = {}
    for component in ("active_coils", "passive_coils", "limiter", "wall"):
        path = os.path.join(machine_config_dir, f"SPARC_{component}.pickle")
        print(path)
        with open(path, "rb") as machine_file:
            machine_data[component] = pickle.load(machine_file)

    passive_pairs = [
        (f"vacuum_vessel_{index}", f"vacuum_vessel_{index + 1}")
        for index in range(0, 16, 2)
    ]

    passive_pairs.append(("VSC_coil_cover0", "VSC_coil_cover1"))
    prepared = prepare_up_down_symmetric_machine(
        machine_data["active_coils"],
        machine_data["passive_coils"],
        machine_data["limiter"],
        machine_data["wall"],
        passive_pairs=passive_pairs,
        exclude_odd_active=True,
    )

    assert len(prepared.passive_pairs) == len(passive_pairs)
    assert prepared.excluded_odd_active_names == ("VS1",)

    for pair in prepared.passive_pairs:
        upper, lower = (
            next(
                element
                for element in prepared.passive_coils_data
                if element["name"] == name
            )
            for name in (pair.upper_name, pair.lower_name)
        )
        upper_points = np.column_stack((upper["R"], upper["Z"]))
        reflected_lower = np.column_stack((lower["R"], -np.asarray(lower["Z"])))
        nearest_distances = np.min(
            np.linalg.norm(
                upper_points[:, np.newaxis] - reflected_lower[np.newaxis, :],
                axis=-1,
            ),
            axis=1,
        )
        np.testing.assert_allclose(nearest_distances, 0.0, atol=1e-14)

    for boundary in (prepared.limiter_data, prepared.wall_data):
        points = np.asarray([[entry["R"], entry["Z"]] for entry in boundary])
        reflected = points.copy()
        reflected[:, 1] *= -1
        nearest_distances = np.min(
            np.linalg.norm(
                points[:, np.newaxis] - reflected[np.newaxis, :],
                axis=-1,
            ),
            axis=1,
        )
        np.testing.assert_allclose(nearest_distances, 0.0, atol=1e-14)

    components = build_tokamak_components(
        active_coils_data=prepared.even_active_coils_data,
        passive_coils_data=prepared.passive_coils_data,
        limiter_data=prepared.limiter_data,
        wall_data=prepared.wall_data,
    )
    assert components["n_active_coils"] == len(prepared.even_active_names)
    assert components["n_passive_coils"] == len(prepared.passive_coils_data)
    assert components["n_coils"] == (
        components["n_active_coils"] + components["n_passive_coils"]
    )


@pytest.mark.parametrize("closed", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("midplane", [0.0, 0.12])
def test_symmetric_rectangle_keeps_corners_and_area(closed, reverse, midplane):
    """Intersection preserves all rectangle corners, winding and closure."""
    points = np.array([[0.5, -1], [2, -1], [2, 1], [0.5, 1]], dtype=float)
    points[:, 1] += midplane
    if reverse:
        points = np.roll(points[::-1], 1, axis=0)
    if closed:
        points = np.vstack((points, points[0]))
    boundary = [{"R": r, "Z": z} for r, z in points]
    result = symmetrise_boundary(boundary, midplane)
    actual = np.array([[p["R"], p["Z"]] for p in result])
    assert len(actual) == len(points)
    assert Polygon(actual).hausdorff_distance(Polygon(points)) < 1e-14
    assert Polygon(actual).exterior.is_ccw == Polygon(points).exterior.is_ccw
    assert (result[0] == result[-1]) == closed
    assert Polygon(actual).area == pytest.approx(3.0)
    np.testing.assert_array_equal([[p["R"], p["Z"]] for p in boundary], points)


def test_nonuniform_symmetric_vertices_are_not_redistributed():
    """Uneven vertex spacing is part of the supplied polygon, not a mesh to fix."""
    points = np.array(
        [
            [0.5, 0],
            [0.5, 0.2],
            [0.8, 0.9],
            [2, 0.4],
            [2.2, 0],
            [2, -0.4],
            [0.8, -0.9],
            [0.5, -0.2],
        ]
    )
    boundary = [{"R": r, "Z": z} for r, z in points]
    result = symmetrise_boundary(boundary)
    np.testing.assert_array_equal([[p["R"], p["Z"]] for p in result], points)


def test_limiter_discrepancy_measures_distance_to_reflected_boundary():
    """Limiter diagnostics need no one-to-one correspondence between vertices."""
    points = np.array([[1, -1], [3, -2], [3, 1], [1, 2]])
    boundary = [{"R": r, "Z": z} for r, z in points]
    with pytest.warns(UserWarning, match="Limiter changed"):
        prepared = prepare_up_down_symmetric_machine({}, limiter_data=boundary)
    result = np.array([[p["R"], p["Z"]] for p in prepared.limiter_data])
    expected = Polygon([(1, -1), (2, -1.5), (3, -1), (3, 1), (2, 1.5), (1, 1)])
    assert Polygon(result).equals(expected)
    rms = np.sqrt(0.4)
    assert prepared.maximum_geometry_discrepancy.reflected_rms == pytest.approx(rms)
    with (
        patch("freegsnke.up_down_symmetry.plt.show") as show,
        pytest.raises(ValueError, match="Largest failures"),
    ):
        prepare_up_down_symmetric_machine(
            {}, limiter_data=boundary, max_pair_mismatch=rms / 2
        )
    show.assert_not_called()


def test_unequal_boundary_vertex_counts_are_intersected():
    """The general boundary helper uses intersection despite unequal sampling."""
    points = [(1, -1), (2, -1), (2, 1), (1.5, 1.1), (1, 1)]
    boundary = [{"R": r, "Z": z} for r, z in points]
    with pytest.warns(UserWarning, match="Boundary changed"):
        result = symmetrise_boundary(boundary)
    assert Polygon([(p["R"], p["Z"]) for p in result]).equals(box(1, -1, 2, 1))


@pytest.mark.parametrize("duplicate", [False, True])
def test_symmetric_outline_keeps_redundant_vertices(duplicate):
    """Exact symmetry needs no vertex matching, including OpenSTEP duplicates."""
    points = [(1, -1), (2, -1), (2, 1), (1.5, 1), (1, 1)]
    if duplicate:
        points.insert(1, points[0])
    points.append(points[0])
    boundary = [{"R": r, "Z": z} for r, z in points]
    prepared = prepare_up_down_symmetric_machine({}, limiter_data=boundary)
    assert prepared.limiter_data == boundary
    assert prepared.maximum_geometry_discrepancy.reflected_rms == 0.0


@pytest.mark.parametrize("machine,prefix", [("example", ""), ("SPARC", "SPARC_")])
@pytest.mark.parametrize("component", ["limiter", "wall"])
def test_symmetric_machine_boundaries_keep_original_vertices(
    machine, prefix, component
):
    """The checked-in symmetric boundaries must retain their original geometry."""
    with (REPO_ROOT / "machine_configs" / machine / f"{prefix}{component}.pickle").open(
        "rb"
    ) as f:
        boundary = pickle.load(f)
    result = symmetrise_boundary(boundary)
    expected = np.array([[p["R"], p["Z"]] for p in boundary])
    actual = np.array([[p["R"], p["Z"]] for p in result])
    assert len(actual) == len(expected)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)


@pytest.mark.parametrize("machine,prefix", [("MAST-U", "MAST-U_like_"), ("test", "")])
@pytest.mark.parametrize("component", ["limiter", "wall"])
def test_mastu_boundary_intersection_is_conservative_and_idempotent(
    machine, prefix, component
):
    """Both real boundary outlines stay inside the source and remain symmetric."""
    path = REPO_ROOT / "machine_configs" / machine / f"{prefix}{component}.pickle"
    with path.open("rb") as f:
        boundary = pickle.load(f)
    result = symmetrise_boundary(boundary)
    twice = symmetrise_boundary(result)
    source = np.array([[p["R"], p["Z"]] for p in boundary])
    actual = np.array([[p["R"], p["Z"]] for p in result])
    polygon = Polygon(actual)
    assert polygon.is_valid
    assert polygon.difference(Polygon(source)).area < 1e-12
    assert (
        polygon.symmetric_difference(
            scale(polygon, xfact=1, yfact=-1, origin=(0, 0))
        ).area
        < 1e-12
    )
    assert (
        polygon.hausdorff_distance(Polygon([(p["R"], p["Z"]) for p in twice])) < 1e-12
    )


@pytest.mark.parametrize("component", ["limiter", "wall"])
def test_asymmetric_iter_boundary_uses_symmetric_inner_intersection(component):
    """ITER needs no corresponding vertices for either limiter or wall."""
    path = REPO_ROOT / "machine_configs" / "ITER" / f"ITER_{component}.pickle"
    with path.open("rb") as f:
        boundary = pickle.load(f)
    with pytest.warns(UserWarning, match="Boundary changed"):
        result = symmetrise_boundary(boundary)
    source = Polygon([(p["R"], p["Z"]) for p in boundary])
    inner = Polygon([(p["R"], p["Z"]) for p in result])
    assert inner.is_valid and 0 < inner.area < source.area
    assert inner.difference(source).area < 1e-12
    assert (
        inner.symmetric_difference(scale(inner, xfact=1, yfact=-1, origin=(0, 0))).area
        < 1e-12
    )


@pytest.mark.parametrize(
    "points",
    [
        [(1, 0), (2, 1), (1, 1), (2, 0)],
        [(1, 0), (2, 0), (3, 0), (4, 0)],
        [(1, -1), (2, -1), (2, np.nan), (1, 1)],
    ],
)
def test_invalid_boundary_is_rejected(points):
    """Invalid or non-finite polygons must be rejected before intersection."""
    with pytest.raises(ValueError, match="[Ii]nvalid boundary|finite"):
        symmetrise_boundary([{"R": r, "Z": z} for r, z in points])


@pytest.mark.parametrize("closed", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("midplane", [0.0, 0.12])
@pytest.mark.parametrize("symmetrise", [symmetrise_limiter, symmetrise_boundary])
def test_limiter_intersection_inserts_crossings(closed, reverse, midplane, symmetrise):
    """Known crossing points replace averaging while preserving closure and winding."""
    points = np.array([(1, -1), (3, -2), (3, 1), (1, 2)], dtype=float)
    points[:, 1] += midplane
    if reverse:
        points = points[::-1]
    if closed:
        points = np.vstack((points, points[0]))
    boundary = [{"R": r, "Z": z} for r, z in points]
    original = pickle.dumps(boundary)
    with pytest.warns(UserWarning, match="Limiter changed|Boundary changed"):
        result = symmetrise(boundary, midplane)
    expected = Polygon(
        [
            (r, z + midplane)
            for r, z in [(1, -1), (2, -1.5), (3, -1), (3, 1), (2, 1.5), (1, 1)]
        ]
    )
    polygon = Polygon([(p["R"], p["Z"]) for p in result])
    assert polygon.hausdorff_distance(expected) < 1e-14
    assert polygon.area == pytest.approx(5.0)
    assert len(result) == 6 + int(closed)
    assert (result[0] == result[-1]) == closed
    assert polygon.exterior.is_ccw == Polygon(points).exterior.is_ccw
    assert pickle.dumps(boundary) == original


@pytest.mark.parametrize("entry_point", ["direct", "prepare"])
def test_changed_limiter_warns_and_plots_each_time(entry_point):
    """Every accepted footprint change has a quantified warning and comparison plot."""
    boundary = [{"R": r, "Z": z} for r, z in [(1, -1), (3, -2), (3, 1), (1, 2)]]
    with patch("freegsnke.up_down_symmetry.plt.show") as show:
        for _ in range(2):
            with pytest.warns(UserWarning, match=r"16\.7%.*4 -> 6 vertices"):
                if entry_point == "direct":
                    symmetrise_limiter(boundary)
                else:
                    prepare_up_down_symmetric_machine({}, limiter_data=boundary)
            fig = plt.gcf()
            assert len(fig.axes) == 2
            assert [line.get_label() for line in fig.axes[0].lines[:3]] == [
                "Original limiter",
                "Reflected limiter",
                "Retained intersection",
            ]
            assert len(fig.axes[0].collections[0].get_offsets()) == 2
        assert show.call_count == 2
        show.assert_called_with(block=False)


def test_unchanged_limiter_and_repeated_intersection_are_quiet():
    """An unchanged footprint never produces an extra warning or plot."""
    symmetric = [
        {"R": r, "Z": z}
        for r, z in [(1, -1), (2, -1), (2, 1), (1.5, 1), (1, 1), (1, -1)]
    ]
    asymmetric = [{"R": r, "Z": z} for r, z in [(1, -1), (3, -2), (3, 1), (1, 2)]]
    with pytest.warns(UserWarning, match="Limiter changed"):
        clipped = symmetrise_limiter(asymmetric)
    with (
        warnings.catch_warnings(record=True) as caught,
        patch("freegsnke.up_down_symmetry.plt.show") as show,
    ):
        warnings.simplefilter("always")
        assert symmetrise_limiter(symmetric) == symmetric
        assert symmetrise_limiter(clipped) == clipped
        prepared = prepare_up_down_symmetric_machine({}, limiter_data=symmetric)
        assert prepared.limiter_data == symmetric
    assert not caught
    show.assert_not_called()


def test_preparation_intersects_and_reports_both_limiter_and_wall():
    """Both outlines use the same conservative strategy and get labelled plots."""
    boundary = [{"R": r, "Z": z} for r, z in [(1, -1), (3, -2), (3, 1), (1, 2)]]
    with patch("freegsnke.up_down_symmetry.plt.show") as show:
        with pytest.warns(UserWarning) as notices:
            prepared = prepare_up_down_symmetric_machine(
                {}, limiter_data=boundary, wall_data=boundary
            )
        assert [str(w.message).split(" changed")[0] for w in notices] == [
            "Limiter",
            "Wall",
        ]
        assert show.call_count == 2
        assert prepared.limiter_data == prepared.wall_data
        assert [r.component for r in prepared.geometry_discrepancies] == [
            "limiter",
            "wall",
        ]
        for record in prepared.geometry_discrepancies:
            assert record.reflected_rms == pytest.approx(np.sqrt(0.4))
        titles = [plt.figure(n)._suptitle.get_text() for n in plt.get_fignums()]
        assert any(title.startswith("Limiter symmetrisation") for title in titles)
        assert any(title.startswith("Wall symmetrisation") for title in titles)


def test_invalid_or_rejected_wall_does_not_display_an_accepted_change():
    """Wall validation and tolerance checks happen before either notification."""
    boundary = [{"R": r, "Z": z} for r, z in [(1, -1), (3, -2), (3, 1), (1, 2)]]
    invalid = [{"R": r, "Z": z} for r, z in [(1, 0), (2, 1), (1, 1), (2, 0)]]
    for wall, options in [(invalid, {}), (boundary, {"max_pair_mismatch": 0.1})]:
        with (
            warnings.catch_warnings(record=True) as caught,
            patch("freegsnke.up_down_symmetry.plt.show") as show,
            pytest.raises(ValueError),
        ):
            warnings.simplefilter("always")
            prepare_up_down_symmetric_machine(
                {}, limiter_data=boundary, wall_data=wall, **options
            )
        assert not caught
        show.assert_not_called()


def test_unequal_limiter_vertex_counts_need_no_pairing():
    """Asymmetric upper and lower sampling is accepted without resampling."""
    boundary = [
        {"R": r, "Z": z} for r, z in [(1, -1), (2, -1), (2, 1), (1.5, 1.1), (1, 1)]
    ]
    with pytest.warns(UserWarning, match="Limiter changed"):
        result = symmetrise_limiter(boundary)
    assert Polygon([(p["R"], p["Z"]) for p in result]).equals(box(1, -1, 2, 1))


@pytest.mark.parametrize(
    "machine,prefix",
    [
        ("example", ""),
        ("test", ""),
        ("MAST-U", "MAST-U_like_"),
        ("SPARC", "SPARC_"),
        ("ITER", "ITER_"),
    ],
)
def test_checked_in_limiter_intersections_are_symmetric_and_conservative(
    machine, prefix
):
    """All checked-in limiters produce one valid symmetric inner domain."""
    path = REPO_ROOT / "machine_configs" / machine / f"{prefix}limiter.pickle"
    with path.open("rb") as f:
        boundary = pickle.load(f)
    result = symmetrise_limiter(boundary)
    source = Polygon([(p["R"], p["Z"]) for p in boundary])
    inner = Polygon([(p["R"], p["Z"]) for p in result])
    reflected = scale(inner, xfact=1, yfact=-1, origin=(0, 0))
    assert inner.is_valid and not len(inner.interiors)
    assert inner.difference(source).area < 1e-12
    assert inner.symmetric_difference(reflected).area < 1e-12
    again = symmetrise_limiter(result)
    assert inner.hausdorff_distance(Polygon([(p["R"], p["Z"]) for p in again])) < 1e-12


@pytest.mark.parametrize("kind", ["empty", "disconnected", "invalid", "nonfinite"])
def test_invalid_limiter_intersection_does_not_warn_or_plot(kind):
    """Invalid or disconnected domains must be rejected before notifying a change."""
    if kind == "empty":
        points = list(box(1, 1, 2, 2).exterior.coords)
    elif kind == "disconnected":
        source = unary_union([box(1, -1, 2, 1), box(4, -1, 5, 1), box(2, 0.5, 4, 1)])
        points = list(source.exterior.coords)
    elif kind == "invalid":
        points = [(1, 0), (2, 1), (1, 1), (2, 0)]
    else:
        points = [(1, 0), (2, np.nan), (3, 1)]
    with (
        warnings.catch_warnings(record=True) as caught,
        patch("freegsnke.up_down_symmetry.plt.show") as show,
        pytest.raises(ValueError),
    ):
        warnings.simplefilter("always")
        symmetrise_limiter([{"R": r, "Z": z} for r, z in points])
    assert not caught
    show.assert_not_called()
