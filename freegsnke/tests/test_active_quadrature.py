import numpy as np

from freegs4e.gradshafranov import Greens
from freegs4e.multi_coil import MultiCoil

from freegsnke.build_machine import (
    ActiveQuadratureMultiCoil,
    build_active_coil_component,
    build_active_coil_dict,
    build_actives,
)

UNSET = object()


def _active_coil(active_quadrature=UNSET):
    coil = {
        "R": [1.0],
        "Z": [0.0],
        "dR": 0.2,
        "dZ": 0.4,
        "resistivity": 1.0e-8,
        "polarity": 1,
        "multiplier": 2.0,
    }
    if active_quadrature is not UNSET:
        coil["active_quadrature"] = active_quadrature
    return {"A": coil}


def test_active_quadrature_omitted_preserves_legacy_metadata():
    coils_dict = build_active_coil_dict(_active_coil())

    assert np.allclose(coils_dict["A"]["coords"], [[1.0], [0.0]])
    assert np.allclose(coils_dict["A"]["multiplier"], [2.0])
    assert coils_dict["A"]["dR"] == 0.2
    assert coils_dict["A"]["dZ"] == 0.4


def test_active_quadrature_none_uses_standard_multicoil():
    omitted_coils = build_actives(_active_coil())
    _, omitted_circuit = omitted_coils[0]
    omitted_multicoil = omitted_circuit.coils[0][1]

    none_coils = build_actives(_active_coil(active_quadrature=None))
    _, none_circuit = none_coils[0]
    none_multicoil = none_circuit.coils[0][1]

    assert type(omitted_multicoil) is MultiCoil
    assert type(none_multicoil) is MultiCoil
    assert not isinstance(none_multicoil, ActiveQuadratureMultiCoil)


def test_active_quadrature_expands_each_element_with_equal_weights():
    coils_dict = build_active_coil_dict(_active_coil(active_quadrature=(2, 3)))

    assert coils_dict["A"]["coords"].shape == (2, 6)
    assert np.isclose(np.sum(coils_dict["A"]["multiplier"]), 2.0)
    assert np.allclose(coils_dict["A"]["multiplier"], np.full(6, 2.0 / 6.0))
    assert np.isclose(
        np.average(
            coils_dict["A"]["coords"][0],
            weights=coils_dict["A"]["multiplier"],
        ),
        1.0,
    )
    assert np.isclose(
        np.average(
            coils_dict["A"]["coords"][1],
            weights=coils_dict["A"]["multiplier"],
        ),
        0.0,
    )
    assert np.isclose(coils_dict["A"]["dR"], 0.1)
    assert np.isclose(coils_dict["A"]["dZ"], 0.4 / 3.0)


def test_active_quadrature_machine_greens_match_manual_average():
    coils = build_actives(_active_coil(active_quadrature=(2, 2)))
    _, circuit = coils[0]
    R_eval = np.array([1.4, 1.8])
    Z_eval = np.array([0.1, -0.2])

    actual = circuit.createPsiGreensVec(R_eval, Z_eval)

    offsets_R = np.array([-0.05, 0.05])
    offsets_Z = np.array([-0.1, 0.1])
    expected = 0.0
    for dR in offsets_R:
        for dZ in offsets_Z:
            expected += 0.25 * Greens(1.0 + dR, dZ, R_eval, Z_eval)
    expected *= 2.0

    assert np.allclose(actual, expected)


def test_active_quadrature_auto_uses_nearby_active_windings():
    active_coils = _active_coil(active_quadrature="auto")
    active_coils["B"] = {
        "R": [1.4],
        "Z": [0.0],
        "dR": 0.2,
        "dZ": 0.4,
        "resistivity": 1.0e-8,
        "polarity": 1,
        "multiplier": 1.0,
    }

    coils_dict = build_active_coil_dict(active_coils)

    assert coils_dict["A"]["coords"].shape == (2, 8)
    assert np.isclose(np.sum(coils_dict["A"]["multiplier"]), 2.0)
    assert np.isclose(coils_dict["A"]["dR"], 0.1)
    assert np.isclose(coils_dict["A"]["dZ"], 0.1)
    assert coils_dict["B"]["coords"].shape == (2, 1)


def test_active_quadrature_auto_update_component_uses_full_active_context():
    active_coils = _active_coil(active_quadrature="auto")
    active_coils["B"] = {
        "R": [1.4],
        "Z": [0.0],
        "dR": 0.2,
        "dZ": 0.4,
        "resistivity": 1.0e-8,
        "polarity": 1,
        "multiplier": 1.0,
    }

    _, metadata = build_active_coil_component(
        "A",
        active_coils["A"],
        active_coils_context=active_coils,
    )

    assert metadata["coords"].shape == (2, 8)
