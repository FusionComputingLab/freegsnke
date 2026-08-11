import copy
import pickle

import numpy as np
import pytest

from freegsnke import build_machine, equilibrium_update

MACHINE_CONFIG_PATH = "./machine_configs/test"


def _load_machine_description():
    with open(f"{MACHINE_CONFIG_PATH}/active_coils.pickle", "rb") as f:
        active_coils = pickle.load(f)
    with open(f"{MACHINE_CONFIG_PATH}/passive_coils.pickle", "rb") as f:
        passive_coils = pickle.load(f)
    with open(f"{MACHINE_CONFIG_PATH}/limiter.pickle", "rb") as f:
        limiter = pickle.load(f)
    with open(f"{MACHINE_CONFIG_PATH}/wall.pickle", "rb") as f:
        wall = pickle.load(f)
    with open(f"{MACHINE_CONFIG_PATH}/magnetic_probes.pickle", "rb") as f:
        magnetic_probes = pickle.load(f)

    return active_coils, passive_coils, limiter, wall, magnetic_probes


def _build_tokamak_from_paths():
    return build_machine.tokamak(
        active_coils_path=f"{MACHINE_CONFIG_PATH}/active_coils.pickle",
        passive_coils_path=f"{MACHINE_CONFIG_PATH}/passive_coils.pickle",
        limiter_path=f"{MACHINE_CONFIG_PATH}/limiter.pickle",
        wall_path=f"{MACHINE_CONFIG_PATH}/wall.pickle",
        magnetic_probe_path=f"{MACHINE_CONFIG_PATH}/magnetic_probes.pickle",
    )


def _build_tokamak_from_data(active_coils, passive_coils, limiter, wall, probes):
    return build_machine.tokamak(
        active_coils_data=active_coils,
        passive_coils_data=passive_coils,
        limiter_data=limiter,
        wall_data=wall,
        magnetic_probe_data=probes,
    )


def test_direct_machine_description_matches_pickle_inputs():
    active_coils, passive_coils, limiter, wall, probes = _load_machine_description()

    np.random.seed(1)
    from_paths = _build_tokamak_from_paths()
    np.random.seed(1)
    from_data = _build_tokamak_from_data(
        active_coils, passive_coils, limiter, wall, probes
    )

    assert from_data.coils_list == from_paths.coils_list
    assert from_data.n_active_coils == from_paths.n_active_coils
    assert from_data.n_passive_coils == from_paths.n_passive_coils
    assert from_data.n_coils == from_paths.n_coils
    active_slice = slice(0, from_paths.n_active_coils)
    assert np.allclose(
        from_data.coil_resist[active_slice], from_paths.coil_resist[active_slice]
    )
    assert np.allclose(
        from_data.coil_self_ind[active_slice, active_slice],
        from_paths.coil_self_ind[active_slice, active_slice],
    )
    assert from_data.coil_self_ind.shape == from_paths.coil_self_ind.shape

    for coil_name in from_paths.coils_list[: from_paths.n_active_coils]:
        assert np.allclose(
            from_data.coils_dict[coil_name]["coords"],
            from_paths.coils_dict[coil_name]["coords"],
        )


def test_machine_description_can_be_updated_in_place_from_direct_data():
    active_coils, passive_coils, limiter, wall, probes = _load_machine_description()
    tokamak = _build_tokamak_from_data(
        active_coils, passive_coils, limiter, wall, probes
    )

    tokamak.set_coil_current("Solenoid", 123.0)
    old_id = id(tokamak)
    old_resistance = np.copy(tokamak.coil_resist)
    old_self_ind = np.copy(tokamak.coil_self_ind)
    unchanged_label = next(label for label in tokamak.coils_list if label != "Solenoid")
    unchanged_index = tokamak.coil_order[unchanged_label]
    old_unchanged_coil_object = tokamak[unchanged_label]

    updated_active_coils = copy.deepcopy(active_coils)
    updated_active_coils["Solenoid"]["R"] = [
        r + 1.0e-3 for r in updated_active_coils["Solenoid"]["R"]
    ]

    returned = tokamak.set_machine_description(
        active_coils_data=updated_active_coils,
        passive_coils_data=passive_coils,
        limiter_data=limiter,
        wall_data=wall,
        magnetic_probe_data=probes,
    )

    assert returned is tokamak
    assert id(tokamak) == old_id
    assert tokamak._last_machine_update_changed_coils == ["Solenoid"]
    assert tokamak._last_machine_update_topology_changed is False
    assert tokamak[unchanged_label] is old_unchanged_coil_object
    assert tokamak["Solenoid"].current == 123.0
    assert tokamak.current_vec[tokamak.coil_order["Solenoid"]] == 123.0
    assert np.allclose(
        tokamak.coils_dict["Solenoid"]["coords"][0],
        np.array(active_coils["Solenoid"]["R"]) + 1.0e-3,
    )
    assert not np.allclose(tokamak.coil_resist, old_resistance)
    assert np.isclose(
        tokamak.coil_resist[unchanged_index], old_resistance[unchanged_index]
    )
    assert np.isclose(
        tokamak.coil_self_ind[unchanged_index, unchanged_index],
        old_self_ind[unchanged_index, unchanged_index],
    )


def test_equilibrium_machine_description_update_refreshes_cached_geometry():
    active_coils, passive_coils, limiter, wall, probes = _load_machine_description()
    tokamak = _build_tokamak_from_data(
        active_coils, passive_coils, limiter, wall, probes
    )

    eq = equilibrium_update.Equilibrium(
        tokamak=tokamak,
        Rmin=0.1,
        Rmax=2.0,
        Zmin=-2.2,
        Zmax=2.2,
        nx=17,
        ny=17,
    )
    eq.tokamak.set_coil_current("Solenoid", 123.0)
    old_vgreen = np.copy(eq._vgreen)
    solenoid_index = eq.tokamak.coil_order["Solenoid"]

    updated_active_coils = copy.deepcopy(active_coils)
    updated_active_coils["Solenoid"]["R"] = [
        r + 1.0e-3 for r in updated_active_coils["Solenoid"]["R"]
    ]

    returned = eq.update_machine_description(
        active_coils_data=updated_active_coils,
        passive_coils_data=passive_coils,
        limiter_data=limiter,
        wall_data=wall,
        magnetic_probe_data=probes,
    )

    assert returned is eq
    assert eq._vgreen.shape == old_vgreen.shape
    assert not np.allclose(eq._vgreen[solenoid_index], old_vgreen[solenoid_index])
    for coil_index, coil_name in enumerate(eq.tokamak.coils_list):
        if coil_name != "Solenoid":
            assert np.allclose(eq._vgreen[coil_index], old_vgreen[coil_index])
    assert np.allclose(eq.tokamak_psi, eq.tokamak.calcPsiFromGreens(pgreen=eq._pgreen))
    assert eq.mask_inside_limiter.shape == eq.R.shape
    assert eq.tokamak["Solenoid"].current == 123.0


def test_active_coil_can_be_updated_without_rebuilding_full_machine():
    active_coils, passive_coils, limiter, wall, probes = _load_machine_description()
    tokamak = _build_tokamak_from_data(
        active_coils, passive_coils, limiter, wall, probes
    )

    tokamak.set_coil_current("Solenoid", 123.0)
    old_coil_objects = {label: tokamak[label] for label in tokamak.coils_list}
    old_resistance = np.copy(tokamak.coil_resist)
    old_self_ind = np.copy(tokamak.coil_self_ind)
    solenoid_index = tokamak.coil_order["Solenoid"]

    updated_solenoid = copy.deepcopy(active_coils["Solenoid"])
    updated_solenoid["R"] = [r + 1.0e-3 for r in updated_solenoid["R"]]

    returned = tokamak.update_active_coil("Solenoid", updated_solenoid)

    assert returned is tokamak
    assert tokamak._last_machine_update_changed_coils == ["Solenoid"]
    assert tokamak._last_machine_update_topology_changed is False
    assert tokamak["Solenoid"] is not old_coil_objects["Solenoid"]
    assert tokamak["Solenoid"].current == 123.0
    assert tokamak.current_vec[solenoid_index] == 123.0
    assert np.allclose(
        tokamak.coils_dict["Solenoid"]["coords"][0],
        np.array(active_coils["Solenoid"]["R"]) + 1.0e-3,
    )
    assert np.allclose(
        tokamak._machine_description_data["active_coils"]["Solenoid"]["R"],
        np.array(active_coils["Solenoid"]["R"]) + 1.0e-3,
    )

    unchanged_indices = [
        i for i, label in enumerate(tokamak.coils_list) if label != "Solenoid"
    ]
    for label in tokamak.coils_list:
        if label != "Solenoid":
            assert tokamak[label] is old_coil_objects[label]

    assert not np.isclose(
        tokamak.coil_resist[solenoid_index], old_resistance[solenoid_index]
    )
    assert np.allclose(
        tokamak.coil_resist[unchanged_indices], old_resistance[unchanged_indices]
    )
    assert not np.allclose(
        tokamak.coil_self_ind[solenoid_index], old_self_ind[solenoid_index]
    )
    assert np.allclose(
        tokamak.coil_self_ind[np.ix_(unchanged_indices, unchanged_indices)],
        old_self_ind[np.ix_(unchanged_indices, unchanged_indices)],
    )


def test_active_coil_update_noops_when_data_are_unchanged():
    active_coils, passive_coils, limiter, wall, probes = _load_machine_description()
    tokamak = _build_tokamak_from_data(
        active_coils, passive_coils, limiter, wall, probes
    )

    old_solenoid_object = tokamak["Solenoid"]
    old_resistance = np.copy(tokamak.coil_resist)
    old_self_ind = np.copy(tokamak.coil_self_ind)

    returned = tokamak.update_active_coil(
        "Solenoid", copy.deepcopy(active_coils["Solenoid"])
    )

    assert returned is tokamak
    assert tokamak._last_machine_update_changed_coils == []
    assert tokamak._last_machine_update_topology_changed is False
    assert tokamak["Solenoid"] is old_solenoid_object
    assert np.allclose(tokamak.coil_resist, old_resistance)
    assert np.allclose(tokamak.coil_self_ind, old_self_ind)


def test_active_coil_update_rejects_passive_labels():
    active_coils, passive_coils, limiter, wall, probes = _load_machine_description()
    tokamak = _build_tokamak_from_data(
        active_coils, passive_coils, limiter, wall, probes
    )

    passive_label = tokamak.coils_list[tokamak.n_active_coils]
    with pytest.raises(ValueError, match="not an active coil"):
        tokamak.update_active_coil(
            passive_label, copy.deepcopy(active_coils["Solenoid"])
        )


def test_equilibrium_active_coil_update_refreshes_only_target_greens():
    active_coils, passive_coils, limiter, wall, probes = _load_machine_description()
    tokamak = _build_tokamak_from_data(
        active_coils, passive_coils, limiter, wall, probes
    )
    eq = equilibrium_update.Equilibrium(
        tokamak=tokamak,
        Rmin=0.1,
        Rmax=2.0,
        Zmin=-2.2,
        Zmax=2.2,
        nx=17,
        ny=17,
    )
    eq.tokamak.set_coil_current("Solenoid", 123.0)
    old_vgreen = np.copy(eq._vgreen)
    old_limiter_handler = eq.limiter_handler
    solenoid_index = eq.tokamak.coil_order["Solenoid"]

    updated_solenoid = copy.deepcopy(active_coils["Solenoid"])
    updated_solenoid["R"] = [r + 1.0e-3 for r in updated_solenoid["R"]]

    returned = eq.update_active_coil("Solenoid", updated_solenoid)

    assert returned is eq
    assert eq._vgreen.shape == old_vgreen.shape
    assert eq.limiter_handler is old_limiter_handler
    assert not np.allclose(eq._vgreen[solenoid_index], old_vgreen[solenoid_index])
    for coil_index, coil_name in enumerate(eq.tokamak.coils_list):
        if coil_name != "Solenoid":
            assert np.allclose(eq._vgreen[coil_index], old_vgreen[coil_index])
    assert np.allclose(eq.tokamak_psi, eq.tokamak.calcPsiFromGreens(pgreen=eq._pgreen))
    assert eq.tokamak["Solenoid"].current == 123.0
