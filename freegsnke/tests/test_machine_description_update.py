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


def _point_active_coil_data(
    R=1.3, Z=0.95, dR=0.05, dZ=0.05, resistivity=1.55e-8, polarity=1.0, multiplier=1.0
):
    return {
        "R": [R],
        "Z": [Z],
        "dR": dR,
        "dZ": dZ,
        "resistivity": resistivity,
        "polarity": polarity,
        "multiplier": multiplier,
    }


def test_active_coil_can_be_added_without_rebuilding_full_machine():
    active_coils, passive_coils, limiter, wall, probes = _load_machine_description()
    tokamak = _build_tokamak_from_data(
        active_coils, passive_coils, limiter, wall, probes
    )

    old_coil_objects = {label: tokamak[label] for label in tokamak.coils_list}
    old_resistance = np.copy(tokamak.coil_resist)
    old_self_ind = np.copy(tokamak.coil_self_ind)
    n_coils_before = tokamak.n_coils
    n_active_before = tokamak.n_active_coils

    returned = tokamak.add_active_coil("NewCoil", _point_active_coil_data())

    assert returned is tokamak
    assert tokamak.n_coils == n_coils_before + 1
    assert tokamak.n_active_coils == n_active_before + 1
    # new active coil is inserted right before the first passive structure,
    # keeping coils_list[:n_active_coils] exactly the active coils
    assert tokamak.coils_list[tokamak.n_active_coils - 1] == "NewCoil"
    assert all(
        tokamak.coils_dict[label]["active"]
        for label in tokamak.coils_list[: tokamak.n_active_coils]
    )
    assert all(
        not tokamak.coils_dict[label]["active"]
        for label in tokamak.coils_list[tokamak.n_active_coils :]
    )
    assert tokamak["NewCoil"].current == 0
    assert tokamak.coil_resist.shape == (n_coils_before + 1,)
    assert tokamak.coil_self_ind.shape == (n_coils_before + 1, n_coils_before + 1)

    for label in old_coil_objects:
        assert tokamak[label] is old_coil_objects[label]

    old_indices = [tokamak.coil_order[label] for label in old_coil_objects]
    assert np.allclose(tokamak.coil_resist[old_indices], old_resistance)
    assert np.allclose(
        tokamak.coil_self_ind[np.ix_(old_indices, old_indices)], old_self_ind
    )

    with pytest.raises(ValueError, match="already contains"):
        tokamak.add_active_coil("NewCoil", _point_active_coil_data())


def test_active_coil_can_be_removed_without_rebuilding_full_machine():
    active_coils, passive_coils, limiter, wall, probes = _load_machine_description()
    tokamak = _build_tokamak_from_data(
        active_coils, passive_coils, limiter, wall, probes
    )

    old_resistance = np.copy(tokamak.coil_resist)
    old_self_ind = np.copy(tokamak.coil_self_ind)
    n_coils_before = tokamak.n_coils
    n_active_before = tokamak.n_active_coils

    tokamak.add_active_coil("NewCoil", _point_active_coil_data())
    remaining_labels = [label for label in tokamak.coils_list if label != "NewCoil"]
    old_coil_objects = {label: tokamak[label] for label in remaining_labels}

    returned = tokamak.remove_active_coil("NewCoil")

    assert returned is tokamak
    assert tokamak.n_coils == n_coils_before
    assert tokamak.n_active_coils == n_active_before
    assert "NewCoil" not in tokamak.coil_order
    assert "NewCoil" not in tokamak.coils_dict
    assert tokamak.coils_list == remaining_labels
    assert np.allclose(tokamak.coil_resist, old_resistance)
    assert np.allclose(tokamak.coil_self_ind, old_self_ind)

    for label in remaining_labels:
        assert tokamak[label] is old_coil_objects[label]

    with pytest.raises(ValueError, match="does not contain"):
        tokamak.remove_active_coil("NewCoil")

    passive_label = tokamak.coils_list[tokamak.n_active_coils]
    with pytest.raises(ValueError, match="not an active coil"):
        tokamak.remove_active_coil(passive_label)


def test_equilibrium_add_and_remove_active_coil_updates_greens():
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
    old_vgreen = np.copy(eq._vgreen)
    old_limiter_handler = eq.limiter_handler
    n_coils_before = eq.tokamak.n_coils
    n_active_before = eq.tokamak.n_active_coils

    returned = eq.add_active_coil("NewCoil", _point_active_coil_data())

    assert returned is eq
    assert eq.tokamak.n_coils == n_coils_before + 1
    assert eq.tokamak.n_active_coils == n_active_before + 1
    assert eq._vgreen.shape[0] == n_coils_before + 1
    assert "NewCoil" in eq._pgreen
    insert_index = eq.tokamak.coil_order["NewCoil"]
    assert np.allclose(eq._vgreen[:insert_index], old_vgreen[:insert_index])
    assert np.allclose(eq._vgreen[insert_index + 1 :], old_vgreen[insert_index:])
    assert eq.limiter_handler is old_limiter_handler
    assert np.allclose(eq.tokamak_psi, eq.tokamak.calcPsiFromGreens(pgreen=eq._pgreen))

    returned = eq.remove_active_coil("NewCoil")

    assert returned is eq
    assert eq.tokamak.n_coils == n_coils_before
    assert eq.tokamak.n_active_coils == n_active_before
    assert eq._vgreen.shape[0] == n_coils_before
    assert "NewCoil" not in eq._pgreen
    assert np.allclose(eq._vgreen, old_vgreen)
    assert eq.limiter_handler is old_limiter_handler
    assert np.allclose(eq.tokamak_psi, eq.tokamak.calcPsiFromGreens(pgreen=eq._pgreen))


def _point_passive_data(R=1.0, Z=0.9, dR=0.02, dZ=0.02, resistivity=5.5e-7):
    return {"R": R, "Z": Z, "dR": dR, "dZ": dZ, "resistivity": resistivity}


def test_passive_structure_can_be_updated_without_rebuilding_full_machine():
    active_coils, passive_coils, limiter, wall, probes = _load_machine_description()
    tokamak = _build_tokamak_from_data(
        active_coils, passive_coils, limiter, wall, probes
    )

    passive_label = tokamak.coils_list[tokamak.n_active_coils]
    tokamak.set_coil_current(passive_label, 42.0)
    old_coil_objects = {label: tokamak[label] for label in tokamak.coils_list}
    old_resistance = np.copy(tokamak.coil_resist)
    old_self_ind = np.copy(tokamak.coil_self_ind)
    passive_index = tokamak.coil_order[passive_label]

    new_data = _point_passive_data()
    returned = tokamak.update_passive_structure(passive_label, new_data)

    assert returned is tokamak
    assert tokamak._last_machine_update_changed_coils == [passive_label]
    assert tokamak._last_machine_update_topology_changed is False
    assert tokamak[passive_label] is not old_coil_objects[passive_label]
    assert tokamak[passive_label].current == 42.0
    assert tokamak.current_vec[passive_index] == 42.0
    assert tokamak.coils_dict[passive_label]["dR"] == 0.02

    unchanged_indices = [
        i for i, label in enumerate(tokamak.coils_list) if label != passive_label
    ]
    for label in tokamak.coils_list:
        if label != passive_label:
            assert tokamak[label] is old_coil_objects[label]

    assert not np.isclose(
        tokamak.coil_resist[passive_index], old_resistance[passive_index]
    )
    assert np.allclose(
        tokamak.coil_resist[unchanged_indices], old_resistance[unchanged_indices]
    )
    assert np.allclose(
        tokamak.coil_self_ind[np.ix_(unchanged_indices, unchanged_indices)],
        old_self_ind[np.ix_(unchanged_indices, unchanged_indices)],
    )


def test_passive_structure_update_noops_when_data_are_unchanged():
    active_coils, passive_coils, limiter, wall, probes = _load_machine_description()
    tokamak = _build_tokamak_from_data(
        active_coils, passive_coils, limiter, wall, probes
    )

    passive_label = tokamak.coils_list[tokamak.n_active_coils]
    old_passive_object = tokamak[passive_label]
    old_resistance = np.copy(tokamak.coil_resist)
    old_self_ind = np.copy(tokamak.coil_self_ind)

    returned = tokamak.update_passive_structure(
        passive_label, copy.deepcopy(passive_coils[0])
    )

    assert returned is tokamak
    assert tokamak._last_machine_update_changed_coils == []
    assert tokamak._last_machine_update_topology_changed is False
    assert tokamak[passive_label] is old_passive_object
    assert np.allclose(tokamak.coil_resist, old_resistance)
    assert np.allclose(tokamak.coil_self_ind, old_self_ind)


def test_passive_structure_update_rejects_active_labels():
    active_coils, passive_coils, limiter, wall, probes = _load_machine_description()
    tokamak = _build_tokamak_from_data(
        active_coils, passive_coils, limiter, wall, probes
    )

    with pytest.raises(ValueError, match="not a passive structure"):
        tokamak.update_passive_structure("Solenoid", _point_passive_data())


def test_passive_structure_can_be_added_without_rebuilding_full_machine():
    active_coils, passive_coils, limiter, wall, probes = _load_machine_description()
    tokamak = _build_tokamak_from_data(
        active_coils, passive_coils, limiter, wall, probes
    )

    old_coil_objects = {label: tokamak[label] for label in tokamak.coils_list}
    old_resistance = np.copy(tokamak.coil_resist)
    old_self_ind = np.copy(tokamak.coil_self_ind)
    n_coils_before = tokamak.n_coils
    n_passive_before = tokamak.n_passive_coils

    returned = tokamak.add_passive_structure(
        _point_passive_data(), name="new_passive"
    )

    assert returned is tokamak
    assert tokamak.n_coils == n_coils_before + 1
    assert tokamak.n_passive_coils == n_passive_before + 1
    assert tokamak.coils_list[-1] == "new_passive"
    assert tokamak.coil_order["new_passive"] == n_coils_before
    assert tokamak["new_passive"].current == 0
    assert tokamak.coil_resist.shape == (n_coils_before + 1,)
    assert tokamak.coil_self_ind.shape == (n_coils_before + 1, n_coils_before + 1)

    for label in old_coil_objects:
        assert tokamak[label] is old_coil_objects[label]

    old_indices = [tokamak.coil_order[label] for label in old_coil_objects]
    assert np.allclose(tokamak.coil_resist[old_indices], old_resistance)
    assert np.allclose(
        tokamak.coil_self_ind[np.ix_(old_indices, old_indices)], old_self_ind
    )

    with pytest.raises(ValueError, match="already contains"):
        tokamak.add_passive_structure(_point_passive_data(), name="new_passive")


def test_passive_structure_can_be_removed_without_rebuilding_full_machine():
    active_coils, passive_coils, limiter, wall, probes = _load_machine_description()
    tokamak = _build_tokamak_from_data(
        active_coils, passive_coils, limiter, wall, probes
    )

    passive_label = tokamak.coils_list[tokamak.n_active_coils]
    remaining_labels = [
        label for label in tokamak.coils_list if label != passive_label
    ]
    old_coil_objects = {label: tokamak[label] for label in remaining_labels}
    n_coils_before = tokamak.n_coils
    n_passive_before = tokamak.n_passive_coils

    returned = tokamak.remove_passive_structure(passive_label)

    assert returned is tokamak
    assert tokamak.n_coils == n_coils_before - 1
    assert tokamak.n_passive_coils == n_passive_before - 1
    assert passive_label not in tokamak.coil_order
    assert passive_label not in tokamak.coils_dict
    assert tokamak.coil_resist.shape == (n_coils_before - 1,)
    assert tokamak.coil_self_ind.shape == (n_coils_before - 1, n_coils_before - 1)

    for label in remaining_labels:
        assert tokamak[label] is old_coil_objects[label]

    with pytest.raises(ValueError, match="does not contain"):
        tokamak.remove_passive_structure(passive_label)


def test_equilibrium_passive_structure_update_refreshes_only_target_greens():
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
    passive_label = eq.tokamak.coils_list[eq.tokamak.n_active_coils]
    eq.tokamak.set_coil_current(passive_label, 42.0)
    old_vgreen = np.copy(eq._vgreen)
    old_limiter_handler = eq.limiter_handler
    passive_index = eq.tokamak.coil_order[passive_label]

    returned = eq.update_passive_structure(passive_label, _point_passive_data())

    assert returned is eq
    assert eq._vgreen.shape == old_vgreen.shape
    assert eq.limiter_handler is old_limiter_handler
    assert not np.allclose(eq._vgreen[passive_index], old_vgreen[passive_index])
    for coil_index, coil_name in enumerate(eq.tokamak.coils_list):
        if coil_name != passive_label:
            assert np.allclose(eq._vgreen[coil_index], old_vgreen[coil_index])
    assert np.allclose(eq.tokamak_psi, eq.tokamak.calcPsiFromGreens(pgreen=eq._pgreen))
    assert eq.tokamak[passive_label].current == 42.0


def test_equilibrium_add_and_remove_passive_structure_updates_greens():
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
    old_vgreen = np.copy(eq._vgreen)
    old_limiter_handler = eq.limiter_handler
    n_coils_before = eq.tokamak.n_coils

    returned = eq.add_passive_structure(_point_passive_data(), name="new_passive")

    assert returned is eq
    assert eq.tokamak.n_coils == n_coils_before + 1
    assert eq._vgreen.shape[0] == n_coils_before + 1
    assert "new_passive" in eq._pgreen
    assert np.allclose(eq._vgreen[:n_coils_before], old_vgreen)
    assert eq.limiter_handler is old_limiter_handler
    assert np.allclose(eq.tokamak_psi, eq.tokamak.calcPsiFromGreens(pgreen=eq._pgreen))

    returned = eq.remove_passive_structure("new_passive")

    assert returned is eq
    assert eq.tokamak.n_coils == n_coils_before
    assert eq._vgreen.shape[0] == n_coils_before
    assert "new_passive" not in eq._pgreen
    assert np.allclose(eq._vgreen, old_vgreen)
    assert eq.limiter_handler is old_limiter_handler
    assert np.allclose(eq.tokamak_psi, eq.tokamak.calcPsiFromGreens(pgreen=eq._pgreen))
