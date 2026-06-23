import os
import pickle
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from scipy.stats.qmc import LatinHypercube

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "freegsnke-matplotlib")
)

from freegsnke import build_machine, refine_passive

REPO_ROOT = Path(__file__).resolve().parents[2]
MACHINE_CONFIG_DIR = REPO_ROOT / "machine_configs" / "example"
TEST_DATA_DIR = Path(__file__).resolve().parent / "baselines"

EXAMPLE_MACHINE_BASELINE = TEST_DATA_DIR / "example_machine_build.npz"

EXPECTED_COILS_LIST = [
    "Pz",
    "P1",
    "P2",
    "Solenoid",
    "passive_0",
    "passive_1",
    "passive_2",
    "passive_3",
    "passive_lower_wall",
    "passive_upper_wall",
    "passive_left_wall",
    "passive_right_wall",
]


@pytest.fixture()
def example_paths():
    """Return pickle paths for the example00 machine configuration."""
    return {
        "active_coils_path": str(MACHINE_CONFIG_DIR / "active_coils.pickle"),
        "passive_coils_path": str(MACHINE_CONFIG_DIR / "passive_coils.pickle"),
        "limiter_path": str(MACHINE_CONFIG_DIR / "limiter.pickle"),
        "wall_path": str(MACHINE_CONFIG_DIR / "wall.pickle"),
        "magnetic_probe_path": str(MACHINE_CONFIG_DIR / "magnetic_probes.pickle"),
    }


@pytest.fixture()
def example_data():
    """Load the example00 machine configuration pickle data."""
    data = {}
    for name in [
        "active_coils",
        "passive_coils",
        "limiter",
        "wall",
        "magnetic_probes",
    ]:
        with (MACHINE_CONFIG_DIR / f"{name}.pickle").open("rb") as f:
            data[name] = pickle.load(f)
    return data


@pytest.fixture()
def example_tokamak(example_paths):
    """Build the example00 tokamak from pickle paths."""
    _reset_refinement_engine()
    return build_machine.tokamak(**example_paths)


def _reset_refinement_engine():
    """Reset passive refinement sampling so regression builds are deterministic."""
    refine_passive.engine = LatinHypercube(d=2, seed=42)


@pytest.fixture()
def tiny_machine_data():
    """Return a small in-memory machine covering active/passive/probe variants."""
    active_coils = {
        "TinySingle": {
            "R": np.array([1.0]),
            "Z": np.array([0.0]),
            "dR": 0.05,
            "dZ": 0.04,
            "resistivity": 1.0e-6,
            "polarity": 1,
            "multiplier": 2.0,
        },
        "TinyCircuit": {
            "upper": {
                "R": np.array([1.2, 1.25]),
                "Z": np.array([0.35, 0.4]),
                "dR": 0.03,
                "dZ": 0.03,
                "resistivity": 2.0e-6,
                "polarity": 1,
                "multiplier": 1.5,
            },
            "lower": {
                "R": np.array([1.2, 1.25]),
                "Z": np.array([-0.35, -0.4]),
                "dR": 0.03,
                "dZ": 0.03,
                "resistivity": 2.0e-6,
                "polarity": -1,
                "multiplier": 1.5,
            },
        },
    }

    passive_coils = [
        {
            "name": "passive_pin",
            "R": 0.8,
            "Z": -0.2,
            "dR": 0.02,
            "dZ": 0.03,
            "resistivity": 3.0e-6,
        },
        {
            "name": "passive_plate",
            "R": np.array([1.35, 1.55, 1.55, 1.35]),
            "Z": np.array([-0.1, -0.1, 0.1, 0.1]),
            "resistivity": 4.0e-6,
            "min_refine_per_area": 100,
            "min_refine_per_length": 200,
        },
    ]

    limiter = [
        {"R": 0.6, "Z": -0.6},
        {"R": 1.6, "Z": -0.6},
        {"R": 1.6, "Z": 0.6},
        {"R": 0.6, "Z": 0.6},
        {"R": 0.6, "Z": -0.6},
    ]
    wall = [
        {"R": 0.5, "Z": -0.7},
        {"R": 1.7, "Z": -0.7},
        {"R": 1.7, "Z": 0.7},
        {"R": 0.5, "Z": 0.7},
        {"R": 0.5, "Z": -0.7},
    ]
    magnetic_probes = {
        "flux_loops": [
            {"name": "tiny_flux", "position": np.array([1.05, 0.15])},
        ],
        "pickups": [
            {
                "name": "tiny_pickup",
                "position": np.array([1.1, 0.0, -0.15]),
                "orientation": "PARALLEL",
                "orientation_vector": np.array([0.0, 0.0, 1.0]),
            },
        ],
    }

    return {
        "active_coils": active_coils,
        "passive_coils": passive_coils,
        "limiter": limiter,
        "wall": wall,
        "magnetic_probes": magnetic_probes,
    }


def _build_tiny_tokamak(tiny_machine_data, refine_mode="G"):
    """Build the in-memory test machine with independent input dictionaries."""
    _reset_refinement_engine()
    return build_machine.tokamak(
        active_coils_data=deepcopy(tiny_machine_data["active_coils"]),
        passive_coils_data=deepcopy(tiny_machine_data["passive_coils"]),
        limiter_data=deepcopy(tiny_machine_data["limiter"]),
        wall_data=deepcopy(tiny_machine_data["wall"]),
        magnetic_probe_data=deepcopy(tiny_machine_data["magnetic_probes"]),
        refine_mode=refine_mode,
    )


def _assert_example_machine_structure(tokamak):
    """Assert stable structural invariants of the example00 machine."""
    assert tokamak.coils_list == EXPECTED_COILS_LIST
    assert tokamak.n_active_coils == 4
    assert tokamak.n_passive_coils == 8
    assert tokamak.n_coils == 12

    expected_shapes = {
        "Pz": (2, 1),
        "P1": (2, 8),
        "P2": (2, 6),
        "Solenoid": (2, 32),
        "passive_0": (2, 1),
        "passive_1": (2, 1),
        "passive_2": (2, 1),
        "passive_3": (2, 1),
        "passive_lower_wall": (2, 396),
        "passive_upper_wall": (2, 404),
        "passive_left_wall": (2, 666),
        "passive_right_wall": (2, 846),
    }
    for coil_name, shape in expected_shapes.items():
        assert tokamak.coils_dict[coil_name]["coords"].shape == shape

    for coil_name in ["Pz", "P1", "P2", "Solenoid"]:
        assert tokamak.coils_dict[coil_name]["active"] is True
    for coil_name in EXPECTED_COILS_LIST[4:]:
        assert tokamak.coils_dict[coil_name]["active"] is False

    assert np.array_equal(tokamak.coils_dict["P2"]["polarity"], [1, 1, 1, -1, -1, -1])
    assert np.isclose(np.sum(tokamak.coils_dict["P1"]["multiplier"]), 8)
    assert np.isclose(np.sum(tokamak.coils_dict["Solenoid"]["multiplier"]), 32)
    assert np.isclose(np.sum(tokamak.coils_dict["passive_0"]["multiplier"]), 1)
    assert np.isclose(
        np.sum(tokamak.coils_dict["passive_lower_wall"]["multiplier"]),
        1 / tokamak.coils_dict["passive_lower_wall"]["coords"].shape[1],
    )

    assert len(tokamak.limiter.R) == 20
    assert len(tokamak.wall.R) == 5
    assert len(tokamak.probes.floops) == 2
    assert len(tokamak.probes.pickups) == 2
    assert [probe["name"] for probe in tokamak.probes.floops] == [
        "fl_nu_01",
        "fl_nu_02",
    ]
    assert [probe["name"] for probe in tokamak.probes.pickups] == [
        "b_c1_p01",
        "b_c1_t02",
    ]


def test_example_machine_matches_notebook_structure(example_tokamak):
    """Regression test for the machine built in example00."""
    _assert_example_machine_structure(example_tokamak)


def test_example_machine_matches_numerical_baseline(example_tokamak):
    """Check R/M matrices and representative geometry against stored baselines."""
    baseline = np.load(EXAMPLE_MACHINE_BASELINE)

    assert np.allclose(example_tokamak.coil_resist, baseline["coil_resist"])
    assert np.allclose(example_tokamak.coil_self_ind, baseline["coil_self_ind"])
    assert np.allclose(example_tokamak.limiter.R, baseline["limiter_R"])
    assert np.allclose(example_tokamak.limiter.Z, baseline["limiter_Z"])
    assert np.allclose(example_tokamak.wall.R, baseline["wall_R"])
    assert np.allclose(example_tokamak.wall.Z, baseline["wall_Z"])

    assert np.allclose(
        example_tokamak.coils_dict["P1"]["coords"], baseline["P1_coords"]
    )
    assert np.allclose(
        example_tokamak.coils_dict["P2"]["polarity"], baseline["P2_polarity"]
    )
    assert np.allclose(
        example_tokamak.coils_dict["passive_lower_wall"]["coords"],
        baseline["passive_lower_wall_coords"],
    )
    assert np.allclose(
        example_tokamak.coils_dict["passive_lower_wall"]["area"],
        baseline["passive_lower_wall_area"],
    )

    assert np.all(np.isfinite(example_tokamak.coil_resist))
    assert np.all(np.isfinite(example_tokamak.coil_self_ind))
    assert np.all(example_tokamak.coil_resist > 0)
    assert np.all(np.diag(example_tokamak.coil_self_ind) > 0)
    assert np.allclose(example_tokamak.coil_self_ind, example_tokamak.coil_self_ind.T)


def test_mixed_path_and_direct_data_build_matches_path_build(
    example_paths, example_data
):
    """Cover the alternate build style from example00."""
    _reset_refinement_engine()
    path_tokamak = build_machine.tokamak(**example_paths)
    _reset_refinement_engine()
    mixed_tokamak = build_machine.tokamak(
        active_coils_path=example_paths["active_coils_path"],
        passive_coils_data=deepcopy(example_data["passive_coils"]),
        limiter_data=deepcopy(example_data["limiter"]),
        wall_data=deepcopy(example_data["wall"]),
        magnetic_probe_path=example_paths["magnetic_probe_path"],
    )

    _assert_example_machine_structure(mixed_tokamak)
    assert mixed_tokamak.coils_list == path_tokamak.coils_list
    assert np.allclose(mixed_tokamak.coil_resist, path_tokamak.coil_resist)
    assert np.allclose(mixed_tokamak.coil_self_ind, path_tokamak.coil_self_ind)


def test_optional_machine_inputs_have_documented_defaults(example_data):
    """Check defaults for omitted passives, wall, and magnetic probes."""
    _reset_refinement_engine()
    tokamak = build_machine.tokamak(
        active_coils_data=deepcopy(example_data["active_coils"]),
        limiter_data=deepcopy(example_data["limiter"]),
    )

    assert tokamak.n_active_coils == 4
    assert tokamak.n_passive_coils == 0
    assert tokamak.n_coils == 4
    assert tokamak.coils_list == ["Pz", "P1", "P2", "Solenoid"]
    assert np.allclose(tokamak.wall.R, tokamak.limiter.R)
    assert np.allclose(tokamak.wall.Z, tokamak.limiter.Z)
    assert not hasattr(tokamak.probes, "floops")
    assert tokamak.coil_resist.shape == (4,)
    assert tokamak.coil_self_ind.shape == (4, 4)


def test_in_memory_machine_builds_active_and_passive_variants(tiny_machine_data):
    """Cover direct-data builds with single/circuit actives and scalar/polygon passives."""
    tokamak = _build_tiny_tokamak(tiny_machine_data)

    assert tokamak.coils_list == [
        "TinySingle",
        "TinyCircuit",
        "passive_pin",
        "passive_plate",
    ]
    assert tokamak.n_active_coils == 2
    assert tokamak.n_passive_coils == 2
    assert tokamak.n_coils == 4

    assert tokamak.coils_dict["TinySingle"]["active"] is True
    assert tokamak.coils_dict["TinySingle"]["coords"].shape == (2, 1)
    assert np.array_equal(tokamak.coils_dict["TinySingle"]["polarity"], [1])
    assert np.array_equal(tokamak.coils_dict["TinySingle"]["multiplier"], [2.0])

    assert tokamak.coils_dict["TinyCircuit"]["active"] is True
    assert tokamak.coils_dict["TinyCircuit"]["coords"].shape == (2, 4)
    assert np.array_equal(tokamak.coils_dict["TinyCircuit"]["polarity"], [1, 1, -1, -1])
    assert np.array_equal(
        tokamak.coils_dict["TinyCircuit"]["multiplier"], [1.5, 1.5, 1.5, 1.5]
    )

    assert tokamak.coils_dict["passive_pin"]["active"] is False
    assert tokamak.coils_dict["passive_pin"]["coords"].shape == (2, 1)
    assert tokamak.coils_dict["passive_plate"]["active"] is False
    assert tokamak.coils_dict["passive_plate"]["vertices"].shape == (2, 4)

    passive_plate_coords = tokamak.coils_dict["passive_plate"]["coords"]
    assert passive_plate_coords.shape[0] == 2
    assert passive_plate_coords.shape[1] > 1
    assert np.all(np.isfinite(passive_plate_coords))
    assert np.isclose(
        tokamak.coils_dict["passive_plate"]["multiplier"][0],
        1 / passive_plate_coords.shape[1],
    )

    assert len(tokamak.limiter.R) == 5
    assert len(tokamak.wall.R) == 5
    assert tokamak.coil_resist.shape == (4,)
    assert tokamak.coil_self_ind.shape == (4, 4)
    assert np.all(np.isfinite(tokamak.coil_resist))
    assert np.all(np.isfinite(tokamak.coil_self_ind))
    assert np.all(tokamak.coil_resist > 0)
    assert np.allclose(tokamak.coil_self_ind, tokamak.coil_self_ind.T)


def test_latin_hypercube_refinement_builds_polygon_passive(tiny_machine_data):
    """Cover the LH passive refinement path with deterministic direct input data."""
    tokamak = _build_tiny_tokamak(tiny_machine_data, refine_mode="LH")

    passive_plate = tokamak.coils_dict["passive_plate"]
    coords = passive_plate["coords"]

    assert coords.shape == (2, 56)
    assert np.all(np.isfinite(coords))
    assert np.all(coords[0] >= 1.35)
    assert np.all(coords[0] <= 1.55)
    assert np.all(coords[1] >= -0.1)
    assert np.all(coords[1] <= 0.1)
    assert np.isclose(passive_plate["multiplier"][0], 1 / coords.shape[1])

    assert tokamak.coil_resist.shape == (4,)
    assert tokamak.coil_self_ind.shape == (4, 4)
    assert np.all(np.isfinite(tokamak.coil_resist))
    assert np.all(np.isfinite(tokamak.coil_self_ind))
    assert np.allclose(tokamak.coil_self_ind, tokamak.coil_self_ind.T)


def test_direct_magnetic_probe_data_populates_probe_metadata(tiny_machine_data):
    """Check direct probe data is attached with matching coil metadata."""
    tokamak = _build_tiny_tokamak(tiny_machine_data)

    assert tokamak.probes.coil_names == tokamak.coils_list
    assert tokamak.probes.coils_dict is tokamak.coils_dict

    assert len(tokamak.probes.floops) == 1
    assert tokamak.probes.floops[0]["name"] == "tiny_flux"
    assert np.allclose(
        tokamak.probes.floops[0]["position"],
        tiny_machine_data["magnetic_probes"]["flux_loops"][0]["position"],
    )

    assert len(tokamak.probes.pickups) == 1
    assert tokamak.probes.pickups[0]["name"] == "tiny_pickup"
    assert np.allclose(
        tokamak.probes.pickups[0]["position"],
        tiny_machine_data["magnetic_probes"]["pickups"][0]["position"],
    )
    assert np.allclose(
        tokamak.probes.pickups[0]["orientation_vector"],
        tiny_machine_data["magnetic_probes"]["pickups"][0]["orientation_vector"],
    )


@pytest.mark.parametrize(
    "kwargs,error_message",
    [
        ({}, "active_coils_data"),
        ({"active_coils_data": {}}, "limiter_data"),
    ],
)
def test_required_machine_inputs_are_validated(kwargs, error_message):
    """Check that required active coil and limiter inputs are enforced."""
    with pytest.raises(ValueError, match=error_message):
        build_machine.tokamak(**kwargs)


@pytest.mark.parametrize(
    "data_key,path_key",
    [
        ("active_coils_data", "active_coils_path"),
        ("passive_coils_data", "passive_coils_path"),
        ("limiter_data", "limiter_path"),
        ("wall_data", "wall_path"),
        ("magnetic_probe_data", "magnetic_probe_path"),
    ],
)
def test_data_and_path_inputs_are_mutually_exclusive(
    example_paths, example_data, data_key, path_key
):
    """Check that each data source accepts either direct data or a path."""
    fixture_keys = {
        "active_coils_data": "active_coils",
        "passive_coils_data": "passive_coils",
        "limiter_data": "limiter",
        "wall_data": "wall",
        "magnetic_probe_data": "magnetic_probes",
    }
    kwargs = {
        "active_coils_data": deepcopy(example_data["active_coils"]),
        "limiter_data": deepcopy(example_data["limiter"]),
    }
    kwargs[data_key] = deepcopy(example_data[fixture_keys[data_key]])
    kwargs[path_key] = example_paths[path_key]

    with pytest.raises(ValueError):
        build_machine.tokamak(**kwargs)


def test_copy_tokamak_preserves_machine_regression_data(example_tokamak):
    """Check that copying preserves regression-critical machine attributes."""
    copied = build_machine.copy_tokamak(example_tokamak)

    assert copied is not example_tokamak
    assert copied.coils_list == example_tokamak.coils_list
    assert copied.n_active_coils == example_tokamak.n_active_coils
    assert copied.n_passive_coils == example_tokamak.n_passive_coils
    assert copied.n_coils == example_tokamak.n_coils
    assert np.allclose(copied.coil_resist, example_tokamak.coil_resist)
    assert np.allclose(copied.coil_self_ind, example_tokamak.coil_self_ind)
    assert copied.coil_resist is not example_tokamak.coil_resist
    assert copied.coil_self_ind is not example_tokamak.coil_self_ind
    assert copied.probes is example_tokamak.probes


def test_example_machine_plot_smoke(monkeypatch, tmp_path, example_tokamak):
    """Smoke test plotting the example machine, limiter, and wall."""
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    example_tokamak.plot(axis=ax, show=False)
    ax.plot(example_tokamak.limiter.R, example_tokamak.limiter.Z)
    ax.plot(example_tokamak.wall.R, example_tokamak.wall.Z)
    plt.close(fig)
