import pickle
from pathlib import Path

import numpy as np
import pytest

from freegsnke.mastu_tools import passive_currents_from_efit
from freegsnke.refine_passive import (
    generate_refinement,
    polygon_area,
    subdivide_passive_polygon,
    subdivide_passive_polygons,
)

MASTU_CONFIG_DIR = Path(__file__).resolve().parents[2] / "machine_configs" / "MAST-U"


def _rectangle_passive():
    return {
        "R": [0.0, 0.0, 2.0, 2.0],
        "Z": [0.0, 1.0, 1.0, 0.0],
        "name": "wall",
        "element": "vessel",
        "efitGroup": "V",
        "resistivity": 7.1e-7,
        "current_multiplier": 0.25,
    }


def test_passive_polygon_subdivision_conserves_area_and_current_multiplier():
    parent = _rectangle_passive()

    children = subdivide_passive_polygon(parent, max_edge_length=0.5)

    assert len(children) == 8
    assert {tuple(child["subdivision_shape"]) for child in children} == {(4, 2)}
    assert children[0]["parent_name"] == "wall"
    assert children[0]["parent_index"] is None
    assert children[0]["element"] == "vessel"
    assert children[0]["efitGroup"] == "V"
    assert children[0]["name"] == "wall__0"

    parent_area = polygon_area(parent["R"], parent["Z"])
    child_areas = [polygon_area(child["R"], child["Z"]) for child in children]
    assert np.isclose(sum(child_areas), parent_area)
    assert np.allclose(child_areas, parent_area / len(children))
    assert np.isclose(
        sum(child["current_multiplier"] for child in children),
        parent["current_multiplier"],
    )
    assert np.isclose(sum(child["area_fraction"] for child in children), 1.0)


def test_passive_polygon_subdivision_uses_area_limit_as_guardrail():
    parent = _rectangle_passive()

    children = subdivide_passive_polygon(parent, max_area=0.25)

    assert len(children) == 8
    assert max(polygon_area(child["R"], child["Z"]) for child in children) <= 0.25


def test_passive_polygon_subdivision_handles_skewed_quadrilateral():
    parent = {
        "R": [0.0, 0.35, 2.1, 1.7],
        "Z": [0.0, 0.9, 1.05, -0.1],
        "name": "skewed",
        "resistivity": 7.1e-7,
        "current_multiplier": 1.0,
    }

    children = subdivide_passive_polygon(parent, max_edge_length=0.6, max_area=0.2)

    assert len(children) > 1
    assert np.isclose(
        sum(polygon_area(child["R"], child["Z"]) for child in children),
        polygon_area(parent["R"], parent["Z"]),
    )
    assert np.isclose(sum(child["current_multiplier"] for child in children), 1.0)


def test_subdivide_passive_polygons_copies_single_filaments_unchanged():
    filament = {
        "R": 1.0,
        "Z": 0.0,
        "dR": 0.1,
        "dZ": 0.2,
        "name": "filament",
        "resistivity": 7.1e-7,
    }
    polygon = _rectangle_passive()

    children = subdivide_passive_polygons([filament, polygon], max_edge_length=1.0)

    assert children[0] == filament
    assert children[1]["parent_index"] == 1
    assert len(children) == 1 + 2


def test_subdivide_passive_polygon_rejects_non_quadrilateral_polygons():
    with pytest.raises(ValueError, match="four-vertex"):
        subdivide_passive_polygon(
            {
                "R": [0.0, 1.0, 0.0],
                "Z": [0.0, 0.0, 1.0],
                "name": "triangle",
                "resistivity": 7.1e-7,
            },
            max_edge_length=0.5,
        )


def test_subdivide_passive_polygons_rejects_non_quadrilateral_by_default():
    with pytest.raises(ValueError, match="four-vertex"):
        subdivide_passive_polygons(
            [
                _rectangle_passive(),
                {
                    "R": [0.0, 1.0, 1.2, 0.5, 0.0],
                    "Z": [0.0, 0.0, 0.8, 1.1, 0.6],
                    "name": "pentagon",
                    "resistivity": 7.1e-7,
                },
            ],
            max_edge_length=0.5,
        )


def test_subdivide_passive_polygons_can_keep_non_quadrilaterals():
    pentagon = {
        "R": [0.0, 1.0, 1.2, 0.5, 0.0],
        "Z": [0.0, 0.0, 0.8, 1.1, 0.6],
        "name": "pentagon",
        "resistivity": 7.1e-7,
    }

    children = subdivide_passive_polygons(
        [_rectangle_passive(), pentagon],
        max_edge_length=0.5,
        non_quadrilateral="keep",
    )

    assert len(children) == 9
    assert children[-1] == pentagon


def test_gauss_quadrature_refinement_returns_normalized_area_weights():
    R = [0.0, 0.0, 2.0, 2.0]
    Z = [0.0, 1.0, 1.0, 0.0]

    filaments, area, weights = generate_refinement(
        R, Z, n_refine=8, refine_mode="GQ", return_weights=True
    )

    assert len(filaments) >= 8
    assert np.isclose(area, 2.0)
    assert np.isclose(np.sum(weights), 1.0)
    assert np.isclose(np.sum(weights * filaments[:, 0]), 1.0)
    assert np.isclose(np.sum(weights * filaments[:, 1]), 0.5)


def test_mastu_like_passive_currents_are_preserved_after_subdivision():
    with open(MASTU_CONFIG_DIR / "MAST-U_like_passive_coils.pickle", "rb") as file:
        passive_coils = pickle.load(file)

    refined_passive_coils = subdivide_passive_polygons(
        passive_coils,
        max_edge_length=0.1,
    )

    group_labels = sorted(
        {
            coil["efitGroup"] if "efitGroup" in coil else coil["element"]
            for coil in passive_coils
        }
    )
    current_labels = np.asarray(group_labels)
    currents_values = np.vstack(
        [
            np.linspace(1.0e3, 5.0e3, len(group_labels)),
            np.linspace(-2.0e3, 3.0e3, len(group_labels)),
            np.linspace(7.0e3, -1.0e3, len(group_labels)),
        ]
    )

    original_currents = passive_currents_from_efit(
        passive_coils,
        current_labels,
        currents_values,
    )
    refined_currents = passive_currents_from_efit(
        refined_passive_coils,
        current_labels,
        currents_values,
    )

    regrouped_refined_currents = {
        coil["name"]: np.zeros(currents_values.shape[0]) for coil in passive_coils
    }
    for coil in refined_passive_coils:
        parent_name = coil.get("parent_name", coil["name"])
        regrouped_refined_currents[parent_name] += refined_currents[coil["name"]]

    for coil in passive_coils:
        assert np.allclose(
            regrouped_refined_currents[coil["name"]],
            original_currents[coil["name"]],
            rtol=0.0,
            atol=1.0e-10,
        )


def test_mastu_like_passive_polygons_are_all_quadrilaterals():
    with open(MASTU_CONFIG_DIR / "MAST-U_like_passive_coils.pickle", "rb") as file:
        passive_coils = pickle.load(file)

    assert all(len(coil["R"]) == 4 and len(coil["Z"]) == 4 for coil in passive_coils)
