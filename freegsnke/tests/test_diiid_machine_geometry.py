"""Geometry invariants for the illustrative DIII-D machine description."""

import pickle
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Polygon

MACHINE_DIR = Path(__file__).resolve().parents[2] / "machine_configs" / "DIII-D"


@pytest.mark.parametrize("component", ["limiter", "wall"])
def test_diiid_boundary_is_closed_and_valid(component):
    """Boundary vertices must not retrace an edge or intersect themselves."""
    with (MACHINE_DIR / f"DIII-D_{component}.pickle").open("rb") as handle:
        boundary = pickle.load(handle)
    points = np.array([[point["R"], point["Z"]] for point in boundary])
    np.testing.assert_array_equal(points[0], points[-1])
    polygon = Polygon(points)
    assert polygon.is_valid
    assert polygon.area > 0


def test_diiid_passive_polygons_do_not_overlap():
    """Adjacent vessel pieces can touch but cannot double-count conducting area."""
    with (MACHINE_DIR / "DIII-D_passive_coils.pickle").open("rb") as handle:
        passives = pickle.load(handle)
    polygons = {
        part["name"]: Polygon(np.column_stack((part["R"], part["Z"])))
        for part in passives
    }
    assert len(polygons) == 24
    for name, polygon in polygons.items():
        assert polygon.is_valid and polygon.area > 0, name
    for (first, a), (second, b) in combinations(polygons.items(), 2):
        assert a.intersection(b).area < 1e-12, (first, second)
