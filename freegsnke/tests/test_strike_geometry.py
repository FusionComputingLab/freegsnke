from types import SimpleNamespace

import numpy as np

from freegsnke import shape_descriptors


def test_find_strikepoints_combines_single_and_multiple_intersections(monkeypatch):
    contour = SimpleNamespace(
        allsegs=[
            [
                np.array([[1.0, -1.0], [1.0, 1.0]]),
                np.array([[2.0, -1.0], [4.0, 1.0], [6.0, -1.0]]),
            ]
        ]
    )
    monkeypatch.setattr(
        shape_descriptors, "_boundary_contour_lines", lambda *args: contour.allsegs[0]
    )

    strikes = shape_descriptors.find_strikepoints(
        np.zeros((2, 2)),
        np.zeros((2, 2)),
        np.zeros((2, 2)),
        0.0,
        np.array([[0.0, 0.0], [10.0, 0.0]]),
    )

    assert strikes.shape == (3, 2)
    np.testing.assert_allclose(
        strikes[np.argsort(strikes[:, 0])],
        np.array([[1.0, 0.0], [3.0, 0.0], [5.0, 0.0]]),
    )


def test_find_leg_strikepoint_follows_requested_xpoint_branch(monkeypatch):
    contour = SimpleNamespace(
        allsegs=[
            [
                # One connected separatrix line, ordered from the inner wall
                # through the lower X-point to the outer wall.
                np.array(
                    [
                        [0.30, -2.00],
                        [0.43, -1.32],
                        [0.50, -1.20],
                        [0.60, -1.32],
                        [1.40, -2.00],
                    ]
                ),
                # A nearby closed core contour must not be mistaken for a leg.
                np.array(
                    [
                        [0.48, -1.18],
                        [0.55, -1.12],
                        [0.62, -1.18],
                        [0.55, -1.24],
                        [0.48, -1.18],
                    ]
                ),
                # An unrelated contour can intersect the wall but is not
                # connected to the supplied X-point.
                np.array([[1.60, -1.80], [1.70, -2.00]]),
            ]
        ]
    )
    monkeypatch.setattr(
        shape_descriptors, "_boundary_contour_lines", lambda *args: contour.allsegs[0]
    )

    radial = np.linspace(0.0, 2.0, 21)
    vertical = np.linspace(-2.0, 0.0, 21)
    R, Z = np.meshgrid(radial, vertical, indexing="ij")
    wall = np.array([[0.0, -2.0], [2.0, -2.0]])

    outer = shape_descriptors.find_leg_strikepoint(
        R, Z, np.zeros_like(R), 0.0, wall, [0.50, -1.20], leg="outer"
    )
    inner = shape_descriptors.find_leg_strikepoint(
        R, Z, np.zeros_like(R), 0.0, wall, [0.50, -1.20], leg="inner"
    )

    np.testing.assert_allclose(outer, [1.40, -2.00])
    np.testing.assert_allclose(inner, [0.30, -2.00])


def test_find_leg_strikepoint_returns_first_contact_along_leg(monkeypatch):
    contour = SimpleNamespace(
        allsegs=[
            [
                np.array(
                    [
                        [0.30, -2.00],
                        [0.50, -1.20],
                        [0.80, -1.80],
                        [1.00, -1.60],
                        [1.40, -2.00],
                    ]
                )
            ]
        ]
    )
    monkeypatch.setattr(
        shape_descriptors, "_boundary_contour_lines", lambda *args: contour.allsegs[0]
    )

    radial = np.linspace(0.0, 2.0, 21)
    vertical = np.linspace(-2.0, 0.0, 21)
    R, Z = np.meshgrid(radial, vertical, indexing="ij")
    wall = np.array([[0.0, -1.70], [2.0, -1.70]])

    outer = shape_descriptors.find_leg_strikepoint(
        R, Z, np.zeros_like(R), 0.0, wall, [0.50, -1.20], leg="outer"
    )

    # The outer leg crosses Z=-1.7 three times; the first crossing is on the
    # segment from (0.5, -1.2) to (0.8, -1.8).
    np.testing.assert_allclose(outer, [0.75, -1.70])


def test_find_leg_strikepoint_handles_saddle_split_across_contours(monkeypatch):
    contour = SimpleNamespace(
        allsegs=[
            [
                # Marching-squares contouring can put the inner and outer
                # legs in separate open polylines near the same X-point.
                np.array(
                    [
                        [0.30, -2.00],
                        [0.43, -1.32],
                        [0.49, -1.21],
                        [0.40, -0.80],
                    ]
                ),
                np.array(
                    [
                        [0.90, -0.80],
                        [0.56, -1.21],
                        [0.63, -1.35],
                        [1.40, -2.00],
                    ]
                ),
            ]
        ]
    )
    monkeypatch.setattr(
        shape_descriptors, "_boundary_contour_lines", lambda *args: contour.allsegs[0]
    )

    radial = np.linspace(0.0, 2.0, 21)
    vertical = np.linspace(-2.0, 0.0, 21)
    R, Z = np.meshgrid(radial, vertical, indexing="ij")
    wall = np.array([[0.0, -2.0], [2.0, -2.0]])

    outer = shape_descriptors.find_leg_strikepoint(
        R, Z, np.zeros_like(R), 0.0, wall, [0.50, -1.20], leg="outer"
    )
    inner = shape_descriptors.find_leg_strikepoint(
        R, Z, np.zeros_like(R), 0.0, wall, [0.50, -1.20], leg="inner"
    )

    np.testing.assert_allclose(outer, [1.40, -2.00])
    np.testing.assert_allclose(inner, [0.30, -2.00])


def test_find_leg_strikepoint_handles_same_side_saddle_probes(monkeypatch):
    contour = SimpleNamespace(
        allsegs=[
            [
                np.array(
                    [
                        # The long direction reaches the inner target after
                        # wrapping around the plasma boundary.
                        [0.30, -2.00],
                        [0.90, 0.20],
                        [0.68, -1.05],
                        # The discretized contour misses the supplied saddle
                        # to its outboard side. Both local probes therefore
                        # have positive dR and cannot classify the two legs.
                        [0.62, -1.18],
                        [0.60, -1.28],
                        # The short direction is the physical outer leg.
                        [1.40, -2.00],
                    ]
                )
            ]
        ]
    )
    monkeypatch.setattr(
        shape_descriptors, "_boundary_contour_lines", lambda *args: contour.allsegs[0]
    )

    radial = np.linspace(0.0, 2.0, 21)
    vertical = np.linspace(-2.0, 0.0, 21)
    R, Z = np.meshgrid(radial, vertical, indexing="ij")
    wall = np.array([[0.0, -2.0], [2.0, -2.0]])

    outer = shape_descriptors.find_leg_strikepoint(
        R, Z, np.zeros_like(R), 0.0, wall, [0.50, -1.20], leg="outer"
    )
    inner = shape_descriptors.find_leg_strikepoint(
        R, Z, np.zeros_like(R), 0.0, wall, [0.50, -1.20], leg="inner"
    )

    np.testing.assert_allclose(outer, [1.40, -2.00])
    np.testing.assert_allclose(inner, [0.30, -2.00])
