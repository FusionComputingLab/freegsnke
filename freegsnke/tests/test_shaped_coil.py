import numpy as np

from freegsnke.shaped_coil import ShapedCoil


def test_area():
    coil1 = ShapedCoil(
        [(0.95, 0.1), (0.95, 0.2), (1.05, 0.2), (1.05, 0.1)], current=100.0
    )
    assert np.isclose(coil1.area, 0.01)


def test_move_R():
    """
    Changing major radius property R changes all filament locations
    """

    dR = 0.6
    coil1 = ShapedCoil(
        [(0.95, 0.1), (0.95, 0.2), (1.05, 0.2), (1.05, 0.1)], current=100.0
    )
    coil2 = ShapedCoil(
        [
            (0.95 + dR, 0.1),
            (0.95 + dR, 0.2),
            (1.05 + dR, 0.2),
            (1.05 + dR, 0.1),
        ],
        current=100.0,
    )

    # change the copy by a different amount to ensure changing the copy
    # does not impact the original coil
    coil1_copy = coil1.copy()
    coil1_copy.R += dR + 0.4

    # Shift coil1 to same location as coil2
    coil1.R += dR

    # run the test twice: once on the original coil and once on a copy
    # to check both produce the same results
    for _ in range(2):
        assert np.isclose(coil1.controlPsi(0.4, 0.5), coil2.controlPsi(0.4, 0.5))

        assert np.isclose(coil1.controlBr(0.3, -0.2), coil2.controlBr(0.3, -0.2))

        assert np.isclose(coil1.controlBz(1.75, 1.2), coil2.controlBz(1.75, 1.2))

        coil1 = coil1.copy()
        coil2 = coil2.copy()


def test_move_Z():
    """
    Changing height property Z changes all filament locations
    """

    dZ = 0.4
    coil1 = ShapedCoil(
        [(0.95, 0.1), (0.95, 0.2), (1.05, 0.2), (1.05, 0.1)], current=100.0
    )
    coil2 = ShapedCoil(
        [
            (0.95, 0.1 + dZ),
            (0.95, 0.2 + dZ),
            (1.05, 0.2 + dZ),
            (1.05, 0.1 + dZ),
        ],
        current=100.0,
    )

    # Shift coil1 to same location as coil2
    coil1.Z += dZ

    assert np.isclose(coil1.controlPsi(0.4, 0.5), coil2.controlPsi(0.4, 0.5))

    assert np.isclose(coil1.controlBr(0.3, -0.2), coil2.controlBr(0.3, -0.2))

    assert np.isclose(coil1.controlBz(1.75, 1.2), coil2.controlBz(1.75, 1.2))
