import numpy as np

from freegsnke.bilinear_interpolation import biliint


def test_biliint_uniform_grid():
    r = np.linspace(0.5, 2.0, 17)
    z = np.linspace(-1.2, 0.8, 25)
    R, Z = np.meshgrid(r, z, indexing="ij")
    psi = 2.0 + 3.0 * R - 4.0 * Z + 5.0 * R * Z
    points = np.array(
        [
            [0.5, 2.0, 0.5, 2.0, 0.73, 1.41, 1.89],
            [-1.2, -1.2, 0.8, 0.8, -0.64, 0.17, 0.52],
        ]
    )

    expected = 2.0 + 3.0 * points[0] - 4.0 * points[1] + 5.0 * points[0] * points[1]
    assert np.allclose(biliint(R, Z, psi, points), expected)
