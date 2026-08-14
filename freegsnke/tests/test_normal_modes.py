"""Tests for passive-structure normal-mode transformations."""

import numpy as np

from freegsnke.normal_modes import mode_decomposition


def _mode_decomposition():
    """Return a small decomposition whose passive modes are not orthogonal."""

    resistances = np.array([0.8, 1.1, 1.6, 2.0])
    inductances = np.array(
        [
            [4.0, 0.2, 0.1, 0.0],
            [0.2, 3.2, 0.5, 0.1],
            [0.1, 0.5, 2.8, 0.4],
            [0.0, 0.1, 0.4, 2.5],
        ]
    )
    return mode_decomposition(
        coil_resist=resistances,
        coil_self_ind=inductances,
        n_coils=4,
        n_active_coils=1,
    )


def test_current_transform_uses_inverse_not_transpose():
    """The non-orthogonal mode basis requires its inverse for currents."""

    modes = _mode_decomposition()
    modal_currents = np.array([1.0, -0.4, 0.7, 0.2])
    physical_currents = modes.Pmatrix @ modal_currents

    assert not np.allclose(modes.Pmatrix.T, modes.Pmatrix_inverse)
    np.testing.assert_allclose(
        modes.Pmatrix_inverse @ physical_currents,
        modal_currents,
        rtol=1e-13,
        atol=1e-13,
    )


def test_mode_greens_reproduce_physical_current_flux():
    """Modal and reconstructed physical currents must produce the same flux."""

    modes = _mode_decomposition()
    rng = np.random.default_rng(42)
    physical_greens = rng.normal(size=(modes.n_coils, 5, 6))
    modal_currents = np.array([1.0, -0.4, 0.7, 0.2])

    physical_currents = modes.Pmatrix @ modal_currents
    flux_from_physical_currents = np.einsum(
        "i,ijk->jk", physical_currents, physical_greens
    )
    flux_from_modal_currents = np.einsum(
        "i,ijk->jk",
        modal_currents,
        modes.normal_modes_greens(physical_greens),
    )

    np.testing.assert_allclose(
        flux_from_modal_currents,
        flux_from_physical_currents,
        rtol=1e-13,
        atol=1e-13,
    )
