import os

import freegs4e
import numpy as np
import pytest

os.environ["ACTIVE_COILS_PATH"] = "./machine_configs/test/active_coils.pickle"
os.environ["PASSIVE_COILS_PATH"] = "./machine_configs/test/passive_coils.pickle"
os.environ["WALL_PATH"] = "./machine_configs/test/wall.pickle"
os.environ["LIMITER_PATH"] = "./machine_configs/test/limiter.pickle"
os.environ["PROBE_PATH"] = "./machine_configs/test/magnetic_probes.pickle"

from freegsnke import build_machine, limiter_func


@pytest.fixture
def create_machine():
    tokamak = build_machine.tokamak()

    # Creates equilibrium object and initializes it with
    # a "good" solution
    # plasma_psi = np.loadtxt('plasma_psi_example.txt')
    eq = freegs4e.Equilibrium(
        tokamak=tokamak,
        # domains can be changed
        Rmin=0.1,
        Rmax=2.0,  # Radial domain
        Zmin=-2.2,
        Zmax=2.2,  # Height range
        # grid resolution can be changed
        nx=129,
        ny=129,  # Number of grid points
        # psi=plasma_psi[::2,:])
    )
    return tokamak, eq


@pytest.fixture
def plasma_domain_masks(create_machine):
    tokamak, eq = create_machine
    limiter_handler = limiter_func.Limiter_handler(eq, tokamak.limiter)
    limiter_handler.make_layer_mask(limiter_handler.mask_inside_limiter)
    return (
        limiter_handler.mask_inside_limiter,
        limiter_handler.layer_mask,
        limiter_handler.plasma_pts,
    )


def test_plasma_domain_mask(create_machine, plasma_domain_masks):
    """
    Tests if the shape of the limiter mask is correct and if the points
    returned are unique.
    """
    _, eq = create_machine
    mask_inside_limiter, layer_mask, plasma_pts = plasma_domain_masks
    assert (
        mask_inside_limiter.shape == eq.R.shape
    ), "The shape of the limiter  mask is incorrect"


def test_make_layer_mask(create_machine, plasma_domain_masks):
    """
    Tests if the shape of the layer mask is the correct shape and does not
    overlap with the limiter mask.
    """
    _, eq = create_machine
    mask_inside_limiter, layer_mask, plasma_pts = plasma_domain_masks
    assert (
        layer_mask.shape == mask_inside_limiter.shape
    ), "Layer mask is not the correct shape"
    assert (
        np.sum(layer_mask * mask_inside_limiter) == 0
    ), "Layer mask and limiter mask are overlapping"


# def test_Myy(grids):
#     """
#     Tests if the shape of the mutual inductance matrix is correct. The mutual
#     inductance matrix of the plasma on itself should be symmetrical and postive
#     definite.
#     """
#     Myy_ = grids.Myy()

#     assert Myy_.shape == (len(grids.plasma_pts), len(grids.plasma_pts)), (
#         f"Shape of Myy not correct, shape of Myy: {Myy_.shape}, number of"
#         + f" plasma point: {len(grids.plasma_pts)}"
#     )
#     assert np.all(Myy_ == Myy_.T), "Myy not symmetric"
#     assert np.all(np.linalg.eigvals(Myy_) > 0), "Myy not positive definite"


# def test_Mey(create_machine, grids):
#     """
#     Tests if the shape of the mutual inductance matrix of the plasma gridpoints
#     and all vessel coils is correct.
#     """
#     _, eq = create_machine

#     Mey_ = grids.Mey()

#     assert Mey_.shape == (len(eq.tokamak.coils), len(grids.plasma_pts)), (
#         f"Shape of Myy not correct, shape of Myy: {Mey_.shape}, number of"
#         + f" plasma point: {len(eq.tokomak.coils), len(grids.plasma_pts)}"
#     )


@pytest.mark.skip(reason="Not implemented yet")
def test_Iy_from_jtor():
    raise NotImplementedError


@pytest.mark.skip(reason="Not implemented yet")
def test_normalise_sum():
    raise NotImplementedError


@pytest.mark.skip(reason="Not implemented yet")
def test_hat_Iy_from_jtor():
    raise NotImplementedError


@pytest.mark.skip(reason="Not implemented yet")
def test_rebuild_map2d():
    raise NotImplementedError
