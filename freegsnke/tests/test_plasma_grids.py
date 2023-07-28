import os
import pytest
import freegs
import numpy as np

os.environ["ACTIVE_COILS_PATH"] = "./machine_configs/MAST-U/active_coils.pickle"
os.environ["PASSIVE_COILS_PATH"] = "./machine_configs/MAST-U/passive_coils.pickle"
os.environ["WALL_PATH"] = "./machine_configs/MAST-U/wall.pickle"

from freegsnke import build_machine
from freegsnke.plasma_grids import Grids

@pytest.fixture
def eq():
    tokamak = build_machine.tokamak()


    # Creates equilibrium object and initializes it with 
    # a "good" solution
    # plasma_psi = np.loadtxt('plasma_psi_example.txt')
    eq = freegs.Equilibrium(tokamak=tokamak,
                            #domains can be changed 
                            Rmin=0.1, Rmax=2.0,    # Radial domain
                            Zmin=-2.2, Zmax=2.2,   # Height range
                            #grid resolution can be changed
                            nx=129, ny=129, # Number of grid points
                            # psi=plasma_psi[::2,:])   
                            )  
    return eq


@pytest.fixture
def grids(eq):
    return Grids(eq, None)


def test_plasma_domain_mask(eq, grids):
    """Tests if the shape of the limiter mask is correct and if the points returned are unique

    Parameters
    ----------
    eq : freegs.Equilibrium
        An equilbrium object
    """
    assert grids.plasma_domain_mask.shape == eq.R.shape, "The shape of the limiter  mask is incorrect"
    assert len(np.unique(grids.plasma_pts, axis=0)) == len(grids.plasma_pts), f"There are {len(np.nunique(grids.plasma_pts, axis=0))} unique points out of {len(grids.plasma_pts)}"

def test_make_layer_mask(eq, grids):
    """Tests if the shape of the layer mask is the correct shape and does not overlap with the limiter mask

    Parameters
    ----------
    eq : freegs.Equilibrium
        An equilbrium object
    """
    assert grids.layer_mask.shape == grids.plasma_domain_mask.shape, "Layer mask is not the correct shape"
    assert np.sum(grids.layer_mask*grids.plasma_domain_mask) == 0, "Layer mask and limiter mask are overlapping"

def test_Myy(grids):
    """Tests if the shape of the mutual inductance matrix is correct. The mutual inductance matrix of the plasma on itself should be symmetrical and postive definite.

    Parameters
    ----------
    eq : freegs.Equilibrium
        An equilbrium object
    """
    Myy_ = grids.Myy()

    assert Myy_.shape == (len(grids.plasma_pts), len(grids.plasma_pts)), f"Shape of Myy not correct, shape of Myy: {Myy_.shape}, number of plasma point: {len(grids.plasma_pts)}"
    assert np.all(Myy_ == Myy_.T), "Myy not symmetric"
    assert np.all(np.linalg.eigvals(Myy_ > 0)), "Myy not positive definite"

def test_Mey(eq, grids):
    """Tests if the shape of the mutual inductance matrix of the plasma gridpoints and all vessel coils is correct. 

    Parameters
    ----------
    eq : freegs.Equilibrium
        An equilbrium object
    """

    Mey_ = grids.Mey()

    assert Mey_.shape == (len(eq.tokamak.coils), len(grids.plasma_pts)), f"Shape of Myy not correct, shape of Myy: {Mey_.shape}, number of plasma point: {len(eq.tokomak.coils), len(grids.plasma_pts)}"
    
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