import pytest
import freegs
import numpy as np
from freegsnke import MASTU_coils
from freegsnke.plasma_grids import define_reduced_plasma_grid, make_layer_mask, Myy, Mey

@pytest.fixture
def eq():
    tokamak = MASTU_coils.MASTU_wpass()


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

def test_define_reduced_plasma_grid(eq):
    """Tests if the shape of the limiter mask is correct and if the points returned are unique

    Parameters
    ----------
    eq : freegs.Equilibrium
        An equilbrium object
    """
    plasma_pts, mask_inside_limiter = define_reduced_plasma_grid(eq.R, eq.Z)

    assert mask_inside_limiter.shape == eq.R.shape, "The shape of the limiter  mask is incorrect"
    assert len(np.unique(plasma_pts, axis=0)) == len(plasma_pts), f"There are {len(np.nunique(plasma_pts, axis=0))} unique points out of {len(plasma_pts)}"

def test_make_layer_mask(eq):
    """Tests if the shape of the layer mask is the correct shape and does not overlap with the limiter mask

    Parameters
    ----------
    eq : freegs.Equilibrium
        An equilbrium object
    """
    plasma_pts, mask_inside_limiter = define_reduced_plasma_grid(eq.R, eq.Z)
    layer_mask = make_layer_mask(mask_inside_limiter)

    assert layer_mask.shape == mask_inside_limiter.shape, "Layer mask is not the correct shape"
    assert np.sum(layer_mask*mask_inside_limiter) == 0, "Layer mask and limiter mask are overlapping"

def test_Myy(eq):
    """Tests if the shape of the mutual inductance matrix is correct. The mutual inductance matrix of the plasma on itself should be symmetrical and postive definite.

    Parameters
    ----------
    eq : freegs.Equilibrium
        An equilbrium object
    """
    plasma_pts, mask_inside_limiter = define_reduced_plasma_grid(eq.R, eq.Z)

    Myy_ = Myy(plasma_pts)

    assert Myy_.shape == (len(plasma_pts), len(plasma_pts)), f"Shape of Myy not correct, shape of Myy: {Myy_.shape}, number of plasma point: {len(plasma_pts)}"
    assert np.all(Myy_ == Myy_.T), "Myy not symmetric"
    assert np.all(np.linalg.eigvals(Myy_ > 0)), "Myy not positive definite"

def test_Mey(eq):
    """Tests if the shape of the mutual inductance matrix of the plasma gridpoints and all vessel coils is correct. 

    Parameters
    ----------
    eq : freegs.Equilibrium
        An equilbrium object
    """
    plasma_pts, mask_inside_limiter = define_reduced_plasma_grid(eq.R, eq.Z)

    Mey_ = Mey(plasma_pts)

    assert Mey_.shape == (len(MASTU_coils.coils_dict.keys()), len(plasma_pts)), f"Shape of Myy not correct, shape of Myy: {Mey_.shape}, number of plasma point: {len(MASTU_coils.coils_dict.keys()), len(plasma_pts)}"
    
