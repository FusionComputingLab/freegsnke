import pytest

import numpy as np
import freegs
from freegs.plotting import plotConstraints
from freegs.critical import find_critical
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import sys
import matplotlib.pyplot as plt
from copy import deepcopy
from IPython.display import display, clear_output
import time


@pytest.fixture()
def create_machine():
    # Create the machine, which specifies coil locations
    # and equilibrium, specifying the domain to solve over
    # this has to be either
    # freegs.machine.MASTU(), in which case:
    #tokamak = freegs.machine.MASTU()
    # or
    # MASTU_coils.MASTU_wpass()
    from freegsnke import MASTU_coils
    tokamak = MASTU_coils.MASTU_wpass()


    # Creates equilibrium object and initializes it with 
    # a "good" solution
    # plasma_psi = np.loadtxt('plasma_psi_example.txt')
    eq = freegs.Equilibrium(tokamak=tokamak,
                            #domains can be changed 
                            Rmin=0.1, Rmax=2.0,    # Radial domain
                            Zmin=-2.2, Zmax=2.2,   # Height range
                            #grid resolution can be changed
                            nx=65, ny=129, # Number of grid points
                            # psi=plasma_psi[::2,:])   
                            )  

    # Sets desired plasma properties for the 'starting equilibrium'
    # values can be changed
    from freegsnke.jtor_update import ConstrainPaxisIp
    profiles = ConstrainPaxisIp(8.1e3, # Plasma pressure on axis [Pascals]
                                6.2e5, # Plasma current [Amps]
                                0.5, # vacuum f = R*Bt
                                alpha_m = 1.8,
                                alpha_n = 1.2)


    # Sets some shape constraints (here very close to those used for initialization)
    Rx = 0.6
    Zx = 1.1

    Rmid = 1.4   # Outboard midplane
    Rin = 0.4  # Inboard midplane

    xpoints = [(Rx, -Zx-.01),   # (R,Z) locations of X-points
            (Rx,  Zx)]
    isoflux = [
            (Rx,Zx, Rx,-Zx),
            (Rmid, 0, Rin, 0.0),
            (Rmid,0, Rx,Zx),
        
            # Link inner and outer midplane locations
            (Rx, Zx, .85, 1.7),
            (Rx, Zx, .75, 1.6),
            (Rx, Zx, Rin, 0.2),
            (Rx, Zx, Rin, 0.1),
            (Rx,-Zx, Rin, -0.1),
            (Rx,-Zx, Rin, -0.2),
            (Rx,-Zx, .85, -1.7),
            (Rx,-Zx, .75, -1.6),

            (Rx,-Zx, 0.45, -1.8),
            (Rx, Zx, 0.45,  1.8),
            ]

    eq.tokamak['P6'].current = 0
    eq.tokamak['P6'].control = False
    eq.tokamak['Solenoid'].control = False

    constrain = freegs.control.constrain(xpoints=xpoints, 
                                        gamma=5e-6, 
                                        isoflux=isoflux
                                        )
    constrain(eq)

    return eq, profiles, constrain

def create_test_files_static_solve(create_machine):
    """Saves the control currents and psi map needed for testing the static solver. This should not be run every test, just if there is a major change that changes the machine. 

    Parameters
    ----------
    create_machine : pytest.fixture
        the equilibirum, profiles and constrain object to generate the test set from. 
    """
    eq, profiles, constrain = create_machine


    from freegsnke import newtonkrylov
    NK = newtonkrylov.NewtonKrylov(eq)

    eq.tokamak['P6'].current = 0
    eq.tokamak['P6'].control = False
    eq.tokamak['Solenoid'].control = False
    eq.tokamak['Solenoid'].current = 15000

    freegs.solve(eq,          # The equilibrium to adjust
                profiles,    # The plasma profiles
                constrain,   # Plasma control constraints
                show=False,
                rtol=3e-3) 

    controlCurrents = np.load("./freegsnke/tests/test_controlCurrents.npy")
    eq.tokamak.setControlCurrents(controlCurrents)


    NK.solve(eq, profiles, rel_convergence=1e-8)

    test_psi = np.load("./freegsnke/tests/test_psi.npy")


def test_static_solve(create_machine):
    """Tests the implementation of the static solver. 

    Parameters
    ----------    
    create_machine : pytest.fixture
        the equilibirum, profiles and constrain object to generate the test set from. 
    """
    eq, profiles, constrain = create_machine


    from freegsnke import newtonkrylov
    NK = newtonkrylov.NewtonKrylov(eq)

    eq.tokamak['P6'].current = 0
    eq.tokamak['P6'].control = False
    eq.tokamak['Solenoid'].control = False
    eq.tokamak['Solenoid'].current = 15000

    freegs.solve(eq,          # The equilibrium to adjust
                profiles,    # The plasma profiles
                constrain,   # Plasma control constraints
                show=False,
                rtol=3e-3) 

    controlCurrents = np.load("./freegsnke/tests/test_controlCurrents.npy")
    eq.tokamak.setControlCurrents(controlCurrents)


    NK.solve(eq, profiles, rel_convergence=1e-8)

    test_psi = np.load("./freegsnke/tests/test_psi.npy")

    assert np.allclose(eq.psi(), test_psi), "Psi map doesn't match the test map"