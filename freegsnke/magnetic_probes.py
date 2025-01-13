"""
Class to implement magnetic probes (flux loops and pick ups at the moment):
- sets up probe object, containing the types and locations of the probes
- methods to extract the 'measurements' by each probe from an equilibrium.

Copyright 2025 UKAEA, UKRI-STFC, and The Authors, as per the COPYRIGHT and README files.

This file is part of FreeGSNKE.

FreeGSNKE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

FreeGSNKE is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
  
You should have received a copy of the GNU Lesser General Public License
along with FreeGSNKE.  If not, see <http://www.gnu.org/licenses/>.   
"""

import os
import pickle

import numpy as np
from deepdiff import DeepDiff
from freegs4e.gradshafranov import Greens, GreensBr, GreensBz


class Probes:
    """
    Class to implement magnetic probes:
    - flux loops: compute psi(R,Z)
    - pickup coils: compute B(R,phi,Z).nhat (nhat is unit vector orientation of the probe)

    Inputs:
    - equilibrium object - contains grid, plasma profile, plasma and coil currents, coil positions.
    N.B:- in init/setup the eq object only provides machine /domain/position information
        - in methods the equilibrium provides currents and other aspects that evolve in solve().

    Attributes:
    - floops,pickups = dictionaries with name, position, orientation of the probes
    - floops_positions etc.  = extract individual arrays of positions, orientations etc.
    - floops_order / pickups_order = list of fluxloop/pickups names, if individual probe value is required
    - greens_psi_coils_floops = greens functions for psi, evaluated at flux loop positions
    - greens_br/bz_coils_pickups = greens functions for Br/Bz, evaluated at pickup coil positions
    - greens_psi_plasma_floops = dictionary of greens functions for psi from plasma current, evaluated at flux loop positions
    - greens_BrBz_plasma_pickups = dictionary of greens functions for Br and Bz from plasma, evaluated at pickup coil positions

    - more greens function attributes would be added if new probes area added.

    Methods:
    - get_coil_currents(eq): returns current values in all the coils from equilibrium object.
    - get_plasma_current(eq): returns toroidal current values at each plasma grid point, taken from equilibrium input.
    - create_greens_all_coils(): returns array with greens function for each coil and probe combination.
    - psi_all_coils(eq): returns list of psi values at each flux loop position, summed over all coils.
    - psi_from_plasma(eq): returns list of psi values at each flux loop position from plasma itself.
    - create_greens_B_oriented_plasma(eq) : creates oriented greens functions for pickup coils.
    - calculate_fluxloop_value(eq): returns total flux at each probe position (sum of previous two outputs).
    - calculate_pickup_value(eq): returns pickup values at each probe position.


    - Br(eq)/ Bz(eq) : computes radial/z component of magnetic field, sum of coil and plasma contributions
    - Btor(eq) : extracts toroidal magnetic field (outside of plasma), evaluated at

    Methods currently have floop or pickup positions as default, but these can be changed with optional argument.
    """

    def __init__(self, coils_dict):
        """
        Initialise the following
        - read the probe dictionaries from file
        """
        # extract probe dictionaries, and set variables for each probe type
        try:
            probe_path = os.environ.get("PROBE_PATH", None)
            if probe_path is None:
                raise ValueError("PROBE_PATH environment variable not set.")

            with open(probe_path, "rb") as file:
                probe_dict = pickle.load(file)

            # set coil lists and probe lists
            self.floops = probe_dict["flux_loops"]
            self.pickups = probe_dict["pickups"]
        except ValueError:
            print("No probe configuration is provided")

        # store coil info
        self.coil_names = [name for name in coils_dict]
        self.coils_dict = coils_dict

    def initialise_setup(self, eq):
        """
        Setup attributes: positions, orientations and greens functions
        - set probe positions and orientations
        - set coil positions from equilibrium object
        - create arrays/dictionaries of greens functions with positions of coils/plasma currents and probes
        """

        check = DeepDiff(eq.tokamak.coils_dict, self.coils_dict) == {}
        if check is not True:
            raise AssertionError(
                "The supplied equilibrium uses a different tokamak. Probes values can not be computed."
            )

        eq_key = self.create_eq_key(eq)

        # self.limiter_handler = {}
        # self.limiter_handler[eq_key] = eq.limiter_handler

        # FLUX LOOPS
        # positions, number of probes, ordering
        self.floop_pos = np.array([probe["position"] for probe in self.floops])
        self.number_floops = np.shape(self.floop_pos)[0]  # number of probes
        self.floop_order = [probe["name"] for probe in self.floops]

        # # Initilaise Greens functions Gpsi
        self.greens_psi_coils_floops = self.create_greens_psi_all_coils(eq, "floops")
        self.greens_psi_plasma_floops = {}
        self.greens_psi_plasma_floops[eq_key] = self.create_green_psi_plasma(
            eq, "floops"
        )

        # # PICKUP COILS
        # # Positions and orientations - 3d vectors of [R, theta, Z]
        self.pickup_pos = np.array([el["position"] for el in self.pickups])
        self.pickup_or = np.array([el["orientation_vector"] for el in self.pickups])
        self.number_pickups = np.shape(self.pickup_pos)[0]
        self.pickup_order = [probe["name"] for probe in self.pickups]

        # # Initialise greens functions for pickups
        self.greens_br_plasma_pickup, self.greens_bz_plasma_pickup = {}, {}
        self.greens_br_coils_pickup, self.greens_bz_coils_pickup = (
            self.greens_BrBz_all_coils(eq, "pickups")
        )
        # not initialised unless needed to save memory
        # (
        #     self.greens_br_plasma_pickup[eq_key],
        #     self.greens_bz_plasma_pickup[eq_key],
        # ) = self.create_greens_BrBz_plasma(eq, "pickups")

        self.greens_B_plasma_oriented = {}
        self.greens_B_plasma_oriented[eq_key] = self.create_greens_B_oriented_plasma(
            eq, "pickups"
        )
        self.greens_B_coils_oriented = self.create_greens_B_oriented_coils(
            eq, "pickups"
        )

        # Other probes - to add in future...

    """
    Things for all probes
    - coil current array
    - plasma current array
    - eq grid key 
    """

    def get_coil_currents(self, eq):
        """
        create list of coil currents from the equilibrium
        """
        array_of_coil_currents = np.zeros(len(self.coil_names))
        for i, label in enumerate(self.coil_names):
            array_of_coil_currents[i] = eq.tokamak[label].current

        # could use eq.tokamak.getcurrents() instead
        return array_of_coil_currents

    def get_plasma_current(self, eq):
        """
        From equilibirium object contains toroidal current distribution on the grid, reduced to the limiter region only.
        """
        return eq.limiter_handler.Iy_from_jtor(eq._profiles.jtor)

    def create_eq_key(self, eq):
        """
        Produces tuple (Rmin,Rmax,Zmin,Zmax,nx,ny) from equilibrium to access correct greens function.
        """
        nx, ny = len(eq.R_1D), len(eq.Z_1D)
        eq_key = (eq.Rmin, eq.Rmax, eq.Zmin, eq.Zmax, nx, ny)
        return eq_key

    """
    Things for flux loops
    """

    def create_greens_psi_single_coil(self, eq, coil_key, probe="floops"):
        """
        Create array of greens functions for given coil evaluate at all probe positions
        - pos_R and pos_Z are arrays of R,Z coordinates of the probes
        - defines array of greens for each filament at each probe.
        - multiplies by polarity and multiplier
        - then sums over filaments to return greens function for probes from a given coil
        - flux loops by default, can apply to other probes too with minor modification
        """
        if probe == "floops":
            pos_R = self.floop_pos[:, 0]
            pos_Z = self.floop_pos[:, 1]

        # pol = self.coils_dict[coil_key]["polarity"][np.newaxis, :]
        # mul = self.coils_dict[coil_key]["multiplier"][np.newaxis, :]

        # greens_filaments = Greens(
        #     self.coils_dict[coil_key]["coords"][0][np.newaxis, :],
        #     self.coils_dict[coil_key]["coords"][1][np.newaxis, :],
        #     pos_R[:, np.newaxis],
        #     pos_Z[:, np.newaxis],
        # )
        # greens_filaments *= pol
        # greens_filaments *= mul
        # greens_psi_coil = np.sum(greens_filaments, axis=1)
        greens_psi_coil = eq.tokamak[coil_key].controlPsi(pos_R, pos_Z)

        return greens_psi_coil

    def create_greens_psi_all_coils(self, eq, probe="floops"):
        """
        Create 2d array of greens functions for all coils and at all probe positions
        - array[i][j] is greens function for coil i evaluated at probe position j
        """

        array = np.array([]).reshape(0, self.number_floops)
        for key in self.coils_dict.keys():
            array = np.vstack(
                (array, self.create_greens_psi_single_coil(eq, key, probe))
            )
        return array

    def psi_floop_all_coils(self, eq, probe="floops"):
        """
        compute flux function summed over all coils.
        returns array of flux values at the positions of the floop probes by default.
        new probes can be used instead (just change which greens function is used)
        """
        array_of_coil_currents = self.get_coil_currents(eq)
        if probe == "floops":
            greens = self.greens_psi_coils_floops

        psi_from_all_coils = np.sum(
            greens * array_of_coil_currents[:, np.newaxis], axis=0
        )
        # self.floop_psi = psi_from_all_coils
        return psi_from_all_coils

    def create_green_psi_plasma(self, eq, probe="floops"):
        """
        Compute greens function at probes from the plasma currents .
        - plasma current source in grid from solve. grid points contained in eq object
        - evaluated on flux loops by default, can apply to other probes too with minor modification
        """

        if probe == "floops":
            pos_R = self.floop_pos[:, 0]
            pos_Z = self.floop_pos[:, 1]

        #   only on the limiter domain pts
        greens = Greens(
            eq.limiter_handler.plasma_pts[:, 0, np.newaxis],
            eq.limiter_handler.plasma_pts[:, 1, np.newaxis],
            pos_R[np.newaxis, :],
            pos_Z[np.newaxis, :],
        )

        return greens

    def psi_from_plasma(self, eq, probe="floops"):
        """
        Calculate flux function contribution from the plasma
        - returns array of flux values from plasma at position of floop probes
        - evaluated on flux loops by default, can apply to other probes too with minor modification
        """
        eq_key = self.create_eq_key(eq)
        plasma_current_distribution = self.get_plasma_current(eq)

        if probe == "floops":
            try:
                plasma_greens = self.greens_psi_plasma_floops[eq_key]
            except:
                #  add new greens functions to dictionary
                self.greens_psi_plasma_floops[eq_key] = self.create_green_psi_plasma(
                    eq, "floops"
                )
                print("new equilibrium grid - computed new greens functions")
                # use newly created dictionary element.
                plasma_greens = self.greens_psi_plasma_floops[eq_key]

        psi_from_plasma = np.sum(
            plasma_greens * plasma_current_distribution[:, np.newaxis], axis=0
        )
        return psi_from_plasma

    def calculate_fluxloop_value(self, eq):
        """
        total flux for all floop probes
        """
        return self.psi_floop_all_coils(eq) + self.psi_from_plasma(eq)

    """
    Things for pickup coils
    """

    def create_greens_BrBz_single_coil(self, eq, coil_key, probe="pickups"):
        """
        Create array of greens functions for given coil evaluate at all pickup positions
        - defines array of greens for each filament at each probe.
        - multiplies by polarity and multiplier
        - then sums over filaments to return greens function for probes from a given coil
        - evaluated on pickups by default, can apply to other probes too with minor modification
        """
        if probe == "pickups":
            pos_R = self.pickup_pos[:, 0]
            pos_Z = self.pickup_pos[:, 2]

        # pol = self.coils_dict[coil_key]["polarity"][np.newaxis, :]
        # mul = self.coils_dict[coil_key]["multiplier"][np.newaxis, :]
        # greens_br_filaments = GreensBr(
        #     self.coils_dict[coil_key]["coords"][0][np.newaxis, :],
        #     self.coils_dict[coil_key]["coords"][1][np.newaxis, :],
        #     pos_R[:, np.newaxis],
        #     pos_Z[:, np.newaxis],
        # )
        # greens_br_filaments *= pol
        # greens_br_filaments *= mul
        # greens_br_coil = np.sum(greens_br_filaments, axis=1)

        # greens_bz_filaments = GreensBz(
        #     self.coils_dict[coil_key]["coords"][0][np.newaxis, :],
        #     self.coils_dict[coil_key]["coords"][1][np.newaxis, :],
        #     pos_R[:, np.newaxis],
        #     pos_Z[:, np.newaxis],
        # )

        # greens_bz_filaments *= pol
        # greens_bz_filaments *= mul
        # greens_bz_coil = np.sum(greens_bz_filaments, axis=1)

        greens_br_coil = eq.tokamak[coil_key].controlBr(pos_R, pos_Z)
        greens_bz_coil = eq.tokamak[coil_key].controlBz(pos_R, pos_Z)

        return greens_br_coil, greens_bz_coil

    def greens_BrBz_all_coils(self, eq, probe="pickups"):
        """
        Create 2d array of greens functions for all coils and at all probe positions
        - array[i][j] is greens function for coil i evaluated at probe position j
        - evaluated on pickups by default, can apply to other probes too with minor modification
        """
        if probe == "pickups":
            array_r = np.array([]).reshape(0, self.number_pickups)
            array_z = np.array([]).reshape(0, self.number_pickups)

        for key in self.coils_dict.keys():
            vals = self.create_greens_BrBz_single_coil(eq, key, probe)
            array_r = np.vstack((array_r, vals[0]))
            array_z = np.vstack((array_z, vals[1]))

        return array_r, array_z

    def create_greens_B_oriented_coils(self, eq, probe="pickups"):
        """
        perform dot product of greens function vector with pickup coil orientation
        """
        if probe == "pickups":
            or_R = self.pickup_or[:, 0]
            or_Z = self.pickup_or[:, 2]

        vals = self.greens_BrBz_all_coils(eq, probe)
        prod = vals[0] * or_R + vals[1] * or_Z

        return prod

    def BrBz_coils(self, eq, probe="pickups"):
        """
        Magnetic fields from coils, radial and z components only.
        evaluated on pickup positions by default.
        """
        coil_currents = self.get_coil_currents(eq)[:, np.newaxis]
        if probe == "pickups":
            br_coil = np.sum(self.greens_br_coils_pickup * coil_currents, axis=0)
            bz_coil = np.sum(self.greens_bz_coils_pickup * coil_currents, axis=0)
        return br_coil, bz_coil

    def create_greens_BrBz_plasma(self, eq, probe="pickups"):
        """
        Compute greens function at probes from the plasma currents .
        - plasma current source in grid from solve. grid points contained in eq object
        - evaluated on pickups by default, can apply to other probes too with minor modification
        """
        if probe == "pickups":
            pos_R = self.pickup_pos[:, 0]
            pos_Z = self.pickup_pos[:, 2]

        # rgrid = eq.R
        # zgrid = eq.Z

        greens_br = GreensBr(
            eq.limiter_handler.plasma_pts[:, 0, np.newaxis],
            eq.limiter_handler.plasma_pts[:, 1, np.newaxis],
            pos_R[np.newaxis, :],
            pos_Z[np.newaxis, :],
        )

        greens_bz = GreensBz(
            eq.limiter_handler.plasma_pts[:, 0, np.newaxis],
            eq.limiter_handler.plasma_pts[:, 1, np.newaxis],
            pos_R[np.newaxis, :],
            pos_Z[np.newaxis, :],
        )

        return greens_br, greens_bz

    def create_greens_B_oriented_plasma(self, eq, probe="pickups"):
        """
        perform dot product of greens function vector with orientation
        """
        br, bz = self.create_greens_BrBz_plasma(eq)

        or_R = self.pickup_or[:, 0]
        or_Z = self.pickup_or[:, 2]
        prod = br * or_R + bz * or_Z

        return prod

    def BrBz_plasma(self, eq, probe="pickups"):
        """
        Magnetic fields from plasma
        """
        eq_key = self.create_eq_key(eq)
        plasma_current = self.get_plasma_current(eq)[:, np.newaxis]

        try:
            greens_br = self.greens_br_plasma_pickup[eq_key]
            greens_bz = self.greens_bz_plasma_pickup[eq_key]
        except:
            (
                self.greens_br_plasma_pickup[eq_key],
                self.greens_bz_plasma_pickup[eq_key],
            ) = self.create_greens_BrBz_plasma(eq, "pickups")
            print("new equilibrium grid - computed new greens functions")
        if probe == "pickups":
            br_plasma = np.sum(greens_br * plasma_current, axis=(0, 1))
            bz_plasma = np.sum(greens_bz * plasma_current, axis=(0, 1))
        return br_plasma, bz_plasma

    def Br(self, eq, probe="pickups"):
        """
        Method to compute total radial magnetic field from coil and plasma
        returns array with Br at each pickup coil probe
        - evaluated on pickups by default, can apply to other probes too with minor modification
        """
        coil_currents = self.get_coil_currents(eq)[:, np.newaxis]
        plasma_current = self.get_plasma_current(eq)[:, np.newaxis]
        eq_key = self.create_eq_key(eq)

        if probe == "pickups":
            try:
                greens_pl = self.greens_br_plasma_pickup[eq_key]
            except:
                self.greens_br_plasma_pickup[eq_key] = self.create_greens_BrBz_plasma(
                    eq, "pickups"
                )[0]
                greens_pl = self.greens_br_plasma_pickup[eq_key]
                print("new equilibrium grid - computed new greens functions")
            br_coil = np.sum(self.greens_br_coils_pickup * coil_currents, axis=0)
            br_plasma = np.sum(greens_pl * plasma_current, axis=(0))
        return br_coil + br_plasma

    def Bz(self, eq, probe="pickups"):
        """
        Method to compute total z component of magnetic field from coil and plasma
        returns array with Bz at each pickup coil probe
        - evaluated on pickups by default, can apply to other probes too with minor modification
        """
        coil_currents = self.get_coil_currents(eq)[:, np.newaxis]
        plasma_current = self.get_plasma_current(eq)[:, np.newaxis]
        eq_key = self.create_eq_key(eq)

        if probe == "pickups":
            try:
                greens_pl = self.greens_bz_plasma_pickup[eq_key]
            except:
                self.greens_bz_plasma_pickup[eq_key] = self.create_greens_BrBz_plasma(
                    eq, "pickups"
                )[1]
                greens_pl = self.greens_bz_plasma_pickup[eq_key]
                print("new equilibrium grid - computed new greens functions")

            bz_coil = np.sum(self.greens_bz_coils_pickup * coil_currents, axis=0)
            bz_plasma = np.sum(greens_pl * plasma_current, axis=(0))
        return bz_coil + bz_plasma

    def Btor(self, eq, probe="pickups"):
        """
        Probes outside of plasma therefore Btor = fvac/R
        returns array of btor for each probe position
        - evaluated on pickups by default, can apply to other probes too with minor modification
        """
        if probe == "pickups":
            pos_R = self.pickup_pos[:, 0]

        btor = eq._profiles.fvac() / pos_R
        return btor

    # def calculate_pickup_value_v1(self,eq,probe = 'pickups'):
    #     """
    #     OLD VERSION
    #     Method to compute and return B.n, for pickup coils
    #     """
    #     orientation = self.pickup_or.transpose()
    #     Btotal = np.vstack((self.Br(eq),self.Btor(eq,probe),self.Bz(eq)))

    #     BdotN = np.sum(orientation*Btotal, axis = 0)

    #     return BdotN

    def calculate_pickup_value(self, eq, probe="pickups"):
        """
        Compute B.n at pickup probes, using oriented greens functions.
        """
        coil_current = self.get_coil_currents(eq)[:, np.newaxis]
        plasma_current = self.get_plasma_current(eq)[:, np.newaxis]
        eq_key = self.create_eq_key(eq)
        if probe == "pickups":
            try:
                greens_pl = self.greens_B_plasma_oriented[eq_key]
            except:
                #  add new greens functions to dictionary
                self.greens_B_plasma_oriented[eq_key] = (
                    self.create_greens_B_oriented_plasma(eq, "floops")
                )
                print("new equilibrium grid - computed new greens functions")
                # use newly created dictionary element.
                greens_pl = self.greens_B_plasma_oriented[eq_key]

            pickup_tor = self.Btor(eq, probe) * self.pickup_or[:, 1]
            pickup_pol_coil = np.sum(
                self.greens_B_coils_oriented * coil_current, axis=0
            )
            pickup_pol_pl = np.sum(greens_pl * plasma_current, axis=(0))

        return pickup_pol_coil + pickup_pol_pl + pickup_tor

    def plot(self, axis=None, show=True, floops=True, pickups=True, pickups_scale=0.05):
        """
        Plot the magnetic probes.

        axis     - Specify the axis on which to plot
        show     - Call matplotlib.pyplot.show() before returning
        floops   - Plot the fluxloops
        pickups  - Plot the pickup coils

        Returns
        -------

        axis  object from Matplotlib

        """
        from freegs4e.plotting import plotProbes

        return plotProbes(
            self,
            axis=axis,
            show=show,
            floops=floops,
            pickups=pickups,
            pickups_scale=pickups_scale,
        )
