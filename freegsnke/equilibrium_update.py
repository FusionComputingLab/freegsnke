"""
Defines the FreeGSNKE equilibrium Object, which inherits from the FreeGS4E equilibrium object.

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

import freegs4e.equilibrium
import numpy as np
from freegs4e import critical
from scipy import interpolate

from . import limiter_func
from .build_machine import copy_tokamak
from .copying import copy_into


class Equilibrium(freegs4e.equilibrium.Equilibrium):
    """FreeGS4E equilibrium class with optional initialization."""

    def __init__(self, *args, **kwargs):
        """Instantiates the object."""
        super().__init__(*args, **kwargs)

        self.equilibrium_path = os.environ.get("EQUILIBRIUM_PATH", None)
        if self.equilibrium_path is not None:
            self.initialize_from_equilibrium()

        # redefine interpolating function
        self.psi_func_interp = interpolate.RectBivariateSpline(
            self.R[:, 0], self.Z[0, :], self.plasma_psi
        )

        self.nxh = len(self.R) // 2
        self.nyh = len(self.Z[0]) // 2
        self.Rnxh = self.R[self.nxh, 0]
        self.Znyh = self.Z[0, self.nyh]

        # It's not a GS solution:
        self.solved = False

        # set up for limiter functionality
        self.limiter_handler = limiter_func.Limiter_handler(self, self.tokamak.limiter)
        self.mask_inside_limiter = 1.0 * self.limiter_handler.mask_inside_limiter
        # the factor 2 is needed by critical routines
        self.mask_outside_limiter = 2 * np.logical_not(self.mask_inside_limiter).astype(
            float
        )

    def create_auxiliary_equilibrium(self):
        """Creates the auxiliary equilibrium object.

        The auxiliary object returned from this method is essentially
        a copy of the equilibrium object (self) however it is manually
        setup and so won't contain all attributes on self (especially custom
        attributes). It is NOT _guaranteed_ to be the same as a deepcopy, or even
        a shallow copy.
        """
        # __new__ stops __init__ being called.
        # This is necessary because the __init__ method does expensive
        # calculations which we can just copy the results of
        equilibrium = Equilibrium.__new__(Equilibrium)

        # attributes that FreeGS4e sets
        equilibrium.tokamak = copy_tokamak(self.tokamak)
        equilibrium.Rmin = self.Rmin
        equilibrium.Rmax = self.Rmax
        equilibrium.Zmin = self.Zmin
        equilibrium.Zmax = self.Zmax
        equilibrium.nx = self.nx
        equilibrium.ny = self.ny
        equilibrium.dR = self.dR
        equilibrium.dZ = self.dZ
        equilibrium._applyBoundary = self._applyBoundary
        equilibrium._current = self._current
        equilibrium.order = self.order
        equilibrium._solver = self._solver

        # attributes the FreeGSNKE sets
        equilibrium.solved = self.solved
        equilibrium.psi_func_interp = self.psi_func_interp
        equilibrium.nxh = self.nxh
        equilibrium.nyh = self.nyh
        equilibrium.Rnxh = self.Rnxh
        equilibrium.Znyh = self.Znyh
        equilibrium.limiter_handler = self.limiter_handler  # should be safe not to copy

        # attributes that actually need to be copied
        equilibrium.R_1D = np.copy(self.R_1D)
        equilibrium.Z_1D = np.copy(self.Z_1D)
        equilibrium.R = np.copy(self.R)
        equilibrium.Z = np.copy(self.Z)
        equilibrium.tokamak_psi = np.copy(self.tokamak_psi)
        equilibrium.plasma_psi = np.copy(self.plasma_psi)
        equilibrium.mask_inside_limiter = np.copy(self.mask_inside_limiter)
        equilibrium.mask_outside_limiter = np.copy(self.mask_outside_limiter)
        equilibrium._pgreen = self._pgreen.copy()
        equilibrium._vgreen = self._vgreen.copy()
        copy_into(self, equilibrium, "current_vec", mutable=True, strict=False)

        copy_into(
            self, equilibrium, "opt", mutable=True, strict=False, allow_deepcopy=True
        )
        copy_into(
            self, equilibrium, "xpt", mutable=True, strict=False, allow_deepcopy=True
        )
        copy_into(self, equilibrium, "psi_bndry", strict=False)

        if hasattr(self, "_profiles"):
            equilibrium._profiles = self._profiles.copy()

        return equilibrium

    def adjust_psi_plasma(
        self,
    ):
        """Operates an initial rescaling of the psi_plasma guess so to ensure a viable O-point
        and at least an X-point within the domain.

        Only use after appropriate coil currents have been set as desired!
        """
        self.tokamak_psi = self.tokamak.calcPsiFromGreens(pgreen=self._pgreen)

        n_up = 0
        self.gmod = 0
        self.gexp = 2
        opoint_flag = False
        while (n_up < 10) and (opoint_flag == False):
            try:
                # Analyse the equilibrium, finding O- and X-points
                opt, xpt = critical.find_critical(
                    self.R,
                    self.Z,
                    self.tokamak_psi + self.plasma_psi,
                    self.mask_inside_limiter,
                    None,
                )
                opoint_flag = True
            except:
                self.plasma_psi *= 1.5
                self.gmod += np.log(1.5)
                n_up += 1
        if opoint_flag == False:
            print("O-point could not be generated by simply scaling up psi_plasma.")
            print("Manual initialization advised.")
            return

        # O-point is in place
        xpoint_flag = len(xpt) > 0
        print("O-point is in place. Flag for X-point in place =", xpoint_flag)
        n_plasma_psi = self.plasma_psi.copy()
        n_exp = 0
        if (xpoint_flag == False) and (n_exp < 10):
            # if it didn't work, try by making psi more compact
            psi_max = np.amax(self.plasma_psi)
            e_plasma_psi = self.plasma_psi / psi_max
            while (xpoint_flag == False) and (n_exp < 10):
                n_exp += 1
                n_plasma_psi = psi_max * e_plasma_psi ** (n_exp * 1.1)
                try:
                    opt, xpt = critical.find_critical(
                        self.R,
                        self.Z,
                        self.tokamak_psi + n_plasma_psi,
                        self.mask_inside_limiter,
                        None,
                    )
                    xpoint_flag = len(xpt) > 0
                    self.gmod *= 1.1
                except:
                    # here if exponentiation causes the o-point to disappear
                    print(
                        "Failed to introduce an xpoint on the domain by exponentiating psi_plasma."
                    )
                    print("Manual initialization advised.")
                    return

        # assign from exponentiation if successful
        if xpoint_flag == False:
            print(
                "Failed to introduce an xpoint on the domain by exponentiating psi_plasma."
            )
            print("Manual initialization advised.")
        else:
            self.plasma_psi = n_plasma_psi.copy()

            n_up = 0

            # try to increase the size of the diverted mask
            diverted_core_mask = critical.inside_mask(
                self.R,
                self.Z,
                self.tokamak_psi + n_plasma_psi,
                opt,
                xpt,
            )
            limiter_size = np.sum(self.mask_inside_limiter)
            diverted_size = np.sum(diverted_core_mask)
            print("Size of the diverted core in number of domain pts =", diverted_size)

            diverted_flag = diverted_size > 0.5 * limiter_size
            while diverted_flag == False and n_up < 6:
                # try:
                opt, xpt = critical.find_critical(
                    self.R,
                    self.Z,
                    self.tokamak_psi + n_plasma_psi * 1.1,
                    self.mask_inside_limiter,
                    None,
                )
                xpoint_flag = len(xpt) > 0
                if xpoint_flag:
                    n_plasma_psi *= 1.15
                    self.gmod += np.log(1.15)
                    n_up += 1
                    diverted_core_mask = critical.inside_mask(
                        self.R,
                        self.Z,
                        self.tokamak_psi + n_plasma_psi,
                        opt,
                        xpt,
                    )
                    diverted_size = np.sum(diverted_core_mask)
                    print("diverted_size", diverted_size)
                # except:
                #     diverted_flag = True

        self.plasma_psi = n_plasma_psi.copy()

    def psi_func(self, R, Z, *args, **kwargs):
        """Scipy interpolation of plasma_psi function.
        Replaces the original FreeGS interpolation.
        It now includes a check which leads to the update of the interpolation when needed.

        Parameters
        ----------
        R : ndarray
            R coordinates where the interpolation is needed
        Z : ndarray
            Z coordinates where the interpolation is needed

        Returns
        -------
        ndarray
            Interpolated values of plasma_psi
        """
        check = (
            np.abs(
                np.max(self.psi_func_interp(self.Rnxh, self.Znyh))
                - self.plasma_psi[self.nxh, self.nyh]
            )
            > 1e-5
        )
        if check:
            print(
                "Dicrepancy between psi_func and plasma_psi detected. psi_func has been re-set."
            )
            # redefine interpolating function
            self.psi_func_interp = interpolate.RectBivariateSpline(
                self.R[:, 0], self.Z[0, :], self.plasma_psi
            )

        return self.psi_func_interp(R, Z, *args, **kwargs)

    def initialize_from_equilibrium(self):
        """
        This function loads a pickle file containing an initial guess for the plasma
        flux (and the corners of the grid points it is located on).

        Interpolation is carried out and mapped to the computational grid specified in the
        eq class.

        Parameters
        ----------

        Returns
        -------

        """

        # load the data from the pickle file
        with open(self.equilibrium_path, "rb") as f:
            data = pickle.load(f)

        # extract the data (will fail if not in this format)
        try:
            Rmin = data["Rmin"]
            Rmax = data["Rmax"]
            Zmin = data["Zmin"]
            Zmax = data["Zmax"]
            psi_plasma = data["psi_plasma"]
        except:
            raise ValueError(
                "Data in EQUILIBRIUM_PATH pickle not in correct format or missing."
            )

        # interpolate the plasma psi on the grid given in the data file
        plasma_psi_func = interpolate.RectBivariateSpline(
            np.linspace(Rmin, Rmax, psi_plasma.shape[0]),
            np.linspace(Zmin, Zmax, psi_plasma.shape[1]),
            psi_plasma,
        )

        # extract the values on the grid given in the eq object (this is the initial guess)
        self.plasma_psi = plasma_psi_func(self.R, self.Z, grid=False)

        print(
            "Initial guess for plasma flux initialised using file provided at EQUILIBRIUM_PATH."
        )
