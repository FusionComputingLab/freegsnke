"""
Applies the Newton Krylov solver Object to the static GS problem.
Implements both forward and inverse GS solvers.

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

import warnings
from copy import deepcopy

import freegs4e
import numpy as np
from scipy import interpolate


class Gradient_inverse:
    """This class implements a gradient based optimiser for the coil currents,
    used to perform (static) inverse GS solves.
    """

    def __init__(
        self,
        isoflux_set=None,
        null_points=None,
        psi_vals=None,
        gradient_weights=None,
    ):
        """Instantiates the object and sets all magnetic constraints to be used.

        Parameters
        ----------
        isoflux_set : list or np.array, optional
            list of isoflux objects, each with structure
            [Rcoords, Zcoords]
            with Rcoords and Zcoords being 1D lists of the coords of all points that are requested to have the same flux value
        null_points : list or np.array, optional
            structure [Rcoords, Zcoords], with Rcoords and Zcoords being 1D lists
            Sets the coordinates of the desired null points, including both Xpoints and Opoints
        psi_vals : list or np.array, optional
            structure [Rcoords, Zcoords, psi_values]
            with Rcoords, Zcoords and psi_values having the same shape
            Sets the desired values of psi for a set of coordinates, possibly an entire map
        gradient_weights : list or array, optional
            Sets the relative importance of the 3 types of magnetic constraints, by default None
        rescale_coeff : float, optional
            Sets the relative size of the update to the current vector, in terms
            of the Newton update, by default 0.2
        """

        self.isoflux_set = isoflux_set
        if isoflux_set is not None:
            try:
                type(self.isoflux_set[0][0][0])
            except:
                self.isoflux_set = np.array(self.isoflux_set)[np.newaxis]
            self.isoflux_set_n = [len(isoflux[0]) for isoflux in self.isoflux_set]
            self.isoflux_set_n = [n * (n - 1) / 2 for n in self.isoflux_set_n]

        self.null_points = null_points

        self.psi_vals = psi_vals
        if self.psi_vals is not None:
            self.full_grid = False
            self.psi_vals = np.array(self.psi_vals)
            self.psi_vals = self.psi_vals.reshape((3, -1))

        if gradient_weights is None:
            gradient_weights = np.ones(3)
        self.gradient_weights = gradient_weights

    def prepare_for_solve(self, eq):
        """To be called after object is instantiated.
        Prepares necessary quantities for loss/gradient calculations.

        Parameters
        ----------
        eq : freegsnke equilibrium object
            Sources information on:
            -   coils available for control
            -   coil current values
            -   green functions
        """
        self.build_control_coils(eq)
        self.build_full_current_vec(eq)
        self.build_control_currents_Vec(self.full_currents_vec)
        self.build_greens(eq)

    def build_control_coils(self, eq):
        """Records what coils are available for control

        Parameters
        ----------
        eq : freegsnke equilibrium object
        """

        self.control_coils = [
            (label, coil) for label, coil in eq.tokamak.coils if coil.control
        ]
        self.control_mask = np.array(
            [coil.control for label, coil in eq.tokamak.coils]
        ).astype(bool)
        self.no_control_mask = np.logical_not(self.control_mask)
        self.n_control_coils = np.sum(self.control_mask)
        self.coil_order = eq.tokamak.coil_order
        self.n_coils = len(eq.tokamak.coils)
        self.full_current_dummy = np.zeros(self.n_coils)
        self.eqR = eq.R
        self.eqZ = eq.Z

    def build_control_currents(self, eq):
        """Builds vector of coil current values, including only those coils
        that are available for control. Values are extracted from the equilibrium itself.

        Parameters
        ----------
        eq : freegsnke equilibrium object
        """
        self.control_currents = eq.tokamak.getCurrentsVec(coils=self.control_coils)

    def build_control_currents_Vec(self, full_currents_vec):
        """Builds vector of coil current values, including only those coils
        that are available for control. Values are extracted from the full current vector.

        Parameters
        ----------
        full_currents_vec : np.array
            Vector of all coil current values. For example as returned by eq.tokamak.getCurrentsVec()
        """
        self.control_currents = full_currents_vec[self.control_mask]

    def build_full_current_vec(self, eq):
        """Builds full vector of coil current values.

        Parameters
        ----------
        eq : freegsnke equilibrium object
        """
        self.full_currents_vec = eq.tokamak.getCurrentsVec()

    def rebuild_full_current_vec(self, control_currents):
        """Builds a full_current vector using the input values.
        Only the coil currents of the coils available for control are filled in.

        Parameters
        ----------
        control_currents : np.array
            Vector of coil currents for those coils available for control.
        """
        full_current_vec = np.zeros_like(self.full_current_dummy)
        for i, current in enumerate(control_currents):
            full_current_vec[self.coil_order[self.control_coils[i][0]]] = current
        return full_current_vec

    def build_greens(self, eq):
        """Calculates and stores all of the needed green function values.

        Parameters
        ----------
            eq : freegsnke equilibrium object
        """

        if self.isoflux_set is not None:
            self.dG_set = []
            for isoflux in self.isoflux_set:
                G = eq.tokamak.createPsiGreensVec(R=isoflux[0], Z=isoflux[1])
                self.dG_set.append(G[:, :, np.newaxis] - G[:, np.newaxis, :])

        if self.null_points is not None:
            self.Gbr = eq.tokamak.createBrGreensVec(
                R=self.null_points[0], Z=self.null_points[1]
            )
            self.Gbz = eq.tokamak.createBzGreensVec(
                R=self.null_points[0], Z=self.null_points[1]
            )

        if self.psi_vals is not None:
            if np.all(self.psi_vals[0] == eq.R.reshape(-1)) and np.all(
                self.psi_vals[1] == eq.Z.reshape(-1)
            ):
                self.full_grid = True
                self.G = np.copy(eq._vgreen).reshape((self.n_coils, -1))
            else:
                self.G = eq.tokamak.createPsiGreensVec(
                    R=self.psi_vals[0], Z=self.psi_vals[1]
                )

    def build_plasma_vals(self, trial_plasma_psi):
        """Builds and stores all the values relative to the plasma,
        based on the provided plasma_psi

        Parameters
        ----------
        trial_plasma_psi : np.array
            Flux due to the plasma. Same shape as eq.R
        """

        psi_func = interpolate.RectBivariateSpline(
            self.eqR[:, 0], self.eqZ[0, :], trial_plasma_psi
        )

        if self.null_points is not None:
            self.brp = (
                -psi_func(self.null_points[0], self.null_points[1], dy=1, grid=False)
                / self.null_points[0]
            )
            self.bzp = (
                psi_func(self.null_points[0], self.null_points[1], dx=1, grid=False)
                / self.null_points[0]
            )

        if self.isoflux_set is not None:
            self.psi_plasma_vals_iso = []
            for isoflux in self.isoflux_set:
                self.psi_plasma_vals_iso.append(
                    psi_func(isoflux[0], isoflux[1], grid=False)
                )

        if self.psi_vals is not None:
            if self.full_grid:
                self.psi_plasma_vals = trial_plasma_psi.reshape(-1)
            else:
                self.psi_plasma_vals = psi_func(
                    self.psi_vals[0], self.psi_vals[1], grid=False
                )

    def build_isoflux_lsq(
        self,
    ):

        loss = 0
        A = []
        b = []
        for i, isoflux in enumerate(self.isoflux_set):

            self.dG_set[i]

    def build_isoflux_gradient(
        self,
    ):
        """Builds the loss and gradient associated to the isoflux constraints."""

        gradient = np.zeros(len(self.control_currents))
        loss = 0

        for i, isoflux in enumerate(self.isoflux_set):
            dGI = np.sum(
                self.dG_set[i] * self.full_currents_vec[:, np.newaxis, np.newaxis],
                axis=0,
            )
            dpsip = (
                self.psi_plasma_vals_iso[i][:, np.newaxis]
                - self.psi_plasma_vals_iso[i][np.newaxis, :]
            )
            Liso = np.triu(dpsip + dGI, k=1)
            dLiso = Liso[np.newaxis, :, :] * self.dG_set[i][self.control_mask]
            gradient += np.sum(dLiso, axis=(1, 2)) / self.isoflux_set_n[i]
            loss += np.sum(Liso**2) / self.isoflux_set_n[i]

        return gradient * self.gradient_weights[0], loss * self.gradient_weights[0]

    def build_null_points_gradient(
        self,
    ):
        """Builds the loss and gradient associated to the null_points constraints."""

        Lbr = (
            np.sum(
                self.Gbr * self.full_currents_vec[:, np.newaxis], axis=0, keepdims=True
            )
            + self.brp[np.newaxis]
        )
        dLbr = np.sum(Lbr * self.Gbr[self.control_mask], axis=1)
        Lbz = (
            np.sum(
                self.Gbz * self.full_currents_vec[:, np.newaxis], axis=0, keepdims=True
            )
            + self.bzp[np.newaxis]
        )
        dLbz = np.sum(Lbz * self.Gbz[self.control_mask], axis=1)
        gradient = (dLbr + dLbz) / len(self.null_points[0])
        loss = np.sum(Lbr**2 + Lbz**2) / len(self.null_points[0])

        return gradient * self.gradient_weights[1], loss * self.gradient_weights[1]

    def build_psi_vals_gradient(
        self,
    ):
        """Builds the loss and gradient associated to the psi_vals constraints."""
        Lpsi = (
            np.sum(
                self.G * self.full_currents_vec[:, np.newaxis], axis=0, keepdims=True
            )
            + self.psi_plasma_vals[np.newaxis]
            - self.psi_vals[2][np.newaxis]
        )
        gradient = np.sum(Lpsi * self.G[self.control_mask], axis=1)
        gradient /= np.size(self.psi_vals[0])
        loss = np.sum(Lpsi**2) / np.size(self.psi_vals[0])

        return gradient * self.gradient_weights[2], loss * self.gradient_weights[2]

    def build_gradient(
        self,
    ):
        """Combines all contributions to both loss and gradient."""

        gradient = np.zeros_like(self.control_currents)
        loss = 0
        if self.isoflux_set is not None:
            grad, l = self.build_isoflux_gradient()
            gradient += grad
            loss += l
        if self.null_points is not None:
            grad, l = self.build_null_points_gradient()
            gradient += grad
            loss += l
        if self.psi_vals is not None:
            grad, l = self.build_psi_vals_gradient()
            gradient += grad
            loss += l
        self.gradient = np.copy(gradient)
        self.loss = loss
        return self.gradient, self.loss

    def build_current_gradient_update(
        self, full_currents_vec, trial_plasma_psi, rescale_coeff, plasma_calc=True
    ):
        """Calculates the update to the coil currents available for control
        using gradient descent.

        Parameters
        ----------
        full_currents_vec : np.array
            Vector of all coil current values. For example as returned by eq.tokamak.getCurrentsVec()
        trial_plasma_psi : np.array
            Flux due to the plasma. Same shape as eq.R

        """
        # prepare to build the gradient
        self.full_currents_vec = np.copy(full_currents_vec)
        if plasma_calc:
            self.build_plasma_vals(trial_plasma_psi=trial_plasma_psi)

        g, l = self.build_gradient()
        dc = -l * g / np.linalg.norm(g) ** 2
        dc *= rescale_coeff

        self.delta_current = self.rebuild_full_current_vec(dc)
        return self.delta_current, l

    def plot(self, axis=None, show=True):
        """
        Plots constraints used for coil current control

        axis     - Specify the axis on which to plot
        show     - Call matplotlib.pyplot.show() before returning

        """
        from freegs4e.plotting import plotGIConstraints

        return plotGIConstraints(self, axis=axis, show=show)
