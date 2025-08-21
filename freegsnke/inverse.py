"""
Implements the optimiser for the inverse Grad-Shafranov problem.

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

from copy import deepcopy

import freegs4e
import numpy as np
from scipy import interpolate


class Inverse_optimizer:
    """This class implements a gradient based optimiser for the coil currents,
    used to perform (static) inverse GS solves.
    """

    def __init__(
        self,
        isoflux_set=None,
        null_points=None,
        psi_vals=None,
        curr_vals=None,
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
        curr_vals : list, optional
            structure [[coil indexes in the array of coils available for control], [coil current values]]
        """

        self.isoflux_set = isoflux_set
        if isoflux_set is not None:
            try:
                type(self.isoflux_set[0][0][0])
                self.isoflux_set = []
                for isoflux in isoflux_set:
                    self.isoflux_set.append(np.array(isoflux))
            except:
                self.isoflux_set = np.array(self.isoflux_set)[np.newaxis]
            self.isoflux_set_n = [len(isoflux[0]) for isoflux in self.isoflux_set]
            # self.isoflux_set_n = [n * (n - 1) / 2 for n in self.isoflux_set_n]

        self.null_points = null_points
        if self.null_points is not None:
            self.null_points = np.array(self.null_points)

        self.psi_vals = psi_vals
        if self.psi_vals is not None:
            self.full_grid = False
            self.psi_vals = np.array(self.psi_vals)
            self.psi_vals = self.psi_vals.reshape((3, -1))
            # subtract unimportant vertical shift
            self.psi_vals[2] -= np.mean(self.psi_vals[2])
            self.norm_psi_vals = np.linalg.norm(self.psi_vals[2])

        self.curr_vals = curr_vals
        self.curr_loss = 0
        if self.curr_vals is not None:
            self.curr_vals = [
                np.array(self.curr_vals[0]).astype(int),
                np.array(self.curr_vals[1]).astype(float),
            ]

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
        self.build_greens(eq)

    def source_domain_properties(self, eq):
        self.eqR = eq.R
        self.eqZ = eq.Z

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
        self.source_domain_properties(eq)

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

    def rebuild_full_current_vec(self, control_currents, filling=0):
        """Builds a full_current vector using the input values.
        Only the coil currents of the coils available for control are filled in.

        Parameters
        ----------
        control_currents : np.array
            Vector of coil currents for those coils available for control.
        """
        full_current_vec = filling * np.ones_like(self.full_current_dummy)
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
            self.mask_set = []
            for i, isoflux in enumerate(self.isoflux_set):
                G = eq.tokamak.createPsiGreensVec(R=isoflux[0], Z=isoflux[1])
                mask = np.triu(
                    np.ones((self.isoflux_set_n[i], self.isoflux_set_n[i])), k=1
                ).astype(bool)
                self.mask_set.append(mask)
                dG = G[:, :, np.newaxis] - G[:, np.newaxis, :]
                self.dG_set.append(dG[:, mask])

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
            self.d_psi_plasma_vals_iso = []
            for i, isoflux in enumerate(self.isoflux_set):
                plasma_vals = psi_func(isoflux[0], isoflux[1], grid=False)
                d_plasma_vals = plasma_vals[:, np.newaxis] - plasma_vals[np.newaxis, :]
                self.d_psi_plasma_vals_iso.append(d_plasma_vals[self.mask_set[i]])

        if self.psi_vals is not None:
            if self.full_grid:
                self.psi_plasma_vals = trial_plasma_psi.reshape(-1)
            else:
                self.psi_plasma_vals = psi_func(
                    self.psi_vals[0], self.psi_vals[1], grid=False
                )

    def build_isoflux_lsq(self, full_currents_vec):
        """Builds for the ordinary least sq problem associated to the isoflux constraints

        Parameters
        ----------
        full_currents_vec : np.array
            Full vector of all coil current values. For example as returned by eq.tokamak.getCurrentsVec()

        """

        loss = []
        A = []
        b = []
        for i, isoflux in enumerate(self.isoflux_set):
            A.append(self.dG_set[i][self.control_mask].T)
            b_val = np.sum(self.dG_set[i] * full_currents_vec[:, np.newaxis], axis=0)
            b_val += self.d_psi_plasma_vals_iso[i]
            b.append(-b_val)

            loss.append(np.linalg.norm(b_val))
        return A, b, loss

    def build_null_points_lsq(self, full_currents_vec):
        """Builds for the ordinary least sq problem associated to the null points

        Parameters
        ----------
        full_currents_vec : np.array
            Full vector of all coil current values. For example as returned by eq.tokamak.getCurrentsVec()

        """

        # radial field
        A_r = self.Gbr[self.control_mask].T
        b_r = np.sum(self.Gbr * full_currents_vec[:, np.newaxis], axis=0)
        b_r += self.brp
        loss = [np.linalg.norm(b_r)]

        # vertical field
        A_z = self.Gbz[self.control_mask].T
        b_z = np.sum(self.Gbz * full_currents_vec[:, np.newaxis], axis=0)
        b_z += self.bzp
        loss.append(np.linalg.norm(b_z))

        A = np.concatenate((A_r, A_z), axis=0)
        b = -np.concatenate((b_r, b_z), axis=0)
        return A, b, loss

    def build_psi_vals_lsq(self, full_currents_vec):
        """Builds for the ordinary least sq problem associated to the psi values

        Parameters
        ----------
        full_currents_vec : np.array
            Full vector of all coil current values. For example as returned by eq.tokamak.getCurrentsVec()

        """

        A = self.G[self.control_mask].T
        b = np.sum(self.G * full_currents_vec[:, np.newaxis], axis=0)
        b += self.psi_plasma_vals
        # subtract mean value
        b -= np.mean(b)
        b -= self.psi_vals[2]
        b *= -1
        normalised_loss = np.linalg.norm(b) / self.norm_psi_vals

        return A, b, [normalised_loss]

    def build_curr_vals_lsq(self, full_currents_vec):
        """Builds for the ordinary least sq problem associated to the psi values

        Parameters
        ----------
        full_currents_vec : np.array
            Full vector of all coil current values. For example as returned by eq.tokamak.getCurrentsVec()

        """
        A = np.zeros((len(self.curr_vals[0]), self.n_control_coils))
        A[np.arange(len(self.curr_vals[0])), self.curr_vals[0]] = 1
        b = self.curr_vals[1] - full_currents_vec[self.control_mask][self.curr_vals[0]]
        self.curr_loss = np.linalg.norm(b)
        return A, b, self.curr_loss

    def build_lsq(self, full_currents_vec):
        """Fetches all terms for the least sq problem, combining all types of magnetic constraints

        Parameters
        ----------
        full_currents_vec : np.array
            Full vector of all coil current values. For example as returned by eq.tokamak.getCurrentsVec()

        """

        loss = 0
        A = np.empty(shape=(0, self.n_control_coils))
        b = np.empty(shape=0)
        loss = []
        if self.isoflux_set is not None:
            A_i, b_i, l = self.build_isoflux_lsq(full_currents_vec)
            A = np.concatenate(A_i, axis=0)
            b = np.concatenate(b_i, axis=0)
            self.isoflux_dim = len(b)
            loss = loss + l
        if self.null_points is not None:
            A_np, b_np, l = self.build_null_points_lsq(full_currents_vec)
            A = np.concatenate((A, A_np), axis=0)
            b = np.concatenate((b, b_np), axis=0)
            self.nullp_dim = len(b)
            loss = loss + l
        if self.psi_vals is not None:
            A_pv, b_pv, l = self.build_psi_vals_lsq(full_currents_vec)
            A = np.concatenate((A, A_pv), axis=0)
            b = np.concatenate((b, b_pv), axis=0)
            self.psiv_dim = len(b)
            loss = loss + l
        if self.curr_vals is not None:
            A_cv, b_cv, l = self.build_curr_vals_lsq(full_currents_vec)
            A = np.concatenate((A, A_cv), axis=0)
            b = np.concatenate((b, b_cv), axis=0)
            self.curr_dim = len(b)
            loss = loss + l
        self.A = np.copy(A)
        self.b = np.copy(b)
        self.loss = np.array(loss)
        # return A, b, loss

    def optimize_currents(self, full_currents_vec, trial_plasma_psi, l2_reg):
        """Solves the least square problem. Tikhonov regularization is applied.

        Parameters
        ----------
        full_currents_vec : np.array
            Full vector of all coil current values. For example as returned by eq.tokamak.getCurrentsVec()
        trial_plasma_psi : np.array
            Flux due to the plasma. Same shape as eq.R
        l2_reg : either float or 1d np.array with len=self.n_control_coils
            The regularization factor

        """
        # prepare the plasma-related values
        self.build_plasma_vals(trial_plasma_psi=trial_plasma_psi)

        # build the matrices that define the optimization
        self.build_lsq(full_currents_vec)

        if type(l2_reg) == float:
            reg_matrix = l2_reg * np.eye(self.n_control_coils)
        else:
            if len(l2_reg) != self.n_control_coils:
                raise ValueError(
                    f"Expected l2_reg to have length equal to number of coils being controlled ({self.n_control_coils}), but got {len(l2_reg)}."
                )
            reg_matrix = np.diag(l2_reg)
        mat = np.linalg.inv(np.matmul(self.A.T, self.A) + reg_matrix)
        delta_current = np.dot(mat, np.dot(self.A.T, self.b))

        return delta_current, np.linalg.norm(self.loss)

    def optimize_currents_grad(
        self,
        full_currents_vec,
        trial_plasma_psi,
        isoflux_weight=1.0,
        null_points_weight=1.0,
        psi_vals_weight=1.0,
        current_weight=1.0,
    ):
        """Solves the least square problem. Tikhonov regularization is applied.

        Parameters
        ----------
        full_currents_vec : np.array
            Full vector of all coil current values. For example as returned by eq.tokamak.getCurrentsVec()
        trial_plasma_psi : np.array
            Flux due to the plasma. Same shape as eq.R
        l2_reg : either float or 1d np.array with len=self.n_control_coils
            The regularization factor

        """
        # prepare the plasma-related values
        self.build_plasma_vals(trial_plasma_psi=trial_plasma_psi)

        # build the matrices that define the optimization
        self.build_lsq(full_currents_vec)

        # weight the different terms in the loss
        b_weighted = np.copy(self.b)
        idx = 0
        if self.isoflux_set is not None:
            b_weighted[idx : idx + self.isoflux_dim] *= isoflux_weight
            idx += self.isoflux_dim
        if self.null_points is not None:
            b_weighted[idx : idx + self.nullp_dim] *= null_points_weight
            idx += self.nullp_dim
        if self.psi_vals is not None:
            b_weighted[idx : idx + self.psiv_dim] *= psi_vals_weight
            idx += self.psiv_dim
        if self.curr_vals is not None:
            b_weighted[idx : idx + self.curr_dim] *= current_weight

        grad = np.dot(self.A.T, b_weighted)

        return grad, np.linalg.norm(self.loss)

    def plot(self, axis=None, show=True):
        """
        Plots constraints used for coil current control

        axis     - Specify the axis on which to plot
        show     - Call matplotlib.pyplot.show() before returning

        """
        from freegs4e.plotting import plotIOConstraints

        return plotIOConstraints(self, axis=axis, show=show)

    def prepare_plasma_psi(self, trial_plasma_psi):
        self.min_psi = np.amin(trial_plasma_psi)
        self.psi0 = np.amax(trial_plasma_psi)
        self.min_psi -= 0.001 * (self.psi0 - self.min_psi)
        self.psi0 -= self.min_psi

    def prepare_plasma_vals_for_plasma(self, trial_plasma_psi):

        self.prepare_plasma_psi(trial_plasma_psi=trial_plasma_psi)

        psi_func = interpolate.RectBivariateSpline(
            self.eqR[:, 0], self.eqZ[0, :], trial_plasma_psi
        )

        if self.isoflux_set is not None:
            self.d_psi_plasma_vals_iso = []
            self.d_psi_for_plasma_iso = []
            for i, isoflux in enumerate(self.isoflux_set):
                plasma_vals = psi_func(isoflux[0], isoflux[1], grid=False)
                d_plasma_vals = plasma_vals[:, np.newaxis] - plasma_vals[np.newaxis, :]
                self.d_psi_plasma_vals_iso.append(d_plasma_vals[self.mask_set[i]])
                hat_plasma_vals = (plasma_vals - self.min_psi) / self.psi0
                hat_plasma_vals *= np.log(hat_plasma_vals)
                d_hat_plasma_vals = (
                    hat_plasma_vals[:, np.newaxis] - hat_plasma_vals[np.newaxis, :]
                )
                self.d_psi_for_plasma_iso.append(
                    self.psi0 * d_hat_plasma_vals[self.mask_set[i]]
                )

    def prepare_for_plasma_optimization(self, eq):
        self.source_domain_properties(eq)
        self.build_greens(eq=eq)

    def build_plasma_isoflux_lsq(self, full_currents_vec, trial_plasma_psi):

        self.prepare_plasma_vals_for_plasma(trial_plasma_psi)

        loss = []
        A = []
        b = []
        for i, isoflux in enumerate(self.isoflux_set):
            b_val = np.sum(self.dG_set[i] * full_currents_vec[:, np.newaxis], axis=0)
            b_val += self.d_psi_plasma_vals_iso[i]
            b.append(-b_val)
            loss.append(np.linalg.norm(b_val))
            # build the jacobian
            Amat = np.zeros((len(b_val), 2))
            # gradient with respect to the normalization of psi
            Amat[:, 0] = self.d_psi_plasma_vals_iso[i]
            # gradient with respect to the exponent of psi
            Amat[:, 1] = self.d_psi_for_plasma_iso[i]
            A.append(Amat)

        self.A_plasma = np.concatenate(A, axis=0)
        self.b_plasma = np.concatenate(b, axis=0)
        self.loss_plasma = np.linalg.norm(loss)

    def optimize_plasma_psi(self, full_currents_vec, trial_plasma_psi, l2_reg):
        self.build_plasma_isoflux_lsq(full_currents_vec, trial_plasma_psi)

        if type(l2_reg) == float:
            reg_matrix = l2_reg * np.eye(2)
        else:
            reg_matrix = np.diag(l2_reg)

        mat = np.linalg.inv(np.matmul(self.A_plasma.T, self.A_plasma) + reg_matrix)
        delta_current = np.dot(mat, np.dot(self.A_plasma.T, self.b_plasma))

        return delta_current, self.loss_plasma
