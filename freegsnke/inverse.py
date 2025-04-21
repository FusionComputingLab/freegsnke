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

import numpy as np
from freegs4e import bilinear_interpolation as bilint
from scipy import interpolate


class Gradient_inverse:

    def __init__(
        self,
        isoflux_set=None,
        null_points=None,
        psi_vals=None,
        gradient_weights=None,
        rescale_coeff=0.2,
    ):

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
            self.psi_vals = np.array(self.psi_vals)
            self.psi_vals = self.psi_vals.reshape((3, -1))

        if gradient_weights is None:
            gradient_weights = np.ones(3)
        self.gradient_weights = gradient_weights

        self.rescale_coeff = rescale_coeff

    def prepare_for_solve(self, eq):
        self.build_control_coils(eq)
        self.build_full_current_vec(eq)
        self.build_control_currents(eq)
        self.build_greens(eq)

    def build_control_coils(self, eq):

        self.control_coils = [
            (label, coil) for label, coil in eq.tokamak.coils if coil.control
        ]
        self.control_mask = np.array(
            [coil.control for label, coil in eq.tokamak.coils]
        ).astype(bool)
        self.coil_order = eq.tokamak.coil_order
        self.n_coils = len(eq.tokamak.coils)
        self.full_current_dummy = np.zeros(self.n_coils)
        self.eqR = eq.R
        self.eqZ = eq.Z

    def build_control_currents(self, eq):
        self.control_currents = eq.tokamak.getCurrentsVec(coils=self.control_coils)

    def build_control_currents_Vec(self, full_currents_vec):
        self.control_currents = full_currents_vec[self.control_mask]

    def build_full_current_vec(self, eq):
        self.full_currents_vec = eq.tokamak.getCurrentsVec()

    def rebuild_full_current_vec(self, control_currents):
        full_current_vec = np.zeros_like(self.full_current_dummy)
        for i, current in enumerate(control_currents):
            full_current_vec[self.coil_order[self.control_coils[i][0]]] = current
        return full_current_vec

    def build_greens(self, eq):

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
            if np.all(self.psi_vals[0] == eq.R) and np.all(self.psi_vals[1] == eq.Z):
                self.G = np.copy(eq._vgreen).reshape((self.n_coils, -1))
            else:
                self.G = eq.tokamak.createPsiGreensVec(
                    R=self.psi_vals[0], Z=self.psi_vals[1]
                )

    def build_plasma_vals(self, trial_plasma_psi):
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
            self.psi_plasma_vals = psi_func(
                self.psi_vals[0], self.psi_vals[1], grid=False
            )

    def build_isoflux_gradient(
        self,
    ):

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
        Lpsi = (
            np.sum(
                self.G * self.full_currents_vec[:, np.newaxis], axis=0, keepdims=True
            )
            + self.psi_plasma_vals[np.newaxis]
        )
        gradient = np.sum(Lpsi * self.G[self.control_mask], axis=1)
        gradient /= np.size(self.psi_vals[0])
        loss = np.sum(Lpsi**2) / np.size(self.psi_vals[0])

        return gradient * self.gradient_weights[2], loss * self.gradient_weights[2]

    def build_gradient(
        self,
    ):

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

    def build_current_update(self, full_currents_vec, trial_plasma_psi):
        # prepare to build the gradient
        self.full_currents_vec = np.copy(full_currents_vec)
        self.build_plasma_vals(trial_plasma_psi=trial_plasma_psi)

        g, l = self.build_gradient()
        dc = -self.rescale_coeff * l * g / np.linalg.norm(g) ** 2
        self.delta_current = self.rebuild_full_current_vec(dc)
        return self.delta_current, l
