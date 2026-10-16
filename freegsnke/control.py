"""
Legacy FreeGS plasma-control constraints.

These classes belong to the inherited standalone Picard inverse workflow. They
are not used by FreeGSNKE, which supplies its own inverse-solver constraints.
The module is currently retained for compatibility while the legacy API is
reviewed.

Copyright 2016-2021 Ben Dudson, University of York. Email: benjamin.dudson@york.ac.uk
Copyright 2018 Bhavin Patel
Copyright 2026 Nicola C. Amorisco

Derived from FreeGS through FreeGS4E.

SPDX-License-Identifier: LGPL-3.0-or-later

This file is part of FreeGSNKE.

FreeGSNKE is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FreeGSNKE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with FreeGSNKE.  If not, see <http://www.gnu.org/licenses/>.
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from numpy import array, dot, eye, transpose
from numpy.linalg import inv
from scipy import optimize

from . import critical


class constrain(object):
    """
    Adjust coil currents using the inherited FreeGS constraint implementation.

    This is not a supported FreeGSNKE inverse-solver interface. To use this
    legacy class, first create an instance by specifying the constraints:

    >>> controlsystem = constrain(xpoints = [(1.0, 1.1), (1.0,-1.0)])

    controlsystem will now attempt to create x-points at
    (R,Z) = (1.0, 1.1) and (1.0, -1.1) in any Equilibrium

    >>> controlsystem(eq)

    where eq is an Equilibrium object which is modified by
    the call.

    The constraints which can be set are:

    xpoints - A list of X-point (R,Z) locations

    isoflux - A list of tuples (R1,Z1, R2,Z2)

    psivals - A list of (R,Z,psi) values

    At least one constraint must be included. The coil currents are found by
    Tikhonov-regularised least squares, with `gamma` penalising the magnitude
    of the current changes.
    """

    def __init__(
        self,
        xpoints: list[tuple[float, float]] = [],
        gamma: float = 1e-12,
        isoflux: list[tuple[float, float, float, float]] = [],
        psivals: list[tuple[float, float, float]] = [],
    ) -> None:
        """
        Create an instance, specifying the constraints to apply.

        Parameters
        ----------
        xpoints : list[tuple[float, float]], optional
            X-point ``(R, Z)`` locations.
        gamma : float, optional
            Tikhonov regularisation scalar, which minimises the magnitude of the
            coil currents.
        isoflux : list[tuple[float, float, float, float]], optional
            Tuples ``(R1, Z1, R2, Z2)`` of points constrained to have the same
            flux.
        psivals : list[tuple[float, float, float]], optional
            ``(R, Z, psi)`` values to impose.
        """
        self.xpoints = xpoints
        self.gamma = gamma
        self.isoflux = isoflux
        self.psivals = psivals

    def __call__(self, eq: object) -> None:
        """
        Apply the constraints to an Equilibrium, modifying its coil currents.

        Parameters
        ----------
        eq : object
            Equilibrium object, modified in place.

        Raises
        ------
        ValueError
            If no constraints were given.
        """

        tokamak = eq.getMachine()

        constraint_matrix = []
        constraint_rhs = []
        for xpt in self.xpoints:
            # Each x-point introduces two constraints
            # 1) Br = 0

            Br = eq.Br(xpt[0], xpt[1])

            # Add currents to cancel out this field
            constraint_rhs.append(-Br)
            constraint_matrix.append(tokamak.controlBr(xpt[0], xpt[1]))

            # 2) Bz = 0

            Bz = eq.Bz(xpt[0], xpt[1])

            # Add currents to cancel out this field
            constraint_rhs.append(-Bz)
            constraint_matrix.append(tokamak.controlBz(xpt[0], xpt[1]))

        # Constrain points to have the same flux
        for r1, z1, r2, z2 in self.isoflux:
            # Get Psi at (r1,z1) and (r2,z2)
            p1 = eq.psiRZ(r1, z1)
            p2 = eq.psiRZ(r2, z2)
            constraint_rhs.append(p2 - p1)

            # Coil responses
            c1 = tokamak.controlPsi(r1, z1)
            c2 = tokamak.controlPsi(r2, z2)
            # Control for the difference between p1 and p2
            c = [c1val - c2val for c1val, c2val in zip(c1, c2)]
            constraint_matrix.append(c)

        # Constrain the value of psi
        for r, z, psi in self.psivals:
            p1 = eq.psiRZ(r, z)
            constraint_rhs.append(psi - p1)

            # Coil responses
            c = tokamak.controlPsi(r, z)
            constraint_matrix.append(c)

        if not constraint_rhs:
            raise ValueError("No constraints given")

        # Constraint matrix
        A = array(constraint_matrix)
        b = np.reshape(array(constraint_rhs), (-1,))

        # Solve by Tikhonov regularisation
        # minimise || Ax - b ||^2 + ||gamma x ||^2
        #
        # x = (A^T A + gamma^2 I)^{-1}A^T b

        # Number of controls (length of x)
        ncontrols = A.shape[1]

        # Calculate the change in coil current
        current_change = dot(
            inv(dot(transpose(A), A) + self.gamma**2 * eye(ncontrols)),
            dot(transpose(A), b),
        )
        # print("Current changes: " + str(current_change))
        tokamak.controlAdjust(current_change)

        # Ensure that the last constraint used is set in the Equilibrium
        eq._constraints = self

    def plot(self, axis: Axes | None = None, show: bool = True) -> Axes:
        """
        Plot the constraints used for coil current control.

        Parameters
        ----------
        axis : matplotlib.axes.Axes or None, optional
            Axis on which to plot; a new one is created if None.
        show : bool, optional
            Call ``matplotlib.pyplot.show()`` before returning.

        Returns
        -------
        matplotlib.axes.Axes
            The axis containing the plotted constraints.
        """
        from .plotting import plotConstraints

        return plotConstraints(self, axis=axis, show=show)


class ConstrainPsi2D(object):
    """
    Adjusts coil currents to minimise the square differences
    between psi[R,Z] and a target psi.

    Ignores constant offset differences between psi array
    """

    def __init__(
        self, target_psi: np.ndarray, weights: float | np.ndarray | None = None
    ) -> None:
        """
        Store the target psi and the weights.

        Parameters
        ----------
        target_psi : np.ndarray
            2D (R, Z) array; must be the same size as the equilibrium psi.
        weights : float or np.ndarray or None, optional
            Relative importance of each (R, Z) point in the fitting, a float or
            a 2D array of the same size as `target_psi`. By default every point
            is equally weighted. Set points to zero to ignore them in fitting.
        """
        if weights is None:
            weights = np.full(target_psi.shape, 1.0)

        # Remove the average so constant offsets are ignored
        self.target_psi = target_psi - np.average(target_psi, weights=weights)

        self.weights = weights

    def __call__(self, eq: object) -> None:
        """
        Fit the coil currents to the target psi, modifying the Equilibrium.

        Parameters
        ----------
        eq : object
            Equilibrium object, modified in place.
        """

        tokamak = eq.getMachine()
        start_currents = tokamak.controlCurrents()

        end_currents, _ = optimize.leastsq(
            self.psi_difference, start_currents, args=(eq,)
        )

        tokamak.setControlCurrents(end_currents)

        # Ensure that the last constraint used is set in the Equilibrium
        eq._constraints = self

    def psi_difference(self, currents: np.ndarray, eq: object) -> np.ndarray:
        """
        Compute the weighted difference between psi and the target psi.

        Sets the control currents of the equilibrium to `currents` and
        removes the constant offset from its psi before comparing.

        Parameters
        ----------
        currents : np.ndarray
            Control currents to apply to the equilibrium.
        eq : object
            Equilibrium object, modified in place.

        Returns
        -------
        np.ndarray
            Flattened weighted difference between psi and the target.
        """
        eq.getMachine().setControlCurrents(currents)
        psi = eq.psi()
        psi_av = np.average(psi, weights=self.weights)
        return (
            (psi - psi_av - self.target_psi) * self.weights
        ).ravel()  # flatten array


class ConstrainPsiNorm2D(object):
    """
    Adjusts coil currents to minimise the square differences
    between normalised psi[R,Z] and a target normalised psi.
    """

    def __init__(
        self, target_psinorm: np.ndarray, weights: float | np.ndarray = 1.0
    ) -> None:
        """
        Store the target normalised psi and the weights.

        Parameters
        ----------
        target_psinorm : np.ndarray
            2D (R, Z) array; must be the same size as the equilibrium psi.
        weights : float or np.ndarray, optional
            Relative importance of each (R, Z) point in the fitting, a float or
            a 2D array of the same size as `target_psinorm`. By default every
            point is equally weighted. Set points to zero to ignore them.
        """
        self.target_psinorm = target_psinorm
        self.weights = weights

    def __call__(self, eq: object) -> None:
        """
        Fit the coil currents to the target psinorm, modifying the Equilibrium.

        Parameters
        ----------
        eq : object
            Equilibrium object, modified in place.
        """

        tokamak = eq.getMachine()
        start_currents = tokamak.controlCurrents()

        end_currents, _ = optimize.leastsq(
            self.psinorm_difference, start_currents, args=(eq,)
        )

        tokamak.setControlCurrents(end_currents)

        # Ensure that the last constraint used is set in the Equilibrium
        eq._constraints = self

    def psinorm_difference(self, currents: np.ndarray, eq: object) -> np.ndarray:
        """
        Compute the weighted difference between normalised psi and the target.

        Sets the control currents of the equilibrium to `currents`. Normalised
        psi is 0 at the magnetic axis and 1 at the plasma boundary (first
        X-point).

        Parameters
        ----------
        currents : np.ndarray
            Control currents to apply to the equilibrium.
        eq : object
            Equilibrium object, modified in place.

        Returns
        -------
        np.ndarray
            Flattened weighted difference between psi_norm and the target.

        Raises
        ------
        ValueError
            If no O-points or no X-points are found.
        """
        eq.getMachine().setControlCurrents(currents)
        psi = eq.psi()

        opt, xpt = critical.find_critical(eq.R, eq.Z, psi)
        if len(opt) == 0:
            print("No O-points found!")
            print(opt, xpt)
            eq.plot()
            raise ValueError("No O-points found!")
        psi_axis = opt[0][2]

        if len(xpt) == 0:
            print("No X-points found!")
            eq.plot()
            raise ValueError("No X-points found")
        psi_bndry = xpt[0][2]

        # Calculate normalised psi.
        # 0 = magnetic axis
        # 1 = plasma boundary
        psi_norm = (psi - psi_axis) / (psi_bndry - psi_axis)

        return (
            (psi_norm - self.target_psinorm) * self.weights
        ).ravel()  # flatten array
