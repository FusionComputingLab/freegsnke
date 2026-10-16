"""
Legacy routines for solving the nonlinear part of the Grad-Shafranov equation.

This inherited standalone solver is not used or supported by FreeGSNKE. Use
the static or evolutive solvers provided by FreeGSNKE instead. The module is
currently retained for compatibility while the legacy API is reviewed.

Copyright 2024 Nicola C. Amorisco, Adriano Agnello, George K. Holt, Ben Dudson.

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

from typing import Callable

import numpy as np
from numpy import amax, amin, array


def solve(
    eq: object,
    profiles: object,
    constrain: Callable[[object], None] | None = None,
    verbose: bool = False,
    rtol: float = 1e-3,
    atol: float = 1e-10,
    blend: float = 0.0,
    show: bool = False,
    axis: object | None = None,
    pause: float = 0.0001,
    psi_bndry: float | None = None,
    maxits: int | None = 50,
    convergenceInfo: bool = False,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Perform the inherited standalone Picard iteration.

    Notes
    -----
    This is not a supported FreeGSNKE backend entry point. In particular, the
    original inverse workflow using ``freegsnke.control`` is known not to satisfy
    its inherited regression setup.

    Parameters
    ----------
    eq : object
        Equilibrium object (equilibrium.py).
    profiles : object
        Profile object for the toroidal current (jtor.py).
    constrain : Callable[[object], None] or None, optional
        Called with ``eq`` to adjust the coil currents (e.g. to place
        X-points) before and during the iteration.
    verbose : bool, optional
        If True, print the iteration log.
    rtol : float, optional
        Relative tolerance, (change in psi) / (max(psi) - min(psi)).
    atol : float, optional
        Absolute tolerance on the change in psi.
    blend : float, optional
        Blending of previous and next psi solution,
        psi{n+1} <- psi{n+1} * (1-blend) + blend * psi{n}.
    show : bool, optional
        If True, plot the plasma equilibrium at each nonlinear step.
    axis : object or None, optional
        Matplotlib axis to plot onto. If None, a new figure is created.
    pause : float, optional
        Delay between output plots. If negative, waits for the window to
        be closed.
    psi_bndry : float or None, optional
        Poloidal flux at the plasma boundary, passed to ``eq.solve``.
    maxits : int or None, optional
        Maximum number of iterations. None means no limit.
    convergenceInfo : bool, optional
        If True, return the per-iteration convergence history.

    Returns
    -------
    tuple[np.ndarray, np.ndarray] or None
        If ``convergenceInfo`` is True, the maximum absolute change in psi
        and the relative change in psi for each iteration; otherwise None.

    Raises
    ------
    RuntimeError
        If ``maxits`` is exceeded before convergence.
    """

    log = []

    if constrain is not None:
        # Set the coil currents to get X-points in desired locations
        constrain(eq)

    # Get the total psi = plasma + coils
    psi = eq.psi()

    if show:
        import matplotlib.pyplot as plt

        from .plotting import plotEquilibrium

        if pause > 0.0 and axis is None:
            # No axis specified, so create a new figure
            fig = plt.figure()
            axis = fig.add_subplot(111)

    iteration = 0  # Count number of iterations
    psi_maxchange_iterations, psi_relchange_iterations = [], []
    # Start main loop
    while True:

        log.append("-----")
        log.append("Picard iteration: " + str(iteration))

        if show:
            try:
                # Plot state of plasma equilibrium
                if pause < 0:
                    fig = plt.figure()
                    axis = fig.add_subplot(111)
                else:
                    axis.clear()

                plotEquilibrium(eq, axis=axis, show=False)

                if pause < 0:
                    # Wait for user to close the window
                    plt.show()
                else:
                    # Update the canvas and pause
                    # Note, a short pause is needed to force drawing update
                    axis.figure.canvas.draw()
                    plt.pause(pause)
            except:
                pass

        # Copy psi to compare at the end
        psi_last = psi.copy()

        # Solve equilbrium, using the given psi to calculate Jtor
        eq.solve(profiles, psi=psi, psi_bndry=psi_bndry)

        # Get the new psi, including coils
        psi = eq.psi()

        # Compare against last solution
        psi_change = psi_last - psi
        psi_maxchange = amax(abs(psi_change))
        psi_relchange = psi_maxchange / (amax(psi) - amin(psi))

        log.append("...relative error =  " + str(psi_relchange))

        if verbose:
            for x in log:
                print(x)
        log = []

        psi_maxchange_iterations.append(psi_maxchange)
        psi_relchange_iterations.append(psi_relchange)

        # Check if the relative change in psi is small enough
        if (psi_maxchange < atol) or (psi_relchange < rtol):
            break

        # Adjust the coil currents
        if constrain is not None:
            constrain(eq)

        psi = (1.0 - blend) * eq.psi() + blend * psi_last

        # Check if the maximum iterations has been exceeded
        iteration += 1
        if maxits and iteration > maxits:
            raise RuntimeError(
                "Inverse solver failed to converge to requested tolerance."
            )

        eq._profiles = profiles

    if convergenceInfo:
        return array(psi_maxchange_iterations), array(psi_relchange_iterations)
