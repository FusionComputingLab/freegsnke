"""
Routines for optimising equilibria.

These make use of the generic "optimiser" routines.

Copyright 2019 Ben Dudson, University of York. Email: benjamin.dudson@york.ac.uk

This file is part of FreeGSNKE.

FreeGSNKE is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FreeGSNKE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with FreeGSNKE. If not, see <http://www.gnu.org/licenses/>.
"""

from __future__ import annotations

from math import sqrt
from typing import Callable

import matplotlib.pyplot as plt

from . import optimiser, picard
from .plotting import plotEquilibrium

# Measures which operate on Equilibrium objects


def max_abs_coil_current(eq: object) -> float:
    """
    Return the maximum absolute coil current of an equilibrium.

    Parameters
    ----------
    eq : object
        Equilibrium object.

    Returns
    -------
    float
        Largest absolute coil current.
    """
    currents = eq.tokamak.getCurrents()  # dictionary
    return max(abs(current) for current in currents.values())


def max_coil_force(eq: object) -> float:
    """
    Return the maximum magnitude of the force on any coil.

    Parameters
    ----------
    eq : object
        Equilibrium object.

    Returns
    -------
    float
        Largest value of ``sqrt(force[0]**2 + force[1]**2)`` over the coils.
    """
    forces = eq.tokamak.getForces()  # dictionary
    return max(sqrt(force[0] ** 2 + force[1] ** 2) for force in forces.values())


def no_wall_intersection(eq: object) -> float:
    """
    Penalise intersection of the LCFS with the walls.

    Parameters
    ----------
    eq : object
        Equilibrium object.

    Returns
    -------
    float
        ``inf`` if the equilibrium intersects the wall, otherwise 0.0.
    """
    if eq.intersectsWall():
        return float("inf")
    return 0.0  # No intersection


# Combine measures


def weighted_sum(
    *args: Callable[[object], float] | tuple[Callable[[object], float], float],
) -> Callable[[object], float]:
    """
    Combine measures together as a weighted sum.

    The returned function takes a single argument (the equilibrium) and passes
    it to each given function. The returned values are multiplied by their
    weights and summed.

    Parameters
    ----------
    *args : Callable or tuple[Callable, float]
        Measure functions, or ``(function, weight)`` pairs. If no weight is
        supplied a weight of 1.0 is assumed.

    Returns
    -------
    Callable[[object], float]
        Function mapping an equilibrium to the weighted sum of the measures.
    """
    args_with_weights = []
    for arg in args:
        if callable(arg):
            args_with_weights.append((arg, 1.0))
        else:
            args_with_weights.append(arg)

    def combined_measure(eq: object) -> float:
        """
        Evaluate the weighted sum of the measures for an equilibrium.

        Parameters
        ----------
        eq : object
            Equilibrium object.

        Returns
        -------
        float
            Weighted sum of the measures.
        """
        return sum(func(eq) * weight for func, weight in args_with_weights)

    return combined_measure


# Controls for Equilibrium objects


class CoilRadius:
    """Control to modify the radius of a specified coil."""

    def __init__(
        self, label: str, minimum: float | None = 0.0, maximum: float | None = None
    ) -> None:
        """
        Store the coil label and bounds.

        Parameters
        ----------
        label : str
            Label of the coil to be modified.
        minimum : float or None, optional
            Minimum allowed radius.
        maximum : float or None, optional
            Maximum allowed radius.
        """
        self.label = label
        self.minimum = minimum
        self.maximum = maximum

    def set(self, eq: object, R: float) -> None:
        """
        Set the coil radius, clipped to the allowed bounds.

        Parameters
        ----------
        eq : object
            Equilibrium object, modified in place.
        R : float
            Requested coil radius.
        """
        if self.minimum and (R < self.minimum):
            R = self.minimum
        if self.maximum and (R > self.maximum):
            R = self.maximum
        eq.tokamak[self.label].R = R

    def get(self, eq: object) -> float:
        """
        Get the coil radius.

        Parameters
        ----------
        eq : object
            Equilibrium object.

        Returns
        -------
        float
            Current radius of the coil.
        """
        return eq.tokamak[self.label].R


class CoilHeight:
    """Control to modify the height of a specified coil."""

    def __init__(
        self, label: str, minimum: float | None = None, maximum: float | None = None
    ) -> None:
        """
        Store the coil label and bounds.

        Parameters
        ----------
        label : str
            Label of the coil to be modified.
        minimum : float or None, optional
            Minimum allowed height.
        maximum : float or None, optional
            Maximum allowed height.
        """
        self.label = label
        self.minimum = minimum
        self.maximum = maximum

    def set(self, eq: object, Z: float) -> None:
        """
        Set the coil height, clipped to the allowed bounds.

        Parameters
        ----------
        eq : object
            Equilibrium object, modified in place.
        Z : float
            Requested coil height.
        """
        if self.minimum and (Z < self.minimum):
            Z = self.minimum
        if self.maximum and (Z > self.maximum):
            Z = self.maximum
        eq.tokamak[self.label].Z = Z

    def get(self, eq: object) -> float:
        """
        Get the coil height.

        Parameters
        ----------
        eq : object
            Equilibrium object.

        Returns
        -------
        float
            Current height of the coil.
        """
        return eq.tokamak[self.label].Z


# Monitor optimisation solutions


# Plot and save the best equilibrium each generation
class PlotMonitor:
    """
    Plot the best solution at the end of each generation.

    The plot is saved to a PDF file named ``generation_<n>.pdf``.
    """

    def __init__(self) -> None:
        """Create the figure and axis used for plotting."""
        self.fig = plt.figure()
        self.axis = self.fig.add_subplot(111)

    def __call__(
        self,
        generation: int,
        best: tuple[float, object],
        population: list[tuple[float, object]],
    ) -> None:
        """
        Plot and save the best equilibrium of a generation.

        Parameters
        ----------
        generation : int
            Generation number.
        best : tuple[float, object]
            ``(score, equilibrium)`` of the best candidate.
        population : list[tuple[float, object]]
            ``(score, equilibrium)`` pairs of the whole population (unused).
        """
        self.axis.clear()
        plotEquilibrium(best[1], axis=self.axis, show=False)
        # Update the canvas and pause
        # Note, a short pause is needed to force drawing update
        self.fig.canvas.draw()
        self.axis.set_title("Generation: {} Score: {}".format(generation, best[0]))
        self.fig.savefig("generation_{}.pdf".format(generation))
        plt.pause(0.5)


def optimise(
    eq: object,
    controls: list[object],
    measure: Callable[[object], float],
    maxgen: int = 10,
    N: int = 10,
    CR: float = 0.3,
    F: float = 1.0,
    monitor: (
        Callable[[int, tuple[float, object], list[tuple[float, object]]], None] | None
    ) = None,
) -> object:
    """
    Use Differential Evolution to optimise an Equilibrium.

    See https://en.wikipedia.org/wiki/Differential_evolution

    Parameters
    ----------
    eq : object
        Equilibrium used as starting solution. It is deep-copied and passed to
        `controls` and `measure`.
    controls : list[object]
        Control objects, each with methods ``set(eq, value)``, which modifies
        the given Equilibrium, and ``get(eq)``, which returns the value from it.
    measure : Callable[[object], float]
        Returns a score for a given Equilibrium; the optimiser minimises it.
    maxgen : int, optional
        Maximum number of generations.
    N : int, optional
        Population size (must be >= 4).
    CR : float, optional
        Crossover probability, in [0, 1].
    F : float, optional
        Differential weight, in [0, 2].
    monitor : Callable or None, optional
        Called each generation as ``monitor(generation, best, population)``,
        where `generation` is an int, `best` is a ``(score, equilibrium)``
        tuple and `population` is a list of such tuples.

    Returns
    -------
    object
        The Equilibrium with the lowest score.
    """

    def solve_and_measure(eq: object) -> float:
        """
        Re-solve an equilibrium and evaluate the measure.

        Parameters
        ----------
        eq : object
            Equilibrium object, modified in place.

        Returns
        -------
        float
            Score from `measure`, or ``inf`` if the solve fails.
        """
        # Need to update some internal caches
        eq._pgreen = eq.tokamak.createPsiGreens(eq.R, eq.Z)
        try:
            # Re-solve
            picard.solve(eq, eq._profiles, eq._constraints)
            # Call user-supplied evaluation function
            return measure(eq)
        except:
            # Solve failed.
            return float("inf")

    # Call the generic optimiser,
    return optimiser.optimise(
        eq,
        controls,
        solve_and_measure,
        maxgen=maxgen,
        N=N,
        CR=CR,
        F=F,
        monitor=monitor,
    )
