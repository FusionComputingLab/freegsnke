"""
Poloidal field coil.

Used in machine to define coils. Can also be a base class for other coil types.

License
-------

Copyright 2016-2019 Ben Dudson, University of York. Email: benjamin.dudson@york.ac.uk

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

import numbers

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from .gradshafranov import (
    Greens,
    GreensBr,
    GreensBz,
    GreensdBrdr,
    GreensdBrdz,
    GreensdBzdr,
    GreensdBzdz,
    mu0,
)


class AreaCurrentLimit:
    """
    Calculate the coil area based on a fixed current density limit.
    """

    def __init__(self, current_density: float = 3.5e9) -> None:
        """
        Set the maximum current density.

        Limits in general depend on the magnetic field.
        Typical values Nb3Sn ~ 3.5e9 A/m^2
        https://doi.org/10.1016/0167-899X(86)90010-8

        Parameters
        ----------
        current_density : float, optional
            Maximum current density in A/m^2.
        """
        self._current_density = current_density

    def __call__(self, coil: Coil) -> float:
        """
        Return the coil area required to carry its current at the density limit.

        Parameters
        ----------
        coil : Coil
            Coil object with ``current`` (A) and ``turns`` attributes.

        Returns
        -------
        float
            Area in m^2.
        """
        return abs(coil.current * coil.turns) / self._current_density


class Coil:
    """
    Represents a poloidal field coil.

    The total toroidal current carried by the coil is ``current * turns``.
    """

    # A dtype for converting to Numpy array and storing in HDF5 files
    dtype = np.dtype(
        [
            (str("R"), np.float64),
            (str("Z"), np.float64),
            (str("current"), np.float64),
            (str("turns"), int),
            (str("control"), bool),
        ]
    )

    def __init__(
        self,
        R: float,
        Z: float,
        current: float = 0.0,
        turns: int = 1,
        control: bool = True,
        area: float | AreaCurrentLimit = AreaCurrentLimit(),
    ) -> None:
        """
        Set the coil location, current, turns, control flag and area.

        The most important effect of the area is on the coil self-force:
        the smaller the area the larger the hoop force for a given current.

        Parameters
        ----------
        R : float
            Radial location of the coil in m.
        Z : float
            Vertical location of the coil in m.
        current : float, optional
            Current in each turn of the coil in A.
        turns : int, optional
            Number of turns. Total coil current is ``current * turns``.
        control : bool, optional
            Enable or disable control system.
        area : float or callable, optional
            Cross-section area in m^2. Either a fixed value (e.g. 0.025 for
            a 5x5 cm coil) or a callable taking the coil as its argument,
            e.g. ``AreaCurrentLimit(current_density)`` with the density in
            A/m^2, in which case the area is recalculated as the coil
            current is changed.
        """
        self.R = R
        self.Z = Z

        self.current = current
        self.turns = turns
        self.control = control
        self.area = area

    def copy(self) -> Coil:
        """
        Create a new object that has identical attributes to self (a copy).

        The copy method will need to be implemented for any subclasses of Coil
        to ensure the correct __init__ signature is called and to ensure
        lists/other object attributes are copied where appropriate (not passed
        by reference).

        Returns
        -------
        Coil
            The copied coil.
        """
        new_obj = type(self)(self.R, self.Z, self.current, self.turns, self.area)
        new_obj._area = self.area
        return new_obj

    def psi(self, R: float | np.ndarray, Z: float | np.ndarray) -> float | np.ndarray:
        """
        Calculate poloidal flux at (R, Z) due to the coil current.

        Parameters
        ----------
        R : float or np.ndarray
            Radial coordinate(s) in m.
        Z : float or np.ndarray
            Vertical coordinate(s) in m.

        Returns
        -------
        float or np.ndarray
            Poloidal flux at (R, Z); 0 if the coil current is negligible
            (|current| <= 1e-5 A).
        """
        if np.abs(self.current) > 1e-5:
            psi_ = self.controlPsi(R, Z) * self.current
        else:
            psi_ = 0
        return psi_

    def createPsiGreens(
        self, R: float | np.ndarray, Z: float | np.ndarray
    ) -> np.ndarray:
        """
        Calculate the poloidal flux Green's function at every point.

        Parameters
        ----------
        R : float or np.ndarray
            Radial coordinate(s) in m.
        Z : float or np.ndarray
            Vertical coordinate(s) in m.

        Returns
        -------
        np.ndarray
            Green's function values (per unit current) at the given points.
        """
        return self.controlPsi(R, Z)

    def createPsiGreensVec(
        self, R: float | np.ndarray, Z: float | np.ndarray
    ) -> np.ndarray:
        """
        Calculate the poloidal flux Green's function at every point.

        Parameters
        ----------
        R : float or np.ndarray
            Radial coordinate(s) in m.
        Z : float or np.ndarray
            Vertical coordinate(s) in m.

        Returns
        -------
        np.ndarray
            Green's function values (per unit current) at the given points.
        """
        return self.controlPsi(R, Z)

    def createBrGreensVec(
        self, R: float | np.ndarray, Z: float | np.ndarray
    ) -> np.ndarray:
        """
        Calculate the Br Green's function at every point.

        Parameters
        ----------
        R : float or np.ndarray
            Radial coordinate(s) in m.
        Z : float or np.ndarray
            Vertical coordinate(s) in m.

        Returns
        -------
        np.ndarray
            Green's function values (per unit current) at the given points.
        """
        return self.controlBr(R, Z)

    def createBzGreensVec(
        self, R: float | np.ndarray, Z: float | np.ndarray
    ) -> np.ndarray:
        """
        Calculate the Bz Green's function at every point.

        Parameters
        ----------
        R : float or np.ndarray
            Radial coordinate(s) in m.
        Z : float or np.ndarray
            Vertical coordinate(s) in m.

        Returns
        -------
        np.ndarray
            Green's function values (per unit current) at the given points.
        """
        return self.controlBz(R, Z)

    def createdBrdrGreensVec(
        self, R: float | np.ndarray, Z: float | np.ndarray
    ) -> np.ndarray:
        """
        Calculate the dBrdr Green's function at every point.

        Parameters
        ----------
        R : float or np.ndarray
            Radial coordinate(s) in m.
        Z : float or np.ndarray
            Vertical coordinate(s) in m.

        Returns
        -------
        np.ndarray
            Green's function values (per unit current) at the given points.
        """
        return self.controldBrdr(R, Z)

    def createdBzdzGreensVec(
        self, R: float | np.ndarray, Z: float | np.ndarray
    ) -> np.ndarray:
        """
        Calculate the dBzdz Green's function at every point.

        Parameters
        ----------
        R : float or np.ndarray
            Radial coordinate(s) in m.
        Z : float or np.ndarray
            Vertical coordinate(s) in m.

        Returns
        -------
        np.ndarray
            Green's function values (per unit current) at the given points.
        """
        return self.controldBzdz(R, Z)

    def createdBrdzGreensVec(
        self, R: float | np.ndarray, Z: float | np.ndarray
    ) -> np.ndarray:
        """
        Calculate the dBrdz Green's function at every point.

        Parameters
        ----------
        R : float or np.ndarray
            Radial coordinate(s) in m.
        Z : float or np.ndarray
            Vertical coordinate(s) in m.

        Returns
        -------
        np.ndarray
            Green's function values (per unit current) at the given points.
        """
        return self.controldBrdz(R, Z)

    def createdBzdrGreensVec(
        self, R: float | np.ndarray, Z: float | np.ndarray
    ) -> np.ndarray:
        """
        Calculate the dBzdr Green's function at every point.

        Parameters
        ----------
        R : float or np.ndarray
            Radial coordinate(s) in m.
        Z : float or np.ndarray
            Vertical coordinate(s) in m.

        Returns
        -------
        np.ndarray
            Green's function values (per unit current) at the given points.
        """
        return self.controldBzdr(R, Z)

    def calcPsiFromGreens(self, pgreen: np.ndarray) -> np.ndarray:
        """
        Calculate plasma psi from Greens functions and current.

        Parameters
        ----------
        pgreen : np.ndarray
            Poloidal flux Green's function values (per unit current).

        Returns
        -------
        np.ndarray
            Poloidal flux due to the coil, ``current * pgreen``.
        """
        return self.current * pgreen

    def Br(self, R: float | np.ndarray, Z: float | np.ndarray) -> float | np.ndarray:
        """
        Calculate radial magnetic field Br at (R, Z) due to the coil current.

        Parameters
        ----------
        R : float or np.ndarray
            Radial coordinate(s) in m.
        Z : float or np.ndarray
            Vertical coordinate(s) in m.

        Returns
        -------
        float or np.ndarray
            The quantity at (R, Z); 0 if the coil current is negligible
            (|current| <= 1e-5 A).
        """
        if np.abs(self.current) > 1e-5:
            val = self.controlBr(R, Z) * self.current
        else:
            val = 0
        return val

    def Bz(self, R: float | np.ndarray, Z: float | np.ndarray) -> float | np.ndarray:
        """
        Calculate vertical magnetic field Bz at (R, Z) due to the coil current.

        Parameters
        ----------
        R : float or np.ndarray
            Radial coordinate(s) in m.
        Z : float or np.ndarray
            Vertical coordinate(s) in m.

        Returns
        -------
        float or np.ndarray
            The quantity at (R, Z); 0 if the coil current is negligible
            (|current| <= 1e-5 A).
        """
        if np.abs(self.current) > 1e-5:
            val = self.controlBz(R, Z) * self.current
        else:
            val = 0
        return val

    def dBrdr(self, R: float | np.ndarray, Z: float | np.ndarray) -> float | np.ndarray:
        """
        Calculate magnetic field gradient dBr/dr at (R, Z) due to the coil current.

        Parameters
        ----------
        R : float or np.ndarray
            Radial coordinate(s) in m.
        Z : float or np.ndarray
            Vertical coordinate(s) in m.

        Returns
        -------
        float or np.ndarray
            The quantity at (R, Z); 0 if the coil current is negligible
            (|current| <= 1e-5 A).
        """
        if np.abs(self.current) > 1e-5:
            val = self.controldBrdr(R, Z) * self.current
        else:
            val = 0
        return val

    def dBzdz(self, R: float | np.ndarray, Z: float | np.ndarray) -> float | np.ndarray:
        """
        Calculate magnetic field gradient dBz/dz at (R, Z) due to the coil current.

        Parameters
        ----------
        R : float or np.ndarray
            Radial coordinate(s) in m.
        Z : float or np.ndarray
            Vertical coordinate(s) in m.

        Returns
        -------
        float or np.ndarray
            The quantity at (R, Z); 0 if the coil current is negligible
            (|current| <= 1e-5 A).
        """
        if np.abs(self.current) > 1e-5:
            val = self.controldBzdz(R, Z) * self.current
        else:
            val = 0
        return val

    def dBrdz(self, R: float | np.ndarray, Z: float | np.ndarray) -> float | np.ndarray:
        """
        Calculate magnetic field gradient dBr/dz at (R, Z) due to the coil current.

        Parameters
        ----------
        R : float or np.ndarray
            Radial coordinate(s) in m.
        Z : float or np.ndarray
            Vertical coordinate(s) in m.

        Returns
        -------
        float or np.ndarray
            The quantity at (R, Z); 0 if the coil current is negligible
            (|current| <= 1e-5 A).
        """
        if np.abs(self.current) > 1e-5:
            val = self.controldBrdz(R, Z) * self.current
        else:
            val = 0
        return val

    def dBzdr(self, R: float | np.ndarray, Z: float | np.ndarray) -> float | np.ndarray:
        """
        Calculate magnetic field gradient dBz/dr at (R, Z) due to the coil current.

        Parameters
        ----------
        R : float or np.ndarray
            Radial coordinate(s) in m.
        Z : float or np.ndarray
            Vertical coordinate(s) in m.

        Returns
        -------
        float or np.ndarray
            The quantity at (R, Z); 0 if the coil current is negligible
            (|current| <= 1e-5 A).
        """
        if np.abs(self.current) > 1e-5:
            val = self.controldBzdr(R, Z) * self.current
        else:
            val = 0
        return val

    def controlPsi(
        self, R: float | np.ndarray, Z: float | np.ndarray
    ) -> float | np.ndarray:
        """
        Calculate poloidal flux at (R, Z) due to a unit current.

        Parameters
        ----------
        R : float or np.ndarray
            Radial coordinate(s) in m.
        Z : float or np.ndarray
            Vertical coordinate(s) in m.

        Returns
        -------
        float or np.ndarray
            The quantity per unit coil current (scaled by the number of turns).
        """
        return Greens(self.R, self.Z, R, Z) * self.turns

    def controlBr(
        self, R: float | np.ndarray, Z: float | np.ndarray
    ) -> float | np.ndarray:
        """
        Calculate radial magnetic field Br at (R, Z) due to a unit current.

        Parameters
        ----------
        R : float or np.ndarray
            Radial coordinate(s) in m.
        Z : float or np.ndarray
            Vertical coordinate(s) in m.

        Returns
        -------
        float or np.ndarray
            The quantity per unit coil current (scaled by the number of turns).
        """
        return GreensBr(self.R, self.Z, R, Z) * self.turns

    def controlBz(
        self, R: float | np.ndarray, Z: float | np.ndarray
    ) -> float | np.ndarray:
        """
        Calculate vertical magnetic field Bz at (R, Z) due to a unit current.

        Parameters
        ----------
        R : float or np.ndarray
            Radial coordinate(s) in m.
        Z : float or np.ndarray
            Vertical coordinate(s) in m.

        Returns
        -------
        float or np.ndarray
            The quantity per unit coil current (scaled by the number of turns).
        """
        return GreensBz(self.R, self.Z, R, Z) * self.turns

    def controldBrdr(
        self, R: float | np.ndarray, Z: float | np.ndarray
    ) -> float | np.ndarray:
        """
        Calculate dBr/dr at (R, Z) due to a unit current.

        Parameters
        ----------
        R : float or np.ndarray
            Radial coordinate(s) in m.
        Z : float or np.ndarray
            Vertical coordinate(s) in m.

        Returns
        -------
        float or np.ndarray
            The quantity per unit coil current (scaled by the number of turns).
        """
        return GreensdBrdr(self.R, self.Z, R, Z) * self.turns

    def controldBzdz(
        self, R: float | np.ndarray, Z: float | np.ndarray
    ) -> float | np.ndarray:
        """
        Calculate dBz/dz at (R, Z) due to a unit current.

        Parameters
        ----------
        R : float or np.ndarray
            Radial coordinate(s) in m.
        Z : float or np.ndarray
            Vertical coordinate(s) in m.

        Returns
        -------
        float or np.ndarray
            The quantity per unit coil current (scaled by the number of turns).
        """
        return GreensdBzdz(self.R, self.Z, R, Z) * self.turns

    def controldBrdz(
        self, R: float | np.ndarray, Z: float | np.ndarray
    ) -> float | np.ndarray:
        """
        Calculate dBr/dz at (R, Z) due to a unit current.

        Parameters
        ----------
        R : float or np.ndarray
            Radial coordinate(s) in m.
        Z : float or np.ndarray
            Vertical coordinate(s) in m.

        Returns
        -------
        float or np.ndarray
            The quantity per unit coil current (scaled by the number of turns).
        """
        return GreensdBrdz(self.R, self.Z, R, Z) * self.turns

    def controldBzdr(
        self, R: float | np.ndarray, Z: float | np.ndarray
    ) -> float | np.ndarray:
        """
        Calculate dBz/dr at (R, Z) due to a unit current.

        Parameters
        ----------
        R : float or np.ndarray
            Radial coordinate(s) in m.
        Z : float or np.ndarray
            Vertical coordinate(s) in m.

        Returns
        -------
        float or np.ndarray
            The quantity per unit coil current (scaled by the number of turns).
        """
        return GreensdBzdr(self.R, self.Z, R, Z) * self.turns

    def getForces(self, equilibrium: object) -> np.ndarray:
        """
        Calculate forces on the coil in Newtons.

        Force on coil due to its own current:
            Lorentz self-forces on curved current loops
            Physics of Plasmas 1, 3425 (1998); https://doi.org/10.1063/1.870491
            David A. Garren and James Chen

        The coil current is temporarily set to zero to evaluate the field
        due to all other coils and the plasma, then restored.

        Parameters
        ----------
        equilibrium : object
            Equilibrium providing ``Br(R, Z)`` and ``Bz(R, Z)`` methods.

        Returns
        -------
        np.ndarray
            Array of two elements, ``[Fr, Fz]``, in N.
        """
        current = self.current  # current per turn
        total_current = current * self.turns  # Total toroidal current

        # Calculate field at this coil due to all other coils
        # and plasma. Need to zero this coil's current
        self.current = 0.0
        Br = equilibrium.Br(self.R, self.Z)
        Bz = equilibrium.Bz(self.R, self.Z)
        self.current = current

        # Assume circular cross-section for hoop (self) force
        minor_radius = np.sqrt(self.area / np.pi)

        # Self inductance factor, depending on internal current
        # distribution. 0.5 for uniform current, 0 for surface current
        self_inductance = 0.5

        # Force per unit length.
        # In cgs units f = I^2/(c^2 * R) * (ln(8*R/a) - 1 + xi/2)
        # In SI units f = mu0 * I^2 / (4*pi*R) * (ln(8*R/a) - 1 + xi/2)
        self_fr = (mu0 * total_current**2 / (4.0 * np.pi * self.R)) * (
            np.log(8.0 * self.R / minor_radius) - 1 + self_inductance / 2.0
        )

        Ltor = 2 * np.pi * self.R  # Length of coil
        return np.array(
            [
                (total_current * Bz + self_fr)
                * Ltor,  # Jphi x Bz = Fr, self force always outwards
                -total_current * Br * Ltor,
            ]
        )  # Jphi x Br = - Fz

    def __repr__(self) -> str:
        """
        Describe the coil position, current, turns and control flag.

        Returns
        -------
        str
            String representation of the coil.
        """
        return "Coil(R={0}, Z={1}, current={2:.1f}, turns={3}, control={4})".format(
            self.R, self.Z, self.current, self.turns, self.control
        )

    def __eq__(self, other: Coil) -> bool:
        """
        Compare coil position, current, turns and control flag.

        Parameters
        ----------
        other : Coil
            Coil to compare against.

        Returns
        -------
        bool
            True if all of the compared attributes are equal.
        """
        return (
            self.R == other.R
            and self.Z == other.Z
            and self.current == other.current
            and self.turns == other.turns
            and self.control == other.control
        )

    def __ne__(self, other: Coil) -> bool:
        """
        Return whether the coils differ under the equality comparison.

        Parameters
        ----------
        other : Coil
            Coil to compare against.

        Returns
        -------
        bool
            True if the coils are not equal.
        """
        return not self == other

    def to_numpy_array(self) -> np.ndarray:
        """
        Convert the coil to a structured array, for writing output.

        Returns
        -------
        np.ndarray
            Structured 0-d array with the coil serialization ``dtype``.
        """
        return np.array(
            (self.R, self.Z, self.current, self.turns, self.control),
            dtype=self.dtype,
        )

    @classmethod
    def from_numpy_array(cls, value: np.ndarray) -> Coil:
        """
        Restore a coil from a structured record with the coil serialization dtype.

        Parameters
        ----------
        value : np.ndarray
            Structured array with dtype ``Coil.dtype``.

        Returns
        -------
        Coil
            The restored coil.

        Raises
        ------
        ValueError
            If ``value.dtype`` does not match ``Coil.dtype``.
        """
        if value.dtype != cls.dtype:
            raise ValueError(
                "Can't create {this} from dtype: {got} (expected: {dtype})".format(
                    this=type(cls), got=value.dtype, dtype=cls.dtype
                )
            )
        return Coil(*value[()])

    @property
    def area(self) -> float:
        """
        The cross-section area of the coil in m^2.

        If the stored area is a callable it is evaluated with the coil.

        Returns
        -------
        float
            Area in m^2.
        """
        if isinstance(self._area, numbers.Number):
            assert self._area >= 0
            return self._area
        # Calculate using functor
        area = self._area(self)
        assert area >= 0
        return area

    @area.setter
    def area(self, area: float | AreaCurrentLimit) -> None:
        """
        Set the cross-sectional area or a callable that computes it.

        Parameters
        ----------
        area : float or callable
            Area in m^2, or a callable taking the coil and returning the area.
        """
        self._area = area

    def plot(self, axis: object = None, show: bool = False) -> object:
        """
        Plot the coil location, using axis if given.

        The area of the coil is used to set the radius, unless the coil has
        rectangular dimensions ``dR`` and ``dZ`` (see ``plot_nke``).

        Parameters
        ----------
        axis : matplotlib.axes.Axes, optional
            Axis to plot on. A new figure is created if None.
        show : bool, optional
            Passed to ``plot_nke``; unused otherwise.

        Returns
        -------
        matplotlib.axes.Axes
            The axis plotted on.
        """

        try:
            axis = self.plot_nke(axis, show)
        except:

            if axis is None:
                fig = plt.figure()
                axis = fig.add_subplot(111)

            minor_radius = np.sqrt(self.area / np.pi)

            circle = plt.Circle(
                (self.R, self.Z),
                minor_radius,
                facecolor="grey",
                edgecolor="k",
                linewidth=0.75,
            )
            axis.add_artist(circle)

        return axis

    def plot_nke(self, axis: object = None, show: bool = False) -> object:
        """
        Plot the coil as a rectangle of size ``dR`` by ``dZ``.

        Requires the coil to have ``dR`` and ``dZ`` attributes.

        Parameters
        ----------
        axis : matplotlib.axes.Axes, optional
            Axis to plot on. A new figure is created if None.
        show : bool, optional
            Unused.

        Returns
        -------
        matplotlib.axes.Axes
            The axis plotted on.
        """
        self.rectangle = Rectangle(
            (self.R - self.dR / 2, self.Z - self.dZ / 2),
            width=self.dR,
            height=self.dZ,
            facecolor="b",
        )

        if axis is None:
            fig = plt.figure()
            axis = fig.add_subplot(111)

        axis.add_patch(self.rectangle)
        return axis
