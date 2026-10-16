"""
Poloidal field coil

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
    Calculate the coil area based on a fixed current density limit
    """

    def __init__(self, current_density=3.5e9):
        """
        current_density   - Maximum current density in A/m^2

        Limits in general depend on the magnetic field
        Typical values Nb3Sn ~ 3.5e9 A/m^2
        https://doi.org/10.1016/0167-899X(86)90010-8

        """
        self._current_density = current_density

    def __call__(self, coil):
        """
        Return the area in m^2, given a Coil object
        """
        return abs(coil.current * coil.turns) / self._current_density


class Coil:
    """
    Represents a poloidal field coil

    public members
    --------------

    R, Z - Location of the coil
    current - current in the coil in Amps
    turns   - Number of turns
    control - enable or disable control system
    area    - Cross-section area in m^2

    The total toroidal current carried by the coil is current * turns
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
        self, R, Z, current=0.0, turns=1, control=True, area=AreaCurrentLimit()
    ):
        """
        R, Z - Location of the coil

        current - current in each turn of the coil in Amps
        turns   - Number of turns. Total coil current is current * turns
        control - enable or disable control system
        area    - Cross-section area in m^2

        Area can be a fixed value (e.g. 0.025 for 5x5cm coil), or can be specified
        using a function which takes a coil as an input argument.
        To specify a current density limit, use:

        area = AreaCurrentLimit(current_density)

        where current_density is in A/m^2. The area of the coil will be recalculated
        as the coil current is changed.

        The most important effect of the area is on the coil self-force:
        The smaller the area the larger the hoop force for a given current.
        """
        self.R = R
        self.Z = Z

        self.current = current
        self.turns = turns
        self.control = control
        self.area = area

    def copy(self):
        """Creates a new object that has identical attributes to self (a copy).

        The copy method will need to be implemented for any subclasses of Coil
        to ensure the correct __init__ signature is called and to ensure
        lists/other object attributes are copied where appropriate (not passed
        by reference).
        """
        new_obj = type(self)(self.R, self.Z, self.current, self.turns, self.area)
        new_obj._area = self.area
        return new_obj

    def psi(self, R, Z):
        """
        Calculate poloidal flux at (R,Z)
        """
        if np.abs(self.current) > 1e-5:
            psi_ = self.controlPsi(R, Z) * self.current
        else:
            psi_ = 0
        return psi_

    def createPsiGreens(self, R, Z):
        """
        Calculate the Greens function at every point, and return
        array. This will be passed back to evaluate Psi in
        calcPsiFromGreens()
        """
        return self.controlPsi(R, Z)

    def createPsiGreensVec(self, R, Z):
        """
        Calculate the Greens function at every point, and return
        array. This will be passed back to evaluate Psi in
        calcPsiFromGreens()
        """
        return self.controlPsi(R, Z)

    def createBrGreensVec(self, R, Z):
        """
        Calculate Br Greens functions
        """
        return self.controlBr(R, Z)

    def createBzGreensVec(self, R, Z):
        """
        Calculate Bz Greens functions
        """
        return self.controlBz(R, Z)

    def createdBrdrGreensVec(self, R, Z):
        """
        Calculate dBrdr Greens functions
        """
        return self.controldBrdr(R, Z)

    def createdBzdzGreensVec(self, R, Z):
        """
        Calculate dBzdz Greens functions
        """
        return self.controldBzdz(R, Z)

    def createdBrdzGreensVec(self, R, Z):
        """
        Calculate dBrdz Greens functions
        """
        return self.controldBrdz(R, Z)

    def createdBzdrGreensVec(self, R, Z):
        """
        Calculate dBzdr Greens functions
        """
        return self.controldBzdr(R, Z)

    def calcPsiFromGreens(self, pgreen):
        """
        Calculate plasma psi from Greens functions and current
        """
        return self.current * pgreen

    def Br(self, R, Z):
        """
        Calculate radial magnetic field Br at (R,Z)
        """
        if np.abs(self.current) > 1e-5:
            br = self.controlBr(R, Z) * self.current
        else:
            br = 0
        return br

    def Bz(self, R, Z):
        """
        Calculate vertical magnetic field Bz at (R,Z)
        """
        if np.abs(self.current) > 1e-5:
            bz = self.controlBz(R, Z) * self.current
        else:
            bz = 0
        return bz

    def dBrdr(self, R, Z):
        """
        Calculate magnetic field dBrdr at (R,Z)
        """
        if np.abs(self.current) > 1e-5:
            br = self.controldBrdr(R, Z) * self.current
        else:
            br = 0
        return br

    def dBzdz(self, R, Z):
        """
        Calculate magnetic field dBzdz at (R,Z)
        """
        if np.abs(self.current) > 1e-5:
            bz = self.controldBzdz(R, Z) * self.current
        else:
            bz = 0
        return bz

    def dBrdz(self, R, Z):
        """
        Calculate magnetic field dBrdz at (R,Z)
        """
        if np.abs(self.current) > 1e-5:
            br = self.controldBrdz(R, Z) * self.current
        else:
            br = 0
        return br

    def dBzdr(self, R, Z):
        """
        Calculate magnetic field dBzdr at (R,Z)
        """
        if np.abs(self.current) > 1e-5:
            bz = self.controldBzdr(R, Z) * self.current
        else:
            bz = 0
        return bz

    def controlPsi(self, R, Z):
        """
        Calculate poloidal flux at (R,Z) due to a unit current
        """
        return Greens(self.R, self.Z, R, Z) * self.turns

    def controlBr(self, R, Z):
        """
        Calculate radial magnetic field Br at (R,Z) due to a unit current
        """
        return GreensBr(self.R, self.Z, R, Z) * self.turns

    def controlBz(self, R, Z):
        """
        Calculate vertical magnetic field Bz at (R,Z) due to a unit current
        """
        return GreensBz(self.R, self.Z, R, Z) * self.turns

    def controldBrdr(self, R, Z):
        """
        Calculate magnetic field dBrdr at (R,Z) due to a unit current
        """
        return GreensdBrdr(self.R, self.Z, R, Z) * self.turns

    def controldBzdz(self, R, Z):
        """
        Calculate magnetic field dBzdz at (R,Z) due to a unit current
        """
        return GreensdBzdz(self.R, self.Z, R, Z) * self.turns

    def controldBrdz(self, R, Z):
        """
        Calculate magnetic field dBrdz at (R,Z) due to a unit current
        """
        return GreensdBrdz(self.R, self.Z, R, Z) * self.turns

    def controldBzdr(self, R, Z):
        """
        Calculate magnetic field dBzdr at (R,Z) due to a unit current
        """
        return GreensdBzdr(self.R, self.Z, R, Z) * self.turns

    def getForces(self, equilibrium):
        """
        Calculate forces on the coils in Newtons

        Returns an array of two elements: [ Fr, Fz ]


        Force on coil due to its own current:
            Lorentz self‐forces on curved current loops
            Physics of Plasmas 1, 3425 (1998); https://doi.org/10.1063/1.870491
            David A. Garren and James Chen
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

    def __repr__(self):
        """Describe the coil position, current, turns and control flag."""
        return "Coil(R={0}, Z={1}, current={2:.1f}, turns={3}, control={4})".format(
            self.R, self.Z, self.current, self.turns, self.control
        )

    def __eq__(self, other):
        """Compare coil position, current, turns and control flag."""
        return (
            self.R == other.R
            and self.Z == other.Z
            and self.current == other.current
            and self.turns == other.turns
            and self.control == other.control
        )

    def __ne__(self, other):
        """Return whether the coils differ under the equality comparison."""
        return not self == other

    def to_numpy_array(self):
        """
        Helper method for writing output
        """
        return np.array(
            (self.R, self.Z, self.current, self.turns, self.control),
            dtype=self.dtype,
        )

    @classmethod
    def from_numpy_array(cls, value):
        """Restore a coil from a structured record with the coil serialization dtype."""
        if value.dtype != cls.dtype:
            raise ValueError(
                "Can't create {this} from dtype: {got} (expected: {dtype})".format(
                    this=type(cls), got=value.dtype, dtype=cls.dtype
                )
            )
        return Coil(*value[()])

    @property
    def area(self):
        """
        The cross-section area of the coil in m^2
        """
        if isinstance(self._area, numbers.Number):
            assert self._area >= 0
            return self._area
        # Calculate using functor
        area = self._area(self)
        assert area >= 0
        return area

    @area.setter
    def area(self, area):
        """Set the cross-sectional area or a callable that computes it in square metres."""
        self._area = area

    def plot(self, axis=None, show=False):
        """
        Plot the coil location, using axis if given

        The area of the coil is used to set the radius
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

    def plot_nke(self, axis=None, show=False):
        """
        Plot the coil location, using axis if given

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
