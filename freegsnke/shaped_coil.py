"""
Define a class of coil which contains a uniform current density
over a shaped region.

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

import numpy as np

from . import polygons, quadrature
from .coil import Coil
from .gradshafranov import Greens, GreensBr, GreensBz


class ShapedCoil(Coil):
    """
    Represents a coil with a uniform current density over a polygonal region.

    The total toroidal current carried by the coil block is ``current * turns``.
    """

    # A dtype for converting to Numpy array and storing in HDF5 files
    dtype = np.dtype(
        [
            (str("RZlen"), int),  # Length of the R and Z arrays
            (str("R"), "10f8"),  # Note: Up to 10 points
            (str("Z"), "10f8"),  # Note: Up to 10 points
            (str("current"), np.float64),
            (str("turns"), int),
            (str("control"), bool),
            (str("npoints"), int),
        ]
    )

    def __init__(
        self,
        shape: list[tuple[float, float]],
        current: float = 0.0,
        turns: int = 1,
        control: bool = True,
        npoints: int = 6,
    ) -> None:
        """
        Set the coil outline, current, turns and quadrature order.

        Parameters
        ----------
        shape : list of (float, float)
            Outline of the coil shape as a list of points
            ``[(r1, z1), (r2, z2), ...]`` in m. Must have more than two points.
        current : float, optional
            The current in the circuit in A. The total current is
            ``current * turns``.
        turns : int, optional
            Number of turns in the coil block.
        control : bool, optional
            Enable or disable control system.
        npoints : int, optional
            Number of quadrature points per triangle. Valid choices: 1, 3, 6.
        """
        assert len(shape) > 2

        # Find the geometric middle of the coil
        # The R,Z properties have accessor functions to handle modifications
        self._R_centre = sum(r for r, z in shape) / len(shape)
        self._Z_centre = sum(z for r, z in shape) / len(shape)

        self.current = current
        self.turns = turns
        self.control = control
        self._area = abs(polygons.area(shape))
        self.shape = shape

        # The quadrature points to be used
        self.npoints_per_triangle = npoints
        self._points = quadrature.polygon_quad(shape, n=npoints)

    def copy(self) -> ShapedCoil:
        """
        Copy the polygon and quadrature points while preserving coil settings.

        Returns
        -------
        ShapedCoil
            The copied coil.
        """
        R_centre = np.copy(self._R_centre)
        Z_centre = np.copy(self._Z_centre)
        points = np.copy(self._points)

        new_obj = type(self)(
            np.copy(self.shape),
            self.current,
            self.turns,
            self.control,
            self.npoints_per_triangle,
        )

        new_obj._R_centre = R_centre
        new_obj._Z_centre = Z_centre
        new_obj._points = points

        return new_obj

    def controlPsi(
        self, R: float | np.ndarray, Z: float | np.ndarray
    ) -> float | np.ndarray:
        """
        Calculate poloidal flux at (R, Z) due to a unit current.

        Sums the weighted contributions of the quadrature points.

        Parameters
        ----------
        R : float or np.ndarray
            Radial coordinate(s) in m.
        Z : float or np.ndarray
            Vertical coordinate(s) in m.

        Returns
        -------
        float or np.ndarray
            The quantity at (R, Z) per unit coil current.
        """
        result = 0.0
        for R_fil, Z_fil, weight in self._points:
            result += Greens(R_fil, Z_fil, R, Z) * weight
        return result

    def controlBr(
        self, R: float | np.ndarray, Z: float | np.ndarray
    ) -> float | np.ndarray:
        """
        Calculate radial magnetic field Br at (R, Z) due to a unit current.

        Sums the weighted contributions of the quadrature points.

        Parameters
        ----------
        R : float or np.ndarray
            Radial coordinate(s) in m.
        Z : float or np.ndarray
            Vertical coordinate(s) in m.

        Returns
        -------
        float or np.ndarray
            The quantity at (R, Z) per unit coil current.
        """
        result = 0.0
        for R_fil, Z_fil, weight in self._points:
            result += GreensBr(R_fil, Z_fil, R, Z) * weight
        return result

    def controlBz(
        self, R: float | np.ndarray, Z: float | np.ndarray
    ) -> float | np.ndarray:
        """
        Calculate vertical magnetic field Bz at (R, Z) due to a unit current.

        Sums the weighted contributions of the quadrature points.

        Parameters
        ----------
        R : float or np.ndarray
            Radial coordinate(s) in m.
        Z : float or np.ndarray
            Vertical coordinate(s) in m.

        Returns
        -------
        float or np.ndarray
            The quantity at (R, Z) per unit coil current.
        """
        result = 0.0
        for R_fil, Z_fil, weight in self._points:
            result += GreensBz(R_fil, Z_fil, R, Z) * weight
        return result

    def __repr__(self) -> str:
        """
        Describe the polygon, current, turns and control flag.

        Returns
        -------
        str
            String representation of the coil.
        """
        return "ShapedCoil({0}, current={1:.1f}, turns={2}, control={3})".format(
            self.shape, self.current, self.turns, self.control
        )

    @property
    def R(self) -> float:
        """
        Major radius of the coil centre in m.

        Returns
        -------
        float
            Mean radial position of the outline vertices.
        """
        return self._R_centre

    @R.setter
    def R(self, Rnew: float) -> None:
        """
        Translate radial quadrature coordinates to the requested centre.

        Parameters
        ----------
        Rnew : float
            New radial position of the coil centre in m.
        """
        # Need to shift all points
        Rshift = Rnew - self._R_centre
        self._points = [(r + Rshift, z, w) for r, z, w in self._points]
        self._R_centre = Rnew

    @property
    def Z(self) -> float:
        """
        Height of the coil centre in m.

        Returns
        -------
        float
            Mean vertical position of the outline vertices.
        """
        return self._Z_centre

    @Z.setter
    def Z(self, Znew: float) -> None:
        """
        Translate vertical quadrature coordinates to the requested centre.

        Parameters
        ----------
        Znew : float
            New vertical position of the coil centre in m.
        """
        # Need to shift all points
        Zshift = Znew - self._Z_centre
        self._points = [(r, z + Zshift, w) for r, z, w in self._points]
        self._Z_centre = Znew

    @property
    def area(self) -> float:
        """
        Fixed polygon cross-sectional area.

        Returns
        -------
        float
            Area in m^2.
        """
        return self._area

    @area.setter
    def area(self, area: float) -> None:
        """
        Reject area changes because the cross-section is fixed by the polygon.

        Parameters
        ----------
        area : float
            Requested area (ignored).

        Raises
        ------
        ValueError
            Always, since the area of a ShapedCoil is fixed.
        """
        raise ValueError("Area of a ShapedCoil is fixed")

    def plot(self, axis: object = None, show: bool = False) -> object:
        """
        Plot the coil shape, using axis if given.

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
        import matplotlib.pyplot as plt

        if axis is None:
            fig = plt.figure()
            axis = fig.add_subplot(111)

        r = [r for r, z in self.shape]
        z = [z for r, z in self.shape]
        axis.fill(r, z, color="blue")

        # Quadrature points
        # rquad = [r for r,z,w in self._points]
        # zquad = [z for r,z,w in self._points]
        # axis.plot(rquad, zquad, 'ro')

        return axis

    def to_numpy_array(self) -> np.ndarray:
        """
        Convert the coil to a structured array, for writing output.

        The outline is zero-padded to 10 points.

        Returns
        -------
        np.ndarray
            Structured 0-d array with the coil serialization ``dtype``.
        """
        RZlen = len(self.shape)
        R = np.zeros(10)
        Z = np.zeros(10)
        R[:RZlen] = [R for R, Z in self.shape]
        Z[:RZlen] = [Z for R, Z in self.shape]

        return np.array(
            (
                RZlen,
                R,
                Z,
                self.current,
                self.turns,
                self.control,
                self.npoints_per_triangle,
            ),
            dtype=self.dtype,
        )

    @classmethod
    def from_numpy_array(cls, value: np.ndarray) -> ShapedCoil:
        """
        Restore the coil polygon and settings from a structured record.

        Parameters
        ----------
        value : np.ndarray
            Structured array with dtype ``ShapedCoil.dtype``.

        Returns
        -------
        ShapedCoil
            The restored coil.

        Raises
        ------
        ValueError
            If ``value.dtype`` does not match ``ShapedCoil.dtype``.
        """
        if value.dtype != cls.dtype:
            raise ValueError(
                "Can't create {this} from dtype: {got} (expected: {dtype})".format(
                    this=type(cls), got=value.dtype, dtype=cls.dtype
                )
            )
        RZlen = value["RZlen"]
        R = value["R"][:RZlen]
        Z = value["Z"][:RZlen]
        current = value["current"]
        turns = value["turns"]
        control = value["control"]
        npoints = value["npoints"]

        return ShapedCoil(
            list(zip(R, Z)),
            current=current,
            turns=turns,
            control=control,
            npoints=npoints,
        )

    def __eq__(self, other: ShapedCoil) -> bool:
        """
        Compare polygon coordinates, current, turns and quadrature settings.

        Parameters
        ----------
        other : ShapedCoil
            Coil to compare against.

        Returns
        -------
        bool
            True if the compared attributes are equal.
        """
        return (
            np.allclose(self.shape, other.shape)
            and self.current == other.current
            and self.turns == other.turns
            and self.control == other.control
            and self.npoints_per_triangle == other.npoints_per_triangle
        )

    def __ne__(self, other: ShapedCoil) -> bool:
        """
        Return whether the shaped coils differ under the equality comparison.

        Parameters
        ----------
        other : ShapedCoil
            Coil to compare against.

        Returns
        -------
        bool
            True if the coils are not equal.
        """
        return not self == other
