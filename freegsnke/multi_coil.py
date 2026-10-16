"""
Classes and routines to represent coils and circuits.

Copyright 2019 Chris Marsden, Tokamak Energy. Email: Chris.Marsden@tokamakenergy.co.uk

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

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from .coil import AreaCurrentLimit, Coil
from .gradshafranov import Greens, GreensBr, GreensBz


class MultiCoil(Coil):
    """
    A coil represented by one or more filaments, optionally mirrored in Z=0.

    This class is multifunctional and can model several coil arrangements:
    1. A single PF coil
    2. A block of several PF coil filaments, modelled as
        a) a single point in (R,Z)
        b) a list of (R,Z) points for each filament in the block

    In both cases, the specified coil(s) can also be mirrored in Z=0 and
    connected in a circuit sharing the same power supply. In such a case,
    there is also the option to specify the polarity of the currents in
    the upper and lower coil blocks, enabling the wiring of these blocks
    in opposite directions.

    For multiple point coils, the total toroidal current carried by the
    point coil block is ``current * turns``.
    """

    # A dtype for converting to Numpy array and storing in HDF5 files
    dtype = np.dtype(
        [
            (str("RZlen"), int),  # Length of R and Z arrays
            (str("R"), "500f8"),  # Up to 100 points
            (str("Z"), "500f8"),
            (str("current"), np.float64),
            (str("turns"), int),
            (str("control"), bool),
            (str("mirror"), bool),
            (str("polarity"), "2f8"),
        ]
    )

    def __init__(
        self,
        R: float | list[float] | np.ndarray,
        Z: float | list[float] | np.ndarray,
        current: float = 0.0,
        turns: int = 1,
        control: bool = True,
        mirror: bool = False,
        polarity: list[float] | np.ndarray | None = None,
        area: float | AreaCurrentLimit = AreaCurrentLimit(),
    ) -> None:
        """
        Set the filament locations, current, turns, mirroring and polarity.

        Parameters
        ----------
        R : float or array-like
            Radial location(s) in m. A single value is the coil centre (moving
            it moves all filaments); a sequence gives the filament locations.
        Z : float or array-like
            Vertical location(s) in m, with the same length as ``R``.
        current : float, optional
            Current in each turn of the coil in A.
        turns : int, optional
            Number of turns in the point coil block; total block current is
            ``current * turns``. Only used if ``R`` and ``Z`` are single
            points, otherwise it is set to the number of filaments.
        control : bool, optional
            Enable or disable control system.
        mirror : bool, optional
            Mirror the point/detailed coil block in Z=0, creating a circuit.
        polarity : array-like of float, optional
            Wiring of the circuit, same or opposite direction, as
            ``[block 1, block 2]``. Defaults to ``[1.0, 1.0]``.
        area : float or callable, optional
            Cross-section area of the block in m^2. Either a fixed value
            (e.g. 0.025 for a 5x5 cm coil) or a callable taking the coil as
            its argument, e.g. ``AreaCurrentLimit(current_density)`` with
            the density in A/m^2, in which case the area is recalculated as
            the coil current is changed.
        """
        # Store locations as an internal list
        if hasattr(R, "__len__"):
            assert len(R) == len(Z)
            self.Rfil = R
            self.Zfil = Z
            self.turns = len(R)
        else:
            # Assume a single R, Z. Turn into a list
            self.Rfil = [R]
            self.Zfil = [Z]
            self.turns = turns

        self.current = current
        self.control = control
        self.mirror = mirror
        self.polarity = [1.0, 1.0] if polarity is None else polarity
        self.area = area

        # Internal (R,Z) value, should not be modified directly
        self._R_centre = np.mean(self.Rfil)
        self._Z_centre = np.mean(self.Zfil)

    def copy(self) -> MultiCoil:
        """
        Create a copy of this object.

        The extra attributes ``rectangle``, ``resistivity``, ``dR`` and ``dZ``
        are copied (by reference) if present.

        Returns
        -------
        MultiCoil
            A new instance of the same class with copied attributes.
        """

        new_obj = type(self)(
            np.copy(self.Rfil),
            np.copy(self.Zfil),
            self.current,
            self.turns,
            self.control,
            self.mirror,
            self.polarity,
            self.area,
        )

        # copy any additional attributes if they exist on the original
        extra_attrs = ["rectangle", "resistivity", "dR", "dZ"]
        for key in extra_attrs:
            try:
                value = getattr(self, key)
                setattr(new_obj, key, value)
            except AttributeError:
                # Skip if the attribute doesn't exist on the original
                pass

        return new_obj

    def controlPsi(
        self, R: float | np.ndarray, Z: float | np.ndarray
    ) -> float | np.ndarray:
        """
        Calculate poloidal flux at (R, Z) due to a unit current.

        Sums the contributions of all filaments (and their Z=0 mirror images
        if ``mirror`` is set), weighted by ``polarity``.

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
        for R_fil, Z_fil in zip(self.Rfil, self.Zfil):
            result += Greens(R_fil, Z_fil, R, Z) * self.polarity[0]

        if self.mirror:
            for R_fil, Z_fil in zip(self.Rfil, self.Zfil):
                result += Greens(R_fil, -Z_fil, R, Z) * self.polarity[1]
        return result

    def controlBr(
        self, R: float | np.ndarray, Z: float | np.ndarray
    ) -> float | np.ndarray:
        """
        Calculate radial magnetic field Br at (R, Z) due to a unit current.

        Sums the contributions of all filaments (and their Z=0 mirror images
        if ``mirror`` is set), weighted by ``polarity``.

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
        for R_fil, Z_fil in zip(self.Rfil, self.Zfil):
            result += GreensBr(R_fil, Z_fil, R, Z) * self.polarity[0]

        if self.mirror:  # Mirror coil(s) in Z, creating circuit
            for R_fil, Z_fil in zip(self.Rfil, self.Zfil):
                result += GreensBr(R_fil, -Z_fil, R, Z) * self.polarity[1]
        return result

    def controlBz(
        self, R: float | np.ndarray, Z: float | np.ndarray
    ) -> float | np.ndarray:
        """
        Calculate vertical magnetic field Bz at (R, Z) due to a unit current.

        Sums the contributions of all filaments (and their Z=0 mirror images
        if ``mirror`` is set), weighted by ``polarity``.

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
        for R_fil, Z_fil in zip(self.Rfil, self.Zfil):
            result += GreensBz(R_fil, Z_fil, R, Z) * self.polarity[0]

        if self.mirror:  # Mirror coil(s) in Z, creating circuit
            for R_fil, Z_fil in zip(self.Rfil, self.Zfil):
                result += GreensBz(R_fil, -Z_fil, R, Z) * self.polarity[1]
        return result

    def __repr__(self) -> str:
        """
        Describe the filaments, current, turns and mirrored-coil configuration.

        Returns
        -------
        str
            String representation of the coil.
        """
        return "MultiCoil(R={0}, Z={1}, current={2:.1f}, turns={3}, control={4}, mirror={5}, polarity={6})".format(
            self.Rfil,
            self.Zfil,
            self.current,
            self.turns,
            self.control,
            self.mirror,
            self.polarity,
        )

    def __eq__(self, other: MultiCoil) -> bool:
        """
        Compare coil centres, currents, turns and control flags.

        Parameters
        ----------
        other : MultiCoil
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

    def __ne__(self, other: MultiCoil) -> bool:
        """
        Return whether the coil groups differ under the equality comparison.

        Parameters
        ----------
        other : MultiCoil
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

        Filament coordinates are zero-padded to 500 entries.

        Returns
        -------
        np.ndarray
            Structured 0-d array with the coil serialization ``dtype``.
        """

        RZlen = len(self.Rfil)
        assert RZlen <= 500
        R = np.zeros(500)
        Z = np.zeros(500)
        R[:RZlen] = self.Rfil
        Z[:RZlen] = self.Zfil

        return np.array(
            (
                RZlen,
                R,
                Z,
                self.current,
                self.turns,
                self.control,
                self.mirror,
                self.polarity,
            ),
            dtype=self.dtype,
        )

    @classmethod
    def from_numpy_array(cls, value: np.ndarray) -> MultiCoil:
        """
        Restore a filament group from a structured serialization record.

        Parameters
        ----------
        value : np.ndarray
            Structured array with dtype ``MultiCoil.dtype``.

        Returns
        -------
        MultiCoil
            The restored coil.

        Raises
        ------
        ValueError
            If ``value.dtype`` does not match ``MultiCoil.dtype``.
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
        mirror = value["npoints"]
        polarity = value["polarity"]

        return MultiCoil(
            R,
            Z,
            current=current,
            turns=turns,
            control=control,
            mirror=mirror,
            polarity=polarity,
        )

    def plot(self, axis: object = None, show: bool = False) -> object:
        """
        Plot the coil geometry including the turn (filament) locations.

        This method attempts to plot the coil using the class's `plot_nke` method.
        If that fails (e.g., due to a missing method or plotting error), it falls
        back to plotting the coil turn positions directly using `matplotlib`.

        Parameters
        ----------
        axis : matplotlib.axes.Axes, optional
            The Matplotlib axis to plot on. If None, a new figure and axis are created.
        show : bool, optional
            Passed to ``plot_nke``; ``plt.show()`` is not called.

        Returns
        -------
        matplotlib.axes.Axes
            The axis object containing the plot.
        """

        try:
            axis = self.plot_nke(axis, show)
        except Exception as e:
            if axis is None:
                fig = plt.figure()
                axis = fig.add_subplot(111)

            plt.plot(self.Rfil, self.Zfil, "bo")

        return axis

    def plot_nke(self, axis: object = None, show: bool = False) -> object:
        """
        Plot each filament as a rectangle of size ``dR`` by ``dZ``.

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

        if axis is None:
            fig = plt.figure()
            axis = fig.add_subplot(111)

        for i in np.arange(len(self.Rfil)):
            self.rectangle = Rectangle(
                (self.Rfil[i] - self.dR / 2, self.Zfil[i] - self.dZ / 2),
                width=self.dR,
                height=self.dZ,
                facecolor="dodgerblue",
                edgecolor="b",
                linewidth=1,
            )
            axis.add_patch(self.rectangle)
        return axis

    @property
    def R(self) -> float:
        """
        Major radius of the coil centre in m.

        Returns
        -------
        float
            Mean radial position of the filaments.
        """
        return self._R_centre

    @R.setter
    def R(self, Rnew: float) -> None:
        """
        Translate every radial filament coordinate to the new coil centre.

        Parameters
        ----------
        Rnew : float
            New radial position of the coil centre in m.
        """
        # Need to shift all points
        Rshift = Rnew - self._R_centre
        for i in range(len(self.Rfil)):
            self.Rfil[i] += Rshift
        self._R_centre = Rnew

    @property
    def Z(self) -> float:
        """
        Height of the coil centre in m.

        Returns
        -------
        float
            Mean vertical position of the filaments.
        """
        return self._Z_centre

    @Z.setter
    def Z(self, Znew: float) -> None:
        """
        Translate every vertical filament coordinate to the new coil centre.

        Parameters
        ----------
        Znew : float
            New vertical position of the coil centre in m.
        """
        # Need to shift all points
        Zshift = Znew - self._Z_centre
        for i in range(len(self.Zfil)):
            self.Zfil[i] += Zshift
        self._Z_centre = Znew
