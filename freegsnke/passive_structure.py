"""
Implements the FreeGSNKE object used to deal with extended vessel structures.
Current is distributed uniformly over each extended structure.

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

import freegs4e
import matplotlib.pyplot as plt
import numpy as np
from freegs4e.gradshafranov import Greens, GreensBr, GreensBz, mu0
from matplotlib.patches import Polygon

from .refine_passive import find_area, generate_refinement


class PassiveStructure(freegs4e.coil.Coil):
    """Inherits from freegs4e.coil.Coil.
    Object to implement passive structures.
    Rather than listing large number of filaments it averages the
    relevant green functions so that currents are distributed over
    the structure -- uniformly.
    """

    def __init__(
        self,
        R,
        Z,
        min_refine_per_area,
        min_refine_per_length,
        refine_mode="G",
    ):
        """Instantiates the object and builds the refinement of the provided polygonal shape.

        Parameters
        ----------
        R : array
            List of vertex coordinates, defining a passive structure polygon.
        Z : array
            List of vertex coordinates, defining a passive structure polygon.
        refine_mode : str, optional
            refinement mode for passive structures inputted as polygons, by default 'G' for 'grid'
            Use 'LH' for alternative mode using a Latin Hypercube implementation.
        """

        res = find_area(R, Z, 1e3)
        self.area = res[0]
        self.R = res[-2]
        self.Z = res[-1]
        self.Len = np.linalg.norm(res[-3])

        self.turns = 1
        self.control = False
        self.current = 0

        self.Rpolygon = np.array(R)
        self.Zpolygon = np.array(Z)
        self.vertices = np.concatenate(
            (self.Rpolygon[:, np.newaxis], self.Zpolygon[:, np.newaxis]), axis=-1
        )
        self.polygon = Polygon(self.vertices, facecolor="k", alpha=0.75)

        self.refine_mode = refine_mode
        self.n_refine = int(
            max(1, self.area * min_refine_per_area, self.Len * min_refine_per_length)
        )
        self.filaments = self.build_refining_filaments()

        self.greens = {}

    def copy(self):
        """
        Create a shallow copy of the control object without reinitialising
        geometry or recomputing Green's functions.

        This method avoids calling the constructor for performance reasons and
        instead manually copies attributes that define the control geometry and
        cached electromagnetic response.

        Returns
        -------
        object
            A new instance of the same class with duplicated state.

        Notes
        -----
        - Geometry-related arrays (e.g. `R`, `Z`, `vertices`, `filaments`) are
        shared by reference and are assumed to be immutable.
        - The `greens` dictionary is shallow-copied:
            - Safe usage: replacing entries (e.g. `greens["psi"] = new_array`)
            - Unsafe usage: in-place modification (e.g. `greens["psi"][:] = ...`)
        - This copy is therefore *not fully independent* and should be used only
        when immutability assumptions hold.
        - The attribute `control` is currently assigned from `self.turns`
        (this may be intentional or a likely typo depending on design intent).
        """
        # dont instantiate the new object, it will be slow
        new_obj = type(self).__new__(type(self))

        new_obj.turns = self.turns
        new_obj.control = self.turns
        new_obj.current = self.current
        new_obj.refine_mode = self.refine_mode

        # ASSUMING the shape will never be modified in-place
        new_obj.area = self.area
        new_obj.R = self.R
        new_obj.Z = self.Z
        new_obj.Len = self.Len
        new_obj.Rpolygon = self.Rpolygon
        new_obj.Zpolygon = self.Zpolygon
        new_obj.vertices = self.vertices
        new_obj.polygon = self.polygon
        new_obj.n_refine = self.n_refine
        new_obj.filaments = self.filaments

        # This performs a shallow copy of the greens dictionary.
        # This implicitly assumes that the dictionary might be modified
        # e.g. self.greens["psi"] = new_array (this would be fine)
        # but its values WON't be modified in place
        # e.g. self.greens["psi"][:] = new_array (this would cause problems)
        new_obj.greens = self.greens.copy()

        return new_obj

    def create_RZ_key(self, R, Z):
        """
        Create a hashable key identifying a specific R–Z grid for caching Green's functions.

        The key is based on the grid bounds and size, and is used to ensure that
        Green's function evaluations are reused only when the underlying spatial
        grid is identical.

        Parameters
        ----------
        R : np.ndarray
            Radial coordinate grid (typically `eq.R`).
        Z : np.ndarray
            Vertical coordinate grid (typically `eq.Z`).

        Returns
        -------
        tuple
            A hashable identifier of the form:
            (R_min, R_max, Z_min, Z_max, N),
            where N = total number of grid points in `R`.
        """
        RZ_key = (np.min(R), np.max(R), np.min(Z), np.max(Z), np.size(R))
        return RZ_key

    def build_refining_filaments(
        self,
    ):
        """
        Construct the set of refining filaments used to discretise the control
        region at higher resolution.

        This routine generates a refined filament representation of the control
        geometry, typically used to improve spatial resolution of control
        response calculations.

        Returns
        -------
        np.ndarray
            Array of filament coordinates generated by the refinement procedure.

        Notes
        -----
        - The filaments are generated from the control polygon defined by
        `self.Rpolygon` and `self.Zpolygon`.
        - The refinement density is controlled by `self.n_refine`.
        - The refinement strategy is determined by `self.refine_mode`.
        - The function `generate_refinement` also returns an area estimate,
        which is currently ignored.
        """

        filaments, area = generate_refinement(
            self.Rpolygon, self.Zpolygon, self.n_refine, self.refine_mode
        )
        return filaments

    def build_control_psi(self, R, Z):
        """
        Compute and cache the Green's function for the poloidal flux (ψ)
        induced by the control filaments on a specified R–Z grid.

        The result is obtained by evaluating the contribution from each filament
        (via `Greens`) and averaging across all filaments.

        Parameters
        ----------
        R : np.ndarray
            2D array defining the radial coordinate grid (e.g. equilibrium R grid).
        Z : np.ndarray
            2D array defining the vertical coordinate grid (e.g. equilibrium Z grid).

        Notes
        -----
        - Each filament is evaluated over the full grid using broadcasting.
        - Filament contributions are averaged to produce the total response.
        - The result is cached in `self.greens[(R, Z)]["psi"]`.

        Returns
        -------
        None
            The computed Green's function is stored internally.
        """

        greens_psi = Greens(
            self.filaments[:, 0].reshape([-1] + [1] * R.ndim),
            self.filaments[:, 1].reshape([-1] + [1] * R.ndim),
            R[np.newaxis],
            Z[np.newaxis],
        )
        greens_psi = np.mean(greens_psi, axis=0)

        RZ_key = self.create_RZ_key(R, Z)
        try:
            self.greens[RZ_key]["psi"] = greens_psi
        except:
            self.greens[RZ_key] = {"psi": greens_psi}

    def build_control_br(self, R, Z):
        """
        Compute and cache the Green's function for the radial magnetic field (Br)
        induced by the control filaments on a specified R–Z grid.

        The field is obtained by evaluating the Biot–Savart contribution from each
        filament (via `GreensBr`) and averaging over all filaments.

        Parameters
        ----------
        R : np.ndarray
            2D array defining the radial coordinate grid (e.g. equilibrium R grid).
        Z : np.ndarray
            2D array defining the vertical coordinate grid (e.g. equilibrium Z grid).

        Notes
        -----
        - Each filament is evaluated on the full R–Z grid using broadcasting.
        - Contributions are averaged across filaments to form the final field.
        - The result is cached in `self.greens[(R, Z)]["Br"]` to avoid recomputation.

        Returns
        -------
        None
            The computed Green's function is stored in `self.greens`.
        """

        greens_br = GreensBr(
            self.filaments[:, 0].reshape([-1] + [1] * R.ndim),
            self.filaments[:, 1].reshape([-1] + [1] * R.ndim),
            R[np.newaxis],
            Z[np.newaxis],
        )
        greens_br = np.mean(greens_br, axis=0)

        RZ_key = self.create_RZ_key(R, Z)
        try:
            self.greens[RZ_key]["Br"] = greens_br
        except:
            self.greens[RZ_key] = {"Br": greens_br}

    def build_control_bz(self, R, Z):
        """
        Compute and cache the Green's function for the vertical magnetic field (Bz)
        induced by the control filaments on a specified R–Z grid.

        The result is computed by averaging the filament-wise Biot–Savart
        contributions and stored in the internal `self.greens` cache using a
        grid-dependent key.

        Parameters
        ----------
        R : np.ndarray
            2D array defining the radial coordinate grid (e.g. equilibrium R grid).
        Z : np.ndarray
            2D array defining the vertical coordinate grid (e.g. equilibrium Z grid).

        Notes
        -----
        - The computation is performed using `GreensBz` evaluated for each filament.
        - Filament contributions are averaged to produce the final field.
        - Results are cached in `self.greens[(R, Z)]["Bz"]` to avoid recomputation.

        Returns
        -------
        None
            The result is stored internally in `self.greens`.
        """

        greens_bz = GreensBz(
            self.filaments[:, 0].reshape([-1] + [1] * R.ndim),
            self.filaments[:, 1].reshape([-1] + [1] * R.ndim),
            R[np.newaxis],
            Z[np.newaxis],
        )
        greens_bz = np.mean(greens_bz, axis=0)

        RZ_key = self.create_RZ_key(R, Z)
        try:
            self.greens[RZ_key]["Bz"] = greens_bz
        except:
            self.greens[RZ_key] = {"Bz": greens_bz}

    def controlPsi(self, R, Z):
        """
        Return the poloidal flux ψ at a given observation point
        due to a unit current in the control element.

        The value is retrieved from a cached Green's function table if available;
        otherwise it is computed and stored.

        Parameters
        ----------
        R : float
            Major radius coordinate of the evaluation point.
        Z : float
            Vertical coordinate of the evaluation point.

        Returns
        -------
        np.ndarray or float
            Green's function contribution to ψ at (R, Z) for unit current.
        """

        RZ_key = self.create_RZ_key(R, Z)
        try:
            greens_ = self.greens[RZ_key]["psi"]
        except:
            self.build_control_psi(R, Z)
            greens_ = self.greens[RZ_key]["psi"]
        return greens_

    def controlBr(self, R, Z):
        """
        Retrieve the radial magnetic field component (Br) at a given
        point (R, Z) due to a unit current source.

        This method accesses a cached Green’s function if available.
        If the requested value is not already cached, it is computed
        on demand and stored for future reuse.

        Parameters
        ----------
        R : float
            Radial coordinate of the evaluation point.
        Z : float
            Vertical coordinate of the evaluation point.

        Returns
        -------
        float or np.ndarray
            Radial magnetic field component Br at (R, Z) due to a unit current.

        Notes
        -----
        - Values are stored in `self.greens` using a key derived from (R, Z).
        - If the entry is missing, `self.build_control_br(R, Z)` is called
        to populate the cache.
        - Repeated queries at the same location are retrieved from cache,
        avoiding recomputation.
        """

        RZ_key = self.create_RZ_key(R, Z)
        try:
            greens_ = self.greens[RZ_key]["Br"]
        except:
            self.build_control_br(R, Z)
            greens_ = self.greens[RZ_key]["Br"]
        return greens_

    def controlBz(self, R, Z):
        """
        Retrieve the vertical magnetic field component (Bz) at a given
        point (R, Z) due to a unit current source.

        The method uses a cached Green’s function lookup when available.
        If the requested value is not present in the cache, it is computed
        on demand and stored for future use.

        Parameters
        ----------
        R : float
            Radial coordinate of evaluation point.
        Z : float
            Vertical coordinate of evaluation point.

        Returns
        -------
        float or np.ndarray
            Bz field contribution at (R, Z) due to a unit current source.

        Notes
        -----
        - Results are cached in `self.greens` indexed by a key derived from (R, Z).
        - If the entry is missing, `self.build_control_bz(R, Z)` is called,
        which populates the cache.
        - Subsequent calls with the same (R, Z) avoid recomputation.
        """

        RZ_key = self.create_RZ_key(R, Z)
        try:
            greens_ = self.greens[RZ_key]["Bz"]
        except:
            self.build_control_bz(R, Z)
            greens_ = self.greens[RZ_key]["Bz"]
        return greens_

    def plot(self, axis=None, show=False):
        """
        Plot the passive structure polygon on a Matplotlib axis.

        This method constructs a `matplotlib.patches.Polygon` from the
        stored vertex coordinates and adds it to the provided axis. If no
        axis is supplied, a new Matplotlib figure and axis are created.

        Parameters
        ----------
        axis : matplotlib.axes.Axes, optional
            Existing Matplotlib axis to plot onto. If None, a new figure
            and axis are created.
        show : bool, optional
            Unused. Present for API compatibility; does not affect behaviour.

        Returns
        -------
        matplotlib.axes.Axes
            Axis containing the plotted polygon.

        Notes
        -----
        - The polygon is also stored as `self.polygon` for later access.
        - The polygon is rendered with fixed styling (grey fill, black edge).
        - This method does not call `plt.show()` even if `show=True`.
        """

        if axis is None:
            fig = plt.figure()
            axis = fig.add_subplot(111)
        self.polygon = Polygon(
            self.vertices, facecolor="grey", edgecolor="k", linewidth=0.75
        )

        axis.add_patch(self.polygon)
        return axis
