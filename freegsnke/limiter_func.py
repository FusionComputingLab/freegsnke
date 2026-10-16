"""
Defines the functionality related to the implementation of the limiter in FreeGSNKE. 

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

from __future__ import annotations

import numpy as np
from matplotlib.path import Path


class Limiter_handler:
    """
    Handles limiter-related geometric operations for FreeGSNKE profile evaluation.

    This class provides functionality for enforcing and querying the plasma
    boundary defined by the limiter geometry. It is primarily used by profile
    objects to determine whether grid points lie inside the allowable plasma
    region and to support limiter-dependent calculations.

    An equilibrium owns one handler; profile objects share its fixed grid geometry
    and limiter masks.

    Notes
    -----
    The limiter is treated as a closed boundary defining the valid plasma domain.
    """

    def __init__(self, eq: object, limiter: object) -> None:
        """
        Object to handle additional calculations due to the limiter.

        This is primarily used by the profile functions.

        Parameters
        ----------
        eq : FreeGSNKE equilibrium object
            Used as a source of info on the solver's grid.
        limiter : a tokamak.Wall object
            Contains a list of R and Z coordinates (the vertices) which define the region accessible to the plasma.
            The boundary itself is the limiter.
        """

        self.limiter = limiter
        self.eqR = eq.R
        self.eqZ = eq.Z
        self.eqR_1D = self.eqR[:, 0]
        self.eqZ_1D = self.eqZ[0, :]

        self.dR = self.eqR[1, 0] - self.eqR[0, 0]
        self.dZ = self.eqZ[0, 1] - self.eqZ[0, 0]
        self.dR_dZ = np.array([self.dR, self.dZ])
        self.R0Z0 = np.array([self.eqR_1D[0], self.eqZ_1D[0]])
        self.dRdZ = self.dR * self.dZ
        self.nx, self.ny = np.shape(eq.R)
        self.nxny = self.nx * self.ny
        self.map2d = np.zeros_like(eq.R)
        self.eqRidx = np.tile(np.arange(self.nx)[:, np.newaxis], (1, self.ny))
        self.eqZidx = np.tile(np.arange(self.ny)[:, np.newaxis], (1, self.nx)).T

        self.validate_limiter_inside_domain()
        self.build_mask_inside_limiter()
        self.mask_outside_limiter = (
            2 * np.logical_not(self.mask_inside_limiter)
        ).astype(float)
        self.limiter_points()
        self.plasma_pts = self.extract_plasma_pts(eq.R, eq.Z, self.mask_inside_limiter)
        self.idxs_mask = self.extract_index_mask(self.mask_inside_limiter)
        self._idx_grid_points = None

    @property
    def idx_grid_points(self) -> np.ndarray:
        """
        Grid indices for contour fallback, built only when first required.

        Returns
        -------
        np.ndarray
            Array of shape (nx*ny, 2) of (R, Z) grid indices.
        """
        if self._idx_grid_points is None:
            self._idx_grid_points = np.column_stack(
                (self.eqRidx.reshape(-1), self.eqZidx.reshape(-1))
            )
        return self._idx_grid_points

    def validate_limiter_inside_domain(self) -> None:
        """
        Raise a clear error if the limiter is not inside the solution domain.

        Limiter boundary interpolation assumes that every limiter segment lies
        strictly inside the rectangular solver grid, so that each refined
        boundary point belongs to a valid interpolation cell.

        Raises
        ------
        ValueError
            If any limiter vertex lies on or outside the edge of the solver grid.
        """
        limiter_R = np.asarray(self.limiter.R)
        limiter_Z = np.asarray(self.limiter.Z)
        Rmin, Rmax = self.eqR_1D[0], self.eqR_1D[-1]
        Zmin, Zmax = self.eqZ_1D[0], self.eqZ_1D[-1]

        limiter_inside_domain = (
            (np.amin(limiter_R) > Rmin)
            and (np.amax(limiter_R) < Rmax)
            and (np.amin(limiter_Z) > Zmin)
            and (np.amax(limiter_Z) < Zmax)
        )
        if not limiter_inside_domain:
            raise ValueError(
                "Limiter coordinates must be strictly inside the solution domain. "
                f"Limiter bounds are R=[{np.amin(limiter_R):.6g}, "
                f"{np.amax(limiter_R):.6g}], Z=[{np.amin(limiter_Z):.6g}, "
                f"{np.amax(limiter_Z):.6g}], while the solution domain is "
                f"R=[{Rmin:.6g}, {Rmax:.6g}], Z=[{Zmin:.6g}, {Zmax:.6g}]."
            )

    def extract_index_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Extract the grid indices of the points selected by a mask (e.g. inside the limiter).

        Parameters
        ----------
        mask : np.ndarray of bool
            Specifies the mask of the relevant region, shape (nx, ny).

        Returns
        -------
        np.ndarray
            Integer array of shape (2, n_points) with the R and Z indices of the
            points where `mask` is True.
        """

        nx, ny = np.shape(mask)
        idxs_mask = np.mgrid[0:nx, 0:ny][np.tile(mask, (2, 1, 1))].reshape(2, -1)

        return idxs_mask

    def extract_plasma_pts(
        self, R: np.ndarray, Z: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """
        Extract R and Z coordinates of the grid points selected by a mask (e.g. inside the limiter).

        Parameters
        ----------
        R : np.ndarray
            R coordinates on the domain grid, e.g. eq.R
        Z : np.ndarray
            Z coordinates on the domain grid, e.g. eq.Z
        mask : np.ndarray of bool
            Specifies the mask of the relevant region

        Returns
        -------
        np.ndarray
            Array of shape (n_points, 2) of (R, Z) coordinates of the masked points.
        """
        plasma_pts = np.concatenate(
            (
                R[mask][:, np.newaxis],
                Z[mask][:, np.newaxis],
            ),
            axis=-1,
        )

        return plasma_pts

    def reduce_rect_domain(self, map: np.ndarray) -> np.ndarray:
        """
        Reduce map from the whole domain to the smallest rectangular domain around limiter mask.

        Requires ``build_reduced_rect_domain`` to have been called (sets ``Rrange``, ``Zrange``).

        Parameters
        ----------
        map : np.ndarray
            Same dimensions as eq.R

        Returns
        -------
        np.ndarray
            The sub-array of `map` covering the reduced rectangular domain.
        """

        return map[self.Rrange[0] : self.Rrange[1], self.Zrange[0] : self.Zrange[1]]

    def build_reduced_rect_domain(
        self,
    ) -> None:
        """
        Build smallest rectangular domain around limiter mask.

        Sets ``Rrange``, ``Zrange`` (index ranges) and ``mask_inside_limiter_red``.
        """

        self.Rrange = (min(self.idxs_mask[0]), max(self.idxs_mask[0]) + 1)
        self.Zrange = (min(self.idxs_mask[1]), max(self.idxs_mask[1]) + 1)

        # self.eqR_red = self.reduce_rect_domain(self.eqR)
        # self.eqZ_red = self.reduce_rect_domain(self.eqZ)
        self.mask_inside_limiter_red = self.reduce_rect_domain(self.mask_inside_limiter)

    def build_mask_inside_limiter(
        self,
    ) -> None:
        """
        Use the coordinates of points along the edge of the limiter region
        to generate the mask of contained domain points.

        Sets ``mask_inside_limiter`` (boolean mask over the full domain of grid
        points inside the limiter region) and ``path`` (the limiter
        ``matplotlib.path.Path``).
        """
        verts = np.concatenate(
            (
                np.array(self.limiter.R)[:, np.newaxis],
                np.array(self.limiter.Z)[:, np.newaxis],
            ),
            axis=-1,
        )
        path = Path(verts)

        points = np.concatenate(
            (self.eqR[:, :, np.newaxis], self.eqZ[:, :, np.newaxis]), axis=-1
        )

        mask_inside_limiter = path.contains_points(points.reshape(-1, 2))
        mask_inside_limiter = mask_inside_limiter.reshape(self.nx, self.ny)
        self.mask_inside_limiter = mask_inside_limiter
        self.path = path

    def broaden_mask(self, mask: np.ndarray, layer_size: int = 3) -> np.ndarray:
        """
        Create a mask that is wider than the input mask, by a width=`layer_size`.

        Parameters
        ----------
        mask : np.ndarray of bool
            Mask to broaden, shape (nx, ny).
        layer_size : int, optional
            Width of the layer, by default 3

        Returns
        -------
        layer_mask : np.ndarray of bool
            Broader mask, same shape as `mask`.
        """
        nx, ny = np.shape(mask)

        layer_mask = np.zeros(
            np.array([nx, ny]) + 2 * np.array([layer_size, layer_size])
        )

        for i in np.arange(-layer_size, layer_size + 1) + layer_size:
            for j in np.arange(-layer_size, layer_size + 1) + layer_size:
                layer_mask[i : i + nx, j : j + ny] += mask
        layer_mask = layer_mask[
            layer_size : layer_size + nx, layer_size : layer_size + ny
        ]
        layer_mask = (layer_mask > 0).astype(bool)
        return layer_mask

    def make_layer_mask(self, mask: np.ndarray, layer_size: int = 3) -> np.ndarray:
        """
        Create a mask for the points just outside the input mask, with a width=`layer_size`.

        Parameters
        ----------
        mask : np.ndarray of bool
            Input mask, shape (nx, ny).
        layer_size : int, optional
            Width of the layer outside the limiter, by default 3

        Returns
        -------
        layer_mask : np.ndarray of bool
            Mask of the points outside the mask within a distance of `layer_size`
        """

        layer_mask = self.broaden_mask(mask, layer_size=layer_size)
        layer_mask = layer_mask * np.logical_not(mask)
        return layer_mask.astype(bool)

    def limiter_points(self, refine: int = 16) -> None:
        """
        Build the refined list of points on the limiter boundary from the limiter vertices.

        These refined boundary points (the intersections of the limiter segments with
        the solver grid lines, plus the vertices) are those on which the flux
        function is interpolated to find the value of psi_boundary in the case of a
        limiter plasma. Sets ``fine_point``, ``grid_per_limiter_fine_point``,
        ``mask_limiter_cells``, ``limiter_mask_out``, ``offending_mask`` and the
        ``fine_point_per_cell*`` dictionaries.

        Parameters
        ----------
        refine : int, optional
            Upsampling ratio with respect to the solver's grid. Currently unused
            (the fine points are the grid-line intersections), by default 16.
        """
        verts = np.concatenate(
            (
                np.array(self.limiter.R)[:, np.newaxis],
                np.array(self.limiter.Z)[:, np.newaxis],
            ),
            axis=-1,
        )
        dverts = verts[1:] - verts[:-1]
        idxR = (
            np.sum(
                np.tile(self.eqR_1D[np.newaxis], (len(verts), 1)) < verts[:, :1], axis=1
            )
            - 1
        )
        idxZ = (
            np.sum(
                np.tile(self.eqZ_1D[np.newaxis], (len(verts), 1)) < verts[:, 1:], axis=1
            )
            - 1
        )

        all_fine_verts = []
        for i in range(len(verts) - 1):

            if dverts[i, 0] == 0:
                # line is vertical
                # line can only intersect the horizontal grid
                min_Z = min(idxZ[i : i + 2])
                max_Z = max(idxZ[i : i + 2])
                Zvals = self.eqZ_1D[min_Z + 1 : max_Z + 1]
                Rvals = np.array([verts[i, 0]] * len(Zvals))
                fine_verts = np.array([Rvals, Zvals]).T

            elif dverts[i, 1] == 0:
                # line is horizontal
                # line can only intersect the vertical grid
                min_R = min(idxR[i : i + 2])
                max_R = max(idxR[i : i + 2])
                Rvals = self.eqR_1D[min_R + 1 : max_R + 1]
                Zvals = np.array([verts[i, 1]] * len(Rvals))
                fine_verts = np.array([Rvals, Zvals]).T

            else:
                # not vertical
                aa = dverts[i, 1] / dverts[i, 0]
                bb = verts[i, 1] - aa * verts[i, 0]

                # add all intersections with the vertical grid
                min_R = min(idxR[i : i + 2])
                max_R = max(idxR[i : i + 2])
                Rvals = self.eqR_1D[min_R + 1 : max_R + 1]
                Zvals = aa * Rvals + bb
                fine_verts = np.array([Rvals, Zvals]).T
                # add all intersections with the horizontal grid
                min_Z = min(idxZ[i : i + 2])
                max_Z = max(idxZ[i : i + 2])
                Zvals = self.eqZ_1D[min_Z + 1 : max_Z + 1]
                Rvals = (Zvals - bb) / aa
                fine_verts = np.concatenate(
                    (fine_verts, np.array([Rvals, Zvals]).T), axis=0
                )

            # add second vertex
            fine_verts = np.concatenate((fine_verts, verts[i + 1 : i + 2]), axis=0)
            # sort locally
            fine_verts = fine_verts[
                np.argsort(np.linalg.norm(fine_verts - verts[i + 1 : i + 2], axis=1))
            ]

            all_fine_verts.append(fine_verts)

        fine_points = np.concatenate(all_fine_verts, axis=0)

        # refined_ddiag = (self.dR**2 + self.dZ**2) ** 0.5 / refine
        # fine_points = []
        # for i in range(1, len(verts)):
        #     dv = verts[i : i + 1] - verts[i - 1 : i]
        #     ndv = np.linalg.norm(dv)
        #     nn = np.round(ndv // refined_ddiag).astype(int)
        #     if nn:
        #         points = dv * np.arange(nn)[:, np.newaxis] / nn
        #         points += verts[i - 1 : i]
        #         fine_points.append(points)
        # fine_points = np.concatenate(fine_points, axis=0)

        # finds the grid vertex with coords just left-below each of the fine_points along the limiter
        Rvals = self.eqR[:, 0]
        Ridxs = np.sum(Rvals[np.newaxis, :] < fine_points[:, :1], axis=1) - 1
        Zvals = self.eqZ[0, :]
        Zidxs = np.sum(Zvals[np.newaxis, :] < fine_points[:, 1:2], axis=1) - 1
        self.grid_per_limiter_fine_point = np.concatenate(
            (Ridxs[:, np.newaxis], Zidxs[:, np.newaxis]), axis=-1
        )
        # saves the mask of all gridpoints that are just left-below any limiter fine-point
        self.mask_limiter_cells = np.zeros_like(self.eqR)
        self.mask_limiter_cells[
            self.grid_per_limiter_fine_point[:, 0],
            self.grid_per_limiter_fine_point[:, 1],
        ] = 1
        self.mask_limiter_cells = self.mask_limiter_cells.astype(bool)

        self.limiter_mask_out = self.make_layer_mask(
            np.logical_not(self.mask_inside_limiter), 1
        )
        self.offending_mask = np.zeros_like(self.eqR).astype(bool)

        self.fine_point_per_cell = {}
        self.fine_point_per_cell_R = {}
        self.fine_point_per_cell_Z = {}
        for i in range(len(fine_points)):
            if (Ridxs[i], Zidxs[i]) not in self.fine_point_per_cell.keys():
                self.fine_point_per_cell[Ridxs[i], Zidxs[i]] = []
                self.fine_point_per_cell_R[Ridxs[i], Zidxs[i]] = []
                self.fine_point_per_cell_Z[Ridxs[i], Zidxs[i]] = []
            self.fine_point_per_cell[Ridxs[i], Zidxs[i]].append(i)
            self.fine_point_per_cell_R[Ridxs[i], Zidxs[i]].append(
                [
                    self.eqR[Ridxs[i] + 1, Zidxs[i]] - fine_points[i, 0],
                    -(self.eqR[Ridxs[i], Zidxs[i]] - fine_points[i, 0]),
                ]
            )
            self.fine_point_per_cell_Z[Ridxs[i], Zidxs[i]].append(
                [
                    [
                        self.eqZ[Ridxs[i], Zidxs[i] + 1] - fine_points[i, 1],
                        -(self.eqZ[Ridxs[i], Zidxs[i]] - fine_points[i, 1]),
                    ]
                ]
            )
        for key in self.fine_point_per_cell.keys():
            self.fine_point_per_cell_R[key] = np.array(self.fine_point_per_cell_R[key])
            self.fine_point_per_cell_Z[key] = np.array(self.fine_point_per_cell_Z[key])
        self.fine_point = fine_points

    def interp_on_limiter_points_cell(
        self, id_R: int, id_Z: int, psi: np.ndarray
    ) -> tuple[np.ndarray | list, list]:
        """
        Bilinearly interpolate the flux function psi in one grid cell.

        The cell is [eq.R[id_R], eq.R[id_R + 1]] x [eq.Z[id_Z], eq.Z[id_Z + 1]]. The
        interpolation is returned directly for the refined points on the limiter
        boundary that fall in that grid cell, as assigned through the
        self.fine_point_per_cell objects.

        Parameters
        ----------
        id_R : int
            Index of the R coordinate for the relevant grid cell.
        id_Z : int
            Index of the Z coordinate for the relevant grid cell.
        psi : np.ndarray
            Values of the total flux function on the solver's grid, shape (nx, ny).

        Returns
        -------
        vals : np.ndarray or list
            Interpolated values of the flux function at the
            self.fine_point_per_cell[id_R, id_Z] locations (empty list if the cell
            contains no limiter points).
        idxs : list
            Indices into ``self.fine_point`` of the corresponding points (empty list
            if the cell contains no limiter points).
        """
        if (id_R, id_Z) in self.fine_point_per_cell_Z.keys():
            ker = psi[id_R : id_R + 2, id_Z : id_Z + 2][np.newaxis, :, :]
            # ker *= self.ker_signs
            vals = np.sum(ker * self.fine_point_per_cell_Z[id_R, id_Z], axis=-1)
            vals = np.sum(vals * self.fine_point_per_cell_R[id_R, id_Z], axis=-1)
            vals /= self.dRdZ
            idxs = self.fine_point_per_cell[id_R, id_Z]
        else:
            vals = []
            idxs = []
        return vals, idxs

    def interp_on_limiter_points(
        self, id_R: int, id_Z: int, psi: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Interpolate psi on the limiter boundary points of the 9 cells around a cell.

        Uses interp_on_limiter_points_cell to interpolate the flux function psi
        on the refined limiter boundary points relevant to the 9 cells
        {id_R-1, id_R, id_R+1} X {id_Z-1, id_Z, id_Z+1}. Interpolated values on the
        boundary points relevant to the cells above are collated and returned.
        Not currently called by ``core_mask_limiter`` (the corresponding code there
        is commented out).

        Parameters
        ----------
        id_R : int
            Index of the R coordinate for the central grid cell.
        id_Z : int
            Index of the Z coordinate for the central grid cell.
        psi : np.ndarray
            Values of the total flux function on the solver's grid, shape (nx, ny).

        Returns
        -------
        vals : np.ndarray
            Interpolated values of the flux function at the self.fine_point_per_cell
            locations relevant to all of the 9 cells.
        idxs : np.ndarray
            Indices into ``self.fine_point`` of the corresponding points.
        """
        vals = []
        idxs = []
        for i in np.arange(-1, 2):
            for j in np.arange(-1, 2):
                vals_, idxs_ = self.interp_on_limiter_points_cell(
                    id_R + i, id_Z + j, psi
                )
                vals = np.concatenate((vals, vals_))
                idxs = np.concatenate((idxs, idxs_))
        # vals = np.concatenate(vals)
        return vals, idxs

    def psi_on_limiter_boundary(self, psi: np.ndarray) -> np.ndarray:
        """Interpolate the flux function on all refined limiter boundary points.

        Parameters
        ----------
        psi : np.ndarray
            Total poloidal flux on the solver grid.

        Returns
        -------
        np.ndarray
            Flux values interpolated on the refined limiter boundary.
        """
        vals = []
        for id_R, id_Z in self.fine_point_per_cell:
            vals_, _ = self.interp_on_limiter_points_cell(id_R, id_Z, psi)
            if len(vals_):
                vals.append(vals_)

        if not vals:
            return np.array([])

        return np.concatenate(vals)

    def core_mask_limiter(
        self,
        psi: np.ndarray,
        psi_bndry: float,
        core_mask: np.ndarray,
        limiter_mask_out: np.ndarray,
        current_sign: float = 1.0,
        #   limiter_mask_in,
        #   linear_coeff=.5
    ) -> tuple[float, np.ndarray, bool]:
        """
        Check if the plasma is in a limiter configuration rather than a diverted one.

        This is obtained by checking whether the core mask deriving from the assumption of a diverted configuration
        implies an overlap with the limiter. If so, an interpolation of psi on the limiter boundary points
        is called to determine the value of psi_boundary and to recalculate the core_mask accordingly.

        Parameters
        ----------
        psi : np.ndarray
            The flux function, including both plasma and metal components.
            np.shape(psi) = (eq.nx, eq.ny)
        psi_bndry : float
            The value of the flux function at the boundary.
            This is xpt[0][2] for a diverted configuration, where xpt is the output of critical.find_critical
        core_mask : np.ndarray
            The mask identifying the plasma region under the assumption of a diverted configuration.
            This is the result of FreeGSNKE's critical.core_mask
            Same size as psi.
        limiter_mask_out : np.ndarray
            The mask identifying the border of the limiter, including points just inside it, the 'last' accessible to the plasma.
            Same size as psi. Currently unused.
        current_sign : float, optional
            Sign of the plasma current, used to orient flux comparisons.

        Returns
        -------
        psi_bndry : float
            The value of the flux function at the boundary.
        core_mask : np.ndarray
            The core mask after correction (float array if no correction was
            applied, as the input is cast to float).
        flag_limiter : bool
            Flag to identify if the plasma is in a diverted or limiter configuration.
        """
        core_mask = core_mask.astype(float)
        # identify the grid points just left-below of points on the limiter that need checking
        offending_mask = (
            core_mask[:-1, :-1]
            + core_mask[1:, :-1]
            + core_mask[:-1, 1:]
            + core_mask[1:, 1:]
        )
        offending_mask = (offending_mask > 0) * (offending_mask < 4)
        self.offending_mask[:-1, :-1] = np.copy(offending_mask)
        self.offending_mask *= self.mask_limiter_cells
        # self.offending_mask = self.offending_mask.astype(bool)

        self.flag_limiter = False

        offending_cells_id_R = self.eqRidx[self.offending_mask]
        offending_cells_id_Z = self.eqZidx[self.offending_mask]

        self.interpolated_on_limiter = []
        for i in range(len(offending_cells_id_R)):
            vals_, idxs_ = self.interp_on_limiter_points_cell(
                offending_cells_id_R[i], offending_cells_id_Z[i], psi
            )
            self.interpolated_on_limiter.append(vals_)

        if len(self.interpolated_on_limiter):
            self.interpolated_on_limiter = np.concatenate(self.interpolated_on_limiter)
            psi_on_limiter = self.interpolated_on_limiter[
                np.argmax(current_sign * self.interpolated_on_limiter)
            ]
            if current_sign * (psi_on_limiter - psi_bndry) > 0:
                self.flag_limiter = True
                psi_bndry = 1.0 * psi_on_limiter
                core_mask = (current_sign * (psi - psi_bndry) > 0) * core_mask

        # if np.any(offending_mask):
        #     # psi_max_out = np.amax(psi[offending_mask])
        #     # psi_max_in = np.amax(psi[(core_mask * limiter_mask_in).astype(bool)])
        #     # psi_bndry = linear_coeff*psi_max_out + (1-linear_coeff)*psi_max_in
        #     # core_mask = (psi > psi_bndry)*core_mask

        #     id_psi_max_out = np.unravel_index(
        #         np.argmax(psi - (10**6) * (1 - offending_mask)), (self.nx, self.ny)
        #     )
        #     self.interpolated_on_limiter, self.interpolated_idxs = self.interp_on_limiter_points(
        #         id_psi_max_out[0], id_psi_max_out[1], psi
        #     )
        #     psi_on_limiter = np.amax(self.interpolated_on_limiter)
        #     if psi_on_limiter > psi_bndry:
        #         self.flag_limiter = True
        #         psi_bndry = 1.0 * psi_on_limiter
        #         core_mask = (psi > psi_bndry) * core_mask

        return psi_bndry, core_mask, self.flag_limiter

    def Iy_from_jtor(self, jtor: np.ndarray) -> np.ndarray:
        """
        Generate 1d vector of plasma current values at the grid points of the reduced plasma domain.

        Parameters
        ----------
        jtor : np.ndarray
            Plasma current density on full domain. np.shape(jtor) = np.shape(eq.R)

        Returns
        -------
        Iy : np.ndarray
            Reduced 1d plasma current vector (jtor * dR * dZ inside the limiter).
        """
        Iy = jtor[self.mask_inside_limiter] * self.dRdZ
        return Iy

    def normalize_sum(self, Iy: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
        """
        Normalise any vector by the linear sum of its elements.

        Parameters
        ----------
        Iy : np.ndarray
            Vector to normalise.
        epsilon : float, optional
            Added to the sum to avoid divergences, by default 1e-6

        Returns
        -------
        np.ndarray
            ``Iy / (sum(Iy) + epsilon)``.
        """
        hat_Iy = Iy / (np.sum(Iy) + epsilon)
        return hat_Iy

    def hat_Iy_from_jtor(self, jtor: np.ndarray) -> np.ndarray:
        """
        Generate 1d vector on reduced plasma domain for the normalised vector
        $$ Jtor*dR*dZ/I_p $$.

        Parameters
        ----------
        jtor : np.ndarray
            Plasma current density on full domain. np.shape(jtor) = np.shape(eq.R)

        Returns
        -------
        hat_Iy : np.ndarray
            Reduced 1d plasma current vector, normalized to its (regularised) sum
        """
        hat_Iy = jtor[self.mask_inside_limiter]
        hat_Iy = self.normalize_sum(hat_Iy)
        return hat_Iy

    def rebuild_map2d(
        self, reduced_vector: np.ndarray, map_dummy: np.ndarray, idxs_mask: np.ndarray
    ) -> np.ndarray:
        """
        Rebuild 2d map on full domain corresponding to 1d vector
        reduced_vector on smaller plasma domain.

        Parameters
        ----------
        reduced_vector : np.ndarray
            1d vector on reduced plasma domain
        map_dummy : np.ndarray
            Specifies the size of the desired rectangular map
        idxs_mask : np.ndarray
            Specifies the location of the pixels of the reduced vector in the rectangular domain
            Note this is specific to map_dummy, e.g. use self.extract_index_mask

        Returns
        -------
        map2d : np.ndarray
            2d map on domain as map_dummy. Values on gridpoints outside the
            reduced plasma domain are set to zero.
        """

        map2d = np.zeros_like(map_dummy.astype(reduced_vector.dtype))
        map2d[idxs_mask[0], idxs_mask[1]] = reduced_vector
        return map2d
