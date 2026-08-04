"""
Contains various functions for refining the plasma current density map.  

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

import numpy as np

from .copying import copy_into


class Jtor_refiner:
    """
    Refines the toroidal plasma current (Jtor) on a structured grid.

    This class implements a local sub-grid refinement strategy for Jtor,
    currently used primarily for Lao85-type profiles when refinement is enabled.

    Refinement is applied selectively to grid cells, typically those:
    - intersected by the separatrix (LCFS),
    - exhibiting large Jtor magnitude, or
    - showing strong spatial gradients.

    The goal is to improve resolution of sharp features without globally
    increasing grid resolution.
    """

    def __init__(self, eq, nnx, nny):
        """
        Initialise the Jtor refiner and precompute geometric and indexing data.

        Parameters
        ----------
        eq : freegs4e.Equilibrium
            Equilibrium object defining the computational grid and limiter geometry.
        nnx : int (even)
            Refinement factor in the R-direction (number of subcells per cell).
        nny : int (even)
            Refinement factor in the Z-direction (number of subcells per cell).

        Notes
        -----
        The grid is assumed to be uniform in R and Z, and refinement is performed
        by subdividing each selected cell into an nnx × nny subgrid.
        """

        self.eqR = eq.R
        self.eqZ = eq.Z
        self.dR = self.eqR[1, 0] - self.eqR[0, 0]
        self.dZ = self.eqZ[0, 1] - self.eqZ[0, 0]
        self.dRdZ = self.dR * self.dZ
        self.nx, self.ny = np.shape(eq.R)
        self.nxny = self.nx * self.ny

        self.nnx = nnx
        self.nny = nny
        self.hnnx = nnx // 2
        self.hnny = nny // 2
        self.nnxy = nnx * nny

        self.path = eq.limiter_handler.path
        self.prepare_for_refinement()

        self.edges_mask = np.ones_like(eq.R)
        self.edges_mask[0, :] = 0
        self.edges_mask[:, 0] = 0
        self.edges_mask[-1, :] = 0
        self.edges_mask[:, -1] = 0

    def copy(self):
        """
        Create a deep copy of the Jtor_refiner object.

        Returns
        -------
        Jtor_refiner
            A new instance with the same configuration and precomputed refinement
            structures.

        Notes
        -----
        - Array-like attributes such as grid geometry and masks are copied using
        `copy_into`.
        - Derived refinement structures are recomputed via `prepare_for_refinement()`.
        - Optional attributes (e.g. LCFS and refinement masks) are copied if present,
        otherwise they are ignored (`strict=False`).
        """
        obj = type(self).__new__(type(self))

        copy_into(self, obj, "eqR", mutable=True)
        copy_into(self, obj, "eqZ", mutable=True)

        copy_into(self, obj, "dR")
        copy_into(self, obj, "dZ")
        copy_into(self, obj, "dRdZ")
        copy_into(self, obj, "nx")
        copy_into(self, obj, "ny")
        copy_into(self, obj, "nxny")
        copy_into(self, obj, "nnx")
        copy_into(self, obj, "nny")
        copy_into(self, obj, "hnnx")
        copy_into(self, obj, "hnny")
        copy_into(self, obj, "nnxy")

        obj.prepare_for_refinement()

        copy_into(self, obj, "edges_mask", mutable=True)
        copy_into(self, obj, "lcfs_mask", mutable=True, strict=False)
        copy_into(self, obj, "value_mask", mutable=True, strict=False)
        copy_into(self, obj, "gradient_mask", mutable=True, strict=False)
        copy_into(self, obj, "mask_to_refine", mutable=True, strict=False)

        return obj

    def prepare_for_refinement(
        self,
    ):
        """
        Precompute geometric, interpolation, and masking structures used in Jtor refinement.

        This method builds all static quantities required for sub-grid refinement,
        including:
        - coarse-grid index maps (Ridx, Zidx),
        - sub-cell coordinate systems,
        - bilinear interpolation weights,
        - limiter-based inside/outside masks at refined resolution,
        - quadrant decomposition masks for vectorised operations.

        These arrays are reused across refinement calls to avoid recomputation.

        Notes
        -----
        - Assumes a uniform structured grid in (R, Z).
        - Sub-cell structure is defined by (nnx × nny) refinement per cell.
        - Limiter geometry is accessed via `self.path.contains_points`.
        """

        self.Ridx = np.tile(np.arange(self.nx), (self.ny, 1)).T
        self.Zidx = np.tile(np.arange(self.ny), (self.nx, 1))

        self.xx = np.linspace(0, 1 - 1 / self.nnx, self.nnx) + 1 / (2 * self.nnx)
        self.yy = np.linspace(0, 1 - 1 / self.nny, self.nny) + 1 / (2 * self.nny)
        self.xxc = self.xx - 0.5
        self.yyc = self.yy - 0.5

        self.xxx = np.concatenate(
            (1 - self.xx[:, np.newaxis], self.xx[:, np.newaxis]), axis=-1
        )
        self.yyy = np.concatenate(
            (1 - self.yy[:, np.newaxis], self.yy[:, np.newaxis]), axis=-1
        )
        self.xxxx = np.concatenate(
            (
                self.xxx[np.newaxis, : self.hnnx],
                self.xxx[np.newaxis, self.hnnx :],
                self.xxx[np.newaxis, self.hnnx :],
                self.xxx[np.newaxis, : self.hnnx],
            ),
            axis=0,
        )
        self.yyyy = np.concatenate(
            (
                self.yyy[np.newaxis, : self.hnny],
                self.yyy[np.newaxis, : self.hnny],
                self.yyy[np.newaxis, self.hnny :],
                self.yyy[np.newaxis, self.hnny :],
            ),
            axis=0,
        )

        fullr = np.tile(
            (
                self.eqR[:, :, np.newaxis]
                + self.dR * self.xxc[np.newaxis, np.newaxis, :]
            )[:, :, :, np.newaxis],
            [1, 1, 1, self.nny],
        )
        fullz = np.tile(
            (
                self.eqZ[:, :, np.newaxis]
                + self.dZ * self.yyc[np.newaxis, np.newaxis, :]
            )[:, :, np.newaxis, :],
            [1, 1, self.nnx, 1],
        )
        fullg = np.concatenate(
            (fullr[:, :, :, :, np.newaxis], fullz[:, :, :, :, np.newaxis]), axis=-1
        )
        full_masks = self.path.contains_points(fullg.reshape(-1, 2))
        # these are the refined masks of points inside the limiter
        self.full_masks = full_masks.reshape(self.nx, self.ny, self.nnx, self.nny)

        srr, szz = np.meshgrid(np.arange(self.nnx), np.arange(self.nny), indexing="ij")
        quartermasks = np.zeros((self.nnx, self.nny, 4))
        quartermasks[:, :, 2] = (srr < (self.nnx / 2)) * (szz < (self.nny / 2))
        quartermasks[:, :, 3] = (srr >= (self.nnx / 2)) * (szz < (self.nny / 2))
        quartermasks[:, :, 1] = (srr < (self.nnx / 2)) * (szz >= (self.nny / 2))
        quartermasks[:, :, 0] = (srr >= (self.nnx / 2)) * (szz >= (self.nny / 2))
        self.quartermasks = quartermasks

    def get_indexes_for_refinement(self, mask_to_refine):
        """
        Construct index arrays for bilinear interpolation on refined cells.

        For each selected coarse grid cell, this function returns the indices of
        the 2×2 stencil (four surrounding vertices) required to perform bilinear
        interpolation of ψ on the refined subgrid.

        Parameters
        ----------
        mask_to_refine : np.ndarray (bool)
            Boolean mask of coarse grid cells selected for refinement.

        Returns
        -------
        RRidxs : np.ndarray
            R-index stencil for each refined cell, shape
            (n_cells, 2, 2, 2) depending on internal packing.
        ZZidxs : np.ndarray
            Z-index stencil matching RRidxs structure.

        Notes
        -----
        Each cell contributes a structured 2×2 vertex stencil used for vectorised
        bilinear interpolation of ψ on the nnx × nny subgrid.
        """
        RRidxs = np.concatenate(
            (
                np.concatenate(
                    (
                        np.concatenate(
                            (
                                self.Ridx[mask_to_refine][:, np.newaxis],
                                self.Ridx[mask_to_refine][:, np.newaxis],
                            ),
                            axis=-1,
                        )[:, np.newaxis, :],
                        np.concatenate(
                            (
                                self.Ridx[mask_to_refine][:, np.newaxis] + 1,
                                self.Ridx[mask_to_refine][:, np.newaxis] + 1,
                            ),
                            axis=-1,
                        )[:, np.newaxis, :],
                    ),
                    axis=1,
                )[:, np.newaxis, :, :],
                np.concatenate(
                    (
                        np.concatenate(
                            (
                                self.Ridx[mask_to_refine][:, np.newaxis] - 1,
                                self.Ridx[mask_to_refine][:, np.newaxis] - 1,
                            ),
                            axis=-1,
                        )[:, np.newaxis, :],
                        np.concatenate(
                            (
                                self.Ridx[mask_to_refine][:, np.newaxis],
                                self.Ridx[mask_to_refine][:, np.newaxis],
                            ),
                            axis=-1,
                        )[:, np.newaxis, :],
                    ),
                    axis=1,
                )[:, np.newaxis, :, :],
                np.concatenate(
                    (
                        np.concatenate(
                            (
                                self.Ridx[mask_to_refine][:, np.newaxis] - 1,
                                self.Ridx[mask_to_refine][:, np.newaxis] - 1,
                            ),
                            axis=-1,
                        )[:, np.newaxis, :],
                        np.concatenate(
                            (
                                self.Ridx[mask_to_refine][:, np.newaxis],
                                self.Ridx[mask_to_refine][:, np.newaxis],
                            ),
                            axis=-1,
                        )[:, np.newaxis, :],
                    ),
                    axis=1,
                )[:, np.newaxis, :, :],
                np.concatenate(
                    (
                        np.concatenate(
                            (
                                self.Ridx[mask_to_refine][:, np.newaxis],
                                self.Ridx[mask_to_refine][:, np.newaxis],
                            ),
                            axis=-1,
                        )[:, np.newaxis, :],
                        np.concatenate(
                            (
                                self.Ridx[mask_to_refine][:, np.newaxis] + 1,
                                self.Ridx[mask_to_refine][:, np.newaxis] + 1,
                            ),
                            axis=-1,
                        )[:, np.newaxis, :],
                    ),
                    axis=1,
                )[:, np.newaxis, :, :],
            ),
            axis=1,
        )

        ZZidxs = np.concatenate(
            (
                np.concatenate(
                    (
                        np.concatenate(
                            (
                                self.Zidx[mask_to_refine][:, np.newaxis],
                                self.Zidx[mask_to_refine][:, np.newaxis] + 1,
                            ),
                            axis=-1,
                        )[:, np.newaxis, :],
                        np.concatenate(
                            (
                                self.Zidx[mask_to_refine][:, np.newaxis],
                                self.Zidx[mask_to_refine][:, np.newaxis] + 1,
                            ),
                            axis=-1,
                        )[:, np.newaxis, :],
                    ),
                    axis=1,
                )[:, np.newaxis, :, :],
                np.concatenate(
                    (
                        np.concatenate(
                            (
                                self.Zidx[mask_to_refine][:, np.newaxis],
                                self.Zidx[mask_to_refine][:, np.newaxis] + 1,
                            ),
                            axis=-1,
                        )[:, np.newaxis, :],
                        np.concatenate(
                            (
                                self.Zidx[mask_to_refine][:, np.newaxis],
                                self.Zidx[mask_to_refine][:, np.newaxis] + 1,
                            ),
                            axis=-1,
                        )[:, np.newaxis, :],
                    ),
                    axis=1,
                )[:, np.newaxis, :, :],
                np.concatenate(
                    (
                        np.concatenate(
                            (
                                self.Zidx[mask_to_refine][:, np.newaxis] - 1,
                                self.Zidx[mask_to_refine][:, np.newaxis],
                            ),
                            axis=-1,
                        )[:, np.newaxis, :],
                        np.concatenate(
                            (
                                self.Zidx[mask_to_refine][:, np.newaxis] - 1,
                                self.Zidx[mask_to_refine][:, np.newaxis],
                            ),
                            axis=-1,
                        )[:, np.newaxis, :],
                    ),
                    axis=1,
                )[:, np.newaxis, :, :],
                np.concatenate(
                    (
                        np.concatenate(
                            (
                                self.Zidx[mask_to_refine][:, np.newaxis] - 1,
                                self.Zidx[mask_to_refine][:, np.newaxis],
                            ),
                            axis=-1,
                        )[:, np.newaxis, :],
                        np.concatenate(
                            (
                                self.Zidx[mask_to_refine][:, np.newaxis] - 1,
                                self.Zidx[mask_to_refine][:, np.newaxis],
                            ),
                            axis=-1,
                        )[:, np.newaxis, :],
                    ),
                    axis=1,
                )[:, np.newaxis, :, :],
            ),
            axis=1,
        )

        return RRidxs, ZZidxs

    def build_jtor_value_mask(self, unrefined_jtor, threshold, quantiles=(0.5, 0.9)):
        """
        Construct a refinement mask based on the magnitude of Jtor.

        Cells are selected for refinement if their Jtor value exceeds a scaled
        threshold defined relative to two quantiles of the Jtor distribution.

        Parameters
        ----------
        unrefined_jtor : np.ndarray
            Coarse-grid toroidal current density field.
        threshold : float
            Scaling factor applied to the inter-quantile range to determine
            refinement sensitivity.
        quantiles : tuple of float, optional
            Two quantiles (q_low, q_high) used to define a reference range.
            Default is (0.5, 0.9).

        Returns
        -------
        np.ndarray (bool)
            Boolean mask indicating cells selected for refinement.
        """

        jtor_quantiles = np.quantile(unrefined_jtor.reshape(-1), quantiles)
        mask = (unrefined_jtor - jtor_quantiles[0]) > threshold * (
            jtor_quantiles[1] - jtor_quantiles[0]
        )
        return mask

    def build_jtor_gradient_mask(self, unrefined_jtor, threshold, quantiles=(0.5, 0.9)):
        """
        Construct a refinement mask based on local finite-difference variations of Jtor.

        Cells are selected when the magnitude of local Jtor differences between
        neighbouring grid points exceeds a threshold derived from the distribution
        of these differences.

        The method approximates spatial variation using forward finite differences
        in both R and Z directions, and applies a quantile-based thresholding
        strategy (via `build_jtor_value_mask`) to identify large-gradient regions.

        Parameters
        ----------
        unrefined_jtor : np.ndarray
            Coarse-grid toroidal current density field.
        threshold : float
            Scaling factor applied to the inter-quantile range of gradient
            magnitudes used for refinement.
        quantiles : tuple of float, optional
            (q_low, q_high) quantiles used to normalise gradient magnitude
            thresholds. Default is (0.5, 0.9).

        Returns
        -------
        np.ndarray (bool)
            Boolean mask indicating cells selected for refinement.
        """
        gradient_mask = np.zeros_like(unrefined_jtor)

        # right
        right_gradient = np.abs(unrefined_jtor[:-1, :-1] - unrefined_jtor[1:, :-1])
        right_gradient = self.build_jtor_value_mask(
            right_gradient, threshold, quantiles
        )
        # include both indexes in refinement:
        gradient_mask[:-1, :-1] += right_gradient
        gradient_mask[1:, :-1] += right_gradient

        # up
        up_gradient = np.abs(unrefined_jtor[:-1, :-1] - unrefined_jtor[:-1, 1:])
        up_gradient = self.build_jtor_value_mask(up_gradient, threshold, quantiles)
        # include both indexes in refinement:
        gradient_mask[:-1, :-1] += up_gradient
        gradient_mask[:-1, 1:] += up_gradient

        return gradient_mask > 0

    def build_LCFS_mask(self, core_mask):
        """
        Construct a refinement mask identifying grid points adjacent to the LCFS.

        The mask is built by detecting interfaces between plasma and vacuum
        cells in the binary `core_mask`. Any grid cell that shares an edge
        with a cell of opposite classification (inside/outside plasma core)
        is marked for refinement.

        The resulting mask is additive: cells adjacent to multiple LCFS edges
        may accumulate values greater than 1. This is primarily used as a
        selection indicator, not a strict boolean mask.

        Parameters
        ----------
        core_mask : np.ndarray of shape (nx, ny)
            Binary plasma core mask on the structured grid.
            Expected values:
                - 1 (or True): inside plasma core
                - 0 (or False): outside plasma core

        Returns
        -------
        np.ndarray of shape (nx, ny)
            LCFS refinement indicator array. Entries are non-zero where grid
            cells are adjacent to a change in `core_mask` across a grid edge.
            Higher values indicate multiple adjacent LCFS crossings.
        """

        core_mask = core_mask.astype(float)
        lcfs_mask = np.zeros_like(core_mask)
        # right
        right_mask = core_mask[:-1, :] + core_mask[1:, :] == 1
        # include both indexes in refinement:
        lcfs_mask[:-1, :] += right_mask
        lcfs_mask[1:, :] += right_mask
        # # include one more pixel
        # lcfs_mask[:-2, :] += right_mask[1:,:]
        # lcfs_mask[2:, :] += right_mask[:-1,:]
        # up
        up_mask = core_mask[:, :-1] + core_mask[:, 1:] == 1
        # include both indexes in refinement:
        lcfs_mask[:, :-1] += up_mask
        lcfs_mask[:, 1:] += up_mask
        # # include one more pixel
        # lcfs_mask[:, :-2] += up_mask[:,1:]
        # lcfs_mask[:, 2:] += up_mask[:,:-1]
        return lcfs_mask

    def build_mask_to_refine(self, unrefined_jtor, core_mask, thresholds):
        """
        Construct the global refinement mask combining LCFS location,
        Jtor magnitude, and Jtor gradient criteria.

        This method aggregates multiple refinement indicators into a single
        cell-wise mask. A cell is marked for refinement if it satisfies any
        of the following conditions:

            1. It lies adjacent to the LCFS (core–vacuum interface)
            2. Its Jtor value exceeds a threshold based on distribution quantiles
            3. Its Jtor gradient exceeds a threshold based on distribution quantiles

        Boundary cells are excluded from refinement.

        The intermediate masks are also stored as attributes for diagnostics:
            - self.lcfs_mask
            - self.value_mask
            - self.gradient_mask

        Parameters
        ----------
        unrefined_jtor : np.ndarray of shape (nx, ny)
            Toroidal current density on the coarse grid.

        core_mask : np.ndarray of shape (nx, ny)
            Binary mask identifying plasma core cells.
            Typically 1 inside plasma, 0 outside.

        thresholds : tuple of float
            (jtor_threshold, gradient_threshold)
            Scaling factors applied to inter-quantile ranges used to define
            refinement sensitivity.

        Returns
        -------
        None
            The result is stored in:
                self.mask_to_refine : np.ndarray (bool)
        """

        mask_to_refine = np.zeros_like(unrefined_jtor)

        # include all cells that are crossed by the lcfs:
        self.lcfs_mask = self.build_LCFS_mask(core_mask)
        mask_to_refine += self.lcfs_mask

        # include cells that warrant refinement according to criterion on jtor value:
        self.value_mask = self.build_jtor_value_mask(unrefined_jtor, thresholds[0])
        mask_to_refine += self.value_mask

        # include cells that warrant refinement according to criterion on gradient value:
        self.gradient_mask = self.build_jtor_gradient_mask(
            unrefined_jtor, thresholds[1]
        )
        mask_to_refine += self.gradient_mask

        # remove all edges, as these cannot be refined
        mask_to_refine *= self.edges_mask

        # make bool mask
        self.mask_to_refine = mask_to_refine.astype(bool)

    def build_bilinear_psi_interp(self, psi, core_mask, unrefined_jtor, thresholds):
        """
        Construct a refined representation of the poloidal flux `psi` on a
        sub-grid using bilinear interpolation in selected refinement cells.

        Cells are selected for refinement based on a combined criterion:
            - proximity to the LCFS (core–vacuum interface)
            - large values of Jtor (based on quantile thresholding)
            - large gradients in Jtor (based on quantile thresholding)

        For each selected coarse-grid cell, a higher-resolution sub-grid is
        generated and psi is reconstructed via bilinear interpolation.

        Parameters
        ----------
        psi : np.ndarray of shape (nx, ny)
            Poloidal flux on the coarse grid.

        core_mask : np.ndarray of shape (nx, ny)
            Binary plasma core mask defining LCFS location.

        unrefined_jtor : np.ndarray of shape (nx, ny)
            Toroidal current density on the coarse grid.

        thresholds : tuple of float
            (jtor_threshold, gradient_threshold)
            Scaling factors used in refinement criteria via inter-quantile ranges.

        Returns
        -------
        format_bilinear_psi : np.ndarray of shape (n_refined, nnx, nny)
            Bilinearly interpolated psi values on refined sub-grids for each
            selected coarse cell.

        refined_R : np.ndarray of shape (n_refined, nnx, nny)
            R-coordinate values corresponding to each refined sub-grid point.

        Notes
        -----
        - The refinement mask is computed internally via `build_mask_to_refine`.
        - Each selected coarse cell is subdivided into an `nnx × nny` sub-grid.
        - Bilinear interpolation is performed using precomputed vertex weights
        (`self.xxxx`, `self.yyyy`).
        - Output is structured per refined cell, not a full fine global grid.
        """

        self.build_mask_to_refine(unrefined_jtor, core_mask, thresholds)

        # this is a vector of R values at the refined calculation points
        refined_R = np.tile(
            (
                self.eqR[self.Ridx[self.mask_to_refine], 0][:, np.newaxis]
                + self.dR * self.xxc[np.newaxis, :]
            )[:, :, np.newaxis],
            (1, 1, self.nny),
        )

        # build refined psi
        # get indexes to build psi for bilinear interp
        RRidxs, ZZidxs = self.get_indexes_for_refinement(self.mask_to_refine)
        # this is psi on the vertices as needed for each grid point to be refined
        psi_where_needed = psi[RRidxs, ZZidxs]
        # this is psi refined at the refined calculation points
        bilinear_psi = np.sum(
            np.sum(
                psi_where_needed[:, :, np.newaxis, :, :]
                * self.yyyy[np.newaxis, :, :, np.newaxis, :],
                -1,
            )[:, :, np.newaxis, :, :]
            * self.xxxx[np.newaxis, :, :, np.newaxis, :],
            axis=-1,
        )
        # reformat so to have same structure as refined_R
        format_bilinear_psi = np.zeros(
            (np.sum(self.mask_to_refine), self.nnx, self.nny)
        )
        format_bilinear_psi[:, self.hnnx :, self.hnny :] = bilinear_psi[:, 0]
        format_bilinear_psi[:, : self.hnnx, self.hnny :] = bilinear_psi[:, 1]
        format_bilinear_psi[:, : self.hnnx :, : self.hnny] = bilinear_psi[:, 2]
        format_bilinear_psi[:, self.hnnx :, : self.hnny] = bilinear_psi[:, 3]

        return format_bilinear_psi, refined_R

    def build_from_refined_jtor(self, unrefined_jtor, refined_jtor):
        """
        Reconstruct a coarse-grid Jtor field by averaging refined sub-grid values
        back onto the original (nx, ny) mesh.

        For each selected coarse cell, the corresponding refined sub-grid
        (nnx × nny) is first masked to remove points outside the limiter.
        The remaining values are then spatially averaged and used to replace
        the coarse-grid value.

        Parameters
        ----------
        unrefined_jtor : np.ndarray of shape (nx, ny)
            Original coarse-grid toroidal current density.

        refined_jtor : np.ndarray of shape (n_refined, nnx, nny)
            Refined Jtor values on sub-grids for each selected coarse cell.

        Returns
        -------
        np.ndarray of shape (nx, ny)
            Updated Jtor field where selected cells have been replaced by
            averaged refined values and all other cells remain unchanged.

        Notes
        -----
        - Refinement contributions are masked using `self.full_masks` to
        exclude points outside the limiter.
        - Each coarse cell is updated independently; no smoothing is applied
        between neighboring refined regions.
        - The output preserves the original grid structure.
        """
        # mask out refinement points that are outside the limiter
        masked_refined_jtor = (
            refined_jtor
            * self.full_masks[
                self.Ridx[self.mask_to_refine], self.Zidx[self.mask_to_refine], :, :
            ]
        )
        # average in each refinement region
        masked_refined_jtor = np.sum(masked_refined_jtor, axis=(1, 2)) / self.nnxy

        # assign to jtor
        jtor = 1.0 * unrefined_jtor
        jtor[self.Ridx[self.mask_to_refine], self.Zidx[self.mask_to_refine]] = (
            masked_refined_jtor
        )

        return jtor
