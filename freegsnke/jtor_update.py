"""
Defines the FreeGSNKE profile Object, which inherits from the FreeGS4E profile object.

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

import freegs4e.jtor
import numpy as np
from freegs4e.gradshafranov import mu0
from matplotlib.path import Path
from scipy.ndimage import maximum_filter
from skimage import measure

from . import jtor_refinement
from . import switch_profile as swp
from .copying import copy_into


class Jtor_universal:
    def __init__(self, refine_jtor=False):
        """Sets default unrefined Jtor."""
        self._refine_jtor = refine_jtor

    def Jtor(self, *args, **kwargs):
        if self._refine_jtor:
            return self.Jtor_refined(*args, **kwargs)
        else:
            return self.Jtor_unrefined(*args, **kwargs)

    def copy(self, obj=None):
        """Creates a copy the object.

        obj : Jtor_universal
            An instance of self that the attributes are copied into instead of
            creating a new object
        """

        obj = type(self).__new__(type(self)) if obj is None else obj

        copy_into(self, obj, "_refine_jtor")
        copy_into(self, obj, "dR")
        copy_into(self, obj, "dZ")
        copy_into(self, obj, "dRdZ")
        copy_into(self, obj, "nx")

        copy_into(self, obj, "dR_dZ", mutable=True)
        copy_into(self, obj, "grid_points", mutable=True)
        copy_into(self, obj, "eqRidx", mutable=True)
        copy_into(self, obj, "eqZidx", mutable=True)
        copy_into(self, obj, "idx_grid_points", mutable=True)
        copy_into(self, obj, "R0Z0", mutable=True)
        copy_into(self, obj, "mask_inside_limiter", mutable=True)
        copy_into(self, obj, "mask_outside_limiter", mutable=True)
        copy_into(self, obj, "limiter_mask_out", mutable=True)
        copy_into(self, obj, "limiter_mask_for_plotting", mutable=True)
        copy_into(self, obj, "edge_mask", mutable=True)
        obj.inputs = self.inputs[::]  # shallow copy suffices

        # *Should* not be necessary to copy this
        obj.limiter_handler = self.limiter_handler

        # the following attributes won't always be present...
        if hasattr(self, "jtor_refiner"):
            obj.refinement_thresholds = self.refinement_thresholds[::]
            obj.jtor_refiner = self.jtor_refiner.copy()

        copy_into(self, obj, "psi_bndry", strict=False)
        copy_into(self, obj, "psi_axis", strict=False)
        copy_into(self, obj, "psi_axis", strict=False)
        copy_into(self, obj, "flag_limiter", strict=False)
        copy_into(self, obj, "Ip_logic", strict=False)

        copy_into(self, obj, "psi_map", mutable=True, strict=False)
        copy_into(
            self,
            obj,
            "record_xpt",
            mutable=True,
            strict=False,
            allow_deepcopy=True,
        )
        copy_into(self, obj, "lcfs", mutable=True, strict=False)
        copy_into(self, obj, "jtor", mutable=True, strict=False)
        copy_into(self, obj, "diverted_core_mask", mutable=True, strict=False)
        copy_into(self, obj, "limiter_core_mask", mutable=True, strict=False)
        copy_into(self, obj, "unrefined_jtor", mutable=True, strict=False)
        copy_into(self, obj, "unrefined_djtordpsi", mutable=True, strict=False)
        copy_into(self, obj, "pure_jtor", mutable=True, strict=False)
        copy_into(self, obj, "pure_djtordpsi", mutable=True, strict=False)
        copy_into(self, obj, "dJtordpsi", mutable=True, strict=False)

        copy_into(self, obj, "xpt", mutable=True, strict=False, allow_deepcopy=True)
        copy_into(self, obj, "opt", mutable=True, strict=False, allow_deepcopy=True)

        return obj

    def set_masks(self, eq):
        """Universal function to set all masks related to the limiter.

        Parameters
        ----------
        eq : FreeGSNKE Equilibrium object
            Specifies the domain properties
        """
        self.dR = eq.R_1D[1] - eq.R_1D[0]
        self.dZ = eq.Z_1D[1] - eq.Z_1D[0]
        self.dR_dZ = np.array([self.dR, self.dZ])
        self.R0Z0 = np.array([eq.R_1D[0], eq.Z_1D[0]])
        self.dRdZ = self.dR * self.dZ
        self.grid_points = np.concatenate(
            (eq.R[:, :, np.newaxis], eq.Z[:, :, np.newaxis]), axis=-1
        )
        self.nx, self.ny = np.shape(eq.R)
        self.eqRidx = np.tile(np.arange(self.nx)[:, np.newaxis], (1, self.ny))
        self.eqZidx = np.tile(np.arange(self.ny)[:, np.newaxis], (1, self.nx)).T
        self.idx_grid_points = np.concatenate(
            (self.eqRidx[:, :, np.newaxis], self.eqZidx[:, :, np.newaxis]), axis=-1
        ).reshape(-1, 2)

        self.limiter_handler = eq.limiter_handler

        # self.core_mask_limiter = eq.limiter_handler.core_mask_limiter

        self.mask_inside_limiter = eq.limiter_handler.mask_inside_limiter

        mask_outside_limiter = np.logical_not(eq.limiter_handler.mask_inside_limiter)
        # Note the factor 2 is not a typo: used in critical.inside_mask
        self.mask_outside_limiter = (2 * mask_outside_limiter).astype(float)

        self.limiter_mask_out = eq.limiter_handler.limiter_mask_out

        self.limiter_mask_for_plotting = (
            eq.limiter_handler.mask_inside_limiter
            + eq.limiter_handler.make_layer_mask(
                eq.limiter_handler.mask_inside_limiter, layer_size=1
            )
        ) > 0

        # set mask of the edge domain pixels
        self.edge_mask = np.zeros_like(eq.R)
        self.edge_mask[0, :] = self.edge_mask[:, 0] = self.edge_mask[-1, :] = (
            self.edge_mask[:, -1]
        ) = 1

    def select_refinement(self, eq, refine_jtor, nnx, nny):
        """Initializes the object that handles the subgrid refinement of jtor

        Parameters
        ----------
        eq : freegs4e Equilibrium object
            Specifies the domain properties
        refine_jtor : bool
            Flag to select whether to apply sug-grid refinement of plasma current distribution jtor
        nnx : even integer
            refinement factor in the R direction
        nny : even integer
            refinement factor in the Z direction
        """
        self._refine_jtor = refine_jtor
        if refine_jtor:
            self.jtor_refiner = jtor_refinement.Jtor_refiner(eq, nnx, nny)
            self.set_refinement_thresholds()

    def set_refinement_thresholds(self, thresholds=(1.0, 1.0)):
        """Sets the default criteria for refinement -- used when not directly set.

        Parameters
        ----------
        thresholds : tuple (threshold for jtor criterion, threshold for gradient criterion)
            tuple of values used to identify where to apply refinement
        """
        self.refinement_thresholds = thresholds

    # def all_open(self, contours):
    #     checks = []
    #     for contour in contours:
    #         checks.append(
    #             np.any(
    #                 [
    #                     np.any(contour[:, 0] <= 1),
    #                     np.any(contour[:, 0] >= self.nx - 2),
    #                     np.any(contour[:, 1] <= 1),
    #                     np.any(contour[:, 1] >= self.ny - 2),
    #                 ]
    #             )
    #         )
    #     return np.all(checks), checks

    def diverted_critical(
        self,
        R,
        Z,
        psi,
        psi_bndry=None,
        mask_outside_limiter=None,
        rel_tolerance_xpt=1e-10,
        starting_dx=0.05,
    ):
        """
        Replaces Jtor_part1 when that fails. Implements a new algorithm to define the LCFS.
        This is considerably more time consuming, but essential when the default routines in
        critical fail, as for example when the Xpt is not correctly identified.


        Parameters
        ----------
        R : np.ndarray
            Radial coordinates of the grid points.
        Z : np.ndarray
            Vertical coordinates of the grid points.
        psi : np.ndarray
            Total poloidal field flux at each grid point [Webers/2pi].
        psi_bndry : float, optional
            Value of the poloidal field flux at the boundary of the plasma (last closed
            flux surface).
        mask_outside_limiter : np.ndarray
            Mask of points outside the limiter, if any.

        Returns
        -------
        np.array
            Each row represents an O-point of the form [R, Z, ψ(R,Z)] [m, m, Webers/2pi].
        np.array
            Each row represents an X-point of the form [R, Z, ψ(R,Z)] [m, m, Webers/2pi].
        np.bool
            An array, the same shape as the computational grid, indicating the locations
            at which the core plasma resides (True) and where it does not (False).
        float
            Value of the poloidal field flux at the boundary of the plasma (last closed
            flux surface).
        """

        # prepare psi_map to use
        psi_map = np.copy(psi)
        self.psi_map = psi_map
        min_psi = np.amin(psi_map)
        psi_map[:, 0] = psi_map[0, :] = psi_map[-1, :] = psi_map[:, -1] = min_psi
        del_psi = np.amax(psi_map) - min_psi
        psi_map /= del_psi

        # find all the local maxima
        maxima_psi_mask = (maximum_filter(psi_map, size=3)) == psi_map
        # select those inside the limiter region
        maxima_psi_mask_in = maxima_psi_mask * self.mask_inside_limiter
        if np.sum(maxima_psi_mask_in) < 1:
            raise ValueError(
                "No O-point in the limiter region. Guess psi_plasma is likely inappropriate."
            )

        # identify the location of the local maximum inside the limiter
        valid_max_psi = np.amax(psi_map[maxima_psi_mask_in])
        mask = psi_map * maxima_psi_mask_in == valid_max_psi
        idx_valid_max = np.array([self.eqRidx[mask][0], self.eqZidx[mask][0]])

        # select the local maxima outside the limiter region
        maxima_psi_mask_out = maxima_psi_mask * mask_outside_limiter
        # include the edges of the map to the excluded region
        maxima_psi_mask_out[1, :] = maxima_psi_mask_out[:, 1] = maxima_psi_mask_out[
            -1, :
        ] = maxima_psi_mask_out[:, -1] = True
        maxima_psi_mask_out = maxima_psi_mask_out.astype(bool)
        idx_excluded_max = np.array(
            [self.eqRidx[maxima_psi_mask_out], self.eqZidx[maxima_psi_mask_out]]
        ).T

        # start root finding for the xpoint flux value
        increment = -starting_dx
        desired_check_larger = True
        current_psi_level = valid_max_psi + increment
        self.record_xpt = [valid_max_psi, current_psi_level]

        while abs(increment) > rel_tolerance_xpt or desired_check_larger is False:
            # design regions
            all_regions = measure.find_contours(psi_map, current_psi_level)
            # sort them by distance to the valid maximum
            mean_dist = [
                np.linalg.norm(np.mean(region, axis=0) - idx_valid_max)
                for region in all_regions
            ]
            regions_order = np.argsort(mean_dist)
            # identify the region containing the valid local maximum
            region_found = False
            idx = -1
            while region_found is False:
                idx += 1
                path = Path(all_regions[regions_order[idx]])
                region_found = path.contains_point(idx_valid_max)
            # check if any excluded points have been included
            check_larger = np.any(path.contains_points(idx_excluded_max.astype(float)))
            if check_larger == desired_check_larger:
                # invert sign and decrease size
                desired_check_larger = np.logical_not(desired_check_larger)
                increment *= -0.5
            # else:
            # keep exploring in the same direction
            # so no action needed
            current_psi_level += increment
            self.record_xpt.append(current_psi_level)

        # build opt, xpt and diverted core mask accordingly
        self.lcfs = all_regions[regions_order[idx]][:-1]
        self.lcfs = self.lcfs * self.dR_dZ[np.newaxis] + self.R0Z0[np.newaxis]
        # build xpt
        psi_bndry = current_psi_level * del_psi
        dist = np.linalg.norm(
            self.lcfs[:, np.newaxis] - self.lcfs[np.newaxis, :], axis=-1
        ) + 10 * np.eye(len(self.lcfs))
        mask = dist == np.amin(dist)
        xpt_coords = (
            np.mean(self.lcfs[np.any(mask, axis=0)], axis=0) * self.dR_dZ + self.R0Z0
        )
        xpt = np.concatenate((xpt_coords, [psi_bndry]))[np.newaxis]
        # build opt
        opt = np.concatenate(
            (idx_valid_max * self.dR_dZ + self.R0Z0, [valid_max_psi * del_psi])
        )[np.newaxis]
        # build diverted_core_mask
        diverted_core_mask = path.contains_points(self.idx_grid_points).reshape(
            (self.nx, self.ny)
        )

        return opt, xpt, diverted_core_mask, psi_bndry

    def diverted_critical_complete(
        self,
        R,
        Z,
        psi,
        psi_bndry=None,
        mask_outside_limiter=None,
        rel_tolerance_xpt=1e-4,
        starting_dx=0.05,
    ):
        try:
            opt, xpt, diverted_core_mask, psi_bndry = self.Jtor_part1(
                R, Z, psi, psi_bndry, mask_outside_limiter
            )
        except:
            opt, xpt, diverted_core_mask, psi_bndry = self.diverted_critical(
                R,
                Z,
                psi,
                psi_bndry,
                mask_outside_limiter,
                rel_tolerance_xpt,
                starting_dx,
            )

        return opt, xpt, diverted_core_mask, psi_bndry

    # def diverted_critical_old(self, R, Z, psi, psi_bndry=None, mask_outside_limiter=None, xpt_tol=1e-4):
    #     # this

    #     # find O- and X-points of equilibrium
    #     opt, xpt = critical.fastcrit(
    #         R, Z, psi, self.mask_inside_limiter, #self.Ip
    #     )
    #     len_xpt = len(xpt)
    #     len_opt = len(opt)

    #     # find core plasma mask (using user-defined psi_bndry)
    #     if psi_bndry is not None:
    #         diverted_core_mask = critical.inside_mask(
    #             R, Z, psi, opt, xpt, mask_outside_limiter, psi_bndry
    #         )

    #     elif len_xpt:
    #         del_psi = np.max(psi)-np.min(psi)
    #         # order xpt according to psi
    #         xpt = xpt[np.argsort(xpt[:,2])]
    #         i = -1
    #         xpt_found = False
    #         while xpt_found==False and i<len_xpt-1:
    #             i += 1
    #             # cs = plt.contour(R, Z, psi, levels=[xpt[i,2]-xpt_tol*del_psi, xpt[i,2]+xpt_tol*del_psi])
    #             # all_coords = cs.allsegs
    #             all_coords = [measure.find_contours(psi, xpt[i,2] - xpt_tol*del_psi),
    #                           measure.find_contours(psi, xpt[i,2] + xpt_tol*del_psi)]
    #             open_close = [self.all_open(all_coords[0]), self.all_open(all_coords[1])]
    #             # check that lines are open for fluxes 'belox' the xpoint and closed 'above'
    #             xpt_found = (open_close[0][0]==True) and (open_close[1][0]==False)
    #             if xpt_found:
    #                 # check that the closed region has overlap with the limiter region
    #                 # use countour to find diverted mask
    #                 candidate_lcfs = all_coords[1][np.arange(len(all_coords[1]))[np.logical_not(open_close[1][1])][0]]
    #                 # normalize spatial coordinates
    #                 candidate_lcfs *= self.dR_dZ
    #                 candidate_lcfs += self.R0Z0
    #                 LCFS = Path(candidate_lcfs)
    #                 candidate_diverted_core_mask = LCFS.contains_points(self.grid_points.reshape(-1, 2)).reshape(np.shape(R))
    #                 candidate_diverted_core_mask *= self.mask_inside_limiter
    #                 xpt_found = np.any(candidate_diverted_core_mask)

    #         if xpt_found:
    #             # use point with highest psi as opt
    #             psi_in_core = psi[candidate_diverted_core_mask]
    #             psi_max = max(psi_in_core)
    #             opt_idx = np.arange(len(psi_in_core))[psi_in_core==psi_max]
    #             new_opt = [[(R[candidate_diverted_core_mask])[opt_idx[0]],
    #                         (Z[candidate_diverted_core_mask])[opt_idx[0]],
    #                         psi_max]]
    #             # update opt list accordingly
    #             if len_opt:
    #                 # check if already in the list
    #                 dist = np.abs(opt - new_opt)
    #                 check_opt = (dist[:,0] < self.dR) * (dist[:,1] < self.dZ)
    #                 if np.any(check_opt):
    #                     # bring to first position
    #                     opt_idx = np.arange(len_opt)[check_opt][0]
    #                     aux = np.copy(opt[0])
    #                     opt[0] = np.copy(opt[opt_idx])
    #                     opt[opt_idx] = np.copy(aux)
    #                 else:
    #                     # add to the list
    #                     opt = np.concatenate((new_opt, opt), axis=0)
    #             else:
    #                 # add to list
    #                 opt = np.concatenate((new_opt, opt), axis=0)
    #             # set xpt-related quantities
    #             psi_bndry = xpt[i,2]
    #             self.lcfs = 1.0*candidate_lcfs
    #             # put xpt[i] to first position
    #             aux = np.copy(xpt[0])
    #             xpt[0] = np.copy(xpt[i])
    #             xpt[i] = np.copy(aux)
    #             # refine edge to recover any pixels lost due to the xpt_tol
    #             diverted_core_mask = self.limiter_handler.broaden_mask(candidate_diverted_core_mask, layer_size=1)
    #             diverted_core_mask *= (psi > psi_bndry)

    #         else:
    #             # no useful xpt found
    #             psi_bndry = psi[0, 0]
    #             diverted_core_mask = None

    #     else:
    #         # No X-points
    #         psi_bndry = psi[0, 0]
    #         diverted_core_mask = None

    #     return opt, xpt, diverted_core_mask, psi_bndry

    def Jtor_build(
        self,
        Jtor_part1,
        Jtor_part2,
        core_mask_limiter,
        R,
        Z,
        psi,
        psi_bndry,
        mask_outside_limiter,
        limiter_mask_out,
    ):
        """Universal function that calculates the plasma current distribution,
        common to all of the different types of profile parametrizations used in FreeGSNKE.

        Parameters
        ----------
        Jtor_part1 : method
            method from the freegs4e Profile class
            returns opt, xpt, diverted_core_mask
        Jtor_part2 : method
            method from each individual profile class
            returns jtor itself
        core_mask_limiter : method
            method of the limiter_handler class
            returns the refined core_mask where jtor>0 accounting for the limiter
        R : np.ndarray
            R coordinates of the domain grid points
        Z : np.ndarray
            Z coordinates of the domain grid points
        psi : np.ndarray
            Poloidal field flux / 2*pi at each grid points (for example as returned by Equilibrium.psi())
        psi_bndry : float, optional
            Value of the poloidal field flux at the boundary of the plasma (last closed flux surface), by default None
        mask_outside_limiter : np.ndarray
            Mask of points outside the limiter, if any, optional
        limiter_mask_out : np.ndarray
            The mask identifying the border of the limiter, including points just inside it, the 'last' accessible to the plasma.
            Same size as psi.
        """

        opt, xpt, diverted_core_mask, self.diverted_psi_bndry = Jtor_part1(
            R, Z, psi, psi_bndry, mask_outside_limiter
        )

        if diverted_core_mask is None:
            # print('no xpt')
            psi_bndry, limiter_core_mask, flag_limiter = (
                self.diverted_psi_bndry,
                None,
                False,
            )
            # psi_bndry = np.amin(psi[self.limiter_mask_out])
            # diverted_core_mask = np.copy(self.mask_inside_limiter)

        else:
            psi_bndry, limiter_core_mask, flag_limiter = core_mask_limiter(
                psi,
                self.diverted_psi_bndry,
                diverted_core_mask * self.mask_inside_limiter,
                limiter_mask_out,
            )
            if np.sum(limiter_core_mask * self.mask_inside_limiter) == 0:
                limiter_core_mask = diverted_core_mask * self.mask_inside_limiter
                psi_bndry = 1.0 * self.diverted_psi_bndry

        self.inputs = [opt[0][2], psi_bndry, limiter_core_mask]

        jtor = Jtor_part2(R, Z, psi, opt[0][2], psi_bndry, limiter_core_mask)
        return (
            jtor,
            opt,
            xpt,
            psi_bndry,
            diverted_core_mask,
            limiter_core_mask,
            flag_limiter,
        )

    def Jtor_unrefined(self, R, Z, psi, psi_bndry=None):
        """Replaces the FreeGS4E call, while maintaining the same input structure.

        Parameters
        ----------
        R : np.ndarray
            R coordinates of the domain grid points
        Z : np.ndarray
            Z coordinates of the domain grid points
        psi : np.ndarray
            Poloidal field flux / 2*pi at each grid points (for example as returned by Equilibrium.psi())
        psi_bndry : float, optional
            Value of the poloidal field flux at the boundary of the plasma (last closed flux surface), by default None

        Returns
        -------
        ndarray
            2d map of toroidal current values
        """
        (
            self.jtor,
            self.opt,
            self.xpt,
            self.psi_bndry,
            self.diverted_core_mask,
            self.limiter_core_mask,
            self.flag_limiter,
        ) = self.Jtor_build(
            self.diverted_critical_complete,
            # self.Jtor_part1,
            self.Jtor_part2,
            self.limiter_handler.core_mask_limiter,
            # self.core_mask_limiter,
            R,
            Z,
            psi,
            psi_bndry,
            self.mask_outside_limiter,
            self.limiter_mask_out,
        )
        return self.jtor

    def Jtor_refined(self, R, Z, psi, psi_bndry=None, thresholds=None):
        """Implements the call to the Jtor method for the case in which the subgrid refinement is used.

         Parameters
        ----------
        R : np.ndarray
            R coordinates of the domain grid points
        Z : np.ndarray
            Z coordinates of the domain grid points
        psi : np.ndarray
            Poloidal field flux / 2*pi at each grid points (for example as returned by Equilibrium.psi())
        psi_bndry : float, optional
            Value of the poloidal field flux at the boundary of the plasma (last closed flux surface), by default None
        thresholds : tuple (threshold for jtor criterion, threshold for gradient criterion)
            tuple of values used to identify where to apply refinement
            when None, the default refinement_thresholds are used

        Returns
        -------
        ndarray
            2d map of toroidal current values
        """

        unrefined_jtor = self.Jtor_unrefined(R, Z, psi, psi_bndry)
        self.unrefined_jtor = np.copy(unrefined_jtor)
        self.unrefined_djtordpsi = np.copy(self.dJtordpsi)
        self.pure_jtor = unrefined_jtor / self.L
        self.pure_djtordpsi = self.dJtordpsi / self.L
        core_mask = 1.0 * self.limiter_core_mask

        if thresholds is None:
            thresholds = self.refinement_thresholds

        bilinear_psi_interp, refined_R = self.jtor_refiner.build_bilinear_psi_interp(
            psi, core_mask, unrefined_jtor, thresholds
        )
        refined_jtor = self.Jtor_part2(
            R,
            Z,
            bilinear_psi_interp.reshape(-1, self.jtor_refiner.nny),
            self.psi_axis,
            self.psi_bndry,
            mask=None,
            torefine=True,
            refineR=refined_R.reshape(-1, self.jtor_refiner.nny),
        )
        refined_jtor = refined_jtor.reshape(
            -1, self.jtor_refiner.nnx, self.jtor_refiner.nny
        )
        self.dJtordpsi = self.jtor_refiner.build_from_refined_jtor(
            self.pure_djtordpsi,
            self.dJtordpsi.reshape(-1),
            self.jtor_refiner.nnx,
            self.jtor_refiner.nny,
        )

        self.jtor = self.jtor_refiner.build_from_refined_jtor(
            self.pure_jtor, refined_jtor
        )
        if self.Ip_logic:
            self.L = self.Ip / (np.sum(self.jtor) * self.dRdZ)
            self.jtor *= self.L
            self.dJtordpsi *= self.L

        return self.jtor


class ConstrainBetapIp(freegs4e.jtor.ConstrainBetapIp, Jtor_universal):
    """FreeGSNKE profile class adapting the original FreeGS object with the same name,
    with a few modifications, to:
    - retain memory of critical point calculation;
    - deal with limiter plasma configurations

    """

    def __init__(self, eq, *args, **kwargs):
        """Instantiates the object.

        Parameters
        ----------
        eq : FreeGSNKE Equilibrium object
            Specifies the domain properties
        """
        freegs4e.jtor.ConstrainBetapIp.__init__(self, *args, **kwargs)
        Jtor_universal.__init__(self)

        # profiles need Ip normalization
        self.Ip_logic = True
        self.profile_parameter = self.betap

        self.set_masks(eq=eq)

    def copy(self):
        obj = super().copy()

        copy_into(self, obj, "profile_parameter")
        copy_into(self, obj, "betap")
        copy_into(self, obj, "Ip")
        copy_into(self, obj, "_fvac")
        copy_into(self, obj, "alpha_m")
        copy_into(self, obj, "alpha_n")
        copy_into(self, obj, "Raxis")
        copy_into(self, obj, "fast")
        copy_into(self, obj, "L")
        copy_into(self, obj, "Beta0")

        return obj

    def Lao_parameters(
        self, n_alpha, n_beta, alpha_logic=True, beta_logic=True, Ip_logic=True, nn=100
    ):
        """Finds best fitting alpha, beta parameters for a Lao85 profile,
        to reproduce the input pprime_ and ffprime_
        n_alpha and n_beta represent the number of free parameters

        See Lao_parameters_finder.
        """

        pn_ = np.linspace(0, 1, nn)
        pprime_ = self.pprime(pn_)
        ffprime_ = self.ffprime(pn_)

        alpha, beta = swp.Lao_parameters_finder(
            pn_,
            pprime_,
            ffprime_,
            n_alpha,
            n_beta,
            alpha_logic,
            beta_logic,
            Ip_logic,
        )

        return alpha, beta


class ConstrainPaxisIp(freegs4e.jtor.ConstrainPaxisIp, Jtor_universal):
    """FreeGSNKE profile class adapting the original FreeGS object with the same name,
    with a few modifications, to:
    - retain memory of critical point calculation;
    - deal with limiter plasma configurations

    """

    def __init__(self, eq, *args, **kwargs):
        """Instantiates the object.

        Parameters
        ----------
        eq : FreeGSNKE Equilibrium object
            Specifies the domain properties
        """
        freegs4e.jtor.ConstrainPaxisIp.__init__(self, *args, **kwargs)
        Jtor_universal.__init__(self)

        # profiles need Ip normalization
        self.Ip_logic = True
        self.profile_parameter = self.paxis

        self.set_masks(eq=eq)

    def copy(self):
        obj = super().copy()

        copy_into(self, obj, "profile_parameter")
        copy_into(self, obj, "paxis")
        copy_into(self, obj, "Ip")
        copy_into(self, obj, "_fvac")
        copy_into(self, obj, "alpha_m")
        copy_into(self, obj, "alpha_n")
        copy_into(self, obj, "Raxis")
        copy_into(self, obj, "fast")
        copy_into(self, obj, "L")
        copy_into(self, obj, "Beta0")

        return obj

    def Lao_parameters(
        self, n_alpha, n_beta, alpha_logic=True, beta_logic=True, Ip_logic=True, nn=100
    ):
        """Finds best fitting alpha, beta parameters for a Lao85 profile,
        to reproduce the input pprime_ and ffprime_
        n_alpha and n_beta represent the number of free parameters

        See Lao_parameters_finder.
        """

        pn_ = np.linspace(0, 1, nn)
        pprime_ = self.pprime(pn_)
        ffprime_ = self.ffprime(pn_)

        alpha, beta = swp.Lao_parameters_finder(
            pn_,
            pprime_,
            ffprime_,
            n_alpha,
            n_beta,
            alpha_logic,
            beta_logic,
            Ip_logic,
        )

        return alpha, beta


class Fiesta_Topeol(freegs4e.jtor.Fiesta_Topeol, Jtor_universal):
    """FreeGSNKE profile class adapting the FreeGS4E object with the same name,
    with a few modifications, to:
    - retain memory of critical point calculation;
    - deal with limiter plasma configurations

    """

    def __init__(self, eq, *args, **kwargs):
        """Instantiates the object.

        Parameters
        ----------
        eq : FreeGSNKE Equilibrium object
            Specifies the domain properties
        """
        freegs4e.jtor.Fiesta_Topeol.__init__(self, *args, **kwargs)
        Jtor_universal.__init__(self)

        # profiles need Ip normalization
        self.Ip_logic = True
        self.profile_parameter = self.Beta0

        self.set_masks(eq=eq)

    def copy(self):
        obj = super().copy()

        copy_into(self, obj, "profile_parameter")
        copy_into(self, obj, "Ip")
        copy_into(self, obj, "_fvac")
        copy_into(self, obj, "alpha_m")
        copy_into(self, obj, "alpha_n")
        copy_into(self, obj, "Raxis")
        copy_into(self, obj, "fast")
        copy_into(self, obj, "L")
        copy_into(self, obj, "Beta0")

        return obj

    def Lao_parameters(
        self, n_alpha, n_beta, alpha_logic=True, beta_logic=True, Ip_logic=True, nn=100
    ):
        """Finds best fitting alpha, beta parameters for a Lao85 profile,
        to reproduce the input pprime_ and ffprime_
        n_alpha and n_beta represent the number of free parameters

        See Lao_parameters_finder.
        """

        pn_ = np.linspace(0, 1, nn)
        pprime_ = self.pprime(pn_)
        ffprime_ = self.ffprime(pn_)

        alpha, beta = swp.Lao_parameters_finder(
            pn_,
            pprime_,
            ffprime_,
            n_alpha,
            n_beta,
            alpha_logic,
            beta_logic,
            Ip_logic,
        )

        return alpha, beta


class Lao85(freegs4e.jtor.Lao85, Jtor_universal):
    """FreeGSNKE profile class adapting the FreeGS4E object with the same name,
    with a few modifications, to:
    - retain memory of critical point calculation;
    - deal with limiter plasma configurations

    """

    def __init__(self, eq, *args, refine_jtor=False, nnx=None, nny=None, **kwargs):
        """Instantiates the object.

        Parameters
        ----------
        eq : freegs4e Equilibrium object
            Specifies the domain properties
        refine_jtor : bool
            Flag to select whether to apply sug-grid refinement of plasma current distribution jtor
        nnx : even integer
            refinement factor in the R direction
        nny : even integer
            refinement factor in the Z direction
        """
        freegs4e.jtor.Lao85.__init__(self, *args, **kwargs)
        self.set_masks(eq=eq)
        self.select_refinement(eq, refine_jtor, nnx, nny)

    def copy(self):
        obj = super().copy()

        copy_into(self, obj, "Ip")
        copy_into(self, obj, "_fvac")
        copy_into(self, obj, "alpha_logic")
        copy_into(self, obj, "beta_logic")
        copy_into(self, obj, "Raxis")
        copy_into(self, obj, "fast")
        copy_into(self, obj, "Ip_logic")
        copy_into(self, obj, "L")
        copy_into(self, obj, "alpha", mutable=True)
        copy_into(self, obj, "beta", mutable=True)
        copy_into(self, obj, "alpha_exp", mutable=True)
        copy_into(self, obj, "beta_exp", mutable=True)
        copy_into(self, obj, "dJtorpsin1", strict=False)
        copy_into(self, obj, "dJtordpsi", mutable=True, strict=False)
        copy_into(self, obj, "problem_psi", mutable=True, strict=False)

        return obj

    def Topeol_parameters(self, nn=100, max_it=100, tol=1e-5):
        """Fids best combination of
        (alpha_m, alpha_n, beta_0)
        to instantiate a Topeol profile object as similar as possible to self

        Parameters
        ----------
        nn : int, optional
            number of points to sample 0,1 interval in the normalised psi, by default 100
        max_it : int,
            maximum number of iterations in the optimization
        tol : float
            iterations stop when change in the optimised parameters in smaller than tol
        """

        x = np.linspace(1 / (100 * nn), 1 - 1 / (100 * nn), nn)
        tp = self.pprime(x)
        tf = self.ffprime(x) / mu0

        pars = swp.Topeol_opt(
            tp,
            tf,
            x,
            max_it,
            tol,
        )

        return pars


class TensionSpline(freegs4e.jtor.TensionSpline, Jtor_universal):
    """FreeGSNKE profile class adapting the FreeGS4E object with the same name,
    with a few modifications, to:
    - retain memory of critical point calculation;
    - deal with limiter plasma configurations
    """

    def __init__(self, eq, *args, **kwargs):
        """Instantiates the object.

        Parameters
        ----------
        eq : FreeGSNKE Equilibrium object
            Specifies the domain properties
        """

        freegs4e.jtor.TensionSpline.__init__(self, *args, **kwargs)
        Jtor_universal.__init__(self)

        self.profile_parameter = [
            self.pp_knots,
            self.pp_values,
            self.pp_values_2,
            self.pp_sigma,
            self.ffp_knots,
            self.ffp_values,
            self.ffp_values_2,
            self.ffp_sigma,
        ]

        self.set_masks(eq=eq)

    def copy(self):
        obj = super().copy()

        copy_into(self, obj, "Ip")
        copy_into(self, obj, "_fvac")
        copy_into(self, obj, "Raxis")
        copy_into(self, obj, "Ip_logic")
        copy_into(self, obj, "fast")
        copy_into(self, obj, "L")
        copy_into(self, obj, "pp_knots", mutable=True)
        copy_into(self, obj, "pp_values", mutable=True)
        copy_into(self, obj, "pp_values_2", mutable=True)
        copy_into(self, obj, "pp_sigma")
        copy_into(self, obj, "ffp_knots", mutable=True)
        copy_into(self, obj, "ffp_values", mutable=True)
        copy_into(self, obj, "ffp_values_2", mutable=True)
        copy_into(self, obj, "ffp_sigma")

        obj.profile_parameter = [
            obj.pp_knots,
            obj.pp_values,
            obj.pp_values_2,
            obj.pp_sigma,
            obj.ffp_knots,
            obj.ffp_values,
            obj.ffp_values_2,
            obj.ffp_sigma,
        ]

        return obj

    def assign_profile_parameter(
        self,
        pp_knots,
        pp_values,
        pp_values_2,
        pp_sigma,
        ffp_knots,
        ffp_values,
        ffp_values_2,
        ffp_sigma,
    ):
        """Assigns to the profile object new values for the profile parameters"""
        self.pp_knots = pp_knots
        self.pp_values = pp_values
        self.pp_values_2 = pp_values_2
        self.pp_sigma = pp_sigma
        self.ffp_knots = ffp_knots
        self.ffp_values = ffp_values
        self.ffp_values_2 = ffp_values_2
        self.ffp_sigma = ffp_sigma

        self.profile_parameter = [
            pp_knots,
            pp_values,
            pp_values_2,
            pp_sigma,
            ffp_knots,
            ffp_values,
            ffp_values_2,
            ffp_sigma,
        ]


class GeneralPprimeFFprime(freegs4e.jtor.GeneralPprimeFFprime, Jtor_universal):
    """FreeGSNKE profile class adapting the FreeGS4E object with the same name,
    with a few modifications, to:
    - retain memory of critical point calculation;
    - deal with limiter plasma configurations
    """

    def __init__(self, eq, *args, **kwargs):
        """Instantiates the object.

        Parameters
        ----------
        eq : FreeGSNKE Equilibrium object
            Specifies the domain properties
        """

        freegs4e.jtor.GeneralPprimeFFprime.__init__(self, *args, **kwargs)
        Jtor_universal.__init__(self)

        self.profile_parameter = []
        self.set_masks(eq=eq)

    def copy(self):
        obj = super().copy()

        copy_into(self, obj, "profile_parameter")
        copy_into(self, obj, "Ip")
        copy_into(self, obj, "_fvac")
        copy_into(self, obj, "Raxis")
        copy_into(self, obj, "Ip_logic")
        copy_into(self, obj, "L")
        copy_into(self, obj, "fast")
        copy_into(self, obj, "psi_n", mutable=True)
        copy_into(self, obj, "pprime_data", mutable=True)
        copy_into(self, obj, "ffprime_data", mutable=True)
        copy_into(self, obj, "p_data", mutable=True)
        copy_into(self, obj, "f_data", mutable=True)

        obj.initialize_profile()

        return obj

    def assign_profile_parameter(
        self,
    ):
        """Assigns to the profile object new values for the profile parameters"""

        self.profile_parameter = []
