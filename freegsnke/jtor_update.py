"""
Defines the FreeGSNKE profile object, which inherits from the FreeGS4E profile object.

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

from dataclasses import dataclass

import freegs4e.jtor
import numpy as np
from freegs4e.gradshafranov import mu0
from matplotlib.path import Path
from scipy.ndimage import maximum_filter
from scipy.optimize import least_squares
from skimage import measure

from . import jtor_refinement
from . import switch_profile as swp
from .copying import copy_into


@dataclass
class LaoBetapLiFitResult:
    """Result returned by :func:`fit_lao85_betap_li_ip`.

    Attributes
    ----------
    alpha, beta : ndarray
        The fitted two-coefficient Lao pressure and FF' vectors supplied to
        :class:`Lao85` before optional logic coefficients are appended.
    effective_alpha, effective_beta : ndarray
        Full coefficient vectors used by the returned profile object after
        applying ``alpha_logic`` and ``beta_logic``.
    target_betap, final_betap : float
        Requested and achieved poloidal beta. This uses ``eq.poloidalBeta1()``,
        matching the definition used by ``ConstrainBetapIp``.
    target_li, final_li : float
        Requested and achieved internal inductance using ``li_method``.
    metric_jacobian : ndarray or None
        Physical Jacobian ``d[beta_p, li] / d[alpha0, alpha1, beta0, beta1]``
        supplied to, or built for, the least-squares optimiser.
    optimizer_result : scipy.optimize.OptimizeResult
        Raw optimiser result from ``scipy.optimize.least_squares``.
    evaluations : list of dict
        Per-objective-call diagnostics, including failed static solves.
    """

    alpha: np.ndarray
    beta: np.ndarray
    effective_alpha: np.ndarray
    effective_beta: np.ndarray
    alpha_logic: bool
    beta_logic: bool
    target_betap: float
    final_betap: float
    target_li: float
    final_li: float
    li_method: str
    metric_jacobian: object
    optimizer_result: object
    evaluations: list


class Jtor_universal:
    """
    Wrapper class providing a unified interface for toroidal current density (Jtor)
    evaluation, with optional refinement.

    This class selects between two implementations of the toroidal current density
    model depending on whether refinement is enabled:

    - Unrefined Jtor: fast, standard evaluation
    - Refined Jtor: higher-resolution or corrected evaluation using additional
      numerical processing

    The interface ensures that downstream code can call `Jtor()` without needing
    to know which implementation is being used.
    """

    def __init__(self, refine_jtor=False):
        """Sets default unrefined Jtor."""
        self._refine_jtor = refine_jtor

    def Jtor(self, *args, **kwargs):
        """
        Evaluate toroidal current density (Jtor), dispatching to either the
        refined or unrefined implementation.

        This method acts as a unified interface:
        - If `_refine_jtor` is True, it calls `Jtor_refined`
        - Otherwise, it calls `Jtor_unrefined`

        Parameters
        ----------
        *args, **kwargs
            Arguments passed directly to the selected Jtor implementation.

        Returns
        -------
        ndarray
            Toroidal current density evaluated on the plasma grid.
        """
        if self._refine_jtor:
            return self.Jtor_refined(*args, **kwargs)
        else:
            return self.Jtor_unrefined(*args, **kwargs)

    def copy(self, obj=None):
        """
        Create a copy of the current Jtor_universal instance.

        This method performs a selective copy of internal attributes,
        combining shallow copies, deep copies, and shared references
        depending on the nature of each attribute.

        Parameters
        ----------
        obj : Jtor_universal, optional
            Existing instance to copy attributes into. If None, a new
            instance of the same class is created.

        Returns
        -------
        Jtor_universal
            A copied instance of the current object.

        Notes
        -----
        Copy semantics:
        - Immutable / scalar attributes are copied directly
        - NumPy arrays and simple containers are shallow-copied where safe
        - Selected complex objects are deep-copied (e.g. via copy.deepcopy)
        - Some large system objects are shared by reference (e.g. limiter_handler)

        Optional attributes:
        The method safely ignores missing attributes using strict=False
        for certain fields.

        Warning
        -------
        This is not a full deep copy. Some internal objects are shared
        between the original and the copy, particularly:
        - limiter_handler
        - any attributes not explicitly copied via `copy_into`
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
        if hasattr(self, "inputs"):
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
        """
        Initialise grid geometry and limiter-related masks from an equilibrium object.

        This method constructs and stores all spatial grid metadata, index mappings,
        and limiter masks required for subsequent calculations. It modifies the
        object in-place.

        Parameters
        ----------
        eq : FreeGSNKE Equilibrium object
            Equilibrium defining the computational domain, including 1D and 2D
            coordinate grids and limiter geometry.
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
        """
        Initialise optional subgrid refinement for toroidal current density (jtor).

        This method enables and configures subgrid refinement of the plasma
        current density if requested, constructing the refinement handler and
        associated parameters.

        Parameters
        ----------
        eq : FreeGS4E Equilibrium object
            Equilibrium object defining the computational domain and geometry.
        refine_jtor : bool
            If True, enable subgrid refinement of the toroidal current density.
            If False, refinement is disabled.
        nnx : int (even)
            Refinement factor in the R-direction.
        nny : int (even)
            Refinement factor in the Z-direction.

        Returns
        -------
        None
        """
        self._refine_jtor = refine_jtor
        if refine_jtor:
            self.jtor_refiner = jtor_refinement.Jtor_refiner(eq, nnx, nny)
            self.set_refinement_thresholds()

    def set_refinement_thresholds(self, thresholds=(1.0, 1.0)):
        """
        Set the criteria used to control jtor subgrid refinement.

        These thresholds determine where refinement is applied based on the
        current density and its gradient.

        Parameters
        ----------
        thresholds : tuple of float, optional
            (jtor_threshold, gradient_threshold), where each value controls
            the activation of refinement criteria.

        Returns
        -------
        None
        """
        self.refinement_thresholds = thresholds

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
        Compute LCFS, O-point, X-point, and core mask using a contour-based fallback algorithm.

        This method replaces the standard LCFS/X-point detection routine when it fails,
        providing a more robust (but more expensive) contour-tracking approach.

        Parameters
        ----------
        R : ndarray
            Radial grid coordinates.
        Z : ndarray
            Vertical grid coordinates.
        psi : ndarray
            Poloidal flux on the computational grid.
        psi_bndry : float, optional
            Prescribed boundary flux value. If None, it is computed internally.
        mask_outside_limiter : ndarray, optional
            Boolean mask identifying points outside the limiter.
        rel_tolerance_xpt : float, optional
            Relative tolerance controlling convergence of X-point search.
        starting_dx : float, optional
            Initial step size in normalized flux space for contour search.

        Returns
        -------
        opt : ndarray, shape (1, 3)
            O-point coordinates and flux value [R, Z, psi].
        xpt : ndarray, shape (1, 3)
            X-point coordinates and flux value [R, Z, psi].
        diverted_core_mask : ndarray, shape (nx, ny)
            Boolean mask identifying plasma core region.
        psi_bndry : float
            Flux value at the last closed flux surface (LCFS).
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
        """
        Robust LCFS, O-point, X-point, and core mask detection with fallback logic.

        This method attempts to compute plasma boundary information using the
        primary routine `Jtor_part1`. If this fails (raises an exception), it
        falls back to a more robust but computationally expensive contour-based
        method implemented in `diverted_critical`.

        Parameters
        ----------
        R : ndarray
            Radial grid coordinates.
        Z : ndarray
            Vertical grid coordinates.
        psi : ndarray
            Poloidal flux on the computational grid.
        psi_bndry : float, optional
            Prescribed boundary flux value. If None, it is computed internally.
        mask_outside_limiter : ndarray, optional
            Boolean mask identifying points outside the limiter.
        rel_tolerance_xpt : float, optional
            Convergence tolerance for X-point search in fallback method.
        starting_dx : float, optional
            Initial step size for contour-based X-point search.

        Returns
        -------
        opt : ndarray, shape (1, 3)
            O-point coordinates and flux value [R, Z, psi].
        xpt : ndarray, shape (1, 3)
            X-point coordinates and flux value [R, Z, psi].
        diverted_core_mask : ndarray
            Boolean mask of plasma core region.
        psi_bndry : float
            Flux value at the last closed flux surface (LCFS).
        """

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
        """
        Construct the toroidal current density (Jtor) using a modular profile pipeline.

        This function is the main assembly routine for the plasma current density.
        It combines:
        - geometric reconstruction of plasma boundaries (Jtor_part1),
        - limiter-aware core masking (core_mask_limiter),
        - and evaluation of the current profile itself (Jtor_part2).

        The implementation is designed to support multiple profile parametrisations
        in a unified framework.

        Parameters
        ----------
        Jtor_part1 : callable
            Function that computes geometric plasma descriptors:
            returns (opt, xpt, diverted_core_mask).
        Jtor_part2 : callable
            Function that evaluates the toroidal current density jtor.
        core_mask_limiter : callable
            Function that refines the core mask using limiter geometry.
        R : ndarray
            Radial grid coordinates.
        Z : ndarray
            Vertical grid coordinates.
        psi : ndarray
            Poloidal flux on the grid.
        psi_bndry : float
            Flux value at the last closed flux surface (LCFS).
        mask_outside_limiter : ndarray
            Boolean mask for points outside the limiter.
        limiter_mask_out : ndarray
            Limiter boundary mask (including edge-adjacent region).

        Returns
        -------
        jtor : ndarray
            Toroidal current density on the grid.
        opt : ndarray
            O-point coordinates and flux value.
        xpt : ndarray
            X-point coordinates and flux value.
        psi_bndry : float
            Updated LCFS flux value.
        diverted_core_mask : ndarray
            Core plasma mask from geometric reconstruction.
        limiter_core_mask : ndarray
            Core mask refined with limiter constraints.
        flag_limiter : bool or int
            Indicator of whether limiter correction was applied successfully.
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
        """
        Compute the toroidal current density without subgrid refinement.

        This method provides a direct replacement for the FreeGS4E current
        computation interface, using the internal Jtor pipeline but disabling
        any refined current reconstruction.

        Parameters
        ----------
        R : ndarray
            Radial grid coordinates.
        Z : ndarray
            Vertical grid coordinates.
        psi : ndarray
            Poloidal flux on the grid (Webers / 2π).
        psi_bndry : float, optional
            Flux value at the last closed flux surface (LCFS). If None,
            it is determined internally.

        Returns
        -------
        jtor : ndarray
            Toroidal current density on the computational grid.
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
            self.Jtor_part2,
            self.limiter_handler.core_mask_limiter,
            R,
            Z,
            psi,
            psi_bndry,
            self.mask_outside_limiter,
            self.limiter_mask_out,
        )
        return self.jtor

    def Jtor_refined(self, R, Z, psi, psi_bndry=None, thresholds=None):
        """
        Compute toroidal current density using subgrid refinement.

        This method evaluates the unrefined current density first and then applies
        a subgrid refinement procedure in regions where higher resolution is required.
        The refinement is controlled by a threshold-based criterion acting on the
        current density and its gradients.

        Parameters
        ----------
        R : ndarray
            Radial grid coordinates.
        Z : ndarray
            Vertical grid coordinates.
        psi : ndarray
            Poloidal flux on the grid (typically ψ / 2π).
        psi_bndry : float, optional
            Flux value at the last closed flux surface (LCFS). If None,
            it is determined internally.
        thresholds : tuple of float, optional
            (jtor_threshold, gradient_threshold) controlling where refinement is
            applied. If None, the default `self.refinement_thresholds` is used.

        Returns
        -------
        jtor : ndarray
            Refined toroidal current density on the computational grid.
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
    """
    Betap–Ip constrained toroidal current profile with FreeGSNKE extensions.

    """

    Jtor = Jtor_universal.Jtor

    def __init__(self, eq, *args, **kwargs):
        """
        Initialise the constrained profile.

        Parameters
        ----------
        eq : FreeGSNKE Equilibrium object
            Equilibrium object defining grid geometry and limiter structure.
        """
        freegs4e.jtor.ConstrainBetapIp.__init__(self, *args, **kwargs)
        Jtor_universal.__init__(self)

        # profiles need Ip normalization
        self.Ip_logic = True
        self.profile_parameter = self.betap

        self.set_masks(eq=eq)

    def copy(self):
        """
        Create a deep-ish copy of the profile object.

        Returns
        -------
        ConstrainBetapIp
            A new instance with copied scalar parameters and shared/replicated
            internal state depending on attribute type.
        """
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
        """
        Fit Lao85 profile parameters to the current pprime and ffprime profiles.

        This method constructs a discrete sampling of the normalized flux coordinate
        and evaluates the current profile derivatives. It then fits a Lao85-type
        polynomial representation to obtain optimal alpha and beta parameters.

        Parameters
        ----------
        n_alpha : int
            Number of free parameters used in the pprime (alpha) expansion.
        n_beta : int
            Number of free parameters used in the ffprime (beta) expansion.
        alpha_logic : bool, optional
            If True, enforce boundary-consistent modification of the alpha basis.
        beta_logic : bool, optional
            If True, enforce boundary-consistent modification of the beta basis.
        Ip_logic : bool, optional
            If True, apply total current normalisation during fitting.
        nn : int, optional
            Number of sampling points in the normalized flux coordinate.

        Returns
        -------
        alpha : ndarray
            Fitted alpha coefficients for the pprime expansion.
        beta : ndarray
            Fitted beta coefficients for the ffprime expansion.
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
    """
    Paxis–Ip constrained toroidal current profile with FreeGSNKE extensions.

    """

    Jtor = Jtor_universal.Jtor

    def __init__(self, eq, *args, **kwargs):
        """
        Initialise the constrained profile.

        Parameters
        ----------
        eq : FreeGSNKE Equilibrium object
            Equilibrium object defining grid geometry and limiter structure.
        """
        freegs4e.jtor.ConstrainPaxisIp.__init__(self, *args, **kwargs)
        Jtor_universal.__init__(self)

        # profiles need Ip normalization
        self.Ip_logic = True
        self.profile_parameter = self.paxis

        self.set_masks(eq=eq)

    def copy(self):
        """
        Create a copy of the current profile instance.

        Returns
        -------
        ConstrainPaxisIp
            A copied instance with duplicated scalar parameters and appropriately
            handled internal state (deep or shallow depending on attribute type).
        """

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
        """
        Fit Lao85 profile coefficients from pprime and ffprime evaluations.

        This method samples the normalized flux coordinate and evaluates the
        pressure derivative (pprime) and flux function derivative (ffprime).
        It then computes best-fit Lao85 polynomial coefficients using a linear
        fitting procedure.

        Parameters
        ----------
        n_alpha : int
            Number of coefficients in the alpha (pprime) expansion.
        n_beta : int
            Number of coefficients in the beta (ffprime) expansion.
        alpha_logic : bool, optional
            If True, enforces boundary-consistent modification of the alpha basis.
        beta_logic : bool, optional
            If True, enforces boundary-consistent modification of the beta basis.
        Ip_logic : bool, optional
            If True, applies total current normalisation in the fitting procedure.
        nn : int, optional
            Number of points used to sample the normalized flux coordinate.

        Returns
        -------
        alpha : ndarray
            Fitted coefficients for the pprime expansion.
        beta : ndarray
            Fitted coefficients for the ffprime expansion.
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
    """
    Fiesta Topeol constrained toroidal current profile with FreeGSNKE extensions.

    """

    Jtor = Jtor_universal.Jtor

    def __init__(self, eq, *args, **kwargs):
        """
        Initialise the Fiesta-Topeol constrained current profile.

        Parameters
        ----------
        eq : FreeGSNKE Equilibrium object
            Equilibrium object defining grid geometry and limiter structure.
        """
        freegs4e.jtor.Fiesta_Topeol.__init__(self, *args, **kwargs)
        Jtor_universal.__init__(self)

        # profiles need Ip normalization
        self.Ip_logic = True
        self.profile_parameter = self.Beta0

        self.set_masks(eq=eq)

    def copy(self):
        """
        Create a copy of the Fiesta-Topeol profile instance.

        Returns
        -------
        Fiesta_Topeol
            A copied instance with duplicated scalar parameters and appropriately
            handled internal state (deep or shallow depending on attribute type).
        """
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
        """
        Fit Lao85 profile coefficients from sampled pprime and ffprime data.

        This method evaluates the pressure derivative (pprime) and flux function
        derivative (ffprime) on a uniform grid in the normalized flux coordinate,
        then fits Lao85 polynomial coefficients using a linear least-squares
        procedure.

        Parameters
        ----------
        n_alpha : int
            Number of coefficients in the alpha (pprime) expansion.
        n_beta : int
            Number of coefficients in the beta (ffprime) expansion.
        alpha_logic : bool, optional
            If True, enforces boundary-consistent modification of the alpha basis.
        beta_logic : bool, optional
            If True, enforces boundary-consistent modification of the beta basis.
        Ip_logic : bool, optional
            If True, applies total current normalisation in the fitting procedure.
        nn : int, optional
            Number of sample points in the normalized flux coordinate.

        Returns
        -------
        alpha : ndarray
            Fitted coefficients for the pprime expansion.
        beta : ndarray
            Fitted coefficients for the ffprime expansion.
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
    """
    Lao 1985 constrained toroidal current profile with FreeGSNKE extensions.

    """

    Jtor = Jtor_universal.Jtor

    def __init__(self, eq, *args, refine_jtor=False, nnx=None, nny=None, **kwargs):
        """
        Initialise the Lao85 current profile.

        Parameters
        ----------
        eq : FreeGSNKE Equilibrium object
            Equilibrium object defining grid geometry and limiter structure.
        refine_jtor : bool
            If True, enable subgrid refinement of the toroidal current density.
        nnx : int
            Refinement factor in the R-direction (must be even if used).
        nny : int
            Refinement factor in the Z-direction (must be even if used).
        """
        freegs4e.jtor.Lao85.__init__(self, *args, **kwargs)
        self.set_masks(eq=eq)
        self.select_refinement(eq, refine_jtor, nnx, nny)

    def copy(self):
        """
        Create a copy of the Lao85 profile instance.

        Returns
        -------
        Lao85
            A copied instance with duplicated profile parameters and internal
            state (with controlled shallow/deep copying depending on attribute type).
        """
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
        """
        Fit optimal Topeol profile parameters from target pprime and ffprime data.

        This method determines the best-fitting parameters
        (alpha_m, alpha_n, beta_0) for a Topeol current profile by minimising
        a mismatch between the model and the target profile derivatives
        evaluated on a sampled normalized flux grid.

        Parameters
        ----------
        nn : int, optional
            Number of sampling points in the normalized flux interval (0, 1).
        max_it : int, optional
            Maximum number of optimisation iterations.
        tol : float, optional
            Convergence tolerance on parameter updates.

        Returns
        -------
        pars : ndarray, shape (3,)
            Optimised parameters (alpha_m, alpha_n, beta_0).
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


def fit_lao85_betap_li_ip(
    eq,
    static_solver,
    Ip,
    fvac,
    betap,
    li,
    alpha=None,
    beta=None,
    alpha_logic=True,
    beta_logic=True,
    li_method="internalInductance2",
    Raxis=1,
    target_relative_tolerance=1e-8,
    max_solving_iterations=100,
    solve_kwargs=None,
    optimizer_kwargs=None,
    residual_scales=None,
    regularization_weight=1e-6,
    coefficient_scales=None,
    use_metric_jacobian=False,
    metric_jacobian=None,
    jacobian_rel_step=1e-4,
):
    """Fit a four-coefficient Lao85 profile to static ``Ip``, ``betap`` and ``li``.

    The fitted profile uses two pressure coefficients and two FF' coefficients:
    ``alpha = [alpha0, alpha1]`` and ``beta = [beta0, beta1]``. These are the
    coefficients supplied to :class:`Lao85`; when ``alpha_logic`` or
    ``beta_logic`` is true, :class:`Lao85` appends the usual boundary coefficient
    so that the corresponding profile vanishes at the plasma edge.

    The total plasma current is enforced through the standard ``Lao85``
    ``Ip_logic`` global normalisation. The fitted scalar targets are therefore
    ``betap`` from ``eq.poloidalBeta1()`` and ``li`` from
    ``getattr(eq, li_method)()``. The default ``li_method`` is
    ``internalInductance2``.

    Four coefficients are fitted to two scalar targets after the current
    normalisation has removed one scale direction. A small regularisation term
    resolves the remaining freedom by favouring coefficients close to the
    supplied initial ``alpha`` and ``beta`` values.

    Parameters
    ----------
    eq : FreeGSNKE equilibrium
        Equilibrium object to solve. It is updated in-place to the final fitted
        static equilibrium.
    static_solver : freegsnke.GSstaticsolver.NKGSsolver
        Static Grad-Shafranov solver compatible with ``eq``.
    Ip : float
        Target total plasma current [A].
    fvac : float
        Vacuum toroidal field parameter ``R * B_tor`` [T m].
    betap : float
        Target poloidal beta using the ``ConstrainBetapIp`` definition.
    li : float
        Target internal inductance using ``li_method``.
    alpha, beta : array-like of length 2, optional
        Initial Lao coefficients before optional boundary coefficients are
        appended.
    alpha_logic, beta_logic : bool, optional
        Whether to append the standard Lao boundary coefficient for p' or FF'.
    li_method : str, optional
        Name of the equilibrium method used to evaluate internal inductance.
    target_relative_tolerance : float, optional
        Static solve tolerance used for each trial profile.
    max_solving_iterations : int, optional
        Maximum static solve iterations used for each trial profile.
    solve_kwargs, optimizer_kwargs : dict, optional
        Additional keyword arguments forwarded to the static solve and
        ``scipy.optimize.least_squares`` respectively.
    residual_scales : sequence of two floats, optional
        Scaling applied to the beta_p and li residuals.
    regularization_weight : float, optional
        Weight applied to the coefficient regularisation residual. Set to zero
        to disable regularisation.
    coefficient_scales : sequence of four floats, optional
        Scaling applied to the coefficient regularisation residual.
    use_metric_jacobian : bool, optional
        If true, build an explicit finite-difference metric Jacobian for the
        optimiser. This is more expensive but usually reduces optimiser steps
        for nearby targets.
    metric_jacobian : array-like, optional
        Reusable physical Jacobian ``d[beta_p, li] / d[alpha0, alpha1, beta0,
        beta1]``. If supplied, this constant Jacobian is used directly.
    jacobian_rel_step : float, optional
        Relative coefficient perturbation used by the explicit metric Jacobian.

    Returns
    -------
    profiles : Lao85
        Fitted Lao profile object after a final static solve.
    result : LaoBetapLiFitResult
        Fit diagnostics and raw optimiser result.
    """

    alpha = _validate_lao_pair("alpha", [1.0, -0.5] if alpha is None else alpha)
    beta = _validate_lao_pair("beta", [1.0, -0.5] if beta is None else beta)

    if not hasattr(eq, li_method):
        raise ValueError(f"Equilibrium object has no li method '{li_method}'.")

    solve_kwargs = {} if solve_kwargs is None else dict(solve_kwargs)
    optimizer_kwargs = {} if optimizer_kwargs is None else dict(optimizer_kwargs)

    x0 = np.concatenate((alpha, beta)).astype(float)
    if residual_scales is None:
        residual_scales = np.array([max(abs(betap), 1e-12), max(abs(li), 1e-12)])
    else:
        residual_scales = np.asarray(residual_scales, dtype=float)
        if residual_scales.shape != (2,):
            raise ValueError("residual_scales must contain two values.")
        if np.any(residual_scales <= 0):
            raise ValueError("residual_scales entries must be positive.")

    if coefficient_scales is None:
        coefficient_scales = np.maximum(np.abs(x0), 1.0)
    else:
        coefficient_scales = np.asarray(coefficient_scales, dtype=float)
        if coefficient_scales.shape != (4,):
            raise ValueError("coefficient_scales must contain four values.")
        if np.any(coefficient_scales <= 0):
            raise ValueError("coefficient_scales entries must be positive.")

    evaluations = []
    calculated_metric_jacobian = [None]
    if metric_jacobian is not None:
        metric_jacobian = np.asarray(metric_jacobian, dtype=float)
        if metric_jacobian.shape != (2, 4):
            raise ValueError("metric_jacobian must have shape (2, 4).")

    def build_profiles(coefficients):
        coefficients = np.asarray(coefficients, dtype=float)
        return Lao85(
            eq,
            Ip=Ip,
            fvac=fvac,
            alpha=coefficients[:2],
            beta=coefficients[2:],
            alpha_logic=alpha_logic,
            beta_logic=beta_logic,
            Raxis=Raxis,
            Ip_logic=True,
        )

    def evaluate_metric_residual(coefficients, record=True):
        profiles = build_profiles(coefficients)
        try:
            static_solver.forward_solve(
                eq,
                profiles,
                target_relative_tolerance,
                max_solving_iterations=max_solving_iterations,
                suppress=True,
                **solve_kwargs,
            )
            _sync_profile_flux_normalisation(eq, profiles)
            current_betap = eq.poloidalBeta1()
            current_li = getattr(eq, li_method)()
            scaled_residual = (
                np.array([current_betap - betap, current_li - li], dtype=float)
                / residual_scales
            )
            if record:
                evaluations.append(
                    {
                        "coefficients": np.asarray(coefficients, dtype=float).copy(),
                        "betap": current_betap,
                        "li": current_li,
                        "failed": False,
                    }
                )
        except Exception as exc:
            scaled_residual = np.array([1e6, 1e6], dtype=float)
            if record:
                evaluations.append(
                    {
                        "coefficients": np.asarray(coefficients, dtype=float).copy(),
                        "betap": None,
                        "li": None,
                        "failed": True,
                        "error": repr(exc),
                    }
                )

        return scaled_residual

    def objective(coefficients):
        scaled_residual = evaluate_metric_residual(coefficients)
        if regularization_weight:
            regularization = (
                np.sqrt(regularization_weight)
                * (np.asarray(coefficients, dtype=float) - x0)
                / coefficient_scales
            )
            return np.concatenate((scaled_residual, regularization))

        return scaled_residual

    def append_regularization_jacobian(scaled_metric_jacobian):
        if regularization_weight:
            regularization_jacobian = np.sqrt(regularization_weight) * np.diag(
                1.0 / coefficient_scales
            )
            return np.vstack((scaled_metric_jacobian, regularization_jacobian))

        return scaled_metric_jacobian

    def constant_metric_jacobian(coefficients):
        del coefficients
        scaled_metric_jacobian = metric_jacobian / residual_scales[:, np.newaxis]
        return append_regularization_jacobian(scaled_metric_jacobian)

    def finite_difference_metric_jacobian(coefficients):
        coefficients = np.asarray(coefficients, dtype=float)
        scaled_metric_jacobian = np.zeros((2, len(coefficients)))

        base_plasma_psi = np.copy(eq.plasma_psi)
        base_solved = getattr(eq, "solved", None)

        for i, coefficient in enumerate(coefficients):
            step = jacobian_rel_step * max(abs(coefficient), coefficient_scales[i])
            if step == 0:
                step = jacobian_rel_step

            plus = coefficients.copy()
            plus[i] += step
            _restore_equilibrium_solver_state(eq, base_plasma_psi, base_solved)
            plus_residual = evaluate_metric_residual(plus, record=False)

            minus = coefficients.copy()
            minus[i] -= step
            _restore_equilibrium_solver_state(eq, base_plasma_psi, base_solved)
            minus_residual = evaluate_metric_residual(minus, record=False)

            scaled_metric_jacobian[:, i] = (plus_residual - minus_residual) / (
                2.0 * step
            )

        _restore_equilibrium_solver_state(eq, base_plasma_psi, base_solved)
        calculated_metric_jacobian[0] = (
            scaled_metric_jacobian * residual_scales[:, np.newaxis]
        )

        return append_regularization_jacobian(scaled_metric_jacobian)

    optimizer_kwargs.setdefault("max_nfev", 30)
    if metric_jacobian is not None:
        optimizer_kwargs.setdefault("jac", constant_metric_jacobian)
    elif use_metric_jacobian:
        optimizer_kwargs.setdefault("jac", finite_difference_metric_jacobian)
    optimizer_result = least_squares(objective, x0, **optimizer_kwargs)

    if (
        use_metric_jacobian
        and metric_jacobian is None
        and calculated_metric_jacobian[0] is None
    ):
        finite_difference_metric_jacobian(optimizer_result.x)

    profiles = build_profiles(optimizer_result.x)
    static_solver.forward_solve(
        eq,
        profiles,
        target_relative_tolerance,
        max_solving_iterations=max_solving_iterations,
        suppress=True,
        **solve_kwargs,
    )
    _sync_profile_flux_normalisation(eq, profiles)

    result = LaoBetapLiFitResult(
        alpha=optimizer_result.x[:2].copy(),
        beta=optimizer_result.x[2:].copy(),
        effective_alpha=profiles.alpha.copy(),
        effective_beta=profiles.beta.copy(),
        alpha_logic=alpha_logic,
        beta_logic=beta_logic,
        target_betap=betap,
        final_betap=eq.poloidalBeta1(),
        target_li=li,
        final_li=getattr(eq, li_method)(),
        li_method=li_method,
        metric_jacobian=(
            metric_jacobian
            if metric_jacobian is not None
            else calculated_metric_jacobian[0]
        ),
        optimizer_result=optimizer_result,
        evaluations=evaluations,
    )

    return profiles, result


def _validate_lao_pair(name, values):
    """Return a validated two-coefficient Lao parameter vector."""
    values = np.asarray(values, dtype=float)
    if values.shape != (2,):
        raise ValueError(
            f"{name} must contain exactly two coefficients before Lao logic is applied."
        )
    return values


def _sync_profile_flux_normalisation(eq, profiles):
    """Ensure solved profile copies retain flux-axis normalisation fields."""
    if hasattr(profiles, "inputs"):
        psi_axis = profiles.inputs[0]
        psi_bndry = profiles.inputs[1]
    else:
        psi_axis = getattr(profiles, "psi_axis", getattr(eq, "psi_axis", None))
        psi_bndry = getattr(profiles, "psi_bndry", getattr(eq, "psi_bndry", None))

    for target in (profiles, getattr(eq, "_profiles", None)):
        if target is None:
            continue
        if psi_axis is not None and not hasattr(target, "psi_axis"):
            target.psi_axis = psi_axis
        if psi_bndry is not None and not hasattr(target, "psi_bndry"):
            target.psi_bndry = psi_bndry

    if hasattr(eq, "_profiles"):
        eq._profiles = profiles.copy()


def _restore_equilibrium_solver_state(eq, plasma_psi, solved):
    """Restore the minimal equilibrium state that static solves mutate."""
    eq.plasma_psi = np.copy(plasma_psi)
    if solved is not None:
        eq.solved = solved


class TensionSpline(freegs4e.jtor.TensionSpline, Jtor_universal):
    """
    Tension spline constrained toroidal current profile with FreeGSNKE extensions.

    """

    Jtor = Jtor_universal.Jtor

    def __init__(self, eq, *args, **kwargs):
        """
        Initialise the tension spline current profile.

        Parameters
        ----------
        eq : FreeGSNKE Equilibrium object
            Equilibrium object defining grid geometry and limiter structure.
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
        """
        Create a copy of the TensionSpline profile instance.

        Returns
        -------
        TensionSpline
            A copied instance with all spline parameters duplicated and internal
            state consistently updated.
        """
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
        """
        Assign new spline parameters to the profile object.

        Parameters
        ----------
        pp_knots : ndarray
            Knot locations for pprime spline.
        pp_values : ndarray
            Spline values for pprime.
        pp_values_2 : ndarray
            Second derivative (or auxiliary) values for pprime spline.
        pp_sigma : ndarray
            Regularisation / smoothing parameters for pprime spline.
        ffp_knots : ndarray
            Knot locations for ffprime spline.
        ffp_values : ndarray
            Spline values for ffprime.
        ffp_values_2 : ndarray
            Second derivative (or auxiliary) values for ffprime spline.
        ffp_sigma : ndarray
            Regularisation / smoothing parameters for ffprime spline.
        """
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
    """
    General unconstrained toroidal current profile with FreeGSNKE extensions.

    """

    Jtor = Jtor_universal.Jtor

    def __init__(self, eq, *args, **kwargs):
        """
        Initialise the general pprime/ffprime current profile.

        Parameters
        ----------
        eq : FreeGSNKE Equilibrium object
            Equilibrium object defining grid geometry and limiter structure.
        """

        freegs4e.jtor.GeneralPprimeFFprime.__init__(self, *args, **kwargs)
        Jtor_universal.__init__(self)

        self.profile_parameter = []
        self.set_masks(eq=eq)

    def copy(self):
        """
        Create a copy of the GeneralPprimeFFprime profile instance.

        Returns
        -------
        GeneralPprimeFFprime
            A copied instance with all profile data and grid-dependent state
            duplicated and reinitialised appropriately.
        """
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
        """
        Reset profile parameter container.

        This profile is fully non-parametric, so no scalar or vector parameter
        set is required; the parameter container is explicitly cleared.
        """

        self.profile_parameter = []
