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

import freegs4e
import numpy as np
from freegs4e.gradshafranov import mu0

from . import jtor_refinement
from . import switch_profile as swp


class Jtor_universal:

    def __init__(self, refine_jtor=False):
        """Sets default unrefined Jtor."""
        if refine_jtor:
            self.Jtor = self.Jtor_refined
        else:
            self.Jtor = self.Jtor_unrefined

    def set_masks(self, eq):
        """Universal function to set all masks related to the limiter.

        Parameters
        ----------
        eq : FreeGSNKE Equilibrium object
            Specifies the domain properties
        """
        self.dRdZ = (eq.R_1D[1] - eq.R_1D[0]) * (eq.Z_1D[1] - eq.Z_1D[0])

        self.core_mask_limiter = eq.limiter_handler.core_mask_limiter

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
        if refine_jtor:
            self.jtor_refiner = jtor_refinement.Jtor_refiner(eq, nnx, nny)
            self.set_refinement_thresholds()
            self.Jtor = self.Jtor_refined
        else:
            self.Jtor = self.Jtor_unrefined

    def set_refinement_thresholds(self, thresholds=(1.0, 1.0)):
        """Sets the default criteria for refinement -- used when not directly set.

        Parameters
        ----------
        thresholds : tuple (threshold for jtor criterion, threshold for gradient criterion)
            tuple of values used to identify where to apply refinement
        """
        self.refinement_thresholds = thresholds

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

        opt, xpt, diverted_core_mask, diverted_psi_bndry = Jtor_part1(
            R, Z, psi, psi_bndry, mask_outside_limiter
        )

        if diverted_core_mask is None:
            # print('no xpt')
            psi_bndry, limiter_core_mask, flag_limiter = (
                diverted_psi_bndry,
                None,
                False,
            )
            # psi_bndry = np.amin(psi[self.limiter_mask_out])
            # diverted_core_mask = np.copy(self.mask_inside_limiter)

        else:
            psi_bndry, limiter_core_mask, flag_limiter = core_mask_limiter(
                psi,
                diverted_psi_bndry,
                diverted_core_mask,
                limiter_mask_out,
            )
            if np.sum(limiter_core_mask * self.mask_inside_limiter) == 0:
                limiter_core_mask = diverted_core_mask * self.mask_inside_limiter
                psi_bndry = 1.0 * diverted_psi_bndry

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
            self.Jtor_part1,
            self.Jtor_part2,
            # self.limiter_handler.core_mask_limiter,
            self.core_mask_limiter,
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
        self.unrefined_jtor = 1.0 * unrefined_jtor
        self.pure_jtor = unrefined_jtor / self.L
        core_mask = 1.0 * self.limiter_core_mask

        if thresholds == None:
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
        self.jtor = self.jtor_refiner.build_from_refined_jtor(
            self.pure_jtor, refined_jtor
        )
        if self.Ip_logic:
            self.L = self.Ip / (np.sum(self.jtor) * self.dRdZ)
            self.jtor *= self.L

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
