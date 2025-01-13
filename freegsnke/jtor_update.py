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
from freegs4e import critical
from freegs4e.gradshafranov import mu0

from . import limiter_func
from . import switch_profile as swp


class Jtor_universal:
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

        opt, xpt, diverted_core_mask, psi_bndry = Jtor_part1(
            R, Z, psi, psi_bndry, mask_outside_limiter
        )

        if diverted_core_mask is None:
            # print('no xpt')
            psi_bndry, limiter_core_mask, flag_limiter = (
                psi_bndry,
                None,
                False,
            )
        else:
            psi_bndry, limiter_core_mask, flag_limiter = core_mask_limiter(
                psi,
                psi_bndry,
                diverted_core_mask,
                limiter_mask_out,
            )

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

    def Jtor(self, R, Z, psi, psi_bndry=None):
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
            self.limiter_handler.core_mask_limiter,
            R,
            Z,
            psi,
            psi_bndry,
            self.mask_outside_limiter,
            self.limiter_mask_out,
        )
        return self.jtor


class ConstrainBetapIp(freegs4e.jtor.ConstrainBetapIp, Jtor_universal):
    """FreeGSNKE profile class adapting the original FreeGS object with the same name,
    with a few modifications, to:
    - retain memory of critical point calculation;
    - deal with limiter plasma configurations

    """

    def __init__(self, eq, limiter=None, *args, **kwargs):
        """Instantiates the object.

        Parameters
        ----------
        eq : FreeGSNKE Equilibrium object
            Specifies the domain properties
        limiter : freegs4e.machine.Wall object
            Specifies the limiter contour points.
            Only needs to be set here if purposedly using a limiter that is different from eq.tokamak.limiter.
            Otherwise use None
        """
        super().__init__(*args, **kwargs)
        self.profile_parameter = self.betap

        if limiter is None:
            self.limiter_handler = eq.limiter_handler
        else:
            self.limiter_handler = limiter_func.Limiter_handler(eq, limiter)

        self.mask_inside_limiter = self.limiter_handler.mask_inside_limiter
        self.mask_outside_limiter = np.logical_not(self.mask_inside_limiter)
        self.limiter_mask_out = self.limiter_handler.limiter_mask_out

        # Note the factor 2 is not a typo: used in critical.inside_mask
        self.mask_outside_limiter = (2 * self.mask_outside_limiter).astype(float)

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

    def __init__(self, eq, limiter=None, *args, **kwargs):
        """Instantiates the object.

        Parameters
        ----------
        eq : FreeGSNKE Equilibrium object
            Specifies the domain properties
        limiter : freegs4e.machine.Wall object
            Specifies the limiter contour points
            Only needs to be set here if purposedly using a limiter that is different from eq.tokamak.limiter.
            Otherwise use None
        """
        super().__init__(*args, **kwargs)
        self.profile_parameter = self.paxis

        if limiter is None:
            self.limiter_handler = eq.limiter_handler
        else:
            self.limiter_handler = limiter_func.Limiter_handler(eq, limiter)

        self.mask_inside_limiter = self.limiter_handler.mask_inside_limiter
        self.mask_outside_limiter = np.logical_not(self.mask_inside_limiter)
        self.limiter_mask_out = self.limiter_handler.limiter_mask_out
        self.limiter_mask_for_plotting = (
            self.mask_inside_limiter
            + self.limiter_handler.make_layer_mask(
                self.mask_inside_limiter, layer_size=1
            )
        ) > 0

        # Note the factor 2 is not a typo: used in critical.inside_mask
        self.mask_outside_limiter = (2 * self.mask_outside_limiter).astype(float)

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

    def __init__(self, eq, limiter=None, *args, **kwargs):
        """Instantiates the object.

        Parameters
        ----------
        eq : FreeGSNKE Equilibrium object
            Specifies the domain properties
        limiter : freegs4e.machine.Wall object
            Specifies the limiter contour points
            Only needs to be set here if purposedly using a limiter that is different from eq.tokamak.limiter.
            Otherwise use None
        """
        super().__init__(*args, **kwargs)
        self.profile_parameter = self.Beta0

        if limiter is None:
            self.limiter_handler = eq.limiter_handler
        else:
            self.limiter_handler = limiter_func.Limiter_handler(eq, limiter)

        self.mask_inside_limiter = self.limiter_handler.mask_inside_limiter
        self.mask_outside_limiter = np.logical_not(self.mask_inside_limiter)
        self.limiter_mask_out = self.limiter_handler.limiter_mask_out
        self.limiter_mask_for_plotting = (
            self.mask_inside_limiter
            + self.limiter_handler.make_layer_mask(
                self.mask_inside_limiter, layer_size=1
            )
        ) > 0

        # Note the factor 2 is not a typo: used in critical.inside_mask
        self.mask_outside_limiter = (2 * self.mask_outside_limiter).astype(float)

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

    def __init__(self, eq, limiter=None, *args, **kwargs):
        """Instantiates the object.

        Parameters
        ----------
        eq : freegs4e Equilibrium object
            Specifies the domain properties
        limiter : freegs4e.machine.Wall object
            Specifies the limiter contour points
            Only set if a limiter different from eq.tokamak.limiter is to be used.

        """
        super().__init__(*args, **kwargs)

        if limiter is None:
            self.limiter_handler = eq.limiter_handler
        else:
            self.limiter_handler = limiter_func.Limiter_handler(eq, limiter)

        self.mask_inside_limiter = self.limiter_handler.mask_inside_limiter
        self.mask_outside_limiter = np.logical_not(self.mask_inside_limiter)
        self.limiter_mask_out = self.limiter_handler.limiter_mask_out
        self.limiter_mask_for_plotting = (
            self.mask_inside_limiter
            + self.limiter_handler.make_layer_mask(
                self.mask_inside_limiter, layer_size=1
            )
        ) > 0

        # Note the factor 2 is not a typo: used in critical.inside_mask
        self.mask_outside_limiter = (2 * self.mask_outside_limiter).astype(float)

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

    def __init__(self, eq, limiter=None, *args, **kwargs):
        """Instantiates the object.

        Parameters
        ----------
        eq : FreeGSNKE Equilibrium object
            Specifies the domain properties
        limiter : freegs4e.machine.Wall object
            Specifies the limiter contour points
            Only needs to be set here if purposedly using a limiter that is different from eq.tokamak.limiter.
            Otherwise use None
        """

        super().__init__(*args, **kwargs)
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

        if limiter is None:
            self.limiter_handler = eq.limiter_handler
        else:
            self.limiter_handler = limiter_func.Limiter_handler(eq, limiter)

        self.mask_inside_limiter = self.limiter_handler.mask_inside_limiter
        self.mask_outside_limiter = np.logical_not(self.mask_inside_limiter)
        self.limiter_mask_out = self.limiter_handler.limiter_mask_out
        self.limiter_mask_for_plotting = (
            self.mask_inside_limiter
            + self.limiter_handler.make_layer_mask(
                self.mask_inside_limiter, layer_size=1
            )
        ) > 0

        # Note the factor 2 is not a typo: used in critical.inside_mask
        self.mask_outside_limiter = (2 * self.mask_outside_limiter).astype(float)

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
