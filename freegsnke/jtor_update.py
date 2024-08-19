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
        """Universal function that calculates the plasma current distribution
        over the different types of profile parametrizations.

        Parameters
        ----------
        Jtor_part1 : method
            method from the freegs4e Profile class
            returns opt, xpt, diverted_core_mask
        Jtor_part2 : method
            method from each individual profile class
            returns jtor itself
        core_mask_limiter : method
            method of the limiter_handles class
            returns the refined core_mask where jtor>0 accounting for the limiter
        R : np.ndarray
                R coordinates of the grid points
        Z : np.ndarray
            Z coordinates of the grid points
        psi : np.ndarray
            Poloidal field flux / 2*pi at each grid points (as returned by FreeGS.Equilibrium.psi())
        psi_bndry : float, optional
            Value of the poloidal field flux at the boundary of the plasma (last closed flux surface), by default None
        mask_outside_limiter : np.ndarray
            Mask of points outside the limiter, if any, optional

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
    """FreeGS profile class with a few modifications, to:
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
            Specifies the limiter contour points.
            Only set here if a limiter different from eq.tokamak.limiter is to be used.
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

        # this is used in critical.inside_mask
        self.mask_outside_limiter = (2 * self.mask_outside_limiter).astype(float)

    def get_pars(
        self,
    ):
        """Fetches all profile parameters and returns them in a single array"""
        return np.array([self.alpha_m, self.alpha_n, self.betap])

    def assign_profile_parameter(self, betap):
        """Assigns to the profile object a new value of the profile parameter betap"""
        self.betap = betap
        self.profile_parameter = betap

    def assign_profile_coefficients(self, alpha_m, alpha_n):
        """Assigns to the profile object new value of the coefficients (alpha_m, alpha_n)"""
        self.alpha_m = alpha_m
        self.alpha_n = alpha_n

    # def _Jtor(self, R, Z, psi, psi_bndry=None, rel_psi_error=0):
    #     """Replaces the original FreeGS Jtor method if FreeGS4E is not available."""
    #     self.jtor = super().Jtor(R, Z, psi, psi_bndry)
    #     self.opt, self.xpt = critical.find_critical(R, Z, psi)

    #     self.diverted_core_mask = self.jtor > 0
    #     self.psi_bndry, mask, self.limiter_flag = (
    #         self.limiter_handler.core_mask_limiter(
    #             psi,
    #             self.xpt[0][2],
    #             self.diverted_core_mask,
    #             self.limiter_mask_out,
    #         )
    #     )
    #     self.jtor = super().Jtor(R, Z, psi, self.psi_bndry)
    #     return self.jtor

    # def Jtor_fast(self, R, Z, psi, psi_bndry=None, rel_psi_error=0):
    #     """Used when FreeGS4E is available."""
    #     self.diverted_core_mask = super().Jtor_part1(R, Z, psi, psi_bndry)
    #     if self.diverted_core_mask is None:
    #         # print('no xpt')
    #         self.psi_bndry, self.limiter_core_mask, self.flag_limiter = (
    #             psi_bndry,
    #             None,
    #             False,
    #         )
    #     elif rel_psi_error < 0.02:
    #         self.psi_bndry, self.limiter_core_mask, self.flag_limiter = (
    #             self.limiter_handler.core_mask_limiter(
    #                 psi,
    #                 self.psi_bndry,
    #                 self.diverted_core_mask,
    #                 self.limiter_mask_out,
    #             )
    #         )
    #     else:
    #         self.limiter_core_mask = self.diverted_core_mask.copy()
    #     self.jtor = super().Jtor_part2(
    #         R, Z, psi, self.psi_bndry, self.limiter_core_mask
    #     )
    #     return self.jtor

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
    """FreeGS4E profile class with a few modifications, to:
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
        self.mask_outside_limiter = (2 * self.mask_outside_limiter).astype(float)

        # if not hasattr(self, "fast"):
        #     self.Jtor = self._Jtor
        # else:
        #     self.Jtor = self.Jtor_fast

    def get_pars(
        self,
    ):
        """Fetches all profile parameters and returns them in a single array"""
        return np.array([self.alpha_m, self.alpha_n, self.paxis])

    def assign_profile_parameter(self, paxis):
        """Assigns to the profile object a new value of the profile parameter paxis"""
        self.paxis = paxis
        self.profile_parameter = paxis

    def assign_profile_coefficients(self, alpha_m, alpha_n):
        """Assigns to the profile object new value of the coefficients (alpha_m, alpha_n)"""
        self.alpha_m = alpha_m
        self.alpha_n = alpha_n

    # def _Jtor(self, R, Z, psi, psi_bndry=None, rel_psi_error=0):
    #     """Replaces the original FreeGS Jtor method if FreeGS4E is not available."""
    #     self.jtor = super().Jtor(R, Z, psi, psi_bndry)
    #     self.opt, self.xpt = critical.find_critical(R, Z, psi)

    #     self.diverted_core_mask = self.jtor > 0
    #     self.psi_bndry, mask, self.limiter_flag = (
    #         self.limiter_handler.core_mask_limiter(
    #             psi,
    #             self.xpt[0][2],
    #             self.diverted_core_mask,
    #             self.limiter_mask_out,
    #         )
    #     )
    #     self.jtor = super().Jtor(R, Z, psi, self.psi_bndry)
    #     return self.jtor

    # def Jtor_fast(self, R, Z, psi, psi_bndry=None, rel_psi_error=0):
    #     """Used when FreeGS4E is available."""

    #     opt, xpt = super().Jtor_part1(R, Z, psi, psi_bndry)

    #     if psi_bndry is not None:
    #         self.diverted_core_mask = critical.inside_mask(R, Z, psi, opt, xpt, self.mask_outside_limiter, psi_bndry)
    #     elif xpt:
    #         psi_bndry = xpt[0][2]
    #         self.diverted_core_mask = critical.inside_mask(R, Z, psi, opt, xpt, self.mask_outside_limiter, psi_bndry)
    #     else:
    #         # No X-points
    #         psi_bndry = psi[0, 0]
    #         self.diverted_core_mask = None

    #     psi_axis = opt[0][2]
    #     # # check correct sorting between psi_axis and psi_bndry
    #     if (psi_axis-psi_bndry)*self.Ip < 0:
    #         raise ValueError("Incorrect critical points! Likely due to not suitable psi_plasma")

    #     # added with respect to original Jtor
    #     self.xpt = xpt
    #     self.opt = opt
    #     self.psi_bndry = psi_bndry
    #     self.psi_axis = psi_axis

    #     if self.diverted_core_mask is None:
    #         # print('no xpt')
    #         self.psi_bndry, self.limiter_core_mask, self.flag_limiter = (
    #             psi_bndry,
    #             None,
    #             False,
    #         )
    #     elif rel_psi_error < 0.02:
    #         self.psi_bndry, self.limiter_core_mask, self.flag_limiter = (
    #             self.limiter_handler.core_mask_limiter(
    #                 psi,
    #                 self.psi_bndry,
    #                 self.diverted_core_mask,
    #                 self.limiter_mask_out,
    #             )
    #         )
    #     else:
    #         self.limiter_core_mask = self.diverted_core_mask.copy()

    #     self.jtor = super().Jtor_part2(
    #         R, Z, psi, self.psi_bndry, self.limiter_core_mask
    #     )
    #     return self.jtor

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
    """FreeGS profile class with a few modifications, to:
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
        self.mask_outside_limiter = (2 * self.mask_outside_limiter).astype(float)

        # if not hasattr(self, "fast"):
        #     self.Jtor = self._Jtor
        # else:
        #     self.Jtor = self.Jtor_fast

    def assign_profile_parameter(self, Beta0):
        """Assigns to the profile object a new value of the profile parameter paxis"""
        self.Beta0 = Beta0
        self.profile_parameter = Beta0

    def assign_profile_coefficients(self, alpha_m, alpha_n):
        """Assigns to the profile object new value of the coefficients (alpha_m, alpha_n)"""
        self.alpha_m = alpha_m
        self.alpha_n = alpha_n

    # def _Jtor(self, R, Z, psi, psi_bndry=None, rel_psi_error=0):
    #     """Replaces the original FreeGS Jtor method if FreeGS4E is not available."""
    #     self.jtor = super().Jtor(R, Z, psi, psi_bndry)
    #     self.opt, self.xpt = critical.find_critical(R, Z, psi)

    #     self.diverted_core_mask = self.jtor > 0
    #     self.psi_bndry, mask, self.limiter_flag = (
    #         self.limiter_handler.core_mask_limiter(
    #             psi,
    #             self.xpt[0][2],
    #             self.diverted_core_mask,
    #             self.limiter_mask_out,
    #         )
    #     )
    #     self.jtor = super().Jtor(R, Z, psi, self.psi_bndry)
    #     return self.jtor

    # def Jtor_fast(self, R, Z, psi, psi_bndry=None, rel_psi_error=0):
    #     """Used when FreeGS4E is available."""
    #     opt, xpt = super().Jtor_part1(R, Z, psi, psi_bndry)

    #     if psi_bndry is not None:
    #         self.diverted_core_mask = critical.inside_mask(R, Z, psi, opt, xpt, self.mask_outside_limiter, psi_bndry)
    #     elif xpt:
    #         psi_bndry = xpt[0][2]
    #         self.diverted_core_mask = critical.inside_mask(R, Z, psi, opt, xpt, self.mask_outside_limiter, psi_bndry)
    #     else:
    #         # No X-points
    #         psi_bndry = psi[0, 0]
    #         self.diverted_core_mask = None

    #     psi_axis = opt[0][2]
    #     # # check correct sorting between psi_axis and psi_bndry
    #     if (psi_axis-psi_bndry)*self.Ip < 0:
    #         raise ValueError("Incorrect critical points! Likely due to not suitable psi_plasma")

    #     # added with respect to original Jtor
    #     self.xpt = xpt
    #     self.opt = opt
    #     self.psi_bndry = psi_bndry
    #     self.psi_axis = psi_axis

    #     if self.diverted_core_mask is None:
    #         # print('no xpt')
    #         self.psi_bndry, self.limiter_core_mask, self.flag_limiter = (
    #             psi_bndry,
    #             None,
    #             False,
    #         )
    #     elif rel_psi_error < 0.02:
    #         self.psi_bndry, self.limiter_core_mask, self.flag_limiter = (
    #             self.limiter_handler.core_mask_limiter(
    #                 psi,
    #                 self.psi_bndry,
    #                 self.diverted_core_mask,
    #                 self.limiter_mask_out,
    #             )
    #         )
    #     else:
    #         self.limiter_core_mask = self.diverted_core_mask.copy()

    #     self.jtor = super().Jtor_part2(
    #         R, Z, psi, self.psi_bndry, self.limiter_core_mask
    #     )
    #     return self.jtor

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
    """FreeGS profile class with a few modifications, to:
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
        self.profile_parameter = [self.alpha, self.beta]

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
        self.mask_outside_limiter = (2 * self.mask_outside_limiter).astype(float)

        # if not hasattr(self, "fast"):
        #     self.Jtor = self._Jtor
        # else:
        #     self.Jtor = self.Jtor_fast

    def assign_profile_parameter(self, alpha, beta):
        """Assigns to the profile object a new value of the profile parameter paxis"""
        self.alpha = alpha
        self.beta = beta
        self.profile_parameter = [alpha, beta]

    def get_pars(
        self,
    ):
        """Fetches all profile parameters and returns them in a single array"""
        # This is a temporary fix that allows the linearization to work on lao profiles
        # Changes in the profile are ignored in the linearised dynamics
        return np.array([])

    # def _Jtor(self, R, Z, psi, psi_bndry=None, rel_psi_error=0):
    #     """Replaces the original FreeGS Jtor method if FreeGS4E is not available."""
    #     self.jtor = super().Jtor(R, Z, psi, psi_bndry)
    #     self.opt, self.xpt = critical.find_critical(R, Z, psi)

    #     self.diverted_core_mask = self.jtor > 0
    #     self.psi_bndry, mask, self.limiter_flag = (
    #         self.limiter_handler.core_mask_limiter(
    #             psi,
    #             self.xpt[0][2],
    #             self.diverted_core_mask,
    #             self.limiter_mask_out,
    #         )
    #     )
    #     self.jtor = super().Jtor(R, Z, psi, self.psi_bndry)
    #     return self.jtor

    # def Jtor_fast_old(self, R, Z, psi, psi_bndry=None, rel_psi_error=0):
    #     """Used when FreeGS4E is available."""

    #     opt, xpt = super().Jtor_part1(R, Z, psi, psi_bndry)

    #     if psi_bndry is not None:
    #         self.diverted_core_mask = critical.inside_mask(R, Z, psi, opt, xpt, self.mask_outside_limiter, psi_bndry)
    #     elif xpt:
    #         psi_bndry = xpt[0][2]
    #         psi_axis = opt[0][2]
    #         # # check correct sorting between psi_axis and psi_bndry
    #         if (psi_axis-psi_bndry)*self.Ip < 0:
    #             raise ValueError("Incorrect critical points! Likely due to not suitable psi_plasma")
    #         self.diverted_core_mask = critical.inside_mask(R, Z, psi, opt, xpt, self.mask_outside_limiter, psi_bndry)
    #     else:
    #         # No X-points
    #         psi_bndry = psi[0, 0]
    #         self.diverted_core_mask = None

    #     # added with respect to original Jtor
    #     self.xpt = xpt
    #     self.opt = opt
    #     self.psi_bndry = psi_bndry
    #     self.psi_axis = psi_axis

    #     if self.diverted_core_mask is None:
    #         # print('no xpt')
    #         self.psi_bndry, self.limiter_core_mask, self.flag_limiter = (
    #             psi_bndry,
    #             None,
    #             False,
    #         )
    #     elif True: #rel_psi_error < 0.02:
    #         self.psi_bndry, self.limiter_core_mask, self.flag_limiter = (
    #             self.limiter_handler.core_mask_limiter(
    #                 psi,
    #                 self.psi_bndry,
    #                 self.diverted_core_mask,
    #                 self.limiter_mask_out,
    #             )
    #         )
    #     else:
    #         self.limiter_core_mask = self.diverted_core_mask.copy()

    #     self.jtor = super().Jtor_part2(
    #         R, Z, psi, self.psi_bndry, self.limiter_core_mask
    #     )
    #     return self.jtor

    # def Jtor(self, R, Z, psi, psi_bndry=None):
    #     self.jtor, self.opt, self.xpt, self.psi_bndry, self.diverted_core_mask, self.limiter_core_mask, self.flag_limiter = Jtor_fast(self.Jtor_part1,
    #                      self.Jtor_part2,
    #                      self.limiter_handler.core_mask_limiter,
    #                      R, Z, psi, psi_bndry,
    #                      self.mask_outside_limiter,
    #                      self.limiter_mask_out)
    #     return self.jtor

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
    """FreeGS profile class with a few modifications, to:
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
        self.mask_outside_limiter = (2 * self.mask_outside_limiter).astype(float)

        # if not hasattr(self, "fast"):
        #     self.Jtor = self._Jtor
        # else:
        #     self.Jtor = self.Jtor_fast

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

    def get_pars(
        self,
    ):
        """Fetches all profile parameters and returns them in a single array"""
        # This is a temporary fix that allows the linearization to work on lao profiles
        # Changes in the profile are ignored in the linearised dynamics
        return np.array([])
