import freegs
import numpy as np
from freegs import critical
from freegs.gradshafranov import mu0

from . import limiter_func


def Lao_parameters_finder(
    pn_,
    pprime_,
    ffprime_,
    n_alpha,
    n_beta,
    alpha_logic=True,
    beta_logic=True,
    Ip_logic=True,
):
    """Finds best fitting alpha, beta parameters for a Lao85 profile,
    to reproduce the input pprime_ and ffprime_
    n_alpha and n_beta represent the number of free parameters
    Simple linear fit.

    Parameters
    ----------
    pn_ : np.array
        Normalised plasma function, array of values btw 0 and 1.
    pprime_ : np.array
        pprime(pn_)
    ffprime_ : np.array
        ffprime(pn_)
    n_alpha : int
        Number of free parameters for the pprime term
    n_beta : int
        Number of free parameters for the ffprime term
    alpha_logic : bool, optional
        add polynomial term in Lao85 profile so that ppprime(1)=0, by default True
    beta_logic : bool, optional
        add polynomial term so that ffpprime(1)=0,, by default True
    Ip_logic : bool, optional
        if False, scale coefficients, by default True
    nn : int, optional
        number of points to be used for fit, by default 100

    Returns
    -------
    np.array, np.array
        optimal alpha and beta parameters
        Note these need to be used with the same alpha and beta logic terms provided as inputs.
    """

    pprime0_ = pprime_[0]
    pprime_ /= pprime0_
    ffprime0_ = ffprime_[0]
    ffprime_ /= ffprime0_

    alpha = np.arange(n_alpha)
    ppn = pn_[:, np.newaxis] ** alpha[np.newaxis, :]
    if alpha_logic is True:
        ppn -= pn_[:, np.newaxis] ** n_alpha
    alpha = np.matmul(np.matmul(np.linalg.inv(np.matmul(ppn.T, ppn)), ppn.T), pprime_)

    beta = np.arange(n_beta)
    ppn = pn_[:, np.newaxis] ** beta[np.newaxis, :]
    if beta_logic is True:
        ppn -= pn_[:, np.newaxis] ** n_beta
    beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(ppn.T, ppn)), ppn.T), ffprime_)

    if Ip_logic is False:
        alpha *= pprime0_
        beta *= ffprime0_
    else:
        beta *= ffprime0_ / pprime0_

    return alpha, beta


# Functions for the Lao85.Topeol_parameters optimiser
def Topeol_std(x, alpha_m, alpha_n, beta_0):
    return (1 - x**alpha_m) ** alpha_n


def d2Ldb2(t, x, alpha_m, alpha_n, beta_0, Tstd=None):
    if Tstd is None:
        Tstd = Topeol_std(x, alpha_m, alpha_n, beta_0)
    res = 2 * Tstd**2
    return res


def d2Ldbdn(t, x, alpha_m, alpha_n, beta_0, Tstd=None):
    if Tstd is None:
        Tstd = Topeol_std(x, alpha_m, alpha_n, beta_0)
    res = 2 * beta_0 * Tstd - t
    res *= 2 * Tstd * np.log(Topeol_std(x, alpha_m, 1, beta_0))
    return res


def d2Ldbdm(t, x, alpha_m, alpha_n, beta_0, Tstd=None):
    if Tstd is None:
        Tstd = Topeol_std(x, alpha_m, alpha_n, beta_0)
    res = 2 * beta_0 * Tstd - t
    res *= Topeol_std(x, alpha_m, alpha_n - 1, beta_0)
    res *= -2 * alpha_n * np.log(x) * x**alpha_m
    return res


def d2Ldm2(t, x, alpha_m, alpha_n, beta_0, Tstd=None):
    if Tstd is None:
        Tstd = Topeol_std(x, alpha_m, alpha_n, beta_0)
    res = beta_0 * Tstd
    res *= 2 * alpha_n * x**alpha_m - 1
    res += t - alpha_n * t * x**alpha_m
    res *= Topeol_std(x, alpha_m, alpha_n - 2, beta_0)
    res *= 2 * beta_0 * alpha_n * x**alpha_m * np.log(x) ** 2
    return res


def d2Ldn2(t, x, alpha_m, alpha_n, beta_0, Tstd=None):
    if Tstd is None:
        Tstd = Topeol_std(x, alpha_m, alpha_n, beta_0)
    res = 2 * beta_0 * Tstd - t
    res *= np.log(Topeol_std(x, alpha_m, 1, beta_0)) ** 2
    res *= Tstd
    res *= 2 * beta_0
    return res


def d2Ldmdn(t, x, alpha_m, alpha_n, beta_0, Tstd=None):
    if Tstd is None:
        Tstd = Topeol_std(x, alpha_m, alpha_n, beta_0)
    res = 2 * beta_0 * Tstd - t
    res *= alpha_n * np.log(Topeol_std(x, alpha_m, 1, beta_0))
    res += beta_0 * Tstd - t
    res *= Topeol_std(x, alpha_m, alpha_n - 1, beta_0)
    res *= -2 * beta_0 * x**alpha_m * np.log(x)
    return res


def dLdn(t, x, alpha_m, alpha_n, beta_0, Tstd=None):
    if Tstd is None:
        Tstd = Topeol_std(x, alpha_m, alpha_n, beta_0)
    res = t - beta_0 * Tstd
    res *= np.log(Topeol_std(x, alpha_m, 1, beta_0))
    res *= -2 * beta_0 * Tstd
    return res


def dLdm(t, x, alpha_m, alpha_n, beta_0, Tstd=None):
    if Tstd is None:
        Tstd = Topeol_std(x, alpha_m, alpha_n, beta_0)
    res = t - beta_0 * Tstd
    res *= np.log(x) * Topeol_std(x, alpha_m, alpha_n - 1, beta_0)
    res *= 2 * beta_0 * alpha_n * x**alpha_m
    return res


def dLdb(t, x, alpha_m, alpha_n, beta_0, Tstd=None):
    if Tstd is None:
        Tstd = Topeol_std(x, alpha_m, alpha_n, beta_0)
    res = t - beta_0 * Tstd
    res *= -2 * Tstd
    return res


def dLdpars(tp, tf, x, alpha_m, alpha_n, beta_0, Tstd=None):
    dLpdpars = np.zeros((len(tp), 3))
    dLpdpars[:, 0] = dLdm(tp, x, alpha_m, alpha_n, beta_0, Tstd)
    dLpdpars[:, 1] = dLdn(tp, x, alpha_m, alpha_n, beta_0, Tstd)
    dLpdpars[:, 2] = dLdb(tp, x, alpha_m, alpha_n, beta_0, Tstd)
    dLpdpars = np.sum(dLpdpars, axis=0)

    dLfdpars = np.zeros((len(tp), 3))
    dLfdpars[:, 0] = dLdm(tf, x, alpha_m, alpha_n, 1 - beta_0, Tstd)
    dLfdpars[:, 1] = dLdn(tf, x, alpha_m, alpha_n, 1 - beta_0, Tstd)
    dLfdpars[:, 2] = -dLdb(tf, x, alpha_m, alpha_n, 1 - beta_0, Tstd)
    dLfdpars = np.sum(dLfdpars, axis=0)
    return dLpdpars + dLfdpars


def d2Ldpars2(tp, tf, x, alpha_m, alpha_n, beta_0, Tstd=None):
    d2Lpdpars2 = np.zeros((3, 3))
    d2Lpdpars2[0, 0] = np.sum(d2Ldm2(tp, x, alpha_m, alpha_n, beta_0, Tstd))
    d2Lpdpars2[1, 1] = np.sum(d2Ldn2(tp, x, alpha_m, alpha_n, beta_0, Tstd))
    d2Lpdpars2[2, 2] = np.sum(d2Ldb2(tp, x, alpha_m, alpha_n, beta_0, Tstd))
    d2Lpdpars2[0, 1] = d2Lpdpars2[1, 0] = np.sum(
        d2Ldmdn(tp, x, alpha_m, alpha_n, beta_0, Tstd)
    )
    d2Lpdpars2[0, 2] = d2Lpdpars2[2, 0] = np.sum(
        d2Ldbdm(tp, x, alpha_m, alpha_n, beta_0, Tstd)
    )
    d2Lpdpars2[1, 2] = d2Lpdpars2[2, 1] = np.sum(
        d2Ldbdn(tp, x, alpha_m, alpha_n, beta_0, Tstd)
    )

    d2Lfdpars2 = np.zeros((3, 3))
    d2Lfdpars2[0, 0] = np.sum(d2Ldm2(tf, x, alpha_m, alpha_n, 1 - beta_0, Tstd))
    d2Lfdpars2[1, 1] = np.sum(d2Ldn2(tf, x, alpha_m, alpha_n, 1 - beta_0, Tstd))
    d2Lfdpars2[2, 2] = np.sum(d2Ldb2(tf, x, alpha_m, alpha_n, 1 - beta_0, Tstd))
    d2Lfdpars2[0, 1] = d2Lfdpars2[1, 0] = np.sum(
        d2Ldmdn(tf, x, alpha_m, alpha_n, 1 - beta_0, Tstd)
    )
    d2Lfdpars2[0, 2] = d2Lfdpars2[2, 0] = -np.sum(
        d2Ldbdm(tf, x, alpha_m, alpha_n, 1 - beta_0, Tstd)
    )
    d2Lfdpars2[1, 2] = d2Lfdpars2[2, 1] = -np.sum(
        d2Ldbdn(tf, x, alpha_m, alpha_n, 1 - beta_0, Tstd)
    )
    return d2Lpdpars2 + d2Lfdpars2


def Lpars(tp, tf, x, alpha_m, alpha_n, beta_0, Tstd=None):
    if Tstd is None:
        Tstd = Topeol_std(x, alpha_m, alpha_n, beta_0)
    Lp = (tp - beta_0 * Tstd) ** 2
    Lf = (tf - (1 - beta_0) * Tstd) ** 2
    return np.sum(Lp + Lf, axis=0)


def Topeol_opt_init(tp, tf):
    tpn = tp / max(tp[0], tf[0])
    tfn = tf / max(tp[0], tf[0])

    mask = tfn > 0
    rr = np.mean(tpn[mask] / tfn[mask])
    b0 = rr / (1 + rr)

    tpn = b0 * tp / tp[0]
    tfn = (1 - b0) * tf / tf[0]
    return tpn, tfn, b0


def Topeol_opt_stepper(tp, tf, x, pars):
    alpha_m, alpha_n, beta_0 = pars
    Tstd = Topeol_std(x, alpha_m, alpha_n, beta_0)
    dLdp = dLdpars(tp, tf, x, alpha_m, alpha_n, beta_0, Tstd)
    d2Ldp2 = d2Ldpars2(tp, tf, x, alpha_m, alpha_n, beta_0, Tstd)
    eigvals = np.linalg.eigvals(d2Ldp2)
    if np.sum(eigvals > 0) > 2:
        dpars = np.dot(np.linalg.inv(d2Ldp2), -dLdp)
    else:
        ll = Lpars(tp, tf, x, alpha_m, alpha_n, beta_0, Tstd)
        dpars = -ll * dLdp / np.linalg.norm(dLdp)
    ratio = pars / np.abs(dpars)
    if np.any(ratio > 2):
        dpars = np.where(ratio > 2, dpars, np.sign(dpars) * pars / 2)
    return pars + dpars


def Topeol_opt(tp, tf, x, max_it, tol):
    tpn, tfn, b0 = Topeol_opt_init(tp, tf)
    it = 0
    pars = np.array([2, 1, b0])
    new_pars = Topeol_opt_stepper(tpn, tfn, x, pars)
    control = np.any(np.abs(pars - new_pars) > tol)
    while control and it < max_it:
        pars = new_pars.copy()
        new_pars = Topeol_opt_stepper(tpn, tfn, x, pars)
        control = np.any(np.abs(pars - new_pars) > tol)
        it += 1
    if it == max_it:
        print("Optimization failed to converge in", max_it, "iterations.")
    return new_pars


class ConstrainBetapIp(freegs.jtor.ConstrainBetapIp):
    """FreeGS profile class with a few modifications, to:
    - retain memory of critical point calculation;
    - deal with limiter plasma configurations

    """

    def __init__(self, eq, limiter=None, *args, **kwargs):
        """Instantiates the object.

         Parameters
        ----------
        eq : freeGS Equilibrium object
            Specifies the domain properties
        limiter : freeGS.machine.Wall object
            Specifies the limiter contour points.
            Only set if a limiter different from eq.tokamak.limiter is to be used.
        """
        super().__init__(*args, **kwargs)
        self.profile_parameter = self.betap

        if limiter is None:
            self.limiter_handler = eq.limiter_handler

            self.limiter_mask_out = eq.limiter_mask_out
            self.mask_inside_limiter = eq.mask_inside_limiter
            self.limiter_mask_for_plotting = (
                self.mask_inside_limiter + self.limiter_mask_out
            ) > 0
        else:
            self.limiter_handler = limiter_func.Limiter_handler(eq, limiter)

            self.mask_inside_limiter = self.limiter_handler.mask_inside_limiter
            self.limiter_mask_out = self.limiter_handler.make_layer_mask(layer_size=1)
            self.limiter_mask_for_plotting = (
                self.mask_inside_limiter + self.limiter_mask_out
            ) > 0

        if not hasattr(self, "fast"):
            self.Jtor = self._Jtor
        else:
            self.Jtor = self.Jtor_fast

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

    def _Jtor(self, R, Z, psi, psi_bndry=None, rel_psi_error=0):
        """Replaces the original FreeGS Jtor method if FreeGSfast is not available."""
        self.jtor = super().Jtor(R, Z, psi, psi_bndry)
        self.opt, self.xpt = critical.find_critical(R, Z, psi)

        self.diverted_core_mask = self.jtor > 0
        self.psi_bndry, mask, self.limiter_flag = (
            self.limiter_handler.core_mask_limiter(
                psi,
                self.xpt[0][2],
                self.diverted_core_mask,
                self.limiter_mask_out,
            )
        )
        self.jtor = super().Jtor(R, Z, psi, self.psi_bndry)
        return self.jtor

    def Jtor_fast(self, R, Z, psi, psi_bndry=None, rel_psi_error=0):
        """Used when FreeGSfast is available."""
        self.diverted_core_mask = super().Jtor_part1(R, Z, psi, psi_bndry)
        if self.diverted_core_mask is None:
            # print('no xpt')
            self.psi_bndry, self.limiter_core_mask, self.flag_limiter = (
                psi_bndry,
                None,
                False,
            )
        elif rel_psi_error < 0.02:
            self.psi_bndry, self.limiter_core_mask, self.flag_limiter = (
                self.limiter_handler.core_mask_limiter(
                    psi,
                    self.psi_bndry,
                    self.diverted_core_mask,
                    self.limiter_mask_out,
                )
            )
        else:
            self.limiter_core_mask = self.diverted_core_mask.copy()
        self.jtor = super().Jtor_part2(
            R, Z, psi, self.psi_bndry, self.limiter_core_mask
        )
        return self.jtor

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

        alpha, beta = Lao_parameters_finder(
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


class ConstrainPaxisIp(freegs.jtor.ConstrainPaxisIp):
    """FreeGS profile class with a few modifications, to:
    - retain memory of critical point calculation;
    - deal with limiter plasma configurations

    """

    def __init__(self, eq, limiter=None, *args, **kwargs):
        """Instantiates the object.

        Parameters
        ----------
        eq : freeGS Equilibrium object
            Specifies the domain properties
        limiter : freeGS.machine.Wall object
            Specifies the limiter contour points
            Only set if a limiter different from eq.tokamak.limiter is to be used.

        """
        super().__init__(*args, **kwargs)
        self.profile_parameter = self.paxis

        if limiter is None:
            self.limiter_handler = eq.limiter_handler

            self.limiter_mask_out = eq.limiter_mask_out
            self.mask_inside_limiter = eq.mask_inside_limiter
            self.limiter_mask_for_plotting = (
                self.mask_inside_limiter + self.limiter_mask_out
            ) > 0
        else:
            self.limiter_handler = limiter_func.Limiter_handler(eq, limiter)

            self.mask_inside_limiter = self.limiter_handler.mask_inside_limiter
            self.limiter_mask_out = self.limiter_handler.make_layer_mask(layer_size=1)
            self.limiter_mask_for_plotting = (
                self.mask_inside_limiter + self.limiter_mask_out
            ) > 0

        if not hasattr(self, "fast"):
            self.Jtor = self._Jtor
        else:
            self.Jtor = self.Jtor_fast

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

    def _Jtor(self, R, Z, psi, psi_bndry=None, rel_psi_error=0):
        """Replaces the original FreeGS Jtor method if FreeGSfast is not available."""
        self.jtor = super().Jtor(R, Z, psi, psi_bndry)
        self.opt, self.xpt = critical.find_critical(R, Z, psi)

        self.diverted_core_mask = self.jtor > 0
        self.psi_bndry, mask, self.limiter_flag = (
            self.limiter_handler.core_mask_limiter(
                psi,
                self.xpt[0][2],
                self.diverted_core_mask,
                self.limiter_mask_out,
            )
        )
        self.jtor = super().Jtor(R, Z, psi, self.psi_bndry)
        return self.jtor

    def Jtor_fast(self, R, Z, psi, psi_bndry=None, rel_psi_error=0):
        """Used when FreeGSfast is available."""
        self.diverted_core_mask = super().Jtor_part1(R, Z, psi, psi_bndry)
        if self.diverted_core_mask is None:
            # print('no xpt')
            self.psi_bndry, self.limiter_core_mask, self.flag_limiter = (
                psi_bndry,
                None,
                False,
            )
        elif rel_psi_error < 0.02:
            self.psi_bndry, self.limiter_core_mask, self.flag_limiter = (
                self.limiter_handler.core_mask_limiter(
                    psi,
                    self.psi_bndry,
                    self.diverted_core_mask,
                    self.limiter_mask_out,
                )
            )
        else:
            self.limiter_core_mask = self.diverted_core_mask.copy()

        self.jtor = super().Jtor_part2(
            R, Z, psi, self.psi_bndry, self.limiter_core_mask
        )
        return self.jtor

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

        alpha, beta = Lao_parameters_finder(
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


class Fiesta_Topeol(freegs.jtor.Fiesta_Topeol):
    """FreeGS profile class with a few modifications, to:
    - retain memory of critical point calculation;
    - deal with limiter plasma configurations

    """

    def __init__(self, eq, limiter=None, *args, **kwargs):
        """Instantiates the object.

        Parameters
        ----------
        eq : freeGS Equilibrium object
            Specifies the domain properties
        limiter : freeGS.machine.Wall object
            Specifies the limiter contour points
            Only set if a limiter different from eq.tokamak.limiter is to be used.

        """
        super().__init__(*args, **kwargs)
        self.profile_parameter = self.Beta0

        if limiter is None:
            self.limiter_handler = eq.limiter_handler

            self.limiter_mask_out = eq.limiter_mask_out
            self.mask_inside_limiter = eq.mask_inside_limiter
            self.limiter_mask_for_plotting = (
                self.mask_inside_limiter + self.limiter_mask_out
            ) > 0
        else:
            self.limiter_handler = limiter_func.Limiter_handler(eq, limiter)

            self.mask_inside_limiter = self.limiter_handler.mask_inside_limiter
            self.limiter_mask_out = self.limiter_handler.make_layer_mask(layer_size=1)
            self.limiter_mask_for_plotting = (
                self.mask_inside_limiter + self.limiter_mask_out
            ) > 0

        if not hasattr(self, "fast"):
            self.Jtor = self._Jtor
        else:
            self.Jtor = self.Jtor_fast

    def assign_profile_parameter(self, Beta0):
        """Assigns to the profile object a new value of the profile parameter paxis"""
        self.Beta0 = Beta0
        self.profile_parameter = Beta0

    def assign_profile_coefficients(self, alpha_m, alpha_n):
        """Assigns to the profile object new value of the coefficients (alpha_m, alpha_n)"""
        self.alpha_m = alpha_m
        self.alpha_n = alpha_n

    def _Jtor(self, R, Z, psi, psi_bndry=None, rel_psi_error=0):
        """Replaces the original FreeGS Jtor method if FreeGSfast is not available."""
        self.jtor = super().Jtor(R, Z, psi, psi_bndry)
        self.opt, self.xpt = critical.find_critical(R, Z, psi)

        self.diverted_core_mask = self.jtor > 0
        self.psi_bndry, mask, self.limiter_flag = (
            self.limiter_handler.core_mask_limiter(
                psi,
                self.xpt[0][2],
                self.diverted_core_mask,
                self.limiter_mask_out,
            )
        )
        self.jtor = super().Jtor(R, Z, psi, self.psi_bndry)
        return self.jtor

    def Jtor_fast(self, R, Z, psi, psi_bndry=None, rel_psi_error=0):
        """Used when FreeGSfast is available."""
        self.diverted_core_mask = super().Jtor_part1(R, Z, psi, psi_bndry)
        if self.diverted_core_mask is None:
            # print('no xpt')
            self.psi_bndry, self.limiter_core_mask, self.flag_limiter = (
                psi_bndry,
                None,
                False,
            )
        elif rel_psi_error < 0.02:
            self.psi_bndry, self.limiter_core_mask, self.flag_limiter = (
                self.limiter_handler.core_mask_limiter(
                    psi,
                    self.psi_bndry,
                    self.diverted_core_mask,
                    self.limiter_mask_out,
                )
            )
        else:
            self.limiter_core_mask = self.diverted_core_mask.copy()

        self.jtor = super().Jtor_part2(
            R, Z, psi, self.psi_bndry, self.limiter_core_mask
        )
        return self.jtor

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

        alpha, beta = Lao_parameters_finder(
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


class Lao85(freegs.jtor.Lao85):
    """FreeGS profile class with a few modifications, to:
    - retain memory of critical point calculation;
    - deal with limiter plasma configurations

    """

    def __init__(self, eq, limiter=None, *args, **kwargs):
        """Instantiates the object.

        Parameters
        ----------
        eq : freeGS Equilibrium object
            Specifies the domain properties
        limiter : freeGS.machine.Wall object
            Specifies the limiter contour points
            Only set if a limiter different from eq.tokamak.limiter is to be used.

        """
        super().__init__(*args, **kwargs)
        self.profile_parameter = [self.alpha, self.beta]

        if limiter is None:
            self.limiter_handler = eq.limiter_handler

            self.limiter_mask_out = eq.limiter_mask_out
            self.mask_inside_limiter = eq.mask_inside_limiter
            self.limiter_mask_for_plotting = (
                self.mask_inside_limiter + self.limiter_mask_out
            ) > 0
        else:
            self.limiter_handler = limiter_func.Limiter_handler(eq, limiter)

            self.mask_inside_limiter = self.limiter_handler.mask_inside_limiter
            self.limiter_mask_out = self.limiter_handler.make_layer_mask(layer_size=1)
            self.limiter_mask_for_plotting = (
                self.mask_inside_limiter + self.limiter_mask_out
            ) > 0

        if not hasattr(self, "fast"):
            self.Jtor = self._Jtor
        else:
            self.Jtor = self.Jtor_fast

    def assign_profile_parameter(self, alpha, beta):
        """Assigns to the profile object a new value of the profile parameter paxis"""
        self.alpha = alpha
        self.beta = beta
        self.profile_parameter = [alpha, beta]

    def _Jtor(self, R, Z, psi, psi_bndry=None, rel_psi_error=0):
        """Replaces the original FreeGS Jtor method if FreeGSfast is not available."""
        self.jtor = super().Jtor(R, Z, psi, psi_bndry)
        self.opt, self.xpt = critical.find_critical(R, Z, psi)

        self.diverted_core_mask = self.jtor > 0
        self.psi_bndry, mask, self.limiter_flag = (
            self.limiter_handler.core_mask_limiter(
                psi,
                self.xpt[0][2],
                self.diverted_core_mask,
                self.limiter_mask_out,
            )
        )
        self.jtor = super().Jtor(R, Z, psi, self.psi_bndry)
        return self.jtor

    def Jtor_fast(self, R, Z, psi, psi_bndry=None, rel_psi_error=0):
        """Used when FreeGSfast is available."""
        self.diverted_core_mask = super().Jtor_part1(R, Z, psi, psi_bndry)
        if self.diverted_core_mask is None:
            # print('no xpt')
            self.psi_bndry, self.limiter_core_mask, self.flag_limiter = (
                psi_bndry,
                None,
                False,
            )
        elif rel_psi_error < 0.02:
            self.psi_bndry, self.limiter_core_mask, self.flag_limiter = (
                self.limiter_handler.core_mask_limiter(
                    psi,
                    self.psi_bndry,
                    self.diverted_core_mask,
                    self.limiter_mask_out,
                )
            )
        else:
            self.limiter_core_mask = self.diverted_core_mask.copy()

        self.jtor = super().Jtor_part2(
            R, Z, psi, self.psi_bndry, self.limiter_core_mask
        )
        return self.jtor

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

        pars = Topeol_opt(
            tp,
            tf,
            x,
            max_it,
            tol,
        )

        return pars
