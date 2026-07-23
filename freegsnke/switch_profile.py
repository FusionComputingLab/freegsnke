"""
Implements some functionality needed by the FreeGSNKE profile object to find optimised coefficients
when switching between different profile parametrisations.

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
    """
    Fit Lao85 profile coefficients to prescribed pprime and ffprime profiles.

    The profiles are represented as polynomial expansions in the normalised
    flux coordinate ``pn_``. The coefficients are obtained via a linear
    least-squares fit.

    Parameters
    ----------
    pn_ : ndarray
        Normalised flux coordinate values in the interval [0, 1].
    pprime_ : ndarray
        Values of pprime evaluated at ``pn_``.
    ffprime_ : ndarray
        Values of ffprime evaluated at ``pn_``.
    n_alpha : int
        Number of free coefficients in the pprime expansion.
    n_beta : int
        Number of free coefficients in the ffprime expansion.
    alpha_logic : bool, optional
        If True, include the Lao85 constraint term so that the fitted
        pprime profile satisfies the boundary condition at ``pn = 1``.
        Default is True.
    beta_logic : bool, optional
        If True, include the Lao85 constraint term so that the fitted
        ffprime profile satisfies the boundary condition at ``pn = 1``.
        Default is True.
    Ip_logic : bool, optional
        If False, rescale the fitted coefficients using the original
        profile amplitudes. Default is True.

    Returns
    -------
    alpha : ndarray
        Fitted coefficients for the pprime profile.
    beta : ndarray
        Fitted coefficients for the ffprime profile.

    Notes
    -----
    The returned coefficients must be used with the same values of
    ``alpha_logic`` and ``beta_logic`` that were used during fitting.

    This routine normalises ``pprime_`` and ``ffprime_`` by their values
    at ``pn_[0]`` before fitting.
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


def Topeol_std(x, alpha_m, alpha_n, beta_0):
    """
    Evaluate the standard Topeol profile.

    The profile is defined as

        T(x) = (1 - x^{alpha_m})^{alpha_n}.

    Parameters
    ----------
    x : float or ndarray
        Independent variable, typically a normalised coordinate in the
        interval [0, 1].
    alpha_m : float
        Inner exponent controlling the profile shape.
    alpha_n : float
        Outer exponent controlling the profile peaking and edge behaviour.
    beta_0 : float
        Unused parameter. Included for consistency with related Topeol
        profile functions and fitting routines.

    Returns
    -------
    float or ndarray
        Value of the standard Topeol profile evaluated at ``x``.

    Notes
    -----
    This function does not currently depend on ``beta_0``.
    """
    return (1 - x**alpha_m) ** alpha_n


def d2Ldb2(t, x, alpha_m, alpha_n, beta_0, Tstd=None):
    """
    Evaluate the second derivative of the loss function with respect to
    beta_0.

    Parameters
    ----------
    t : float or ndarray
        Target values used in the loss function.
    x : float or ndarray
        Independent variable, typically a normalised coordinate in the
        interval [0, 1].
    alpha_m : float
        Inner exponent of the standard Topeol profile.
    alpha_n : float
        Outer exponent of the standard Topeol profile.
    beta_0 : float
        Amplitude parameter of the fitted profile.
    Tstd : float or ndarray, optional
        Precomputed value of ``Topeol_std(x, alpha_m, alpha_n, beta_0)``.
        If not provided, it is evaluated internally.

    Returns
    -------
    float or ndarray
        Second derivative of the loss function with respect to beta_0.

    Notes
    -----
    This function returns

        d²L/dβ₀² = 2 Tstd²,

    where Tstd is the standard Topeol profile. Providing ``Tstd`` avoids
    recomputing the profile when multiple derivatives are evaluated at
    the same point.
    """

    if Tstd is None:
        Tstd = Topeol_std(x, alpha_m, alpha_n, beta_0)
    res = 2 * Tstd**2
    return res


def d2Ldbdn(t, x, alpha_m, alpha_n, beta_0, Tstd=None):
    """
    Evaluate the mixed second derivative of the loss function with respect to
    beta_0 and alpha_n.

    Parameters
    ----------
    t : float or ndarray
        Target values used in the loss function.
    x : float or ndarray
        Independent variable, typically a normalised coordinate in the interval [0, 1].
    alpha_m : float
        Inner exponent of the standard Topeol profile.
    alpha_n : float
        Outer exponent of the standard Topeol profile.
    beta_0 : float
        Amplitude parameter of the fitted profile.
    Tstd : float or ndarray, optional
        Precomputed value of Topeol_std(x, alpha_m, alpha_n, beta_0).
        If not provided, it is evaluated internally.

    Returns
    -------
    float or ndarray
        Mixed second derivative of the loss function with respect to beta_0 and alpha_n.

    Notes
    -----
    This function evaluates

        d²L / (d beta_0 d alpha_n)
        = (2 * beta_0 * Tstd - t) * (2 * Tstd * log(1 - x^alpha_m)),

    where

        Tstd = (1 - x^alpha_m)^alpha_n.

    The logarithm term comes from differentiating the profile with respect to alpha_n.

    Warning
    -------
    This expression becomes singular when x = 1, since log(0) is undefined.
    """

    if Tstd is None:
        Tstd = Topeol_std(x, alpha_m, alpha_n, beta_0)
    res = 2 * beta_0 * Tstd - t
    res *= 2 * Tstd * np.log(Topeol_std(x, alpha_m, 1, beta_0))
    return res


def d2Ldbdm(t, x, alpha_m, alpha_n, beta_0, Tstd=None):
    """
    Evaluate the mixed second derivative of the loss function with respect to
    beta_0 and alpha_m.

    Parameters
    ----------
    t : float or ndarray
        Target values used in the loss function.
    x : float or ndarray
        Independent variable, typically a normalised coordinate in the interval [0, 1].
    alpha_m : float
        Inner exponent of the standard Topeol profile.
    alpha_n : float
        Outer exponent of the standard Topeol profile.
    beta_0 : float
        Amplitude parameter of the fitted profile.
    Tstd : float or ndarray, optional
        Precomputed value of Topeol_std(x, alpha_m, alpha_n, beta_0).
        If not provided, it is evaluated internally.

    Returns
    -------
    float or ndarray
        Mixed second derivative of the loss function with respect to beta_0 and alpha_m.

    Notes
    -----
    This function evaluates the mixed derivative

        d²L / (d beta_0 d alpha_m),

    using the standard Topeol profile

        T(x) = (1 - x^alpha_m)^alpha_n.

    The expression includes logarithmic and power-law sensitivity terms arising
    from differentiation with respect to alpha_m.

    Warning
    -------
    The term log(x) becomes singular at x = 0, so the expression is undefined
    at that point.
    """

    if Tstd is None:
        Tstd = Topeol_std(x, alpha_m, alpha_n, beta_0)
    res = 2 * beta_0 * Tstd - t
    res *= Topeol_std(x, alpha_m, alpha_n - 1, beta_0)
    res *= -2 * alpha_n * np.log(x) * x**alpha_m
    return res


def d2Ldm2(t, x, alpha_m, alpha_n, beta_0, Tstd=None):
    """
    Evaluate the second derivative of the loss function with respect to
    alpha_m.

    Parameters
    ----------
    t : float or ndarray
        Target values used in the loss function.
    x : float or ndarray
        Independent variable, typically a normalised coordinate in the interval [0, 1].
    alpha_m : float
        Inner exponent of the standard Topeol profile.
    alpha_n : float
        Outer exponent of the standard Topeol profile.
    beta_0 : float
        Amplitude parameter of the fitted profile.
    Tstd : float or ndarray, optional
        Precomputed value of Topeol_std(x, alpha_m, alpha_n, beta_0).
        If not provided, it is evaluated internally.

    Returns
    -------
    float or ndarray
        Second derivative of the loss function with respect to alpha_m.

    Notes
    -----
    This function evaluates

        d²L / d alpha_m²,

    using the standard Topeol profile

        T(x) = (1 - x^alpha_m)^alpha_n.

    The expression contains logarithmic terms (log(x)) and power-law factors
    arising from repeated differentiation with respect to alpha_m.

    Warning
    -------
    The function is undefined at x = 0 due to log(x), and may suffer
    numerical instability near x = 0 or x = 1.
    """
    if Tstd is None:
        Tstd = Topeol_std(x, alpha_m, alpha_n, beta_0)
    res = beta_0 * Tstd
    res *= 2 * alpha_n * x**alpha_m - 1
    res += t - alpha_n * t * x**alpha_m
    res *= Topeol_std(x, alpha_m, alpha_n - 2, beta_0)
    res *= 2 * beta_0 * alpha_n * x**alpha_m * np.log(x) ** 2
    return res


def d2Ldn2(t, x, alpha_m, alpha_n, beta_0, Tstd=None):
    """
    Evaluate the second derivative of the loss function with respect to
    alpha_n.

    Parameters
    ----------
    t : float or ndarray
        Target values used in the loss function.
    x : float or ndarray
        Independent variable, typically a normalised coordinate in the interval [0, 1].
    alpha_m : float
        Inner exponent of the standard Topeol profile.
    alpha_n : float
        Outer exponent of the standard Topeol profile.
    beta_0 : float
        Amplitude parameter of the fitted profile.
    Tstd : float or ndarray, optional
        Precomputed value of Topeol_std(x, alpha_m, alpha_n, beta_0).
        If not provided, it is evaluated internally.

    Returns
    -------
    float or ndarray
        Second derivative of the loss function with respect to alpha_n.

    Notes
    -----
    This function evaluates

        d²L / d alpha_n²,

    using the standard Topeol profile

        T(x) = (1 - x^alpha_m)^alpha_n.

    The dependence on alpha_n enters through logarithmic terms of the form
    log(1 - x^alpha_m), which are squared in this second derivative.

    Warning
    -------
    The expression becomes singular when x = 1 since log(0) is undefined.
    """

    if Tstd is None:
        Tstd = Topeol_std(x, alpha_m, alpha_n, beta_0)
    res = 2 * beta_0 * Tstd - t
    res *= np.log(Topeol_std(x, alpha_m, 1, beta_0)) ** 2
    res *= Tstd
    res *= 2 * beta_0
    return res


def d2Ldmdn(t, x, alpha_m, alpha_n, beta_0, Tstd=None):
    """
    Evaluate the mixed second derivative of the loss function with respect to
    alpha_m and alpha_n.

    Parameters
    ----------
    t : float or ndarray
        Target values used in the loss function.
    x : float or ndarray
        Independent variable, typically a normalised coordinate in the interval [0, 1].
    alpha_m : float
        Inner exponent of the standard Topeol profile.
    alpha_n : float
        Outer exponent of the standard Topeol profile.
    beta_0 : float
        Amplitude parameter of the fitted profile.
    Tstd : float or ndarray, optional
        Precomputed value of Topeol_std(x, alpha_m, alpha_n, beta_0).
        If not provided, it is evaluated internally.

    Returns
    -------
    float or ndarray
        Mixed second derivative of the loss function with respect to alpha_m and alpha_n.

    Notes
    -----
    This function evaluates the mixed derivative

        d²L / (d alpha_m d alpha_n),

    for the standard Topeol profile

        T(x) = (1 - x^alpha_m)^alpha_n.

    The expression contains coupled logarithmic sensitivities in both
    alpha_m and alpha_n arising from differentiation of the power-law
    structure of the profile.

    Warning
    -------
    The expression is undefined at x = 0 (log(x)) and may become singular
    at x = 1 due to log(1 - x^alpha_m).
    """

    if Tstd is None:
        Tstd = Topeol_std(x, alpha_m, alpha_n, beta_0)
    res = 2 * beta_0 * Tstd - t
    res *= alpha_n * np.log(Topeol_std(x, alpha_m, 1, beta_0))
    res += beta_0 * Tstd - t
    res *= Topeol_std(x, alpha_m, alpha_n - 1, beta_0)
    res *= -2 * beta_0 * x**alpha_m * np.log(x)
    return res


def dLdn(t, x, alpha_m, alpha_n, beta_0, Tstd=None):
    """
    Evaluate the first derivative of the loss function with respect to
    alpha_n.

    Parameters
    ----------
    t : float or ndarray
        Target values used in the loss function.
    x : float or ndarray
        Independent variable, typically a normalised coordinate in the interval [0, 1].
    alpha_m : float
        Inner exponent of the standard Topeol profile.
    alpha_n : float
        Outer exponent of the standard Topeol profile.
    beta_0 : float
        Amplitude parameter of the fitted profile.
    Tstd : float or ndarray, optional
        Precomputed value of Topeol_std(x, alpha_m, alpha_n, beta_0).
        If not provided, it is evaluated internally.

    Returns
    -------
    float or ndarray
        First derivative of the loss function with respect to alpha_n.

    Notes
    -----
    This function evaluates

        dL / d alpha_n,

    using the standard Topeol profile

        T(x) = (1 - x^alpha_m)^alpha_n.

    The dependence on alpha_n enters through a logarithmic term of the form
    log(1 - x^alpha_m).

    Warning
    -------
    The expression becomes singular when x = 1 since log(0) is undefined.
    """

    if Tstd is None:
        Tstd = Topeol_std(x, alpha_m, alpha_n, beta_0)
    res = t - beta_0 * Tstd
    res *= np.log(Topeol_std(x, alpha_m, 1, beta_0))
    res *= -2 * beta_0 * Tstd
    return res


def dLdm(t, x, alpha_m, alpha_n, beta_0, Tstd=None):
    """
    Evaluate the first derivative of the loss function with respect to
    alpha_m.

    Parameters
    ----------
    t : float or ndarray
        Target values used in the loss function.
    x : float or ndarray
        Independent variable, typically a normalised coordinate in the interval [0, 1].
    alpha_m : float
        Inner exponent of the standard Topeol profile.
    alpha_n : float
        Outer exponent of the standard Topeol profile.
    beta_0 : float
        Amplitude parameter of the fitted profile.
    Tstd : float or ndarray, optional
        Precomputed value of Topeol_std(x, alpha_m, alpha_n, beta_0).
        If not provided, it is evaluated internally.

    Returns
    -------
    float or ndarray
        First derivative of the loss function with respect to alpha_m.

    Notes
    -----
    This function evaluates

        dL / d alpha_m,

    using the standard Topeol profile

        T(x) = (1 - x^alpha_m)^alpha_n.

    The dependence on alpha_m enters through logarithmic and power-law terms
    involving log(x).

    Warning
    -------
    The expression is undefined at x = 0 due to log(x), and may become
    numerically unstable near x = 0 or x = 1.
    """

    if Tstd is None:
        Tstd = Topeol_std(x, alpha_m, alpha_n, beta_0)
    res = t - beta_0 * Tstd
    res *= np.log(x) * Topeol_std(x, alpha_m, alpha_n - 1, beta_0)
    res *= 2 * beta_0 * alpha_n * x**alpha_m
    return res


def dLdb(t, x, alpha_m, alpha_n, beta_0, Tstd=None):
    """
    Evaluate the first derivative of the loss function with respect to
    beta_0.

    Parameters
    ----------
    t : float or ndarray
        Target values used in the loss function.
    x : float or ndarray
        Independent variable, typically a normalised coordinate in the interval [0, 1].
    alpha_m : float
        Inner exponent of the standard Topeol profile.
    alpha_n : float
        Outer exponent of the standard Topeol profile.
    beta_0 : float
        Amplitude parameter of the fitted profile.
    Tstd : float or ndarray, optional
        Precomputed value of Topeol_std(x, alpha_m, alpha_n, beta_0).
        If not provided, it is evaluated internally.

    Returns
    -------
    float or ndarray
        First derivative of the loss function with respect to beta_0.

    Notes
    -----
    This function evaluates

        dL / d beta_0,

    using the standard Topeol profile

        T(x) = (1 - x^alpha_m)^alpha_n.

    The derivative reflects a simple quadratic residual structure in beta_0.

    Warning
    -------
    None.
    """
    if Tstd is None:
        Tstd = Topeol_std(x, alpha_m, alpha_n, beta_0)
    res = t - beta_0 * Tstd
    res *= -2 * Tstd
    return res


def dLdpars(tp, tf, x, alpha_m, alpha_n, beta_0, Tstd=None):
    """
    Compute the gradient of the total loss with respect to the model parameters.

    The total loss is assumed to be the sum of two contributions:
    one associated with ``tp`` and one associated with ``tf``, each depending
    on the same Topeol profile but with different amplitude structure.

    Parameters
    ----------
    tp : float or ndarray
        First target dataset (e.g. pressure-like contribution).
    tf : float or ndarray
        Second target dataset (e.g. flux-function-like contribution).
    x : float or ndarray
        Independent variable, typically a normalised coordinate in the interval [0, 1].
    alpha_m : float
        Inner exponent of the standard Topeol profile.
    alpha_n : float
        Outer exponent of the standard Topeol profile.
    beta_0 : float
        Amplitude parameter of the fitted profile.
    Tstd : float or ndarray, optional
        Precomputed value of Topeol_std(x, alpha_m, alpha_n, beta_0).
        If not provided, it is evaluated internally.

    Returns
    -------
    ndarray, shape (3,)
        Gradient of the total loss with respect to:
        [alpha_m, alpha_n, beta_0].

    Notes
    -----
    The gradient is constructed as the sum of two contributions:

    1. tp contribution:
       - uses standard derivatives dLdm, dLdn, dLdb

    2. tf contribution:
       - uses modified amplitude (1 - beta_0)
       - flips sign of beta_0 derivative term

    The final result is:
        grad L = grad L(tp) + grad L(tf)
    """

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
    """
    Compute the Hessian matrix of the total loss with respect to the model parameters.

    The total loss is assumed to be the sum of two contributions:
    one associated with ``tp`` and one associated with ``tf``, each depending
    on the same Topeol profile but with different amplitude structure.

    Parameters
    ----------
    tp : float or ndarray
        First target dataset (e.g. pressure-like contribution).
    tf : float or ndarray
        Second target dataset (e.g. flux-function-like contribution).
    x : float or ndarray
        Independent variable, typically a normalised coordinate in the interval [0, 1].
    alpha_m : float
        Inner exponent of the standard Topeol profile.
    alpha_n : float
        Outer exponent of the standard Topeol profile.
    beta_0 : float
        Amplitude parameter of the fitted profile.
    Tstd : float or ndarray, optional
        Precomputed value of Topeol_std(x, alpha_m, alpha_n, beta_0).
        If not provided, it is evaluated internally.

    Returns
    -------
    ndarray, shape (3, 3)
        Hessian matrix of the total loss with respect to:
        [alpha_m, alpha_n, beta_0].

    Notes
    -----
    The Hessian is constructed as the sum of two contributions:

    1. tp contribution:
       - uses second derivatives of the loss with respect to all parameter pairs
       - evaluated using d2Ldm2, d2Ldn2, d2Ldb2, d2Ldmdn, d2Ldbdm, d2Ldbdn

    2. tf contribution:
       - uses modified amplitude (1 - beta_0)
       - includes sign flips in mixed derivatives involving beta_0 due to chain rule

    The final Hessian is:
        H = H(tp) + H(tf)
    """

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
    """
    Compute the total loss for the parameterised Topeol model.

    The loss is defined as the sum of two squared-error contributions:
    one for ``tp`` and one for ``tf``, using a shared profile.

    Parameters
    ----------
    tp : float or ndarray
        First target dataset (e.g. pressure-like contribution).
    tf : float or ndarray
        Second target dataset (e.g. flux-function-like contribution).
    x : float or ndarray
        Independent variable, typically a normalised coordinate in the interval [0, 1].
    alpha_m : float
        Inner exponent of the standard Topeol profile.
    alpha_n : float
        Outer exponent of the standard Topeol profile.
    beta_0 : float
        Amplitude parameter controlling the split between the two datasets.
    Tstd : float or ndarray, optional
        Precomputed value of Topeol_std(x, alpha_m, alpha_n, beta_0).
        If not provided, it is evaluated internally.

    Returns
    -------
    float
        Scalar total loss value.

    Notes
    -----
    The loss is

        L = sum( (tp - beta_0 * T(x))^2 + (tf - (1 - beta_0) * T(x))^2 )

    where

        T(x) = (1 - x^alpha_m)^alpha_n.

    The summation is performed over the input grid in ``x``.
    """

    if Tstd is None:
        Tstd = Topeol_std(x, alpha_m, alpha_n, beta_0)
    Lp = (tp - beta_0 * Tstd) ** 2
    Lf = (tf - (1 - beta_0) * Tstd) ** 2
    return np.sum(Lp + Lf, axis=0)


def Topeol_opt_init(tp, tf):
    """
    Compute an initial guess for Topeol model scaling and amplitude split.

    This routine provides a heuristic initialisation for optimisation by
    normalising the input profiles and estimating a mixing parameter ``b0``
    that balances contributions between ``tp`` and ``tf``.

    Parameters
    ----------
    tp : ndarray
        First input profile (e.g. pressure-like quantity).
    tf : ndarray
        Second input profile (e.g. flux-function-like quantity).

    Returns
    -------
    tpn : ndarray
        Normalised and rescaled version of ``tp`` using the estimated split.
    tfn : ndarray
        Normalised and rescaled version of ``tf`` using the estimated split.
    b0 : float
        Estimated initial amplitude split parameter in [0, 1].

    Notes
    -----
    The initial estimate is computed as follows:

    1. A shared normalisation scale is defined using the larger of the
       initial values of ``tp`` and ``tf``.
    2. A mask is applied where ``tf`` is positive to form a stable ratio.
    3. The mean ratio ``tp/tf`` over this region is used to estimate:

           rr = mean(tp/tf)
           b0 = rr / (1 + rr)

    4. The profiles are then rescaled consistently using ``b0`` and their
       initial values.

    Warning
    -------
    This heuristic assumes:
    - ``tp[0]`` and ``tf[0]`` are non-zero
    - There exists a region where ``tf > 0``
    - Ratios are numerically stable in the masked region

    Results may be unstable if these assumptions are violated.
    """

    tpn = tp / max(tp[0], tf[0])
    tfn = tf / max(tp[0], tf[0])

    mask = tfn > 0
    rr = np.mean(tpn[mask] / tfn[mask])
    b0 = rr / (1 + rr)

    tpn = b0 * tp / tp[0]
    tfn = (1 - b0) * tf / tf[0]
    return tpn, tfn, b0


def Topeol_opt_stepper(tp, tf, x, pars):
    """
    Perform one optimisation step for the Topeol parameter fit.

    This routine computes a parameter update using either a Newton-like step
    (based on the Hessian) or a fallback gradient-scaled step when the Hessian
    is not sufficiently well-conditioned.

    Parameters
    ----------
    tp : ndarray
        First target dataset (e.g. pressure-like contribution).
    tf : ndarray
        Second target dataset (e.g. flux-function-like contribution).
    x : ndarray
        Independent variable, typically a normalised coordinate in the interval [0, 1].
    pars : array-like, shape (3,)
        Current parameter vector [alpha_m, alpha_n, beta_0].

    Returns
    -------
    ndarray, shape (3,)
        Updated parameter vector after one optimisation step.

    Notes
    -----
    The update proceeds as follows:

    1. Compute the current profile:
           T(x) = Topeol_std(x, alpha_m, alpha_n, beta_0)

    2. Compute gradient and Hessian:
           grad = dLdpars(...)
           H = d2Ldpars2(...)

    3. Check Hessian conditioning via eigenvalues:
       - If sufficiently many positive eigenvalues are present,
         solve Newton step:
             dpars = -H^{-1} grad
       - Otherwise, fall back to a scaled gradient step:
             dpars = -L * grad / ||grad||

    4. Apply step limiting:
       Prevent excessively large updates by clipping relative changes
       in parameters.

    Warning
    -------
    This is a heuristic optimizer:
    - It is not guaranteed to converge globally.
    - Hessian inversion may be unstable if ill-conditioned.
    - Step limiting is ad hoc and may bias convergence.

    """

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
    """
    Optimise Topeol model parameters using an iterative stepper.

    This function performs a fixed-point style optimisation of the parameters
    [alpha_m, alpha_n, beta_0] using repeated calls to the Topeol_opt_stepper
    until convergence or a maximum number of iterations is reached.

    Parameters
    ----------
    tp : ndarray
        First target dataset (e.g. pressure-like contribution).
    tf : ndarray
        Second target dataset (e.g. flux-function-like contribution).
    x : ndarray
        Independent variable, typically a normalised coordinate in the interval [0, 1].
    max_it : int
        Maximum number of optimisation iterations.
    tol : float
        Convergence tolerance. Iteration stops when all parameter updates
        satisfy |Δpars| ≤ tol.

    Returns
    -------
    ndarray, shape (3,)
        Optimised parameters [alpha_m, alpha_n, beta_0].

    Notes
    -----
    The optimisation procedure is:

    1. Initialise parameters using Topeol_opt_init:
           tpn, tfn, b0 = Topeol_opt_init(tp, tf)

    2. Set initial guess:
           pars = [2, 1, b0]

    3. Iterate:
           pars_{k+1} = Topeol_opt_stepper(tpn, tfn, x, pars_k)

    4. Stop when:
           max|pars_{k+1} - pars_k| ≤ tol
       or when max_it is reached.

    Warning
    -------
    - Convergence is not guaranteed.
    - The algorithm depends strongly on the quality of the initial guess.
    - If max_it is reached, the returned parameters may not be optimal.
    """

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
