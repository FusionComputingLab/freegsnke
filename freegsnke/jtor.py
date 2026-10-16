"""
Classes representing different toroidal current density profiles
in the plasma.
 
Modified substantially from the original FreeGS code.

Copyright 2024 Nicola C. Amorisco, Adriano Agnello, George K. Holt, Ben Dudson.

FreeGSNKE is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FreeGSNKE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with FreeGSNKE.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
from numpy import clip, pi, reshape, sqrt, zeros
from scipy.integrate import quad, romb
from scipy.interpolate import CubicSpline, UnivariateSpline
from scipy.special import beta as spbeta
from scipy.special import betainc as spbinc

from . import critical
from .gradshafranov import mu0


class Profile(object):
    """
    Base class from which profiles classes can inherit.
    """

    def Jtor(self, R, Z, psi, psi_bndry=None):
        """
        Calculate the toroidal current density on the computational grid.

        This convenience wrapper combines the generic critical-point and
        core-mask identification in :meth:`Jtor_part1` with the profile-specific
        current-density calculation implemented by each subclass in
        :meth:`Jtor_part2`.

        Parameters
        ----------
        R : np.ndarray
            Radial coordinates of the grid points.
        Z : np.ndarray
            Vertical coordinates of the grid points.
        psi : np.ndarray
            Total poloidal field flux at each grid point [Webers/2pi].
        psi_bndry : float, optional
            Value of the poloidal field flux at the plasma boundary.

        Returns
        -------
        np.array
            Toroidal current density on the computational grid [A/m^2].
        """

        if not hasattr(self, "mask_inside_limiter"):
            self.mask_inside_limiter = np.ones_like(psi, dtype=bool)
        if not hasattr(self, "mask_outside_limiter"):
            self.mask_outside_limiter = np.zeros_like(psi, dtype=float)

        opt, xpt, core_mask, psi_bndry = self.Jtor_part1(
            R,
            Z,
            psi,
            psi_bndry=psi_bndry,
            mask_outside_limiter=self.mask_outside_limiter,
        )
        if core_mask is None:
            core_mask = np.ones_like(psi, dtype=bool)
        uses_xpoint_boundary = len(xpt) > 0 and np.isclose(psi_bndry, xpt[0][2])
        self.opt = opt
        self.xpt = xpt
        self.psi_axis = opt[0][2]
        self.psi_bndry = psi_bndry
        self.diverted_core_mask = core_mask
        self.limiter_core_mask = core_mask
        self.flag_limiter = not uses_xpoint_boundary
        return self.Jtor_part2(R, Z, psi, opt[0][2], psi_bndry, core_mask)

    def pressure(self, psinorm):
        """
        Calculate the 1D pressure profile in the plasma (vs. the normalised
        poloidal flux) by integrating the pprime profile.

        Parameters
        ----------
        psinorm : np.array
            Array of normalised psi values between 0 and 1.

        Returns
        -------
        np.array
            The computed pressure values at each normalised psi value.
        """

        # to store psi and integral values
        pvals = reshape(psinorm, -1)
        ovals = reshape(zeros(psinorm.shape), -1)

        if len(pvals) != len(ovals):
            raise ValueError("Input and output arrays of different lengths")

        for i in range(len(pvals)):
            # integrate
            val, _ = quad(self.pprime, pvals[i], 1.0)

            # convert from integral in normalised psi to integral in psi
            val *= self.psi_axis - self.psi_bndry
            ovals[i] = val

        # pressure at the edge is zero so no need for constant of integration

        return reshape(ovals, psinorm.shape)

    def fpol(self, psinorm):
        """
        Calculate the 1D toroidal magnetic profile in the plasma (vs. the normalised
        poloidal flux) by integrating the ffprime profile (ffprime = 0.5*d/dpsi(f^2)).

        Parameters
        ----------
        psinorm : np.array
            Array of normalised psi values between 0 and 1.

        Returns
        -------
        np.array
            The computed F values at each normalised psi value.
        """

        # if a single value
        if not hasattr(psinorm, "__len__"):
            # integrate
            val, _ = quad(self.ffprime, psinorm, 1.0)
            # convert from integral in normalised psi to integral in psi
            val *= self.psi_axis - self.psi_bndry

            # ffprime = 0.5*d/dpsi(f^2)
            # apply boundary condition at psinorm=1 val = fvac**2
            return sqrt(2.0 * val + self.fvac() ** 2)

        # to store psi and integral values
        pvals = reshape(psinorm, -1)
        ovals = reshape(zeros(psinorm.shape), -1)

        if len(pvals) != len(ovals):
            raise ValueError("Input and output arrays of different lengths")
        for i in range(len(pvals)):
            # integrate
            val, _ = quad(self.ffprime, pvals[i], 1.0)

            # convert from integral in normalised psi to integral in psi
            val *= self.psi_axis - self.psi_bndry

            # ffprime = 0.5*d/dpsi(f^2)
            # apply boundary condition at psinorm=1 val = fvac**2
            ovals[i] = sqrt(2.0 * val + self.fvac() ** 2)

        return reshape(ovals, psinorm.shape)

    def Jtor_part1(self, R, Z, psi, psi_bndry=None, mask_outside_limiter=None):
        """
        First part of the Jtor calculation that identifies critical points in the
        poloidal flux (generic to all profile functions). The second stage (in
        specific classes below) will be used for calculating Jtor using their
        explicit profile parameterisations.

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

        # find O- and X-points of equilibrium
        opt, xpt = critical.find_critical(R, Z, psi, self.mask_inside_limiter, self.Ip)

        # flux on the magnetic axis (needed by fpol, pressure, and q, also without X-points)
        self.psi_axis = opt[0][2]

        # find core plasma mask (using user-defined psi_bndry)
        if psi_bndry is not None:
            diverted_core_mask = critical.inside_mask(
                R, Z, psi, opt, xpt, mask_outside_limiter, psi_bndry
            )
        elif len(xpt) > 0:
            # find core plasma mask using psi_bndry from xpt
            psi_bndry = xpt[0][2]
            # check correct sorting between psi_axis and psi_bndry
            if (self.psi_axis - psi_bndry) * self.Ip < 0:
                raise ValueError(
                    "Incorrect critical points! Likely due to unsuitable 'psi_plasma' guess."
                )
            diverted_core_mask = critical.inside_mask(
                R, Z, psi, opt, xpt, mask_outside_limiter, psi_bndry
            )
            # check of any abrupt change of size in the diverted core mask
            if hasattr(self, "diverted_core_mask"):
                if self.diverted_core_mask is not None:
                    previous_core_size = np.sum(self.diverted_core_mask)
                    # check size change
                    check = np.sum(diverted_core_mask) / previous_core_size < 0.5
                    candidate_index = 1
                    while check and candidate_index < len(xpt):
                        # try using the next xpt as primary xpt
                        candidate_xpt = xpt[candidate_index:]
                        alt_diverted_core_mask = critical.inside_mask(
                            R,
                            Z,
                            psi,
                            opt,
                            candidate_xpt,
                            mask_outside_limiter,
                            candidate_xpt[0, 2],
                        )
                        # check the alternative Xpoint gives rise to a valid core
                        edge_pixels = np.sum(self.edge_mask * alt_diverted_core_mask)
                        if edge_pixels == 0:
                            # the candidate is valid
                            xpt = candidate_xpt
                            psi_bndry = xpt[0, 2]
                            diverted_core_mask = alt_diverted_core_mask.copy()

                            # check if there could be better candidates
                            check = (
                                np.sum(diverted_core_mask) / previous_core_size < 0.5
                            )
                            candidate_index = 1
                        else:
                            candidate_index += 1
        else:
            # No X-points
            psi_bndry = psi[0, 0]
            diverted_core_mask = None

        return opt, xpt, diverted_core_mask, psi_bndry


class ConstrainBetapIp(Profile):
    """
    Class describing the toroidal current density profile formulation defined in:
    Y. M. Jeon, 2015, "Development of a free-boundary tokamak equilibrium solver
    for advanced study of tokamak equilibria".
    (https://link.springer.com/article/10.3938/jkps.67.843)

    This formulation constrains profiles using the poloidal beta and plasma
    current.
    """

    def __init__(self, betap, Ip, fvac, alpha_m=1.0, alpha_n=2.0, Raxis=1.0):
        """
        Initialise the class.

        Parameters
        ----------
        betap : float
            Poloidal beta.
        Ip : float
            Total plasma current [A].
        fvac : float
            Vacuum toroidal field strength (f = R*B_tor) [T].
        alpha_m : float
            Shape/peakedness parameter (non-negative).
        alpha_n : float
            Shape/peakedness parameter (non-negative).
        Raxis : float
            Radial scaling parameter (non-negative).
        """

        # set parameters
        self.betap = betap
        self.Ip = Ip
        self._fvac = fvac

        if alpha_m < 0 or alpha_n < 0:
            raise ValueError("alpha_m and alpha_n must be positive.")
        self.alpha_m = alpha_m
        self.alpha_n = alpha_n

        if Raxis < 0:
            raise ValueError("Raxis must be positive.")
        self.Raxis = Raxis

        # Compatibility flag for native current-density profiles.
        self.fast = True

    def Jtor_part2(self, R, Z, psi, psi_axis, psi_bndry, mask):
        """
        Second part of the calculation that will use the explicit
        parameterisation of the chosen profile function to calculate Jtor.

        This is given by:
            Jtor(ψ, R, Z) = L * [ (beta0 * R / Raxis) + ((1 - Beta0) * Raxis / R) ] * (1 - ψ^alpha_m)^alpha_n.

        Parameters
        ----------
        R : np.ndarray
            Radial coordinates of the grid points.
        Z : np.ndarray
            Vertical coordinates of the grid points.
        psi : np.ndarray
            Total poloidal field flux at each grid point [Webers/2pi].
        psi_axis : float
            Value of the poloidal field flux at the magnetic axis of the plasma.
        psi_bndry : float
            Value of the poloidal field flux at the boundary of the plasma (last closed
            flux surface).
        mask : np.ndarray
            Mask of points inside the last closed flux surface.

        Returns
        -------
        np.array
            Toroidal current density on the computational grid [A/m^2].
        """

        # set flux on boundary
        if psi_bndry is None:
            psi_bndry = psi[0, 0]
        self.psi_bndry = psi_bndry

        # grid sizes
        dR = R[1, 0] - R[0, 0]
        dZ = Z[0, 1] - Z[0, 0]

        # calculate normalised psi
        psi_norm = (psi - psi_axis) / (psi_bndry - psi_axis)

        # shape function
        jtorshape = (1.0 - np.clip(psi_norm, 0.0, 1.0) ** self.alpha_m) ** self.alpha_n

        # if there is a masking function, use it
        if mask is not None:
            jtorshape *= mask

        # now apply constraints to define constants
        shapeintegral0 = spbeta(1.0 / self.alpha_m, 1.0 + self.alpha_n) / self.alpha_m

        # pressure function
        pfunc = (
            shapeintegral0
            * (psi_bndry - psi_axis)
            * (
                1.0
                - spbinc(
                    1.0 / self.alpha_m,
                    1.0 + self.alpha_n,
                    np.clip(psi_norm, 0.0, 1.0) ** self.alpha_m,
                )
            )
        )

        # mask with core plasma
        if mask is not None:
            pfunc *= mask

        # integrate over plasma
        intp = np.sum(pfunc) * dR * dZ  # romb(romb(pfunc)) * dR * dZ

        # find Lbeta0 scaling
        LBeta0 = -self.betap * (mu0 / (8.0 * pi)) * self.Raxis * self.Ip**2 / intp

        # integrate current density components
        IR = (
            np.sum(jtorshape * R / self.Raxis) * dR * dZ
        )  # romb(romb(jtorshape * R / self.Raxis)) * dR * dZ
        if IR == 0:
            raise ValueError("No core mask!")
        I_R = (
            np.sum(jtorshape * self.Raxis / R) * dR * dZ
        )  # romb(romb(jtorshape * self.Raxis / R)) * dR * dZ

        # find L scaling parameter and scaled beta
        L = self.Ip / I_R - LBeta0 * (IR / I_R - 1)
        Beta0 = LBeta0 / L

        # calculate final toroidal current density
        Jtor = L * (Beta0 * R / self.Raxis + (1 - Beta0) * self.Raxis / R) * jtorshape

        # store parameters
        self.jtor = Jtor
        self.L = L
        self.Beta0 = Beta0

        return Jtor

    def pprime(self, pn):
        """
        Calculate the 1D pprime (dp/dpsi) profile in the plasma (vs. the
        normalised poloidal flux).

        Parameters
        ----------
        pn : np.array
            Array of normalised psi values between 0 and 1.

        Returns
        -------
        np.array
            The computed pprime values at each normalised psi value.
        """

        shape = (1.0 - np.clip(pn, 0.0, 1.0) ** self.alpha_m) ** self.alpha_n
        if hasattr(self, "Beta0") is False:
            raise ValueError(
                "The pprime profile cannot be normalised. "
                "Please first calculate Jtor for this profile. "
            )

        return self.L * self.Beta0 / self.Raxis * shape

    def ffprime(self, pn):
        """
        Calculate the 1D ffprime (FdF/dpsi) profile in the plasma (vs. the
        normalised poloidal flux).

        Parameters
        ----------
        pn : np.array
            Array of normalised psi values between 0 and 1.

        Returns
        -------
        np.array
            The computed ffprime values at each normalised psi value.
        """

        shape = (1.0 - np.clip(pn, 0.0, 1.0) ** self.alpha_m) ** self.alpha_n
        if hasattr(self, "Beta0") is False:
            raise ValueError(
                "The ffprime profile cannot be normalised. "
                "Please first calculate Jtor for this profile. "
            )

        return mu0 * self.L * (1 - self.Beta0) * self.Raxis * shape

    def fvac(self):
        """
        Return the vaccum field parameter fvac = R*Btor.

        Parameters
        ----------

        Returns
        -------
        float
            The vaccum field parameter fvac = R*Btor.
        """
        return self._fvac


class ConstrainPaxisIp(Profile):
    """
    Class describing the toroidal current density profile formulation defined in:
    Y. M. Jeon, 2015, "Development of a free-boundary tokamak equilibrium solver
    for advanced study of tokamak equilibria".
    (https://link.springer.com/article/10.3938/jkps.67.843)

    This formulation constrains profiles using the pressure on axis and plasma
    current.
    """

    def __init__(self, paxis, Ip, fvac, alpha_m=1.0, alpha_n=2.0, Raxis=1.0):
        """
        Initialise the class.

        Parameters
        ----------
        paxis : float
            Pressure on the magnetic axis [Pa].
        Ip : float
            Total plasma current [A].
        fvac : float
            Vacuum toroidal field strength (f = R*B_tor) [T].
        alpha_m : float
            Shape/peakedness parameter (non-negative).
        alpha_n : float
            Shape/peakedness parameter (non-negative).
        Raxis : float
            Radial scaling parameter (non-negative).
        """

        # set parameters
        self.paxis = paxis
        self.Ip = Ip
        self._fvac = fvac

        if alpha_m < 0 or alpha_n < 0:
            raise ValueError("alpha_m and alpha_n must be positive.")
        self.alpha_m = alpha_m
        self.alpha_n = alpha_n

        if Raxis < 0:
            raise ValueError("Raxis must be positive.")
        self.Raxis = Raxis

        # Compatibility flag for native current-density profiles.
        self.fast = True

    def Jtor_part2(self, R, Z, psi, psi_axis, psi_bndry, mask):
        """
        Second part of the calculation that will use the explicit
        parameterisation of the chosen profile function to calculate Jtor.

        This is given by:
            Jtor(ψ, R, Z) = L * [ (beta0 * R / Raxis) + ((1 - Beta0) * Raxis / R) ] * (1 - ψ^alpha_m)^alpha_n.

        Parameters
        ----------
        R : np.ndarray
            Radial coordinates of the grid points.
        Z : np.ndarray
            Vertical coordinates of the grid points.
        psi : np.ndarray
            Total poloidal field flux at each grid point [Webers/2pi].
        psi_axis : float
            Value of the poloidal field flux at the magnetic axis of the plasma.
        psi_bndry : float
            Value of the poloidal field flux at the boundary of the plasma (last closed
            flux surface).
        mask : np.ndarray
            Mask of points inside the last closed flux surface.

        Returns
        -------
        np.array
            Toroidal current density on the computational grid [A/m^2].
        """

        # set flux on boundary
        if psi_bndry is None:
            psi_bndry = psi[0, 0]
        self.psi_bndry = psi_bndry

        # grid sizes
        dR = R[1, 0] - R[0, 0]
        dZ = Z[0, 1] - Z[0, 0]

        # calculate normalised psi
        psi_norm = (psi - psi_axis) / (psi_bndry - psi_axis)

        # shape function
        jtorshape = (1.0 - np.clip(psi_norm, 0.0, 1.0) ** self.alpha_m) ** self.alpha_n

        # if there is a masking function, use it
        if mask is not None:
            jtorshape *= mask

        # now apply constraints to define constants
        shapeintegral = spbeta(1.0 / self.alpha_m, 1.0 + self.alpha_n) / self.alpha_m
        shapeintegral *= psi_bndry - psi_axis

        # integrate current density components
        IR = (
            np.sum(jtorshape * R / self.Raxis) * dR * dZ
        )  # romb(romb(jtorshape * R / self.Raxis)) * dR * dZ
        I_R = (
            np.sum(jtorshape * self.Raxis / R) * dR * dZ
        )  # romb(romb(jtorshape * self.Raxis / R)) * dR * dZ

        # find L scaling parameter and scaled beta
        LBeta0 = -self.paxis * self.Raxis / shapeintegral
        L = self.Ip / I_R - LBeta0 * (IR / I_R - 1)
        Beta0 = LBeta0 / L

        # calculate final toroidal current density
        Jtor = L * (Beta0 * R / self.Raxis + (1 - Beta0) * self.Raxis / R) * jtorshape

        # store parameters
        self.jtor = Jtor
        self.L = L
        self.Beta0 = Beta0

        return Jtor

    def pprime(self, pn):
        """
        Calculate the 1D pprime (dp/dpsi) profile in the plasma (vs. the
        normalised poloidal flux).

        Parameters
        ----------
        pn : np.array
            Array of normalised psi values between 0 and 1.

        Returns
        -------
        np.array
            The computed pprime values at each normalised psi value.
        """

        shape = (1.0 - np.clip(pn, 0.0, 1.0) ** self.alpha_m) ** self.alpha_n
        if hasattr(self, "Beta0") is False:
            raise ValueError(
                "The pprime profile cannot be normalised. "
                "Please first calculate Jtor for this profile. "
            )
            # self.L = 1
            # print(
            #     "This is using self.L=1, which is likely not appropriate. Please calculate Jtor first to ensure the correct normalization."
            # )
        return self.L * self.Beta0 / self.Raxis * shape

    def ffprime(self, pn):
        """
        Calculate the 1D ffprime (FdF/dpsi) profile in the plasma (vs. the
        normalised poloidal flux).

        Parameters
        ----------
        pn : np.array
            Array of normalised psi values between 0 and 1.

        Returns
        -------
        np.array
            The computed ffprime values at each normalised psi value.
        """

        shape = (1.0 - np.clip(pn, 0.0, 1.0) ** self.alpha_m) ** self.alpha_n
        if hasattr(self, "Beta0") is False:
            raise ValueError(
                "The ffprime profile cannot be normalised. "
                "Please first calculate Jtor for this profile. "
            )
        return mu0 * self.L * (1 - self.Beta0) * self.Raxis * shape

    def fvac(self):
        """
        Return the vaccum field parameter fvac = R*Btor.

        Parameters
        ----------

        Returns
        -------
        float
            The vaccum field parameter fvac = R*Btor.
        """
        return self._fvac


class Fiesta_Topeol(Profile):
    """
    Class describing the toroidal current density profile formulation defined in:
    Y. M. Jeon, 2015, "Development of a free-boundary tokamak equilibrium solver
    for advanced study of tokamak equilibria".
    (https://link.springer.com/article/10.3938/jkps.67.843)

    In this class, Beta0 is specified directly.

    """

    def __init__(self, Beta0, Ip, fvac, alpha_m=1.0, alpha_n=2.0, Raxis=1.0):
        """
        Initialise the class.

        Parameters
        ----------
        Beta0 : float
            Scaling parameter.
        Ip : float
            Total plasma current [A].
        fvac : float
            Vacuum toroidal field strength (f = R*B_tor) [T].
        alpha_m : float
            Shape/peakedness parameter (non-negative).
        alpha_n : float
            Shape/peakedness parameter (non-negative).
        Raxis : float
            Radial scaling parameter (non-negative).
        """

        # set parameters
        self.Beta0 = Beta0
        self.Ip = Ip
        self._fvac = fvac

        if alpha_m < 0 or alpha_n < 0:
            raise ValueError("alpha_m and alpha_n must be positive.")
        self.alpha_m = alpha_m
        self.alpha_n = alpha_n

        if Raxis < 0:
            raise ValueError("Raxis must be positive.")
        self.Raxis = Raxis

        # Compatibility flag for native current-density profiles.
        self.fast = True

    def Jtor_part2(self, R, Z, psi, psi_axis, psi_bndry, mask):
        """
        Second part of the calculation that will use the explicit
        parameterisation of the chosen profile function to calculate Jtor.

        This is given by:
            Jtor(ψ, R, Z) = L * [ (beta0 * R / Raxis) + ((1 - Beta0) * Raxis / R) ] * (1 - ψ^alpha_m)^alpha_n.

        Parameters
        ----------
        R : np.ndarray
            Radial coordinates of the grid points.
        Z : np.ndarray
            Vertical coordinates of the grid points.
        psi : np.ndarray
            Total poloidal field flux at each grid point [Webers/2pi].
        psi_axis : float
            Value of the poloidal field flux at the magnetic axis of the plasma.
        psi_bndry : float
            Value of the poloidal field flux at the boundary of the plasma (last closed
            flux surface).
        mask : np.ndarray
            Mask of points inside the last closed flux surface.

        Returns
        -------
        np.array
            Toroidal current density on the computational grid [A/m^2].
        """

        # set flux on boundary
        if psi_bndry is None:
            psi_bndry = psi[0, 0]
        self.psi_bndry = psi_bndry

        # grid sizes
        dR = R[1, 0] - R[0, 0]
        dZ = Z[0, 1] - Z[0, 0]

        # calculate normalised psi
        psi_norm = (psi - psi_axis) / (psi_bndry - psi_axis)

        # shape function
        jtorshape = (1.0 - np.clip(psi_norm, 0.0, 1.0) ** self.alpha_m) ** self.alpha_n

        # if there is a masking function, use it
        if mask is not None:
            jtorshape *= mask

        # calculate final toroidal current density
        Jtor = (
            self.Beta0 * R / self.Raxis + (1 - self.Beta0) * self.Raxis / R
        ) * jtorshape
        L = self.Ip / (np.sum(Jtor) * dR * dZ)

        # store parameters
        self.jtor = L * Jtor
        self.L = L

        return self.jtor

    def pprime(self, pn):
        """
        Calculate the 1D pprime (dp/dpsi) profile in the plasma (vs. the
        normalised poloidal flux).

        Parameters
        ----------
        pn : np.array
            Array of normalised psi values between 0 and 1.

        Returns
        -------
        np.array
            The computed pprime values at each normalised psi value.
        """

        shape = (1.0 - np.clip(pn, 0.0, 1.0) ** self.alpha_m) ** self.alpha_n
        if hasattr(self, "Beta0") is False:
            raise ValueError(
                "The pprime profile cannot be normalised. "
                "Please first calculate Jtor for this profile. "
            )
        return self.L * self.Beta0 / self.Raxis * shape

    def ffprime(self, pn):
        """
        Calculate the 1D ffprime (FdF/dpsi) profile in the plasma (vs. the
        normalised poloidal flux).

        Parameters
        ----------
        pn : np.array
            Array of normalised psi values between 0 and 1.

        Returns
        -------
        np.array
            The computed ffprime values at each normalised psi value.
        """

        shape = (1.0 - np.clip(pn, 0.0, 1.0) ** self.alpha_m) ** self.alpha_n
        if hasattr(self, "Beta0") is False:
            raise ValueError(
                "The ffprime profile cannot be normalised. "
                "Please first calculate Jtor for this profile. "
            )
        return mu0 * self.L * (1 - self.Beta0) * self.Raxis * shape

    def fvac(self):
        """
        Return the vaccum field parameter fvac = R*Btor.

        Parameters
        ----------

        Returns
        -------
        float
            The vaccum field parameter fvac = R*Btor.
        """
        return self._fvac


class Lao85(Profile):
    """
    Class describing the toroidal current density profile formulation defined in:
    Lao et al, 1985, "Reconstruction of current profile parameters and plasma
    shapes in tokamaks".
    (https://dx.doi.org/10.1088/0029-5515/25/11/007)

    See equations 2, 4, and 5.
    """

    def __init__(
        self,
        Ip,
        fvac,
        alpha,
        beta,
        alpha_logic=True,
        beta_logic=True,
        Raxis=1,
        Ip_logic=True,
    ):
        """
        Initialise the class.

        Parameters
        ----------
        Ip : float
            Total plasma current [A].
        fvac : float
            Vacuum toroidal field strength (f = R*B_tor) [T].
        alpha : list or np.array
            Polynomial coefficients for pprime.
        beta : list or np.array
            Polynomial coefficients for ffprime.
        alpha_logic : bool
            If True, include the n_{P+1} term to ensure pprime(1)=0 (see Jtor2).
        beta_logic : bool
            If True, include the n_{F+1} term to ensure ffprime(1)=0 (see Jtor2).
        Raxis : float
            Radial scaling parameter (non-negative).
        Ip_logic : bool
            If True, entire profile is re-normalised to satisfy Ip identically.
        """

        # set parameters for later use
        self.Ip = Ip
        self._fvac = fvac
        self.alpha = np.array(alpha)
        self.alpha_logic = alpha_logic
        self.beta = np.array(beta)
        self.beta_logic = beta_logic
        self.Raxis = Raxis
        self.Ip_logic = Ip_logic
        if self.Ip_logic is False:
            self.L = 1  # no normalisation in this case

        # initialise profile ready for calculation
        self.initialize_profile()

        # Compatibility flag for native current-density profiles.
        self.fast = True

    def initialize_profile(self):
        """
        Set up the attributes in the class ready for Jtor calculations.

        Parameters
        ----------

        Returns
        -------

        """

        # set up the required exponents in the Lao profile
        if self.alpha_logic:
            self.alpha = np.concatenate((self.alpha, [-np.sum(self.alpha)]))
        self.alpha_exp = np.arange(0, len(self.alpha))
        if self.beta_logic:
            self.beta = np.concatenate((self.beta, [-np.sum(self.beta)]))
        self.beta_exp = np.arange(0, len(self.beta))

    def build_dJtorpsin1(
        self,
    ):
        # calculate dJ/dpsi_n at psi_n=1
        pprime_term = (
            self.alpha[1:, np.newaxis, np.newaxis]
            * self.alpha_exp[1:, np.newaxis, np.newaxis]
        )
        pprime_term = np.sum(pprime_term, axis=0)
        pprime_term *= self.eqR / self.Raxis

        ffprime_term = (
            self.beta[1:, np.newaxis, np.newaxis]
            * self.beta_exp[1:, np.newaxis, np.newaxis]
        )
        ffprime_term = np.sum(ffprime_term, axis=0)
        ffprime_term *= self.Raxis / self.eqR
        ffprime_term /= mu0

        self.dJtorpsin1 = pprime_term + ffprime_term

    def Jtor_part2(
        self,
        R,
        Z,
        psi,
        psi_axis,
        psi_bndry,
        mask,
        torefine=False,
        refineR=None,
    ):
        """
        Second part of the calculation that will use the explicit
        parameterisation of the chosen profile function to calculate Jtor.

        This is given by:
            Jtor(ψ, R, Z) = L * [ (R / Raxis) * p'(ψ) + (Raxis / R) * FF'(ψ) / mu0 ],

        where
            p'(ψ) = sum(alpha_i ψ^i) - sum(alpha_i) ψ^(n_P+1),

            FF'(ψ) = sum(beta_i ψ^i) - sum(beta_i) ψ^(n_F+1),

        and note ψ here is the normalised psi.

        Parameters
        ----------
        R : np.ndarray
            Radial coordinates of the grid points.
        Z : np.ndarray
            Vertical coordinates of the grid points.
        psi : np.ndarray
            Total poloidal field flux at each grid point [Webers/2pi].
        psi_axis : float
            Value of the poloidal field flux at the magnetic axis of the plasma.
        psi_bndry : float
            Value of the poloidal field flux at the boundary of the plasma (last closed
            flux surface).
        mask : np.ndarray
            Mask of points inside the last closed flux surface.


        Returns
        -------
        np.array
            Toroidal current density on the computational grid [A/m^2].
        """

        # set flux on boundary
        if psi_bndry is None:
            psi_bndry = psi[0, 0]
        self.psi_bndry = psi_bndry

        if torefine:
            R = 1.0 * refineR

        # calculate normalised psi
        psi_norm = (psi - psi_axis) / (psi_bndry - psi_axis)
        psi_norm = np.clip(psi_norm, 0.0, 1.0)

        # calculate the p' and FF' profiles
        pprime_term = (
            psi_norm[np.newaxis, :, :] ** self.alpha_exp[:, np.newaxis, np.newaxis]
        )
        pprime_term *= self.alpha[:, np.newaxis, np.newaxis]
        pprime_term = np.sum(pprime_term, axis=0)
        pprime_term *= R / self.Raxis

        ffprime_term = (
            psi_norm[np.newaxis, :, :] ** self.beta_exp[:, np.newaxis, np.newaxis]
        )
        ffprime_term *= self.beta[:, np.newaxis, np.newaxis]
        ffprime_term = np.sum(ffprime_term, axis=0)
        ffprime_term *= self.Raxis / R
        ffprime_term /= mu0

        # sum together
        Jtor = pprime_term + ffprime_term

        # calculate dJ/dpsi_n
        pprime_term = (
            psi_norm[np.newaxis, :, :] ** self.alpha_exp[:-1, np.newaxis, np.newaxis]
        )
        pprime_term *= (
            self.alpha[1:, np.newaxis, np.newaxis]
            * self.alpha_exp[1:, np.newaxis, np.newaxis]
        )
        pprime_term = np.sum(pprime_term, axis=0)
        pprime_term *= R / self.Raxis

        ffprime_term = (
            psi_norm[np.newaxis, :, :] ** self.beta_exp[:-1, np.newaxis, np.newaxis]
        )
        ffprime_term *= (
            self.beta[1:, np.newaxis, np.newaxis]
            * self.beta_exp[1:, np.newaxis, np.newaxis]
        )
        ffprime_term = np.sum(ffprime_term, axis=0)
        ffprime_term *= self.Raxis / R
        ffprime_term /= mu0

        dJtordpsin = pprime_term + ffprime_term
        self.dJtordpsi = dJtordpsin / (psi_axis - psi_bndry)

        # put to zero all current outside the LCFS
        # Jtor *= psi > psi_bndry

        # Jtor *= self.Ip * Jtor > 0

        if torefine:
            return Jtor

        if mask is not None:
            Jtor *= mask
            self.dJtordpsi *= mask

        # if Ip normalisation is required, do it
        if self.Ip_logic:
            # grid sizes
            dR = R[1, 0] - R[0, 0]
            dZ = Z[0, 1] - Z[0, 0]

            jtorIp = np.sum(Jtor)
            if jtorIp == 0:
                self.problem_psi = psi
                raise ValueError("Total plasma current is zero! Cannot renormalise.")
            L = self.Ip / (jtorIp * dR * dZ)
            Jtor = L * Jtor
            self.dJtordpsi *= L

        else:
            L = 1.0

        # store parameters
        self.jtor = Jtor.copy()
        self.L = L

        return self.jtor

    def pprime(self, pn):
        """
        Calculate the 1D pprime (dp/dpsi) profile in the plasma (vs. the
        normalised poloidal flux).

        Parameters
        ----------
        pn : np.array
            Array of normalised psi values between 0 and 1.

        Returns
        -------
        np.array
            The computed pprime values at each normalised psi value.
        """

        pn_ = np.clip(np.array(pn), 0, 1)
        shape_pn = np.shape(pn_)

        shape = pn_[np.newaxis] ** self.alpha_exp.reshape(
            list(np.shape(self.alpha_exp)) + [1] * len(shape_pn)
        )
        shape *= self.alpha.reshape(list(np.shape(self.alpha)) + [1] * len(shape_pn))
        shape = np.sum(shape, axis=0)
        if self.Ip_logic is True:
            if hasattr(self, "L") is False:
                raise ValueError(
                    "The pprime profile cannot be normalised. "
                    "Please first calculate Jtor for this profile. "
                )
        return self.L * shape / self.Raxis

    def ffprime(self, pn):
        """
        Calculate the 1D ffprime (FdF/dpsi) profile in the plasma (vs. the
        normalised poloidal flux).

        Parameters
        ----------
        pn : np.array
            Array of normalised psi values between 0 and 1.

        Returns
        -------
        np.array
            The computed ffprime values at each normalised psi value.
        """

        pn_ = np.clip(np.array(pn), 0, 1)
        shape_pn = np.shape(pn_)

        shape = pn_[np.newaxis] ** self.beta_exp.reshape(
            list(np.shape(self.beta_exp)) + [1] * len(shape_pn)
        )
        shape *= self.beta.reshape(list(np.shape(self.beta)) + [1] * len(shape_pn))
        shape = np.sum(shape, axis=0)
        if self.Ip_logic is True:
            if hasattr(self, "L") is False:
                raise ValueError(
                    "The ffprime profile cannot be normalised. "
                    "Please first calculate Jtor for this profile. "
                )
        return self.L * shape * self.Raxis

    def pressure(self, pn):
        """
        Calculate the 1D pressure profile in the plasma (vs. the normalised
        poloidal flux) without integrating the pprime profile (specifically
        for the Lao profile).

        Parameters
        ----------
        pn : np.array
            Array of normalised psi values between 0 and 1.

        Returns
        -------
        np.array
            The computed pressure values at each normalised psi value.
        """

        pn_ = np.clip(np.array(pn), 0, 1)[np.newaxis]
        shape_pn = np.shape(pn_)

        ones = np.ones_like(pn)
        integrated_coeffs = self.alpha / np.arange(1, len(self.alpha_exp) + 1)
        norm_pressure = np.sum(
            integrated_coeffs.reshape(
                list(np.shape(integrated_coeffs)) + [1] * len(shape_pn)
            )
            * (
                ones
                - pn
                ** (
                    self.alpha_exp.reshape(
                        list(np.shape(self.alpha_exp)) + [1] * len(shape_pn)
                    )
                    + 1
                )
            ),
            axis=0,
        )
        if hasattr(self, "L") is False:
            self.L = 1
            print(
                "This is using self.L=1, which is likely not appropriate. Please calculate Jtor first to ensure the correct normalization."
            )
        pressure = self.L * norm_pressure * (self.psi_axis - self.psi_bndry)
        return pressure

    def fvac(self):
        """
        Return the vaccum field parameter fvac = R*Btor.

        Parameters
        ----------

        Returns
        -------
        float
            The vaccum field parameter fvac = R*Btor.
        """
        return self._fvac


class TensionSpline(Profile):
    """
    Class describing the toroidal current density profile formulation that uses
    tension spline.
    """

    def __init__(
        self,
        Ip,
        fvac,
        pp_knots,
        pp_values,
        pp_values_2,
        pp_sigma,
        ffp_knots,
        ffp_values,
        ffp_values_2,
        ffp_sigma,
        Raxis=1,
        Ip_logic=True,
    ):
        """
        Initialise the class.

        Parameters
        ----------
        Ip : float
            Total plasma current [A].
        fvac : float
            Vacuum toroidal field strength (f = R*B_tor) [T].
        pp_knots : list or np.array
            Knot points of pprime profile.
        pp_values : list or np.array
            Values of pprime at knot points.
        pp_values_2 : list or np.array
            Values of 2nd derivative of pprime at knot points.
        pp_sigma : float
            Tension spline parameter for pprime (non-neative).
        ffp_knots : list or np.array
            Knot points of ffprime profile.
        ffp_values : list or np.array
            Values of ffprime at knot points.
        ffp_values_2 : list or np.array
            Values of 2nd derivative of ffprime at knot points.
        ffp_sigma : float
            Tension spline parameter for ffprime (non-neative).
        Raxis : float
            Radial scaling parameter (non-negative).
        Ip_logic : bool
            If True, entire profile is re-normalised to satisfy Ip identically.
        """

        # set parameters for later use
        self.Ip = Ip
        self._fvac = fvac
        self.pp_knots = np.array(pp_knots)
        self.pp_values = np.array(pp_values)
        self.pp_values_2 = np.array(pp_values_2)
        self.pp_sigma = pp_sigma
        self.ffp_knots = np.array(ffp_knots)
        self.ffp_values = np.array(ffp_values)
        self.ffp_values_2 = np.array(ffp_values_2)
        self.ffp_sigma = ffp_sigma
        self.Raxis = Raxis
        self.Ip_logic = Ip_logic

        # Compatibility flag for native current-density profiles.
        self.fast = True

    def Jtor_part2(self, R, Z, psi, psi_axis, psi_bndry, mask):
        """
        Second part of the calculation that will use the explicit
        parameterisation of the chosen profile function to calculate Jtor.
        Typically used for more modelling more complex shaped profiles
        (from magnetics + MSE plasma reconstructions).

        This is given by:
            Jtor(ψ, R, Z) = L * [ (R / Raxis) * p'(ψ) + (Raxis / R) * FF'(ψ) / mu0 ],

        where
            p'(ψ) = sum(f_n),

            FF'(ψ) = sum(f_n),

        where f_n is the tension spline.

        See https://catxmai.github.io/pdfs/Math212_ProjectReport.pdf for definition.

        Parameters
        ----------
        R : np.ndarray
            Radial coordinates of the grid points.
        Z : np.ndarray
            Vertical coordinates of the grid points.
        psi : np.ndarray
            Total poloidal field flux at each grid point [Webers/2pi].
        psi_axis : float
            Value of the poloidal field flux at the magnetic axis of the plasma.
        psi_bndry : float
            Value of the poloidal field flux at the boundary of the plasma (last closed
            flux surface).
        mask : np.ndarray
            Mask of points inside the last closed flux surface.

        Returns
        -------
        np.array
            Toroidal current density on the computational grid [A/m^2].
        """

        # set flux on boundary
        if psi_bndry is None:
            psi_bndry = psi[0, 0]
        self.psi_bndry = psi_bndry

        # grid sizes
        dR = R[1, 0] - R[0, 0]
        dZ = Z[0, 1] - Z[0, 0]

        # calculate normalised psi
        psi_norm = (psi - psi_axis) / (psi_bndry - psi_axis)
        psi_norm = np.clip(psi_norm, 0.0, 1.0)

        # calculate pprime and ffprime terms
        pprime_term = self.tension_spline(
            psi_norm,
            self.pp_knots,
            self.pp_values,
            self.pp_values_2,
            self.pp_sigma,
        )
        pprime_term *= R / self.Raxis

        ffprime_term = self.tension_spline(
            psi_norm,
            self.ffp_knots,
            self.ffp_values,
            self.ffp_values_2,
            self.ffp_sigma,
        )
        ffprime_term *= self.Raxis / R
        ffprime_term /= mu0

        # sum together
        Jtor = pprime_term + ffprime_term

        # if there is a masking function, use it
        if mask is not None:
            Jtor *= mask

        # normalise with respect to plasma current if needed
        if self.Ip_logic:
            jtorIp = np.sum(Jtor)
            if jtorIp == 0:
                self.problem_psi = psi
                raise ValueError("Total plasma current is zero! Cannot renormalise.")
            L = self.Ip / (jtorIp * dR * dZ)
            Jtor = L * Jtor
        else:
            L = 1.0

        # store results
        self.jtor = Jtor.copy()
        self.L = L

        return self.jtor

    def pprime(self, pn):
        """
        Calculate the 1D pprime (dp/dpsi) profile in the plasma (vs. the
        normalised poloidal flux).

        Parameters
        ----------
        pn : np.array
            Array of normalised psi values between 0 and 1.

        Returns
        -------
        np.array
            The computed pprime values at each normalised psi value.
        """

        shape = self.tension_spline(
            pn, self.pp_knots, self.pp_values, self.pp_values_2, self.pp_sigma
        )
        return self.L * shape / self.Raxis

    def ffprime(self, pn):
        """
        Calculate the 1D ffprime (FdF/dpsi) profile in the plasma (vs. the
        normalised poloidal flux).

        Parameters
        ----------
        pn : np.array
            Array of normalised psi values between 0 and 1.

        Returns
        -------
        np.array
            The computed ffprime values at each normalised psi value.
        """

        shape = self.tension_spline(
            pn,
            self.ffp_knots,
            self.ffp_values,
            self.ffp_values_2,
            self.ffp_sigma,
        )
        return self.L * shape * self.Raxis

    def fvac(self):
        """
        Return the vaccum field parameter fvac = R*Btor.

        Parameters
        ----------

        Returns
        -------
        float
            The vaccum field parameter fvac = R*Btor.
        """
        return self._fvac

    def tension_spline(self, x, xn, yn, zn, sigma):
        """
        Evaluate the tension spline at locations in x using knot points xn,
        values at knot points yn, and second derivative values at knot points zn.
        Tension parameter is a non-negative float sigma.

        Parameters
        ----------
        x : float or np.array
            Normalised psi value(s) between 0 and 1 (a scalar or an array of any shape).
        xn : np.array
            Array of knot points between 0 and 1.
        yn : np.array
            Values of the function at the knot points.
        zn : np.array
            Values of the 2nd derivative of the function at the knot points.
        sigma : float
            Tension spline parameter (non-negative).

        Returns
        -------
        float or np.array
            The tension spline evaluated at the points in x (a float if x is a
            scalar, otherwise an array with the same shape as x).
        """

        # accept scalars as well as arrays of any shape
        x = np.asarray(x, dtype=float)
        size = x.shape
        x = x.reshape(-1)

        # fixed parameters
        x_diffs = xn[1:] - xn[0:-1]
        sinh_diffs = np.sinh(sigma * x_diffs)

        # initial solution array (each column is f_n(x) for a different n)
        X = np.tile(x, (len(x_diffs), 1)).T

        # calculate the terms in the spline (vectorised for speed)
        t1 = (yn[0:-1] - zn[0:-1] / (sigma**2)) * ((xn[1:] - X) / x_diffs)
        t2 = (
            zn[0:-1] * np.sinh(sigma * (xn[1:] - X))
            + zn[1:] * np.sinh(sigma * (X - xn[0:-1]))
        ) / ((sigma**2) * sinh_diffs)
        t3 = (yn[1:] - zn[1:] / (sigma**2)) * ((X - xn[0:-1]) / x_diffs)

        # sum the values
        sol = t1 + t2 + t3

        # zero out values outisde range of each f_n(x) as they're not valid (recall definition of tension spline)
        for n in range(0, len(xn) - 1):
            ind = (xn[n] <= x) & (x <= xn[n + 1])
            sol[~ind, n] = 0

        # sum to find (alomst) final solution
        f = np.sum(sol, axis=1)

        # points on an interior knot lie in two adjacent intervals so are double counted
        f[np.isin(x, xn[1:-1])] /= 2

        # restore the shape of the input (a plain float for a scalar input)
        if len(size) == 0:
            return float(f[0])
        return f.reshape(size)


class GeneralPprimeFFprime(Profile):
    """
    Class describing the general toroidal current density profile formulation:

    Jtor(ψ, R, Z) = R * p'(ψ) + FF'(ψ) / (R * mu0).

    """

    def __init__(
        self,
        Ip,
        fvac,
        psi_n,
        pprime_data=None,
        ffprime_data=None,
        p_data=None,
        f_data=None,
        Raxis=1,
        Ip_logic=True,
        interpolator="univariate_spline",
    ):
        """
        Initialise the class.

        Parameters
        ----------
        Ip : float
            Total plasma current [A].
        fvac : float
            Vacuum toroidal field strength (f = R*B_tor) [T].
        psi_n : np.array
            A 1D array of normalised poloidal flux values, corresponding to the data in pprime_data, ffprime_data,
            p_data, and f_data.
        pprime_data : np.array
            A 1D array of dp/dpsi_n (p is pressure) values at each value of normalised flux in psi_n.
        ffprime_data : np.array
            A 1D array of fdf/dpsi_n (f is the toroidal magnetic field) values at each value of normalised flux in psi_n.
        p_data : np.array
            A 1D array of p (p is pressure) values at each value of normalised flux in psi_n.
        f_data : np.array
            A 1D array of f (f is the toroidal magnetic field) values at each value of normalised flux in psi_n.
        Raxis : float
            Radial scaling parameter (non-negative).
        Ip_logic : bool
            If True, entire profile is re-normalised to satisfy Ip identically.
        interpolator : str
            Defaults to using a "univariate_spline" or optionally a "cubic_spline", data points in both are interpolated exactly.
        """

        # set parameters for later use
        self.Ip = Ip
        self._fvac = fvac
        self.psi_n = psi_n
        self.pprime_data = pprime_data
        self.ffprime_data = ffprime_data
        self.p_data = p_data
        self.f_data = f_data
        self.Raxis = Raxis
        self.Ip_logic = Ip_logic
        if self.Ip_logic is False:
            self.L = 1  # no normalisation in this case
        self.interpolator = interpolator

        # initialise profile ready for calculation
        self.initialize_profile()

        # Compatibility flag for native current-density profiles.
        self.fast = True

    def initialize_profile(self):
        """
        Set up the attributes in the class ready for Jtor calculations.

        Parameters
        ----------

        Returns
        -------

        """

        # interpolate the p data
        self.pprime_func = None
        self.p_func = None

        if self.pprime_data is not None:
            if self.interpolator == "cubic_spline":
                self.pprime_func = CubicSpline(
                    self.psi_n, self.pprime_data, bc_type="natural"
                )
            else:
                self.pprime_func = UnivariateSpline(
                    self.psi_n,
                    self.pprime_data,
                    s=0,
                    k=1,
                )

        if self.p_data is not None:
            if self.interpolator == "cubic_spline":
                self.p_func = CubicSpline(self.psi_n, self.p_data, bc_type="natural")
            else:
                self.p_func = UnivariateSpline(self.psi_n, self.p_data, s=0, k=1)

        # if pprime_func still not provided, use p_func derivative, else throw error
        if self.pprime_func is None and self.p_func is not None:
            self.pprime_func = self.p_func.derivative(n=1)
        elif self.pprime_func is None and self.p_func is None:
            raise ValueError("Need to provide either 'pprime_data' or 'p_data'")

        # interpolate the f data
        self.ffprime_func = None
        self.f_func = None

        if self.ffprime_data is not None:
            if self.interpolator == "cubic_spline":
                self.ffprime_func = CubicSpline(
                    self.psi_n, self.ffprime_data, bc_type="natural"
                )
            else:
                self.ffprime_func = UnivariateSpline(
                    self.psi_n, self.ffprime_data, s=0, k=1
                )

        if self.f_data is not None:
            if self.interpolator == "cubic_spline":
                self.f_func = CubicSpline(self.psi_n, self.f_data, bc_type="natural")
            else:
                self.f_func = UnivariateSpline(self.psi_n, self.f_data, s=0, k=1)

        # if ffprime_func still not provided, use f_func derivative, else throw error
        if self.ffprime_func is None and self.f_func is not None:
            fprime_func = self.f_func.derivative(n=1)
            self.ffprime_func = lambda x: self.f_func(x) * fprime_func(x)
        elif self.ffprime_func is None and self.f_func is None:
            raise ValueError("Need to provide either 'ffprime_data' or 'f_data'")

    def Jtor_part2(
        self,
        R,
        Z,
        psi,
        psi_axis,
        psi_bndry,
        mask,
        torefine=False,
        refineR=None,
    ):
        """
        Second part of the calculation that will use the interpolated profile
        functions to calculate Jtor.

        This is given by:
            Jtor(ψ, R, Z) = L * [ (R / Raxis) * p'(ψ) + (Raxis / R) * FF'(ψ) / mu0 ],

        where
            p'(ψ) = pprime_func(ψ)

            FF'(ψ) = ffprime_func(ψ)

        and note ψ here is the normalised psi.

        Parameters
        ----------
        R : np.ndarray
            Radial coordinates of the grid points.
        Z : np.ndarray
            Vertical coordinates of the grid points.
        psi : np.ndarray
            Total poloidal field flux at each grid point [Webers/2pi].
        psi_axis : float
            Value of the poloidal field flux at the magnetic axis of the plasma.
        psi_bndry : float
            Value of the poloidal field flux at the boundary of the plasma (last closed
            flux surface).
        mask : np.ndarray
            Mask of points inside the last closed flux surface.


        Returns
        -------
        np.array
            Toroidal current density on the computational grid [A/m^2].
        """

        # set flux on boundary
        if psi_bndry is None:
            psi_bndry = psi[0, 0]
        self.psi_bndry = psi_bndry

        # grid sizes
        dR = R[1, 0] - R[0, 0]
        dZ = Z[0, 1] - Z[0, 0]

        if torefine:
            R = 1.0 * refineR

        # calculate normalised psi
        psi_norm = (psi - psi_axis) / (psi_bndry - psi_axis)
        psi_norm = np.clip(psi_norm, 0.0, 1.0)

        # calculate the p' and FF' profiles
        pprime_term = self.pprime_func(psi_norm)
        pprime_term *= R / self.Raxis

        ffprime_term = self.ffprime_func(psi_norm)
        ffprime_term *= self.Raxis / R
        ffprime_term /= mu0

        # sum together
        Jtor = pprime_term + ffprime_term

        if torefine:
            return Jtor

        # put to zero all current outside the LCFS
        if mask is not None:
            Jtor *= mask

        # if Ip normalisation is required, do it
        if self.Ip_logic:
            jtorIp = np.sum(Jtor)
            if jtorIp == 0:
                self.problem_psi = psi
                raise ValueError("Total plasma current is zero! Cannot renormalise.")
            L = self.Ip / (jtorIp * dR * dZ)
            Jtor = L * Jtor
        else:
            L = 1.0

        # store parameters
        self.jtor = Jtor.copy()
        self.L = L

        return self.jtor

    def pprime(self, pn):
        """
        Calculate the 1D pprime (dp/dpsi) profile in the plasma (vs. the
        normalised poloidal flux).

        Parameters
        ----------
        pn : np.array
            Array of normalised psi values between 0 and 1.

        Returns
        -------
        np.array
            The computed pprime values at each normalised psi value.
        """

        pn_ = np.clip(np.array(pn), 0, 1)

        if self.Ip_logic:
            if hasattr(self, "L") is False:
                raise ValueError(
                    "The pprime profile cannot be normalised. "
                    "Please first calculate Jtor for this profile. "
                )
        return self.L * self.pprime_func(pn_) / self.Raxis

    def ffprime(self, pn):
        """
        Calculate the 1D ffprime (FdF/dpsi) profile in the plasma (vs. the
        normalised poloidal flux).

        Parameters
        ----------
        pn : np.array
            Array of normalised psi values between 0 and 1.

        Returns
        -------
        np.array
            The computed ffprime values at each normalised psi value.
        """

        pn_ = np.clip(np.array(pn), 0, 1)

        if self.Ip_logic:
            if hasattr(self, "L") is False:
                raise ValueError(
                    "The ffprime profile cannot be normalised. "
                    "Please first calculate Jtor for this profile. "
                )
        return self.L * self.ffprime_func(pn_) * self.Raxis

    def pressure(self, pn):
        """
        Calculate the 1D pressure profile in the plasma (vs. the normalised
        poloidal flux).

        Parameters
        ----------
        pn : np.array
            Array of normalised psi values between 0 and 1.

        Returns
        -------
        np.array
            The computed pressure values at each normalised psi value.
        """

        pn_ = np.clip(np.array(pn), 0, 1)

        if self.p_func is not None:
            return self.p_func(pn_)
        else:
            return super(GeneralPprimeFFprime, self).pressure(pn_)

    def fpol(self, pn):
        """
        Calculate the 1D toroidal magnetic profile in the plasma (vs. the normalised
        poloidal flux).

        Parameters
        ----------
        pn : np.array
            Array of normalised psi values between 0 and 1.

        Returns
        -------
        np.array
            The computed toroidal magnetic field values at each normalised psi value.
        """

        pn_ = np.clip(np.array(pn), 0, 1)

        if self.f_func is not None:
            return self.f_func(pn_)
        else:
            return super(GeneralPprimeFFprime, self).fpol(pn_)

    def fvac(self):
        """
        Return the vaccum field parameter fvac = R*Btor.

        Parameters
        ----------

        Returns
        -------
        float
            The vaccum field parameter fvac = R*Btor.
        """
        return self._fvac
