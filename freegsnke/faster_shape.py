import numpy as np
from numpy import (
    abs,
    amax,
    arctan2,
    argmax,
    argmin,
    clip,
    cos,
    dot,
    linspace,
    pi,
    sin,
    sqrt,
    sum,
    zeros,
)
from numpy.linalg import inv
from scipy import interpolate


def innerOuterSeparatrix(eq, profiles, Z=0.0):
    """
    Locate R co ordinates of separatrix at both
    inboard and outboard poloidal midplane (Z = 0)
    """
    # Find the closest index to requested Z
    Zindex = np.argmin(abs(eq.Z[0, :] - Z))

    # Normalise psi at this Z index
    psinorm = (eq.psi()[:, Zindex] - eq.psi_axis) / (eq.psi_bndry - eq.psi_axis)

    # Start from the magnetic axis
    Rindex_axis = np.argmin(abs(eq.R[:, 0] - profiles.opt[0][0]))

    # Inner separatrix
    # Get the maximum index where psi > 1 in the R index range from 0 to Rindex_axis
    outside_inds = np.argwhere(psinorm[:Rindex_axis] > 1.0)

    if outside_inds.size == 0:
        R_sep_in = eq.Rmin
    else:
        Rindex_inner = np.amax(outside_inds)

        # Separatrix should now be between Rindex_inner and Rindex_inner+1
        # Linear interpolation
        R_sep_in = (
            eq.R[Rindex_inner, Zindex] * (1.0 - psinorm[Rindex_inner + 1])
            + eq.R[Rindex_inner + 1, Zindex] * (psinorm[Rindex_inner] - 1.0)
        ) / (psinorm[Rindex_inner] - psinorm[Rindex_inner + 1])

    # Outer separatrix
    # Find the minimum index where psi > 1
    outside_inds = np.argwhere(psinorm[Rindex_axis:] > 1.0)

    if outside_inds.size == 0:
        R_sep_out = eq.Rmax
    else:
        Rindex_outer = np.amin(outside_inds) + Rindex_axis

        # Separatrix should now be between Rindex_outer-1 and Rindex_outer
        R_sep_out = (
            eq.R[Rindex_outer, Zindex] * (1.0 - psinorm[Rindex_outer - 1])
            + eq.R[Rindex_outer - 1, Zindex] * (psinorm[Rindex_outer] - 1.0)
        ) / (psinorm[Rindex_outer] - psinorm[Rindex_outer - 1])

    return R_sep_in, R_sep_out


def calculate_width(eq, profiles):
    inout = innerOuterSeparatrix(eq, profiles)
    return inout[1] - inout[0]


def find_psisurface(eq, psifunc, r0, z0, r1, z1, psival=1.0, n=100):
    """
    eq      - Equilibrium object
    (r0,z0) - Start location inside separatrix
    (r1,z1) - Location outside separatrix

    n - Number of starting points to use
    """
    # Clip (r1,z1) to be inside domain
    # Shorten the line so that the direction is unchanged
    if abs(r1 - r0) > 1e-6:
        rclip = clip(r1, eq.Rmin, eq.Rmax)
        z1 = z0 + (z1 - z0) * abs((rclip - r0) / (r1 - r0))
        r1 = rclip

    if abs(z1 - z0) > 1e-6:
        zclip = clip(z1, eq.Zmin, eq.Zmax)
        r1 = r0 + (r1 - r0) * abs((zclip - z0) / (z1 - z0))
        z1 = zclip

    r = linspace(r0, r1, n)
    z = linspace(z0, z1, n)

    pnorm = psifunc(r, z, grid=False)

    if hasattr(psival, "__len__"):
        pass

    else:
        # Only one value
        ind = argmax(pnorm > psival)

        # Edited by Bhavin 31/07/18
        # Changed 1.0 to psival in f
        # make f gradient to psival surface
        f = (pnorm[ind] - psival) / (pnorm[ind] - pnorm[ind - 1])

        r = (1.0 - f) * r[ind] + f * r[ind - 1]
        z = (1.0 - f) * z[ind] + f * z[ind - 1]

    return r, z


def Separatrix(eq, profiles, ntheta, psival=1.0):
    """Find the R, Z coordinates of the separatrix for equilbrium
    eq. Returns a tuple of (R, Z, R_X, Z_X), where R_X, Z_X are the
    coordinates of the X-point on the separatrix. Points are equally
    spaced in geometric poloidal angle.

    If opoint, xpoint or psi are not given, they are calculated from eq

    eq - Equilibrium object
    opoint - List of O-point tuples of (R, Z, psi)
    xpoint - List of X-point tuples of (R, Z, psi)
    ntheta - Number of points to find
    psi - Grid of psi on (R, Z)
    axis - A matplotlib axis object to plot points on
    """
    # if psi is None:
    psi = eq.psi()

    # if (opoint is None) or (xpoint is None):
    opoint, xpoint = profiles.opt, profiles.xpt

    psinorm = (psi - opoint[0][2]) / (xpoint[0][2] - opoint[0][2])

    psifunc = interpolate.RectBivariateSpline(eq.R[:, 0], eq.Z[0, :], psinorm)

    r0, z0 = opoint[0][0:2]

    theta_grid = linspace(0, 2 * pi, ntheta, endpoint=False)
    dtheta = theta_grid[1] - theta_grid[0]

    # Avoid putting theta grid points exactly on the X-points
    xpoint_theta = arctan2(xpoint[0][0] - r0, xpoint[0][1] - z0)
    xpoint_theta = xpoint_theta * (xpoint_theta >= 0) + (xpoint_theta + 2 * pi) * (
        xpoint_theta < 0
    )  # let's make it between 0 and 2*pi
    # How close in theta to allow theta grid points to the X-point
    TOLERANCE = 1.0e-3
    if any(abs(theta_grid - xpoint_theta) < TOLERANCE):
        # warn("Theta grid too close to X-point, shifting by half-step")
        theta_grid += dtheta / 2

    isoflux = []
    for theta in theta_grid:
        r, z = find_psisurface(
            eq,
            psifunc,
            r0,
            z0,
            r0 + 10.0 * sin(theta),
            z0 + 10.0 * cos(theta),
            psival=psival,
            n=1000,
        )
        isoflux.append((r, z, xpoint[0][0], xpoint[0][1]))

    return np.array(isoflux)


def geometricElongation(eq, profiles, npoints=20):
    """Calculates the elongation of a plasma using the range of R and Z of
    the separatrix

    """
    separatrix = Separatrix(eq, profiles, ntheta=npoints)[:, 0:2]  # [:,2]
    # Range in Z / range in R
    return (max(separatrix[:, 1]) - min(separatrix[:, 1])) / (
        max(separatrix[:, 0]) - min(separatrix[:, 0])
    )


def shapes_f(eq, profiles):
    width = calculate_width(eq, profiles)  # simple width at z=0:
    opoint = np.array(profiles.opt[0])[np.newaxis]
    # Rvals = eq.R*self.plasma_mask
    # Zvals = eq.Z*self.plasma_mask
    # geometricElongation = (np.max(Zvals)-np.min(Zvals))/(np.max(Rvals)-np.min(Rvals))
    gE = geometricElongation(eq, profiles, npoints=20)
    return width, opoint, gE
