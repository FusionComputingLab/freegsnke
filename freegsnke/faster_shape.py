import numpy as np


def innerOuterSeparatrix(eq, profiles, Z=0.0):
        """
        Locate R co ordinates of separatrix at both
        inboard and outboard poloidal midplane (Z = 0)
        """
        # Find the closest index to requested Z
        Zindex = np.argmin(abs(eq.Z[0, :] - Z))

        # Normalise psi at this Z index
        psinorm = (eq.psi()[:, Zindex] - eq.psi_axis) / (
            eq.psi_bndry - eq.psi_axis
        )

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

