"""
Enables FreeGSNKE-simulated equilibrium data to be read/written to/from IMAS
IDS (via the IMAS-Python package), including netCDF serialisation.

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

from datetime import date

import contourpy
import imas
import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.interpolate import RectBivariateSpline

import freegsnke


def _flux_surface_geometry(eq, psi_n, fpol_1d):
    """
    Traces each requested normalised-flux surface once (via `contourpy`, the
    same technique used by `eq.flux_averaged_function`) and returns the
    geometric quantities and flux-surface averages needed to build several
    `profiles_1d` IDS fields, without re-tracing the surfaces for each one.

    Parameters
    ----------
    eq : freegsnke.equilibrium_update.Equilibrium
        Solved equilibrium object.
    psi_n : np.array
        Normalised flux values to trace (each strictly between 0 and 1).
    fpol_1d : np.array
        F = R*Btor at each value in `psi_n` (e.g. `profiles.fpol(psi_n)`).
        F is constant on a flux surface, so this gives Btor = F/R at every
        traced point without any extra profile evaluation.

    Returns
    -------
    dict of np.array
        `r_inboard`, `r_outboard` : major radius of the surface at the
            magnetic-axis height, on the inboard/outboard side [m].
        `volume` : volume enclosed by the surface [m^3], from the exact
            line-integral form of Pappus's theorem, V = |∮ pi R^2 dZ|.
        `elongation`, `triangularity_upper`, `triangularity_lower` : standard
            Miller-style shape parameters, using the surface's own
            (Rmin, Rmax, Zmin, Zmax) extent - the same formulas as
            freegs4e's `geometricElongation`/`triangularity_upper/lower`,
            generalised from the LCFS to an arbitrary internal surface.
        `avg_inv_R`, `avg_inv_R2` : flux-surface averages <1/R>, <1/R^2>.
        `avg_R_Bp`, `avg_R2_Bp2`, `avg_Bp2` : flux-surface averages
            <R*Bp>, <R^2*Bp^2>, <Bp^2>, from which gm2/gm3/gm7 are built
            (see `write_equilibrium_to_ids`).
        `avg_B2`, `avg_inv_B2` : flux-surface averages <B^2>, <1/B^2> of the
            *total* field (poloidal + toroidal), for gm5/gm4.
    """

    masked_psi = np.ma.array(eq.psiNRZ(R=eq.R, Z=eq.Z), mask=eq.mask_outside_limiter)
    mag_r, mag_z = eq.magneticAxis()[0:2]

    # total-psi spline, reused for cheap Br/Bz on each surface (see the
    # equivalent optimisation in eq.flux_averaged_function)
    psi_total_func = RectBivariateSpline(eq.R_1D, eq.Z_1D, eq.psi())

    cont_gen = contourpy.contour_generator(
        x=eq.R, y=eq.Z, z=masked_psi, line_type=contourpy.LineType.Separate
    )

    n = len(psi_n)
    result = {
        key: np.full(n, np.nan)
        for key in [
            "r_inboard",
            "r_outboard",
            "volume",
            "elongation",
            "triangularity_upper",
            "triangularity_lower",
            "avg_inv_R",
            "avg_inv_R2",
            "avg_R_Bp",
            "avg_R2_Bp2",
            "avg_Bp2",
            "avg_B2",
            "avg_inv_B2",
        ]
    }

    for i, val in enumerate(psi_n):
        lines = [line for line in cont_gen.lines(val) if line.shape[0] > 0]
        distances = [
            np.min(np.linalg.norm(line - [mag_r, mag_z], axis=1)) for line in lines
        ]
        flux_surface = lines[np.argmin(distances)]
        Rc, Zc = flux_surface[:, 0], flux_surface[:, 1]

        # arc length and Bp along the surface
        dl = np.sqrt(np.diff(Rc) ** 2 + np.diff(Zc) ** 2)
        l_cum = np.concatenate(([0], np.cumsum(dl)))
        Br = -psi_total_func(Rc, Zc, dy=1, grid=False) / Rc
        Bz = psi_total_func(Rc, Zc, dx=1, grid=False) / Rc
        Bp2 = Br**2 + Bz**2
        Bp_inv = 1 / np.sqrt(Bp2)
        norm = trapezoid(Bp_inv, l_cum)  # = integral of dl/Bp

        def flux_average(values, Bp_inv=Bp_inv, l_cum=l_cum, norm=norm):
            return trapezoid(values * Bp_inv, l_cum) / norm

        result["avg_inv_R"][i] = flux_average(1 / Rc)
        result["avg_inv_R2"][i] = flux_average(1 / Rc**2)
        result["avg_R_Bp"][i] = flux_average(Rc * np.sqrt(Bp2))
        result["avg_R2_Bp2"][i] = flux_average(Rc**2 * Bp2)
        result["avg_Bp2"][i] = flux_average(Bp2)

        # total field: F is constant on the surface, so Btor = F/R here
        Btor = fpol_1d[i] / Rc
        B2 = Bp2 + Btor**2
        result["avg_B2"][i] = flux_average(B2)
        result["avg_inv_B2"][i] = flux_average(1 / B2)

        # Miller-style shape parameters
        Rmax_i, Rmin_i = np.max(Rc), np.min(Rc)
        Zmax_i, Zmin_i = np.max(Zc), np.min(Zc)
        R_geo = 0.5 * (Rmax_i + Rmin_i)
        a_minor = 0.5 * (Rmax_i - Rmin_i)
        result["elongation"][i] = (Zmax_i - Zmin_i) / (Rmax_i - Rmin_i)
        result["triangularity_upper"][i] = (R_geo - Rc[np.argmax(Zc)]) / a_minor
        result["triangularity_lower"][i] = (R_geo - Rc[np.argmin(Zc)]) / a_minor

        # r_inboard/r_outboard: where the surface crosses the magnetic-axis
        # height (the midplane), on either side of the axis
        dz = Zc - mag_z
        crossing_indices = np.where(np.diff(np.sign(dz)) != 0)[0]
        crossings = []
        for k in crossing_indices:
            f = (mag_z - Zc[k]) / (Zc[k + 1] - Zc[k])
            crossings.append(Rc[k] + f * (Rc[k + 1] - Rc[k]))
        crossings = np.array(crossings)
        inboard = crossings[crossings < mag_r]
        outboard = crossings[crossings > mag_r]
        if inboard.size:
            result["r_inboard"][i] = np.min(inboard)
        if outboard.size:
            result["r_outboard"][i] = np.max(outboard)

        # enclosed volume, exact for the piecewise-linear traced surface:
        # Green's theorem with Q = pi*R^2 gives ∫∫ 2*pi*R dR dZ = ∮ pi*R^2 dZ
        result["volume"][i] = np.abs(trapezoid(np.pi * Rc**2, Zc))

    return result


def write_equilibrium_to_ids(
    eq,
    profiles,
):
    """
    Populates an IMAS `equilibrium` IDS (single time slice) with quantities taken
    from a solved FreeGSNKE equilibrium.

    Parameters
    ----------
    eq : freegsnke.equilibrium_update.Equilibrium
        Solved equilibrium object.
    profiles : freegsnke.jtor_update profile object
        The profile object used to solve for `eq` (e.g. a `ConstrainPaxisIp` or
        `GeneralPprimeFFprime` instance).

    Returns
    -------
    ids_out : imas.ids_toplevel.IDSToplevel
        The populated `equilibrium` IDS.
    """

    # initialise an empty equilibrium IDS
    ids_factory = imas.IDSFactory()
    ids_out = ids_factory.equilibrium()

    # high-level ids properties
    ids_out.ids_properties.name = "FreeGSNKE-generated equilibrium IDS"
    ids_out.ids_properties.homogeneous_time = 1
    ids_out.ids_properties.creation_date = date.today().strftime("%d-%m-%Y")

    # code properties
    ids_out.code.name = freegsnke.__name__
    ids_out.code.description = (
        "A Python-based free-boundary evolutive Grad-Shafranov equilibrium solver."
    )
    ids_out.code.version = freegsnke.__version__
    ids_out.code.repository = "https://github.com/FusionComputingLab/freegsnke"

    # vacuum toroidal field properties (rcentr taken as centre of limiter geometry)
    rcentr = 0.5 * (np.min(eq.tokamak.limiter.R) + np.max(eq.tokamak.limiter.R))
    ids_out.vacuum_toroidal_field.r0 = rcentr
    ids_out.vacuum_toroidal_field.b0 = np.array([profiles.fvac() / rcentr])

    ids_out.time = np.array([0.0])
    ids_out.time_slice.resize(1)
    time_slice = ids_out.time_slice[0]
    time_slice.time = 0.0

    # boundary quantities
    time_slice.boundary.type = 1 - profiles.flag_limiter  # 0 = limited, 1 = diverted
    time_slice.boundary.psi_norm = 1.0
    time_slice.boundary.psi = 2 * np.pi * eq.psi_bndry
    time_slice.boundary.minor_radius = eq.minorRadius()
    boundary = eq.separatrix(ntheta=360)
    time_slice.boundary.outline.r = boundary[:, 0]
    time_slice.boundary.outline.z = boundary[:, 1]

    # global quantities
    time_slice.global_quantities.ip = eq.plasmaCurrent()
    time_slice.global_quantities.psi_axis = 2 * np.pi * eq.psi_axis
    time_slice.global_quantities.psi_boundary = 2 * np.pi * eq.psi_bndry
    mag_r, mag_z = eq.magneticAxis()[0:2]
    time_slice.global_quantities.magnetic_axis.r = mag_r
    time_slice.global_quantities.magnetic_axis.z = mag_z

    # 1D profile quantities. psi_n is clipped away from the exact axis/
    # boundary values (0, 1): q (and everything derived from it below) is
    # singular there, and eq.psiN_1D(N) would otherwise include them exactly.
    # Matches the default clip range already used by eq.flux_averaged_function.
    N = eq.nx
    psi_n = np.clip(eq.psiN_1D(N), 0.01, 0.99)
    psi_actual = eq.psi_axis + psi_n * (eq.psi_bndry - eq.psi_axis)

    time_slice.profiles_1d.psi = (2 * np.pi * psi_actual).squeeze()
    time_slice.profiles_1d.psi_norm = psi_n.squeeze()
    time_slice.profiles_1d.pressure = profiles.pressure(psi_n).squeeze()
    fpol_1d = profiles.fpol(psi_n)
    time_slice.profiles_1d.f = fpol_1d.squeeze()
    time_slice.profiles_1d.dpressure_dpsi = profiles.pprime(psi_n).squeeze()
    time_slice.profiles_1d.f_df_dpsi = profiles.ffprime(psi_n).squeeze()
    q_1d = eq.q(psi_n)
    time_slice.profiles_1d.q = q_1d.squeeze()

    # flux-averaged toroidal current density
    jtor_interp = RectBivariateSpline(eq.R_1D, eq.Z_1D, profiles.jtor)
    flux_averaged_jtor, _ = eq.flux_averaged_function(
        f=lambda R, Z: jtor_interp(R, Z, grid=False),
        psi_n=psi_n,
    )
    time_slice.profiles_1d.j_phi = flux_averaged_jtor.squeeze()

    # Remaining profiles all derive from one pass of flux-surface tracing,
    # plus the safety factor q above. See the write-up accompanying this
    # change for the full derivation; in short:
    #
    # q is defined as q(psi) = (1/2*pi) * d(Phi_tor)/d|psi|, where Phi_tor is
    # the toroidal flux enclosed by the surface and |psi| = psi_n * |psi_bndry
    # - psi_axis| is the (unsigned) distance in psi from the axis - unsigned
    # because q itself is built from |Bp|, not signed Bp, so it already gives
    # the *magnitude* of the flux-expansion rate regardless of which way psi
    # runs from axis to edge (freegs4e's psi_axis can be above or below
    # psi_bndry depending on the current/field directions - here that would
    # otherwise flip the sign of a naively-integrated Phi_tor). Integrating
    # gives:
    #   Phi_tor(psi_n) = integral_0^psi_n  2*pi*q(psi_n') |psi_bndry-psi_axis| dpsi_n'
    # which we verified against a direct 2D area integral of B_tor over the
    # enclosed region (agreement to ~1-3%, consistent with the grid
    # resolution and the missing "wedge" between the axis and our innermost
    # traced surface, added back in below assuming q is ~constant over that
    # small residual gap).
    geom = _flux_surface_geometry(eq, psi_n, fpol_1d)

    psi_mag = psi_n * abs(eq.psi_bndry - eq.psi_axis)
    phi_1d = cumulative_trapezoid(2 * np.pi * q_1d, psi_mag, initial=0.0)
    phi_1d = phi_1d + 2 * np.pi * q_1d[0] * psi_mag[0]  # innermost wedge

    b0 = ids_out.vacuum_toroidal_field.b0[0]
    # rho_tor = sqrt(Phi/(pi*b0)) (IMAS DD definition); rho_tor_norm is its
    # ratio to the boundary value, which cancels the pi*b0 factor
    rho_tor = np.sqrt(phi_1d / (np.pi * b0))
    rho_tor_norm = np.sqrt(phi_1d / phi_1d[-1])

    # |grad(rho_tor)| = (d(rho_tor)/d|psi|) * |grad(psi)| = (d(rho_tor)/d|psi|) * R*Bp,
    # and d(rho_tor)/d|psi| = q/(b0*rho_tor) (chain rule through Phi_tor, using
    # the same relation as above) - constant on a given flux surface, so it
    # factors out of each flux-surface average below
    drho_dpsi_mag = q_1d / (b0 * rho_tor)
    gm1 = geom["avg_inv_R2"]  # <1/R^2>
    gm9 = geom["avg_inv_R"]  # <1/R>
    gm7 = drho_dpsi_mag * geom["avg_R_Bp"]  # <|grad(rho_tor)|>
    gm3 = drho_dpsi_mag**2 * geom["avg_R2_Bp2"]  # <|grad(rho_tor)|^2>
    gm2 = drho_dpsi_mag**2 * geom["avg_Bp2"]  # <|grad(rho_tor)|^2/R^2>

    time_slice.profiles_1d.phi = phi_1d.squeeze()
    time_slice.profiles_1d.rho_tor_norm = rho_tor_norm.squeeze()
    time_slice.profiles_1d.gm1 = gm1.squeeze()
    time_slice.profiles_1d.gm2 = gm2.squeeze()
    time_slice.profiles_1d.gm3 = gm3.squeeze()
    time_slice.profiles_1d.gm7 = gm7.squeeze()
    time_slice.profiles_1d.gm9 = gm9.squeeze()
    time_slice.profiles_1d.gm4 = geom["avg_inv_B2"].squeeze()  # <1/B^2>
    time_slice.profiles_1d.gm5 = geom["avg_B2"].squeeze()  # <B^2>
    time_slice.profiles_1d.r_inboard = geom["r_inboard"].squeeze()
    time_slice.profiles_1d.r_outboard = geom["r_outboard"].squeeze()
    time_slice.profiles_1d.volume = geom["volume"].squeeze()
    time_slice.profiles_1d.elongation = geom["elongation"].squeeze()
    time_slice.profiles_1d.triangularity_upper = geom["triangularity_upper"].squeeze()
    time_slice.profiles_1d.triangularity_lower = geom["triangularity_lower"].squeeze()

    # 2D fields (total, plasma, and tokamak flux - jtor also stored)
    tokamak_psi = eq.tokamak.getPsitokamak(eq._vgreen)
    two_d_fields = [
        (0, "total", 2 * np.pi * eq.psi(), profiles.jtor),
        (4, "plasma", 2 * np.pi * eq.plasma_psi, None),
        (1, "vacuum", 2 * np.pi * tokamak_psi, None),
    ]

    time_slice.profiles_2d.resize(len(two_d_fields))
    for i, (type_index, type_name, psi_2d, j_phi_2d) in enumerate(two_d_fields):
        profiles_2d = time_slice.profiles_2d[i]
        profiles_2d.type.index = type_index
        profiles_2d.type.name = type_name
        profiles_2d.grid_type.name = "rectangular"
        profiles_2d.grid_type.index = 1
        profiles_2d.grid_type.description = "Rectangular grid with dims (R, Z)."
        profiles_2d.grid.dim1 = eq.R_1D
        profiles_2d.grid.dim2 = eq.Z_1D
        profiles_2d.psi = psi_2d
        if j_phi_2d is not None:
            profiles_2d.j_phi = j_phi_2d

    return ids_out


def save_equilibrium_ids(ids, path):
    """
    Writes an `equilibrium` IDS to a netCDF file.

    Parameters
    ----------
    ids : imas.ids_toplevel.IDSToplevel
        The `equilibrium` IDS to save (e.g. as returned by `write_equilibrium_to_ids`).
    path : str
        Destination netCDF file path (should end in `.nc`).
    """

    with imas.DBEntry(path, "w") as db_entry:
        db_entry.put(ids)


def load_equilibrium_ids(path):
    """
    Reads an `equilibrium` IDS back from a netCDF file.

    Parameters
    ----------
    path : str
        Path to a netCDF file previously written by `save_equilibrium_ids`.

    Returns
    -------
    ids : imas.ids_toplevel.IDSToplevel
        The `equilibrium` IDS loaded from file.
    """

    with imas.DBEntry(path, "r") as db_entry:
        return db_entry.get("equilibrium")
