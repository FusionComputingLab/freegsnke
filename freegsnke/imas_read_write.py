"""
Enables FreeGSNKE-simulated equilibrium data to be read/written into IMAS IDS formats. 

Copyright 2024 Nicola C. Amorisco, George K. Holt, Kamran Pentland, Adriano Agnello, Alasdair Ross, Matthijs Mars.

FreeGSNKE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
"""

from copy import deepcopy
from datetime import date

import imas
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.linalg import lstsq
from scipy.interpolate import RectBivariateSpline

import freegsnke
from freegsnke import equilibrium_update, jtor_update


def write_equilibrium_to_ids(
    eq,
):
    """
     Function that will take a FreeGSNKE equilibrium and fill in the relevant quantities within
     the IMAS equilbirium IDS.

     Parameters
    -------
     eq : class
         Equilbirium class from freegsnke containing the simulated equilibrium data.

     Returns
    ----
     ids : class
         The imas ids that contains all the required equilibrium data.
    """

    print("---")
    print("IDS being populated...")
    print("   high level properties...")

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
    # ids_out.code.version = freegsnke.__version__
    ids_out.code.repository = "https://github.com/FusionComputingLab/freegsnke"

    # vacuum toroidal field properties
    print("   vacuum field properties...")
    r0 = (
        np.min(eq.tokamak.limiter.R) + np.max(eq.tokamak.limiter.R)
    ) / 2  # taken as centre of limiter geometry
    ids_out.vacuum_toroidal_field.r0 = r0
    ids_out.vacuum_toroidal_field.b0 = np.array([eq._profiles.fvac() / r0])

    # set default time
    ids_out.time = np.array([0.0])

    # create a single timeslice
    ids_out.time_slice.resize(1)
    time_slice = ids_out.time_slice[
        0
    ]  # store in temp object for each access throughout this script
    time_slice.time = 0.0

    # time slices: boundary quantities
    print("   plasma boundary quantities...")
    time_slice.boundary.type = (
        1 - eq._profiles.flag_limiter
    )  # 0 = limited, 1 = diverted
    time_slice.boundary.psi_norm = 1.0
    time_slice.boundary.psi = 2 * np.pi * eq.psi_bndry
    boundary = eq.separatrix(ntheta=360)
    time_slice.boundary.outline.r = boundary[:, 0]
    time_slice.boundary.outline.z = boundary[:, 1]
    # time_slice.boundary.geometric_axis.r = eq.Rgeometric()
    # time_slice.boundary.geometric_axis.z = eq.Zgeometric()
    # time_slice.boundary.minor_radius = eq.minorRadius()
    # time_slice.boundary.elongation = eq.geometricElongation()
    # time_slice.boundary.triangularity_lower = eq.triangularity_lower()
    # time_slice.boundary.triangularity_upper = eq.triangularity_upper()
    # time_slice.boundary.triangularity = eq.triangularity()
    # s_uo, s_ui, s_lo, s_li = eq.squareness()
    # time_slice.boundary.squareness_upper_outer = s_uo
    # time_slice.boundary.squareness_upper_inner = s_ui
    # time_slice.boundary.squareness_lower_outer = s_lo
    # time_slice.boundary.squareness_lower_inner = s_li

    # time slices: global quantities
    print("   global quantities...")
    time_slice.global_quantities.ip = eq._profiles.Ip
    # time_slice.global_quantities.area = eq._sep_area
    # time_slice.global_quantities.volume = eq.plasmaVolume()
    # time_slice.global_quantities.length_pol = eq._sep_length
    time_slice.global_quantities.psi_axis = 2 * np.pi * eq.psi_axis
    time_slice.global_quantities.psi_boundary = 2 * np.pi * eq.psi_bndry
    mag_axis = eq.magneticAxis()
    time_slice.global_quantities.magnetic_axis.r = mag_axis[0]
    time_slice.global_quantities.magnetic_axis.z = mag_axis[1]
    # time_slice.global_quantities.magnetic_axis.b_field_phi = eq.Btor(
    #     mag_axis[0], mag_axis[1]
    # )
    # time_slice.global_quantities.current_centre.r = eq.Rcurrent()
    # time_slice.global_quantities.current_centre.z = eq.Zcurrent()

    N = 101
    psi_norm = eq.psiN_1D(N)
    # time_slice.global_quantities.beta_pol = eq.poloidalBeta4()
    # time_slice.global_quantities.beta_tor = eq.toroidalBeta4()
    # time_slice.global_quantities.li_3 = eq.internalInductance3()
    q_profile = eq.q(psi_norm)
    # q_interp = interpolate.UnivariateSpline(psi_norm, q_profile)
    # time_slice.global_quantities.q_axis = q_interp(0)
    # time_slice.global_quantities.q_95 = q_interp(0.95)
    # time_slice.global_quantities.q_min.value = np.min(q_profile)
    # time_slice.global_quantities.q_min.psi_norm = psi_norm[np.argmin(q_profile)]
    # time_slice.global_quantities.q_min.psi = (
    #     2
    #     * np.pi
    #     * (psi_norm[np.argmin(q_profile)] * (eq.psi_bndry - eq.psi_axis) + eq.psi_axis)
    # )

    # # time slices: x-points
    # xpts = eq.xpt
    # time_slice.constraints.x_point.resize(xpts.shape[0])
    # for i in range(0, xpts.shape[0]):
    #     time_slice.constraints.x_point[i].exact = 1
    #     time_slice.constraints.x_point[i].position_measured.r = xpts[i, 0]
    #     time_slice.constraints.x_point[i].position_measured.z = xpts[i, 1]

    # # time slices: strikepoints
    # strikes = eq.strikepoints()
    # time_slice.constraints.strike_point.resize(strikes.shape[0])
    # for i in range(0, strikes.shape[0]):
    #     time_slice.constraints.strike_point[i].exact = 1
    #     time_slice.constraints.strike_point[i].position_measured.r = strikes[i, 0]
    #     time_slice.constraints.strike_point[i].position_measured.z = strikes[i, 1]

    # # time slices: coil currents (retains ordering of coils used in the tokamak object)
    # print("   coil currents...")
    # currents = eq.tokamak.getCurrentsVec()

    # # active coil currents
    # time_slice.constraints.pf_current.resize(eq.tokamak.n_active_coils)
    # for i in range(0, eq.tokamak.n_active_coils):
    #     time_slice.constraints.pf_current[i].exact = 1
    #     time_slice.constraints.pf_current[i].measured = currents[i]

    # # passive structure currents
    # time_slice.constraints.pf_passive_current.resize(eq.tokamak.n_passive_coils)
    # for i in range(0, eq.tokamak.n_passive_coils):
    #     time_slice.constraints.pf_passive_current[i].exact = 1
    #     time_slice.constraints.pf_passive_current[i].measured = currents[i]

    # time slices: 1D profile quantities
    print("   1D profile quantities...")
    time_slice.profiles_1d.psi = 2 * np.pi * np.linspace(eq.psi_axis, eq.psi_bndry, N).squeeze()
    time_slice.profiles_1d.psi_norm = psi_norm.squeeze()
    time_slice.profiles_1d.pressure = eq.pressure(time_slice.profiles_1d.psi_norm).squeeze()
    time_slice.profiles_1d.f = eq.fpol(time_slice.profiles_1d.psi_norm).squeeze()
    # time_slice.profiles_1d.dpressure_dpsi = eq._profiles.pprime(
    #     time_slice.profiles_1d.psi_norm
    # ).squeeze()
    # time_slice.profiles_1d.f_df_dpsi = eq._profiles.ffprime(
    #     time_slice.profiles_1d.psi_norm
    # ).squeeze()
    time_slice.profiles_1d.q = q_profile.squeeze()

    # flux-averaged jtor
    def f(R,Z):
        jtor = RectBivariateSpline(eq.R_1D, eq.Z_1D, eq._profiles.jtor)
        return jtor(R, Z, grid=False)
    # call the method
    flux_averaged_jtor, psi_n = eq.flux_averaged_function(
        f=f,
        psi_n=psi_norm,
        )
    time_slice.profiles_1d.j_phi = flux_averaged_jtor.squeeze()


    # time slices: 2D profile quantities
    print("   2D profile quantities...")

    # allows us to split up the contributions from the total, plasma, and coil regions (can also include pf_active and pf_passive contributions)
    time_slice.profiles_2d.resize(1)  # one each for total, plasma, and coil flux

    # total poloidal flux
    time_slice.profiles_2d[0].type.name = "Total poloidal flux field"
    time_slice.profiles_2d[0].type.index = 0
    time_slice.profiles_2d[0].grid_type.name = "rectangular"
    time_slice.profiles_2d[0].grid_type.index = 1
    time_slice.profiles_2d[0].grid_type.description = (
        "Here we use a rectangular grid with dims (R,Z)."
    )
    time_slice.profiles_2d[0].grid.dim1 = eq.R_1D
    time_slice.profiles_2d[0].grid.dim2 = eq.Z_1D
    # time_slice.profiles_2d[0].r = eq.R
    # time_slice.profiles_2d[0].z = eq.Z
    time_slice.profiles_2d[0].psi = 2 * np.pi * eq.psi()
    # time_slice.profiles_2d[0].j_phi = eq._profiles.jtor
    # time_slice.profiles_2d[0].b_field_r = eq.Br(eq.R, eq.Z)
    # time_slice.profiles_2d[0].b_field_phi = eq.Btor(eq.R, eq.Z)
    # time_slice.profiles_2d[0].b_field_z = eq.Bz(eq.R, eq.Z)

    # # plasma poloidal flux
    # time_slice.profiles_2d[1].type.name = "Plasma poloidal flux field"
    # time_slice.profiles_2d[1].type.index = 0
    # time_slice.profiles_2d[1].grid_type.name = "Rectangular grid type"
    # time_slice.profiles_2d[1].grid_type.index = 1
    # time_slice.profiles_2d[1].grid_type.description = (
    #     "Here we use a rectangular grid with dims (R,Z)."
    # )
    # time_slice.profiles_2d[1].grid.dim1 = eq.R_1D
    # time_slice.profiles_2d[1].grid.dim2 = eq.Z_1D
    # time_slice.profiles_2d[1].r = eq.R
    # time_slice.profiles_2d[1].z = eq.Z
    # time_slice.profiles_2d[1].psi = 2 * np.pi * eq.plasma_psi
    # time_slice.profiles_2d[1].b_field_r = eq.plasmaBr(eq.R, eq.Z)
    # # time_slice.profiles_2d[1].b_field_phi = eq.Btor(eq.R,eq.Z)
    # time_slice.profiles_2d[1].b_field_z = eq.plasmaBz(eq.R, eq.Z)

    # # tokamak poloidal flux (from coils and passives)
    # time_slice.profiles_2d[2].type.name = "Plasma poloidal flux field"
    # time_slice.profiles_2d[2].type.index = 0
    # time_slice.profiles_2d[2].grid_type.name = "Rectangular grid type"
    # time_slice.profiles_2d[2].grid_type.index = 1
    # time_slice.profiles_2d[2].grid_type.description = (
    #     "Here we use a rectangular grid with dims (R,Z)."
    # )
    # time_slice.profiles_2d[2].grid.dim1 = eq.R_1D
    # time_slice.profiles_2d[2].grid.dim2 = eq.Z_1D
    # time_slice.profiles_2d[2].r = eq.R
    # time_slice.profiles_2d[2].z = eq.Z
    # time_slice.profiles_2d[2].psi = 2 * np.pi * eq.tokamak_psi
    # Br_tokamak = 0.0
    # Bz_tokamak = 0.0
    # for name in eq.tokamak.coils_list:
    #     Br_tokamak += eq.tokamak[name].Br(eq.R, eq.Z)
    #     Bz_tokamak += eq.tokamak[name].Bz(eq.R, eq.Z)
    # time_slice.profiles_2d[2].b_field_r = Br_tokamak
    # # time_slice.profiles_2d[2].b_field_phi = eq.Btor(eq.R,eq.Z)
    # time_slice.profiles_2d[2].b_field_z = Bz_tokamak

    print("done.")

    return ids_out
