"""
Defines the FreeGSNKE equilibrium Object, which inherits from the FreeGS4E equilibrium object.

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

import os
import pickle

import freegs4e.equilibrium
import matplotlib.pyplot as plt
import numpy as np
from freegs4e import critical
from scipy import interpolate

from . import limiter_func
from .build_machine import copy_tokamak
from .copying import copy_into


class Equilibrium(freegs4e.equilibrium.Equilibrium):
    """FreeGS4E equilibrium class with optional initialization."""

    def __init__(self, *args, **kwargs):
        """Instantiates the object."""
        super().__init__(*args, **kwargs)

        self.equilibrium_path = os.environ.get("EQUILIBRIUM_PATH", None)
        if self.equilibrium_path is not None:
            self.initialize_from_equilibrium()

        # redefine interpolating function
        self.psi_func_interp = interpolate.RectBivariateSpline(
            self.R[:, 0], self.Z[0, :], self.plasma_psi
        )

        self.nxh = len(self.R) // 2
        self.nyh = len(self.Z[0]) // 2
        self.Rnxh = self.R[self.nxh, 0]
        self.Znyh = self.Z[0, self.nyh]

        # It's not a GS solution:
        self.solved = False

        # set up for limiter functionality
        self.limiter_handler = limiter_func.Limiter_handler(self, self.tokamak.limiter)
        self.mask_inside_limiter = 1.0 * self.limiter_handler.mask_inside_limiter
        # the factor 2 is needed by critical routines
        self.mask_outside_limiter = 2 * np.logical_not(self.mask_inside_limiter).astype(
            float
        )

    def update_machine_description(
        self,
        active_coils_data=None,
        passive_coils_data=None,
        limiter_data=None,
        wall_data=None,
        magnetic_probe_data=None,
        active_coils_path=None,
        passive_coils_path=None,
        limiter_path=None,
        wall_path=None,
        magnetic_probe_path=None,
        refine_mode="G",
        preserve_currents=True,
    ):
        """
        Update the equilibrium's tokamak directly and refresh machine-dependent caches.

        The computational domain, plasma current, and plasma flux are left in place.
        Coil Greens functions, tokamak flux, limiter masks, and probe descriptions
        are rebuilt for the updated machine.

        Parameters
        ----------
        active_coils_data : dict, optional
            Dictionary containing the active coil description.
        passive_coils_data : list, optional
            List containing passive structure descriptions. If omitted, no
            passive structures are added.
        limiter_data : list, optional
            List of limiter boundary points.
        wall_data : list, optional
            List of wall boundary points. If omitted, the limiter is used as the
            wall.
        magnetic_probe_data : dict, optional
            Dictionary containing magnetic probe descriptions.
        active_coils_path : str, optional
            Path to a pickle file containing the active coil description.
        passive_coils_path : str, optional
            Path to a pickle file containing passive structure descriptions.
        limiter_path : str, optional
            Path to a pickle file containing limiter boundary points.
        wall_path : str, optional
            Path to a pickle file containing wall boundary points.
        magnetic_probe_path : str, optional
            Path to a pickle file containing magnetic probe descriptions.
        refine_mode : str, optional
            Refinement mode for extended passive structures. Defaults to ``"G"``.
        preserve_currents : bool, optional
            If True, currents for labels that are present in both the old and new
            machine descriptions are copied onto the updated coil objects.

        Returns
        -------
        Equilibrium
            This equilibrium object, updated in place.

        Notes
        -----
        This method refreshes equilibrium-level data derived from the tokamak,
        but it does not update existing solver objects. Reinstantiate static or
        nonlinear solvers after changing machine geometry.
        """

        self.tokamak.set_machine_description(
            active_coils_data=active_coils_data,
            passive_coils_data=passive_coils_data,
            limiter_data=limiter_data,
            wall_data=wall_data,
            magnetic_probe_data=magnetic_probe_data,
            active_coils_path=active_coils_path,
            passive_coils_path=passive_coils_path,
            limiter_path=limiter_path,
            wall_path=wall_path,
            magnetic_probe_path=magnetic_probe_path,
            refine_mode=refine_mode,
            preserve_currents=preserve_currents,
        )
        self.refresh_machine_dependent_state()
        return self

    def update_active_coil(self, coil_name, active_coil_data, preserve_current=True):
        """
        Update one active coil/circuit and refresh equilibrium-level coil caches.

        Parameters
        ----------
        coil_name : str
            Existing active coil/circuit label to replace.
        active_coil_data : dict
            Machine-description entry for ``coil_name``.
        preserve_current : bool, optional
            If True, the old current on ``coil_name`` is copied onto the
            replacement coil/circuit. Defaults to True.

        Returns
        -------
        Equilibrium
            This equilibrium object, updated in place.

        Notes
        -----
        This method refreshes only the Greens functions affected by the updated
        coil when the coil ordering is unchanged. Limiter masks are reused
        because the limiter geometry is not modified by an active-coil update.
        Existing solver objects should still be reinstantiated after a geometry
        change because they cache machine-dependent matrices and mode data.
        """

        self.tokamak.update_active_coil(
            coil_name=coil_name,
            active_coil_data=active_coil_data,
            preserve_current=preserve_current,
        )
        self.refresh_machine_dependent_state(refresh_limiter=False)
        return self

    def refresh_machine_dependent_state(self, refresh_limiter=True):
        """
        Refresh cached data derived from the current tokamak description.

        Rebuilds the coil Greens functions on this equilibrium grid for changed
        coil labels only when possible, updates the tokamak flux from the
        refreshed Greens functions and current vector, and optionally recreates
        the limiter handler and inside/outside-limiter masks. A full Greens
        rebuild is used when the coil ordering or number of coils changed, or
        when no previous cached Greens functions are available.

        Parameters
        ----------
        refresh_limiter : bool, optional
            If True, rebuild the limiter handler and limiter masks. Set to False
            for coil-only updates that leave limiter geometry unchanged.

        Returns
        -------
        Equilibrium
            This equilibrium object, refreshed in place.
        """

        changed_coils = getattr(
            self.tokamak, "_last_machine_update_changed_coils", None
        )
        topology_changed = getattr(
            self.tokamak, "_last_machine_update_topology_changed", True
        )
        can_update_partially = (
            not topology_changed
            and changed_coils is not None
            and hasattr(self, "_pgreen")
            and hasattr(self, "_vgreen")
            and np.shape(self._vgreen)[0] == self.tokamak.n_coils
        )

        if can_update_partially:
            self._pgreen = self._pgreen.copy()
            self._vgreen = np.copy(self._vgreen)
            for label in changed_coils:
                if label not in self.tokamak.coil_order:
                    continue
                coil = self.tokamak[label]
                self._pgreen[label] = coil.createPsiGreens(self.R, self.Z)
                self._vgreen[self.tokamak.coil_order[label]] = coil.createPsiGreensVec(
                    self.R, self.Z
                )
        else:
            self._pgreen = self.tokamak.createPsiGreens(self.R, self.Z)
            self._vgreen = self.tokamak.createPsiGreensVec(self.R, self.Z)

        self.tokamak_psi = self.tokamak.calcPsiFromGreens(pgreen=self._pgreen)

        if refresh_limiter:
            self.limiter_handler = limiter_func.Limiter_handler(
                self, self.tokamak.limiter
            )
            self.mask_inside_limiter = 1.0 * self.limiter_handler.mask_inside_limiter
            self.mask_outside_limiter = 2 * np.logical_not(
                self.mask_inside_limiter
            ).astype(float)

        return self

    def create_auxiliary_equilibrium(self):
        """Creates the auxiliary equilibrium object.

        The auxiliary object returned from this method is essentially
        a copy of the equilibrium object (self) however it is manually
        setup and so won't contain all attributes on self (especially custom
        attributes). It is NOT _guaranteed_ to be the same as a deepcopy, or even
        a shallow copy.
        """
        # __new__ stops __init__ being called.
        # This is necessary because the __init__ method does expensive
        # calculations which we can just copy the results of
        equilibrium = Equilibrium.__new__(Equilibrium)

        # attributes that FreeGS4e sets
        equilibrium.tokamak = copy_tokamak(self.tokamak)
        equilibrium.Rmin = self.Rmin
        equilibrium.Rmax = self.Rmax
        equilibrium.Zmin = self.Zmin
        equilibrium.Zmax = self.Zmax
        equilibrium.nx = self.nx
        equilibrium.ny = self.ny
        equilibrium.dR = self.dR
        equilibrium.dZ = self.dZ
        equilibrium._applyBoundary = self._applyBoundary
        equilibrium._current = self._current
        equilibrium.order = self.order
        equilibrium._solver = self._solver

        # attributes the FreeGSNKE sets
        equilibrium.solved = self.solved
        equilibrium.psi_func_interp = self.psi_func_interp
        equilibrium.nxh = self.nxh
        equilibrium.nyh = self.nyh
        equilibrium.Rnxh = self.Rnxh
        equilibrium.Znyh = self.Znyh
        equilibrium.limiter_handler = self.limiter_handler  # should be safe not to copy

        # attributes that actually need to be copied
        equilibrium.R_1D = np.copy(self.R_1D)
        equilibrium.Z_1D = np.copy(self.Z_1D)
        equilibrium.R = np.copy(self.R)
        equilibrium.Z = np.copy(self.Z)
        equilibrium.tokamak_psi = np.copy(self.tokamak_psi)
        equilibrium.plasma_psi = np.copy(self.plasma_psi)
        equilibrium.psi_axis = np.copy(self.psi_axis)
        equilibrium.psi_bndry = np.copy(self.psi_bndry)
        equilibrium.mask_inside_limiter = np.copy(self.mask_inside_limiter)
        equilibrium.mask_outside_limiter = np.copy(self.mask_outside_limiter)
        equilibrium._pgreen = self._pgreen.copy()
        equilibrium._vgreen = self._vgreen.copy()
        copy_into(
            self,
            equilibrium,
            "flag_limiter",
            mutable=True,
            strict=False,
            allow_deepcopy=True,
        )
        copy_into(self, equilibrium, "has_relevant_xpoint", strict=False)
        copy_into(self, equilibrium, "current_vec", mutable=True, strict=False)

        copy_into(
            self, equilibrium, "opt", mutable=True, strict=False, allow_deepcopy=True
        )
        copy_into(
            self, equilibrium, "xpt", mutable=True, strict=False, allow_deepcopy=True
        )
        # copy_into(self, equilibrium, "psi_bndry", strict=False)

        if hasattr(self, "_profiles"):
            equilibrium._profiles = self._profiles.copy()

        return equilibrium

    def plot(
        self,
        axis=None,
        xpoints=True,
        opoints=True,
        wall=True,
        limiter=True,
        legend=False,
        show=True,
    ):
        """Plot a solved FreeGSNKE equilibrium.

        This overrides the FreeGS4E plotting wrapper so limited equilibria
        without a relevant X-point in the solution domain can still be plotted.
        In that case the LCFS is drawn from ``psi_bndry`` and no primary
        X-point separatrix is requested.
        """
        try:
            psi = self.psi()
            opt = self._profiles.opt
            xpt = self._profiles.xpt
        except AttributeError as e:
            raise RuntimeError(
                "This equilibrium has not been solved: please solve for an "
                "equilibrium first!"
            ) from e

        has_relevant_xpoint = getattr(
            self,
            "has_relevant_xpoint",
            getattr(self._profiles, "has_relevant_xpoint", len(xpt) > 0),
        )

        if axis is None:
            fig = plt.figure()
            axis = fig.add_subplot(111)
        axis.set_aspect("equal")
        axis.set_xlabel("Major radius [m]")
        axis.set_ylabel("Height [m]")

        levels = np.linspace(np.amin(psi), np.amax(psi), 35)
        axis.contour(self.R, self.Z, psi, levels=levels)

        if has_relevant_xpoint:
            colour = "r"
            style = "solid"
            axis.contour(
                self.R,
                self.Z,
                psi,
                levels=[xpt[0][2]],
                colors=colour,
                linestyles=style,
            )
            axis.plot(
                [],
                [],
                colour,
                label="Separatrix (primary X-point)",
                linestyle=style,
            )

        if self._profiles.flag_limiter:
            colour = "k"
            style = "dashed"
            axis.contour(
                self.R,
                self.Z,
                psi,
                levels=[self.psi_bndry],
                colors=colour,
                linestyles=style,
            )
            axis.plot([], [], colour, label="LCFS (limited plasma)", linestyle=style)

        if xpoints and has_relevant_xpoint:
            for r, z, _ in xpt:
                axis.plot(r, z, "rx", markersize=9)
            axis.plot(xpt[0][0], xpt[0][1], "rx", markersize=9, markeredgewidth=2.5)
            axis.plot([], [], "rx", markersize=9, label="X-points")
            axis.plot(
                [],
                [],
                "rx",
                markersize=9,
                markeredgewidth=2.5,
                label="X-point (primary)",
            )

        if opoints:
            for r, z, _ in opt:
                axis.plot(r, z, "g2", markersize=9)
            axis.plot([], [], "g2", markersize=9, label="O-points")

        if wall and self.tokamak.wall and len(self.tokamak.wall.R):
            axis.plot(
                list(self.tokamak.wall.R) + [self.tokamak.wall.R[0]],
                list(self.tokamak.wall.Z) + [self.tokamak.wall.Z[0]],
                "k",
            )

        if limiter and self.tokamak.limiter and len(self.tokamak.limiter.R):
            axis.plot(
                list(self.tokamak.limiter.R) + [self.tokamak.limiter.R[0]],
                list(self.tokamak.limiter.Z) + [self.tokamak.limiter.Z[0]],
                "k:",
            )

        if legend:
            axis.legend(loc="upper right")

        if show:
            plt.show()

        return axis

    def adjust_psi_plasma(
        self,
    ):
        """Operates an initial rescaling of the psi_plasma guess so to ensure a viable O-point
        and at least an X-point within the domain.

        Only use after appropriate coil currents have been set as desired!
        """
        self.tokamak_psi = self.tokamak.calcPsiFromGreens(pgreen=self._pgreen)

        n_up = 0
        self.gmod = 0
        self.gexp = 2
        opoint_flag = False
        while (n_up < 10) and (opoint_flag == False):
            try:
                # Analyse the equilibrium, finding O- and X-points
                opt, xpt = critical.find_critical(
                    self.R,
                    self.Z,
                    self.tokamak_psi + self.plasma_psi,
                    self.mask_inside_limiter,
                    None,
                )
                opoint_flag = True
            except:
                self.plasma_psi *= 1.5
                self.gmod += np.log(1.5)
                n_up += 1
        if opoint_flag == False:
            print("O-point could not be generated by simply scaling up psi_plasma.")
            print("Manual initialization advised.")
            return

        # O-point is in place
        xpoint_flag = len(xpt) > 0
        print("O-point is in place. Flag for X-point in place =", xpoint_flag)
        n_plasma_psi = self.plasma_psi.copy()
        n_exp = 0
        if (xpoint_flag == False) and (n_exp < 10):
            # if it didn't work, try by making psi more compact
            psi_max = np.amax(self.plasma_psi)
            e_plasma_psi = self.plasma_psi / psi_max
            while (xpoint_flag == False) and (n_exp < 10):
                n_exp += 1
                n_plasma_psi = psi_max * e_plasma_psi ** (n_exp * 1.1)
                try:
                    opt, xpt = critical.find_critical(
                        self.R,
                        self.Z,
                        self.tokamak_psi + n_plasma_psi,
                        self.mask_inside_limiter,
                        None,
                    )
                    xpoint_flag = len(xpt) > 0
                    self.gmod *= 1.1
                except:
                    # here if exponentiation causes the o-point to disappear
                    print(
                        "Failed to introduce an xpoint on the domain by exponentiating psi_plasma."
                    )
                    print("Manual initialization advised.")
                    return

        # assign from exponentiation if successful
        if xpoint_flag == False:
            print(
                "Failed to introduce an xpoint on the domain by exponentiating psi_plasma."
            )
            print("Manual initialization advised.")
        else:
            self.plasma_psi = n_plasma_psi.copy()

            n_up = 0

            # try to increase the size of the diverted mask
            diverted_core_mask = critical.inside_mask(
                self.R,
                self.Z,
                self.tokamak_psi + n_plasma_psi,
                opt,
                xpt,
            )
            limiter_size = np.sum(self.mask_inside_limiter)
            diverted_size = np.sum(diverted_core_mask)
            print("Size of the diverted core in number of domain pts =", diverted_size)

            diverted_flag = diverted_size > 0.5 * limiter_size
            while diverted_flag == False and n_up < 6:
                # try:
                opt, xpt = critical.find_critical(
                    self.R,
                    self.Z,
                    self.tokamak_psi + n_plasma_psi * 1.1,
                    self.mask_inside_limiter,
                    None,
                )
                xpoint_flag = len(xpt) > 0
                if xpoint_flag:
                    n_plasma_psi *= 1.15
                    self.gmod += np.log(1.15)
                    n_up += 1
                    diverted_core_mask = critical.inside_mask(
                        self.R,
                        self.Z,
                        self.tokamak_psi + n_plasma_psi,
                        opt,
                        xpt,
                    )
                    diverted_size = np.sum(diverted_core_mask)
                    print("diverted_size", diverted_size)
                # except:
                #     diverted_flag = True

        self.plasma_psi = n_plasma_psi.copy()

    def psi_func(self, R, Z, *args, **kwargs):
        """Scipy interpolation of plasma_psi function.
        Replaces the original FreeGS interpolation.
        It now includes a check which leads to the update of the interpolation when needed.

        Parameters
        ----------
        R : ndarray
            R coordinates where the interpolation is needed
        Z : ndarray
            Z coordinates where the interpolation is needed

        Returns
        -------
        ndarray
            Interpolated values of plasma_psi
        """
        check = (
            np.abs(
                np.max(self.psi_func_interp(self.Rnxh, self.Znyh))
                - self.plasma_psi[self.nxh, self.nyh]
            )
            > 1e-5
        )
        if check:
            print(
                "Dicrepancy between psi_func and plasma_psi detected. psi_func has been re-set."
            )
            # redefine interpolating function
            self.psi_func_interp = interpolate.RectBivariateSpline(
                self.R[:, 0], self.Z[0, :], self.plasma_psi
            )

        return self.psi_func_interp(R, Z, *args, **kwargs)

    def initialize_from_equilibrium(self):
        """
        This function loads a pickle file containing an initial guess for the plasma
        flux (and the corners of the grid points it is located on).

        Interpolation is carried out and mapped to the computational grid specified in the
        eq class.

        Parameters
        ----------

        Returns
        -------

        """

        # load the data from the pickle file
        with open(self.equilibrium_path, "rb") as f:
            data = pickle.load(f)

        # extract the data (will fail if not in this format)
        try:
            Rmin = data["Rmin"]
            Rmax = data["Rmax"]
            Zmin = data["Zmin"]
            Zmax = data["Zmax"]
            psi_plasma = data["psi_plasma"]
        except:
            raise ValueError(
                "Data in EQUILIBRIUM_PATH pickle not in correct format or missing."
            )

        # interpolate the plasma psi on the grid given in the data file
        plasma_psi_func = interpolate.RectBivariateSpline(
            np.linspace(Rmin, Rmax, psi_plasma.shape[0]),
            np.linspace(Zmin, Zmax, psi_plasma.shape[1]),
            psi_plasma,
        )

        # extract the values on the grid given in the eq object (this is the initial guess)
        self.plasma_psi = plasma_psi_func(self.R, self.Z, grid=False)

        print(
            "Initial guess for plasma flux initialised using file provided at EQUILIBRIUM_PATH."
        )
