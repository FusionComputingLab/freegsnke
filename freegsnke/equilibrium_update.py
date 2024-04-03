import os
import pickle

import freegs
import numpy as np
from scipy import interpolate


class Equilibrium(freegs.equilibrium.Equilibrium):
    """FreeGS equilibrium class with optional initialization."""

    def __init__(self, *args, **kwargs):
        """Instantiates the object."""
        super().__init__(*args, **kwargs)

        self.equilibrium_path = os.environ.get("EQUILIBRIUM_PATH", None)
        self.reinitialize_from_file()

        # redefine interpolating function
        self.psi_func_interp = interpolate.RectBivariateSpline(
            self.R[:, 0], self.Z[0, :], self.plasma_psi
        )

        self.nxh = len(self.R) // 2
        self.nyh = len(self.Z[0]) // 2
        self.Rnxh = self.R[self.nxh, 0]
        self.Znyh = self.Z[0, self.nyh]

    def psi_func(self, R, Z, *args, **kwargs):
        """Scipy interpolation of plasma function.
        Replaces the original FreeGS interpolation.
        It now includes a check which leads to the update of the interpolation when needed.

        Parameters
        ----------
        R : _type_
            _description_
        Z : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        check = (
            np.abs(
                np.max(self.psi_func_interp(self.Rnxh, self.Znyh))
                - self.plasma_psi[self.nxh, self.nyh]
            )
            > 1e-5
        )
        print(
            "doing check!",
            np.max(self.psi_func_interp(self.Rnxh, self.Znyh)),
            self.plasma_psi[self.nxh, self.nyh],
            check,
        )
        if check:
            print("re-calculating!")
            # redefine interpolating function
            self.psi_func_interp = interpolate.RectBivariateSpline(
                self.R[:, 0], self.Z[0, :], self.plasma_psi
            )

        return self.psi_func_interp(R, Z, *args, **kwargs)

    def reinitialize_from_file(
        self,
    ):
        """Initializes the equilibrium with data from file."""
        if self.equilibrium_path is not None:
            self.initialize_from_equilibrium()
        else:
            print(
                "Equilibrium data was not provided. Proceeded with default initialization."
            )

    def initialize_from_equilibrium(
        self,
    ):
        """Executes the initialization if data from file is available"""

        with open(self.equilibrium_path, "rb") as f:
            equilibrium_data = pickle.load(f)

        coil_currents = equilibrium_data["coil_currents"]
        plasma_psi = equilibrium_data["plasma_psi"]

        # check that machine descriptions correspond
        # on file first
        coils_on_file = list(coil_currents.keys())
        # select active coils only
        active_coils_on_file = [coil for coil in coils_on_file if coil[:7] != "passive"]
        # in tokamak
        coils_in_tokamak = list((self.tokamak.getCurrents()).keys())
        # select active coils only
        active_coils_in_tokamak = [
            coil for coil in coils_in_tokamak if coil[:7] != "passive"
        ]
        if active_coils_on_file == active_coils_in_tokamak:
            # assign coil current values
            for coil in active_coils_in_tokamak:
                self.tokamak[coil].current = coil_currents[coil]

            # assign plasma_psi
            self.initialize_plasma_psi(plasma_psi)

            print(
                "Equilibrium initialised using file provided as part of the machine description."
            )

        else:
            print(
                "Although the machine description was provided with an equilibrium for initialization, this was not used as the coil set does not correspond."
            )

    def initialize_plasma_psi(self, plasma_psi):
        """Assigns the input plasma_psi to the equilibrium being instantiated.
        Checks and corrects any disagreements in the grid sizes.

        Parameters
        ----------
        plasma_psi : np.array
            plasma flux function to be used for the initialization
        """

        nx, ny = np.shape(self.plasma_psi)
        nx_file, ny_file = np.shape(plasma_psi)

        if (nx, ny) != (nx_file, ny_file):

            # assume solving domain was as in current equilibrium
            psi_func = interpolate.RectBivariateSpline(
                np.linspace(self.Rmin, self.Rmax, nx_file),
                np.linspace(self.Zmin, self.Zmax, ny_file),
                plasma_psi,
            )

            plasma_psi = psi_func(self.R, self.Z, grid=False)

        # note the factor 2 here. This moves the initialization away from being a GS solution
        # but this shift is helpful when performing glitch-y inverse solves
        self.plasma_psi = 2 * plasma_psi
