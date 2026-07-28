"""
Class to implement magnetic probes (flux loops and pick ups at the moment):
- sets up probe object, containing the types and locations of the probes
- methods to extract the 'measurements' by each probe from an equilibrium.

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

import numpy as np
from deepdiff import DeepDiff
from freegs4e.gradshafranov import Greens, GreensBr, GreensBz


class Probes:
    """
    Representation of magnetic diagnostics for a tokamak equilibrium.

    This class implements synthetic magnetic probe signals, including:

    - Flux loops: measure poloidal flux ψ(R, Z)
    - Pickup coils: measure magnetic field components projected onto a
      probe orientation vector (B · n̂)

    The signals are computed from contributions of both coil currents and
    plasma current distributions via precomputed Green's functions.

    Parameters
    ----------
    coils_dict : dict
        Dictionary describing active (and possibly passive) coils, including
        geometry and identifiers used for Green's function evaluation.

    magnetic_probe_data : dict, optional
        Dictionary containing pre-defined probe configurations. Must include
        keys:
            - "flux_loops"
            - "pickups"

    magnetic_probe_path : str, optional
        Path to a pickled file containing `magnetic_probe_data`.

    Notes
    -----
    Exactly one of `magnetic_probe_data` or `magnetic_probe_path` must be
    provided. If both are given, a ValueError is raised. If neither is
    provided, no probe geometry is initialised.

    Attributes
    ----------
    floops : dict
        Flux loop definitions (positions, identifiers, etc.).
    pickups : dict
        Pickup coil definitions (positions, orientations, etc.).
    coil_names : list of str
        Ordered list of coil identifiers.
    coils_dict : dict
        Copy of input coil dictionary.

    Other Attributes
    -----------------
    The class also stores multiple Green's function caches used to compute:

    - Flux loop signals from coils and plasma
    - Pickup coil signals (Br, Bz projections)
    - Combined diagnostic outputs

    These are populated during later setup routines.
    """

    def __init__(
        self,
        coils_dict,
        magnetic_probe_data,
        magnetic_probe_path,
    ):
        """
        Sets up the magnetic probes object if the required data is passed to it via
        'magnetic_probe_data' or 'magnetic_probe_path'.

        Parameters
        ----------
        coils_dict : dict
            Dictionary containing the active coil data.
        magnetic_probe_data : dict
            Dictionary containing the magnetic probes data.
        magnetic_probe_path : str
            Path to the pickle file containing the magnetic probe data.

        """

        # magnetic probes not strictly required
        if magnetic_probe_data is not None and magnetic_probe_path is not None:
            raise ValueError(
                "Provide only one of 'magnetic_probe_data' or 'magnetic_probe_path', not both."
            )
        elif magnetic_probe_data is None and magnetic_probe_path is None:
            print("Magnetic probes --> none provided.")
        else:
            if magnetic_probe_path is not None:
                with open(magnetic_probe_path, "rb") as f:
                    magnetic_probe_data = pickle.load(f)
                print("Magnetic probes --> built from pickle file.")
            else:
                print("Magnetic probes --> built from user-provided data.")

            self.floops = magnetic_probe_data["flux_loops"]
            self.pickups = magnetic_probe_data["pickups"]
            self.coil_names = list(coils_dict.keys())
            self.coils_dict = coils_dict

    def initialise_setup(self, eq):
        """
        Initialise probe geometry and precompute Green's functions for a given
        equilibrium configuration.

        This method validates compatibility between the probe set and the supplied
        equilibrium, then constructs all probe-related quantities (positions,
        orientations, and Green's function caches) required for synthetic
        diagnostic evaluation.

        Parameters
        ----------
        eq : Equilibrium
            Equilibrium object providing tokamak geometry, coil configuration,
            and (optionally) plasma state information.

        Raises
        ------
        AssertionError
            If the coil configuration in `eq.tokamak` does not match the coil
            configuration used to define the probes.

        Notes
        -----
        This routine performs expensive precomputation of Green's functions and
        should typically be called once per equilibrium setup.

        Flux Loops
        ----------
        - Extracts flux loop positions and ordering from `self.floops`
        - Computes coil contribution Green's functions:
            `greens_psi_coils_floops`
        - Computes plasma contribution Green's functions:
            `greens_psi_plasma_floops` (cached per equilibrium key)

        Pickup Coils
        -------------
        - Extracts pickup positions and orientation vectors
        - Computes coil contributions:
            `greens_br_coils_pickup`, `greens_bz_coils_pickup`
        - Computes oriented magnetic field responses:
            `greens_B_coils_oriented`, `greens_B_plasma_oriented`

        Caching
        -------
        All plasma-dependent Green's functions are stored in dictionaries keyed by
        an equilibrium identifier (`eq_key`) to avoid recomputation when possible.

        Returns
        -------
        None
            All results are stored in-place as object attributes.
        """

        check = DeepDiff(eq.tokamak.coils_dict, self.coils_dict) == {}
        if check is not True:
            raise AssertionError(
                "The supplied equilibrium uses a different tokamak. Probes values can not be computed."
            )

        eq_key = self.create_eq_key(eq)

        # FLUX LOOPS
        # positions, number of probes, ordering
        self.floop_pos = np.array([probe["position"] for probe in self.floops])
        self.number_floops = np.shape(self.floop_pos)[0]  # number of probes
        self.floop_order = [probe["name"] for probe in self.floops]

        # # Initilaise Greens functions Gpsi
        self.greens_psi_coils_floops = self.create_greens_psi_all_coils(eq, "floops")
        self.greens_psi_plasma_floops = {}
        self.greens_psi_plasma_floops[eq_key] = self.create_green_psi_plasma(
            eq, "floops"
        )

        # # PICKUP COILS
        # # Positions and orientations - 3d vectors of [R, theta, Z]
        self.pickup_pos = np.array([el["position"] for el in self.pickups])
        self.pickup_or = np.array([el["orientation_vector"] for el in self.pickups])
        self.number_pickups = np.shape(self.pickup_pos)[0]
        self.pickup_order = [probe["name"] for probe in self.pickups]

        # # Initialise greens functions for pickups
        self.greens_br_plasma_pickup, self.greens_bz_plasma_pickup = {}, {}
        self.greens_br_coils_pickup, self.greens_bz_coils_pickup = (
            self.greens_BrBz_all_coils(eq, "pickups")
        )

        self.greens_B_plasma_oriented = {}
        self.greens_B_plasma_oriented[eq_key] = self.create_greens_B_oriented_plasma(
            eq, "pickups"
        )
        self.greens_B_coils_oriented = self.create_greens_B_oriented_coils(
            eq, "pickups"
        )

    def get_coil_currents(self, eq):
        """
        Extract coil current values from an equilibrium object in a fixed ordering.

        This method builds a vector of coil currents corresponding to
        `self.coil_names`, ensuring a consistent ordering between the probe
        model and the equilibrium representation.

        Parameters
        ----------
        eq : Equilibrium
            Equilibrium object containing a `tokamak` attribute with coil objects
            indexed by name.

        Returns
        -------
        np.ndarray
            Array of coil currents ordered according to `self.coil_names`.

        Notes
        -----
        - The ordering of currents is determined by `self.coil_names`, which is
        fixed during probe initialisation.
        - Each coil is accessed via `eq.tokamak[label].current`.
        - This could be replaced by a vectorised call such as
        `eq.tokamak.getcurrents()` if ordering consistency is guaranteed.
        """
        array_of_coil_currents = np.zeros(len(self.coil_names))
        for i, label in enumerate(self.coil_names):
            array_of_coil_currents[i] = eq.tokamak[label].current

        # could use eq.tokamak.getcurrents() instead
        return array_of_coil_currents

    def get_plasma_current(self, eq):
        """
        Extract the toroidal plasma current distribution from an equilibrium object.

        The returned quantity is the grid-based plasma current density mapped to
        the physically valid domain (typically restricted by the limiter region).

        Parameters
        ----------
        eq : Equilibrium
            Equilibrium object containing the plasma current density profile and
            limiter geometry.

        Returns
        -------
        np.ndarray
            Toroidal current distribution mapped onto the limiter-constrained grid.

        Notes
        -----
        - The current density is taken from `eq._profiles.jtor`.
        - The mapping to circuit-relevant currents is performed via:
        `eq.limiter_handler.Iy_from_jtor(...)`
        - This ensures only physically valid (in-limiter) contributions are used.
        """
        return eq.limiter_handler.Iy_from_jtor(eq._profiles.jtor)

    def create_eq_key(self, eq):
        """
        Generate a hashable identifier for an equilibrium grid configuration.

        The key uniquely encodes the spatial domain and resolution of the
        equilibrium, and is used to cache and retrieve precomputed Green's
        functions associated with a specific grid.

        Parameters
        ----------
        eq : Equilibrium
            Equilibrium object containing grid metadata (domain bounds and
            discretisation).

        Returns
        -------
        tuple
            A hashable key of the form:
            (R_min, R_max, Z_min, Z_max, nx, ny)

        Notes
        -----
        - nx and ny are inferred from `eq.R_1D` and `eq.Z_1D`.
        - This key assumes that Green's functions are valid only when both
        spatial bounds and resolution match exactly.
        - Small floating-point differences in bounds will produce distinct keys.
        """
        nx, ny = len(eq.R_1D), len(eq.Z_1D)
        eq_key = (eq.Rmin, eq.Rmax, eq.Zmin, eq.Zmax, nx, ny)
        return eq_key

    def create_greens_psi_single_coil(self, eq, coil_key, probe="floops"):
        """
        Compute the Green's function contribution of a single coil to probe signals.

        This method evaluates the poloidal flux response ψ at all probe locations
        due to a unit current in a specified coil.

        Parameters
        ----------
        eq : Equilibrium
            Equilibrium object containing the tokamak model and coil definitions.
        coil_key : str
            Identifier of the coil in `eq.tokamak` for which the Green's function
            is computed.
        probe : {"floops"}, optional
            Type of diagnostic probe to evaluate. Currently only flux loops
            ("floops") are supported.

        Returns
        -------
        np.ndarray
            Array of ψ values at each probe location due to a unit current in the
            specified coil.

        Notes
        -----
        - Probe positions are taken from `self.floop_pos` when `probe="floops"`.
        - The computation is delegated to:
        `eq.tokamak[coil_key].controlPsi(R, Z)`
        - The result corresponds to a linear response (unit current assumption).
        """

        if probe == "floops":
            pos_R = self.floop_pos[:, 0]
            pos_Z = self.floop_pos[:, 1]

        greens_psi_coil = eq.tokamak[coil_key].controlPsi(pos_R, pos_Z)

        return greens_psi_coil

    def create_greens_psi_all_coils(self, eq, probe="floops"):
        """
        Compute the Green's function matrix relating all coils to all probe
        locations for poloidal flux measurements.

        This builds a full response matrix where each entry represents the
        contribution of a unit current in a coil to the flux measured at a
        specific diagnostic probe.

        Parameters
        ----------
        eq : Equilibrium
            Equilibrium object containing coil geometry and Green's function
            evaluation methods.
        probe : {"floops"}, optional
            Type of diagnostic probe. Currently only flux loops ("floops") are
            supported.

        Returns
        -------
        np.ndarray
            2D array of shape (n_coils, n_probes), where entry [i, j] gives the
            poloidal flux ψ at probe j due to a unit current in coil i.

        Notes
        -----
        - Each row corresponds to a coil in `self.coils_dict`.
        - Each column corresponds to a flux loop in `self.floop_pos`.
        - The matrix assumes linear superposition of coil contributions.
        - Computation is performed via repeated calls to
        `create_greens_psi_single_coil`.
        """

        array = np.array([]).reshape(0, self.number_floops)
        for key in self.coils_dict.keys():
            array = np.vstack(
                (array, self.create_greens_psi_single_coil(eq, key, probe))
            )
        return array

    def psi_floop_all_coils(self, eq, probe="floops"):
        """
        Compute the total poloidal flux at all flux loop locations due to all coils.

        This method forms the linear superposition of coil contributions using the
        precomputed Green's function matrix and the current coil state from the
        equilibrium.

        Parameters
        ----------
        eq : Equilibrium
            Equilibrium object providing the current coil state via `get_coil_currents`.
        probe : {"floops"}, optional
            Diagnostic type. Currently only flux loops ("floops") are supported.

        Returns
        -------
        np.ndarray
            Total poloidal flux ψ at each flux loop location due to all coils.

        Notes
        -----
        - Uses the Green's function matrix:
        `self.greens_psi_coils_floops`
        - Coil contributions are combined via linear superposition:
        ψ_total = Σ_i I_i G_i
        - The ordering of coils is defined by `self.coil_names`.
        """
        array_of_coil_currents = self.get_coil_currents(eq)
        if probe == "floops":
            greens = self.greens_psi_coils_floops

        psi_from_all_coils = np.sum(
            greens * array_of_coil_currents[:, np.newaxis], axis=0
        )
        # self.floop_psi = psi_from_all_coils
        return psi_from_all_coils

    def create_green_psi_plasma(self, eq, probe="floops"):
        """
        Compute the Green's function mapping plasma current density to probe
        measurements of poloidal flux ψ.

        This constructs the response of each diagnostic probe to unit plasma
        current sources distributed over the equilibrium grid (restricted to the
        limiter-allowed region).

        Parameters
        ----------
        eq : Equilibrium
            Equilibrium object containing the plasma current grid and limiter
            mapping.
        probe : {"floops"}, optional
            Diagnostic type. Currently only flux loops ("floops") are supported.

        Returns
        -------
        np.ndarray
            Green's function array mapping plasma current elements to flux loop
            measurements. Shape is typically (n_plasma_cells, n_probes).

        Notes
        -----
        - Plasma source points are taken from:
        `eq.limiter_handler.plasma_pts`
        - Probe positions are taken from `self.floop_pos`.
        - The returned object represents a linear operator for:
            ψ_probe = G_plasma → probe · I_plasma
        - Only in-limiter plasma points are included in the computation.
        """

        if probe == "floops":
            pos_R = self.floop_pos[:, 0]
            pos_Z = self.floop_pos[:, 1]

        #   only on the limiter domain pts
        greens = Greens(
            eq.limiter_handler.plasma_pts[:, 0, np.newaxis],
            eq.limiter_handler.plasma_pts[:, 1, np.newaxis],
            pos_R[np.newaxis, :],
            pos_Z[np.newaxis, :],
        )

        return greens

    def psi_from_plasma(self, eq, probe="floops"):
        """
        Compute the contribution of plasma current to poloidal flux measurements
        at diagnostic probes.

        This evaluates the linear mapping from distributed plasma current density
        to flux loop signals using precomputed or lazily generated Green's
        functions.

        Parameters
        ----------
        eq : Equilibrium
            Equilibrium object containing plasma current distribution and limiter
            mapping information.
        probe : {"floops"}, optional
            Diagnostic type. Currently only flux loops ("floops") are supported.

        Returns
        -------
        np.ndarray
            Contribution to poloidal flux ψ at each probe due to plasma current.

        Notes
        -----
        - Plasma current distribution is obtained via:
        `get_plasma_current(eq)`
        - Green's functions are cached per equilibrium grid using:
        `create_eq_key(eq)`
        - If no cached plasma Green's function exists for the given equilibrium,
        it is computed on demand via `create_green_psi_plasma`.
        - The final signal is computed by linear superposition:
            ψ = Σ_i G_i I_i
        """
        eq_key = self.create_eq_key(eq)
        plasma_current_distribution = self.get_plasma_current(eq)

        if probe == "floops":
            try:
                plasma_greens = self.greens_psi_plasma_floops[eq_key]
            except:
                #  add new greens functions to dictionary
                self.greens_psi_plasma_floops[eq_key] = self.create_green_psi_plasma(
                    eq, "floops"
                )
                print("new equilibrium grid - computed new greens functions")
                # use newly created dictionary element.
                plasma_greens = self.greens_psi_plasma_floops[eq_key]

        psi_from_plasma = np.sum(
            plasma_greens * plasma_current_distribution[:, np.newaxis], axis=0
        )
        return psi_from_plasma

    def calculate_fluxloop_value(self, eq):
        """
        Compute the total flux loop signals by combining coil and plasma
        contributions.

        This method returns the full synthetic flux loop measurement at all
        diagnostic locations by summing the magnetic response from coil currents
        and plasma current distribution.

        Parameters
        ----------
        eq : Equilibrium
            Equilibrium object containing both coil currents and plasma current
            density.

        Returns
        -------
        np.ndarray
            Total poloidal flux ψ at each flux loop location.

        Notes
        -----
        - Coil contribution:
        `psi_floop_all_coils(eq)`
        - Plasma contribution:
        `psi_from_plasma(eq)`
        - Assumes linear superposition of magnetic fields.
        """
        return self.psi_floop_all_coils(eq) + self.psi_from_plasma(eq)

    def create_greens_BrBz_single_coil(self, eq, coil_key, probe="pickups"):
        """
        Compute Green's functions for the magnetic field (Br, Bz) from a single coil,
        evaluated at a set of probe locations.

        This function:
        - Evaluates the radial (Br) and vertical (Bz) magnetic field contributions
        from all filaments in the specified coil.
        - Computes these contributions at the chosen probe positions (default: pickup coils).
        - Returns the Green's function matrices mapping coil filament currents
        to magnetic field components at the probe locations.

        Parameters
        ----------
        eq : object
            Equilibrium object containing the tokamak configuration.
        coil_key : str
            Key identifying the coil within `eq.tokamak`.
        probe : str, optional
            Location set where the field is evaluated.
            Currently only "pickups" is implemented (default).

        Returns
        -------
        greens_br_coil : ndarray
            Green's function matrix for Br at probe locations.
        greens_bz_coil : ndarray
            Green's function matrix for Bz at probe locations.
        """
        if probe == "pickups":
            pos_R = self.pickup_pos[:, 0]
            pos_Z = self.pickup_pos[:, 2]

        greens_br_coil = eq.tokamak[coil_key].controlBr(pos_R, pos_Z)
        greens_bz_coil = eq.tokamak[coil_key].controlBz(pos_R, pos_Z)

        return greens_br_coil, greens_bz_coil

    def greens_BrBz_all_coils(self, eq, probe="pickups"):
        """
        Compute Green's function matrices for (Br, Bz) contributions from all coils
        evaluated at a set of probe locations.

        This function assembles the coil-wise Green's functions into global matrices:
        - Each row corresponds to a coil in `self.coils_dict`
        - Each column corresponds to a probe location
        - Entries represent the magnetic field response (Br or Bz) at a probe due to a given coil

        Parameters
        ----------
        eq : object
            Equilibrium object containing the tokamak configuration.
        probe : str, optional
            Set of probe locations where fields are evaluated.
            Currently only "pickups" is implemented (default).

        Returns
        -------
        array_r : ndarray
            Global Green's function matrix for radial field component Br,
            shape (n_coils, n_probes).
        array_z : ndarray
            Global Green's function matrix for vertical field component Bz,
            shape (n_coils, n_probes).
        """
        if probe == "pickups":
            array_r = np.array([]).reshape(0, self.number_pickups)
            array_z = np.array([]).reshape(0, self.number_pickups)

        for key in self.coils_dict.keys():
            vals = self.create_greens_BrBz_single_coil(eq, key, probe)
            array_r = np.vstack((array_r, vals[0]))
            array_z = np.vstack((array_z, vals[1]))

        return array_r, array_z

    def create_greens_B_oriented_coils(self, eq, probe="pickups"):
        """
        Compute the directional Green's function for coils projected onto probe orientations.

        This function evaluates the magnetic field Green's functions (Br, Bz) from all coils
        and projects them onto the local orientation of each probe. This yields the component
        of the magnetic field aligned with the probe direction.

        Mathematically, for each coil i and probe j:
            G_oriented[i, j] = Br[i, j] * or_R[j] + Bz[i, j] * or_Z[j]

        where (or_R, or_Z) defines the unit orientation vector of the probe in (R, Z).

        Parameters
        ----------
        eq : object
            Equilibrium object containing the tokamak configuration.
        probe : str, optional
            Probe set at which the field is evaluated.
            Currently only "pickups" is implemented (default).

        Returns
        -------
        prod : ndarray
            Oriented Green's function matrix of shape (n_coils, n_probes),
            representing the projection of (Br, Bz) onto probe orientations.
        """
        if probe == "pickups":
            or_R = self.pickup_or[:, 0]
            or_Z = self.pickup_or[:, 2]

        vals = self.greens_BrBz_all_coils(eq, probe)
        prod = vals[0] * or_R + vals[1] * or_Z

        return prod

    def BrBz_coils(self, eq, probe="pickups"):
        """
        Compute magnetic field components (Br, Bz) produced by all coils
        at a set of probe locations.

        The fields are obtained by contracting precomputed Green's functions
        with the coil current vector:

            Br[j] = sum_i G_br[i, j] * I_i
            Bz[j] = sum_i G_bz[i, j] * I_i

        where:
            - i indexes coils
            - j indexes probe locations
            - I_i is the current in coil i

        Parameters
        ----------
        eq : object
            Equilibrium object containing the tokamak configuration.
        probe : str, optional
            Probe set at which fields are evaluated.
            Currently only "pickups" is implemented (default).

        Returns
        -------
        br_coil : ndarray
            Radial magnetic field at probe locations, shape (n_probes,).
        bz_coil : ndarray
            Vertical magnetic field at probe locations, shape (n_probes,).
        """
        coil_currents = self.get_coil_currents(eq)[:, np.newaxis]
        if probe == "pickups":
            br_coil = np.sum(self.greens_br_coils_pickup * coil_currents, axis=0)
            bz_coil = np.sum(self.greens_bz_coils_pickup * coil_currents, axis=0)
        return br_coil, bz_coil

    def create_greens_BrBz_plasma(self, eq, probe="pickups"):
        """
        Compute Green's functions for magnetic field components (Br, Bz)
        produced by plasma current elements and evaluated at probe locations.

        This constructs the plasma-to-probe coupling matrices by evaluating
        the Biot–Savart kernel between discrete plasma grid points and probes.

        The resulting Green's functions map plasma current density (at grid points)
        to magnetic field at probes:

            Br[j] = sum_i G_br[i, j] * J_i
            Bz[j] = sum_i G_bz[i, j] * J_i

        where:
            - i indexes plasma grid / limiter-handler points
            - j indexes probe locations
            - J_i represents plasma current contribution at grid point i

        Parameters
        ----------
        eq : object
            Equilibrium object containing:
            - plasma grid points in `eq.limiter_handler.plasma_pts`
        probe : str, optional
            Probe set where fields are evaluated.
            Currently only "pickups" is implemented (default).

        Returns
        -------
        greens_br : object / ndarray
            Green's function for radial field component Br
            mapping plasma grid points to probes.
        greens_bz : object / ndarray
            Green's function for vertical field component Bz
            mapping plasma grid points to probes.
        """
        if probe == "pickups":
            pos_R = self.pickup_pos[:, 0]
            pos_Z = self.pickup_pos[:, 2]

        # rgrid = eq.R
        # zgrid = eq.Z

        greens_br = GreensBr(
            eq.limiter_handler.plasma_pts[:, 0, np.newaxis],
            eq.limiter_handler.plasma_pts[:, 1, np.newaxis],
            pos_R[np.newaxis, :],
            pos_Z[np.newaxis, :],
        )

        greens_bz = GreensBz(
            eq.limiter_handler.plasma_pts[:, 0, np.newaxis],
            eq.limiter_handler.plasma_pts[:, 1, np.newaxis],
            pos_R[np.newaxis, :],
            pos_Z[np.newaxis, :],
        )

        return greens_br, greens_bz

    def create_greens_B_oriented_plasma(self, eq, probe="pickups"):
        """
        Compute the oriented Green's function for plasma current contributions
        projected onto probe directions.

        This function first computes the plasma-to-probe Green's functions
        for the magnetic field components (Br, Bz), and then projects them
        onto the local orientation of each probe.

        The resulting kernel represents the magnetic field component aligned
        with the probe orientation:

            G_oriented[i, j] = Br[i, j] * or_R[j] + Bz[i, j] * or_Z[j]

        where:
            - i indexes plasma grid points
            - j indexes probe locations
            - (or_R, or_Z) is the unit orientation vector at each probe

        Parameters
        ----------
        eq : object
            Equilibrium object containing plasma grid information.
        probe : str, optional
            Probe set at which the field is evaluated.
            Currently only "pickups" is implemented (default).

        Returns
        -------
        prod : ndarray
            Oriented Green's function matrix mapping plasma currents
            to field components aligned with probe orientation.
        """
        br, bz = self.create_greens_BrBz_plasma(eq)

        or_R = self.pickup_or[:, 0]
        or_Z = self.pickup_or[:, 2]
        prod = br * or_R + bz * or_Z

        return prod

    def BrBz_plasma(self, eq, probe="pickups"):
        """
        Compute magnetic field components (Br, Bz) generated by plasma currents
        at a set of probe locations.

        This function uses precomputed (or lazily generated) Green's functions
        mapping plasma grid current elements to probe measurements, then contracts
        them with the plasma current distribution.

        The field is computed as:

            Br[j] = sum_i G_br[i, j] * I_i
            Bz[j] = sum_i G_bz[i, j] * I_i

        where:
            - i indexes plasma grid points
            - j indexes probe locations
            - I_i is the plasma current at grid point i

        The Green's functions are cached per equilibrium via `eq_key` to avoid
        recomputation when the plasma grid is unchanged.

        Parameters
        ----------
        eq : object
            Equilibrium object containing plasma current and grid information.
        probe : str, optional
            Probe set at which fields are evaluated.
            Currently only "pickups" is implemented (default).

        Returns
        -------
        br_plasma : ndarray
            Radial magnetic field at probe locations, shape (n_probes,).
        bz_plasma : ndarray
            Vertical magnetic field at probe locations, shape (n_probes,).
        """
        eq_key = self.create_eq_key(eq)
        plasma_current = self.get_plasma_current(eq)[:, np.newaxis]

        try:
            greens_br = self.greens_br_plasma_pickup[eq_key]
            greens_bz = self.greens_bz_plasma_pickup[eq_key]
        except:
            (
                self.greens_br_plasma_pickup[eq_key],
                self.greens_bz_plasma_pickup[eq_key],
            ) = self.create_greens_BrBz_plasma(eq, "pickups")
            print("new equilibrium grid - computed new greens functions")
        if probe == "pickups":
            br_plasma = np.sum(greens_br * plasma_current, axis=(0, 1))
            bz_plasma = np.sum(greens_bz * plasma_current, axis=(0, 1))
        return br_plasma, bz_plasma

    def Br(self, eq, probe="pickups"):
        """
        Compute the total radial magnetic field (Br) from both coils and plasma
        at a set of probe locations.

        The total field is the sum of contributions from discrete coils and the
        distributed plasma current:

            Br_total[j] = sum_i G_coil[i, j] * I_coil[i]
                        + sum_k G_plasma[k, j] * I_plasma[k]

        where:
            - i indexes coils
            - k indexes plasma grid points
            - j indexes probe locations

        Green's functions for the plasma contribution are cached per equilibrium
        using `eq_key` and computed lazily if not already available.

        Parameters
        ----------
        eq : object
            Equilibrium object containing coil and plasma current information,
            as well as grid definitions.
        probe : str, optional
            Probe set at which the field is evaluated.
            Currently only "pickups" is implemented (default).

        Returns
        -------
        br_total : ndarray
            Total radial magnetic field at probe locations, shape (n_probes,).
        """
        coil_currents = self.get_coil_currents(eq)[:, np.newaxis]
        plasma_current = self.get_plasma_current(eq)[:, np.newaxis]
        eq_key = self.create_eq_key(eq)

        if probe == "pickups":
            try:
                greens_pl = self.greens_br_plasma_pickup[eq_key]
            except:
                self.greens_br_plasma_pickup[eq_key] = self.create_greens_BrBz_plasma(
                    eq, "pickups"
                )[0]
                greens_pl = self.greens_br_plasma_pickup[eq_key]
                print("new equilibrium grid - computed new greens functions")
            br_coil = np.sum(self.greens_br_coils_pickup * coil_currents, axis=0)
            br_plasma = np.sum(greens_pl * plasma_current, axis=(0))
        return br_coil + br_plasma

    def Bz(self, eq, probe="pickups"):
        """
        Compute the total vertical magnetic field (Bz) from both coils and plasma
        at a set of probe locations.

        The total field is the sum of contributions from discrete coils and the
        distributed plasma current:

            Bz_total[j] = sum_i G_coil[i, j] * I_coil[i]
                        + sum_k G_plasma[k, j] * I_plasma[k]

        where:
            - i indexes coils
            - k indexes plasma grid points
            - j indexes probe locations

        Plasma Green's functions are cached per equilibrium using `eq_key` and
        are computed lazily if not already available.

        Parameters
        ----------
        eq : object
            Equilibrium object containing coil and plasma current information,
            as well as grid definitions.
        probe : str, optional
            Probe set at which the field is evaluated.
            Currently only "pickups" is implemented (default).

        Returns
        -------
        bz_total : ndarray
            Total vertical magnetic field at probe locations, shape (n_probes,).
        """
        coil_currents = self.get_coil_currents(eq)[:, np.newaxis]
        plasma_current = self.get_plasma_current(eq)[:, np.newaxis]
        eq_key = self.create_eq_key(eq)

        if probe == "pickups":
            try:
                greens_pl = self.greens_bz_plasma_pickup[eq_key]
            except:
                self.greens_bz_plasma_pickup[eq_key] = self.create_greens_BrBz_plasma(
                    eq, "pickups"
                )[1]
                greens_pl = self.greens_bz_plasma_pickup[eq_key]
                print("new equilibrium grid - computed new greens functions")

            bz_coil = np.sum(self.greens_bz_coils_pickup * coil_currents, axis=0)
            bz_plasma = np.sum(greens_pl * plasma_current, axis=(0))
        return bz_coil + bz_plasma

    def Btor(self, eq, probe="pickups"):
        """
        Compute the toroidal magnetic field (Btor) at probe locations.

        The toroidal field is assumed to follow the vacuum scaling law:

            Btor(R) = f_vac / R

        where:
            - f_vac is the vacuum toroidal field function from the equilibrium
            - R is the major radius of the probe location

        This approximation assumes probes are located outside the plasma,
        where the toroidal field is purely vacuum-like.

        Parameters
        ----------
        eq : object
            Equilibrium object providing the vacuum toroidal field profile
            via `eq._profiles.fvac()`.
        probe : str, optional
            Probe set at which the field is evaluated.
            Currently only "pickups" is implemented (default).

        Returns
        -------
        btor : ndarray
            Toroidal magnetic field at probe locations, shape (n_probes,).
        """
        if probe == "pickups":
            pos_R = self.pickup_pos[:, 0]

        btor = eq._profiles.fvac() / pos_R
        return btor

    def calculate_pickup_value(self, eq, probe="pickups"):
        """
        Compute the magnetic field projection (B · n) at pickup probes.

        This function evaluates the total magnetic field at pickup locations
        and projects it onto the local probe orientation vector n.

        The total pickup signal is composed of three contributions:

            (B · n)_j = (B_coil · n)_j + (B_plasma · n)_j + (B_tor · n)_j

        where:
            - j indexes pickup probes
            - coil and plasma terms are computed using precomputed oriented
            Green's functions
            - the toroidal field contribution is projected via the pickup
            orientation component in the toroidal direction

        Parameters
        ----------
        eq : object
            Equilibrium object containing coil currents, plasma currents,
            and magnetic field profiles.
        probe : str, optional
            Probe set at which the signal is evaluated.
            Currently only "pickups" is implemented (default).

        Returns
        -------
        signal : ndarray
            Scalar pickup signal (B · n) at each probe location,
            shape (n_probes,).
        """
        coil_current = self.get_coil_currents(eq)[:, np.newaxis]
        plasma_current = self.get_plasma_current(eq)[:, np.newaxis]
        eq_key = self.create_eq_key(eq)
        if probe == "pickups":
            try:
                greens_pl = self.greens_B_plasma_oriented[eq_key]
            except:
                #  add new greens functions to dictionary
                self.greens_B_plasma_oriented[eq_key] = (
                    self.create_greens_B_oriented_plasma(eq, "floops")
                )
                print("new equilibrium grid - computed new greens functions")
                # use newly created dictionary element.
                greens_pl = self.greens_B_plasma_oriented[eq_key]

            pickup_tor = self.Btor(eq, probe) * self.pickup_or[:, 1]
            pickup_pol_coil = np.sum(
                self.greens_B_coils_oriented * coil_current, axis=0
            )
            pickup_pol_pl = np.sum(greens_pl * plasma_current, axis=(0))

        return pickup_pol_coil + pickup_pol_pl + pickup_tor

    def plot(self, axis=None, show=True, floops=True, pickups=True, pickups_scale=0.05):
        """
        Plot magnetic diagnostic probes (fluxloops and pickup coils).

        This is a convenience wrapper around `freegs4e.plotting.plotProbes`.

        Parameters
        ----------
        axis : matplotlib.axes.Axes, optional
            Matplotlib axis to draw on. If None, a new figure/axis is created.
        show : bool, optional
            If True, calls `matplotlib.pyplot.show()` before returning.
        floops : bool, optional
            If True, plots fluxloop diagnostics.
        pickups : bool, optional
            If True, plots pickup coil diagnostics.
        pickups_scale : float, optional
            Scaling factor for visual representation of pickup coil orientation.

        Returns
        -------
        axis : matplotlib.axes.Axes
            Matplotlib axis containing the plotted probes.
        """
        from freegs4e.plotting import plotProbes

        return plotProbes(
            self,
            axis=axis,
            show=show,
            floops=floops,
            pickups=pickups,
            pickups_scale=pickups_scale,
        )
