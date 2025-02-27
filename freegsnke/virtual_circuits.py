"""
Defines class that represents the virtual circuits. 

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

from copy import deepcopy

import numpy as np


class VirtualCircuit:
    """
    The virtual circuits class.
    """

    def __init__(
        self,
    ):
        """
        Initialises the virtual circuits.

        Parameters
        ----------

        """

    def define_solver(self, solver, target_relative_tolerance=1e-7):
        """
        Sets the solver in the VC class.

        Parameters
        ----------
        solver : object
            The static Grad-Shafranov solver object.
        target_relative_tolerance : float
            Target relative tolerance to be met by the solver.

        Returns
        -------
        None
            Modifies the class object in place.
        """

        self.solver = solver
        self.target_relative_tolerance = target_relative_tolerance

    def calculate_targets(self, eq, targets, non_standard_targets=None):
        """
        For the given equilibrium, this function calculates the targets
        specified in the targets list.

        Parameters
        ----------
        eq : object
            The equilibrium object.
        targets : list
            List of strings containing the targets of interest. Currently supported targets
            are:
            - "R_in": inner midplane radius.
            - "R_out": outer midplane radius.
            - "Rx_lower": lower X-point (radial) position.
            - "Zx_lower": lower X-point (vertical) position.
            - "Rx_upper": upper X-point (radial) position.
            - "Zx_upper": upper X-point (vertical) position.
            - "Rs_lower_outer": lower (outer) strikepoint (radial) position.
            - "Rs_upper_outer": upper (outer) strikepoint (radial) position.
        non_standard_targets : list
            List of lists of additional (non-standard) target functions to use. Each sub-list
            takes the form ["new_target_name", function(eq)], where function calcualtes the target
            using the eq object.

        Returns
        -------
        list
            Returns the original list of targets (plus any additional tagets specified
            in non_standard_targets).
        np.array
            Returns a 1D array of the target values (in same order as 'targets' input).
        """

        # flag to ensure we calculate expensive things once
        rinout_flag = False
        xpt_flag = False

        # outputting targets
        final_targets = deepcopy(targets)
        if non_standard_targets is None:
            target_vec = np.zeros(len(targets))
        else:
            target_vec = np.zeros(len(targets) + len(non_standard_targets))

        for i, target in enumerate(targets):

            # inner midplane radius
            if target == "R_in":
                if rinout_flag == False:
                    rin, rout = eq.innerOuterSeparatrix()
                    rinout_flag = True
                target_vec[i] = rin

            # outer midplane radius
            elif target == "R_out":
                if rinout_flag == False:
                    rin, rout = eq.innerOuterSeparatrix()
                    rinout_flag = True
                target_vec[i] = rout

            # lower X-point (radial) position
            elif target == "Rx_lower":
                if xpt_flag == False:
                    xpts = eq.xpt[0:2, 0:2]
                    sorted_xpt = xpts[xpts[:, 1].argsort()]
                    xpt_flag = True
                target_vec[i] = sorted_xpt[0, 0]

            # lower X-point (vertical) position
            elif target == "Zx_lower":
                if xpt_flag == False:
                    xpts = eq.xpt[0:2, 0:2]
                    sorted_xpt = xpts[xpts[:, 1].argsort()]
                    xpt_flag = True
                target_vec[i] = sorted_xpt[0, 1]

            # upper X-point (radial) position
            elif target == "Rx_upper":
                if xpt_flag == False:
                    xpts = eq.xpt[0:2, 0:2]
                    sorted_xpt = xpts[xpts[:, 1].argsort()]
                    xpt_flag = True
                target_vec[i] = sorted_xpt[1, 0]

            # upper X-point (vertical) position
            elif target == "Zx_upper":
                if xpt_flag == False:
                    xpts = eq.xpt[0:2, 0:2]
                    sorted_xpt = xpts[xpts[:, 1].argsort()]
                    xpt_flag = True
                target_vec[i] = sorted_xpt[1, 1]

            # lower (outer) strikepoint (radial) position
            elif target == "Rs_lower_outer":
                target_vec[i] = eq.strikepoints(
                    quadrant="lower right", loc=(eq.xpt[0, 0], 0.0)
                )[0]

            # upper (outer) strikepoint (radial) position
            elif target == "Rs_upper_outer":
                target_vec[i] = eq.strikepoints(
                    quadrant="upper right", loc=(eq.xpt[0, 0], 0.0)
                )[0]

            # catch undefined targets
            else:
                raise ValueError(f"Undefined target: {target}.")

        # add in extra target calculations
        if non_standard_targets is not None:
            for j, target in enumerate(non_standard_targets):
                final_targets.append(target[0])
                target_vec[(i + 1) + j] = target[1](eq)

        return final_targets, target_vec

    def build_current_vec(self, eq, coils):
        """
        For the given equilibrium, this function stores the coil currents
        (for those listed in 'coils') in the class object.

        Parameters
        ----------
        eq : object
            The equilibrium object.
        coils : list
            List of strings containing the names of the coil currents to be stored.

        Returns
        -------
        None
            Modifies the class object in place.
        """

        # empty array for the currents
        self.currents_vec = np.zeros(len(coils))

        # set the currents
        for i, coil in enumerate(coils):
            self.currents_vec[i] = eq.tokamak[coil].current

    def assign_currents(self, currents_vec, coils, eq):
        """
        For the given equilibrium, this function assigns the coil currents
        (for those listed in 'coils') in the class object.

        Parameters
        ----------
        currents_vec : np.array
            Vector of coil currents to be assigned to the eq object using the coil
            names in 'coils.
        eq : object
            The equilibrium object.
        coils : list
            List of strings containing the names of the coil currents to be assigned.

        Returns
        -------
        None
            Modifies the class object in place.
        """

        # directly assign the currents
        for i, coil in enumerate(coils):
            eq.tokamak[coil].current = currents_vec[i]

    def assign_currents_solve_GS(self, currents_vec, coils, target_relative_tolerance):
        """
        Assigns the coil currents in 'currents_vec' to a private equilibrium object and
        then solve using the static GS solver.

        Parameters
        ----------
        currents_vec : np.array
            Input current values to be assigned. Format as in self.assign_currents.
        coils : list
            List of strings containing the names of the coil currents to be assigned.
        target_relative_tolerance : float
            Target relative tolerance to be met by the solver.

        Returns
        -------
        None
            Modifies the class (and other private) object(s) in place.
        """

        # assign currents
        self.assign_currents(currents_vec, coils, eq=self.eq2)

        # solve for equilibrium
        self.solver.forward_solve(
            self.eq2,
            self.profiles2,
            target_relative_tolerance=target_relative_tolerance,
        )

    def prepare_build_dIydI_j(
        self, j, coils, target_dIy, starting_dI, min_curr=1e-4, max_curr=300
    ):
        """
        Prepares to compute the term d(Iy)/dI_j of the Jacobian by
        inferring the value of delta(I_j) corresponding to a change delta(I_y)
        with norm(delta(I_y)) = target_dIy.

        Here:
            - Iy is the flattened vector of plasma currents (on the computational grid).
            - I_j is the current in the jth coil.

        Parameters
        ----------
        j : int
            Index identifying the current to be varied. Indexes as in self.currents_vec.
        coils : list
            List of strings containing the names of the coil currents to be assigned.
        target_dIy : float
            Target value for the norm of delta(I_y), from which the finite difference derivative is calculated.
        starting_dI : float
            Initial value to be used as delta(I_j) to infer the slope of norm(delta(I_y))/delta(I_j).
        min_curr : float, optional, by default 1e-4
            If inferred current value is below min_curr, clip to min_curr.
        max_curr : int, optional, by default 300
            If inferred current value is above max_curr, clip to max_curr.

        Returns
        -------
        None
            Modifies the class (and other private) object(s) in place.
        """

        # copy of currents
        currents = np.copy(self.currents_vec)

        # perturb current j
        currents[j] += starting_dI

        # assign current to the coil and solve static GS problem
        self.assign_currents_solve_GS(
            currents[j : j + 1], coils[j : j + 1], self.target_relative_tolerance
        )

        # difference between plasma current vectors (before and after the solve)
        dIy_0 = (
            self.profiles2.limiter_handler.Iy_from_jtor(self.profiles2.jtor)
            - self.profiles.Iy
        )

        # relative norm of plasma current change
        rel_ndIy_0 = np.linalg.norm(dIy_0) / self.nIy

        # scale the starting_dI to match the target
        final_dI = starting_dI * target_dIy / rel_ndIy_0

        # clip small/large currents
        final_dI = np.clip(final_dI, min_curr, max_curr)

        # store
        self.final_dI_record[j] = final_dI

    def build_dIydI_j(
        self, j, coils, targets=None, non_standard_targets=None, verbose=False
    ):
        """
        Computes the term d(Iy)/dI_j of the Jacobian as a finite difference derivative,
        using the value of delta(I_j) inferred earlier by self.prepare_build_dIydI_j.

        Here:
            - Iy is the flattened vector of plasma currents (on the computational grid).
            - I_j is the current in the jth coil.

        Parameters
        ----------
        j : int
            Index identifying the current to be varied. Indexes as in self.currents_vec.
        coils : list
            List of strings containing the names of the coil currents to be assigned.
        targets : list
            List of strings containing the targets of interest. See above for supported targets.
        non_standard_targets : list
            List of lists of additional (non-standard) target functions to use. Each sub-list
            takes the form ["new_target_name", function(eq)], where function calcualtes the target
            using the eq object.
        verbose: bool
            Display output (or not).

        Returns
        -------
        dIydIj : np.array
            Finite difference derivative d(Iy)/dI_j - this is a 1D vector over flattened vector
            of plasma currents (on the computational grid - reduced to the plasma_domain_mask).
        """

        # store dI
        final_dI = 1.0 * self.final_dI_record[j]

        # copy of currents
        currents = np.copy(self.currents_vec)

        # perturb current
        currents[j] += final_dI

        # assign current to the coil and solve static GS problem
        self.assign_currents_solve_GS(
            currents[j : j + 1], coils[j : j + 1], self.target_relative_tolerance
        )

        # difference between plasma current vectors (before and after the solve)
        dIy_1 = (
            self.profiles2.limiter_handler.Iy_from_jtor(self.profiles2.jtor)
            - self.profiles.Iy
        )

        # calculate the finite difference derivative
        dIydIj = dIy_1 / final_dI

        # calculate finite difference of targets wrt to the coil current
        if targets is not None:
            _, self.target_vec_1 = self.calculate_targets(
                self.eq2, targets, non_standard_targets
            )
            dtargets = self.target_vec_1 - self.targets_vec
            self.dtargetsdIj = dtargets / final_dI

        # print some output
        if verbose:
            print(f"{j}th coil ({coils[j]}) using scaled current shift {final_dI}.")
            # print(
            #     "Direction (coil)",
            #     j,
            #     ", gradient calculated on the finite difference: norm(deltaI) = ",
            #     final_dI,
            #     ", norm(deltaIy) =",
            #     np.linalg.norm(dIy_1),
            # )

        return dIydIj

    def calculate_VC(
        self,
        eq,
        profiles,
        coils,
        targets,
        non_standard_targets=None,
        target_dIy=1e-3,
        starting_dI=None,
        min_starting_dI=50,
        verbose=False,
    ):
        """
        Calculate the "virtual circuits" matrix:

            V = (S^T S)^(-1) S^T,

        which is the Moore-Penrose pseudo-inverse of the shape (Jacobian) matrix S:

            S_ij = dT_i / dI_j.

        This represents the sensitivity of target parameters T_i to changes in coil
        currents I_j.

        Parameters
        ----------
        eq : object
            The equilibrium object.
        profiles : object
            The profiles object.
        coils : list
            List of strings containing the names of the coil currents to be assigned.
        targets : list
            List of strings containing the targets of interest. See above for supported targets.
        non_standard_targets : list
            List of lists of additional (non-standard) target functions to use. Each sub-list
            takes the form ["new_target_name", function(eq)], where function calcualtes the target
            using the eq object.
        target_dIy : float
            Target value for the norm of delta(I_y), from which the finite difference derivative is calculated.
        starting_dI : float
            Initial value to be used as delta(I_j) to infer the slope of norm(delta(I_y))/delta(I_j).
        min_starting_dI : float
            Minimum starting_dI value to be used as delta(I_j): to infer the slope of norm(delta(I_y))/delta(I_j).
        verbose: bool
            Display output (or not).

        Returns
        -------
        None
            Modifies the class (and other private) object(s) in place.

        """

        # store copies of the eq and profile objects
        self.eq = deepcopy(eq)
        self.profiles = deepcopy(profiles)

        # store currents in VC object
        self.build_current_vec(self.eq, coils)

        # # solve static GS problem (it's already solved?)
        # self.solver.forward_solve(
        #     eq=self.eq,
        #     profiles=self.profiles,
        #     target_relative_tolerance=self.target_relative_tolerance,
        # )

        # store the flattened plasma current vector (and its norm)
        self.profiles.Iy = self.profiles.limiter_handler.Iy_from_jtor(
            self.profiles.jtor
        ).copy()
        self.nIy = np.linalg.norm(self.profiles.Iy)

        # calculate the targets from the equilibrium
        targets_new, self.targets_vec = self.calculate_targets(
            self.eq, targets, non_standard_targets
        )

        # make copies of the newly solved equilibrium and profile objects
        self.eq2 = deepcopy(eq)
        self.profiles2 = deepcopy(profiles)

        # define starting_dI using currents if not given
        if starting_dI is None:
            starting_dI = np.abs(self.currents_vec.copy()) * target_dIy
            starting_dI = np.where(
                starting_dI > min_starting_dI, starting_dI, min_starting_dI
            )

        if verbose:
            print("---")
            print("Preparing the scaled current shifts with respect to the:")

        # storage matrices
        self.shape_matrix = np.zeros((len(targets_new), len(coils)))
        self.final_dI_record = np.zeros(len(coils))

        # for each coil, prepare by inferring delta(I_j) corresponding to a change delta(I_y)
        # with norm(delta(I_y)) = target_dIy
        for j in np.arange(len(coils)):
            if verbose:
                print(
                    f"{j}th coil ({coils[j]}) using initial current shift {starting_dI[j]}."
                )
            self.prepare_build_dIydI_j(j, coils, target_dIy, starting_dI[j])

        if verbose:
            print("---")
            print("Building the shape matrix with respect to the:")

        # for each coil, build the Jacobian using the value of delta(I_j) inferred earlier
        # by self.prepare_build_dIydI_j.
        for j in np.arange(len(coils)):
            self.build_dIydI_j(j, coils, targets, non_standard_targets, verbose)

            # each shape matrix row is derivative of targets wrt the final coil current change
            self.shape_matrix[:, j] = self.dtargetsdIj

        # "virtual circuits" are the pseudo-inverse of the shape matrix
        self.VCs = np.linalg.pinv(self.shape_matrix)

        print("---")
        print("Shape and virtual circuit matrices built.")

    def apply_VC(
        self,
        coils,
        targets_shift,
        non_standard_targets_shift=None,
    ):
        """
        Here we apply the VC matrix V to requested shifts in the target quantities (dT),
        obtaining the shift in the currents (in coils, dI) required to achieve this:

            dI = V * dT.

        Applying the current shifts to the existing currents, we
        re-solve the equilibrium and return to user.

        Parameters
        ----------
        coils : list
            List of strings containing the names of the coil currents to be assigned.
        targets_shift : list
            List of floats containing the shifts in the targets of interest. See above for supported targets.
        non_standard_targets_shift : list
            List of floats of additional (non-standard) target shifts to use.

        Returns
        -------
        object
            Returns the equilibrium object after applying the shifted currents.
        object
            Returns the profiles object after applying the shifted currents.

        """

        # add in extra targets (if exist)
        if non_standard_targets_shift is not None:
            shifts = targets_shift + non_standard_targets_shift
        else:
            shifts = targets_shift

        # check dimensionalities
        assert (
            len(shifts) == self.VCs.shape[1]
        ), "No. of target shifts and no. of targets in VCs matrix do not match!"

        # calculate current shifts required using the VCs matrix
        current_shifts = self.VCs @ np.array(shifts)

        # store copies of the eq and profile objects
        eq = deepcopy(self.eq)
        profiles = deepcopy(self.profiles)

        # assign currents to the required coils in the eq object
        new_currents = [
            eq.tokamak.getCurrents()[name] + current_shifts[i]
            for i, name in enumerate(coils)
        ]
        self.assign_currents(new_currents, coils, eq=eq)

        # solve for the new equilibrium
        self.solver.forward_solve(
            eq,
            profiles,
            target_relative_tolerance=self.target_relative_tolerance,
        )

        return eq, profiles
