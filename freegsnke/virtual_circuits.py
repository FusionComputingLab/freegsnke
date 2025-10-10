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
    The class for storing/recording virtual circuits that have been built
    using the VirtualCircuitHandling class.
    """

    def __init__(
        self,
        name,
        eq,
        profiles,
        shape_matrix,
        VCs_matrix,
        targets,
        targets_val,
        targets_options,
        non_standard_targets,
        coils,
    ):
        """
        Store the key quantities from the VirtualCircuitHandling calculations.

        Parameters
        ----------
        name : str
            Name to call the VCs (e.g. super-X VCs).
        eq : object
            The equilibrium object used to build the VCs.
        profiles : object
            The profiles object used to build the VCs.
        shape_matrix : np.array
            The array storing the Jacobians between the targets and coils given in 'targets'
            and 'coils'.
        VCs_matrix : np.array
            The array storing the VCs between the targets and coils given in 'targets'
            and 'coils'.
        targets : list
            The list of targets used to calculate the shape_matrix and VCs_matrix.
        targets_val : np.array
            The array of target values.
        targets_options : dict
            Dictionary of additional parameters required to calculate the
            'targets'.
        non_standard_targets : list
            List of lists of additional (non-standard) target functions to use. Takes the
            form [["new_target_name",...], [function(eq),...]], where function calcualtes the target
            using the eq object.
        coils : list
            The list of coils used to calculate the shape_matrix and VCs_matrix.
        """

        self.name = name
        self.eq = eq
        self.profiles = profiles
        self.shape_matrix = shape_matrix
        self.VCs_matrix = VCs_matrix
        self.targets = targets
        self.targets_val = targets_val
        self.non_standard_targets = non_standard_targets
        self.coils = coils
        self.targets_options = targets_options
        if self.non_standard_targets is not None:
            self.len_non_standard_targets = len(self.non_standard_targets[0])
        else:
            self.len_non_standard_targets = 0


class VirtualCircuitHandling:
    """
    The virtual circuits handling class.
    """

    def __init__(
        self,
    ):
        """
        Initialises the virtual circuits.

        Parameters
        ----------

        """

        # name to store the VC under
        self.default_VC_name = "latest_VC"

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

    def calculate_targets(
        self, eq, targets, targets_options=None, non_standard_targets=None
    ):
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
            - "Rs_lower_outer": lower strikepoint (radial) position.
            - "Rs_upper_outer": upper strikepoint (radial) position.
        targets_options : dict
            Dictionary of additional parameters required to calculate the
            'targets'. Options are required for:
            - "Rx_lower": approx. (R,Z) position of the lower X-point.
            - "Zx_lower": approx. (R,Z) position of the lower X-point.
            - "Rx_upper": approx. (R,Z) position of the upper X-point.
            - "Zx_upper": approx. (R,Z) position of the upper X-point.
            - "Rs_lower_outer": approx. (R,Z) position of the lower outer strikepoint.
            - "Rs_upper_outer": approx. (R,Z) position of the upper outer strikepoint.
        non_standard_targets : list
            List of lists of additional (non-standard) target functions to use. Takes the
            form [["new_target_name",...], [function(eq),...]], where function calcualtes the target
            using the eq object.

        Returns
        -------
        list
            Returns the original list of targets (plus any additional tagets specified
            in non_standard_targets).
        np.array
            Returns a 1D array of the target values (in same order as 'targets' input).
        """

        # flag to ensure we calculate things once
        rinout_flag = False

        # set to empty
        if targets_options is None:
            targets_options = {}

        # outputting targets
        final_targets = deepcopy(targets)
        if non_standard_targets is None:
            target_vec = np.zeros(len(targets))
        else:
            target_vec = np.zeros(len(targets) + len(non_standard_targets[0]))

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

                if target in targets_options:
                    # (R,Z) location where the X-point should roughly be
                    loc = targets_options[target]

                    # find closest x-point to 'loc'
                    xpts = eq.xpt[:, 0:2]
                    x_point_ind = np.argmin(np.sum((xpts - loc) ** 2, axis=1))
                    target_vec[i] = xpts[x_point_ind, 0]
                else:
                    print(f"Use of the 'target_option' input for {target} is advised!")

                    # choose from first two X-points
                    xpts = eq.xpt[0:2, 0:2]
                    x_point_ind = np.argmin(xpts[:, 1])
                    target_vec[i] = xpts[x_point_ind, 0]

            # lower X-point (vertical) position
            elif target == "Zx_lower":

                if target in targets_options:
                    # (R,Z) location where the X-point should roughly be
                    loc = targets_options[target]

                    # find closest x-point to 'loc'
                    xpts = eq.xpt[:, 0:2]
                    x_point_ind = np.argmin(np.sum((xpts - loc) ** 2, axis=1))
                    target_vec[i] = xpts[x_point_ind, 1]
                else:
                    print(f"Use of the 'target_option' input for {target} is advised!")

                    # choose from first two X-points
                    xpts = eq.xpt[0:2, 0:2]
                    x_point_ind = np.argmin(xpts[:, 1])
                    target_vec[i] = xpts[x_point_ind, 1]

            # upper X-point (radial) position
            elif target == "Rx_upper":

                if target in targets_options:
                    # (R,Z) location where the X-point should roughly be
                    loc = targets_options[target]

                    # find closest x-point to 'loc'
                    xpts = eq.xpt[:, 0:2]
                    x_point_ind = np.argmin(np.sum((xpts - loc) ** 2, axis=1))
                    target_vec[i] = xpts[x_point_ind, 0]
                else:
                    print(f"Use of the 'target_option' input for {target} is advised!")

                    # choose from first two X-points
                    xpts = eq.xpt[0:2, 0:2]
                    x_point_ind = np.argmax(xpts[:, 1])
                    target_vec[i] = xpts[x_point_ind, 0]

            # upper X-point (vertical) position
            elif target == "Zx_upper":

                if target in targets_options:
                    # (R,Z) location where the X-point should roughly be
                    loc = targets_options[target]

                    # find closest x-point to 'loc'
                    xpts = eq.xpt[:, 0:2]
                    x_point_ind = np.argmin(np.sum((xpts - loc) ** 2, axis=1))
                    target_vec[i] = xpts[x_point_ind, 1]
                else:
                    print(f"Use of the 'target_option' input for {target} is advised!")

                    # choose from first two X-points
                    xpts = eq.xpt[0:2, 0:2]
                    x_point_ind = np.argmax(xpts[:, 1])
                    target_vec[i] = xpts[x_point_ind, 1]

            # lower outer strikepoint (radial) position
            elif target == "Rs_lower_outer":

                if target in targets_options:
                    # (R,Z) location where the strikepoint should roughly be
                    loc = targets_options[target]

                    # find closets strikepoint to 'loc'
                    strikes = eq.strikepoints()
                    strike_ind = np.argmin(np.sum((strikes - loc) ** 2, axis=1))
                    target_vec[i] = strikes[strike_ind, 0]
                else:
                    print(f"Use of the 'target_option' input for {target} is advised!")

                    # choose the (lower) strikepoint with the largest radial position
                    strikes = eq.strikepoints()
                    if strikes.shape[0] > 4:
                        print(
                            f"More than four strikepoints located, use of 'target_option' input for {target} is strongly advised!"
                        )
                    s_point_ind = strikes[strikes[:, 1] < 0]
                    print
                    target_vec[i] = s_point_ind[np.argmax(s_point_ind[:, 0]), 0]

            # upper outer strikepoint (radial) position
            elif target == "Rs_upper_outer":

                if target in targets_options:
                    # (R,Z) location where the strikepoint should roughly be
                    loc = targets_options[target]

                    # find closets strikepoint to 'loc'
                    strikes = eq.strikepoints()
                    strike_ind = np.argmin(np.sum((strikes - loc) ** 2, axis=1))
                    target_vec[i] = strikes[strike_ind, 0]
                else:
                    print(f"Use of the 'target_option' input for {target} is advised!")

                    # choose the (upper) strikepoint with the largest radial position
                    strikes = eq.strikepoints()
                    if strikes.shape[0] > 4:
                        print(
                            f"More than four strikepoints located, use of 'target_option' input for {target} is strongly advised!"
                        )
                    s_point_ind = strikes[strikes[:, 1] > 0]
                    target_vec[i] = s_point_ind[np.argmax(s_point_ind[:, 0]), 0]

            # catch undefined targets
            else:
                raise ValueError(f"Undefined target: {target}.")

        # add in extra target calculations
        if non_standard_targets is not None:
            final_targets += non_standard_targets[0]
            for j in range(0, len(non_standard_targets[0])):
                target_vec[(i + 1) + j] = non_standard_targets[1][j](eq)

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
            eq.tokamak.set_coil_current(coil, currents_vec[i])

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
        self.assign_currents(currents_vec, coils, eq=self._eq2)

        # solve for equilibrium
        self.solver.forward_solve(
            self._eq2,
            self._profiles2,
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
        self.assign_currents_solve_GS(currents, coils, self.target_relative_tolerance)

        # difference between plasma current vectors (before and after the solve)
        dIy_0 = self._eq2.limiter_handler.Iy_from_jtor(self._profiles2.jtor) - self.Iy

        # relative norm of plasma current change
        rel_ndIy_0 = np.linalg.norm(dIy_0) / self._nIy

        # scale the starting_dI to match the target
        final_dI = starting_dI * target_dIy / rel_ndIy_0

        # clip small/large currents
        final_dI = np.clip(final_dI, min_curr, max_curr)

        # store
        self.final_dI_record[j] = final_dI

    def build_dIydI_j(
        self,
        j,
        coils,
        targets,
        targets_options,
        non_standard_targets=None,
        verbose=False,
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
        targets_options : dict
            Dictionary of additional parameters required to calculate the 'targets' (see above).
        non_standard_targets : list
            List of lists of additional (non-standard) target functions to use. Takes the
            form [["new_target_name",...], [function(eq),...]], where function calcualtes the target
            using the eq object.
        verbose: bool
            Display output (or not).

        Returns
        -------
        None
            VC object modifed in place.
        """

        # store dI
        final_dI = 1.0 * self.final_dI_record[j]

        # copy of currents
        currents = np.copy(self.currents_vec)

        # perturb current
        currents[j] += final_dI

        # assign current to the coil and solve static GS problem
        self.assign_currents_solve_GS(currents, coils, self.target_relative_tolerance)

        # calculate finite difference of targets wrt to the coil current
        _, self._target_vec_1 = self.calculate_targets(
            self._eq2, targets, targets_options, non_standard_targets
        )
        dtargets = self._target_vec_1 - self._targets_vec
        # self._dtargetsdIj = dtargets / final_dI

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

        return dtargets / final_dI

    def calculate_VC(
        self,
        eq,
        profiles,
        coils,
        targets,
        targets_options,
        non_standard_targets=None,
        target_dIy=1e-3,
        starting_dI=None,
        min_starting_dI=50,
        verbose=False,
        VC_name=None,
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
        targets_options : dict
            Dictionary of additional parameters required to calculate the 'targets' (see above).
        non_standard_targets : list
            List of lists of additional (non-standard) target functions to use. Takes the
            form [["new_target_name",...], [function(eq),...]], where function calcualtes the target
            using the eq object.
        target_dIy : float
            Target value for the norm of delta(I_y), from which the finite difference derivative is calculated.
        starting_dI : float
            Initial value to be used as delta(I_j) to infer the slope of norm(delta(I_y))/delta(I_j).
        min_starting_dI : float
            Minimum starting_dI value to be used as delta(I_j): to infer the slope of norm(delta(I_y))/delta(I_j).
        verbose: bool
            Display output (or not).
        VC_name: str
            Name to store the VC under (in the 'VirtualCircuit' class).

        Returns
        -------
        None
            Modifies the class (and other private) object(s) in place.

        """

        # store original currents
        self.build_current_vec(eq, coils)

        # solve static GS problem (it's already solved?)
        self.solver.forward_solve(
            eq=eq,
            profiles=profiles,
            target_relative_tolerance=self.target_relative_tolerance,
        )

        # store the flattened plasma current vector (and its norm)
        self.Iy = eq.limiter_handler.Iy_from_jtor(profiles.jtor).copy()
        self._nIy = np.linalg.norm(self.Iy)

        # calculate the targets from the equilibrium
        targets_new, self._targets_vec = self.calculate_targets(
            eq, targets, targets_options, non_standard_targets
        )

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
        shape_matrix = np.zeros((len(targets_new), len(coils)))
        self.final_dI_record = np.zeros(len(coils))

        # make copies of the newly solved equilibrium and profile objects
        # these are used for all GS solves below
        self._eq2 = eq.create_auxiliary_equilibrium()
        self._profiles2 = profiles.copy()

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
            # each shape matrix row is derivative of targets wrt the final coil current change
            shape_matrix[:, j] = self.build_dIydI_j(
                j, coils, targets, targets_options, non_standard_targets, verbose
            )

        # store the data in its own (new) class
        if VC_name is None:
            VC_name = self.default_VC_name

        # store the VC object dynamically
        store_VC = VirtualCircuit(
            name=VC_name,
            eq=eq,
            profiles=profiles,
            shape_matrix=shape_matrix,
            VCs_matrix=np.linalg.pinv(
                shape_matrix
            ),  # "virtual circuits" are the pseudo-inverse of the shape matrix
            targets=targets_new,
            targets_val=self._targets_vec,
            targets_options=targets_options,
            non_standard_targets=non_standard_targets,
            coils=coils,
        )
        setattr(self, VC_name, store_VC)

        print("---")
        print("Shape and virtual circuit matrices built.")
        print(f"VC object stored under name: '{VC_name}'.")

    def apply_VC(
        self,
        eq,
        profiles,
        VC_object,
        all_requested_target_shifts,
        verbose=False,
    ):
        """
        Here we apply the VC matrix V to requested shifts in the target quantities (dT),
        obtaining the shift in the currents (in coils, dI) required to achieve this:

            dI = V * dT.

        Applying the current shifts to the existing currents, we
        re-solve the equilibrium and return to user.

        Parameters
        ----------
        eq : object
            The equilibrium object upon which to apply the VCs.
        profiles : object
            The profiles object upon which to apply the VCs.
        VC_object : an instance of the VirtualCircuit class
            Specifies the virtual circuit matrix and properties.
        all_requested_targets_shift : list
            List of floats containing the shifts in all of the relevant targets.
            Same order as VC_object.targets.
            Includes both standard and non-standard targets.
            Functions to calculate non-standard targets are sourced from the VC_object.
        verbose: bool
            Display output (or not).

        Returns
        -------
        object
            Returns the equilibrium object after applying the shifted currents.
        object
            Returns the profiles object after applying the shifted currents.
        list
            List of strings containing all of targets of interest.
        np.array
            Array of new target values.
        np.array
            Array of old target values.
        """

        # verify targets, coils, and shifts all match those used to generate VCs
        assert len(all_requested_target_shifts) == len(
            VC_object.targets_val
        ), "The vector of requested shifts does not match the list of targets associated with the supplied VC_object!"
        shifts = all_requested_target_shifts

        # calculate current shifts required using shape matrix (for stability)
        # uses least squares solver to solve S*dI = dT
        # where dT are the target shifts and dI the current shifts
        current_shifts = np.linalg.lstsq(
            VC_object.shape_matrix, np.array(shifts), rcond=None
        )[0]

        if verbose:
            print(f"Currents shifts from VCs:")
            print(f"{VC_object.coils} = {current_shifts}.")

        # re-solve static GS problem (to make sure it's solved already)
        self.solver.forward_solve(
            eq=eq,
            profiles=profiles,
            target_relative_tolerance=self.target_relative_tolerance,
        )

        # calculate the targets
        _, old_target_values = self.calculate_targets(
            eq,
            VC_object.targets[
                0 : len(VC_object.targets) - VC_object.len_non_standard_targets
            ],
            VC_object.targets_options,
            VC_object.non_standard_targets,
        )

        # store copies of the eq and profile objects
        eq_new = eq.create_auxiliary_equilibrium()
        profiles_new = profiles.copy()

        # assign currents to the required coils in the eq object
        new_currents = [
            eq_new.tokamak.getCurrents()[name] + current_shifts[i]
            for i, name in enumerate(VC_object.coils)
        ]
        self.assign_currents(new_currents, VC_object.coils, eq=eq_new)

        # solve for the new equilibrium
        self.solver.forward_solve(
            eq_new,
            profiles_new,
            target_relative_tolerance=self.target_relative_tolerance,
        )

        # calculate new target values and the difference vs. the old
        target_names, new_target_values = self.calculate_targets(
            eq_new,
            VC_object.targets[
                0 : len(VC_object.targets) - VC_object.len_non_standard_targets
            ],
            VC_object.targets_options,
            VC_object.non_standard_targets,
        )

        if verbose:
            print(f"Targets shifts from VCs:")
            print(f"{target_names} = {new_target_values - old_target_values}.")

        return eq_new, profiles_new, target_names, new_target_values, old_target_values
