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

import numpy as np
from copy import deepcopy

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
    
    def define_solver(self, 
                      solver, 
                      target_relative_tolerance=1e-7
                      ):
        """
        Sets the solver in the VC class. 
        
        Parameters
        ----------
        solver : object
            The static Grad-Shafranov solver object.
        target_relative_tolerance : float
            Target relative tolerance to be met. 
                
        Returns
        -------

        """
        
        self.solver = solver
        self.target_relative_tolerance = target_relative_tolerance

    def calculate_targets(self, eq, targets):
        """
        Sets the solver in the VC class. 
        
        Parameters
        ----------
        eq : object
            The equilibrium object.  
        targets : float
            Target relative tolerance to be met. 
                
        Returns
        -------

        """
        rinout_flag = False
        target_vec = np.zeros(len(targets))
        for i, target in enumerate(targets):
            if target == "Rin":
                if rinout_flag == False:
                    rin, rout = eq.innerOuterSeparatrix()
                    rinout_flag = True
                target_vec[i] = rin
            elif target == "Rout":
                if rinout_flag == False:
                    rin, rout = eq.innerOuterSeparatrix()
                    rinout_flag = True
                target_vec[i] = rout
            elif target == "Rxd":
                target_vec[i] = min(eq.xpt[:2, 0])
            elif target == "Zxd":
                target_vec[i] = min(eq.xpt[:2, 1])
            elif target == "Rs":
                target_vec[i] = eq.get_lower_strike()[0]
        return target_vec

    def build_current_vec(self, eq, coils):
        """_summary_

        Parameters
        ----------
        eq : _type_
            _description_
        coils : _type_
            _description_
        """
        self.currents_vec = np.zeros(len(coils))
        for i, coil in enumerate(coils):
            self.currents_vec[i] = eq.tokamak[coil].current

    def assign_currents(self, currents_vec, coils, eq):
        """_summary_

        Parameters
        ----------
        currents_vec : _type_
            _description_
        coils : _type_
            _description_
        eq : _type_
            _description_
        """
        for i, coil in enumerate(coils):
            eq.tokamak[coil].current = currents_vec[i]

    def assign_currents_solve_GS(self, currents_vec, coils, target_relative_tolerance):
        """Assigns current values as in input currents_vec to private self.eq2 and self.profiles2.
        Static GS problem is accordingly solved, which finds the associated plasma flux and current distribution.

        Parameters
        ----------
        currents_vec : np.array
            Input current values to be assigned. Format as in self.assign_currents.
        rtol_NK : float
            Relative tolerance to be used in the static GS problem.
        """
        self.assign_currents(currents_vec, coils, eq=self.eq2)
        self.solver.forward_solve(
            self.eq2,
            self.profiles2,
            target_relative_tolerance=target_relative_tolerance,
        )

    def prepare_build_dIydI_j(
        self, j, coils, target_dIy, starting_dI, min_curr=1e-4, max_curr=300
    ):
        """Prepares to compute the term d(Iy)/dI_j of the Jacobian by
        inferring the value of delta(I_j) corresponding to a change delta(I_y)
        with norm(delta(I_y))=target_dIy.

        Parameters
        ----------
        j : int
            Index identifying the current to be varied. Indexes as in self.currents_vec.
        target_relative_tolerance : float
            Relative tolerance to be used in the static GS problems.
        target_dIy : float
            Target value for the norm of delta(I_y), on which th finite difference derivative is calculated.
        starting_dI : float
            Initial value to be used as delta(I_j) to infer the slope of norm(delta(I_y))/delta(I_j).
        min_curr : float, optional, by default 1e-4
            If inferred current value is below min_curr, clip to min_curr.
        max_curr : int, optional, by default 300
            If inferred current value is above min_curr, clip to max_curr.
        """
        current_ = np.copy(self.currents_vec)
        current_[j] += starting_dI
        self.assign_currents_solve_GS(
            current_[j : j + 1], coils[j : j + 1], self.target_relative_tolerance
        )

        dIy_0 = (
            self.profiles2.limiter_handler.Iy_from_jtor(self.profiles2.jtor)
            - self.profiles.Iy
        )

        rel_ndIy_0 = np.linalg.norm(dIy_0) / self.nIy
        final_dI = starting_dI * target_dIy / rel_ndIy_0
        final_dI = np.clip(final_dI, min_curr, max_curr)
        self.final_dI_record[j] = final_dI

    def build_dIydI_j(self, j, coils, targets=None, verbose=False):
        """Computes the term d(Iy)/dI_j of the Jacobian as a finite difference derivative,
        using the value of delta(I_j) inferred earlier by self.prepare_build_dIydI_j.

        Parameters
        ----------
        j : int
            Index identifying the current to be varied. Indexes as in self.currents_vec.
        target_relative_tolerance : float
            Relative tolerance to be used in the static GS problems.

        Returns
        -------
        dIydIj : np.array finite difference derivative d(Iy)/dI_j.
            This is a 1d vector including all grid points in reduced domain, as from plasma_domain_mask.
        """

        final_dI = 1.0 * self.final_dI_record[j]

        current_ = np.copy(self.currents_vec)
        current_[j] += final_dI
        self.assign_currents_solve_GS(
            current_[j : j + 1], coils[j : j + 1], self.target_relative_tolerance
        )

        dIy_1 = (
            self.profiles2.limiter_handler.Iy_from_jtor(self.profiles2.jtor)
            - self.profiles.Iy
        )
        dIydIj = dIy_1 / final_dI

        if verbose:
            print(
                "dimension",
                j,
                "in the vector of metal currents,",
                "gradient calculated on the finite difference: norm(deltaI) = ",
                final_dI,
                ", norm(deltaIy) =",
                np.linalg.norm(dIy_1),
            )

        if targets is not None:
            self.target_vec_1 = self.calculate_targets(self.eq2, targets)
            dtargets = self.target_vec_1 - self.targets_vec
            self.dtargetsdIj = dtargets / final_dI

        return dIydIj

    def calculate_VC(
        self,
        eq,
        profiles,
        targets,
        coils,
        starting_dI=None,
        target_dIy=1e-3,
        verbose=False,
    ):
        """

        Parameters
        ----------
        eq :
        """
        self.eq = deepcopy(eq)
        self.profiles = deepcopy(profiles)

        self.build_current_vec(self.eq, coils)
        self.solver.forward_solve(
            eq=self.eq,
            profiles=self.profiles,
            target_relative_tolerance=self.target_relative_tolerance,
        )
        self.profiles.Iy = self.profiles.limiter_handler.Iy_from_jtor(
            self.profiles.jtor
        ).copy()
        self.nIy = np.linalg.norm(self.profiles.Iy)
        self.targets_vec = self.calculate_targets(self.eq, targets)

        self.eq2 = deepcopy(eq)
        self.profiles2 = deepcopy(profiles)

        if starting_dI is None:
            starting_dI = np.abs(self.currents_vec.copy()) * target_dIy
            starting_dI[:-1] = np.where(starting_dI[:-1] > 50, starting_dI[:-1], 50)

        print(
            "Linearising with respect to the currents - this may take a minute or two."
        )
        self.shape_matrix = np.zeros((len(targets), len(coils)))
        self.final_dI_record = np.zeros(len(coils))

        for j in np.arange(len(coils)):
            if verbose:
                print("direction", j, "initial current shift", starting_dI[j])
            self.prepare_build_dIydI_j(
                j, coils, self.target_relative_tolerance, target_dIy, starting_dI[j]
            )

        for j in np.arange(len(coils)):
            self.build_dIydI_j(j, coils, targets, verbose)
            self.shape_matrix[:, j] = self.dtargetsdIj

        self.VCs = np.linalg.pinv(self.shape_matrix)
