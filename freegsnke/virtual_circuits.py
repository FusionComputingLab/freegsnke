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
            Target relative tolerance to be met by the solver. 
                
        Returns
        -------
        None
            Modifies the class object in place. 
        """
        
        self.solver = solver
        self.target_relative_tolerance = target_relative_tolerance

    def calculate_targets(self, eq, targets):
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
            - "Rin": inner midplane radius.  
            - "Rout": outer midplane radius.  
            - "Rxd": lower X-point (radial) position.
            - "Zxd": lower X-point (vertical) position.
            - "Rs": lower (outer) strikepoint (radial) position.
                
        Returns
        -------
        np.array
            Returns a 1D array of the target values (in same order as 'targets' input). 
        """
        
        # flag to ensure we calculate Rin and Rout once only
        rinout_flag = False
        
        target_vec = np.zeros(len(targets))
        for i, target in enumerate(targets):
            
            # inner midplane radius
            if target == "Rin":
                if rinout_flag == False:
                    rin, rout = eq.innerOuterSeparatrix()
                    rinout_flag = True
                target_vec[i] = rin
                
            # outer midplane radius
            elif target == "Rout":
                if rinout_flag == False:
                    rin, rout = eq.innerOuterSeparatrix()
                    rinout_flag = True
                target_vec[i] = rout
                
            # lower X-point (radial) position
            elif target == "Rxd":
                target_vec[i] = min(eq.xpt[:2, 0])

            # lower X-point (vertical) position
            elif target == "Zxd":
                target_vec[i] = min(eq.xpt[:2, 1])
                
            # lower (outer) strikepoint (radial) position
            elif target == "Rs":
                target_vec[i] = eq.get_lower_strike()[0]

        return target_vec

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

    def build_dIydI_j(self, j, coils, targets=None, verbose=False):
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
            List of strings containing the targets of interest. Currently supported targets
            are:
            - "Rin": inner midplane radius.  
            - "Rout": outer midplane radius.  
            - "Rxd": lower X-point (radial) position.
            - "Zxd": lower X-point (vertical) position.
            - "Rs": lower (outer) strikepoint (radial) position.
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

        # print some output
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

        # calculate finite difference of targets wrt to the coil current
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
        Calculate the "virtual circuits" matrix:
        
            J^+ = (J^T J)^(-1) J^T,
        
        which is the Moore-Penrose pseudo-inverse of the shape (Jacobian) matrix J:

            J_ij = dT_i / dI_j.
            
        This represents the sensitivity of target parameters T_i to changes in coil 
        currents I_j.
       
       The virtual circuit currents correspond to the rows of J^+.
         
         
        Parameters
        ----------
        eq : object
            The equilibrium object.  
        profiles : object
            The profiles object.
        targets : list
            List of strings containing the targets of interest. Currently supported targets
            are:
            - "Rin": inner midplane radius.  
            - "Rout": outer midplane radius.  
            - "Rxd": lower X-point (radial) position.
            - "Zxd": lower X-point (vertical) position.
            - "Rs": lower (outer) strikepoint (radial) position.
        coils : list
            List of strings containing the names of the coil currents to be assigned.
        starting_dI : float
            Initial value to be used as delta(I_j) to infer the slope of norm(delta(I_y))/delta(I_j).
        target_dIy : float
            Target value for the norm of delta(I_y), from which the finite difference derivative is calculated.
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
        
        # solve static GS problem 
        self.solver.forward_solve(
            eq=self.eq,
            profiles=self.profiles,
            target_relative_tolerance=self.target_relative_tolerance,
        )
        
        # store the flatten plasma current vector (and its norm)
        self.profiles.Iy = self.profiles.limiter_handler.Iy_from_jtor(
            self.profiles.jtor
        ).copy()
        self.nIy = np.linalg.norm(self.profiles.Iy)
        
        # calculate the targets from the equilibrium
        self.targets_vec = self.calculate_targets(self.eq, targets)

        # make copies of the newly solved equilibrium and profile objects
        self.eq2 = deepcopy(eq)
        self.profiles2 = deepcopy(profiles)

        # define starting_dI using currents and target_dIy if not given
        if starting_dI is None:
            starting_dI = np.abs(self.currents_vec.copy()) * target_dIy
            starting_dI[:-1] = np.where(starting_dI[:-1] > 50, starting_dI[:-1], 50) # clip some values

        print("Preparing and building the Jacobians wrt the coil currents - this may take a minute or two.")
        
        # storage matrices
        self.shape_matrix = np.zeros((len(targets), len(coils)))
        self.final_dI_record = np.zeros(len(coils))

        # for each coil, prepare by inferring delta(I_j) corresponding to a change delta(I_y)
        # with norm(delta(I_y)) = target_dIy
        for j in np.arange(len(coils)):
            if verbose:
                print(f"Direction (coil) {j} with initial current shift {starting_dI[j]}.")
            self.prepare_build_dIydI_j(
                j, coils, self.target_relative_tolerance, target_dIy, starting_dI[j]
            )

        # for each coil, build the Jacobian using the value of delta(I_j) inferred earlier 
        # by self.prepare_build_dIydI_j.
        for j in np.arange(len(coils)):
            self.build_dIydI_j(j, coils, targets, verbose)
            
            # each shape matrix row is derivative of targets wrt the final coil current change
            self.shape_matrix[:, j] = self.dtargetsdIj

        # "virtual circuits" are the pseudo-inverse of the shape matrix
        self.VCs = np.linalg.pinv(self.shape_matrix)
