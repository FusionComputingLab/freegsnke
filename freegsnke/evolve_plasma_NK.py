import numpy as np
from numpy.linalg import inv

import freegs

from . import MASTU_coils
from .MASTU_coils import coils_dict
from .MASTU_coils import coil_self_ind

import copy

from . import quants_for_emu
from . import evolve_currents

#from picardfast import fast_solve
from .newtonkrylov import NewtonKrylov

class evolve_plasma_NK:
    #interfaces the circuit equation with freeGS and executes dt evolution
    #uses NK solver rather than picard
    
    def __init__(self, profiles, eq):
        
        #instantiate solver on eq's domain
        self.NK = NewtonKrylov(eq)

        #profiles are the properties of the eq at handoff, 
        #paxis, fvac and alpha values are taken from there and kept fixed thereafter
        self.get_profiles_values(profiles)
        # self.paxis = profiles.paxis
        # self.fvac = profiles.fvac
        # self.alpha_m = profiles.alpha_m
        # self.alpha_n = profiles.alpha_n
        
        #eq needs only have same tokamak and grid structure as those that will be evolved
        #when calling time evolution on some initial eq, the eq object that is actually modified is eq1
        #with profiles1 for changes in the plasma current
        self.eq1 = freegs.Equilibrium(tokamak=eq.getMachine(),
                            Rmin=eq.R[0,0], Rmax=eq.R[-1,-1],    # Radial domain
                            Zmin=eq.Z[0,0], Zmax=eq.Z[-1,-1],    # Height range
                            nx=np.shape(eq.R)[0], ny=np.shape(eq.R)[1], # Number of grid points
                            psi = eq.plasma_psi)  
        self.profiles1 = freegs.jtor.ConstrainPaxisIp(self.paxis, # Plasma pressure on axis [Pascals]
                                            eq.plasmaCurrent(), # Plasma current [Amps]
                                            self.fvac, # vacuum f = R*Bt
                                            alpha_m = self.alpha_m,
                                            alpha_n = self.alpha_n) 
        for i,labeli in enumerate(coils_dict):
            self.eq1.tokamak[labeli].control = False

        #the numerical method used to solve for the time evolution uses iterations
        #the eq object that is modified during the iterations is eq2
        #with profiles2 for changes in the plasma current
        self.eq2 = freegs.Equilibrium(tokamak=eq.getMachine(),
                            Rmin=eq.R[0,0], Rmax=eq.R[-1,-1],    # Radial domain
                            Zmin=eq.Z[0,0], Zmax=eq.Z[-1,-1],    # Height range
                            nx=np.shape(eq.R)[0], ny=np.shape(eq.R)[1], # Number of grid points
                            psi = eq.plasma_psi)  
        self.profiles2 = freegs.jtor.ConstrainPaxisIp(self.paxis, # Plasma pressure on axis [Pascals]
                                            eq.plasmaCurrent(), # Plasma current [Amps]
                                            self.fvac, # vacuum f = R*Bt
                                            alpha_m = self.alpha_m,
                                            alpha_n = self.alpha_n)                     
        for i,labeli in enumerate(coils_dict):
            self.eq2.tokamak[labeli].control = False
        
        
       
        
        
        #qfe is an instance of the quants_for_emulation class
        #calculates fluxes of plasma on coils and on itself
        #as well as all other time-dependent quantities needed for time evolution
        self.qfe = quants_for_emu.quants_for_emulation(eq)
        
        #evol_currents is an instance of the evolve_currents class
        #contains Euler stepper
        self.evol_currents = evolve_currents.evolve_currents()
        self.Rmatrix = self.evol_currents.R_matrix
        

        #quantities for position of plasma
        #used to look if hitting the wall
        mask_inside_reactor = np.ones_like(eq.R)
        mask_inside_reactor *= (eq.R>eq.R[11,0])*(eq.R<eq.R[100,0])
        mask_inside_reactor *= (eq.Z<.96+1*eq.R)*(eq.Z>-.96-1*eq.R)
        mask_inside_reactor *= (eq.Z<-2+9.*eq.R)*(eq.Z>2-9.*eq.R)
        mask_inside_reactor *= (eq.Z<2.28-1.1*eq.R)*(eq.Z>-2.28+1.1*eq.R)
        mask_outside_reactor = 1-mask_inside_reactor
        self.mask_outside_reactor = mask_outside_reactor
        self.plasma_against_wall = 0

        
        # this just creates a vector with all currents
        # including coils and plasma
        # initial self.currents_vec values are taken from eq.tokamak
        n_coils = len(coils_dict.keys())
        currents_vec = np.zeros(n_coils+1)
        eq_currents = eq.tokamak.getCurrents()
        for i,labeli in enumerate(coils_dict.keys()):
            currents_vec[i] = eq_currents[labeli]
        currents_vec[-1] = eq.plasmaCurrent()
        self.currents_vec = currents_vec
        self.new_currents = currents_vec
        
        #self.npshape = np.shape(eq.plasma_psi)
        self.dt_step = 0

        # threshold to calculate rel_change in the currents to set value of dt_step
        # it may be useful to use different values for different coils and for passive/active structures later on
        self.threshold = 1000

        self.void_matrix = np.zeros((self.evol_currents.n_coils+1, self.evol_currents.n_coils+1))

    
    def get_profiles_values(self, profiles):
        #this allows to use the same instantiation of the time_evolution_class
        #on ICs with different paxis, fvac and alpha values
        #if these are different from those used when first instantiating the class
        #just call this function on the new profile object:
        self.paxis = profiles.paxis
        self.fvac = profiles.fvac
        self.alpha_m = profiles.alpha_m
        self.alpha_n = profiles.alpha_n


    def set_currents_eq1(self, eq):
        #sets currents and initial plasma_psi in eq1
        eq_currents = eq.tokamak.getCurrents()
        currents_vec = np.zeros(len(eq_currents)+1)
        for i,labeli in enumerate(coils_dict.keys()):
            currents_vec[i] = eq_currents[labeli]
        currents_vec[-1] = eq.plasmaCurrent()
        self.currents_vec = currents_vec
        self.eq1.plasma_psi = eq.plasma_psi.copy()


        
    def assign_currents_1(self, currents_vec):
        #uses currents_vec to assign currents to both plasma and tokamak in eq/profiles
        self.profiles1 = freegs.jtor.ConstrainPaxisIp(self.paxis, # Plasma pressure on axis [Pascals]
                                                    currents_vec[-1], # Plasma current [Amps]
                                                    self.fvac, # vacuum f = R*Bt
                                                    alpha_m = self.alpha_m,
                                                    alpha_n = self.alpha_n) 
        for i,labeli in enumerate(coils_dict):
            self.eq1.tokamak[labeli].current = currents_vec[i]
    def assign_currents_2(self, currents_vec):
        #uses currents_vec to assign currents to both plasma and tokamak in eq/profiles
        self.profiles2 = freegs.jtor.ConstrainPaxisIp(self.paxis, # Plasma pressure on axis [Pascals]
                                                    currents_vec[-1], # Plasma current [Amps]
                                                    self.fvac, # vacuum f = R*Bt
                                                    alpha_m = self.alpha_m,
                                                    alpha_n = self.alpha_n) 
        for i,labeli in enumerate(coils_dict):
            self.eq2.tokamak[labeli].current = currents_vec[i]

        
        
    def find_dt_evolve(self, U_active, max_rel_change, results=None, dR=0):
        #solves linearized circuit eq on eq1
        #currents at time t are as in eq1.tokamak and profiles1
        #progressively smaller timestep dt_step is used
        #up to achieving a relative change in the currents of max_rel_change
        #since some currents can cross 0, rel change is calculated with respect to 
        #either the current itself or a changeable threshold value, set in init as self.threshold
        #the new currents are in self.new_currents
        #this works using qfe quantities (i.e. inductances) based on eq1/profiles1
        if results==None:
            self.results = self.qfe.quants_out(self.eq1, self.profiles1)
        else: 
            self.results = results
        
        rel_change_curr = np.ones(5)
        dt_step = .001
        while np.sum(abs(rel_change_curr[:])>max_rel_change):
            dt_step /= 1.5
            new_currents = self.evol_currents.new_currents_out(self.eq1, 
                                                               self.results, 
                                                               U_active, 
                                                               dt_step,
                                                               dR)
            rel_change_curr = abs(new_currents-self.currents_vec)/self.vals_for_rel_change

        self.new_currents = new_currents
        self.dt_step = dt_step
       
    
    
    def do_step_LIdot_only(self, U_active, 
                      currents_vec=None, 
                      max_rel_change=.005, 
                      rtol=1e-7,
                      verbose=False):
        """Evolve the equilibrium one time step ignoring the dL/dt term.
        
        Returns
        -------
        flag : int
            1 if the plasma is impinging on the wall, 0 otherwise.
        """
        #flag to alert when hitting wall
        flag = 0
        
        if self.plasma_against_wall>0:
            print('plasma against the wall!')
            flag = 1
        else:
            self.assign_currents_1(currents_vec)
            self.call_NK(rtol, verbose)

            abscurrents = abs(self.currents_vec)       
            self.vals_for_rel_change = np.where(abscurrents>self.threshold, abscurrents, self.threshold)

            self.find_dt_evolve(U_active, max_rel_change)
            self.currents_vec = self.new_currents.copy()
        return(flag)


    def do_LIdot(self, U_active, 
                       currents_vec, 
                       max_rel_change=.005,
                       results=None,
                       dR=0):
        """finds first approx to currents I(t+dt) and sets it in self.trial_currents
        use linearized circuit equation assuming dL/dt=0, or using previous estimate of dL/dt
        assumes GS is already solved at time t for self.eq1, self.profiles1
        Finds an appropriate self.dt_step to get a relative change in the currents that is < max_rel_change"""
        
        self.assign_currents_1(currents_vec)
        self.find_dt_evolve(U_active, max_rel_change, results, dR)
        self.trial_currents = self.new_currents.copy()


    def update_R_matrix(self, trial_currents, rtol_NK=1e-7, verbose_NK=False):
        """calculates the matrix dL/dt using the previous estimate of I(t+dt)
        this is equivalent to a non diagonal resistance term, hence the name"""

        self.assign_currents_2(trial_currents)
        self.NK.solve(self.eq2, self.profiles2, rel_convergence=rtol_NK, verbose=verbose_NK)
        #calculate new fluxes and inductances
        self.results1 = self.qfe.quants_out(self.eq2, self.profiles2)
        
        dLpc = self.results1['plasma_ind_on_coils'] - self.results['plasma_ind_on_coils']
        dLpc /= self.dt_step

        dLcp = self.results1['plasma_coil_ind'] - self.results['plasma_coil_ind']
        dLcp /= self.dt_step
        
        dR = self.void_matrix.copy()
        dR[:,-1] = dLpc
        dR[-1,:-1] = dLcp

        return dR


   
    
   

    #root problem for the circuit equation for NK
    def Fcircuit(self,  U_active, 
                        trial_currents,
                        rtol_NK=1e-7,
                        verbose_NK=False):
        self.dR = self.update_R_matrix(trial_currents, rtol_NK, verbose_NK)
        new_currents = self.evol_currents.stepper(U_active, self.dt_step, dR=self.dR)
        return new_currents-trial_currents


    def dI(self, res0, clip=3):
    #solve the least sq problem in coeffs: min||G.coeffs+res0||^2
        self.coeffs = np.matmul(np.matmul(np.linalg.inv(np.matmul(self.G.T, self.G)),
                                     self.G.T), -res0)
        self.coeffs = np.clip(self.coeffs, -clip, clip)
        #get the associated step in I space
        di = np.sum(self.Q*self.coeffs[np.newaxis,:], axis=1)
        return di


    def Arnoldi_iter(self,      U_active, #active potential
                                trial_currents, #expansion point of the root problem function
                                vec_direction, #first direction for I basis
                                Fresidual=None, #residual of trial currents
                                n_k=3, #max number of basis vectors
                                conv_crit=.1, #add basis vector 
                                              #if orthogonal residual is larger than
                                grad_eps=.0001, #relative magnitude of infinitesimal step, with respect to self.vals_for_rel_change 
                                               #adjust in line to 0.1*max_rel_change
                                verbose_currents=False
                                ):

        n_k = min(n_k, self.evol_currents.n_coils)

        #basis in input space
        Q = np.zeros((self.evol_currents.n_coils+1, n_k+1))
        #orthogonal version of the basis above
        Qn = np.zeros((self.evol_currents.n_coils+1, n_k+1))
        #basis in grandient space
        G = np.zeros((self.evol_currents.n_coils+1, n_k+1))
        
        if Fresidual is None:
            Fresidual = self.Fcircuit(U_active, trial_currents)
        nFresidual = np.linalg.norm(Fresidual)
        
        n_it = 0
        #control on whether to add a new basis vector
        arnoldi_control = 1
        #use at least 1 orthogonal terms, but not more than n_k
        while ((n_it<1)+(arnoldi_control>0))*(n_it<n_k)>0:
            grad_coeff = min(max(grad_eps*min(self.vals_for_rel_change/abs(vec_direction)), .15), 1)

            Q[:,n_it] = vec_direction*grad_coeff
            Qn[:, n_it] = Q[:,n_it]/np.linalg.norm(Q[:,n_it])
            
            
            ri = self.Fcircuit(U_active, trial_currents + Q[:,n_it])
            vec_direction = (ri - Fresidual)
            if verbose_currents:
                print('trial dI = ',Q[:,n_it])
                print('associated residual to use = ', vec_direction)
            G[:,n_it] = vec_direction.copy()
            n_it += 1

            #orthogonalize residual 
            vec_direction -= np.sum(np.sum(Qn[:,:n_it]*vec_direction[:,np.newaxis], axis=0, keepdims=True)*Qn[:,:n_it], axis=1)
            #check if more terms are needed
            arnoldi_control = (np.linalg.norm(vec_direction)/nFresidual > conv_crit)

        #make both basis available
        self.G = G[:,:n_it]
        self.Q = Q[:,:n_it]


    def do_step(self,   U_active, #active potential
                        this_is_first_step, #flag to speed up computations if this is not the first step 
                                            #if it is first time step, then next input MUST be provided
                        eq, #eq for ICs, with all properties set up at time t=0, 
                            # ie with eq.tokamak.currents = I(t) and eq.plasmaCurrent = I_p(t) 
                            # it is assumed that eq has already gone through a GS solver
                            # i.e. that NK.solve(eq, profiles) has been already run  
                            # this is necessary to set the corrent eq.plasmaCurrent
                        max_rel_change=.005, #aim for a dt such that the maximum relative change in the currents is max_rel_change
                        rtol_NK=1e-7, #for convergence of the NK solver of GS
                        verbose_NK=False,

                        rtol_currents=5e-5, #for convergence of the circuit equation
                        verbose_currents=False,
                        max_iter=10, #if more iterative steps are required, dt is reduced
                        n_k=3, #maximum number of terms in Arnoldi expansion
                        conv_crit=1e-1, #add more Arnoldi terms if residual is still larger than
                        grad_eps=.00003): #relative magnitude of infinitesimal step, with respect to self.vals_for_rel_change 
                                        #adjust in line to 0.1*max_rel_change
                        
        """advances both plasma and currents according to complete linearized eq.
        Uses an NK iterative scheme to find consistent values of I(t+dt) and L(t+dt)
        Does not modify the input object eq, rather the evolved plasma is in self.eq1
        Outputs a flag that =1 if plasma is hitting the wall """
    
        if this_is_first_step>0:
            #prepare currents
            self.set_currents_eq1(eq)
            #calculate inductances in do_LIdot
            results = None
            self.dR = 0
        else:
            #use inductances calculated in previous time step
            results = self.results1.copy()
        
        abs_currents = abs(self.currents_vec)
        self.vals_for_rel_change = np.where(abs_currents>self.threshold, abs_currents, self.threshold)

        not_done_flag = 1
        while not_done_flag:
            self.do_LIdot(U_active, 
                          self.currents_vec, 
                          max_rel_change,
                          results,
                          self.dR)

            res_history = []        
            
            self.eq2.plasma_psi = self.eq1.plasma_psi.copy()
            Fresidual = self.Fcircuit(U_active, self.trial_currents, rtol_NK, verbose_NK)
            if verbose_currents:
                print('currents step from LdI/dt only = ', self.trial_currents-self.currents_vec)
                print('initial residual = ', Fresidual)
            rel_change = abs(Fresidual)/self.vals_for_rel_change            
            res_history.append(rel_change)
                
            it=0
            while it<max_iter and not_done_flag:
                if max(rel_change)<rtol_currents: 
                    not_done_flag=0
                else:
                    self.Arnoldi_iter(U_active,
                                        self.trial_currents, 
                                        Fresidual, #starting direction in I space
                                        Fresidual, #F(trial_currents) already calculated
                                        n_k, conv_crit, grad_eps, verbose_currents)
                    di = self.dI(Fresidual, clip=5)
                    self.trial_currents += di
                    Fresidual = self.Fcircuit(U_active, self.trial_currents)
                    rel_change = abs(Fresidual)/self.vals_for_rel_change
                    
                    
                    res_history.append(rel_change)
                    if verbose_currents:
                        print('resulting current step = ', di, ' coeffs=', self.coeffs)
                        print('new residual = ', Fresidual)
                        print('new rel_change = ', rel_change)
                        print('dt_step = ', self.dt_step)
                    
                    it += 1
            
            if it==max_iter:
                #if max_iter was hit, then message:
                print(f'failed to converge with less than {max_iter} iterations.') 
                print(f'Last rel_change={rel_change}.')
                print('Restarting with smaller timestep')
                max_rel_change /= 2
                grad_eps /= 2
        
        self.currents_vec = self.trial_currents.copy()
        self.eq1.plasma_psi = self.eq2.plasma_psi.copy()

        self.plasma_against_wall = np.sum(self.mask_outside_reactor*self.results1['separatrix'][1])
        return self.plasma_against_wall

        
    # def do_step_iterative(self,  U_active, 
    #                         eq, #only useful when setting initial conditions for the first step, for follwing steps set to self.eq1
    #                         max_rel_change=.002,
    #                         rtol_NK=1e-7,
    #                         verbose_NK=False,
    #                         blend = 0.75,
    #                         rtol_currents=1e-6,
    #                         verbose_currents=False,
    #                         max_iter=100):
    #     """purely iterative implementation of the circuit equation solver, including both LdI/dt and dL/dtI terms
    #     NK-implementation is superior in performance, hence use NK version at self.do_step"""

    #     if eq != self.eq1:
    #         #prepare currents
    #         self.set_currents_eq1(eq)
        
    #     abs_currents = abs(self.currents_vec)
    #     self.vals_for_rel_change = np.where(abs_currents>self.threshold, abs_currents, self.threshold)

    #     self.do_LIdot(U_active, 
    #                         self.currents_vec, 
    #                         max_rel_change,
    #                         )

    #     new_currents = self.trial_currents.copy()
    #     niter = 1
    #     control = [1]
    #     while np.sum(control)*(niter<max_iter):
    #         self.trial_currents = blend*self.trial_currents + (1-blend)*new_currents.copy()
    #         self.dR = self.update_R_matrix(self.trial_currents, rtol_NK, verbose_NK)
    #         new_currents = self.evol_currents.stepper(U_active, self.dt_step, dR=self.dR)
    #         rel_change = (abs(new_currents-self.trial_currents)/self.vals_for_rel_change)
    #         control = rel_change>rtol_currents
    #         if verbose_currents:
    #             print(rel_change)
    #         niter += 1
    #         if niter%10==0:
    #             blend = blend**.75
    #     if niter==max_iter:
    #         print(f'failed to converge in less than {max_iter} iterations. max_relative_change={max(rel_change)}')
