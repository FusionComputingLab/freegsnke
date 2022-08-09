import numpy as np
from numpy.linalg import inv

import freegs

from . import MASTU_coils
from .MASTU_coils import coils_dict

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
            self.eq1.tokamak[labeli].control = False
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
        self.len_currents = n_coils+1
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
        self.threshold = np.array([1000]*MASTU_coils.N_active
                                  +[3000]*(len(self.currents_vec)-MASTU_coils.N_active-1)
                                  +[1000])

        self.void_matrix = np.zeros((self.len_currents, self.len_currents))
        self.dummy_vec = np.zeros(self.len_currents)

        #calculate eigenvectors of time evolution:
        invm = np.linalg.inv(self.evol_currents.R_matrix[:-1,:-1]+MASTU_coils.coil_self_ind/.001)
        v, w = np.linalg.eig(np.matmul(invm, MASTU_coils.coil_self_ind/.001))
        w = w[:, np.argsort(-v)[:50]]
        mw = np.mean(w, axis = 0, keepdims = True)
        self.w = np.append(w, mw, axis=0)
        
        

    
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
        self.profiles1 = freegs.jtor.ConstrainPaxisIp(self.paxis, # Plasma pressure on axis [Pascals]
                                            eq.plasmaCurrent(), # Plasma current [Amps]
                                            self.fvac, # vacuum f = R*Bt
                                            alpha_m = self.alpha_m,
                                            alpha_n = self.alpha_n)
        self.eq1.plasma_psi = eq.plasma_psi.copy()
        self.eq2.plasma_psi = eq.plasma_psi.copy()


        
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
        dt_step = .002
        while np.sum(abs(rel_change_curr)>max_rel_change):
            dt_step /= 1.5
            new_currents = self.evol_currents.new_currents_out(self.eq1, 
                                                               self.results, 
                                                               U_active, 
                                                               dt_step,
                                                               dR)
            rel_change_curr = abs(new_currents-self.currents_vec)/self.vals_for_rel_change
        #print('find_dt_evolve dt = ', dt_step)

        #print('rel_change_currents = ', abs(rel_change_curr))
        self.new_currents = new_currents
        self.dt_step = dt_step
       
    


    
    # def do_step_LIdot_only(self, U_active, 
    #                   currents_vec=None, 
    #                   max_rel_change=.005, 
    #                   rtol=1e-7,
    #                   verbose=False):
    #     """Evolve the equilibrium one time step ignoring the dL/dt term.
        
    #     Returns
    #     -------
    #     flag : int
    #         1 if the plasma is impinging on the wall, 0 otherwise.
    #     """
    #     #flag to alert when hitting wall
    #     flag = 0
        
    #     if self.plasma_against_wall>0:
    #         print('plasma against the wall!')
    #         flag = 1
    #     else:
    #         self.assign_currents_1(currents_vec)
    #         self.call_NK(rtol, verbose)

    #         abscurrents = abs(self.currents_vec)       
    #         self.vals_for_rel_change = np.where(abscurrents>self.threshold, abscurrents, self.threshold)

    #         self.find_dt_evolve(U_active, max_rel_change)
    #         self.currents_vec = self.new_currents.copy()
    #     return(flag)


    # def do_LIdot(self, U_active, 
    #                    currents_vec, 
    #                    max_rel_change=.005,
    #                    results=None,
    #                    dR=0):
    #     """finds first approx to currents I(t+dt) and sets it in self.trial_currents
    #     use linearized circuit equation assuming dL/dt=0, or using previous estimate of dL/dt
    #     assumes GS is already solved at time t for self.eq1, self.profiles1
    #     Finds an appropriate self.dt_step to get a relative change in the currents that is < max_rel_change"""
        
    #     self.assign_currents_1(currents_vec)
    #     self.find_dt_evolve(U_active, max_rel_change, results, dR)
    #     self.trial_currents = self.new_currents.copy()


    # def do_LIdot(self, U_active, 
    #                    currents_vec, 
    #                    max_rel_change=.005,
    #                    results=None,
    #                    dR=0):
    #     """finds first approx to currents I(t+dt) and sets it in self.trial_currents
    #     use linearized circuit equation assuming dL/dt=0, or using previous estimate of dL/dt
    #     assumes GS is already solved at time t for self.eq1, self.profiles1
    #     Finds an appropriate self.dt_step to get a relative change in the currents that is < max_rel_change"""
        
    #     self.assign_currents_1(currents_vec)
    #     self.find_dt_evolve(U_active, max_rel_change, results, dR)
    #     self.trial_currents = self.new_currents.copy()


    def update_R_matrix(self, trial_currents, rtol_NK=1e-6):#, verbose_NK=False):
        """calculates the matrix dL/dt using the previous estimate of I(t+dt)
        this is equivalent to a non diagonal resistance term, hence the name"""

        self.assign_currents_2(trial_currents)
        self.NK.solve(self.eq2, self.profiles2, rel_convergence=rtol_NK) #verbose_NK)
        #calculate new fluxes and inductances
        self.results1 = self.qfe.quants_out(self.eq2, self.profiles2)
        #self.Lplus1 = self.results1['plasma_ind_on_coils']
        
        dLpc = self.results1['plasma_ind_on_coils'] - self.results['plasma_ind_on_coils']
        dLpc /= self.dt_step

        dLcp = self.results1['plasma_coil_ind'] - self.results['plasma_coil_ind']
        dLcp /= self.dt_step
        
        dR = self.void_matrix.copy()
        dR[:,-1] = dLpc
        dR[-1,:-1] = dLcp

        return dR

    # def dR_Lplasma(self, Lplus, Lnow):
    #     return (Lplus-Lnow)/self.dt_step

    #root problem for the circuit equation for NK
    # def Fcircuit_noadapt(self,  U_active, 
    #                     trial_currents,
    #                     rtol_NK=1e-7,
    #                   )#  verbose_NK=False):
    #     self.dR = self.update_R_matrix(trial_currents, rtol_NK)#, verbose_NK)
    #     new_currents = self.evol_currents.stepper(U_active, self.dt_step, dR=self.dR)
    #     return new_currents-trial_currents


    # #root problem for the circuit equation for NK
    # def Fcircuit0(self,  U_active, 
    #                     trial_currents,
    #                     rtol_NK=1e-7,
    #                   )#  verbose_NK=False):
    #     self._dR = self.update_R_matrix(trial_currents, rtol_NK)#, verbose_NK)
    #     new_currents = self.evol_currents.stepper_adapt_repeat(self.currents_vec, dR=self._dR)
    #     return new_currents-trial_currents

    def Fcircuit(self,  trial_currents,
                        rtol_NK=1e-7):
                      #,  verbose_NK=False):
        self._dR = self.update_R_matrix(trial_currents, rtol_NK)#, verbose_NK)
        new_currents = self.evol_currents.stepper_adapt_repeat(self.currents_vec, dR=self._dR)
        return (new_currents-trial_currents)/self.vals_for_rel_change


    def dI(self, res0, G, Q, clip=5):
    #solve the least sq problem in coeffs: min||G.coeffs+res0||^2
        self.coeffs = np.matmul(np.matmul(np.linalg.inv(np.matmul(G.T, G)),
                                          G.T), -res0)
        # print('intermediate_coeffs = ', self.coeffs)
        self.coeffs = np.clip(self.coeffs, -clip, clip)
        self.eplained_res = np.sum(G*self.coeffs[np.newaxis,:], axis=1)
        #get the associated step in I space
        self.di_Arnoldi = np.sum(Q*self.coeffs[np.newaxis,:], axis=1)
        


    # def Arnoldi_iter0(self,      U_active, #active potential
    #                             trial_currents, #expansion point of the root problem function
    #                             vec_direction, #first direction for I basis
    #                             Fresidual, #residual of trial currents
    #                             n_k=6, #max number of basis vectors, must be smaller than tot number of coils+1
    #                             conv_crit=.1, #add basis vector 
    #                                           #if orthogonal residual is larger than
    #                             grad_eps=.0001, #relative magnitude of infinitesimal step, with respect to self.vals_for_rel_change 
    #                                            #adjust in line to 0.1*max_rel_change
    #                             clip = 2,
    #                             verbose_currents=False
    #                             ):

    #     # clip_vec = np.array([clip])

    #     ntrial_currents = np.linalg.norm(self.trial_currents)
    #     nFresidual = np.linalg.norm(Fresidual)
    #     dFresidual = Fresidual/nFresidual
    #     print('nFresidual', nFresidual)

    #     #basis in input space
    #     Q = np.zeros((self.len_currents, n_k+1))
    #     #orthonormal version of the basis above
    #     Qn = np.zeros((self.len_currents, n_k+1))
    #     #basis in grandient space
    #     G = np.zeros((self.len_currents, n_k+1))
        
        
    #     n_it = 0
    #     #control on whether to add a new basis vector
    #     arnoldi_control = 1
    #     #use at least 1 orthogonal terms, but not more than n_k
    #     while ((n_it<1)+(arnoldi_control>0))*(n_it<n_k)>0:
    #         # grad_coeff = min(max(grad_eps*min(self.vals_for_rel_change/abs(vec_direction)), .2), 3)
    #         grad_coeff = grad_eps*ntrial_currents/np.linalg.norm(vec_direction)*nFresidual/1000
    #         candidate_di = vec_direction*grad_coeff
    #         solenoid_control = (abs(candidate_di[-1])<200)
    #         ri = self.Fcircuit(U_active, trial_currents + candidate_di)
    #         candidate_usable = ri - Fresidual
    #         ncandidate_usable = np.linalg.norm(candidate_usable)
    #         di_factor = ncandidate_usable/nFresidual

    #         if ((di_factor<.1)*solenoid_control+(di_factor>5)):
    #             print('using di factor = ', .5/di_factor)
    #             candidate_di *= (.5/di_factor)
    #             solenoid_control = (abs(candidate_di[-1])<200)
    #             ri = self.Fcircuit(U_active, trial_currents + candidate_di)
    #             candidate_usable = ri - Fresidual
    #             ncandidate_usable = np.linalg.norm(candidate_usable)
    #             di_factor = ncandidate_usable/nFresidual


    #         counter_ = 0
    #         while (di_factor<.1)*solenoid_control*(counter_<2):
    #             print('multiplying!')
    #             counter_ += 1
    #             candidate_di *= 2
    #             solenoid_control = (abs(candidate_di[-1])<200)
    #             ri = self.Fcircuit(U_active, trial_currents + candidate_di)
    #             candidate_usable = ri - Fresidual
    #             ncandidate_usable = np.linalg.norm(candidate_usable)
    #             di_factor = ncandidate_usable/nFresidual
    #         counter_ = 0
    #         while (di_factor>5)*(counter_<2):
    #             print('dividing!')
    #             counter_ += 1
    #             candidate_di /= 2
    #             ri = self.Fcircuit(U_active, trial_currents + candidate_di)
    #             candidate_usable = ri - Fresidual
    #             ncandidate_usable = np.linalg.norm(candidate_usable)
    #             di_factor = ncandidate_usable/nFresidual
        
    #         # dcandidate_usable = 
            
    #         # if np.sum(candidate_usable*Fresidual)/(nFresidual*ncandidate_usable)>.1:

            

            
            
    #         Q[:,n_it] = candidate_di.copy()
    #         print('dcurrent = ', Q[:,n_it])
    #         Qn[:,n_it] = Q[:,n_it]/np.linalg.norm(Q[:,n_it])
            
    #         vec_direction = candidate_usable.copy()
    #         print('usable/residual = ', vec_direction/Fresidual)
    #         G[:,n_it] = vec_direction
            
    #         n_it += 1

    #         #orthogonalize residual 
    #         vec_direction -= np.sum(np.sum(Qn*vec_direction[:,np.newaxis], axis=0, keepdims=True)*Qn, axis=1)
            
    #         #check if more terms are needed
    #         #arnoldi_control = (np.linalg.norm(vec_direction)/nFresidual > conv_crit)
    #         self.G = G[:,:n_it]
    #         self.Q = Q[:,:n_it]
    #         self.dI(Fresidual, G=self.G, Q=self.Q, clip=clip)
    #         rel_unexpl_res = np.linalg.norm(self.eplained_res+Fresidual)/nFresidual
    #         print('relative_unexplained_residual = ', rel_unexpl_res)
    #         arnoldi_control = (rel_unexpl_res > conv_crit)

    #         # #update clips
    #         # clip_vec = np.where(abs(self.coeffs)<clip, self.coeffs, clip)
    #         # clip_vec = np.append(clip_vec, [clip])

    #         # #to check failures
    #         # vec_a_trial = np.zeros((1, len(candidate_di)+3))
    #         # vec_a_trial[0,:-3] = candidate_di
    #         # vec_a_trial[0,-3] = n_it
    #         # vec_a_trial[0,-2] = rel_unexpl_res
    #         # self.arnoldi_trials = np.append(self.arnoldi_trials, vec_a_trial, axis=0)

    #     # # make both basis available
    #     # self.G = G[:,:n_it]
    #     # self.Q = Q[:,:n_it]
    #     return n_it



    def Arnoldi_iter(self,      trial_currents, #expansion point of the root problem function
                                vec_direction, #first direction for I basis
                                Fresidual, #residual of trial currents
                                n_k=10, #max number of basis vectors, must be smaller than tot number of coils+1
                                conv_crit=.5, #add basis vector 
                                              #if orthogonal residual is larger than
                                grad_eps=.001, #relative magnitude of infinitesimal step, with respect to self.vals_for_rel_change 
                                               #adjust in line to 0.1*max_rel_change
                                accept_threshold = .2,
                                max_collinearity = .6,
                                clip = 1.5
                                #,verbose_currents=False
                                ):

        # clip_vec = np.array([clip])

        ntrial_currents = np.linalg.norm(self.trial_currents)
        nFresidual = np.linalg.norm(Fresidual)
        dFresidual = Fresidual/nFresidual
        #print('nFresidual', nFresidual)

        #basis in input space
        Q = np.zeros((self.len_currents, n_k+1))
        #orthonormal version of the basis above
        Qn = np.zeros((self.len_currents, n_k+1))
        #basis in residual space
        G = np.zeros((self.len_currents, n_k+1))
        #normalized version of the basis above
        Gn = np.zeros((self.len_currents, n_k+1))
        
        n_it = 0
        #control on whether to add a new basis vector
        arnoldi_control = 1
        #use at least 1 orthogonal terms, but not more than n_k
        failure_count = 0
        while ((n_it<1)+(arnoldi_control>0))*(n_it<n_k)*((failure_count<2)+(n_it<1))>0:

            #print('failure_count', failure_count)
            nvec_direction = np.linalg.norm(vec_direction)
            grad_coeff = grad_eps*ntrial_currents/nvec_direction*nFresidual/.01
            candidate_di = vec_direction*min(grad_coeff, 200/abs(vec_direction[-1]))
            ri = self.Fcircuit(trial_currents + candidate_di)
            #print('trial currents in arnoldi', trial_currents)
            #internal_res = np.abs(ri)/self.vals_for_rel_change
            #print('internal residual = ', max(internal_res), np.argmax(internal_res), internal_res.mean())
            #print('all residual', internal_res)
            candidate_usable = ri - Fresidual
            ncandidate_usable = np.linalg.norm(candidate_usable)
            di_factor = ncandidate_usable/nFresidual
            #print('di_factor', di_factor)

            if ((di_factor<.3)*(abs(candidate_di[-1])<150))+(di_factor>6):
                print('using factor = ', 1/di_factor)
                candidate_di *= min(1/di_factor, 200/abs(candidate_di[-1]))
                ri = self.Fcircuit(trial_currents + candidate_di)
                candidate_usable = ri - Fresidual
                ncandidate_usable = np.linalg.norm(candidate_usable)
                di_factor = ncandidate_usable/nFresidual
            
            #print('dcurrent = ', candidate_di)
            #print('usable/residual = ', candidate_usable/Fresidual)
            dcandidate_usable = candidate_usable/ncandidate_usable
            costerm = abs(np.sum(dFresidual*dcandidate_usable)) 
            #print('costerm', costerm)

            if costerm>accept_threshold:
                collinearity = (np.sum(dcandidate_usable[:,np.newaxis]*Gn[:,:n_it], axis=0) > max_collinearity)
                if np.sum(collinearity):
                    #print('not accepting this term!, ', collinearity)
                    idx = np.random.randint(50)
                    vec_direction += nvec_direction*self.w[:, idx]
                    conv_crit = .95
                    #print('reshuffled', idx)
                    failure_count += 1


                else:
                    Q[:,n_it] = candidate_di.copy()
                    Qn[:,n_it] = Q[:,n_it]/np.linalg.norm(Q[:,n_it])
                    
                    G[:,n_it] = candidate_usable.copy()
                    Gn[:,n_it] = dcandidate_usable.copy()

                    n_it += 1
                    self.G = G[:,:n_it]
                    self.Q = Q[:,:n_it]
                    self.dI(Fresidual, G=self.G, Q=self.Q, clip=clip)
                    rel_unexpl_res = np.linalg.norm(self.eplained_res+Fresidual)/nFresidual
                    #print('relative_unexplained_residual = ', rel_unexpl_res)
                    arnoldi_control = (rel_unexpl_res > conv_crit)

                    vec_direction = candidate_usable*self.vals_for_rel_change
                    #vec_direction -= np.sum(np.sum(Qn[:,:n_it]*vec_direction[:,np.newaxis], axis=0, keepdims=True)*Qn[:,:n_it], axis=1)
            else: 
                #print('costerm too small!')
                idx = np.random.randint(50)
                vec_direction += nvec_direction*self.w[:, idx]
                grad_eps = .0001
                accept_threshold /= 1.2
                #print('reshuffled', idx)
                failure_count += 1

                
        return n_it


    # def do_step_noadapt(self,   U_active, # active potential
    #                     this_is_first_step, # flag to speed up computations if this is not the first step 
    #                                         # if it is first time step, then next input MUST be provided
    #                     eq, # eq for ICs, with all properties set up at time t=0, 
    #                         # ie with eq.tokamak.currents = I(t) and eq.plasmaCurrent = I_p(t) 
    #                         # it is assumed that eq has already gone through a GS solver
    #                         # i.e. that NK.solve(eq, profiles) has been already run  
    #                         # this is necessary to set the corrent eq.plasmaCurrent
    #                     max_rel_residual=.001, #aim for a dt such that the maximum relative change in the currents is max_rel_change
    #                     rtol_NK=1e-6, #for convergence of the NK solver of GS
    #                   )#  verbose_NK=False,

    #                     rtol_currents=5e-5, #for convergence of the circuit equation
    #                     verbose_currents=False,
    #                     max_iter=10, #if more iterative steps are required, dt is reduced
    #                     n_k=10, #maximum number of terms in Arnoldi expansion
    #                     conv_crit=5e-1, #add more Arnoldi terms if residual is still larger than
    #                     grad_eps=.00003): #relative magnitude of infinitesimal step, with respect to self.vals_for_rel_change 
    #                                       #adjust in line to 0.1*max_rel_change
                        
    #     """advances both plasma and currents according to complete linearized eq.
    #     Uses an NK iterative scheme to find consistent values of I(t+dt) and L(t+dt)
    #     Does not modify the input object eq, rather the evolved plasma is in self.eq1
    #     Outputs a flag that =1 if plasma is hitting the wall """
    
    #     if this_is_first_step>0:
    #         #prepare currents
    #         self.set_currents_eq1(eq)
    #         #calculate inductances in do_LIdot
    #         results = self.qfe.quants_out(self.eq1, self.profiles1)
    #         self.Lplus = results['plasma_ind_on_coils']
    #         self.dR = 0
    #     else:
    #         #use inductances calculated in previous time step
    #         results = self.results1.copy()
        
    #     abs_currents = abs(self.currents_vec)
    #     self.vals_for_rel_change = np.where(abs_currents>self.threshold, abs_currents, self.threshold)

    #     not_done_flag = 1
    #     while not_done_flag:
    #         curr_max_rel_change = .01
    #         self.do_LIdot(U_active, 
    #                       self.currents_vec, 
    #                       curr_max_rel_change,
    #                       results,
    #                       self.dR)
    #         self.eq2.plasma_psi = self.eq1.plasma_psi.copy()
    #         Fresidual = self.Fcircuit(U_active, self.trial_currents, rtol_NK, verbose_NK)
    #         rel_change = abs(Fresidual)/self.vals_for_rel_change
    #         max_rel_change = max(rel_change)
    #         if max_rel_change > max_rel_residual:
    #             curr_max_rel_change *= max_rel_residual/max_rel_change
    #             self.do_LIdot(U_active, 
    #                       self.currents_vec, 
    #                       curr_max_rel_change,
    #                       results,
    #                       self.dR)
    #             self.eq2.plasma_psi = self.eq1.plasma_psi.copy()
    #             Fresidual = self.Fcircuit(U_active, self.trial_currents, rtol_NK, verbose_NK)
    #             rel_change = abs(Fresidual)/self.vals_for_rel_change
    #             max_rel_change = max(rel_change)
    #         print('final timestep = ', self.dt_step)

    #         res_history = []          
    #         res_history.append(rel_change)

    #         if verbose_currents:
    #             #print('currents step from LdI/dt only = ', self.trial_currents-self.currents_vec)
    #             #print('initial residual = ', Fresidual)
    #             print('initial max_rel_change = ', max(rel_change), np.argmax(rel_change), np.mean(rel_change))
    #             #print('initial relative residual on inductances', max(abs(self.Lplus1-self.Lplus)/self.Lplus))
                
    #         it=0
    #         while it<max_iter and not_done_flag:
    #             if max(rel_change)<rtol_currents: 
    #                 not_done_flag=0
    #                 if verbose_currents:
    #                     print('Arnoldi iterations = ', it)
    #             else:
    #                 conv_crit = conv_crit/1.2
    #                 used_n = self.Arnoldi_iter(U_active,
    #                                     self.trial_currents, 
    #                                     Fresidual, #starting direction in I space
    #                                     Fresidual, #F(trial_currents) already calculated
    #                                     n_k, conv_crit, grad_eps, verbose_currents)
    #                 #self.dI(Fresidual, clip=2)
    #                 self.trial_currents += self.di_Arnoldi
    #                 #self.Lplus = self.Lplus1.copy()
    #                 Fresidual = self.Fcircuit(U_active, self.trial_currents)
    #                 rel_change = abs(Fresidual)/self.vals_for_rel_change
    #                 #check failures
    #                 # self.arnoldi_trials[-1,-1] = max(rel_change)
    #                 if max(rel_change)>max_rel_change:
    #                     print('degrading!')
                    
    #                 else: max_rel_change = max(rel_change)
                    
    #                 res_history.append(rel_change)
    #                 if verbose_currents:
    #                     print(' coeffs= ', self.coeffs, '; terms used= ', used_n)
    #                     #print('new residual = ', Fresidual)
    #                     print('new max_rel_change = ', max_rel_change, np.argmax(rel_change), rel_change.mean())
    #                     print('dt_step = ', self.dt_step)
    #                     #print('relative residual on inductances', max(abs(self.Lplus1-self.Lplus)/self.Lplus))
                    
    #                 it += 1
            
    #         if it==max_iter:
    #             #if max_iter was hit, then message:
    #             print(f'failed to converge with less than {max_iter} iterations.') 
    #             print(f'Last rel_change={rel_change}.')
    #             print('Restarting with smaller timestep')
    #             max_rel_change /= 2
    #             grad_eps /= 2
    #             this_is_first_step = 1
                


        
    #     self.currents_vec = self.trial_currents.copy()
    #     self.eq1.plasma_psi = self.eq2.plasma_psi.copy()

    #     self.plasma_against_wall = np.sum(self.mask_outside_reactor*self.results1['separatrix'][1])
    #     return self.plasma_against_wall

   




    def initialize_from_ICs(self, eq, profile): # eq for ICs, with all properties set up at time t=0, 
                            # ie with eq.tokamak.currents = I(t) and eq.plasmaCurrent = I_p(t) 
                            # it is assumed that eq has already gone through a GS solver
                            # i.e. that NK.solve(eq, profiles) has been already run  
                            # this is necessary to set the corrent eq.plasmaCurrent
        #ensure it's a GS solution
        self.NK.solve(eq, profile, rel_convergence=1e-8)#, verbose=False)
        #get profile parametrization
        self.get_profiles_values(profile)
        #prepare currents
        self.set_currents_eq1(eq)
        #calculate inductances 
        self.results = self.qfe.quants_out(self.eq1, self.profiles1)
        # self.Lplus = results['plasma_ind_on_coils']
        self.dR = 0
        self.dt_step = .001
        

    def do_step(self,  U_active, # active potential
                        max_rel_residual=.001, #aim for a dt such that the maximum relative change in the currents is max_rel_change
                        max_dt_step = .0002, #do multiple timesteps with constant \dotL if larger than
                        rtol_NK=1e-6, #for convergence of the NK solver of GS
                        #verbose_NK=False,

                        rtol_currents=3e-4, #for convergence of the circuit equation
                        verbose_currents=False,
                        max_iter=10, #if more iterative steps are required, dt is reduced
                        n_k=10, #maximum number of terms in Arnoldi expansion
                        conv_crit=.15, #add more Arnoldi terms if residual is still larger than
                        grad_eps=.00003,
                        clip=1.5): #relative magnitude of infinitesimal step, with respect to self.vals_for_rel_change 
                                          #adjust in line to 0.1*max_rel_change
                        
        """advances both plasma and currents according to complete linearized eq.
        Uses an NK iterative scheme to find consistent values of I(t+dt) and L(t+dt)
        Does not modify the input object eq, rather the evolved plasma is in self.eq1
        Outputs a flag that =1 if plasma is hitting the wall """
    
        # if this_is_first_step>0:
        #     #prepare currents
        #     self.set_currents_eq1(eq)
        #     #calculate inductances 
        #     results = self.qfe.quants_out(self.eq1, self.profiles1)
        #     # self.Lplus = results['plasma_ind_on_coils']
        #     self.dR = 0
        #     self.dt_step = .002
        # else:
        #     #use inductances calculated in previous time step
        #     results = self.results1.copy()
        
        abs_currents = abs(self.currents_vec)
        self.vals_for_rel_change = np.where(abs_currents>self.threshold, abs_currents, self.threshold)

        

        not_done_flag = 1
        while not_done_flag:
            self.evol_currents.initialize_time_t(self.results)
            self.evol_currents.determine_stepsize(self.dt_step, max_dt_step)
            self.trial_currents = self.evol_currents.stepper_adapt_first(self.currents_vec, U_active, self.dR)
            Fresidual = self.Fcircuit(self.trial_currents, rtol_NK)#, verbose_NK)
            rel_change = abs(Fresidual)
            max_rel_change = max(rel_change)
            if max_rel_change<.3*max_rel_residual:
                #print('increasing timestep')
                self.dt_step *= min(3,(.5*max_rel_residual/max_rel_change))
                self.evol_currents.determine_stepsize(self.dt_step, max_dt_step)
                self.trial_currents = self.evol_currents.stepper_adapt_repeat(self.currents_vec, self.dR)
                Fresidual = self.Fcircuit(self.trial_currents, rtol_NK)#, verbose_NK)
                rel_change = abs(Fresidual)
                max_rel_change = max(rel_change)
            while (max_rel_change > max_rel_residual)*(self.dt_step > 1e-5):
                #print('reducing timestep:', max_rel_change, self.dt_step, self.evol_currents.n_step, np.argmax(rel_change))
                self.dt_step *= (.75*max_rel_residual/max_rel_change)
                self.evol_currents.determine_stepsize(self.dt_step, max_dt_step)
                self.trial_currents = self.evol_currents.stepper_adapt_first(self.currents_vec, U_active, self.dR)
                Fresidual = self.Fcircuit(self.trial_currents, rtol_NK)#, verbose_NK)
                rel_change = abs(Fresidual)
                max_rel_change = max(rel_change)

            print(max_rel_change, self.dt_step, self.evol_currents.n_step, np.argmax(rel_change))
            # if verbose_currents:
            #     print('final timestep = ', self.dt_step)
            #     #print('currents step from LdI/dt only = ', self.trial_currents-self.currents_vec)
            #     #print('initial residual = ', Fresidual)
            #     print('initial max_rel_change = ', max_rel_change, np.argmax(rel_change), np.mean(rel_change))
            #     #print('initial relative residual on inductances', max(abs(self.Lplus1-self.Lplus)/self.Lplus))
                
            it=0
            while it<max_iter and not_done_flag:
                if max_rel_change<rtol_currents: 
                    not_done_flag=0
                    
                else:
                    #nFresidual = np.linalg.norm(Fresidual) 
                    used_n = self.Arnoldi_iter( self.trial_currents, 
                                                Fresidual*self.vals_for_rel_change, #starting direction in I space
                                                Fresidual, #F(trial_currents)
                                                n_k, 
                                                conv_crit, 
                                                grad_eps, clip=clip)
                                                #,verbose_currents=verbose_currents)
                    #print('self.di_Arnoldi', self.di_Arnoldi)
                    #print('trial current before update', self.trial_currents)
                    self.trial_currents += self.di_Arnoldi
                    Fresidual = self.Fcircuit(self.trial_currents)
                    #print('full residual after update', Fresidual)
                    rel_change = abs(Fresidual)#/self.vals_for_rel_change
                    max_rel_change = max(rel_change)
                    
                    # if verbose_currents:
                    #     print(' coeffs= ', self.coeffs, '; terms used= ', used_n)
                    #     #print('new residual = ', Fresidual)
                    #     print('new max_rel_change = ', max_rel_change, np.argmax(rel_change), rel_change.mean())
                        
                    it += 1
            
            if it==max_iter:
                #if max_iter was hit, then message:
                print(f'failed to converge with less than {max_iter} iterations.') 
                print(f'Last max rel_change={max_rel_change}.')
                print('Restarting with smaller timestep')
                max_rel_residual /= 2
                grad_eps /= 2
                
        # if verbose_currents:
        #     print('number of Arnoldi iterations = ', it)

        #get ready for next step
        self.currents_vec = self.trial_currents.copy()
        self.assign_currents_1(self.currents_vec)
        self.eq1.plasma_psi = self.eq2.plasma_psi.copy()
        self.results = self.results1.copy()
        self.dR = self._dR.copy()
        
        self.plasma_against_wall = np.sum(self.mask_outside_reactor*self.results['separatrix'][1])
        return self.plasma_against_wall
    