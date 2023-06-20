import numpy as np

from . import machine_config
from .machine_config import coils_order

from copy import deepcopy

# from . import quants_for_emu
from .circuit_eq_metal import metal_currents
from .circuit_eq_plasma import plasma_current
from .linear_solve import simplified_solver_dJ
from .linear_solve import simplified_solver_J1
from . import plasma_grids
from . import extrapolate

#from picardfast import fast_solve
from .GSstaticsolver import NKGSsolver
from .jtor_update import ConstrainPaxisIp


from .faster_shape import shapes
from .faster_shape import check_against_the_wall

import matplotlib.pyplot as plt


from scipy.signal import convolve2d


class nl_solver:
    # interfaces the circuit equation with freeGS NK solver 
    # executes dt evolution of fully non linear equations
    
    
    def __init__(self, profiles, eq, 
                 max_mode_frequency, 
                 max_internal_timestep=.0001, #has been fixed to full_timestep
                 full_timestep=.0001,
                 plasma_resistivity=1e-6,
                 extrapolator_input_size=4,
                 extrapolator_order=1,
                 plasma_norm_factor=2000):

        self.shapes = shapes 

        self.nx = np.shape(eq.R)[0]
        self.ny = np.shape(eq.R)[1]
        self.eqR = eq.R
        self.eqZ = eq.Z
        # area factor for Iy
        dR = eq.R[1, 0] - eq.R[0, 0]
        dZ = eq.Z[0, 1] - eq.Z[0, 0]
        self.dRdZ = dR*dZ

        # mask identifying if plasma is hitting the wall
        self.plasma_pts, self.mask_inside_limiter = plasma_grids.define_reduced_plasma_grid(eq.R, eq.Z)
        self.mask_outside_limiter = plasma_grids.make_layer_mask(self.mask_inside_limiter)
        self.plasma_against_wall = 0
        self.idxs_mask = np.mgrid[0:self.nx, 0:self.ny][np.tile(self.mask_inside_limiter,(2,1,1))].reshape(2,-1)

        
        #instantiate solver on eq's domain
        self.NK = NKGSsolver(eq)

        # profiles are kept constant during evolution
        # paxis, fvac and alpha values are taken from ICs and kept fixed thereafter
        self.get_profiles_values(profiles)
        
        
        #eq needs only have same tokamak and grid structure as those that will be evolved
        #when calling time evolution on some initial eq, the eq object that is actually modified is eq1
        #with profiles1 for changes in the plasma current
        self.eq1 = deepcopy(eq)
        self.profiles1 = deepcopy(profiles)   
        # full plasma current density (make sure you start from an actual equilibrium!)
                                   

        #the numerical method used to solve for the time evolution uses iterations
        #the eq object that is modified during the iterations is eq2
        #with profiles2 for changes in the plasma current
        self.eq2 = deepcopy(eq)
        self.profiles2 = deepcopy(profiles) 

        self.eq3 = deepcopy(eq)
        self.profiles3 = deepcopy(profiles)   


        for i,labeli in enumerate(coils_order):
            self.eq1.tokamak[labeli].control = False
            self.eq2.tokamak[labeli].control = False
            self.eq3.tokamak[labeli].control = False
        
        
        self.dt_step = full_timestep
        self.plasma_norm_factor = plasma_norm_factor
    
        # self.max_internal_timestep = max_internal_timestep
        self.max_internal_timestep = 1.0*full_timestep
        self.max_mode_frequency = max_mode_frequency
        self.reset_plasma_resistivity(plasma_resistivity)


        #qfe is an instance of the quants_for_emulation class
        #calculates fluxes of plasma on coils and on itself
        #as well as all other time-dependent quantities needed for time evolution
        # self.qfe = quants_for_emu.quants_for_emulation(eq)
        
        # to calculate residual of metal circuit eq
        self.evol_metal_curr = metal_currents(flag_vessel_eig=1,
                                                flag_plasma=1,
                                                reference_eq=eq,
                                                max_mode_frequency=self.max_mode_frequency,
                                                max_internal_timestep=self.max_internal_timestep,
                                                full_timestep=self.dt_step)
        # this is the number of independent normal mode currents associated to max_mode_frequency
        self.n_metal_modes = self.evol_metal_curr.n_independent_vars
        

        # to calculate residual of plasma collapsed circuit eq
        self.evol_plasma_curr = plasma_current(reference_eq=eq,
                                               Rm12V=np.matmul(np.diag(self.evol_metal_curr.Rm12), self.evol_metal_curr.V),
                                               plasma_resistance_1d=self.plasma_resistance_1d)


        self.simplified_solver_dJ = simplified_solver_dJ(Lambdam1=self.evol_metal_curr.Lambdam1, 
                                                            Vm1Rm12=np.matmul(self.evol_metal_curr.Vm1, np.diag(self.evol_metal_curr.Rm12)), 
                                                            Mey=self.evol_metal_curr.Mey, 
                                                            Myy=self.evol_plasma_curr.Myy,
                                                            plasma_norm_factor=self.plasma_norm_factor,
                                                            plasma_resistance_1d=self.plasma_resistance_1d,
                                                            max_internal_timestep=self.max_internal_timestep,
                                                            full_timestep=self.dt_step)
        self.n_step_range = np.arange(self.simplified_solver_dJ.solver.n_steps)[::-1][np.newaxis] + 1                                                    
        
        self.simplified_solver_J1 = simplified_solver_J1(Lambdam1=self.evol_metal_curr.Lambdam1, 
                                                            Vm1Rm12=np.matmul(self.evol_metal_curr.Vm1, np.diag(self.evol_metal_curr.Rm12)), 
                                                            Mey=self.evol_metal_curr.Mey, 
                                                            Myy=self.evol_plasma_curr.Myy,
                                                            plasma_norm_factor=self.plasma_norm_factor,
                                                            plasma_resistance_1d=self.plasma_resistance_1d,
                                                            max_internal_timestep=self.max_internal_timestep,
                                                            full_timestep=self.dt_step)
        
        self.extrapolator = extrapolate.extrapolator(input_size=extrapolator_input_size,
                                                     interpolation_order=extrapolator_order,
                                                     parallel_dims=self.n_metal_modes+1)
        self.extrapolator_input_size = extrapolator_input_size

        # vector of full coil currents (not normal modes) is self.vessel_currents_vec
        # initial self.vessel_currents_vec values are taken from eq.tokamak
        # does not include plasma current
        # n_coils = len(coils_order)
        # self.len_currents = n_coils
        vessel_currents_vec = np.zeros(machine_config.n_coils)
        eq_currents = eq.tokamak.getCurrents()
        for i,labeli in enumerate(coils_order):
            vessel_currents_vec[i] = eq_currents[labeli]
        self.vessel_currents_vec = 1.0*vessel_currents_vec
        # vector of normal mode currents is self.eig_currents_vec
        self.eig_currents_vec = self.evol_metal_curr.IvesseltoId(Ivessel=self.vessel_currents_vec)
        
        # norm plasma current is divided by plasma_norm_factor to keep homogeneity
        self.norm_plasma_current = eq.plasmaCurrent()/self.plasma_norm_factor
        # vector of currents on which non linear system is solved is self.currents_vec
        currents_vec = np.zeros(self.n_metal_modes + 1)
        currents_vec[:self.n_metal_modes] = 1.0*self.eig_currents_vec
        currents_vec[-1] = self.norm_plasma_current
        self.currents_vec = 1.0*currents_vec
        self.new_currents = 1.0*currents_vec
        self.residual = np.zeros_like(self.currents_vec)
        
        self.mapd = np.zeros_like(self.eq1.R)

        self.step_no = 0

        #self.npshape = np.shape(eq.plasma_psi)
        # self.dt_step_plasma = 0

        # threshold to calculate rel_change in the currents to set value of dt_step
        # it may be useful to use different values for different coils and for passive/active structures later on
        # self.threshold = np.array([1000]*MASTU_coils.N_active
        #                           +[5000]*(len(self.currents_vec)-MASTU_coils.N_active-1)
        #                           +[1000])

        # self.void_matrix = np.zeros((self.len_currents, self.len_currents))
        # self.dummy_vec = np.zeros(self.len_currents)

        # #calculate eigenvectors of time evolution:
        # invm = np.linalg.inv(self.evol_currents.R_matrix[:-1,:-1]+MASTU_coils.coil_self_ind/.001)
        # v, w = np.linalg.eig(np.matmul(invm, MASTU_coils.coil_self_ind/.001))
        # w = w[:, np.argsort(-v)[:50]]
        # mw = np.mean(w, axis = 0, keepdims = True)
        # self.w = np.append(w, mw, axis=0)
        

       
        
    def reset_plasma_resistivity(self, plasma_resistivity):
        self.plasma_resistivity = plasma_resistivity
        plasma_resistance_matrix = self.eq1.R*(2*np.pi/self.dRdZ)*self.plasma_resistivity
        self.plasma_resistance_1d = plasma_resistance_matrix[self.mask_inside_limiter]
    
    def calc_plasma_resistance(self, norm_red_Iy0, norm_red_Iy1):
        plasma_resistance_0d = np.sum(self.plasma_resistance_1d*norm_red_Iy0*norm_red_Iy1)
        return plasma_resistance_0d
    
    def reset_timestep(self, time_step):
        self.dt_step = time_step
    
    def get_profiles_values(self, profiles):
        #this allows to use the same instantiation of the time_evolution_class
        #on ICs with different paxis, fvac and alpha values
        #if these are different from those used when first instantiating the class
        #just call this function on the new profile object:
        self.paxis = profiles.paxis
        self.fvac = profiles.fvac
        self.alpha_m = profiles.alpha_m
        self.alpha_n = profiles.alpha_n

    def Iyplasmafromjtor(self, jtor):
        red_Iy = jtor[self.mask_inside_limiter]*self.dRdZ
        return red_Iy

    def reduce_normalize(self, jtor, epsilon=1e-6):
        red_Iy = jtor[self.mask_inside_limiter]/(np.sum(jtor) + epsilon)
        return red_Iy
    
    def reduce_normalize_l2(self, jtor):
        red_Iy = jtor[self.mask_inside_limiter]/np.linalg.norm(jtor)
        return red_Iy

    def rebuild_grid_map(self, red_vec):
        self.mapd[self.idxs_mask[0], self.idxs_mask[1]] = red_vec
        return self.mapd



    def set_currents_eq1(self, eq, rtol_NK=1e-8):
        #gets initial currents from ICs, note these are before mode truncation!
        eq_currents = eq.tokamak.getCurrents()
        for i,labeli in enumerate(coils_order):
            self.vessel_currents_vec[i] = eq_currents[labeli]
        # vector of normal mode currents is self.eig_currents_vec
        self.eig_currents_vec = self.evol_metal_curr.IvesseltoId(Ivessel=self.vessel_currents_vec)

        self.norm_plasma_current = eq.plasmaCurrent()/self.plasma_norm_factor
        # vector of currents on which non linear system is solved is self.currents_vec
        self.currents_vec[:self.n_metal_modes] = 1.0*self.eig_currents_vec
        self.currents_vec[-1] = self.norm_plasma_current
        # vector of plasma current density (already reduced on grid from plasma_grids)

        self.eq1.plasma_psi = 1.0*eq.plasma_psi
        # solve GS again with vessel currents you get after truncating modes:
        self.assign_currents(self.currents_vec, self.eq1, self.profiles1)
        self.NK.solve(self.eq1, self.profiles1, target_relative_tolerance=rtol_NK)

        self.eq2.plasma_psi = 1.0*self.eq1.plasma_psi



    def assign_vessel_noise(self, eq, noise_level):
        #uses currents_vec to assign currents to both plasma and tokamak in eq/profiles

        noise_vec = np.random.randn(self.n_metal_modes - machine_config.n_active_coils)
        noise_vec *= noise_level
        noise_vec = np.concatenate((np.zeros(machine_config.n_active_coils), noise_vec))
        
        # calculate vessel currents from normal modes and assign
        self.vessel_currents_vec = self.evol_metal_curr.IdtoIvessel(Id=noise_vec)
        for i,labeli in enumerate(coils_order[machine_config.n_active_coils:]):
            eq.tokamak[labeli].current = self.vessel_currents_vec[i+machine_config.n_active_coils]
        


    def initialize_from_ICs(self, 
                            eq, profile, 
                            rtol_NK=1e-8, 
                            reset_dJ=True,
                            new_seed=True,
                            noise_level=.1,
                            ): 
                            # eq for ICs, with all properties set up at time t=0, 
                            # ie with eq.tokamak.currents = I(t) and eq.plasmaCurrent = I_p(t) 
        
        self.step_counter = 0
        self.currents_guess = False
        self.rtol_NK = rtol_NK

        #get profile parametrization
        self.get_profiles_values(profile)

        # perturb passive structures
        if new_seed:
            self.assign_vessel_noise(eq, noise_level)

        #ensure it's a GS solution
        self.NK.solve(eq, profile, target_relative_tolerance=rtol_NK)


        #prepare currents
        self.set_currents_eq1(eq)
        self.currents_vec_m1 = 1.0*self.currents_vec

        self.jtor_m1 = 1.0*self.profiles1.jtor

        self.red_Iy = self.Iyplasmafromjtor(self.profiles1.jtor)
        self.red_Iy_m1 = 1.0*self.red_Iy
        # J0 is the direction of the plasma current vector at time t0
        # self.J0 = self.red_Iy/np.linalg.norm(self.red_Iy)
        self.J0 = self.red_Iy/np.sum(self.red_Iy)
        self.J0_m1 = 1.0*self.J0
        if reset_dJ:
            # dJ in the direction of the plasma current change in the timestep
            self.dJ = 1.0*self.J0#/np.linalg.norm(self.J0)
            self.J1 = 1.0*self.J0

        self.time = 0
        self.step_no = 0

        # check if against the wall
        if check_against_the_wall(jtor=self.profiles1.jtor, 
                                  boole_mask_outside_limiter=self.mask_outside_limiter):
            print('plasma in ICs is touching the wall!')

        

    def step_complete_assign(self, trial_currents):
        self.jtor_m1 = 1.0*self.profiles1.jtor

        self.profiles1.jtor = 1.0*self.profiles2.jtor
        self.eq1.plasma_psi = 1.0*self.eq2.plasma_psi

        self.currents_vec_m1 = 1.0*self.currents_vec
        self.currents_vec = 1.0*trial_currents

        self.red_Iy_m1 = 1.0*self.red_Iy
        self.red_Iy = 1.0*self.red_Iy_trial
        
        self.J0_m1 = 1.0*self.J0
        # self.J0 = self.red_Iy/np.linalg.norm(self.red_Iy)
        self.J0 = self.red_Iy/np.sum(self.red_Iy)

        self.assign_currents(self.currents_vec, self.eq1, self.profiles1)

        self.step_no += 1


    def assign_currents(self, currents_vec, eq, profile):
        #uses currents_vec to assign currents to both plasma and tokamak in eq/profiles

        profile.Ip = self.plasma_norm_factor*currents_vec[-1]

        # calculate vessel currents from normal modes and assign
        self.vessel_currents_vec = self.evol_metal_curr.IdtoIvessel(Id=currents_vec[:-1])
        for i,labeli in enumerate(coils_order):
            eq.tokamak[labeli].current = self.vessel_currents_vec[i]
        # assign plasma current to equilibrium
        eq._current = self.plasma_norm_factor*currents_vec[-1]

    def guess_currents_from_extrapolation(self,):
        # run after step is complete and assigned, prepares for next step

        if self.step_no >= self.extrapolator_input_size:
            self.currents_guess = self.extrapolator.in_out(self.currents_vec)

        else:
            self.extrapolator.set_Y(self.step_no, self.currents_vec)



    def guess_J_from_extrapolation(self, alpha, rtol_NK):
        # run after step is complete and assigned, prepares for next step

        if self.step_no >= self.extrapolator_input_size:
            currents_guess = self.extrapolator.in_out(self.currents_vec)

            self.assign_currents(currents_vec=currents_guess, profile=self.profiles2, eq=self.eq2)
            self.NK.solve(self.eq2, self.profiles2, target_relative_tolerance=rtol_NK)

            self.J1 = (1-alpha)*self.J1 + alpha*self.reduce_normalize(self.profiles2.jtor)
            self.dJ = (1-alpha)*self.dJ + alpha*self.reduce_normalize(self.profiles2.jtor-self.jtor_m1)

        else:
            self.extrapolator.set_Y(self.step_no, self.currents_vec)
        
        




    
   
    

    def Fresidual_nk_dJ(self, dJ,
                              active_voltage_vec,
                              rtol_NK):
        
        # Sp = self.calc_plasma_resistance(self.J0, dJ)/self.Rp
        self.simplified_c1 = 1.0*self.simplified_solver_dJ.stepper(It=self.currents_vec,
                                                            norm_red_Iy0=self.J0, 
                                                            norm_red_Iy_dot=dJ, 
                                                            active_voltage_vec=active_voltage_vec, 
                                                            Rp=self.Rp)
        res = 1.0*self.Fresidual_dJ(trial_currents=self.simplified_solver_dJ.solver.intermediate_results, 
                                    active_voltage_vec=active_voltage_vec, 
                                    rtol_NK=rtol_NK)   
        dJ1 = self.red_Iy_dot/np.linalg.norm(self.red_Iy_dot)
        return dJ1-dJ
    
    
    def currents_from_J1(self, J1,
                                active_voltage_vec,
                                ):
        mix_map = self.rebuild_grid_map(self.J0_m1 + J1)
        self.broad_J0 = self.reduce_normalize(convolve2d(mix_map, np.ones((3,3)), mode='same'))
        # self.broad_J0 = self.J0_m1 + self.J1
        self.broad_J0 /= np.sum(self.broad_J0)
        simplified_c1 = 1.0*self.simplified_solver_J1.stepper(It=self.currents_vec_m1,
                                                                norm_red_Iy_m1=self.J0_m1, 
                                                                norm_red_Iy0=self.broad_J0, 
                                                                norm_red_Iy1=J1, 
                                                                active_voltage_vec=active_voltage_vec,
                                                                central_2=self.central_2)
        return simplified_c1


    def iterative_unit_J1(self, J1,
                                active_voltage_vec,
                                rtol_NK):
        simplified_c1 = self.currents_from_J1(J1,
                                active_voltage_vec,
                                )
        self.Fresidual_dJ(trial_currents=simplified_c1, 
                            active_voltage_vec=active_voltage_vec, 
                            rtol_NK=rtol_NK)   
        return simplified_c1, self.residual


    def nl_step_iterative_J1(self, active_voltage_vec, 
                                J1,
                                alpha=.8, 
                                rtol_NK=5e-4,
                                atol_currents=1e-3,
                                atol_J=1e-3,
                                use_extrapolation=True,
                                verbose=False
                                ):
        
        self.J1 = 1.0*J1
        self.central_2  = (1 + (self.step_no>0))

        
        simplified_c, res = self.iterative_unit_J1(J1=self.J1,
                                                    active_voltage_vec=active_voltage_vec,
                                                    rtol_NK=rtol_NK)

        # dcurrents = np.abs(simplified_c-self.currents_vec)
        # vals_for_check = np.where(dcurrents>threshold, dcurrents, threshold)

        iterative_steps = 0
        control = 1
        while control:
            self.J1n = self.reduce_normalize(self.profiles2.jtor)
            self.ddJ = self.J1n - self.J1
            self.J1n = (1-alpha)*self.J1 + alpha*self.J1n
            self.J1n /= np.sum(self.J1n)
            self.J1 = 1.0*self.J1n
            simplified_c1, res = self.iterative_unit_J1(J1=self.J1,
                                                        active_voltage_vec=active_voltage_vec,
                                                        rtol_NK=rtol_NK)   

            abs_increments = np.abs(simplified_c - simplified_c1)
            # dcurrents = np.abs(simplified_c1-self.currents_vec)
            # vals_for_check = np.where(dcurrents>threshold, dcurrents, threshold)
            # rel_residuals = np.abs(res)#/vals_for_check
            control = np.any(abs_increments>atol_currents)
            # control += np.any(rel_residuals>rtol_residuals)
            control += np.any( abs(self.ddJ) > atol_J )
            if verbose:
                print('max currents change = ', np.max(abs_increments))
                print('max J direction change = ', np.max(np.abs(self.ddJ)), np.linalg.norm(self.ddJ))
                # print('max circuit eq residual (dim of currents) = ', np.argmax(abs(res)), res)
                # print(simplified_c1 - self.currents_vec_m1)

            iterative_steps += 1

            simplified_c = 1.0*simplified_c1
        
        self.time += self.dt_step

        plt.figure()
        plt.imshow(self.profiles2.jtor - self.jtor_m1)
        plt.colorbar()
        plt.show()

        plt.figure()
        plt.imshow(self.eq2.plasma_psi - self.eq1.plasma_psi)
        plt.colorbar()
        plt.show()

        self.step_complete_assign(simplified_c)
        if use_extrapolation:
            self.guess_J_from_extrapolation(alpha=alpha, rtol_NK=rtol_NK)
        
        flag = check_against_the_wall(jtor=self.profiles2.jtor, 
                                      boole_mask_outside_limiter=self.mask_outside_limiter)


        return flag
    
    
    def calculate_J1hat_GS(self, trial_currents, rtol_NK=1e-9):
        self.assign_solve(trial_currents, rtol_NK=1e-8)
        J1hat = self.reduce_normalize(self.profiles2.jtor)
        return J1hat
    
    def Fresidual_curr_GS(self, trial_currents, active_voltage_vec, rtol_NK=1e-9):
        J1hat = self.calculate_J1hat_GS(trial_currents)
        iterated_currs = self.currents_from_J1(J1hat, active_voltage_vec)
        current_res = iterated_currs - trial_currents
        return current_res  

    def nl_step_nk_curr_GS(self, active_voltage_vec, 
                                rtol_NK=1e-9,
                                rtol_currents=.1,
                                verbose=False,
                                n_k=6,
                                conv_crit=.3,
                                max_collinearity=.3,
                                grad_eps=1,
                                clip=3,
                                use_extrapolation=False):
        
        note_tokamak_psi = 1.0*self.NK.tokamak_psi
        
        self.central_2  = (1 + (self.step_no>0))
        if use_extrapolation*(self.step_no > self.extrapolator_input_size):
            self.trial_currents = 1.0*self.currents_guess 
            
        else:
            self.trial_currents, res = self.iterative_unit_J1(self.J0,
                                                            active_voltage_vec,
                                                            rtol_NK=rtol_NK)

        res0 = self.Fresidual_curr_GS(self.trial_currents, active_voltage_vec, rtol_NK)
        # note_psi = 1.0*self.eq2.plasma_psi

        abs_res0 = np.abs(res0)
        # nres0 = np.sum(abs_res0)
        rel_res0 = abs_res0/abs(self.trial_currents - self.currents_vec_m1)
        control = np.any(rel_res0 > rtol_currents)

        print('starting:', np.amax(rel_res0), np.mean(rel_res0))

        while control:
            self.Arnoldi_iteration(trial_sol=self.trial_currents, #trial_current expansion point
                                    vec_direction=res0, #first vector for current basis
                                    Fresidual=res0, #circuit eq. residual at trial_current expansion point: Fresidual(trial_current)
                                    Fresidual_function=self.Fresidual_curr_GS,
                                    active_voltage_vec=active_voltage_vec,
                                    n_k=n_k, # max number of basis vectors (must be less than number of modes + 1)
                                    conv_crit=conv_crit,   #add basis vector 
                                                    #if unexplained orthogonal component is larger than
                                    max_collinearity=max_collinearity,
                                    grad_eps=grad_eps, #infinitesimal step size, when compared to norm(trial)
                                    clip=clip,
                                    rtol_NK=rtol_NK)
            print(self.coeffs)
            self.trial_currents += self.d_sol_step

            res0 = self.Fresidual_curr_GS(self.trial_currents, active_voltage_vec, rtol_NK)
            abs_res0 = np.abs(res0)
            # nres0 = np.sum(abs_res0)

            rel_res0 = abs_res0/abs(self.trial_currents - self.currents_vec_m1)
            control = np.any(rel_res0 > rtol_currents)

            print('cycle:', np.amax(rel_res0), np.mean(rel_res0))
            
            # r_dpsi = abs(self.eq2.plasma_psi - note_psi)
            # r_dpsi /= (np.amax(note_psi) - np.amin(note_psi))
            # control += np.any(r_dpsi > rtol_psi)


        self.time += self.dt_step

        plt.figure()
        plt.imshow(self.profiles2.jtor - self.jtor_m1)
        plt.colorbar()
        plt.show()

        self.dpsi = self.eq2.plasma_psi - self.eq1.plasma_psi
        plt.figure()
        plt.imshow(self.dpsi)
        plt.colorbar()
        plt.show()

        
        plt.figure()
        plt.imshow(self.NK.tokamak_psi - note_tokamak_psi)
        plt.colorbar()
        plt.show()


        self.step_complete_assign(self.trial_currents)
        
        flag = check_against_the_wall(jtor=self.profiles2.jtor, 
                                      boole_mask_outside_limiter=self.mask_outside_limiter)

        if use_extrapolation:
            self.guess_currents_from_extrapolation()

        return flag






    def calculate_J1hat(self, currents, plasma_psi_2d):
        self.assign_currents(currents, profile=self.profiles2, eq=self.eq2)
        self.tokamak_psi = self.eq2.tokamak.calcPsiFromGreens(pgreen=self.eq2._pgreen)
        jtor_ = self.profiles2.Jtor(self.eqR, self.eqZ, self.tokamak_psi + plasma_psi_2d)
        J1hat = self.reduce_normalize(jtor_)
        return J1hat

    def Fresidual_curr(self, trial_currents, active_voltage_vec, rtol_NK=1e-9):
        # this will not call NK
        J1hat = self.calculate_J1hat(trial_currents, self.trial_plasma_psi)
        iterated_currs = self.currents_from_J1(J1hat, active_voltage_vec)
        current_res = iterated_currs - trial_currents
        return current_res                                            

    def find_best_convex_combination(self, previous_residual, 
                                            trial_currents, 
                                            active_voltage_vec, 
                                            pts=[.05,.95],
                                            blend=1.):
        note_plasma_psi = 1.0*self.trial_plasma_psi
        res_list = []
        for alpha in pts:
            self.trial_plasma_psi = (1-alpha)*note_plasma_psi + alpha*self.eq2.plasma_psi
            res_list.append(np.sum(np.abs(self.Fresidual_curr(trial_currents, active_voltage_vec))))
        a = (res_list[1] - res_list[0])/(pts[1] - pts[0])
        if a>0:
            b = res_list[0] - a*pts[0]
            best_alpha = max(.0, min(1, (blend*previous_residual - b)/a))
            print(best_alpha)
        else:
            best_alpha = 1
            print(best_alpha, 'this was negative!')
        
        self.trial_plasma_psi = (1-best_alpha)*note_plasma_psi + best_alpha*self.eq2.plasma_psi
        

    def nl_step_nk_curr(self, active_voltage_vec, 
                                rtol_NK=1e-9,
                                atol_currents=1e-3,
                                rtol_psi=1e-3,
                                verbose=False,
                                n_k=6,
                                conv_crit=.3,
                                max_collinearity=.3,
                                grad_eps=1,
                                clip=3):
        
        self.central_2  = (1 + (self.step_no>0))

        self.trial_currents, res = self.iterative_unit_J1(self.J0,
                                                            active_voltage_vec,
                                                            rtol_NK=rtol_NK)
        self.trial_plasma_psi = 1.0*self.eq2.plasma_psi

        res0 = self.Fresidual_curr(self.trial_currents, active_voltage_vec)
        abs_res0 = np.abs(res0)
        nres0 = np.sum(abs_res0)

        control = np.any(abs_res0 > atol_currents)
        r_dpsi = abs(self.eq2.plasma_psi - self.trial_plasma_psi)
        r_dpsi /= (np.amax(self.trial_plasma_psi) - np.amin(self.trial_plasma_psi))
        control += np.any(r_dpsi > rtol_psi)

        print('starting:', np.amax(abs_res0), np.amax(r_dpsi))

        # self.trial_plasma_psi = 1.0*self.eq2.plasma_psi

        while control:
            self.Arnoldi_iteration(trial_sol=self.trial_currents, #trial_current expansion point
                                    vec_direction=res0, #first vector for current basis
                                    Fresidual=res0, #circuit eq. residual at trial_current expansion point: Fresidual(trial_current)
                                    Fresidual_function=self.Fresidual_curr,
                                    active_voltage_vec=active_voltage_vec,
                                    n_k=n_k, # max number of basis vectors (must be less than number of modes + 1)
                                    conv_crit=conv_crit,   #add basis vector 
                                                    #if unexplained orthogonal component is larger than
                                    max_collinearity=max_collinearity,
                                    grad_eps=grad_eps, #infinitesimal step size, when compared to norm(trial)
                                    clip=clip,
                                    rtol_NK=rtol_NK)
            self.trial_currents += self.d_sol_step

            self.assign_solve(self.trial_currents, rtol_NK)

            r_dpsi = abs(self.eq2.plasma_psi - self.trial_plasma_psi)
            r_dpsi /= (np.amax(self.trial_plasma_psi)-np.amin(self.trial_plasma_psi))
            control = np.any(r_dpsi > rtol_psi)

            self.find_best_convex_combination(nres0,
                                            self.trial_currents,
                                            active_voltage_vec)
            res0 = self.Fresidual_curr(self.trial_currents, active_voltage_vec)                                
            abs_res0 = np.abs(res0)
            nres0 = np.sum(abs_res0)
            control += np.any(abs_res0 > atol_currents)

            print('cycle:', np.amax(abs_res0), np.amax(r_dpsi))

        self.time += self.dt_step

        plt.figure()
        plt.imshow(self.profiles2.jtor - self.jtor_m1)
        plt.colorbar()
        plt.show()

        self.dpsi = self.eq2.plasma_psi - self.eq1.plasma_psi
        plt.figure()
        plt.imshow(self.dpsi)
        plt.colorbar()
        plt.show()

        self.step_complete_assign(self.trial_currents)
        
        flag = check_against_the_wall(jtor=self.profiles2.jtor, 
                                      boole_mask_outside_limiter=self.mask_outside_limiter)


        return flag






    def nl_step_nk_psi_curr(self, active_voltage_vec, 
                                rel_rtol_NK=.1,
                                rtol_currents=.2,
                                rtol_GS=.2,
                                use_extrapolation=False,
                                verbose=False,
                                n_k=5,
                                conv_crit=.5,
                                max_collinearity=.3,
                                grad_eps_psi=2,
                                grad_eps_curr=2,
                                clip=3,
                                NK_psi_switch=6,
                                blend_curr=1.0,
                                blend_psi=1.0,
                                blend_GS=1.0,
                                curr_eps=1e-4,
                                max_no_NK_psi=5
                                ):
        
        self.central_2  = (1 + (self.step_no>0))


        if use_extrapolation*(self.step_no > self.extrapolator_input_size):
            self.trial_currents = 1.0*self.currents_guess 
            self.assign_solve(self.trial_currents, self.rtol_NK)
            
        else:
            self.trial_currents, res = self.iterative_unit_J1(self.J0,
                                                              active_voltage_vec,
                                                              rtol_NK=self.rtol_NK)
            
        self.trial_plasma_psi = 1.0*self.eq2.plasma_psi
        # 

        res0 = self.Fresidual_curr(self.trial_currents, active_voltage_vec)
        curr_step = abs(self.trial_currents - self.currents_vec_m1)
        curr_step = np.where(curr_step>curr_eps, curr_step, curr_eps)
        rel_curr_res = abs(res0 / curr_step)
        control = np.any(rel_curr_res > rtol_currents)
        print('starting: curr residual', np.amax(rel_curr_res), np.mean(rel_curr_res))

        n_no_NK_psi = 0
        n_it = 0

        while control:
            self.NK.solve(self.eq2, self.profiles2, self.rtol_NK)
            self.trial_plasma_psi *= (1-blend_GS)
            self.trial_plasma_psi += blend_GS*self.eq2.plasma_psi
            self.assign_currents(self.trial_currents, self.eq2, self.profiles2)
            self.tokamak_psi = self.eq2.tokamak.calcPsiFromGreens(pgreen=self.eq2._pgreen)
            self.trial_plasma_psi = self.trial_plasma_psi.reshape(-1)

            res_psi = self.Fresidual_psi(trial_plasma_psi=self.trial_plasma_psi,
                                            active_voltage_vec=active_voltage_vec, 
                                            rtol_NK=self.rtol_NK)
            del_res_psi = (np.amax(res_psi) - np.amin(res_psi))

            print('psi_residual', del_res_psi)
            if (del_res_psi > self.rtol_NK/NK_psi_switch)+(n_no_NK_psi > max_no_NK_psi):
                n_no_NK_psi -= 1
                print('NK_psi!')
                self.Arnoldi_iteration(trial_sol=self.trial_plasma_psi, #trial_current expansion point
                                        vec_direction=res_psi, #first vector for current basis
                                        Fresidual=res_psi, #circuit eq. residual at trial_current expansion point: Fresidual(trial_current)
                                        Fresidual_function=self.Fresidual_psi,
                                        active_voltage_vec=active_voltage_vec,
                                        n_k=n_k, # max number of basis vectors (must be less than number of modes + 1)
                                        conv_crit=conv_crit,   #add basis vector 
                                                        #if unexplained orthogonal component is larger than
                                        max_collinearity=max_collinearity,
                                        grad_eps=grad_eps_psi, #infinitesimal step size, when compared to norm(trial)
                                        clip=clip,
                                        rtol_NK=self.rtol_NK)
                print('psi_coeffs = ', self.coeffs)
                self.trial_plasma_psi += blend_psi*self.d_sol_step
            else:
                print('n_no_NK_psi = ', n_no_NK_psi)
                n_no_NK_psi += 1


            self.trial_plasma_psi = 1.0*self.trial_plasma_psi.reshape(65,65)
            res0 = self.Fresidual_curr(self.trial_currents, active_voltage_vec)
            rel_curr_res = abs(res0 / curr_step)
            print('intermediate curr residual', np.amax(abs(res0/(self.trial_currents - self.currents_vec_m1 + curr_eps))), np.mean(abs(res0/(self.trial_currents - self.currents_vec_m1))))
            self.Arnoldi_iteration(trial_sol=self.trial_currents, #trial_current expansion point
                                    vec_direction=res0, #first vector for current basis
                                    Fresidual=res0, #circuit eq. residual at trial_current expansion point: Fresidual(trial_current)
                                    Fresidual_function=self.Fresidual_curr,
                                    active_voltage_vec=active_voltage_vec,
                                    n_k=n_k, # max number of basis vectors (must be less than number of modes + 1)
                                    conv_crit=conv_crit,   #add basis vector 
                                                    #if unexplained orthogonal component is larger than
                                    max_collinearity=max_collinearity,
                                    grad_eps=grad_eps_psi, #infinitesimal step size, when compared to norm(trial)
                                    clip=clip,
                                    rtol_NK=self.rtol_NK)
            print('curr_coeffs = ', self.coeffs)
            self.trial_currents += blend_curr*self.d_sol_step

            res0 = self.Fresidual_curr(self.trial_currents, active_voltage_vec)
            curr_step = abs(self.trial_currents - self.currents_vec_m1)
            curr_step = np.where(curr_step>curr_eps, curr_step, curr_eps)
            rel_curr_res = abs(res0 / curr_step)
            control = np.any(rel_curr_res > rtol_currents)

            self.assign_currents(self.trial_currents, self.eq2, self.profiles2)
            self.tokamak_psi = self.eq2.tokamak.calcPsiFromGreens(pgreen=self.eq2._pgreen)
            a_res_GS = np.amax(abs(self.NK.F_function(self.trial_plasma_psi.reshape(-1),
                                                        self.tokamak_psi.reshape(-1),
                                                        self.profiles2)))
            self.dpsi = self.trial_plasma_psi - self.eq1.plasma_psi
            r_res_GS = a_res_GS/(np.amax(self.dpsi) - np.amin(self.dpsi))
            control += (r_res_GS > rtol_GS)
            
            n_it += 1
            print('cycle: ', np.amax(rel_curr_res), np.mean(rel_curr_res))
            print('GS residual: ',r_res_GS, a_res_GS)

        # update rtol_NK based on step just taken
        self.rtol_NK = rel_rtol_NK*(np.amax(self.dpsi) - np.amin(self.dpsi))

        self.time += self.dt_step

        self.eq2.plasma_psi = 1.0*self.trial_plasma_psi

        # plt.figure()
        # plt.imshow(self.profiles2.jtor - self.jtor_m1)
        # plt.colorbar()
        # plt.title('J1-J0')
        # plt.show()

        
        # plt.figure()
        # plt.imshow(self.dpsi)
        # plt.title('psi1-psi0')
        # plt.colorbar()
        # plt.show()


        self.step_complete_assign(self.simplified_c)
        if use_extrapolation:
            self.guess_currents_from_extrapolation()
        
    
        flag = check_against_the_wall(jtor=self.profiles2.jtor, 
                                      boole_mask_outside_limiter=self.mask_outside_limiter)


        return flag















    
    def Fresidual_nk_J1(self, J1, active_voltage_vec, rtol_NK=1e-8):
        J1 /= np.sum(J1)
        self.simplified_c, self.circ_eq_res = self.iterative_unit_J1(J1, active_voltage_vec, rtol_NK)
        self.J1_new = self.reduce_normalize(self.profiles2.jtor)
        return self.J1_new - J1
    def Fresidual_nk_J1_currents(self, J1, active_voltage_vec, rtol_NK=1e-8):
        dJ1 = self.Fresidual_nk_J1(J1, active_voltage_vec, rtol_NK)
        note_currents = 1.0*self.simplified_c
        iterated_c = self.currents_from_J1(self.J1_new,
                                            active_voltage_vec)
                                            
        d_currents = iterated_c - note_currents
        return dJ1, d_currents


    def nl_step_nk_J1(self, active_voltage_vec, 
                                J1,
                                # alpha=.8, 
                                rtol_NK=1e-9,
                                rtol_currents=.5,
                                atol_J=1e-3,
                                use_extrapolation=True,
                                verbose=False,
                                n_k=6,
                                conv_crit=.3,
                                max_collinearity=.3,
                                grad_eps=2,
                                clip=3):
        
        self.J1 = 1.0*J1
        self.central_2  = (1 + (self.step_no>0))


        resJ, d_currents = self.Fresidual_nk_J1_currents(J1=self.J1,
                                                        active_voltage_vec=active_voltage_vec,
                                                        rtol_NK=rtol_NK)
        
        rel_res_currents = abs(d_currents/(self.simplified_c - self.currents_vec_m1))
        control = np.any(rel_res_currents > rtol_currents)
        control += np.any(resJ > atol_J)

        print('starting: ', max(rel_res_currents), max(abs(resJ)))

        iterative_steps = 0
        simplified_c = 1.0*self.simplified_c

        while control:
            self.Arnoldi_iteration(trial_sol=self.J1, #trial_current expansion point
                                    vec_direction=resJ, #first vector for current basis
                                    Fresidual=resJ, #circuit eq. residual at trial_current expansion point: Fresidual(trial_current)
                                    Fresidual_function=self.Fresidual_nk_J1,
                                    active_voltage_vec=active_voltage_vec,
                                    n_k=n_k, # max number of basis vectors (must be less than number of modes + 1)
                                    conv_crit=conv_crit,   #add basis vector 
                                                    #if unexplained orthogonal component is larger than
                                    max_collinearity=max_collinearity,
                                    grad_eps=grad_eps, #infinitesimal step
                                    clip=clip,
                                    rtol_NK=rtol_NK)
            print(self.coeffs)
            
            # self.J1 += self.d_sol_step
            # self.J1 /= np.sum(self.J1)

            J1new = self.J1 + self.d_sol_step
            J1new /= np.sum(J1new)

            resJ, d_currents = self.Fresidual_nk_J1_currents(J1=J1new,
                                                            active_voltage_vec=active_voltage_vec,
                                                            rtol_NK=rtol_NK)

            self.J1 = 1.0*J1new

           
            rel_res_currents = abs(d_currents/(self.simplified_c - self.currents_vec_m1))
            control = np.any(rel_res_currents > rtol_currents)
            control += np.any(abs(resJ) > atol_J)
            simplified_c = 1.0*self.simplified_c

            print('cycle: ', max(rel_res_currents), max(abs(resJ)))

            if verbose:
                print('max currents change = ', np.max(rel_res_currents))
                print('max J direction change = ', np.max(np.abs(resJ)), np.linalg.norm(resJ))
                # print('max circuit eq residual (dim of currents) = ', np.argmax(abs(self.circ_eq_res)), self.circ_eq_res)
                # print(self.simplified_c - self.currents_vec_m1)

            iterative_steps += 1

        
        self.time += self.dt_step

        plt.figure()
        plt.imshow(self.profiles2.jtor - self.jtor_m1)
        plt.colorbar()
        plt.show()

        self.dpsi = self.eq2.plasma_psi - self.eq1.plasma_psi
        plt.figure()
        plt.imshow(self.dpsi)
        plt.colorbar()
        plt.show()


        self.step_complete_assign(self.simplified_c)
        if use_extrapolation:
            self.guess_J_from_extrapolation(alpha=1, rtol_NK=rtol_NK)
        

    
        flag = check_against_the_wall(jtor=self.profiles2.jtor, 
                                      boole_mask_outside_limiter=self.mask_outside_limiter)

        

        return flag


    def Fresidual_psi(self, trial_plasma_psi, active_voltage_vec, rtol_NK=1e-8):
        trial_plasma_psi_2d = trial_plasma_psi.reshape(self.nx, self.ny)

        jtor_ = self.profiles2.Jtor(self.eqR, self.eqZ, self.tokamak_psi + trial_plasma_psi_2d)
        hatjtor_ = self.reduce_normalize(jtor_)
        self.simplified_c, res = self.iterative_unit_J1(J1=hatjtor_,
                                                    active_voltage_vec=active_voltage_vec,
                                                    rtol_NK=rtol_NK)
        psi_residual = self.eq2.plasma_psi.reshape(-1) - trial_plasma_psi
        # self.currents_nk_psi = np.append(self.currents_nk_psi, self.simplified_c[:,np.newaxis], axis=1)
        return psi_residual
    
    


    def nl_step_nk_psi(self, active_voltage_vec, 
                                trial_currents,
                                rtol_NK=1e-9,
                                rtol_currents=.1,
                                atol_J=1e-3,
                                verbose=False,
                                n_k=6,
                                conv_crit=.3,
                                max_collinearity=.3,
                                grad_eps=1,
                                clip=3,
                                use_extrapolation=False):
        
        self.currents_nk_psi = np.zeros((self.n_metal_modes+1, 0))
        
        self.central_2  = (1 + (self.step_no>0))

        if use_extrapolation*(self.step_no > self.extrapolator_input_size):
            self.trial_currents = 1.0*self.currents_guess 
            
        else:
            self.trial_currents, res = self.iterative_unit_J1(self.J0,
                                                              active_voltage_vec,
                                                              rtol_NK=rtol_NK)

        self.ref_currents = 1.0*trial_currents
        self.tokamak_psi = 1.0*self.NK.tokamak_psi
        psi0 = 1.0*self.eq2.plasma_psi.reshape(-1)

        
        res_psi = self.Fresidual_psi(trial_plasma_psi=psi0,
                                        active_voltage_vec=active_voltage_vec, 
                                        rtol_NK=1e-9)
        self.currents_nk_psi = np.zeros((self.n_metal_modes+1, 0))

        abs_increments = abs(self.simplified_c - trial_currents)
        control = np.any(abs_increments > atol_currents)
        control += np.any(res_psi > atol_J)
        print('starting: ', max(abs(res_psi)), max(abs_increments))
        simplified_c = 1.0*self.simplified_c
         
        while control:
            self.Arnoldi_iteration(trial_sol=psi0, #trial_current expansion point
                                    vec_direction=res_psi, #first vector for current basis
                                    Fresidual=res_psi, #circuit eq. residual at trial_current expansion point: Fresidual(trial_current)
                                    Fresidual_function=self.Fresidual_psi,
                                    active_voltage_vec=active_voltage_vec,
                                    n_k=n_k, # max number of basis vectors (must be less than number of modes + 1)
                                    conv_crit=conv_crit,   #add basis vector 
                                                    #if unexplained orthogonal component is larger than
                                    max_collinearity=max_collinearity,
                                    grad_eps=grad_eps, #infinitesimal step size, when compared to norm(trial)
                                    clip=clip,
                                    rtol_NK=rtol_NK)
            
            res_psi = self.Fresidual_psi(trial_plasma_psi=psi0+self.d_sol_step,
                                            active_voltage_vec=active_voltage_vec, 
                                            rtol_NK=rtol_NK)
            self.Fresidual_dJ(trial_currents=self.simplified_c,
                                active_voltage_vec=active_voltage_vec,
                                rtol_NK=rtol_NK)
            
            abs_increments = abs(self.simplified_c - simplified_c)
            control = np.any(abs_increments > atol_currents)
            control += np.any(res_psi > atol_J)
            print('cycle: ', max(abs(res_psi)), max(abs_increments))

            psi0 = 1.0*self.eq2.plasma_psi.reshape(-1)
            self.tokamak_psi = 1.0*self.NK.tokamak_psi
            self.ref_currents = 1.0*self.simplified_c
            simplified_c = 1.0*self.simplified_c

        self.time += self.dt_step

        plt.figure()
        plt.imshow(self.profiles2.jtor - self.jtor_m1)
        plt.colorbar()
        plt.show()

        self.dpsi = self.eq2.plasma_psi - self.eq1.plasma_psi
        plt.figure()
        plt.imshow(self.dpsi)
        plt.colorbar()
        plt.show()


        self.step_complete_assign(self.simplified_c)
        # if use_extrapolation:
        #     self.guess_J_from_extrapolation(alpha=alpha, rtol_NK=rtol_NK)
        
        flag = check_against_the_wall(jtor=self.profiles2.jtor, 
                                      boole_mask_outside_limiter=self.mask_outside_limiter)

        
        return flag



    def assign_solve(self, trial_currents, rtol_NK=1e-8):
        self.assign_currents(trial_currents, profile=self.profiles2, eq=self.eq2)
        self.NK.solve(self.eq2, self.profiles2, target_relative_tolerance=rtol_NK)
        self.red_Iy_trial = self.Iyplasmafromjtor(self.profiles2.jtor)
        self.red_Iy_dot = (self.red_Iy_trial - self.red_Iy_m1)/(self.central_2*self.dt_step)
        self.Id_dot = ((trial_currents - self.currents_vec_m1)/(self.central_2*self.dt_step))[:-1]


    def calculate_circ_eq_residual(self, trial_currents, active_voltage_vec):
        self.forcing_term = self.evol_metal_curr.forcing_term_eig_plasma(active_voltage_vec=active_voltage_vec, 
                                                                         Iydot=self.red_Iy_dot)

        self.residual[:-1] = 1.0*self.evol_metal_curr.current_residual( Itpdt=trial_currents[:-1], 
                                                                        Iddot=self.Id_dot, 
                                                                        forcing_term=self.forcing_term)

        self.residual[-1] = 1.0*self.evol_plasma_curr.current_residual( red_Iy0=self.red_Iy, 
                                                                        red_Iy1=self.red_Iy_trial,
                                                                        red_Iydot=self.red_Iy_dot,
                                                                        Iddot=self.Id_dot)/self.plasma_norm_factor
        


    def Fresidual_dJ(self, trial_currents, active_voltage_vec, rtol_NK=1e-8):
        # trial_currents is the full array of intermediate results from euler solver
        # root problem for circuit equation
        # collects both metal normal modes and norm_plasma
        
        # current at t+dt
        # d_current_tpdt = np.sum(trial_currents, axis=-1)

        self.assign_solve(trial_currents, rtol_NK=rtol_NK)
        self.calculate_circ_eq_residual(trial_currents, active_voltage_vec)
        




    # def Fresidual_J1(self, trial_currents, active_voltage_vec, rtol_NK=1e-8):
    #     # trial_currents is the full array of intermediate results from euler solver
    #     # root problem for circuit equation
    #     # collects both metal normal modes and norm_plasma
        
    #     # current at t+dt
    #     # current_tpdt = 1.0*trial_currents#[:, -1]
    #     self.assign_currents(trial_currents, profile=self.profiles2, eq=self.eq2)
    #     self.NK.solve(self.eq2, self.profiles2, target_relative_tolerance=rtol_NK)
    #     self.red_Iy_trial = self.Iyplasmafromjtor(self.profiles2.jtor)

    #     self.red_Iy_dot = (self.red_Iy_trial - self.red_Iy_m1)/(2*self.dt_step)
    #     self.Id_dot = ((trial_currents - self.currents_vec_m1)/(2*self.dt_step))[:-1]

    #     self.forcing_term = self.evol_metal_curr.forcing_term_eig_plasma(active_voltage_vec=active_voltage_vec, 
    #                                                                      Iydot=self.red_Iy_dot)

    #     # mean_curr = np.mean(trial_currents, axis=-1)                                                                 
    #     self.residual[:-1] = 1.0*self.evol_metal_curr.current_residual( Itpdt=trial_currents[:-1], 
    #                                                                     Iddot=self.Id_dot, 
    #                                                                     forcing_term=self.forcing_term)


    #     # mean_Iy = trial_currents[-1]*self.J1*self.plasma_norm_factor
    #     # mean_Iy = 1.0*self.red_Iy_trial
    #     self.residual[-1] = 1.0*self.evol_plasma_curr.current_residual( red_Iy0=self.red_Iy, 
    #                                                                     red_Iy1=self.red_Iy_trial,
    #                                                                     red_Iydot=self.red_Iy_dot,
    #                                                                     Iddot=self.Id_dot)/self.plasma_norm_factor
    #     # return self.residual



    
    # def iterative_unit_dJ(self, dJ,
    #                             active_voltage_vec,
    #                             Rp, 
    #                             rtol_NK):
    #     simplified_c1 = self.central_2*self.simplified_solver_dJ.stepper(It=self.currents_vec_m1,
    #                                                         norm_red_Iy0=self.J0, 
    #                                                         norm_red_Iy_dot=dJ, 
    #                                                         active_voltage_vec=active_voltage_vec, 
    #                                                         Rp=Rp,
    #                                                         central_2=self.central_2)
        
    #     # calculate t+dt currents
    #     # plasma
    #     Iy_tpdt = self.red_Iy_m1/self.plasma_norm_factor + simplified_c1[-1]*dJ
    #     simplified_c1[-1] = np.sum(Iy_tpdt)
    #     # metal
    #     simplified_c1[:-1] += self.currents_vec_m1[:-1]
        
    #     self.Fresidual_dJ(trial_currents=simplified_c1, 
    #                             active_voltage_vec=active_voltage_vec, 
    #                             rtol_NK=rtol_NK)   
    #     return simplified_c1, self.residual
    


    # def nl_step_iterative_dJ(self,  active_voltage_vec, 
    #                                 dJ,
    #                                 alpha=.8, 
    #                                 rtol_NK=5e-4,
    #                                 atol_currents=1e-3,
    #                                 atol_J=1e-3,
    #                                 verbose=False,
    #                                 use_extrapolation=True,
    #                                 ):
        
    #     self.central_2  = (1 + (self.step_no>0))
        
    #     Rp = self.calc_plasma_resistance(self.J0, self.J0_m1)
    #     self.dJ = 1.0*dJ
        
    #     simplified_c, res = self.iterative_unit_dJ(dJ=dJ,
    #                                                active_voltage_vec=active_voltage_vec,
    #                                                Rp=Rp, 
    #                                                rtol_NK=rtol_NK)

    #     # dcurrents = np.abs(simplified_c-self.currents_vec)
    #     # vals_for_check = np.where(dcurrents>threshold, dcurrents, threshold)

    #     iterative_steps = 0
    #     control = 1
    #     while control:
            
    #         # if verbose:
    #         #     plt.figure()
    #         #     plt.imshow(self.rebuild_grid_map(self.dJ))
    #         #     plt.colorbar()
    #         #     plt.title(str(np.sum(self.dJ))+'   '+str(simplified_c[-1]-self.currents_vec_m1[-1]))

    #         self.dJ1 = self.reduce_normalize(self.profiles2.jtor - self.jtor_m1)
    #         self.dJ1 = (1-alpha)*self.dJ + alpha*self.dJ1
    #         # self.dJ1 /= np.linalg.norm(self.dJ1)
    #         self.dJ1 /= np.sum(self.dJ1)
    #         self.ddJ = self.dJ1 - self.dJ
    #         self.dJ = 1.0*self.dJ1
    #         simplified_c1, res = self.iterative_unit_dJ(dJ=self.dJ, 
    #                                                     active_voltage_vec=active_voltage_vec,
    #                                                     Rp=Rp, 
    #                                                     rtol_NK=rtol_NK)   

    #         abs_increments = np.abs(simplified_c - simplified_c1)
    #         # dcurrents = np.abs(simplified_c1-self.currents_vec)
    #         # vals_for_check = np.where(dcurrents>threshold, dcurrents, threshold)
    #         # rel_residuals = np.abs(res)#/vals_for_check
    #         control = np.any(abs_increments>atol_currents)
    #         # control += np.any(rel_residuals>rtol_residuals)
    #         control += np.any(np.abs(self.ddJ)>atol_J)         
    #         if verbose:
    #             print('max currents change = ', np.max(abs_increments))
    #             print('max J direction change = ', np.max(np.abs(self.ddJ)), np.linalg.norm(self.ddJ))
    #             print('max circuit eq residual (dim of currents) = ', np.argmax(abs(res)), res)
    #             print(simplified_c1 - self.currents_vec_m1)

    #         iterative_steps += 1

    #         simplified_c = 1.0*simplified_c1
        
    #     self.time += self.dt_step
    #     self.step_complete_assign(simplified_c)

    #     if use_extrapolation:
    #         self.guess_J_from_extrapolation(alpha=alpha, rtol_NK=rtol_NK)

    #     flag = check_against_the_wall(jtor=self.profiles2.jtor, 
    #                                   boole_mask_outside_limiter=self.mask_outside_limiter)

    #     return flag



    # def nl_mix_unit(self, active_voltage_vec,
    #                              Rp, 
    #                              rtol_NK,
    #                              n_k=10, # max number of basis vectors (must be less than number of modes + 1)
    #                              conv_crit=.1,   #add basis vector 
    #                                             #if unexplained orthogonal component is larger than
    #                              max_collinearity=.3,
    #                              grad_eps=.005, #infinitesimal step
    #                              clip=3):

    #     simplified_c, res = self.iterative_unit(active_voltage_vec=active_voltage_vec,
    #                                              Rp=Rp, 
    #                                              rtol_NK=rtol_NK)
        
    #     self.Arnoldi_iteration(trial_sol=simplified_c, #trial_current expansion point
    #                             vec_direction=-res, #first vector for current basis
    #                             Fresidual=res, #circuit eq. residual at trial_current expansion point: Fresidual(trial_current)
    #                             Fresidual_function=self.Fresidual,
    #                             active_voltage_vec=active_voltage_vec,
    #                             n_k=n_k, # max number of basis vectors (must be less than number of modes + 1)
    #                             conv_crit=conv_crit,   #add basis vector 
    #                                             #if unexplained orthogonal component is larger than
    #                             max_collinearity=max_collinearity,
    #                             grad_eps=grad_eps, #infinitesimal step
    #                             clip=clip)

    #     simplified_c1 = simplified_c + self.d_sol_step
    #     res1 = self.Fresidual(trial_currents=simplified_c1, 
    #                          active_voltage_vec=active_voltage_vec, 
    #                          rtol_NK=rtol_NK)   
    #     return simplified_c1, res1




    # def nl_step_mix(self, active_voltage_vec, 
    #                              alpha=.8, 
    #                              rtol_NK=5e-4,
    #                              atol_increments=1e-3,
    #                              rtol_residuals=1e-3,
    #                              n_k=10, # max number of basis vectors (must be less than number of modes + 1)
    #                              conv_crit=.1,   #add basis vector 
    #                                             #if unexplained orthogonal component is larger than
    #                              max_collinearity=.3,
    #                              grad_eps=.005, #infinitesimal step
    #                              clip=3,
    #                              return_n_steps=False,
    #                              verbose=False,
    #                              threshold=.001):
        
    #     Rp = self.calc_plasma_resistance(self.J0, self.J0)
        
    #     simplified_c, res = self.nl_mix_unit(active_voltage_vec=active_voltage_vec,
    #                                                 Rp=Rp, 
    #                                                 rtol_NK=rtol_NK,
    #                                                 n_k=n_k, # max number of basis vectors (must be less than number of modes + 1)
    #                                                 conv_crit=conv_crit,   #add basis vector 
    #                                                             #if unexplained orthogonal component is larger than
    #                                                 max_collinearity=max_collinearity,
    #                                                 grad_eps=grad_eps, #infinitesimal step
    #                                                 clip=clip)

    #     dcurrents = np.abs(simplified_c-self.currents_vec)
    #     vals_for_check = np.where(dcurrents>threshold, dcurrents, threshold)                                        
        
    #     iterative_steps = 0
    #     control = 1
    #     while control:
    #         self.dJ = (1-alpha)*self.dJ + alpha*(self.reduce_normalize(self.profiles2.jtor - self.profiles1.jtor))
    #         simplified_c1, res = self.nl_mix_unit(active_voltage_vec=active_voltage_vec,
    #                                                 Rp=Rp, 
    #                                                 rtol_NK=rtol_NK,
    #                                                 n_k=n_k, # max number of basis vectors (must be less than number of modes + 1)
    #                                                 conv_crit=conv_crit,   #add basis vector 
    #                                                             #if unexplained orthogonal component is larger than
    #                                                 max_collinearity=max_collinearity,
    #                                                 grad_eps=grad_eps, #infinitesimal step
    #                                                 clip=clip)

    #         abs_increments = np.abs(simplified_c-simplified_c1)
    #         rel_residuals = np.abs(res)/vals_for_check
    #         control = np.any(abs_increments>atol_increments)
    #         control += np.any(rel_residuals>rtol_residuals)            
    #         if verbose:
    #             print(np.mean(abs_increments), np.mean(rel_residuals))

    #         iterative_steps += 1

    #         simplified_c = 1.0*simplified_c1
        
    #     self.time += self.dt_step
    #     self.step_complete_assign(simplified_c)

    #     if return_n_steps:
    #         return iterative_steps

    
    # def nl_step_nk_dJ(self, trial_sol, #trial_current expansion point
    #                     active_voltage_vec,
    #                     n_k=10, # max number of basis vectors (must be less than number of modes + 1)
    #                     conv_crit=.2,   #add basis vector 
    #                                     #if unexplained orthogonal component is larger than
    #                     max_collinearity=.3,
    #                     grad_eps=.5, #infinitesimal step
    #                     clip=3,
    #                     rtol_NK=1e-5,
    #                     atol_currents=1e-3,
    #                     atol_J=1e-3,
    #                     verbose=False):
        
    #     self.Rp = self.calc_plasma_resistance(self.J0, self.J0)

    #     resJ = self.Fresidual_nk_dJ(trial_sol, active_voltage_vec=active_voltage_vec, rtol_NK=rtol_NK)
        

    #     simplified_c = 1.0*self.simplified_c1
    #     # dcurrents = np.abs(self.simplified_c1-self.currents_vec)
    #     # vals_for_check = np.where(dcurrents>threshold, dcurrents, threshold)

    #     iterative_steps = 0
    #     control = 1
    #     while control:
    #         self.Arnoldi_iteration(trial_sol=trial_sol, #trial_current expansion point
    #                                 vec_direction=-resJ, #first vector for current basis
    #                                 Fresidual=resJ, #circuit eq. residual at trial_current expansion point: Fresidual(trial_current)
    #                                 Fresidual_function=self.Fresidual_nk_dJ,
    #                                 active_voltage_vec=active_voltage_vec,
    #                                 n_k=n_k, # max number of basis vectors (must be less than number of modes + 1)
    #                                 conv_crit=conv_crit,   #add basis vector 
    #                                                 #if unexplained orthogonal component is larger than
    #                                 max_collinearity=max_collinearity,
    #                                 grad_eps=grad_eps, #infinitesimal step
    #                                 clip=clip,
    #                                 rtol_NK=rtol_NK)
    #         print(self.coeffs)
    #         trial_sol += self.d_sol_step
    #         resJ = self.Fresidual_nk_dJ(trial_sol, 
    #                          active_voltage_vec=active_voltage_vec, 
    #                          rtol_NK=rtol_NK)   
            
            
    #         abs_increments = np.abs(simplified_c-self.simplified_c1)
    #         # dcurrents = np.abs(simplified_c1-self.currents_vec)
    #         # vals_for_check = np.where(dcurrents>threshold, dcurrents, threshold)
    #         rel_residuals = np.abs(self.residual)#/vals_for_check
    #         control = np.any(abs_increments>atol_currents)
    #         # control += np.any(rel_residuals>rtol_residuals)
    #         control += np.any(resJ>atol_J)       
    #         if verbose:
    #             print('max currents change = ', np.max(abs_increments))
    #             print('max J direction change = ', np.max(np.abs(resJ)))
    #             print('max circuit eq residual (dim of currents) = ', np.max(rel_residuals))

    #         iterative_steps += 1

    #         simplified_c = 1.0*self.simplified_c1
        
    #     self.time += self.dt_step
    #     self.step_complete_assign(simplified_c)
        
    #     flag = check_against_the_wall(jtor=self.profiles2.jtor, 
    #                                   boole_mask_outside_limiter=self.mask_outside_limiter)

    #     return flag
    

    def LSQP(self, Fresidual, nFresidual, G, Q, clip, threshold=.99, clip_hard=1.):
        #solve the least sq problem in coeffs: min||G*coeffs+Fresidual||^2
        self.coeffs = np.matmul(np.matmul(np.linalg.inv(np.matmul(G.T, G)),
                                     G.T), -Fresidual)                            
        self.coeffs = np.clip(self.coeffs, -clip, clip)
        self.explained_res = np.sum(G*self.coeffs[np.newaxis,:], axis=1) 
        self.rel_unexpl_res = np.linalg.norm(self.explained_res + Fresidual)/nFresidual
        if self.rel_unexpl_res > threshold:
            self.coeffs = np.clip(self.coeffs, -clip_hard, clip_hard)
        self.d_sol_step = np.sum(Q*self.coeffs[np.newaxis,:], axis=1)


    def Arnoldi_unit(self,  trial_sol, #trial_current expansion point
                            vec_direction, #first vector for current basis
                            Fresidual, #circuit eq. residual at trial_current expansion point: Fresidual(trial_current)
                            Fresidual_function,
                            active_voltage_vec,
                            grad_coeff,
                            rtol_NK
                            ):

        candidate_d_sol = grad_coeff*vec_direction/np.linalg.norm(vec_direction)
        print('norm candidate step', np.linalg.norm(candidate_d_sol))
        candidate_sol = trial_sol + candidate_d_sol
        # candidate_sol /= np.sum(candidate_sol)
        ri = Fresidual_function(candidate_sol, active_voltage_vec=active_voltage_vec, rtol_NK=rtol_NK)
        lvec_direction = ri - Fresidual

        self.Q[:,self.n_it] = 1.0*candidate_d_sol
        # self.Q[:,self.n_it] = candidate_sol - trial_sol
        self.Qn[:,self.n_it] = self.Q[:,self.n_it]/np.linalg.norm(self.Q[:,self.n_it])
        
        self.G[:,self.n_it] = 1.0*lvec_direction
        self.Gn[:,self.n_it] = self.G[:,self.n_it]/np.linalg.norm(self.G[:,self.n_it])

        #orthogonalize residual 
        lvec_direction -= np.sum(np.sum(self.Qn[:,:self.n_it+1]*lvec_direction[:,np.newaxis], axis=0, keepdims=True)*self.Qn[:,:self.n_it+1], axis=1)
        return lvec_direction


    def Arnoldi_iteration(self, trial_sol, #trial_current expansion point
                                vec_direction, #first vector for current basis
                                Fresidual, #circuit eq. residual at trial_current expansion point: Fresidual(trial_current)
                                Fresidual_function,
                                active_voltage_vec,
                                n_k=5, # max number of basis vectors (must be less than number of modes + 1)
                                conv_crit=.1,   #add basis vector 
                                                #if unexplained orthogonal component is larger than
                                max_collinearity=.3,
                                grad_eps=.3, #infinitesimal step size, when compared to norm(trial)
                                clip=3,
                                rtol_NK=1e-9):
        
        nFresidual = np.linalg.norm(Fresidual)
        problem_d = len(trial_sol)

        #basis in Psi space
        self.Q = np.zeros((problem_d, n_k+1))
        #orthonormal basis in Psi space
        self.Qn = np.zeros((problem_d, n_k+1))
        #basis in grandient space
        self.G = np.zeros((problem_d, n_k+1))
        #basis in grandient space
        self.Gn = np.zeros((problem_d, n_k+1))
        
        self.n_it = 0
        self.n_it_tot = 0
        grad_coeff = grad_eps*nFresidual

        print('norm trial_sol', np.linalg.norm(trial_sol))

        control = 1
        while control:
            # do step
            vec_direction = self.Arnoldi_unit(trial_sol, #trial_current expansion point
                                                vec_direction, #first vector for current basis
                                                Fresidual, #circuit eq. residual at trial_current expansion point: Fresidual(trial_current)
                                                Fresidual_function,
                                                active_voltage_vec,
                                                grad_coeff,#/(self.n_it+1)**1.2,
                                                rtol_NK=rtol_NK
                                                )
            collinear_control = 1 - np.any(np.sum(self.Gn[:,:self.n_it]*self.Gn[:,self.n_it:self.n_it+1], axis=0) > max_collinearity)
            self.n_it_tot += 1
            if collinear_control:
                self.n_it += 1
                self.LSQP(Fresidual, nFresidual, G=self.G[:,:self.n_it], Q=self.Q[:,:self.n_it], clip=clip)
                # rel_unexpl_res = np.linalg.norm(self.explained_res + Fresidual)/nFresidual
                print('rel_unexpl_res', self.rel_unexpl_res)
                arnoldi_control = (self.rel_unexpl_res > conv_crit)
            else:
                print('collinear!')
                # self.currents_nk_psi = self.currents_nk_psi[:,:-1]

            control = arnoldi_control*(self.n_it_tot<n_k)
            # if rel_unexpl_res > .6:
            #     clip = 1.5
            # self.LSQP(Fresidual, G=self.G[:,:self.n_it], Q=self.Q[:,:self.n_it], clip=clip)


    











