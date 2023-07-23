import numpy as np

from . import machine_config
from .machine_config import coils_order

from copy import deepcopy

from .circuit_eq_metal import metal_currents
from .circuit_eq_plasma import plasma_current

from . import nk_solver
from .simplified_solve import simplified_solver_J1
from .linear_solve import linear_solver
from . import plasma_grids
from . import extrapolate

from .GSstaticsolver import NKGSsolver

# from .faster_shape import shapes
# from .faster_shape import self.plasma_grids.check_if_outside_domain

import matplotlib.pyplot as plt


from scipy.signal import convolve2d


class nl_solver:
    # interfaces the circuit equation with freeGS NK solver 
    # executes dt evolution of fully non linear equations
    
    
    def __init__(self, profiles, eq, 
                 max_mode_frequency, 
                 full_timestep=.0001,
                 max_internal_timestep=.0001, 
                 plasma_resistivity=1e-6,
                 plasma_norm_factor=1000,
                 plasma_domain_mask=None,
                 nbroad=3,
                #  initial_guess='linearised_solution',
                #  solution_method='NK_on_psi_and_currents',
                #  update_linearization=True,
                #  extrapolator_input_size=4,
                #  extrapolator_order=1,
                 dIydI=None,
                 automatic_timestep=False,
                 mode_removal=True,
                 min_dIy_dI=1):

        

        self.nx = np.shape(eq.R)[0]
        self.ny = np.shape(eq.R)[1]
        self.eqR = eq.R
        self.eqZ = eq.Z
        # area factor for Iy
        dR = eq.R[1, 0] - eq.R[0, 0]
        dZ = eq.Z[0, 1] - eq.Z[0, 0]
        self.dRdZ = dR*dZ

        # self.eq = eq
        # self.profiles = profiles
        

        self.plasma_grids = plasma_grids.Grids(eq, plasma_domain_mask)
        # mask identifying if plasma is hitting the wall
        self.plasma_domain_mask = self.plasma_grids.plasma_domain_mask
        self.plasma_domain_size = np.sum(self.plasma_grids.plasma_domain_mask)
        self.plasma_against_wall = 0

        
        #instantiate solver on eq's domain
        self.NK = NKGSsolver(eq)

        

        # profiles are kept constant during evolution
        # paxis, fvac and alpha values are taken from ICs and kept fixed thereafter
        self.get_profiles_values(profiles)
        
        
        # #eq needs only have same tokamak and grid structure as those that will be evolved
        # #when calling time evolution on some initial eq, the eq object that is actually modified is eq1
        # #with profiles1 for changes in the plasma current
        # self.eq1 = deepcopy(eq)
        # self.profiles1 = deepcopy(profiles)   
        # # full plasma current density (make sure you start from an actual equilibrium!)
                                   

        # #the numerical method used to solve for the time evolution uses iterations
        # #the eq object that is modified during the iterations is eq2
        # #with profiles2 for changes in the plasma current
        # self.eq2 = deepcopy(eq)
        # self.profiles2 = deepcopy(profiles) 


        # for i,labeli in enumerate(coils_order):
        #     self.eq1.tokamak[labeli].control = False
        #     self.eq2.tokamak[labeli].control = False
        
        
        self.dt_step = full_timestep
        self.plasma_norm_factor = plasma_norm_factor
    
        self.max_internal_timestep = max_internal_timestep
        self.max_mode_frequency = max_mode_frequency
        self.reset_plasma_resistivity(plasma_resistivity, eqR=self.eqR)


        #qfe is an instance of the quants_for_emulation class
        #calculates fluxes of plasma on coils and on itself
        #as well as all other time-dependent quantities needed for time evolution
        # self.qfe = quants_for_emu.quants_for_emulation(eq)
        
        # to calculate residual of metal circuit eq
        self.evol_metal_curr = metal_currents(flag_vessel_eig=1,
                                                flag_plasma=1,
                                                plasma_grids=self.plasma_grids,
                                                max_mode_frequency=self.max_mode_frequency,
                                                max_internal_timestep=self.max_internal_timestep,
                                                full_timestep=self.dt_step)
        # this is the number of independent normal mode currents associated to max_mode_frequency
        self.n_metal_modes = self.evol_metal_curr.n_independent_vars
        self.arange_currents = np.arange(self.n_metal_modes+1)
        

        # to calculate residual of plasma collapsed circuit eq
        self.evol_plasma_curr = plasma_current(plasma_grids=self.plasma_grids,
                                               Rm12=np.diag(self.evol_metal_curr.Rm12), 
                                               V=self.evol_metal_curr.V,
                                               plasma_resistance_1d=self.plasma_resistance_1d,
                                               Mye=self.evol_metal_curr.Mey.T)


        # self.simplified_solver_dJ = simplified_solver_dJ(Lambdam1=self.evol_metal_curr.Lambdam1, 
        #                                                     Vm1Rm12=np.matmul(self.evol_metal_curr.Vm1, np.diag(self.evol_metal_curr.Rm12)), 
        #                                                     Mey=self.evol_metal_curr.Mey, 
        #                                                     Myy=self.evol_plasma_curr.Myy,
        #                                                     plasma_norm_factor=self.plasma_norm_factor,
        #                                                     plasma_resistance_1d=self.plasma_resistance_1d,
        #                                                     max_internal_timestep=self.max_internal_timestep,
        #                                                     full_timestep=self.dt_step)
        # self.n_step_range = np.arange(self.simplified_solver_dJ.solver.n_steps)[::-1][np.newaxis] + 1                                                    
        
        self.simplified_solver_J1 = simplified_solver_J1(Lambdam1=self.evol_metal_curr.Lambdam1, 
                                                            Vm1Rm12=np.matmul(self.evol_metal_curr.Vm1, np.diag(self.evol_metal_curr.Rm12)), 
                                                            Mey=self.evol_metal_curr.Mey, 
                                                            Myy=self.evol_plasma_curr.Myy,
                                                            plasma_norm_factor=self.plasma_norm_factor,
                                                            plasma_resistance_1d=self.plasma_resistance_1d,
                                                            # this is used with no internal step subdivision, to help nonlinear convergence 
                                                            max_internal_timestep=self.dt_step,
                                                            full_timestep=self.dt_step)
        
        


        

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
        # self.eig_currents_vec = self.evol_metal_curr.IvesseltoId(Ivessel=self.vessel_currents_vec)
        
        # norm plasma current is divided by plasma_norm_factor to keep homogeneity
        # self.norm_plasma_current = eq.plasmaCurrent()/self.plasma_norm_factor
        # vector of currents on which non linear system is solved is self.currents_vec
        # currents_vec = np.zeros(self.n_metal_modes + 1)
        # currents_vec[:self.n_metal_modes] = 1.0*self.eig_currents_vec
        # currents_vec[-1] = self.norm_plasma_current
        self.currents_vec = np.zeros(self.n_metal_modes + 1)
        # self.new_currents = 1.0*currents_vec
        self.circuit_eq_residual = np.zeros(self.n_metal_modes + 1)
        
        # self.mapd = np.zeros_like(self.eq1.R)

        self.step_no = 0

        self.ones_to_broaden = np.ones((nbroad,nbroad))

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


        # self.linearised_flag = False
        # self.extrapolator_flag = False
        # if initial_guess=='linearised_solution':
        # self.linearised_flag = True
        self.dIydI = dIydI
        self.linearised_sol = linear_solver(Lambdam1=self.evol_metal_curr.Lambdam1, 
                                            Vm1Rm12=np.matmul(self.evol_metal_curr.Vm1, np.diag(self.evol_metal_curr.Rm12)), 
                                            Mey=self.evol_metal_curr.Mey, 
                                            Myy=self.evol_plasma_curr.Myy,
                                            plasma_norm_factor=self.plasma_norm_factor,
                                            plasma_resistance_1d=self.plasma_resistance_1d,
                                            max_internal_timestep=self.max_internal_timestep,
                                            full_timestep=self.dt_step)
        # self.set_initial_trial_solution = self.set_initial_trial_solution_linearization
        # self.update_linearization = update_linearization
        # if update_linearization:
        self.step_no = 0
        # self.update_n_steps = update_n_steps
        # self.current_record = np.zeros((self.n_metal_modes, self.n_metal_modes+1))
        # self.Iy_record = np.zeros((self.n_metal_modes, self.plasma_domain_size))


        # elif initial_guess=='extrapolated_solution':
        #     self.extrapolator_flag = True
        #     self.extrapolator = extrapolate.extrapolator(input_size=extrapolator_input_size,
        #                                              interpolation_order=extrapolator_order,
        #                                              parallel_dims=self.n_metal_modes+1)
        #     self.extrapolator_input_size = extrapolator_input_size
        #     self.set_initial_trial_solution = self.set_initial_trial_solution_extrapolation


        # if solution_method=='NK_on_psi_and_currents':
        self.psi_nk_solver = nk_solver.nksolver(self.nx * self.ny)
        self.currents_nk_solver = nk_solver.nksolver(self.n_metal_modes + 1)
        # self.stepper = self.nl_step_nk_psi_curr
        # elif solution_method=='NK_on_currents_GS':
        #     self.currents_nk_solver = nk_solver.nksolver(self.n_metal_modes + 1)
        #     self.stepper = self.nl_step_nk_curr_GS
        # elif solution_method=='linearised_only':
        #     if self.linearised_flag is False:
        #         print('initial guess has to be set to \'linearised solution\' for this solution method, please adjust')
        #     self.stepper = self.linear_step
        # else:
        #     print('solution method not recognised, use either')
        #     print('NK_on_psi_and_currents')
        #     print('NK_on_currents_GS')
        #     print('linearised_only')

        if automatic_timestep or mode_removal:
            self.initialize_from_ICs(eq, profiles, 
                                    rtol_NK=1e-8,
                                    noise_level=0, 
                                    dIydI=dIydI)
            
        if mode_removal:
            self.selected_modes_mask = np.linalg.norm(self.dIydI, axis=0)>min_dIy_dI
            self.selected_modes_mask = np.concatenate((np.ones(machine_config.n_active_coils),
                                                       self.selected_modes_mask[machine_config.n_active_coils:-1],
                                                       np.ones(1))).astype(bool)
            self.dIydI = self.dIydI[:, self.selected_modes_mask]
            self.updated_dIydI = np.copy(self.dIydI)
            self.ddIyddI = self.ddIyddI[self.selected_modes_mask]
            self.selected_modes_mask = np.concatenate((self.selected_modes_mask[:-1],
                                                       np.zeros(machine_config.n_coils-self.n_metal_modes))).astype(bool)
            self.remove_modes(self.selected_modes_mask)

        self.linearised_sol.calculate_linear_growth_rate()
        if len(self.linearised_sol.growth_rates):
            print('This equilibrium has a linear growth rate of 1/', abs(self.linearised_sol.growth_rates[0]), 's')
        else: 
            print('No unstable modes found.', 
                    'Either plasma is stable or it is Alfven unstable.', 
                    'Try adding more passive modes.')

        if automatic_timestep is None or automatic_timestep is False:
            print('The solver\'s timestep was set at', self.dt_step,
                        'If necessary, reset.')
        else:
            if len(self.linearised_sol.growth_rates):
                dt_step = abs(self.linearised_sol.growth_rates[0]*automatic_timestep)
                self.reset_timestep(full_timestep=dt_step, 
                                    max_internal_timestep=dt_step/10)
                print('The solver\'s timestep has been reset at', self.dt_step)
            else:
                print('No unstable modes found. Impossible to automatically set timestep!')



    def remove_modes(self, selected_modes_mask):
        print('The input min_dIy_dI corresponds to keeping', np.sum(selected_modes_mask),
              'out of the original', self.n_metal_modes, 'metal modes.')
        self.evol_metal_curr.initialize_for_eig(selected_modes_mask)
        self.n_metal_modes = self.evol_metal_curr.n_independent_vars
        self.arange_currents = np.arange(self.n_metal_modes+1)
        self.currents_vec = np.zeros(self.n_metal_modes + 1)
        self.circuit_eq_residual = np.zeros(self.n_metal_modes + 1)
        self.currents_nk_solver = nk_solver.nksolver(self.n_metal_modes + 1)

        self.evol_plasma_curr.reset_modes(V=self.evol_metal_curr.V)

        self.simplified_solver_J1 = simplified_solver_J1(Lambdam1=self.evol_metal_curr.Lambdam1, 
                                                         Vm1Rm12=np.matmul(self.evol_metal_curr.Vm1, np.diag(self.evol_metal_curr.Rm12)), 
                                                            Mey=self.evol_metal_curr.Mey, 
                                                            Myy=self.evol_plasma_curr.Myy,
                                                            plasma_norm_factor=self.plasma_norm_factor,
                                                            plasma_resistance_1d=self.plasma_resistance_1d,
                                                            # this is used with no internal step subdivision, to help nonlinear convergence 
                                                            max_internal_timestep=self.dt_step,
                                                            full_timestep=self.dt_step)

        self.linearised_sol = linear_solver(Lambdam1=self.evol_metal_curr.Lambdam1, 
                                            Vm1Rm12=self.simplified_solver_J1.Vm1Rm12, 
                                            Mey=self.evol_metal_curr.Mey, 
                                            Myy=self.evol_plasma_curr.Myy,
                                            plasma_norm_factor=self.plasma_norm_factor,
                                            plasma_resistance_1d=self.plasma_resistance_1d,
                                            max_internal_timestep=self.max_internal_timestep,
                                            full_timestep=self.dt_step)
        
        self.linearised_sol.set_linearization_point(dIydI=self.dIydI,
                                                    hatIy0=self.broad_J0)

       




    def set_linear_solution(self, active_voltage_vec):
        # solves linearised problem in the currents  
        # solves GS for those currents
        # assigns self.trial_currents and self.trial_plasma_psi accordingly

        self.trial_currents = self.linearised_sol.stepper(It=self.currents_vec, 
                                                          active_voltage_vec=active_voltage_vec)
        self.assign_currents_solve_GS(self.trial_currents, self.rtol_NK)
        self.trial_plasma_psi = np.copy(self.eq2.plasma_psi)   


    # def set_initial_trial_solution_extrapolation(self, active_voltage_vec):
        
    #     if (self.step_no > self.extrapolator_input_size):
    #         self.trial_currents = np.copy(self.currents_guess)
    #         self.assign_currents_solve_GS(self.trial_currents, self.rtol_NK)
            
    #     else:
    #         self.trial_currents = self.hatIy1_iterative_cycle(self.hatIy,
    #                                                           active_voltage_vec,
    #                                                           rtol_NK=self.rtol_NK)
            
    #     self.trial_plasma_psi = np.copy(self.eq2.plasma_psi)    
        
    def prepare_build_dIydI_j(self, j, rtol_NK, target_dIy, starting_dI, min_curr=1e-4, max_curr=10):
        current_ = np.copy(self.currents_vec)
        current_[j] += starting_dI
        self.assign_currents_solve_GS(current_, rtol_NK)

        dIy_0 = self.plasma_grids.Iy_from_jtor(self.profiles2.jtor) - self.Iy
        final_dI = starting_dI*target_dIy/np.linalg.norm(dIy_0)
        final_dI = np.clip(final_dI, min_curr, max_curr)
        self.final_dI_record[j] = final_dI


    def build_dIydI_j(self, j, rtol_NK):
       
        final_dI = self.final_dI_record[j]
        self.current_at_last_linearization[j] = self.currents_vec[j]

        current_ = np.copy(self.currents_vec)
        current_[j] += final_dI
        self.assign_currents_solve_GS(current_, rtol_NK)

        dIy_1 = self.plasma_grids.Iy_from_jtor(self.profiles2.jtor) - self.Iy
        dIydIj = dIy_1/final_dI

        # noise = noise_level*np.random.random(self.n_metal_modes+1)
        # current_[j] += final_dI
        # self.assign_currents_solve_GS(current_, rtol_NK)
        # dIydIj_2 = (self.plasma_grids.Iy_from_jtor(self.profiles2.jtor) - dIy_1 - self.Iy)/final_dI

        # self.ddIyddI[j] = np.linalg.norm(dIydIj_2 - dIydIj)/final_dI
        
        print(j, 'deltaI = ', final_dI, 'norm(deltaIy) =', np.linalg.norm(dIy_1))
        # print('ddIydI = ', self.ddIyddI[j])
        return dIydIj
    

    def build_dIydI_linearization(self, eq, profile, rtol_NK, target_dIy=10, starting_dI=.5):
        print('I\'m building the linearization. This might take a minute or two.')
        self.NK.solve(eq, profile, target_relative_tolerance=rtol_NK)
        self.build_current_vec(eq, profile)

        self.Iy = self.plasma_grids.Iy_from_jtor(profile.jtor)
        self.dIydI = np.zeros((self.plasma_domain_size, self.n_metal_modes+1))
        self.ddIyddI = np.zeros(self.n_metal_modes+1)
        self.final_dI_record = np.zeros(self.n_metal_modes+1)

        for j in self.arange_currents:
            self.prepare_build_dIydI_j(j, rtol_NK, target_dIy, starting_dI)
            
        for j in self.arange_currents:
            self.dIydI[:,j] = self.build_dIydI_j(j, rtol_NK)
        self.updated_dIydI = np.copy(self.dIydI)
        self.norm_updated_dIydI = np.linalg.norm(self.updated_dIydI)


    def calculate_linear_growth_rates(self, rtol=1e-8):
        self.build_dIydI_linearization(self.eq1, self.profiles1, rtol)
        growth_rates = np.sort(np.linalg.eig(self.linearised_sol.solver.Mmatrix)[0])
        self.growth_rates = growth_rates[growth_rates < 0]
        return self.growth_rates 

        
    def reset_plasma_resistivity(self, plasma_resistivity, eqR):
        self.plasma_resistivity = plasma_resistivity
        plasma_resistance_matrix = eqR*(2*np.pi/self.dRdZ)*self.plasma_resistivity
        self.plasma_resistance_1d = plasma_resistance_matrix[self.plasma_domain_mask]
    

    def calc_lumped_plasma_resistance(self, norm_red_Iy0, norm_red_Iy1):
        lumped_plasma_resistance = np.sum(self.plasma_resistance_1d*norm_red_Iy0*norm_red_Iy1)
        return lumped_plasma_resistance
    

    def reset_timestep(self, full_timestep, max_internal_timestep):
        self.dt_step = full_timestep
        self.max_internal_timestep = max_internal_timestep
       
        self.evol_metal_curr.reset_mode(flag_vessel_eig=1,
                                        flag_plasma=1,
                                        plasma_grids=self.plasma_grids,
                                        max_mode_frequency=self.max_mode_frequency,
                                        max_internal_timestep=max_internal_timestep,
                                        full_timestep=full_timestep)
        
        self.simplified_solver_J1.reset_timesteps(full_timestep=full_timestep,
                                                  max_internal_timestep=full_timestep)
        
        self.linearised_sol.reset_timesteps(full_timestep=full_timestep,
                                            max_internal_timestep=max_internal_timestep)
    

    def get_profiles_values(self, profiles):
        #this allows to use the same instantiation of the time_evolution_class
        #on ICs with different paxis, fvac and alpha values
        #if these are different from those used when first instantiating the class
        #just call this function on the new profile object:
        self.paxis = profiles.paxis
        self.fvac = profiles.fvac
        self.alpha_m = profiles.alpha_m
        self.alpha_n = profiles.alpha_n

    # def Iyplasmafromjtor(self, jtor):
    #     red_Iy = jtor[self.plasma_domain_mask]*self.dRdZ
    #     return red_Iy

    # def reduce_normalize(self, jtor, epsilon=1e-6):
    #     red_Iy = jtor[self.plasma_domain_mask]/(np.sum(jtor) + epsilon)
    #     return red_Iy
    
    # def reduce_normalize_l2(self, jtor):
    #     red_Iy = jtor[self.plasma_domain_mask]/np.linalg.norm(jtor)
    #     return red_Iy

    # def rebuild_grid_map(self, red_vec):
    #     self.mapd[self.plasma_grids.idxs_mask[0], self.plasma_grids.idxs_mask[1]] = red_vec
    #     return self.mapd

    def get_vessel_currents(self, eq):
        eq_currents = eq.tokamak.getCurrents()
        for i,labeli in enumerate(coils_order):
            self.vessel_currents_vec[i] = eq_currents[labeli]


    def build_current_vec(self, eq, profile):
        #gets initial currents from ICs, note these are before mode truncation!
        self.get_vessel_currents(eq)
        
        # vector of currents on which non linear system is solved is self.currents_vec
        self.currents_vec[:self.n_metal_modes] = self.evol_metal_curr.IvesseltoId(Ivessel=self.vessel_currents_vec)
        self.currents_vec[-1] = profile.Ip/self.plasma_norm_factor

        self.currents_vec_m1 = np.copy(self.currents_vec)
        # vector of plasma current density (already reduced on grid from plasma_grids)

        # self.eq1.plasma_psi = 1.0*eq.plasma_psi
        # # solve GS again with vessel currents you get after truncating modes:
        # self.assign_currents(self.currents_vec, self.eq1, self.profiles1)
        # self.NK.solve(self.eq1, self.profiles1, target_relative_tolerance=rtol_NK)

        # self.eq2.plasma_psi = 1.0*self.eq1.plasma_psi



    def assign_vessel_noise(self, eq, noise_level, noise_vec=None):
        # adds gaussian noise to vessel normal mode currents,
        # and assigns to eq.tokamak
        # rms is 'noise_level'
        # active coil currents are not affected
        # needs self.vessel_currents_vec

        self.get_vessel_currents(eq)

        if noise_vec is None:
            noise_vec = np.random.randn(self.n_metal_modes - machine_config.n_active_coils)
            noise_vec *= noise_level
            noise_vec = np.concatenate((np.zeros(machine_config.n_active_coils), noise_vec))
            self.noise_vec = noise_vec
        
        # calculate vessel currents from normal modes and assign
        self.vessel_currents_vec += self.evol_metal_curr.IdtoIvessel(Id=noise_vec)
        for i,labeli in enumerate(coils_order[machine_config.n_active_coils:]):
            # runs on passive only
            eq.tokamak[labeli].current = self.vessel_currents_vec[i + machine_config.n_active_coils]
        


    def initialize_from_ICs(self, 
                            eq, profile, 
                            rtol_NK=1e-8, 
                            noise_level=.001,
                            # new_seed=False,
                            noise_vec=None,
                            dIydI=None,
                            update_linearization=False,
                            update_n_steps=16,
                            threshold_svd=.1,
                            max_dIy_update=.01,
                            max_updates=6
                            ): 
                            # eq for ICs, with all properties set up at time t=0, 
                            # ie with eq.tokamak.currents = I(t) and eq.plasmaCurrent = I_p(t) 
        
        self.step_counter = 0
        self.currents_guess = False
        self.rtol_NK = rtol_NK
        self.update_n_steps = update_n_steps
        self.update_linearization = update_linearization
        self.threshold_svd = threshold_svd
        self.max_dIy_update = max_dIy_update
        self.max_updates = max_updates 

        #get profile parametrization
        self.get_profiles_values(profile)

        self.eq1 = deepcopy(eq)
        self.profiles1 = deepcopy(profile)  

        # perturb passive structures
        if (noise_level>0) or (noise_vec is not None):
            self.assign_vessel_noise(self.eq1, noise_level, noise_vec)

        #ensure it's a GS solution
        self.NK.solve(self.eq1, self.profiles1, target_relative_tolerance=rtol_NK)
         # self.jtor_m1 = np.copy(self.profiles1.jtor)
        self.Iy = self.plasma_grids.Iy_from_jtor(self.profiles1.jtor)
        self.hatIy = self.plasma_grids.normalize_sum(self.Iy)
        self.hatIy1 = np.copy(self.hatIy)
        self.broad_J0 = convolve2d(self.profiles1.jtor, self.ones_to_broaden, mode='same')
        self.broad_J0 = self.plasma_grids.hat_Iy_from_jtor(self.broad_J0)

         
        self.eq2 = deepcopy(self.eq1)
        self.profiles2 = deepcopy(self.profiles1)   


        #prepare currents
        self.build_current_vec(self.eq1, self.profiles1)
        self.current_at_last_linearization = np.copy(self.currents_vec)
        

        self.time = 0
        self.step_no = -1

        
        if dIydI is None:
            if self.dIydI is None:
                self.build_dIydI_linearization(eq=eq, profile=profile, rtol_NK=rtol_NK)
        else:
            self.dIydI = dIydI

        self.linearised_sol.set_linearization_point(dIydI=self.dIydI,
                                                    hatIy0=self.broad_J0)

        if self.update_linearization:
            self.current_record = np.zeros((self.update_n_steps, self.n_metal_modes+1))
            self.Iy_record = np.zeros((self.update_n_steps, self.plasma_domain_size))
            self.current_at_last_linearization = np.copy(self.currents_vec)


        # check if against the wall
        if self.plasma_grids.check_if_outside_domain(jtor=self.profiles1.jtor):
            print('plasma in ICs is touching the wall!')




    def run_linearization_update(self, max_dIy_update, max_updates, 
                                 target_dIy=10, starting_dI=.5):
                
        self.linearised_sol.prepare_min_update_linearization(self.current_record[:-1],
                                                             self.Iy_record[:-1],
                                                             self.threshold_svd)
        delta_dIydI = self.linearised_sol.min_update_linearization()

        compare_ = np.linalg.norm(delta_dIydI)/self.norm_updated_dIydI
        print('relative_d_dIydI', compare_)
        control = (compare_>max_dIy_update)
        
        if control:
            print('Need linearization update: starting...')
            i = 0
            urgency = abs((self.current_at_last_linearization - self.currents_vec)/(self.norm_updated_dIydI/self.ddIyddI))
            urgency = np.argsort(-urgency)  

            while control:
                j = urgency[i]
                self.updated_dIydI[:,j] = self.build_dIydI_j(j, rtol_NK=self.rtol_NK)
                self.norm_updated_dIydI = np.linalg.norm(self.updated_dIydI)
                self.linearised_sol.set_linearization_point(dIydI=self.updated_dIydI,
                                                            hatIy0=self.broad_J0)
                
                delta_dIydI = self.linearised_sol.min_update_linearization()

                compare_ = np.linalg.norm(delta_dIydI)/self.norm_updated_dIydI
                print('relative_d_dIydI', compare_)
                control = (compare_>max_dIy_update)*(i<self.max_updates)

                i += 1

                
                





    
        




    def check_linearization_and_update(self, max_dIy_update, max_updates):
        if self.step_no / self.update_n_steps < 1:
            id = self.step_no % self.update_n_steps
            self.current_record[id] = self.currents_vec
            self.Iy_record[id] = self.Iy
        else:
            self.current_record[-1] = self.currents_vec
            self.Iy_record[-1] = self.Iy
            self.current_record[:-1] = self.current_record[1:]
            self.Iy_record[:-1] = self.Iy_record[1:] 

            self.run_linearization_update(max_dIy_update, max_updates)

            

        

    def step_complete_assign(self, working_relative_tol_GS,
                                   from_linear=False):
        # assigns self.trial_currents
        # plasma_psi as self.trial_plasma_psi

        self.time += self.dt_step
        self.step_no += 1

        self.currents_vec_m1 = np.copy(self.currents_vec)
        self.Iy_m1 = np.copy(self.Iy)

        plasma_psi_step = self.eq2.plasma_psi - self.eq1.plasma_psi
        self.d_plasma_psi_step = np.amax(plasma_psi_step) - np.amin(plasma_psi_step)
        # print(self.d_plasma_psi_step)

        self.currents_vec = np.copy(self.trial_currents)
        self.assign_currents(self.currents_vec, self.eq1, self.profiles1)
        
        self.eq1.plasma_psi = np.copy(self.trial_plasma_psi)
        self.profiles1.Ip = self.trial_currents[-1]*self.plasma_norm_factor
        if from_linear:
            self.profiles1.jtor = np.copy(self.profiles2.jtor)
        else:
            # self.tokamak_psi = self.eq1.tokamak.calcPsiFromGreens(pgreen=self.eq1._pgreen)
            self.profiles1.Jtor(self.eqR, self.eqZ, self.tokamak_psi + self.trial_plasma_psi)

        self.Iy = self.plasma_grids.Iy_from_jtor(self.profiles1.jtor)
        self.hatIy = self.plasma_grids.normalize_sum(self.Iy)
        self.broad_J0 = convolve2d(self.profiles1.jtor, self.ones_to_broaden, mode='same')
        self.broad_J0 = self.plasma_grids.hat_Iy_from_jtor(self.broad_J0)

        self.rtol_NK = working_relative_tol_GS*self.d_plasma_psi_step

        if self.update_linearization:
            self.check_linearization_and_update(max_dIy_update=self.max_dIy_update,
                                                max_updates=self.max_updates)
            # if self.step_no and (self.step_no%self.update_n_steps)==0:
            #     self.delta_dIydI = self.linearised_sol.update_linearization(self.current_record[:self.update_n_steps],
            #                                                                 self.Iy_record[:self.update_n_steps],
            #                                                                 self.threshold_svd)
            #     print('linearization update:', np.linalg.norm(self.delta_dIydI), np.linalg.norm(self.linearised_sol.dIydI))
            #     if np.linalg.norm(self.delta_dIydI)<self.max_relative_dIydI_update*np.linalg.norm(self.linearised_sol.dIydI):
            #         self.linearised_sol.set_linearization_point(dIydI=self.linearised_sol.dIydI+self.delta_dIydI, hatIy0=self.broad_J0)




    # def step_complete_assign(self, trial_currents, 
    #                                trial_plasma_psi,
    #                                working_relative_tol_GS):
    #     # assigns trial_currents
    #     # and trial_plasma_psi, 
    #     # recalculates jtor accordingly

    #     self.time += self.dt_step
    #     self.currents_vec_m1 = np.copy(self.currents_vec)
    #     self.Iy_m1 = np.copy(self.Iy)

    #     self.currents_vec = np.copy(trial_currents)
    #     self.assign_currents(self.currents_vec, self.eq1, self.profiles1)
    #     self.tokamak_psi = self.eq1.tokamak.calcPsiFromGreens(pgreen=self.eq1._pgreen)
    #     self.eq1.plasma_psi = np.copy(trial_plasma_psi)
    #     self.profiles1.Jtor(self.eqR, self.eqZ, (self.tokamak_psi + trial_plasma_psi).reshape(self.nx, self.ny))

        
    #     self.Iy = 1.0*self.plasma_grids.Iy_from_jtor(self.profiles1.jtor)
    #     self.hatIy = self.plasma_grids.normalize_sum(self.Iy)

    #     self.step_no += 1

    #     self.rtol_NK = working_relative_tol_GS*self.d_plasma_psi_step

    #     # if self.extrapolator_flag:
    #     #     self.guess_currents_from_extrapolation()
        
    #     if self.update_linearization:
    #         self.check_linearization_and_update()
    #         if self.step_no and (self.step_no%self.update_n_steps)==0:
    #             delta = self.linearised_sol.update_linearization(self.current_record,
    #                                                              self.Iy_record,
    #                                                              self.threshold_svd)
    #             self.dIydI += delta
    #             self.linearised_sol.set_linearization_point(dIydI=self.dIydI, hatIy0=self.hatIy)
                




    def assign_currents(self, currents_vec, eq, profile):
        #uses currents_vec to assign currents to both plasma and tokamak in eq/profiles

        profile.Ip = self.plasma_norm_factor*currents_vec[-1]

        # calculate vessel currents from normal modes and assign
        self.vessel_currents_vec = self.evol_metal_curr.IdtoIvessel(Id=currents_vec[:-1])
        for i,labeli in enumerate(coils_order):
            eq.tokamak[labeli].current = self.vessel_currents_vec[i]
        # assign plasma current to equilibrium
        eq._current = self.plasma_norm_factor*currents_vec[-1]


    # def guess_currents_from_extrapolation(self,):
    #     # run after step is complete and assigned, prepares for next step

    #     if self.step_no >= self.extrapolator_input_size:
    #         self.currents_guess = self.extrapolator.in_out(self.currents_vec)

    #     else:
    #         self.extrapolator.set_Y(self.step_no, self.currents_vec)



    # def guess_J_from_extrapolation(self, alpha, rtol_NK):
    #     # run after step is complete and assigned, prepares for next step

    #     if self.step_no >= self.extrapolator_input_size:
    #         currents_guess = self.extrapolator.in_out(self.currents_vec)

    #         self.assign_currents(currents_vec=currents_guess, profile=self.profiles2, eq=self.eq2)
    #         self.NK.solve(self.eq2, self.profiles2, target_relative_tolerance=rtol_NK)

    #         self.hatIy1 = (1-alpha)*self.hatIy1 + alpha*self.plasma_grids.hat_Iy_from_jtor(self.profiles2.jtor)

    #     else:
    #         self.extrapolator.set_Y(self.step_no, self.currents_vec)
        
        




    
   
    

    # def F_function_nk_dJ(self, dJ,
    #                           active_voltage_vec,
    #                           rtol_NK):
        
    #     # Sp = self.calc_lumped_plasma_resistance(self.hatIy, dJ)/self.Rp
    #     self.simplified_c1 = 1.0*self.simplified_solver_dJ.stepper(It=self.currents_vec,
    #                                                         norm_red_Iy0=self.hatIy, 
    #                                                         norm_red_Iy_dot=dJ, 
    #                                                         active_voltage_vec=active_voltage_vec, 
    #                                                         Rp=self.Rp)
    #     res = 1.0*self.F_function_dJ(trial_currents=self.simplified_solver_dJ.solver.intermediate_results, 
    #                                 active_voltage_vec=active_voltage_vec, 
    #                                 rtol_NK=rtol_NK)   
    #     dJ1 = self.Iy_dot/np.linalg.norm(self.Iy_dot)
    #     return dJ1-dJ
    

    def make_broad_hatIy(self, hatIy1):
        self.broad_J0 = self.plasma_grids.rebuild_map2d(self.hatIy + hatIy1)
        self.broad_J0 = convolve2d(self.broad_J0, self.ones_to_broaden, mode='same')
        self.broad_J0 = self.plasma_grids.hat_Iy_from_jtor(self.broad_J0)


    def assign_currents_solve_GS(self, trial_currents, rtol_NK):
        self.assign_currents(trial_currents, profile=self.profiles2, eq=self.eq2)
        self.NK.solve(self.eq2, self.profiles2, target_relative_tolerance=rtol_NK)
        self.trial_Iy1 = self.plasma_grids.Iy_from_jtor(self.profiles2.jtor)
        self.Iy_dot = (self.trial_Iy1 - self.Iy)/self.dt_step
        self.Id_dot = ((trial_currents - self.currents_vec)/self.dt_step)[:-1]


    def circ_eq_residual_f(self, trial_currents, active_voltage_vec):
        self.forcing_term = self.evol_metal_curr.forcing_term_eig_plasma(active_voltage_vec=active_voltage_vec, 
                                                                         Iydot=self.Iy_dot)

        self.circuit_eq_residual[:-1] = self.evol_metal_curr.current_residual(Itpdt=trial_currents[:-1], 
                                                                                    Iddot=self.Id_dot, 
                                                                                    forcing_term=self.forcing_term)

        self.circuit_eq_residual[-1] = self.evol_plasma_curr.current_residual(red_Iy0=self.Iy, 
                                                                                red_Iy1=self.trial_Iy1,
                                                                                red_Iydot=self.Iy_dot,
                                                                                Iddot=self.Id_dot)/self.plasma_norm_factor
                


    # def F_function_dJ(self, trial_currents, active_voltage_vec, rtol_NK=1e-8):
    #     # trial_currents is the full array of intermediate results from euler solver
    #     # root problem for circuit equation
    #     # collects both metal normal modes and norm_plasma
        
    #     # current at t+dt
    #     # d_current_tpdt = np.sum(trial_currents, axis=-1)

    #     self.assign_currents_solve_GS(trial_currents, rtol_NK=rtol_NK)
    #     self.circ_eq_residual_f(trial_currents, active_voltage_vec)


    
    def currents_from_hatIy(self, hatIy1,
                                  active_voltage_vec):
        self.make_broad_hatIy(hatIy1)
        current_from_hatIy = self.simplified_solver_J1.stepper(It=self.currents_vec,
                                                            hatIy_left=self.broad_J0, 
                                                            hatIy_0=self.hatIy, 
                                                            hatIy_1=hatIy1, 
                                                            active_voltage_vec=active_voltage_vec)
        return current_from_hatIy


    def hatIy1_iterative_cycle(self, hatIy1,
                                     active_voltage_vec,
                                     rtol_NK):
        current_from_hatIy = self.currents_from_hatIy(hatIy1,
                                                active_voltage_vec)
        self.assign_currents_solve_GS(trial_currents=current_from_hatIy, 
                                      rtol_NK=rtol_NK)   
        # return current_from_hatIy


    
    def calculate_hatIy(self, trial_currents, plasma_psi):
        self.assign_currents(trial_currents, profile=self.profiles2, eq=self.eq2)
        self.tokamak_psi = (self.eq2.tokamak.calcPsiFromGreens(pgreen=self.eq2._pgreen))
        jtor_ = self.profiles2.Jtor(self.eqR, self.eqZ, self.tokamak_psi + plasma_psi)
        hat_Iy1 = self.plasma_grids.hat_Iy_from_jtor(jtor_)
        return hat_Iy1

    def F_function_curr(self, trial_currents, active_voltage_vec):
        # circuit eq as a root problem on the current values,
        # assuming fixed self.trial_plasma_psi
        hatIy1 = self.calculate_hatIy(trial_currents, self.trial_plasma_psi)
        iterated_currs = self.currents_from_hatIy(hatIy1, active_voltage_vec)
        current_res = iterated_currs - trial_currents
        return current_res                                            



    
    def calculate_hatIy_GS(self, trial_currents, rtol_NK):
        self.assign_currents_solve_GS(trial_currents, rtol_NK=rtol_NK)
        hatIy1 = self.plasma_grids.hat_Iy_from_jtor(self.profiles2.jtor)
        return hatIy1
    
    def F_function_curr_GS(self, trial_currents, active_voltage_vec, rtol_NK):
        hatIy1 = self.calculate_hatIy_GS(trial_currents, rtol_NK=rtol_NK)
        iterated_currs = self.currents_from_hatIy(hatIy1, active_voltage_vec)
        current_res = iterated_currs - trial_currents
        return current_res  
    



    def F_function_psi(self, trial_plasma_psi, active_voltage_vec, rtol_NK):
        # circuit eqs as a root problem in plasma_psi at fixed self.tokamak_psi
        jtor_ = self.profiles2.Jtor(self.eqR, self.eqZ, (self.tokamak_psi + trial_plasma_psi).reshape(self.nx, self.ny))
        hatIy1 = self.plasma_grids.hat_Iy_from_jtor(jtor_)
        # self.simplified_c = 
        self.hatIy1_iterative_cycle(hatIy1=hatIy1,
                                                        active_voltage_vec=active_voltage_vec,
                                                        rtol_NK=rtol_NK)
        psi_residual = self.eq2.plasma_psi.reshape(-1) - trial_plasma_psi
        return psi_residual
    



    def calculate_rel_tolerance_currents(self, residual, curr_eps):
        # uses self.trial_currents
        curr_step = abs(self.trial_currents - self.currents_vec_m1)
        self.curr_step = np.where(curr_step>curr_eps, curr_step, curr_eps)
        rel_curr_res = abs(residual / self.curr_step)
        return rel_curr_res


    def calculate_rel_tolerance_GS(self, ):
        plasma_psi_step = self.trial_plasma_psi - self.eq1.plasma_psi
        self.d_plasma_psi_step = np.amax(plasma_psi_step) - np.amin(plasma_psi_step)

        a_res_GS = np.amax(abs(self.NK.F_function(self.trial_plasma_psi.reshape(-1),
                                                        self.tokamak_psi.reshape(-1),
                                                        self.profiles2)))
        
        r_res_GS = a_res_GS/self.d_plasma_psi_step
        return r_res_GS



   
    # def linear_step(self, active_voltage_vec,
    #                       working_relative_tol_GS):
        
    #     self.set_linear_solution(active_voltage_vec)
        
    #     self.step_complete_assign_simple(working_relative_tol_GS)
        
    #     flag = self.plasma_grids.check_if_outside_domain(jtor=self.profiles2.jtor)

    #     return flag
        







    def nlstepper(self, active_voltage_vec, 
                                target_relative_tol_currents=.01,
                                        target_relative_tol_GS=.01,
                                        working_relative_tol_GS=.002,
                                        target_relative_unexplained_residual=.5,
                                        max_n_directions=3,
                                        max_Arnoldi_iterations=4,
                                        max_collinearity=.3,
                                        step_size_psi=2.,
                                        step_size_curr=.8,
                                        scaling_with_n=0,
                                        relative_tol_for_nk_psi=.002,
                                        blend_GS=.5,
                                        blend_psi=1,
                                        curr_eps=1e-5,
                                        max_no_NK_psi=1.,
                                        clip=5,
                                        threshold=1.5,
                                        clip_hard=1.5,
                                        verbose=True,
                                        linear_only=False)       :

        self.set_linear_solution(active_voltage_vec)
        # finds linearised solution for the currents and solves GS in eq2 and profiles2

        if linear_only:
            self.step_complete_assign(working_relative_tol_GS, from_linear=True)
        else:
            # this assigns to self.eq2 and self.profiles2 
            # also records self.tokamak_psi corresponding to self.trial_currents in 2d
            res_curr = self.F_function_curr(self.trial_currents, active_voltage_vec)
            # uses self.trial_currents and self.currents_vec_m1 to relate res_curr above to step in the currents
            rel_curr_res = self.calculate_rel_tolerance_currents(res_curr, curr_eps=curr_eps)
            control = np.any(rel_curr_res > target_relative_tol_currents)
            # pair self.trial_currents and self.trial_plasma_psi are a GS solution
            control_GS = 0
            
            if verbose:
                print('starting: curr residual', np.amax(rel_curr_res), np.mean(rel_curr_res))
            log = []

            n_no_NK_psi = 0
            n_it = 0

            args_nk = [active_voltage_vec, self.rtol_NK]

            while control:
                if verbose:
                    for _ in log:
                        print(_)
                    
                log = []

                if control_GS:
                    self.NK.solve(self.eq2, self.profiles2, self.rtol_NK)
                    self.trial_plasma_psi *= (1 - blend_GS)
                    self.trial_plasma_psi += blend_GS * self.eq2.plasma_psi

                # self.assign_currents(self.trial_currents, self.eq2, self.profiles2)
                # self.tokamak_psi = self.eq2.tokamak.calcPsiFromGreens(pgreen=self.eq2._pgreen)
                
                self.trial_plasma_psi = self.trial_plasma_psi.reshape(-1)
                self.tokamak_psi = self.tokamak_psi.reshape(-1)


                res_psi = self.F_function_psi(trial_plasma_psi=self.trial_plasma_psi,
                                                active_voltage_vec=active_voltage_vec, 
                                                rtol_NK=self.rtol_NK)
                del_res_psi = (np.amax(res_psi) - np.amin(res_psi))

                log.append([n_it, 'psi cycle skipped', n_no_NK_psi, 'times, psi_residual', del_res_psi])

                if (del_res_psi > self.rtol_NK/relative_tol_for_nk_psi)+(n_no_NK_psi > max_no_NK_psi):
                    n_no_NK_psi = 0
                    self.psi_nk_solver.Arnoldi_iteration(x0=self.trial_plasma_psi, #trial_current expansion point
                                                        dx=res_psi, #first vector for current basis
                                                        R0=res_psi, #circuit eq. residual at trial_current expansion point: F_function(trial_current)
                                                        F_function=self.F_function_psi,
                                                        args=args_nk,
                                                        step_size=step_size_psi,
                                                        scaling_with_n=scaling_with_n,
                                                        target_relative_unexplained_residual=target_relative_unexplained_residual,   #add basis vector 
                                                        max_n_directions=max_n_directions, # max number of basis vectors (must be less than number of modes + 1)
                                                        max_Arnoldi_iterations=max_Arnoldi_iterations,
                                                        max_collinearity=max_collinearity,
                                                        clip=clip,
                                                        threshold=threshold,
                                                        clip_hard=clip_hard)
                    log.append([n_it, 'psi_coeffs = ', self.psi_nk_solver.coeffs])
                    self.trial_plasma_psi += self.psi_nk_solver.dx*blend_psi

                else:
                    # print('n_no_NK_psi = ', n_no_NK_psi)
                    n_no_NK_psi += 1


                self.trial_plasma_psi = self.trial_plasma_psi.reshape(self.nx, self.ny)
                # assumes the just updated self.trial_plasma_psi
                res_curr = self.F_function_curr(self.trial_currents, active_voltage_vec)
                rel_curr_res = abs(res_curr / self.curr_step)
                log.append([n_it, 'intermediate curr residual', np.amax(rel_curr_res), np.mean(rel_curr_res)])

                
                
                self.currents_nk_solver.Arnoldi_iteration( x0=self.trial_currents, #trial_current expansion point
                                                            dx=res_curr, #first vector for current basis
                                                            R0=res_curr, #circuit eq. residual at trial_current expansion point: F_function(trial_current)
                                                            F_function=self.F_function_curr,
                                                            args=[active_voltage_vec],
                                                            step_size=step_size_curr,
                                                            scaling_with_n=scaling_with_n,
                                                            target_relative_unexplained_residual=target_relative_unexplained_residual,   #add basis vector 
                                                            max_n_directions=max_n_directions, # max number of basis vectors (must be less than number of modes + 1)
                                                            max_Arnoldi_iterations=max_Arnoldi_iterations,
                                                            max_collinearity=max_collinearity,
                                                            clip=clip,
                                                            threshold=threshold,
                                                            clip_hard=clip_hard)
                # print('curr_coeffs = ', self.currents_nk_solver.coeffs)
                self.trial_currents += self.currents_nk_solver.dx#*blend_curr

                res_curr = self.F_function_curr(self.trial_currents, active_voltage_vec)
                rel_curr_res = self.calculate_rel_tolerance_currents(res_curr, curr_eps=curr_eps)
                control = np.any(rel_curr_res > target_relative_tol_currents)

                # self.assign_currents(self.trial_currents, self.eq2, self.profiles2)
                # self.tokamak_psi = self.eq2.tokamak.calcPsiFromGreens(pgreen=self.eq2._pgreen)
                
                r_res_GS = self.calculate_rel_tolerance_GS()
                control_GS = (r_res_GS > target_relative_tol_GS)
                control += control_GS
                
                log.append([n_it, 'curr_coeffs = ', self.currents_nk_solver.coeffs])
                log.append([n_it, 'full cycle curr residual', np.amax(rel_curr_res), np.mean(rel_curr_res)])
                log.append([n_it, 'GS residual: ',r_res_GS])

                
                n_it += 1
                

                # print('cycle: ', np.amax(rel_curr_res), np.mean(rel_curr_res))
                # print('GS residual: ',r_res_GS)

            # update rtol_NK based on step just taken
            

            

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

            # print('im about to assign', np.shape(self.tokamak_psi), np.shape(self.trial_plasma_psi))
            self.step_complete_assign(#self.simplified_c, self.trial_plasma_psi, 
                                      working_relative_tol_GS)
        

        flag = self.plasma_grids.check_if_outside_domain(jtor=self.profiles2.jtor)

        return flag








    def nl_step_nk_curr_GS(self, active_voltage_vec, 
                                target_relative_tol_currents=.1,
                                use_extrapolation=False,
                                working_relative_tol_GS=.01,
                                target_relative_unexplained_residual=.6,
                                max_n_directions=4,
                                max_Arnoldi_iterations=5,
                                max_collinearity=.3,
                                step_size_curr=1,
                                scaling_with_n=0,
                                curr_eps=1e-4,
                                clip=3,
                                threshold=1.5,
                                clip_hard=1.5,
                                verbose=False,
                                ):
        
        # note_tokamak_psi = 1.0*self.NK.tokamak_psi
        
        # self.central_2  = (1 + (self.step_no>0))
        if use_extrapolation*(self.step_no > self.extrapolator_input_size):
            self.trial_currents = 1.0*self.currents_guess 
            
        else:
            self.trial_currents = self.hatIy1_iterative_cycle(self.hatIy,
                                                              active_voltage_vec,
                                                              rtol_NK=self.rtol_NK)

        res_curr = self.F_function_curr_GS(self.trial_currents, active_voltage_vec, self.rtol_NK)
        rel_curr_res = self.calculate_rel_tolerance_currents(res_curr, curr_eps=curr_eps)
        control = np.any(rel_curr_res > target_relative_tol_currents)

        args_nk = [active_voltage_vec, self.rtol_NK]

        if verbose:
            print('starting: curr residual', np.amax(rel_curr_res))
        log = []

        n_it = 0

        while control:

            if verbose:
                for _ in log:
                    print(_)
                
            log = []

            self.currents_nk_solver.Arnoldi_iteration( x0=self.trial_currents, #trial_current expansion point
                                                        dx=res_curr, #first vector for current basis
                                                        R0=res_curr, #circuit eq. residual at trial_current expansion point: F_function(trial_current)
                                                        F_function=self.F_function_curr_GS,
                                                        args=args_nk,
                                                        step_size=step_size_curr,
                                                        scaling_with_n=scaling_with_n,
                                                        target_relative_unexplained_residual=target_relative_unexplained_residual,   #add basis vector 
                                                        max_n_directions=max_n_directions, # max number of basis vectors (must be less than number of modes + 1)
                                                        max_Arnoldi_iterations=max_Arnoldi_iterations,
                                                        max_collinearity=max_collinearity,
                                                        clip=clip,
                                                        threshold=threshold,
                                                        clip_hard=clip_hard)

            self.trial_currents += self.currents_nk_solver.dx#*blend_curr

            res_curr = self.F_function_curr_GS(self.trial_currents, active_voltage_vec, self.rtol_NK)
            rel_curr_res = self.calculate_rel_tolerance_currents(res_curr, curr_eps=curr_eps)
            control = np.any(rel_curr_res > target_relative_tol_currents)
            
            log.append([n_it, 'full cycle curr residual', np.amax(rel_curr_res), np.mean(rel_curr_res)])

            n_it += 1
            # print('cycle:', np.amax(rel_res0), np.mean(rel_res0))
            
            # r_dpsi = abs(self.eq2.plasma_psi - note_psi)
            # r_dpsi /= (np.amax(note_psi) - np.amin(note_psi))
            # control += np.any(r_dpsi > rtol_psi)


        self.time += self.dt_step

        # plt.figure()
        # plt.imshow(self.profiles2.jtor - self.jtor_m1)
        # plt.colorbar()
        # plt.show()

        # self.dpsi = self.eq2.plasma_psi - self.eq1.plasma_psi
        # plt.figure()
        # plt.imshow(self.dpsi)
        # plt.colorbar()
        # plt.show()

        
        # plt.figure()
        # plt.imshow(self.NK.tokamak_psi - note_tokamak_psi)
        # plt.colorbar()
        # plt.show()


        self.step_complete_assign(self.simplified_c, self.eq2.plasma_psi, working_relative_tol_GS)
        
        
    
        flag = self.plasma_grids.check_if_outside_domain(jtor=self.profiles2.jtor)


        return flag





   



    # def find_best_convex_combination(self, previous_residual, 
    #                                         trial_currents, 
    #                                         active_voltage_vec, 
    #                                         pts=[.05,.95],
    #                                         blend=1.):
    #     note_plasma_psi = 1.0*self.trial_plasma_psi
    #     res_list = []
    #     for alpha in pts:
    #         self.trial_plasma_psi = (1-alpha)*note_plasma_psi + alpha*self.eq2.plasma_psi
    #         res_list.append(np.sum(np.abs(self.F_function_curr(trial_currents, active_voltage_vec))))
    #     a = (res_list[1] - res_list[0])/(pts[1] - pts[0])
    #     if a>0:
    #         b = res_list[0] - a*pts[0]
    #         best_alpha = max(.0, min(1, (blend*previous_residual - b)/a))
    #         print(best_alpha)
    #     else:
    #         best_alpha = 1
    #         print(best_alpha, 'this was negative!')
        
    #     self.trial_plasma_psi = (1-best_alpha)*note_plasma_psi + best_alpha*self.eq2.plasma_psi
        

    # def nl_step_nk_curr(self, active_voltage_vec, 
    #                             rtol_NK=1e-9,
    #                             atol_currents=1e-3,
    #                             rtol_psi=1e-3,
    #                             verbose=False,
    #                             max_n_directions=6,
    #                             target_relative_unexplained_residual=.3,
    #                             max_collinearity=.3,
    #                             step_size=1,
    #                             clip=3):
        
    #     # self.central_2  = (1 + (self.step_no>0))

    #     self.trial_currents, res = self.hatIy1_iterative_cycle(self.hatIy,
    #                                                         active_voltage_vec,
    #                                                         rtol_NK=rtol_NK)
    #     self.trial_plasma_psi = 1.0*self.eq2.plasma_psi

    #     res0 = self.F_function_curr(self.trial_currents, active_voltage_vec)
    #     abs_res0 = np.abs(res0)
    #     nres0 = np.sum(abs_res0)

    #     control = np.any(abs_res0 > atol_currents)
    #     r_dpsi = abs(self.eq2.plasma_psi - self.trial_plasma_psi)
    #     r_dpsi /= (np.amax(self.trial_plasma_psi) - np.amin(self.trial_plasma_psi))
    #     control += np.any(r_dpsi > rtol_psi)

    #     print('starting:', np.amax(abs_res0), np.amax(r_dpsi))

    #     # self.trial_plasma_psi = 1.0*self.eq2.plasma_psi

    #     while control:
    #         self.Arnoldi_iteration(trial_sol=self.trial_currents, #trial_current expansion point
    #                                 vec_direction=res0, #first vector for current basis
    #                                 R0=res0, #circuit eq. residual at trial_current expansion point: F_function(trial_current)
    #                                 F_function=self.F_function_curr,
    #                                 active_voltage_vec=active_voltage_vec,
    #                                 max_n_directions=max_n_directions, # max number of basis vectors (must be less than number of modes + 1)
    #                                 target_relative_unexplained_residual=target_relative_unexplained_residual,   #add basis vector 
    #                                                 #if unexplained orthogonal component is larger than
    #                                 max_collinearity=max_collinearity,
    #                                 step_size=step_size, #infinitesimal step size, when compared to norm(trial)
    #                                 clip=clip,
    #                                 rtol_NK=rtol_NK)
    #         self.trial_currents += self.d_sol_step

    #         self.assign_currents_solve_GS(self.trial_currents, rtol_NK)

    #         r_dpsi = abs(self.eq2.plasma_psi - self.trial_plasma_psi)
    #         r_dpsi /= (np.amax(self.trial_plasma_psi)-np.amin(self.trial_plasma_psi))
    #         control = np.any(r_dpsi > rtol_psi)

    #         self.find_best_convex_combination(nres0,
    #                                         self.trial_currents,
    #                                         active_voltage_vec)
    #         res0 = self.F_function_curr(self.trial_currents, active_voltage_vec)                                
    #         abs_res0 = np.abs(res0)
    #         nres0 = np.sum(abs_res0)
    #         control += np.any(abs_res0 > atol_currents)

    #         print('cycle:', np.amax(abs_res0), np.amax(r_dpsi))

    #     self.time += self.dt_step

    #     plt.figure()
    #     plt.imshow(self.profiles2.jtor - self.jtor_m1)
    #     plt.colorbar()
    #     plt.show()

    #     self.dpsi = self.eq2.plasma_psi - self.eq1.plasma_psi
    #     plt.figure()
    #     plt.imshow(self.dpsi)
    #     plt.colorbar()
    #     plt.show()

    #     self.step_complete_assign(self.trial_currents)
        
    #     flag = self.plasma_grids.check_if_outside_domain(jtor=self.profiles2.jtor)


    #     return flag


















    
    # def F_function_nk_J1(self, J1, active_voltage_vec, rtol_NK=1e-8):
    #     J1 /= np.sum(J1)
    #     self.simplified_c, self.circ_eq_res = self.hatIy1_iterative_cycle(J1, active_voltage_vec, rtol_NK)
    #     self.hatIy1_new = self.plasma_grids.hat_Iy_from_jtor(self.profiles2.jtor)
    #     return self.hatIy1_new - J1
    # def F_function_nk_J1_currents(self, J1, active_voltage_vec, rtol_NK=1e-8):
    #     dJ1 = self.F_function_nk_J1(J1, active_voltage_vec, rtol_NK)
    #     note_currents = 1.0*self.simplified_c
    #     iterated_c = self.currents_from_hatIy(self.hatIy1_new,
    #                                         active_voltage_vec)
                                            
    #     d_currents = iterated_c - note_currents
    #     return dJ1, d_currents


    # def nl_step_nk_J1(self, active_voltage_vec, 
    #                             J1,
    #                             # alpha=.8, 
    #                             rtol_NK=1e-9,
    #                             target_relative_tol_currents=.5,
    #                             atol_J=1e-3,
    #                             use_extrapolation=True,
    #                             verbose=False,
    #                             max_n_directions=6,
    #                             target_relative_unexplained_residual=.3,
    #                             max_collinearity=.3,
    #                             step_size=2,
    #                             clip=3):
        
    #     self.hatIy1 = 1.0*J1
    #     # self.central_2  = (1 + (self.step_no>0))


    #     resJ, d_currents = self.F_function_nk_J1_currents(J1=self.hatIy1,
    #                                                     active_voltage_vec=active_voltage_vec,
    #                                                     rtol_NK=rtol_NK)
        
    #     rel_res_currents = abs(d_currents/(self.simplified_c - self.currents_vec_m1))
    #     control = np.any(rel_res_currents > target_relative_tol_currents)
    #     control += np.any(resJ > atol_J)

    #     print('starting: ', max(rel_res_currents), max(abs(resJ)))

    #     iterative_steps = 0
    #     simplified_c = 1.0*self.simplified_c

    #     while control:
    #         self.Arnoldi_iteration(trial_sol=self.hatIy1, #trial_current expansion point
    #                                 vec_direction=resJ, #first vector for current basis
    #                                 R0=resJ, #circuit eq. residual at trial_current expansion point: F_function(trial_current)
    #                                 F_function=self.F_function_nk_J1,
    #                                 active_voltage_vec=active_voltage_vec,
    #                                 max_n_directions=max_n_directions, # max number of basis vectors (must be less than number of modes + 1)
    #                                 target_relative_unexplained_residual=target_relative_unexplained_residual,   #add basis vector 
    #                                                 #if unexplained orthogonal component is larger than
    #                                 max_collinearity=max_collinearity,
    #                                 step_size=step_size, #infinitesimal step
    #                                 clip=clip,
    #                                 rtol_NK=rtol_NK)
    #         print(self.coeffs)
            
    #         # self.hatIy1 += self.d_sol_step
    #         # self.hatIy1 /= np.sum(self.hatIy1)

    #         J1new = self.hatIy1 + self.d_sol_step
    #         J1new /= np.sum(J1new)

    #         resJ, d_currents = self.F_function_nk_J1_currents(J1=J1new,
    #                                                         active_voltage_vec=active_voltage_vec,
    #                                                         rtol_NK=rtol_NK)

    #         self.hatIy1 = 1.0*J1new

           
    #         rel_res_currents = abs(d_currents/(self.simplified_c - self.currents_vec_m1))
    #         control = np.any(rel_res_currents > target_relative_tol_currents)
    #         control += np.any(abs(resJ) > atol_J)
    #         simplified_c = 1.0*self.simplified_c

    #         print('cycle: ', max(rel_res_currents), max(abs(resJ)))

    #         if verbose:
    #             print('max currents change = ', np.max(rel_res_currents))
    #             print('max J direction change = ', np.max(np.abs(resJ)), np.linalg.norm(resJ))
    #             # print('max circuit eq residual (dim of currents) = ', np.argmax(abs(self.circ_eq_res)), self.circ_eq_res)
    #             # print(self.simplified_c - self.currents_vec_m1)

    #         iterative_steps += 1

        
    #     self.time += self.dt_step

    #     plt.figure()
    #     plt.imshow(self.profiles2.jtor - self.jtor_m1)
    #     plt.colorbar()
    #     plt.show()

    #     self.dpsi = self.eq2.plasma_psi - self.eq1.plasma_psi
    #     plt.figure()
    #     plt.imshow(self.dpsi)
    #     plt.colorbar()
    #     plt.show()


    #     self.step_complete_assign(self.simplified_c)
    #     if use_extrapolation:
    #         self.guess_J_from_extrapolation(alpha=1, rtol_NK=rtol_NK)
        

    
    #     flag = self.plasma_grids.check_if_outside_domain(jtor=self.profiles2.jtor)

        

    #     return flag


   
    
    


    # def nl_step_nk_psi(self, active_voltage_vec, 
    #                             trial_currents,
    #                             rtol_NK=1e-9,
    #                             target_relative_tol_currents=.1,
    #                             atol_J=1e-3,
    #                             verbose=False,
    #                             max_n_directions=6,
    #                             target_relative_unexplained_residual=.3,
    #                             max_collinearity=.3,
    #                             step_size=1,
    #                             clip=3,
    #                             use_extrapolation=False):
        
    #     self.currents_nk_psi = np.zeros((self.n_metal_modes+1, 0))
        
    #     # self.central_2  = (1 + (self.step_no>0))

    #     if use_extrapolation*(self.step_no > self.extrapolator_input_size):
    #         self.trial_currents = 1.0*self.currents_guess 
            
    #     else:
    #         self.trial_currents = self.hatIy1_iterative_cycle(self.hatIy,
    #                                                           active_voltage_vec,
    #                                                           rtol_NK=rtol_NK)

    #     self.ref_currents = 1.0*trial_currents
    #     self.tokamak_psi = 1.0*self.NK.tokamak_psi
    #     psi0 = 1.0*self.eq2.plasma_psi.reshape(-1)

        
    #     res_psi = self.F_function_psi(trial_plasma_psi=psi0,
    #                                     active_voltage_vec=active_voltage_vec, 
    #                                     rtol_NK=1e-9)
    #     self.currents_nk_psi = np.zeros((self.n_metal_modes+1, 0))

    #     abs_increments = abs(self.simplified_c - trial_currents)
    #     control = np.any(abs_increments > atol_currents)
    #     control += np.any(res_psi > atol_J)
    #     print('starting: ', max(abs(res_psi)), max(abs_increments))
    #     simplified_c = 1.0*self.simplified_c
         
    #     while control:
    #         self.Arnoldi_iteration(trial_sol=psi0, #trial_current expansion point
    #                                 vec_direction=res_psi, #first vector for current basis
    #                                 R0=res_psi, #circuit eq. residual at trial_current expansion point: F_function(trial_current)
    #                                 F_function=self.F_function_psi,
    #                                 active_voltage_vec=active_voltage_vec,
    #                                 max_n_directions=max_n_directions, # max number of basis vectors (must be less than number of modes + 1)
    #                                 target_relative_unexplained_residual=target_relative_unexplained_residual,   #add basis vector 
    #                                                 #if unexplained orthogonal component is larger than
    #                                 max_collinearity=max_collinearity,
    #                                 step_size=step_size, #infinitesimal step size, when compared to norm(trial)
    #                                 clip=clip,
    #                                 rtol_NK=rtol_NK)
            
    #         res_psi = self.F_function_psi(trial_plasma_psi=psi0+self.d_sol_step,
    #                                         active_voltage_vec=active_voltage_vec, 
    #                                         rtol_NK=rtol_NK)
    #         self.F_function_dJ(trial_currents=self.simplified_c,
    #                             active_voltage_vec=active_voltage_vec,
    #                             rtol_NK=rtol_NK)
            
    #         abs_increments = abs(self.simplified_c - simplified_c)
    #         control = np.any(abs_increments > atol_currents)
    #         control += np.any(res_psi > atol_J)
    #         print('cycle: ', max(abs(res_psi)), max(abs_increments))

    #         psi0 = 1.0*self.eq2.plasma_psi.reshape(-1)
    #         self.tokamak_psi = 1.0*self.NK.tokamak_psi
    #         self.ref_currents = 1.0*self.simplified_c
    #         simplified_c = 1.0*self.simplified_c

    #     self.time += self.dt_step

    #     plt.figure()
    #     plt.imshow(self.profiles2.jtor - self.jtor_m1)
    #     plt.colorbar()
    #     plt.show()

    #     self.dpsi = self.eq2.plasma_psi - self.eq1.plasma_psi
    #     plt.figure()
    #     plt.imshow(self.dpsi)
    #     plt.colorbar()
    #     plt.show()


    #     self.step_complete_assign(self.simplified_c)
    #     # if use_extrapolation:
    #     #     self.guess_J_from_extrapolation(alpha=alpha, rtol_NK=rtol_NK)
        
    #     flag = self.plasma_grids.check_if_outside_domain(jtor=self.profiles2.jtor)

        
    #     return flag



   
        




    # def F_function_J1(self, trial_currents, active_voltage_vec, rtol_NK=1e-8):
    #     # trial_currents is the full array of intermediate results from euler solver
    #     # root problem for circuit equation
    #     # collects both metal normal modes and norm_plasma
        
    #     # current at t+dt
    #     # current_tpdt = 1.0*trial_currents#[:, -1]
    #     self.assign_currents(trial_currents, profile=self.profiles2, eq=self.eq2)
    #     self.NK.solve(self.eq2, self.profiles2, target_relative_tolerance=rtol_NK)
    #     self.trial_Iy1 = self.plasma_grids.Iy_from_jtor(self.profiles2.jtor)

    #     self.Iy_dot = (self.trial_Iy1 - self.Iy_m1)/(2*self.dt_step)
    #     self.Id_dot = ((trial_currents - self.currents_vec_m1)/(2*self.dt_step))[:-1]

    #     self.forcing_term = self.evol_metal_curr.forcing_term_eig_plasma(active_voltage_vec=active_voltage_vec, 
    #                                                                      Iydot=self.Iy_dot)

    #     # mean_curr = np.mean(trial_currents, axis=-1)                                                                 
    #     self.circuit_eq_residual[:-1] = 1.0*self.evol_metal_curr.current_residual( Itpdt=trial_currents[:-1], 
    #                                                                     Iddot=self.Id_dot, 
    #                                                                     forcing_term=self.forcing_term)


    #     # mean_Iy = trial_currents[-1]*self.hatIy1*self.plasma_norm_factor
    #     # mean_Iy = 1.0*self.trial_Iy1
    #     self.circuit_eq_residual[-1] = 1.0*self.evol_plasma_curr.current_residual( red_Iy0=self.Iy, 
    #                                                                     red_Iy1=self.trial_Iy1,
    #                                                                     red_Iydot=self.Iy_dot,
    #                                                                     Iddot=self.Id_dot)/self.plasma_norm_factor
    #     # return self.circuit_eq_residual



    
    # def iterative_unit_dJ(self, dJ,
    #                             active_voltage_vec,
    #                             Rp, 
    #                             rtol_NK):
    #     simplified_c1 = self.central_2*self.simplified_solver_dJ.stepper(It=self.currents_vec_m1,
    #                                                         norm_red_Iy0=self.hatIy, 
    #                                                         norm_red_Iy_dot=dJ, 
    #                                                         active_voltage_vec=active_voltage_vec, 
    #                                                         Rp=Rp,
    #                                                         central_2=self.central_2)
        
    #     # calculate t+dt currents
    #     # plasma
    #     Iy_tpdt = self.Iy_m1/self.plasma_norm_factor + simplified_c1[-1]*dJ
    #     simplified_c1[-1] = np.sum(Iy_tpdt)
    #     # metal
    #     simplified_c1[:-1] += self.currents_vec_m1[:-1]
        
    #     self.F_function_dJ(trial_currents=simplified_c1, 
    #                             active_voltage_vec=active_voltage_vec, 
    #                             rtol_NK=rtol_NK)   
    #     return simplified_c1, self.circuit_eq_residual
    


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
        
    #     Rp = self.calc_lumped_plasma_resistance(self.hatIy, self.hatIy_m1)
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

    #         self.dJ1 = self.plasma_grids.hat_Iy_from_jtor(self.profiles2.jtor - self.jtor_m1)
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

    #     flag = self.plasma_grids.check_if_outside_domain(jtor=self.profiles2.jtor, 
    #                                   boole_mask_outside_limiter=self.mask_outside_limiter)

    #     return flag



    # def nl_mix_unit(self, active_voltage_vec,
    #                              Rp, 
    #                              rtol_NK,
    #                              max_n_directions=10, # max number of basis vectors (must be less than number of modes + 1)
    #                              target_relative_unexplained_residual=.1,   #add basis vector 
    #                                             #if unexplained orthogonal component is larger than
    #                              max_collinearity=.3,
    #                              step_size=.005, #infinitesimal step
    #                              clip=3):

    #     simplified_c, res = self.iterative_unit(active_voltage_vec=active_voltage_vec,
    #                                              Rp=Rp, 
    #                                              rtol_NK=rtol_NK)
        
    #     self.Arnoldi_iteration(trial_sol=simplified_c, #trial_current expansion point
    #                             vec_direction=-res, #first vector for current basis
    #                             R0=res, #circuit eq. residual at trial_current expansion point: F_function(trial_current)
    #                             F_function=self.F_function,
    #                             active_voltage_vec=active_voltage_vec,
    #                             max_n_directions=max_n_directions, # max number of basis vectors (must be less than number of modes + 1)
    #                             target_relative_unexplained_residual=target_relative_unexplained_residual,   #add basis vector 
    #                                             #if unexplained orthogonal component is larger than
    #                             max_collinearity=max_collinearity,
    #                             step_size=step_size, #infinitesimal step
    #                             clip=clip)

    #     simplified_c1 = simplified_c + self.d_sol_step
    #     res1 = self.F_function(trial_currents=simplified_c1, 
    #                          active_voltage_vec=active_voltage_vec, 
    #                          rtol_NK=rtol_NK)   
    #     return simplified_c1, res1




    # def nl_step_mix(self, active_voltage_vec, 
    #                              alpha=.8, 
    #                              rtol_NK=5e-4,
    #                              atol_increments=1e-3,
    #                              rtol_residuals=1e-3,
    #                              max_n_directions=10, # max number of basis vectors (must be less than number of modes + 1)
    #                              target_relative_unexplained_residual=.1,   #add basis vector 
    #                                             #if unexplained orthogonal component is larger than
    #                              max_collinearity=.3,
    #                              step_size=.005, #infinitesimal step
    #                              clip=3,
    #                              return_n_steps=False,
    #                              verbose=False,
    #                              threshold=.001):
        
    #     Rp = self.calc_lumped_plasma_resistance(self.hatIy, self.hatIy)
        
    #     simplified_c, res = self.nl_mix_unit(active_voltage_vec=active_voltage_vec,
    #                                                 Rp=Rp, 
    #                                                 rtol_NK=rtol_NK,
    #                                                 max_n_directions=max_n_directions, # max number of basis vectors (must be less than number of modes + 1)
    #                                                 target_relative_unexplained_residual=target_relative_unexplained_residual,   #add basis vector 
    #                                                             #if unexplained orthogonal component is larger than
    #                                                 max_collinearity=max_collinearity,
    #                                                 step_size=step_size, #infinitesimal step
    #                                                 clip=clip)

    #     dcurrents = np.abs(simplified_c-self.currents_vec)
    #     vals_for_check = np.where(dcurrents>threshold, dcurrents, threshold)                                        
        
    #     iterative_steps = 0
    #     control = 1
    #     while control:
    #         self.dJ = (1-alpha)*self.dJ + alpha*(self.plasma_grids.hat_Iy_from_jtor(self.profiles2.jtor - self.profiles1.jtor))
    #         simplified_c1, res = self.nl_mix_unit(active_voltage_vec=active_voltage_vec,
    #                                                 Rp=Rp, 
    #                                                 rtol_NK=rtol_NK,
    #                                                 max_n_directions=max_n_directions, # max number of basis vectors (must be less than number of modes + 1)
    #                                                 target_relative_unexplained_residual=target_relative_unexplained_residual,   #add basis vector 
    #                                                             #if unexplained orthogonal component is larger than
    #                                                 max_collinearity=max_collinearity,
    #                                                 step_size=step_size, #infinitesimal step
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
    #                     max_n_directions=10, # max number of basis vectors (must be less than number of modes + 1)
    #                     target_relative_unexplained_residual=.2,   #add basis vector 
    #                                     #if unexplained orthogonal component is larger than
    #                     max_collinearity=.3,
    #                     step_size=.5, #infinitesimal step
    #                     clip=3,
    #                     rtol_NK=1e-5,
    #                     atol_currents=1e-3,
    #                     atol_J=1e-3,
    #                     verbose=False):
        
    #     self.Rp = self.calc_lumped_plasma_resistance(self.hatIy, self.hatIy)

    #     resJ = self.F_function_nk_dJ(trial_sol, active_voltage_vec=active_voltage_vec, rtol_NK=rtol_NK)
        

    #     simplified_c = 1.0*self.simplified_c1
    #     # dcurrents = np.abs(self.simplified_c1-self.currents_vec)
    #     # vals_for_check = np.where(dcurrents>threshold, dcurrents, threshold)

    #     iterative_steps = 0
    #     control = 1
    #     while control:
    #         self.Arnoldi_iteration(trial_sol=trial_sol, #trial_current expansion point
    #                                 vec_direction=-resJ, #first vector for current basis
    #                                 R0=resJ, #circuit eq. residual at trial_current expansion point: F_function(trial_current)
    #                                 F_function=self.F_function_nk_dJ,
    #                                 active_voltage_vec=active_voltage_vec,
    #                                 max_n_directions=max_n_directions, # max number of basis vectors (must be less than number of modes + 1)
    #                                 target_relative_unexplained_residual=target_relative_unexplained_residual,   #add basis vector 
    #                                                 #if unexplained orthogonal component is larger than
    #                                 max_collinearity=max_collinearity,
    #                                 step_size=step_size, #infinitesimal step
    #                                 clip=clip,
    #                                 rtol_NK=rtol_NK)
    #         print(self.coeffs)
    #         trial_sol += self.d_sol_step
    #         resJ = self.F_function_nk_dJ(trial_sol, 
    #                          active_voltage_vec=active_voltage_vec, 
    #                          rtol_NK=rtol_NK)   
            
            
    #         abs_increments = np.abs(simplified_c-self.simplified_c1)
    #         # dcurrents = np.abs(simplified_c1-self.currents_vec)
    #         # vals_for_check = np.where(dcurrents>threshold, dcurrents, threshold)
    #         rel_residuals = np.abs(self.circuit_eq_residual)#/vals_for_check
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
        
    #     flag = self.plasma_grids.check_if_outside_domain(jtor=self.profiles2.jtor, 
    #                                   boole_mask_outside_limiter=self.mask_outside_limiter)

    #     return flag
    

    # def LSQP(self, F_function, nF_function, G, Q, clip, threshold=.99, clip_hard=1.):
    #     #solve the least sq problem in coeffs: min||G*coeffs+F_function||^2
    #     self.coeffs = np.matmul(np.matmul(np.linalg.inv(np.matmul(G.T, G)),
    #                                  G.T), -F_function)                            
    #     self.coeffs = np.clip(self.coeffs, -clip, clip)
    #     self.explained_res = np.sum(G*self.coeffs[np.newaxis,:], axis=1) 
    #     self.rel_unexpl_res = np.linalg.norm(self.explained_res + F_function)/nF_function
    #     if self.rel_unexpl_res > threshold:
    #         self.coeffs = np.clip(self.coeffs, -clip_hard, clip_hard)
    #     self.d_sol_step = np.sum(Q*self.coeffs[np.newaxis,:], axis=1)


    # def Arnoldi_unit(self,  trial_sol, #trial_current expansion point
    #                         vec_direction, #first vector for current basis
    #                         F_function, #circuit eq. residual at trial_current expansion point: F_function(trial_current)
    #                         F_function,
    #                         active_voltage_vec,
    #                         grad_coeff,
    #                         rtol_NK
    #                         ):

    #     candidate_d_sol = grad_coeff*vec_direction/np.linalg.norm(vec_direction)
    #     print('norm candidate step', np.linalg.norm(candidate_d_sol))
    #     candidate_sol = trial_sol + candidate_d_sol
    #     # candidate_sol /= np.sum(candidate_sol)
    #     ri = F_function(candidate_sol, active_voltage_vec=active_voltage_vec, rtol_NK=rtol_NK)
    #     lvec_direction = ri - F_function

    #     self.Q[:,self.n_it] = 1.0*candidate_d_sol
    #     # self.Q[:,self.n_it] = candidate_sol - trial_sol
    #     self.Qn[:,self.n_it] = self.Q[:,self.n_it]/np.linalg.norm(self.Q[:,self.n_it])
        
    #     self.G[:,self.n_it] = 1.0*lvec_direction
    #     self.Gn[:,self.n_it] = self.G[:,self.n_it]/np.linalg.norm(self.G[:,self.n_it])

    #     #orthogonalize residual 
    #     lvec_direction -= np.sum(np.sum(self.Qn[:,:self.n_it+1]*lvec_direction[:,np.newaxis], axis=0, keepdims=True)*self.Qn[:,:self.n_it+1], axis=1)
    #     return lvec_direction


    # def Arnoldi_iteration(self, trial_sol, #trial_current expansion point
    #                             vec_direction, #first vector for current basis
    #                             F_function, #circuit eq. residual at trial_current expansion point: F_function(trial_current)
    #                             F_function, 
    #                             active_voltage_vec,
    #                             max_n_directions=5, # max number of basis vectors (must be less than number of modes + 1)
    #                             target_relative_unexplained_residual=.1,   #add basis vector 
    #                                             #if unexplained orthogonal component is larger than
    #                             max_collinearity=.3,
    #                             step_size=.3, #infinitesimal step size, when compared to norm(trial)
    #                             clip=3,
    #                             rtol_NK=1e-9):
        
    #     nF_function = np.linalg.norm(F_function)
    #     problem_d = len(trial_sol)

    #     #basis in Psi space
    #     self.Q = np.zeros((problem_d, max_n_directions+1))
    #     #orthonormal basis in Psi space
    #     self.Qn = np.zeros((problem_d, max_n_directions+1))
    #     #basis in grandient space
    #     self.G = np.zeros((problem_d, max_n_directions+1))
    #     #basis in grandient space
    #     self.Gn = np.zeros((problem_d, max_n_directions+1))
        
    #     self.n_it = 0
    #     self.n_it_tot = 0
    #     grad_coeff = step_size*nF_function

    #     print('norm trial_sol', np.linalg.norm(trial_sol))

    #     control = 1
    #     while control:
    #         # do step
    #         vec_direction = self.Arnoldi_unit(trial_sol, #trial_current expansion point
    #                                             vec_direction, #first vector for current basis
    #                                             F_function, #circuit eq. residual at trial_current expansion point: F_function(trial_current)
    #                                             F_function,
    #                                             active_voltage_vec,
    #                                             grad_coeff,#/(self.n_it+1)**1.2,
    #                                             rtol_NK=rtol_NK
    #                                             )
    #         collinear_control = 1 - np.any(np.sum(self.Gn[:,:self.n_it]*self.Gn[:,self.n_it:self.n_it+1], axis=0) > max_collinearity)
    #         self.n_it_tot += 1
    #         if collinear_control:
    #             self.n_it += 1
    #             self.LSQP(F_function, nF_function, G=self.G[:,:self.n_it], Q=self.Q[:,:self.n_it], clip=clip)
    #             # rel_unexpl_res = np.linalg.norm(self.explained_res + F_function)/nF_function
    #             print('rel_unexpl_res', self.rel_unexpl_res)
    #             arnoldi_control = (self.rel_unexpl_res > target_relative_unexplained_residual)
    #         else:
    #             print('collinear!')
    #             # self.currents_nk_psi = self.currents_nk_psi[:,:-1]

    #         control = arnoldi_control*(self.n_it_tot<max_n_directions)
    #         # if rel_unexpl_res > .6:
    #         #     clip = 1.5
    #         # self.LSQP(F_function, G=self.G[:,:self.n_it], Q=self.Q[:,:self.n_it], clip=clip)


    












    # def nl_step_iterative_J1(self, active_voltage_vec, 
    #                             J1,
    #                             alpha=.8, 
    #                             rtol_NK=5e-4,
    #                             atol_currents=1e-3,
    #                             atol_J=1e-3,
    #                             use_extrapolation=True,
    #                             verbose=False
    #                             ):
        
    #     self.hatIy1 = 1.0*J1
    #     self.central_2  = (1 + (self.step_no>0))

        
    #     simplified_c, res = self.hatIy1_iterative_cycle(J1=self.hatIy1,
    #                                                 active_voltage_vec=active_voltage_vec,
    #                                                 rtol_NK=rtol_NK)

    #     # dcurrents = np.abs(simplified_c-self.currents_vec)
    #     # vals_for_check = np.where(dcurrents>threshold, dcurrents, threshold)

    #     iterative_steps = 0
    #     control = 1
    #     while control:
    #         self.hatIy1n = self.plasma_grids.hat_Iy_from_jtor(self.profiles2.jtor)
    #         self.ddJ = self.hatIy1n - self.hatIy1
    #         self.hatIy1n = (1-alpha)*self.hatIy1 + alpha*self.hatIy1n
    #         self.hatIy1n /= np.sum(self.hatIy1n)
    #         self.hatIy1 = 1.0*self.hatIy1n
    #         simplified_c1, res = self.hatIy1_iterative_cycle(J1=self.hatIy1,
    #                                                     active_voltage_vec=active_voltage_vec,
    #                                                     rtol_NK=rtol_NK)   

    #         abs_increments = np.abs(simplified_c - simplified_c1)
    #         # dcurrents = np.abs(simplified_c1-self.currents_vec)
    #         # vals_for_check = np.where(dcurrents>threshold, dcurrents, threshold)
    #         # rel_residuals = np.abs(res)#/vals_for_check
    #         control = np.any(abs_increments>atol_currents)
    #         # control += np.any(rel_residuals>rtol_residuals)
    #         control += np.any( abs(self.ddJ) > atol_J )
    #         if verbose:
    #             print('max currents change = ', np.max(abs_increments))
    #             print('max J direction change = ', np.max(np.abs(self.ddJ)), np.linalg.norm(self.ddJ))
    #             # print('max circuit eq residual (dim of currents) = ', np.argmax(abs(res)), res)
    #             # print(simplified_c1 - self.currents_vec_m1)

    #         iterative_steps += 1

    #         simplified_c = 1.0*simplified_c1
        
    #     self.time += self.dt_step

    #     plt.figure()
    #     plt.imshow(self.profiles2.jtor - self.jtor_m1)
    #     plt.colorbar()
    #     plt.show()

    #     plt.figure()
    #     plt.imshow(self.eq2.plasma_psi - self.eq1.plasma_psi)
    #     plt.colorbar()
    #     plt.show()

    #     self.step_complete_assign(simplified_c)
    #     if use_extrapolation:
    #         self.guess_J_from_extrapolation(alpha=alpha, rtol_NK=rtol_NK)
        
    #     flag = self.plasma_grids.check_if_outside_domain(jtor=self.profiles2.jtor, 
    #                                   boole_mask_outside_limiter=self.mask_outside_limiter)


    #     return flag