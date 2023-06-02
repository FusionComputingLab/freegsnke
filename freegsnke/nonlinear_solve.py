import numpy as np

from . import MASTU_coils
from .MASTU_coils import coils_dict

from copy import deepcopy

# from . import quants_for_emu
from .circuit_eq_metal import metal_currents
from .circuit_eq_plasma import plasma_current
from .linear_solve import simplified_solver_dJ
from .linear_solve import simplified_solver_J1
from . import plasma_grids
from . import extrapolate

#from picardfast import fast_solve
from .newtonkrylov import NewtonKrylov
from .jtor_update import ConstrainPaxisIp


from .faster_shape import shapes
from .faster_shape import check_against_the_wall

import matplotlib.pyplot as plt


class nl_solver:
    # interfaces the circuit equation with freeGS NK solver 
    # executes dt evolution of fully non linear equations
    
    
    def __init__(self, profiles, eq, 
                 max_mode_frequency, 
                 max_internal_timestep=.0001,
                 full_timestep=.0001,
                 plasma_resistivity=1e-6,
                 extrapolator_input_size=4,
                 extrapolator_order=1,
                 plasma_norm_factor=2000):

        self.shapes = shapes 

        self.nx = np.shape(eq.R)[0]
        self.ny = np.shape(eq.R)[1]
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
        self.NK = NewtonKrylov(eq)

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

        for i,labeli in enumerate(coils_dict):
            self.eq1.tokamak[labeli].control = False
            self.eq2.tokamak[labeli].control = False
        
        
        self.dt_step = full_timestep
        self.plasma_norm_factor = plasma_norm_factor
    
        self.max_internal_timestep = max_internal_timestep
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
        n_coils = len(coils_dict.keys())
        self.len_currents = n_coils
        vessel_currents_vec = np.zeros(n_coils)
        eq_currents = eq.tokamak.getCurrents()
        for i,labeli in enumerate(coils_dict.keys()):
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
        for i,labeli in enumerate(coils_dict.keys()):
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
        self.NK.solve(self.eq1, self.profiles1, rel_convergence=rtol_NK)

        self.eq2.plasma_psi = 1.0*self.eq1.plasma_psi


    def initialize_from_ICs(self, eq, profile, rtol_NK=1e-8, reset_dJ=True): # eq for ICs, with all properties set up at time t=0, 
                            # ie with eq.tokamak.currents = I(t) and eq.plasmaCurrent = I_p(t) 
        
        self.step_counter = 0

        #get profile parametrization
        self.get_profiles_values(profile)

        #ensure it's a GS solution
        self.NK.solve(eq, profile, rel_convergence=rtol_NK)

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
        for i,labeli in enumerate(coils_dict):
            eq.tokamak[labeli].current = self.vessel_currents_vec[i]
        # assign plasma current to equilibrium
        eq._current = self.plasma_norm_factor*currents_vec[-1]



    def guess_J_from_extrapolation(self, alpha, rtol_NK):
        # run after step is complete and assigned, prepares for next step

        if self.step_no >= self.extrapolator_input_size:
            currents_guess = self.extrapolator.in_out(self.currents_vec)

            self.assign_currents(currents_vec=currents_guess, profile=self.profiles2, eq=self.eq2)
            self.NK.solve(self.eq2, self.profiles2, rel_convergence=rtol_NK)

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
    
    



    def iterative_unit_J1(self, J1,
                                active_voltage_vec,
                                rtol_NK):
        simplified_c1 = 1.0*self.simplified_solver_J1.stepper(It=self.currents_vec_m1,
                                                                norm_red_Iy_m1=self.J0_m1, 
                                                                norm_red_Iy0=self.J0, 
                                                                norm_red_Iy1=J1, 
                                                                active_voltage_vec=active_voltage_vec,
                                                                central_2=self.central_2)
        self.Fresidual_J1(trial_currents=simplified_c1, 
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
                                verbose=False):
        
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
            self.J1n = (1-alpha)*self.J1 + alpha*self.reduce_normalize(self.profiles2.jtor)
            self.J1n /= np.sum(self.J1n)
            self.ddJ = self.J1n - self.J1
            self.J1 = 1.0*self.J1n
            simplified_c1, res = self.iterative_unit_J1(J1=self.J1,
                                                        active_voltage_vec=active_voltage_vec,
                                                        rtol_NK=rtol_NK)   

            abs_increments = np.abs(simplified_c-simplified_c1)
            # dcurrents = np.abs(simplified_c1-self.currents_vec)
            # vals_for_check = np.where(dcurrents>threshold, dcurrents, threshold)
            # rel_residuals = np.abs(res)#/vals_for_check
            control = np.any(abs_increments>atol_currents)
            # control += np.any(rel_residuals>rtol_residuals)
            control += np.any(self.ddJ>atol_J)
            if verbose:
                print('max currents change = ', np.max(abs_increments))
                print('max J direction change = ', np.max(np.abs(self.ddJ)), np.linalg.norm(self.ddJ))
                print('max circuit eq residual (dim of currents) = ', np.argmax(abs(res)), res)
                print(simplified_c1 - self.currents_vec_m1)

            iterative_steps += 1

            simplified_c = 1.0*simplified_c1
        
        self.time += self.dt_step
        self.step_complete_assign(simplified_c)
        if use_extrapolation:
            self.guess_J_from_extrapolation(alpha=alpha, rtol_NK=rtol_NK)
        
        flag = check_against_the_wall(jtor=self.profiles2.jtor, 
                                      boole_mask_outside_limiter=self.mask_outside_limiter)

        return flag
    

    def Fresidual_dJ(self, trial_currents, active_voltage_vec, rtol_NK=1e-8):
        # trial_currents is the full array of intermediate results from euler solver
        # root problem for circuit equation
        # collects both metal normal modes and norm_plasma
        
        # current at t+dt
        # d_current_tpdt = np.sum(trial_currents, axis=-1)

        self.assign_currents(trial_currents, profile=self.profiles2, eq=self.eq2)
        self.NK.solve(self.eq2, self.profiles2, rel_convergence=rtol_NK)
        self.red_Iy_trial = self.Iyplasmafromjtor(self.profiles2.jtor)

        # self.red_Iy_dot = (self.red_Iy_trial - self.red_Iy)/self.dt_step
        self.red_Iy_dot = (self.red_Iy_trial - self.red_Iy_m1)/(2*self.dt_step)
        self.Id_dot = ((trial_currents - self.currents_vec_m1)/(2*self.dt_step))[:-1]

        self.forcing_term = self.evol_metal_curr.forcing_term_eig_plasma(active_voltage_vec=active_voltage_vec, 
                                                                         Iydot=self.red_Iy_dot)

        
        self.residual[:-1] = 1.0*self.evol_metal_curr.current_residual( Itpdt=trial_currents[:-1], 
                                                                        Iddot=self.Id_dot, 
                                                                        forcing_term=self.forcing_term)

        
        self.residual[-1] = 1.0*self.evol_plasma_curr.current_residual( red_Iy0=self.red_Iy, 
                                                                        red_Iy1=self.red_Iy_trial,
                                                                        red_Iydot=self.red_Iy_dot,
                                                                        Iddot=self.Id_dot)/self.plasma_norm_factor
        # return self.residual

    def Fresidual_J1(self, trial_currents, active_voltage_vec, rtol_NK=1e-8):
        # trial_currents is the full array of intermediate results from euler solver
        # root problem for circuit equation
        # collects both metal normal modes and norm_plasma
        
        # current at t+dt
        # current_tpdt = 1.0*trial_currents#[:, -1]
        self.assign_currents(trial_currents, profile=self.profiles2, eq=self.eq2)
        self.NK.solve(self.eq2, self.profiles2, rel_convergence=rtol_NK)
        self.red_Iy_trial = self.Iyplasmafromjtor(self.profiles2.jtor)

        self.red_Iy_dot = (self.red_Iy_trial - self.red_Iy_m1)/(2*self.dt_step)
        self.Id_dot = ((trial_currents - self.currents_vec_m1)/(2*self.dt_step))[:-1]

        self.forcing_term = self.evol_metal_curr.forcing_term_eig_plasma(active_voltage_vec=active_voltage_vec, 
                                                                         Iydot=self.red_Iy_dot)

        # mean_curr = np.mean(trial_currents, axis=-1)                                                                 
        self.residual[:-1] = 1.0*self.evol_metal_curr.current_residual( Itpdt=trial_currents[:-1], 
                                                                        Iddot=self.Id_dot, 
                                                                        forcing_term=self.forcing_term)


        # mean_Iy = trial_currents[-1]*self.J1*self.plasma_norm_factor
        mean_Iy = 1.0*self.red_Iy_trial
        self.residual[-1] = 1.0*self.evol_plasma_curr.current_residual( red_Iy0=self.red_Iy, 
                                                                        red_Iy1=mean_Iy,
                                                                        red_Iydot=self.red_Iy_dot,
                                                                        Iddot=self.Id_dot)/self.plasma_norm_factor
        # return self.residual



    
    def iterative_unit_dJ(self, dJ,
                                active_voltage_vec,
                                Rp, 
                                rtol_NK):
        simplified_c1 = self.central_2*self.simplified_solver_dJ.stepper(It=self.currents_vec_m1,
                                                            norm_red_Iy0=self.J0, 
                                                            norm_red_Iy_dot=dJ, 
                                                            active_voltage_vec=active_voltage_vec, 
                                                            Rp=Rp,
                                                            central_2=self.central_2)
        
        # calculate t+dt currents
        # plasma
        Iy_tpdt = self.red_Iy_m1/self.plasma_norm_factor + simplified_c1[-1]*dJ
        simplified_c1[-1] = np.sum(Iy_tpdt)
        # metal
        simplified_c1[:-1] += self.currents_vec_m1[:-1]
        
        self.Fresidual_dJ(trial_currents=simplified_c1, 
                                active_voltage_vec=active_voltage_vec, 
                                rtol_NK=rtol_NK)   
        return simplified_c1, self.residual
    


    def nl_step_iterative_dJ(self,  active_voltage_vec, 
                                    dJ,
                                    alpha=.8, 
                                    rtol_NK=5e-4,
                                    atol_currents=1e-3,
                                    atol_J=1e-3,
                                    verbose=False,
                                    use_extrapolation=True,
                                    ):
        
        self.central_2  = (1 + (self.step_no>0))
        
        Rp = self.calc_plasma_resistance(self.J0, self.J0_m1)
        self.dJ = 1.0*dJ
        
        simplified_c, res = self.iterative_unit_dJ(dJ=dJ,
                                                   active_voltage_vec=active_voltage_vec,
                                                   Rp=Rp, 
                                                   rtol_NK=rtol_NK)

        # dcurrents = np.abs(simplified_c-self.currents_vec)
        # vals_for_check = np.where(dcurrents>threshold, dcurrents, threshold)

        iterative_steps = 0
        control = 1
        while control:
            
            # if verbose:
            #     plt.figure()
            #     plt.imshow(self.rebuild_grid_map(self.dJ))
            #     plt.colorbar()
            #     plt.title(str(np.sum(self.dJ))+'   '+str(simplified_c[-1]-self.currents_vec_m1[-1]))

            self.dJ1 = self.reduce_normalize(self.profiles2.jtor - self.jtor_m1)
            self.dJ1 = (1-alpha)*self.dJ + alpha*self.dJ1
            # self.dJ1 /= np.linalg.norm(self.dJ1)
            self.dJ1 /= np.sum(self.dJ1)
            self.ddJ = self.dJ1 - self.dJ
            self.dJ = 1.0*self.dJ1
            simplified_c1, res = self.iterative_unit_dJ(dJ=self.dJ, 
                                                        active_voltage_vec=active_voltage_vec,
                                                        Rp=Rp, 
                                                        rtol_NK=rtol_NK)   

            abs_increments = np.abs(simplified_c - simplified_c1)
            # dcurrents = np.abs(simplified_c1-self.currents_vec)
            # vals_for_check = np.where(dcurrents>threshold, dcurrents, threshold)
            # rel_residuals = np.abs(res)#/vals_for_check
            control = np.any(abs_increments>atol_currents)
            # control += np.any(rel_residuals>rtol_residuals)
            control += np.any(np.abs(self.ddJ)>atol_J)         
            if verbose:
                print('max currents change = ', np.max(abs_increments))
                print('max J direction change = ', np.max(np.abs(self.ddJ)), np.linalg.norm(self.ddJ))
                print('max circuit eq residual (dim of currents) = ', np.argmax(abs(res)), res)
                print(simplified_c1 - self.currents_vec_m1)

            iterative_steps += 1

            simplified_c = 1.0*simplified_c1
        
        self.time += self.dt_step
        self.step_complete_assign(simplified_c)

        if use_extrapolation:
            self.guess_J_from_extrapolation(alpha=alpha, rtol_NK=rtol_NK)

        flag = check_against_the_wall(jtor=self.profiles2.jtor, 
                                      boole_mask_outside_limiter=self.mask_outside_limiter)

        return flag



    def nl_mix_unit(self, active_voltage_vec,
                                 Rp, 
                                 rtol_NK,
                                 n_k=10, # max number of basis vectors (must be less than number of modes + 1)
                                 conv_crit=.1,   #add basis vector 
                                                #if unexplained orthogonal component is larger than
                                 max_collinearity=.3,
                                 grad_eps=.005, #infinitesimal step
                                 clip=3):

        simplified_c, res = self.iterative_unit(active_voltage_vec=active_voltage_vec,
                                                 Rp=Rp, 
                                                 rtol_NK=rtol_NK)
        
        self.Arnoldi_iteration(trial_sol=simplified_c, #trial_current expansion point
                                vec_direction=-res, #first vector for current basis
                                Fresidual=res, #circuit eq. residual at trial_current expansion point: Fresidual(trial_current)
                                Fresidual_function=self.Fresidual,
                                active_voltage_vec=active_voltage_vec,
                                n_k=n_k, # max number of basis vectors (must be less than number of modes + 1)
                                conv_crit=conv_crit,   #add basis vector 
                                                #if unexplained orthogonal component is larger than
                                max_collinearity=max_collinearity,
                                grad_eps=grad_eps, #infinitesimal step
                                clip=clip)

        simplified_c1 = simplified_c + self.d_sol_step
        res1 = self.Fresidual(trial_currents=simplified_c1, 
                             active_voltage_vec=active_voltage_vec, 
                             rtol_NK=rtol_NK)   
        return simplified_c1, res1




    def nl_step_mix(self, active_voltage_vec, 
                                 alpha=.8, 
                                 rtol_NK=5e-4,
                                 atol_increments=1e-3,
                                 rtol_residuals=1e-3,
                                 n_k=10, # max number of basis vectors (must be less than number of modes + 1)
                                 conv_crit=.1,   #add basis vector 
                                                #if unexplained orthogonal component is larger than
                                 max_collinearity=.3,
                                 grad_eps=.005, #infinitesimal step
                                 clip=3,
                                 return_n_steps=False,
                                 verbose=False,
                                 threshold=.001):
        
        Rp = self.calc_plasma_resistance(self.J0, self.J0)
        
        simplified_c, res = self.nl_mix_unit(active_voltage_vec=active_voltage_vec,
                                                    Rp=Rp, 
                                                    rtol_NK=rtol_NK,
                                                    n_k=n_k, # max number of basis vectors (must be less than number of modes + 1)
                                                    conv_crit=conv_crit,   #add basis vector 
                                                                #if unexplained orthogonal component is larger than
                                                    max_collinearity=max_collinearity,
                                                    grad_eps=grad_eps, #infinitesimal step
                                                    clip=clip)

        dcurrents = np.abs(simplified_c-self.currents_vec)
        vals_for_check = np.where(dcurrents>threshold, dcurrents, threshold)                                        
        
        iterative_steps = 0
        control = 1
        while control:
            self.dJ = (1-alpha)*self.dJ + alpha*(self.reduce_normalize(self.profiles2.jtor - self.profiles1.jtor))
            simplified_c1, res = self.nl_mix_unit(active_voltage_vec=active_voltage_vec,
                                                    Rp=Rp, 
                                                    rtol_NK=rtol_NK,
                                                    n_k=n_k, # max number of basis vectors (must be less than number of modes + 1)
                                                    conv_crit=conv_crit,   #add basis vector 
                                                                #if unexplained orthogonal component is larger than
                                                    max_collinearity=max_collinearity,
                                                    grad_eps=grad_eps, #infinitesimal step
                                                    clip=clip)

            abs_increments = np.abs(simplified_c-simplified_c1)
            rel_residuals = np.abs(res)/vals_for_check
            control = np.any(abs_increments>atol_increments)
            control += np.any(rel_residuals>rtol_residuals)            
            if verbose:
                print(np.mean(abs_increments), np.mean(rel_residuals))

            iterative_steps += 1

            simplified_c = 1.0*simplified_c1
        
        self.time += self.dt_step
        self.step_complete_assign(simplified_c)

        if return_n_steps:
            return iterative_steps

    
    def nl_step_nk(self, trial_sol, #trial_current expansion point
                        active_voltage_vec,
                        n_k=10, # max number of basis vectors (must be less than number of modes + 1)
                        conv_crit=.2,   #add basis vector 
                                        #if unexplained orthogonal component is larger than
                        max_collinearity=.3,
                        grad_eps=.5, #infinitesimal step
                        clip=3,
                        rtol_NK=1e-5,
                        atol_currents=1e-3,
                        atol_J=1e-3,
                        verbose=False):
        
        self.Rp = self.calc_plasma_resistance(self.J0, self.J0)

        resJ = self.Fresidual_nk_dJ(trial_sol, active_voltage_vec=active_voltage_vec, rtol_NK=rtol_NK)
        

        simplified_c = 1.0*self.simplified_c1
        # dcurrents = np.abs(self.simplified_c1-self.currents_vec)
        # vals_for_check = np.where(dcurrents>threshold, dcurrents, threshold)

        iterative_steps = 0
        control = 1
        while control:
            self.Arnoldi_iteration(trial_sol=trial_sol, #trial_current expansion point
                                    vec_direction=-resJ, #first vector for current basis
                                    Fresidual=resJ, #circuit eq. residual at trial_current expansion point: Fresidual(trial_current)
                                    Fresidual_function=self.Fresidual_nk_dJ,
                                    active_voltage_vec=active_voltage_vec,
                                    n_k=n_k, # max number of basis vectors (must be less than number of modes + 1)
                                    conv_crit=conv_crit,   #add basis vector 
                                                    #if unexplained orthogonal component is larger than
                                    max_collinearity=max_collinearity,
                                    grad_eps=grad_eps, #infinitesimal step
                                    clip=clip)
            print(self.coeffs)
            trial_sol += self.d_sol_step
            resJ = self.Fresidual_nk_dJ(trial_sol, 
                             active_voltage_vec=active_voltage_vec, 
                             rtol_NK=rtol_NK)   
            
            
            abs_increments = np.abs(simplified_c-self.simplified_c1)
            # dcurrents = np.abs(simplified_c1-self.currents_vec)
            # vals_for_check = np.where(dcurrents>threshold, dcurrents, threshold)
            rel_residuals = np.abs(self.residual)#/vals_for_check
            control = np.any(abs_increments>atol_currents)
            # control += np.any(rel_residuals>rtol_residuals)
            control += np.any(resJ>atol_J)       
            if verbose:
                print('max currents change = ', np.max(abs_increments))
                print('max J direction change = ', np.max(np.abs(resJ)))
                print('max circuit eq residual (dim of currents) = ', np.max(rel_residuals))

            iterative_steps += 1

            simplified_c = 1.0*self.simplified_c1
        
        self.time += self.dt_step
        self.step_complete_assign(simplified_c)
        
        flag = check_against_the_wall(jtor=self.profiles2.jtor, 
                                      boole_mask_outside_limiter=self.mask_outside_limiter)

        return flag
    

    def LSQP(self, Fresidual, G, Q, clip=1):
        #solve the least sq problem in coeffs: min||G*coeffs+Fresidual||^2
        self.coeffs = np.matmul(np.matmul(np.linalg.inv(np.matmul(G.T, G)),
                                     G.T), -Fresidual)                            
        self.coeffs = np.clip(self.coeffs, -clip, clip)
        self.explained_res = np.sum(G*self.coeffs[np.newaxis,:], axis=1) 
        #get the associated step in candidate_d_sol space
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
                                grad_eps=.05, #infinitesimal step size, when compared to norm(trial)
                                clip=3,
                                rtol_NK=1e-5):
        
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
        grad_coeff = min(grad_eps*nFresidual, .005)

        print('norm trial_sol', np.linalg.norm(trial_sol))

        control = 1
        while control:
            # do step
            vec_direction = self.Arnoldi_unit(trial_sol, #trial_current expansion point
                                                vec_direction, #first vector for current basis
                                                Fresidual, #circuit eq. residual at trial_current expansion point: Fresidual(trial_current)
                                                Fresidual_function,
                                                active_voltage_vec,
                                                grad_coeff,
                                                rtol_NK=rtol_NK
                                                )
            collinear_control = 1 - np.any( np.sum(self.Gn[:,:self.n_it]*self.Gn[:,self.n_it:self.n_it+1], axis=0)>max_collinearity )
            self.n_it_tot += 1
            if collinear_control:
                self.n_it += 1
                self.LSQP(Fresidual, G=self.G[:,:self.n_it], Q=self.Q[:,:self.n_it], clip=clip)
                rel_unexpl_res = np.linalg.norm(self.explained_res + Fresidual)/nFresidual
                arnoldi_control = (rel_unexpl_res > conv_crit)

            control = arnoldi_control*collinear_control*(self.n_it_tot<n_k)



    














    # def set_currents_eq1(self, eq):
    #     #sets currents and initial plasma_psi in eq1
    #     eq_currents = eq.tokamak.getCurrents()
    #     currents_vec = np.zeros(len(eq_currents)+1)
    #     for i,labeli in enumerate(coils_dict.keys()):
    #         currents_vec[i] = eq_currents[labeli]
    #     currents_vec[-1] = eq.plasmaCurrent()
    #     self.currents_vec = currents_vec.copy()
    #     self.profiles1 = ConstrainPaxisIp(self.paxis, # Plasma pressure on axis [Pascals]
    #                                         eq.plasmaCurrent(), # Plasma current [Amps]
    #                                         self.fvac, # vacuum f = R*Bt
    #                                         alpha_m = self.alpha_m,
    #                                         alpha_n = self.alpha_n)
    #     self.eq1.plasma_psi = eq.plasma_psi.copy()
    #     self.eq2.plasma_psi = eq.plasma_psi.copy()


        
    # def assign_currents_1(self, currents_vec):
    #     #uses currents_vec to assign currents to both plasma and tokamak in eq/profiles
    #     self.profiles1 = ConstrainPaxisIp(self.paxis, # Plasma pressure on axis [Pascals]
    #                                                 currents_vec[-1], # Plasma current [Amps]
    #                                                 self.fvac, # vacuum f = R*Bt
    #                                                 alpha_m = self.alpha_m,
    #                                                 alpha_n = self.alpha_n) 
    #     for i,labeli in enumerate(coils_dict):
    #         self.eq1.tokamak[labeli].current = currents_vec[i]
    # def assign_currents_2(self, currents_vec):
    #     #uses currents_vec to assign currents to both plasma and tokamak in eq/profiles
    #     self.profiles2 = ConstrainPaxisIp(self.paxis, # Plasma pressure on axis [Pascals]
    #                                                 currents_vec[-1], # Plasma current [Amps]
    #                                                 self.fvac, # vacuum f = R*Bt
    #                                                 alpha_m = self.alpha_m,
    #                                                 alpha_n = self.alpha_n) 
    #     for i,labeli in enumerate(coils_dict):
    #         self.eq2.tokamak[labeli].current = currents_vec[i]

        
        
    # # def find_dt_evolve(self, U_active, max_rel_change, results=None, dR=0):
    # #     #solves linearized circuit eq on eq1
    # #     #currents at time t are as in eq1.tokamak and profiles1
    # #     #progressively smaller timestep dt_step is used
    # #     #up to achieving a relative change in the currents of max_rel_change
    # #     #since some currents can cross 0, rel change is calculated with respect to 
    # #     #either the current itself or a changeable threshold value, set in init as self.threshold
    # #     #the new currents are in self.new_currents
    # #     #this works using qfe quantities (i.e. inductances) based on eq1/profiles1
    # #     if results==None:
    # #         self.results = self.qfe.quants_out(self.eq1, self.profiles1)
    # #     else: 
    # #         self.results = results
        
    # #     rel_change_curr = np.ones(5)
    # #     dt_step = .002
    # #     while np.sum(abs(rel_change_curr)>max_rel_change):
    # #         dt_step /= 1.5
    # #         new_currents = self.evol_currents.new_currents_out(self.eq1, 
    # #                                                            self.results, 
    # #                                                            U_active, 
    # #                                                            dt_step,
    # #                                                            dR)
    # #         rel_change_curr = abs(new_currents-self.currents_vec)/self.vals_for_rel_change
    # #     #print('find_dt_evolve dt = ', dt_step)

    # #     #print('rel_change_currents = ', abs(rel_change_curr))
    # #     self.new_currents = new_currents
    # #     self.dt_step_plasma = dt_step
       
    


    
    # def update_R_matrix(self, trial_currents, rtol_NK=1e-8):#, verbose_NK=False):
    #     """calculates the matrix dL/dt using the previous estimate of I(t+dt)
    #     this is equivalent to a non diagonal resistance term, hence the name"""

    #     self.assign_currents_2(trial_currents)
    #     self.NK.solve(self.eq2, self.profiles2, rel_convergence=rtol_NK) #verbose_NK)
    #     #calculate new fluxes and inductances
    #     self.results1 = self.qfe.quants_out(self.eq2, self.profiles2)
    #     #self.Lplus1 = self.results1['plasma_ind_on_coils']
        
    #     dLpc = self.results1['plasma_ind_on_coils'] - self.results['plasma_ind_on_coils']
    #     dLpc /= self.dt_step_plasma

    #     dLcp = self.results1['plasma_coil_ind'] - self.results['plasma_coil_ind']
    #     dLcp /= self.dt_step_plasma
        
    #     dR = self.void_matrix.copy()
    #     dR[:,-1] = dLpc
    #     dR[-1,:-1] = dLcp

    #     return dR

 

    # def Fcircuit(self,  trial_currents,
    #                     rtol_NK=1e-7):
    #                   #,  verbose_NK=False):
    #     self._dR = self.update_R_matrix(trial_currents, rtol_NK)#, verbose_NK)
    #     new_currents = self.evol_currents.stepper_adapt_repeat(self.currents_vec, dR=self._dR)
    #     return (new_currents-trial_currents)/self.vals_for_rel_change


    # def dI(self, res0, G, Q, clip=5):
    # #solve the least sq problem in coeffs: min||G.coeffs+res0||^2
    #     self.coeffs = np.matmul(np.matmul(np.linalg.inv(np.matmul(G.T, G)),
    #                                       G.T), -res0)
    #     # print('intermediate_coeffs = ', self.coeffs)
    #     self.coeffs = np.clip(self.coeffs, -clip, clip)
    #     self.eplained_res = np.sum(G*self.coeffs[np.newaxis,:], axis=1)
    #     #get the associated step in I space
    #     self.di_Arnoldi = np.sum(Q*self.coeffs[np.newaxis,:], axis=1)

            
  


    # def Arnoldi_iter(self,      trial_currents, #expansion point of the root problem function
    #                             vec_direction, #first direction for I basis
    #                             Fresidual, #residual of trial currents
    #                             n_k=10, #max number of basis vectors, must be smaller than tot number of coils+1
    #                             conv_crit=.5, #add basis vector 
    #                                           #if orthogonal residual is larger than
    #                             grad_eps=.001, #relative magnitude of infinitesimal step, with respect to self.vals_for_rel_change 
    #                                            #adjust in line to 0.1*max_rel_change
    #                             accept_threshold = .2,
    #                             max_collinearity = .6,
    #                             clip = 1.5,
    #                             verbose=False
    #                             ):

    #     # clip_vec = np.array([clip])

    #     ntrial_currents = np.linalg.norm(self.trial_currents)
    #     nFresidual = np.linalg.norm(Fresidual)
    #     dFresidual = Fresidual/nFresidual
    #     #print('nFresidual', nFresidual)

    #     #basis in input space
    #     Q = np.zeros((self.len_currents, n_k+1))
    #     #orthonormal version of the basis above
    #     Qn = np.zeros((self.len_currents, n_k+1))
    #     #basis in residual space
    #     G = np.zeros((self.len_currents, n_k+1))
    #     #normalized version of the basis above
    #     Gn = np.zeros((self.len_currents, n_k+1))
        
    #     n_it = 0
    #     #control on whether to add a new basis vector
    #     arnoldi_control = 1
    #     #use at least 1 orthogonal terms, but not more than n_k
    #     failure_count = 0
    #     while ((n_it<1)+(arnoldi_control>0))*(n_it<n_k)*((failure_count<2)+(n_it<1))>0:

    #         #print('failure_count', failure_count)
    #         nvec_direction = np.linalg.norm(vec_direction)
    #         grad_coeff = grad_eps*ntrial_currents/nvec_direction*nFresidual/.01
    #         candidate_di = vec_direction*min(grad_coeff, 200/abs(vec_direction[-1]))
    #         ri = self.Fcircuit(trial_currents + candidate_di)
    #         #print('trial currents in arnoldi', trial_currents)
    #         #internal_res = np.abs(ri)/self.vals_for_rel_change
    #         #print('internal residual = ', max(internal_res), np.argmax(internal_res), internal_res.mean())
    #         #print('all residual', internal_res)
    #         candidate_usable = ri - Fresidual
    #         ncandidate_usable = np.linalg.norm(candidate_usable)
    #         di_factor = ncandidate_usable/nFresidual
    #         #print('di_factor', di_factor)

    #         if ((di_factor<.3)*(abs(candidate_di[-1])<150))+(di_factor>6):
    #             #print('using factor = ', 1/di_factor)
    #             candidate_di *= min(1/di_factor, 200/abs(candidate_di[-1]))
    #             ri = self.Fcircuit(trial_currents + candidate_di)
    #             candidate_usable = ri - Fresidual
    #             ncandidate_usable = np.linalg.norm(candidate_usable)
    #             di_factor = ncandidate_usable/nFresidual
            
    #         #print('dcurrent = ', candidate_di)
            
    #         dcandidate_usable = candidate_usable/ncandidate_usable
    #         costerm = abs(np.sum(dFresidual*dcandidate_usable)) 
    #         if verbose:
    #             print('candidate_di = ', candidate_di)
    #             print('usable/residual = ', candidate_usable/Fresidual)
    #             print('costerm', costerm)

    #         if costerm>accept_threshold:
    #             collinearity = (np.sum(dcandidate_usable[:,np.newaxis]*Gn[:,:n_it], axis=0) > max_collinearity)
    #             if np.sum(collinearity):
    #                 #print('not accepting this term!, ', collinearity)
    #                 idx = np.random.randint(50)
    #                 vec_direction += nvec_direction*self.w[:, idx]
    #                 conv_crit = .95
    #                 #print('reshuffled', idx)
    #                 failure_count += 1


    #             else:
    #                 Q[:,n_it] = candidate_di.copy()
    #                 Qn[:,n_it] = Q[:,n_it]/np.linalg.norm(Q[:,n_it])
                    
    #                 G[:,n_it] = candidate_usable.copy()
    #                 Gn[:,n_it] = dcandidate_usable.copy()

    #                 n_it += 1
    #                 self.G = G[:,:n_it]
    #                 self.Q = Q[:,:n_it]
    #                 self.dI(Fresidual, G=self.G, Q=self.Q, clip=clip)
    #                 rel_unexpl_res = np.linalg.norm(self.eplained_res+Fresidual)/nFresidual
    #                 if verbose:
    #                     print('relative_unexplained_residual = ', rel_unexpl_res)
    #                 arnoldi_control = (rel_unexpl_res > conv_crit)

    #                 vec_direction = candidate_usable*self.vals_for_rel_change
    #                 #vec_direction -= np.sum(np.sum(Qn[:,:n_it]*vec_direction[:,np.newaxis], axis=0, keepdims=True)*Qn[:,:n_it], axis=1)
    #         else: 
    #             #print('costerm too small!')
    #             idx = np.random.randint(50)
    #             vec_direction += nvec_direction*self.w[:, idx]
    #             grad_eps = .0001
    #             accept_threshold /= 1.2
    #             #print('reshuffled', idx)
    #             failure_count += 1

                
    #     return n_it

   

    

    
    # def get_max_rel_change(self, U_active, # active potential
    #                             max_dt_step_currents = .00002, # do multiple timesteps in Euler using constant \dotL if dt_step_plasma larger than
    #                             rtol_NK=1e-6, # for convergence of the NK solver of GS
    #                             ): 
    #     self.evol_currents.initialize_time_t(self.results)
    #     self.evol_currents.determine_currents_stepsize(self.dt_step_plasma, max_dt_step_currents)
    #     self.trial_currents = self.evol_currents.stepper_adapt_first(self.currents_vec, U_active, self.dR)
    #     Fresidual = self.Fcircuit(self.trial_currents, rtol_NK)#, verbose_NK)
    #     rel_change = abs(Fresidual)
    #     return max(rel_change), Fresidual


    # def do_step_free(self,  U_active, # active potential
    #                     max_rel_residual=.0013, # aim for a dt such that the maximum relative change in the currents is max_rel_change
    #                     max_dt_step_currents = .00002, # do multiple timesteps in Euler using constant \dotL if dt_step_plasma larger than
    #                     rtol_NK=1e-6, # for convergence of the NK solver of GS
    #                     # verbose_NK=False,
    #                     rtol_currents=3e-4, #for convergence of the circuit equation
    #                     max_iter=10, # if more iterative steps are required, dt is reduced
    #                     n_k=10, # maximum number of terms in Arnoldi expansion
    #                     conv_crit=.15, # add more Arnoldi terms if residual is still larger than
    #                     grad_eps=.00003,
    #                     clip=1.5,
    #                     cap_timestep=1, # if self-determined timestep is longer than, it is clipped at cap_timestep
    #                     verbose=False
    #                     ): 
        

    #     """advances both plasma and currents according to complete linearized eq.
    #     Uses an NK iterative scheme to find consistent values of I(t+dt) and L(t+dt)
    #     Does not modify the input object eq, rather the evolved plasma is in self.eq1
    #     Outputs a flag that =1 if plasma is hitting the wall 
    #     Timestepping necessary for convergence is self-determined, hence not uniform"""
        
    #     abs_currents = abs(self.currents_vec)
    #     self.vals_for_rel_change = np.where(abs_currents>self.threshold, abs_currents, self.threshold)

    #     not_done_flag = 1
    #     while not_done_flag:

    #         # determine if dt_step_plasma is of suitable size by checking residual
    #         max_rel_change, Fresidual = self.get_max_rel_change(U_active, # active potential
    #                                             max_dt_step_currents, # do multiple timesteps in Euler using constant \dotL if dt_step_plasma larger than
    #                                             rtol_NK, # for convergence of the NK solver of GS
    #                                             )

    #         if max_rel_change<.6*max_rel_residual:
    #             #print('increasing timestep')
    #             self.dt_step_plasma *= min(3,(.8*max_rel_residual/max_rel_change))
    #             max_rel_change, Fresidual = self.get_max_rel_change(U_active, # active potential
    #                                             max_dt_step_currents, # do multiple timesteps in Euler using constant \dotL if dt_step_plasma larger than
    #                                             rtol_NK, # for convergence of the NK solver of GS
    #                                             )
    #         while (max_rel_change > max_rel_residual)*(self.dt_step_plasma > 1e-5):
    #             #print('reducing timestep')
    #             self.dt_step_plasma *= (.8*max_rel_residual/max_rel_change)
    #             max_rel_change, Fresidual = self.get_max_rel_change(U_active, # active potential
    #                                             max_dt_step_currents, # do multiple timesteps in Euler using constant \dotL if dt_step_plasma larger than
    #                                             rtol_NK, # for convergence of the NK solver of GS
    #                                             )

            
    #         if self.dt_step_plasma>cap_timestep:
    #             self.dt_step_plasma = cap_timestep
    #             max_rel_change, Fresidual = self.get_max_rel_change(U_active, # active potential
    #                                             max_dt_step_currents, # do multiple timesteps in Euler using constant \dotL if dt_step_plasma larger than
    #                                             rtol_NK, # for convergence of the NK solver of GS
    #                                             )


    #         if verbose:
    #             print('chosen timestep = ', self.dt_step_plasma)
    #             #print('currents step from LdI/dt only = ', self.trial_currents-self.currents_vec)
    #             #print('initial residual = ', Fresidual)
    #             print('initial max_rel_change = ', max_rel_change)
    #             #print('initial relative residual on inductances', max(abs(self.Lplus1-self.Lplus)/self.Lplus))
            
    #         if max_rel_change<rtol_currents: 
    #             not_done_flag=0

    #         it=0
    #         while it<max_iter and not_done_flag:
    #             self.Arnoldi_iter( self.trial_currents, 
    #                                 Fresidual*self.vals_for_rel_change, #starting direction in I space
    #                                 Fresidual, #F(trial_currents)
    #                                 n_k, 
    #                                 conv_crit, 
    #                                 grad_eps, 
    #                                 clip=clip,
    #                                 verbose=verbose)
    #             #print('self.di_Arnoldi', self.di_Arnoldi)
    #             #print('trial current before update', self.trial_currents)
    #             self.trial_currents += self.di_Arnoldi
    #             Fresidual = self.Fcircuit(self.trial_currents)
    #             #print('full residual after update', Fresidual)
    #             rel_change = abs(Fresidual)#/self.vals_for_rel_change
    #             max_rel_change = max(rel_change)
    #             if max_rel_change<rtol_currents: 
    #                 not_done_flag=0
                
    #             if verbose:
    #                 print(' coeffs= ', self.coeffs)
    #                 #print('new residual = ', Fresidual)
    #                 print('new max_rel_change = ', max_rel_change, np.argmax(rel_change), rel_change.mean())
                    
    #             it += 1
            
    #         if it==max_iter:
    #             #if max_iter was hit, then message:
    #             print(f'failed to converge with less than {max_iter} iterations.') 
    #             print(f'Last max rel_change={max_rel_change}.')
    #             print('Restarting with smaller timestep')
    #             max_rel_residual /= 2
    #             grad_eps /= 2
                
    #     # if verbose:
    #     #     print('number of Arnoldi iterations = ', it)

    #     #get ready for next step
    #     self.old_currents = self.currents_vec.copy()
    #     self.currents_vec = self.trial_currents.copy()
    #     self.assign_currents_1(self.currents_vec)
    #     self.eq1.plasma_psi = self.eq2.plasma_psi.copy()
    #     self.old_results = self.results.copy()
    #     # self.results = self.results1.copy()
    #     self.dR = self._dR.copy()
        
    #     self.plasma_against_wall = np.sum(self.mask_outside_reactor*self.results['separatrix'][1])
    #     return self.plasma_against_wall



    # def do_step(self,  U_active, # active potential
    #                     max_rel_residual=.0016, # aim for a dt such that the maximum relative change in the currents is max_rel_change
    #                     max_dt_step_currents = .00002, # do multiple timesteps in Euler using constant \dotL if dt_step_plasma larger than
    #                     rtol_NK=1e-6, # for convergence of the NK solver of GS
    #                     # verbose_NK=False,
    #                     rtol_currents=3e-4, #for convergence of the circuit equation
    #                     verbose=False,
    #                     max_iter=10, # if more iterative steps are required, dt is reduced
    #                     n_k=10, # maximum number of terms in Arnoldi expansion
    #                     conv_crit=.15, # add more Arnoldi terms if residual is still larger than
    #                     grad_eps=.00003,
    #                     clip=1.5,
    #                     force_timestep=0.0002 # set desired timestep                        
    #                     ): 
                        
    #     """advances both plasma and currents according to complete linearized eq.
    #     Uses an NK iterative scheme to find consistent values of I(t+dt) and L(t+dt)
    #     Does not modify the input object eq, rather the evolved plasma is in self.eq1
    #     Outputs a flag that =1 if plasma is hitting the wall 
    #     Timestepping necessary for convergence is self-determined, hence not uniform"""
        
    #     dt_advance = 0
    #     self.n_plasma_steps = 0

    #     while (dt_advance<force_timestep)*(self.plasma_against_wall<1):
    #         self.n_plasma_steps += 1
    #         cap_timestep = force_timestep - dt_advance
    #         self.do_step_free(U_active, # active potential
    #                     max_rel_residual, # aim for a dt such that the maximum relative change in the currents is max_rel_change
    #                     max_dt_step_currents, # do multiple timesteps in Euler using constant \dotL if dt_step_plasma larger than
    #                     rtol_NK, # for convergence of the NK solver of GS
    #                     # verbose_NK=False,
    #                     rtol_currents, # for convergence of the circuit equation
    #                     max_iter, # if more iterative steps are required, dt is reduced
    #                     n_k, # maximum number of terms in Arnoldi expansion
    #                     conv_crit, # add more Arnoldi terms if residual is still larger than
    #                     grad_eps,
    #                     clip,
    #                     cap_timestep=cap_timestep, # if self-determined timestep is longer than, it is clipped at cap_timestep
    #                     verbose=verbose
    #                     )
    #         dt_advance += self.dt_step_plasma
        
    #     self.dt_step = dt_advance

    #     return self.plasma_against_wall



    # def do_step_(self,  U_active, # active potential
    #                     force_timestep=0.0002, # set desired timestep    
    #                     max_dt_step_currents = .00002, # do multiple timesteps in Euler using constant \dotL if dt_step_plasma larger than
    #                     rtol_NK=1e-6, # for convergence of the NK solver of GS
    #                     rtol_dI=3e-4, #for convergence of the circuit equation
    #                     verbose=False
    #                     ): 

        
        
    #     return 