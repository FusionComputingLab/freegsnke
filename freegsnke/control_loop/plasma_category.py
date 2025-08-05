"""
Module to implement plasma control in FreeGSNKE control loops. 

"""

import numpy as np
from useful_functions import interpolate_spline, interpolate_step

class PlasmaController:
    """
    ADD DESCRIP.

    Parameters
    ----------


    Attributes
    ----------

    """

    def __init__(
        self,
        plasma_data,
    ):

        # create an internal copy of the data
        self.data = plasma_data

        # interpolate what needs to be interpolated
        self.ip_fb = self.interpolate_spline(data=self.data["ip_fb"])
        self.ip_blend = self.interpolate_spline(data=self.data["ip_blend"])
        self.vloop_ff = self.interpolate_spline(data=self.data["vloop_ff"])
        self.k_prop = self.interpolate_step(data=self.data["k_prop"])
        self.k_int = self.interpolate_step(data=self.data["k_int"])
        self.M_solenoid = self.interpolate_step(data=self.data["M_solenoid"])

    def calculate_solenoid_delta(
        self,
        Kp,
        Ki,
        dt,
        blend,
        Ip_req,
        Ip_obs_tprev,
        Ip_obs_t,
        Vloop_ff,
        sol_vc,
        M_sp=1.5801 * 1e-5 * 1e3,
        dt_pi=0.0001,
        reshape=False,
    ):
        """
        Calculates the vector of current trajectories ΔI/Δt, as prescribed
        in the plasma category of the MAST-U PCS. The equations followed are:

        Ip_error = (Ip_req - Ip_obs)
        integral = internal_state + 0.5 * Ip_error * dt
        internal_state = internal_state + Ip_error * dt
        ΔIsol_fb/Δt = Kp * Ip_error + Ki * integral
        ΔIsol/Δt = ΔIsol_fb/Δt * blend - Vloop_ff * (1 - blend)/M_sp
        ΔI/Δt = ΔIsol/Δt * vc_vector

        It should be noted that the PI controller works at a frequency twice as
        high as the data recording system. This is why the PI controller goes
        through two cycles in this method.

        Parameters
        ----------
        - Kp : float
            Proportional term used in the Vloop_fb computation.
        - Ki : float
            Integral term used in the Vloop_fb computation.
        - dt : float
            Interval of time between two timestamps of the recorded data.
        - blend : float
            Value between 0 and 1 that acts as a weight in the Vloops sum.
        - Ip_req : float
            The plasma current requested at this PI controller cycle.
        - Ip_obs_tprev : float
            The plasma current observed at the previous PI controller.
        - Ip_obs_t : float
            The plasma current observed.
        - Vloop_ff : float
            The feedforward Vloop.
        - sol_vc : 1D numpy array
            The solenoid Virtual Circuit. The element corresponding to the
            solenoid of this array should always be 1.
        - M_sp : float
            Mutual inductance betwen the solenoid and the plasma. The default
            value is 1.5801*1e-2 Henrys.
        - dt_pi : float
            Interval of time between two cycles of the PI controller. The
            default value is 0.0001
        - reshape : bool
            flag as to whether to reshape vector of currents. Defaults to False.

        Returns
        -------
        - dI_dt : 1D numpy array
            Array of delta currents requests that will be part of the input of
            Circuits category.

        """

        # PI controller cycles. The integral sum is done following the
        # trapezoidal rule.
        # First cycle

        # Compute the required change for Ip
        delta_Ip_tprev = Ip_req - Ip_obs_tprev
        # integral_tprev = self.internal + (delta_Ip_tprev * dt_pi) / 2
        self.internal += delta_Ip_tprev * dt_pi
        # dIsol_fb_tprev = Kp * delta_Ip_tprev + Ki * integral_tprev

        # Second cycle
        delta_Ip_t = Ip_req - Ip_obs_t
        integral_t = self.internal + (delta_Ip_t * dt_pi) / 2
        self.internal += delta_Ip_t * dt_pi
        dIsol_fb_t = Kp * delta_Ip_t + Ki * integral_t

        # Compute the loop voltage as a weighted sum
        dIsoldt = blend * dIsol_fb_t - (1 - blend) * Vloop_ff * (1 / M_sp)

        # Apply dIsoldt to virtual circuit vector to get the current
        # trajectories requests for the active coils.
        # dI_dt = dIsoldt * sol_vc
        dI_dt = sol_vc @ dIsoldt
        if reshape == False:
            return dI_dt
        elif reshape == True:
            # option 1 reshape currents, fill in zeros and multiply by inductance matrix
            print("reshaping current to match active coils order")
            reshaped_currents = np.zeros(len(self.active_coils))
            for i, coil in enumerate(self.vc_coil_order):
                # PCO patch until we sort this out
                if coil == "pc":
                    continue
                reshaped_currents[self.active_coil_order_dictionary[coil]] = (
                    1.0
                    * dI_dt[
                        i
                    ]  # multiply by 1.0 to get around the pointer/reference feature of python.
                )

            return reshaped_currents

    def ip_control(self, ts, prev_ts, Ip_obs_tprev=None, Ip_obs_t=None, eq=None):
        """
        Execute all the steps in the pipeline for the control of the solenoid
        current, as prescribed by the PCS. This method is the API by design of
        the class; it takes a time_stamp and with the knowledge of the status
        of the plasma it computes the control voltage for the solenoid current.

        Parameters
        ----------
        - ts : float (4 decimal places)
            Timestamp for which this pipeline should provide a control voltage.
        - prev_ts : float (4 decimal places)
             Timestamp of the previous recorded data
        - Ip_obs_tprev : float
            The observed plasma current at `prev_ts` timestamp. Defaults to
            None, in which case the observed plasma current is taken from the
            equilibrium/control params dictionary.
        - Ip_obs_t : float
            The observed plasma current. Defaults to None, in which case the
            observed plasma current is taken from the equilibrium/control
            params dictionary.
        - eq : equilibrium object
            optional equilibrium object

        Returns
        -------
        numpy 1D array
           Trajectories for the active coil currents due to the control on the
           solenoid coil.

        """
        # Implement the plasma category. First, the relevant entities should be
        # retrieved from the scheduler
        # Ip_req = self.scheduler.get_waveform_value(
        #     param_type="Ip", param="plasma", time_stamp=ts
        # )
        Ip_req = self.scheduler.get_wave_values(time_stamp=ts, wave_type="Ip")[0]

        if not Ip_req:
            print(f"  The plasma current is not controlled at t: {ts}")
            # return array of zeros if not controlled
            dI_dt = np.zeros_like(self.scheduler.vc_dict)
            return dI_dt

        # Vloop_req = self.scheduler.get_waveform_value(
        #     param_type="ff", param="plasma", time_stamp=ts
        # )
        Vloop_req = self.scheduler.get_wave_values(time_stamp=ts, wave_type="ff")

        # gain_p, _ = self.scheduler.get_gains(["plasma"], time_stamp=ts, K_type="Kprop")
        # gain_int, _ = self.scheduler.get_gains(["plasma"], time_stamp=ts, K_type="Kint")
        gain_p = self.scheduler.get_gains(time_stamp=ts, K_type="Kprop")
        gain_int = self.scheduler.get_gains(time_stamp=ts, K_type="Kint")

        # blend = self.scheduler.get_waveform_value(
        #     param_type="blends", param="plasma", time_stamp=ts
        # )
        blend = self.scheduler.get_blends(time_stamp=ts)[0]

        # sol_vc, vc_coil_order = self.scheduler.get_vc(ts)
        sol_vc = self.scheduler.get_vc(ts)

        dI_dt = self.calculate_solenoid_delta(
            Kp=gain_p[0],
            Ki=gain_int[0],
            dt=(ts - prev_ts),
            blend=blend,
            Ip_req=Ip_req,
            Ip_obs_tprev=Ip_obs_tprev,
            Ip_obs_t=Ip_obs_t,
            Vloop_ff=Vloop_req,
            sol_vc=sol_vc,
        )

        return dI_dt


# class SolenoidScheduler(TargetScheduler):
#     """
#     Child class of TargetScheduler specified for the solenoid. It stores the
#     times series information of the plasma current, the blend factor, the
#     plasma and the solenoid gains and the feedforward loop voltage.

#     Attributes
#     ----------
#     - All those of TargetScheduler.
#     - control_params : dictionary
#         A dictionary that contains the times series information of the control
#         parameters: plasma and solenoid gains, blend and feedforward loop
#         voltage.

#     Methods
#     -------
#     - retrieve_parameter : Retrieves the value of the queried control parameter
#     at time_stamp.

#     """

#     def __init__(
#         self,
#         waveform_dict,
#         schedule_dict,
#         sol_vc_dict,
#         controlled_targets_all=["Ip"],
#         solenoid_name="Solenoid",
#     ):
#         """
#         Initialise the Solenoid scheduler.

#         Arguments
#         ---------
#         - waveform_dict : dict
#             dictionary containing target waveform.
#         - schedule_dict : dict
#             dictionary containing target schedule.
#         - sol_vc_dict : dict
#             dictionary containing the virtual circuit sequence.
#         - solenoid_name : str
#             A string to denote the solenoid current ("Solenoid", "P1", etc).
#             Defaults to "Solenoid" if not given.

#         Returns
#         -------
#         None

#         """

#         # Execute the parent __init__()
#         super().__init__(waveform_dict, schedule_dict, controlled_targets_all)

#         # Ip requested
#         self.ips = waveform_dict["Ip"]

#         if solenoid_name is None:
#             print(
#                 "A name for the solenoid is not provided, using 'Solenoid' "
#                 "as label for the solenoid current."
#             )
#             self.solenoid_name = "Solenoid"
#         else:
#             self.solenoid_name = solenoid_name

#         # Load the control parameters into a dictionary
#         self.vc_dict = sol_vc_dict

#     def get_vc(self, time_stamp):
#         """
#         Patch-job method to give access to the solenoid VC. The `vc` object is
#         assigned to the `vc` class attribute.

#         Returns
#         -------
#         numpy 1D array
#             The solenoid virtual circuit.

#         """
#         time_pos = self.get_schedule_time(time_stamp=time_stamp)
#         sol_vc = self.vc_dict[time_pos]["vc"]
#         # coil_order = self.vc_dict[time_pos]["coil_order"]

#         return np.array(sol_vc)  # coil_order
