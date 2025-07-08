"""
Module to control plasma current Ip during a tokamak shot.

"""

import numpy as np
from .target_scheduler import TargetScheduler


class ControlSolenoid:
    """
    Class to control the solenoid voltage applied on the solenoid to steer the
    plasma current.

    Parameters
    ----------
    - waveform_dict : dict
        dictionary containing ff, fb, blends, Ip waveforms.
    - schedule_dict : dict
        dictionary containing plasma schedule.
    - sol_vc_dict : dict
        dictionary containing VC schedule for the solenoid
    - contr_params_dict : str
        dictionary containing control parameters sequence.
    - integral_term_0 : float
        Value of the integral term at the beginning of the control.
    - solenoid_name : str
        A string to denote the solenoid current ("Solenoid", "P1", etc).
        Defaults to "Solenoid" if not given.

    Attributes
    ----------
    - scheduler : TargetScheduler
        An object that store information of the controlled target Ip and other
        control parameters: the plasma and solenoid gain, Vloop_ff, and blend.
    - vc : numpy 1D array
        The solenoid virtual circuit.

    """

    def __init__(
        self,
        waveform_dict,
        schedule_dict,
        sol_vc_dict,
        integral_term_0=0,
        solenoid_name=None,
    ):
        """
        Initialises the ControlSolenoid class.

        Returns
        -------
        None

        """
        # Load the scheduler
        self.scheduler = SolenoidScheduler(
            waveform_dict, schedule_dict, sol_vc_dict, solenoid_name
        )

        # The accumulated error in the plasma category. Defaults to 0 if not
        # given.
        self.internal = integral_term_0

    def calculate_solenoid_delta(self,
                                 Kp,
                                 Ki,
                                 dt,
                                 blend,
                                 Ip_req,
                                 Ip_obs_tprev,
                                 Ip_obs_t,
                                 Vloop_ff,
                                 sol_vc,
                                 M_sp=1.5801*1e-5 * 1e3,
                                 dt_pi=0.0001
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
        dI_dt = dIsoldt * sol_vc

        return dI_dt

    def ip_control(self,
                   ts,
                   prev_ts,
                   Ip_obs_tprev=None,
                   Ip_obs_t=None,
                   eq=None):
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

        if Ip_obs_t is None:
            Ip_obs_t = self.scheduler.get_observed_current(ts,
                                                           "Ip_obs",
                                                           eq)
            print(f"  Ip from equilibrium at {ts}: {Ip_obs_t}")
        if Ip_obs_tprev is None:
            Ip_obs_tprev = self.scheduler.get_observed_current(prev_ts,
                                                               "Ip_obs",
                                                               eq)
            print(f"  Ip from equilibrium at {prev_ts}: {Ip_obs_tprev}")

        # Implement the plasma category. First, the relevant entities should be
        # retrieved from the scheduler
        Ip_req = self.scheduler.get_waveform_value(
            param_type="Ip", param="plasma", time_stamp=ts
        )

        if not Ip_req:
            print(f"  The plasma current is not controlled at t: {ts}")
            # return array of zeros if not controlled
            dI_dt = np.zeros_like(self.scheduler.vc_dict)
            return dI_dt

        Vloop_req = self.scheduler.get_waveform_value(
            param_type="ff", param="plasma", time_stamp=ts
        )

        gain_p, _ = self.scheduler.get_gains(
                ["plasma"], time_stamp=ts, K_type="Kprop")
        gain_int, _ = self.scheduler.get_gains(
                ["plasma"], time_stamp=ts, K_type="Kint")

        blend = self.scheduler.get_waveform_value(
            param_type="blends", param="plasma", time_stamp=ts
        )

        dI_dt = self.calculate_solenoid_delta(
            Kp=gain_p[0],
            Ki=gain_int[0],
            dt=(ts - prev_ts),
            blend=blend,
            Ip_req=Ip_req,
            Ip_obs_tprev=Ip_obs_tprev,
            Ip_obs_t=Ip_obs_t,
            Vloop_ff=Vloop_req,
            sol_vc=self.scheduler.retrieve_vc(ts),
        )

        return dI_dt


class SolenoidScheduler(TargetScheduler):
    """
    Child class of TargetScheduler specified for the solenoid. It stores the
    times series information of the plasma current, the blend factor, the
    plasma and the solenoid gains and the feedforward loop voltage.

    Attributes
    ----------
    - All those of TargetScheduler.
    - control_params : dictionary
        A dictionary that contains the times series information of the control
        parameters: plasma and solenoid gains, blend and feedforward loop
        voltage.

    Methods
    -------
    - retrieve_parameter : Retrieves the value of the queried control parameter
    at time_stamp.

    """

    def __init__(
        self,
        waveform_dict,
        schedule_dict,
        sol_vc_dict,
        solenoid_name,
    ):
        """
        Initialise the Solenoid scheduler.

        Arguments
        ---------
        - waveform_dict : dict
            dictionary containing target waveform.
        - schedule_dict : dict
            dictionary containing target schedule.
        - sol_vc_dict : dict
            dictionary containing the virtual circuit sequence.
        - solenoid_name : str
            A string to denote the solenoid current ("Solenoid", "P1", etc).
            Defaults to "Solenoid" if not given.

        Returns
        -------
        None

        """

        # Execute the parent __init__()
        super().__init__(waveform_dict, schedule_dict)

        # Ip requested
        self.ips = waveform_dict["Ip"]

        if solenoid_name is None:
            print(
                "A name for the solenoid is not provided, using 'Solenoid' "
                "as label for the solenoid current."
            )
            self.solenoid_name = "Solenoid"
        else:
            self.solenoid_name = solenoid_name

        # Load the control parameters into a dictionary
        self.vc_dict = sol_vc_dict

    def retrieve_vc(self, time_stamp):
        """
        Patch-job method to give access to the solenoid VC. The `vc` object is
        assigned to the `vc` class attribute.

        Returns
        -------
        numpy 1D array
            The solenoid virtual circuit.

        """
        closest_key = max(
            (key for key in self.vc_dict if key <= time_stamp),
            default=None,
        )
        # print(closest_key)
        if closest_key is None:
            print(
                "time requested is before first target schedule time - return empty list"
            )

            return []

        sol_vc = self.vc_dict[closest_key]["vc"]

        return np.array(sol_vc)

    def get_vloop_blends(self, time_stamp):
        """
        Retrieves the vloop blends for the target at time_stamp, given the target schedule.

        """
        # get set of targets being controlled at this time
        print("--- loading  gains")
        gains = []
        # dict format is {time : {target : tau, target_2 : tau_2, ...}}
        # more likely this if single set of gains for all time.
        blend = self.get_blends(time_stamp=time_stamp, target="Vloop")
        return blend
