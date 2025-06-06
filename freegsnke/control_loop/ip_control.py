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
        dictionary containing target waveform.
    - schedule_dict : dict
        dictionary containing target schedule.
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

        # print(f"  The virtual circuit vector: {self.scheduler.vc_dict}")

        # The accumulated error in the plasma category. Defaults to 0 if not
        # given.
        self.integral = integral_term_0

    def calculate_solenoid_delta(self,
                                 Kp,
                                 Ki,
                                 dt,
                                 blend,
                                 Ip_req,
                                 Ip_obs,
                                 Vloop_ff,
                                 sol_vc):
        """
        Calculates the vector of current trajectories ΔI/Δt, as prescribed
        in the plasma category (and circuits category, supposedly) of the
        MAST-U PCS. The equations followed are:

        Ip_error = (Ip_req - Ip_obs)
        integral = Ip_error * dt
        Vloop_fb = Kp * Ip_error + Ki * (integral + 0.5 * Ip_error * dt)
        Vloop = Vloop_fb * blend + Vloop_ff * (1 - blend)/M_sp
        ΔIsol/Δt = -Vloop
        ΔI/Δt = ΔIsol/Δt * vc_vector

        Parameters
        ----------
        - inductances_pl : dict
            A dictionary with all the required inductacnes_pl.
        - Kp : float
            A proportional term used in the Vloop_fb computation.
        - Ki : float
            An integral term used in the Vloop_fb computation.
        - blend : float
            A value between 0 and 1 that acts as a weight in the Vloops sum.
        - Ip_req : float
            The plasma current requested.
        - Ip_obs : float
            The plasma current observed.
        - Vloop_ff : float
            The feedforward Vloop.

        Returns
        -------
        - dIsoldt : float
            dIsoldt stands for ΔIsol/Δt.

        """

        # Compute the required change for Ip
        delta_Ip = Ip_req - Ip_obs
        print(f"    The delta plasma current: {delta_Ip}")

        self.integral += delta_Ip * dt

        integral_term = self.integral + 0.5 * delta_Ip * dt
        Vloop_fb = Kp * delta_Ip + Ki * integral_term
        print(f"    The feedback loop voltage: {Vloop_fb}")

        # This suggests that Vloop_fb is multiplied by the inductance, as an
        # output (so it's actually a current), and that it's given in KA.
        return Vloop_fb*-1.58*1e-2

        # Compute the loop voltage as a weighted sum
        M_sp = 1.580*1e-5 * 1e3
        dIsoldt = blend * Vloop_fb - (1 - blend) * Vloop_ff * (1 / M_sp)

        # return dIsoldt

        # Compute the rate of change of the solenoid current
        print(f"    The trajectory for the solenoid current: {dIsoldt}")

        # Apply dIsoldt to virtual circuit vector to get the current
        # trajectories of the active coils
        dI_dt = dIsoldt * sol_vc
        print(f"    The trajectory for the solenoid current, after VC: {dI_dt[0]}")

        return dI_dt

    def ip_control(self, ts, prev_ts, Ip_obs=None, eq=None):
        """
        Execute all the steps in the pipeline for the control of the solenoid
        current, as prescribed by the PCS. This method is the API by design of
        the class; it takes a time_stamp and with the knowledge of the status
        of the plasma it computes the control voltage for the solenoid current.

        Parameters
        ----------
        - time_stamp : float (4 decimal places)
            Timestamp for which this pipeline should provide a control voltage.
        - Rp : float
            The plasma resistivity.
        - inductacnes_pl : dict
            A dictionary with all the required inductacnes_pl.
        - Ip_obs : float
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
        if Ip_obs is None:
            Ip_obs = self.scheduler.get_observed_current(ts, "Ip_obs", eq)
            print(f"  Ip from equilibrium: {Ip_obs}")
        else:
            print(f"User defined Ip_obs: {Ip_obs}")

        # print(f"  The observed Ip: {Ip_obs}")
        # Implement the plasma category. First, the relevant entities should be
        # retrieved from the scheduler
        Ip_req = self.scheduler.get_waveform_value(
            param_type="Ip",  param="plasma", time_stamp=ts
        )

        # TODO check if this is monitored this way
        if not Ip_req:
            print(f"  The plasma current is not controlled at t: {ts}")
            # return array of zeros if not controlled
            dI_dt = np.zeros_like(self.scheduler.vc_dict)
            return dI_dt
        print(f"  The requested Ip: {Ip_req}")

        Vloop_req = self.scheduler.get_waveform_value(
            param_type="ff",  param="plasma", time_stamp=ts
        )
        print(f"  The requested FF_Vloop: {Vloop_req}")

        gain_p, _ = self.scheduler.get_gains(
                ["plasma"], time_stamp=ts, K_type="Kprop")
        gain_int, _ = self.scheduler.get_gains(
                ["plasma"], time_stamp=ts, K_type="Kint")
        print(f"  The plasma gains: Kp={gain_p}, Kint={gain_int}")

        blend = self.scheduler.get_waveform_value(
            param_type="blends",  param="plasma", time_stamp=ts
        )
        print(f"  The blend value: {blend}")
        print("dt: ", (ts - prev_ts))

        dI_dt = self.calculate_solenoid_delta(
            Kp=gain_p[0],
            Ki=gain_int[0],
            dt=(ts - prev_ts),
            blend=blend,
            Ip_req=Ip_req,
            Ip_obs=Ip_obs,
            Vloop_ff=Vloop_req,
            sol_vc=self.scheduler.retrieve_vc(ts)
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

    # def retrieve_parameter(self, time_stamp, query):
    #     """
    #     Retrieves the value of the queried control parameter at time_stamp.

    #     Arguments
    #     ---------
    #     - time_stamp : float
    #         Time at which the gains are queried.
    #     - query : str
    #         Control parameter requested.

    #     Returns
    #     -------
    #     float
    #         Value of the requested parameter at time_stamp.

    #     """

    #     if query not in self.control_params:
    #         print(
    #             f"{query} is not present in control_params, returning None "
    #             "from retrieve_parameter()"
    #         )
    #         requested_parameter = None
    #     else:
    #         # Compute the position in the list for query that is the closest
    #         # lower time to time_stamp
    #         closest_pos = max(
    #             (key for key in self.control_params[query].keys() if key <= time_stamp),
    #             default=None,
    #         )
    #         if closest_pos is None:
    #             print(
    #                 "time requested is before first control parameter time, "
    #                 "returning None from retrieve_parameter()"
    #             )
    #             requested_parameter = None
    #         else:
    #             # Retrieved the value for this parameter at the chosen position
    #             requested_parameter = self.control_params[query][closest_pos]

    #     return requested_parameter

    # def get_observed_current(self, query, eq=None):
    #     """
    #     Provides the current value for `query` at `time_stamp`, either via
    #     user-defined sequences (if `query` is present in `control_params` on
    #     `time_stamp`) or via an estimation given by the Equilibrium `eq`.

    #     Parameters
    #     ----------
    #     # - time_stamp : float (4 decimal places)
    #     #     Timestamp for which this pipeline should provide a control voltage.
    #     - query : str
    #         Current queried. It can be either "Ip_obs" or "measured_Isol".

    #         "Ip_obs" refers to the required plasma current for this time_stamp.
    #         It defaults to the current value stored in the Equilibrium
    #         (argument eq) if not given.

    #         "measured_Isol" refers to the measured solenoid current from the
    #         tokamak. It defaults to the current value stored in the Equilibrium
    #         (argument eq) if not given.
    #     - eq : Equilibrium
    #         An equilibrium object from which we get information about the
    #         plasma or solenoid current when Ip_obs or measured_Isol are not
    #         given by the user.

    #     Returns
    #     -------
    #     float
    #         The current value for the queried entity.

    #     """
    #     # current = self.retrieve_parameter(time_stamp, query)

    #     if eq is None:
    #         raise Exception(
    #             "An Equilibrium object should be provided to "
    #             f"ip_control() if {query} is not provided."
    #         )

    #     if query == "Ip_obs":
    #         print(
    #             "Ip_obs is not provided, using the equilibrium given to " "estimate it."
    #         )
    #         current = eq.plasmaCurrent()

    #     if query == "measured_Isol":
    #         print(
    #             "measured_Isol is not provided, using the equilibrium "
    #             "given to estimate it."
    #         )
    #         current = eq.tokamak[self.solenoid_name].current

    #     return current

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
