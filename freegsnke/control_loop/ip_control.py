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
        contr_params_dict,
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
            waveform_dict, schedule_dict, contr_params_dict, solenoid_name
        )

        self.vc = self.scheduler.retrieve_vc()
        print(f"  The virtual circuit vector: {self.vc}")

    def calculate_solenoid_delta(
        self, inductacnes_pl, gain, blend, Ip_req, Ip_obs, Vloop_ff
    ):
        """
        Calculates the vector of current trajectories ΔI/Δt, as prescribed
        in the plasma category (and circuits category, supposedly) of the
        MAST-U PCS. The equations followed are:

        Vloop_fb = gain * (Ip_req - Ip_obs) * M_p
        Vloop = Vloop_fb * blend + Vloop_ff * (1 - blend)
        ΔIsol/Δt = -Vloop / M_sp
        ΔI/Δt = ΔIsol/Δt * vc_vector

        Parameters
        ----------
        - inductacnes_pl : dict
            A dictionary with all the required inductacnes_pl.
        - gain : float
            A proportional term used in the Vloop_fb computation.
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

        # Compute the feedback loop voltage
        M_p = inductacnes_pl["plasma"]
        Vloop_fb = gain * delta_Ip * M_p
        print(f"    The feedback loop voltage: {Vloop_fb}")

        # Compute the loop voltage as a weighted sum
        Vloop = blend * Vloop_fb + (1 - blend) * Vloop_ff
        print(f"    The full loop voltage: {Vloop}")

        # Compute the rate of change of the solenoid current
        M_sp = inductacnes_pl["mutual"]
        dIsoldt = -Vloop * (1 / M_sp)
        print(f"    The trajectory for the solenoid current: {dIsoldt}")

        # Apply dIsoldt to virtual circuit vector to get the current trajectories
        # of the active coils
        dI_dt = dIsoldt * self.vc
        return dI_dt

    def ip_control(self, time_stamp, Rp, inductacnes_pl, Ip_obs=None, eq=None):
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
            The observed plasma current. Defaults to None, in which case the observed plasma current is taken from the equilibrium/control params dictionary.
        - eq : equilibrium object
            optional equilibrium object

        Returns
        -------
        numpy 1D array
           Trajectories for the active coil currents due to the control on the
           solenoid coil.
        """
        if Ip_obs is None:
            Ip_obs = self.scheduler.get_observed_current(time_stamp, "Ip_obs", eq)
            print(f"  Ip from equilibrium: {Ip_obs}")
        else:
            print(f"User defined Ip_obs: {Ip_obs}")

        # print(f"  The observed Ip: {Ip_obs}")
        # Implement the plasma category. First, the relevant entities should be
        # retrieved from the scheduler
        controlled_targets = self.scheduler.desired_target_values(time_stamp)
        if not controlled_targets:
            print(f"  The plasma current is not controlled at t: {time_stamp}")
            # return None
            dI_dt = np.zeros_like(self.vc)  # return array of zeros if not controlled
            return dI_dt
        Ip_req = controlled_targets[0]
        control_params = self.scheduler.control_params
        print(f"  The requested Ip: {Ip_req}")
        Vloop_req = self.scheduler.retrieve_control_param(
            param_dict=control_params, time_stamp=time_stamp, param="Vloop"
        )
        print(f"  The requested Vloop: {Vloop_req}")
        gain_p = self.scheduler.retrieve_control_param(
            param_dict=control_params, time_stamp=time_stamp, param="Kp"
        )
        print(f"  The plasma gain: {gain_p}")
        blend = self.scheduler.retrieve_control_param(
            param_dict=control_params, time_stamp=time_stamp, param="blend"
        )
        print(f"  The blend value: {blend}")
        dI_dt = self.calculate_solenoid_delta(
            inductacnes_pl=inductacnes_pl,
            Ip_obs=Ip_obs,
            Ip_req=Ip_req,
            Vloop_ff=Vloop_req,
            gain=gain_p,
            blend=blend,
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

    def get_observed_current(self, query, eq=None):
        """
        Provides the current value for `query` at `time_stamp`, either via
        user-defined sequences (if `query` is present in `control_params` on
        `time_stamp`) or via an estimation given by the Equilibrium `eq`.

        Parameters
        ----------
        # - time_stamp : float (4 decimal places)
        #     Timestamp for which this pipeline should provide a control voltage.
        - query : str
            Current queried. It can be either "Ip_obs" or "measured_Isol".

            "Ip_obs" refers to the required plasma current for this time_stamp.
            It defaults to the current value stored in the Equilibrium
            (argument eq) if not given.

            "measured_Isol" refers to the measured solenoid current from the
            tokamak. It defaults to the current value stored in the Equilibrium
            (argument eq) if not given.
        - eq : Equilibrium
            An equilibrium object from which we get information about the
            plasma or solenoid current when Ip_obs or measured_Isol are not
            given by the user.

        Returns
        -------
        float
            The current value for the queried entity.

        """
        # current = self.retrieve_parameter(time_stamp, query)

        if eq is None:
            raise Exception(
                "An Equilibrium object should be provided to "
                f"ip_control() if {query} is not provided."
            )

        if query == "Ip_obs":
            print(
                "Ip_obs is not provided, using the equilibrium given to " "estimate it."
            )
            current = eq.plasmaCurrent()

        if query == "measured_Isol":
            print(
                "measured_Isol is not provided, using the equilibrium "
                "given to estimate it."
            )
            current = eq.tokamak[self.solenoid_name].current

        return current

    def get_vloop_blends(self, time_stamp):
        """
        Retrieves the vloop blends for the target at time_stamp, given the target schedule.

        """
        # get set of targets being controlled at this time
        print("--- loading shape gains")
        gains = []
        # dict format is {time : {target : tau, target_2 : tau_2, ...}}
        # more likely this if single set of gains for all time.
        blend = self.retrieve_control_param(
            param_dict=self.target_waveform_dict,
            param="blends",
            time_stamp=time_stamp,
        )
        return blend
