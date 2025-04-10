"""
Module to control plasma current Ip during a tokamak shot.

"""

from sys import float_info
from numpy.random import normal
from target_scheduler import TargetScheduler


class ControlSolenoid:
    """
    Class to control the solenoid voltage applied on the solenoid to steer the
    plasma current.

    Attributes
    ----------
    - scheduler : TargetScheduler
        An object that store information of the controlled target Ip and other
        control parameters: the plasma and solenoid gain, Vloop_ff, and blend.

    Methods
    -------
    - calculate_solenoid_delta: It calculates the feedback solenoid current
      from the observed and required plasma current.
    - compute_accumulated_currents: It accumulates all the feedback solenoid
      currents, effectively predicting the current value of the solenoid
      current.
    - check_currents: It checks whether the feedback and feedforward solenoid
      currents are within the prescribed limits.
    - calculate_solenoid_voltage: It corrects the error thrown by
      compute_accumulated_currents() and combines the feedback, feedforward and
      resistance voltages to provide the final output solenoid voltage.
    - ip_control: API method, it calls sequentially the other methods of the
      class, performing all the required control actions on the plasma current
      and the solenoid as a result.

    """

    def __init__(self, target_seq_path, target_sched_path, contr_params_path):
        """
        Initialises the ControlSolenoid class.

        Parameters
        ----------
        - target_waveform_path : str
            path to the file containing target sequence.
        - target_schedule_path : str
            path to the file containing target schedule.
        - contr_params_path : str
            path to the file containing control parameters sequence.

        Returns
        -------
        None

        """

        # Load the scheduler
        self.scheduler = SolenoidScheduler(
            target_seq_path, target_sched_path, contr_params_path
        )

    def calculate_solenoid_delta(self,
                                 inductances,
                                 gain,
                                 vc_vector,
                                 blend,
                                 Ip_req,
                                 Ip_obs,
                                 Vloop_ff):
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
        - inductances : dict
            A dictionary with all the required inductances.
        - gain : float
            A proportional term used in the Vloop_fb computation.
        - vc_vector : numpy array of size (12,1).
            Virtual Circuit 12-element vector for the solenoid current. First
            element is 1.
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
        - dIsol : float
            dIsol stands for ΔIsol/Δt.

        """

        # Compute the required change for Ip
        delta_Ip = Ip_req - Ip_obs
        print(f"    The delta plasma current: {delta_Ip}")

        # Compute the feedback loop voltage
        M_p = inductances["plasma"]
        Vloop_fb = gain * delta_Ip * M_p
        print(f"    The feedback loop voltage: {Vloop_fb}")

        # Compute the loop voltage as a weighted sum
        Vloop = blend * Vloop_fb + (1 - blend) * Vloop_ff
        print(f"    The full loop voltage: {Vloop}")

        # Compute the rate of change of the solenoid current
        M_sp = inductances["mutual"]
        dIsol = -Vloop * (1 / M_sp)

        # Apply dIsol to virtual circuit vector to get the current trajectories
        # of the active coils
        dI = dIsol * vc_vector

        return dI

    def compute_accumulated_currents(self, real_Isol):
        """
        Computes the predicted current for the solenoid so far in the shot, as
        prescribed in the VC subcategory of the circuit category. At the moment
        this method just mimics the real procedure.

        Parameters
        ----------
        - real_Isol : float
            This is just to get this going, it'll be changed.

        Returns
        -------
        - predicted_Isol : float
            The solenoid current, as predicted by the changes made by the PCS.

        """

        # Here I mimic the integral for dIsol's
        print(f"    The measured solenoid current: {real_Isol}")
        predicted_Isol = real_Isol + normal(0, 100)

        return predicted_Isol

    def check_currents(self, dIsol, Isol):
        """
        Check the values of the ΔIsol, Isol as prescribed in the system
        category of the PCS. At the moment we just check against python
        float limits.

        Parameters
        ----------
        - dIsol : float
            The change of rate estimated for the solenoid current.
        - Isol : float
            The absolute valued estimated for the solenoid current.

        Returns
        -------
        - approved_dIsol : float
            The change of rate estimated for the solenoid current, approved by
            the system category.
        - approved_Isol : float
            The absolute valued estimated for the solenoid current, approved by
            the system category.

        """

        # Check the rate of change
        if dIsol > float_info.max:
            approved_dIsol = float_info.max
        else:
            approved_dIsol = dIsol

        # Check the absolute value
        if Isol > float_info.max:
            approved_Isol = float_info.max
        else:
            approved_Isol = Isol

        return approved_dIsol, approved_Isol

    def calculate_solenoid_voltage(
        self,
        Rp,
        inductances,
        gain,
        approved_dIsol,
        approved_Isol,
        measured_Isol
    ):
        """
        Calculate the output voltage to apply on the solenoid, as prescribed in
        the PF category of the PCS. The equations followed are:

        Vsol_fb = gain_s * (approved_Isol - measured_Isol) * Msol_fb
        Vsol_ff = approved_dIsol * M_sol_ff
        Vsol_res = measured_Isol * Rp
        Vsol = Vsol_fb + Vsol_ff + Vsol_res

        Parameters
        ----------
        - Rp : float
            The plasma resistivity.
        - inductances : dict
            A dictionary with all the required inductances.
        - gain : float
            A proportional term used in the Vsol_fb computation.
        - approved_dIsol : float
            Approved change of rate of the solenoid current by the system
            category of the PCS.
        - approved_Isol : float
            Approved solenoid current by the system category of the PCS.
        - measured_Isol : float
            Actual solenoid current as registered by the Equilibrium object.

        Returns
        -------
        - Vsol : float
            Output control voltage of PCS to apply on the solenoid.

        """

        # Compute the feedback voltage
        Corrected_Isol = gain * (approved_Isol - measured_Isol)
        Msol_fb = inductances["sol_fb"]
        Vsol_fb = Corrected_Isol * Msol_fb
        print(f"    The feedback voltage: {Vsol_fb}")

        # Compute the feedforward voltage
        M_sol_ff = inductances["sol_ff"]
        Vsol_ff = approved_dIsol * M_sol_ff
        print(f"    The feedforward voltage: {Vsol_ff}")

        # Compute the resistive voltage
        Vsol_res = measured_Isol * Rp
        print(f"    The resistive voltage: {Vsol_res}")

        # Combine all the voltages into the output voltage
        Vsol = Vsol_fb + Vsol_ff + Vsol_res

        return Vsol

    def ip_control(self,
                   time_stamp,
                   Rp,
                   inductances,
                   Ip_obs=None,
                   measured_Isol=None,
                   eq=None,
                   solenoid_name=None):
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
        - inductances : dict
            A dictionary with all the required inductances.
        - Ip_obs : float
            Required plasma current for this time_stamp. It defaults to the
            current value stored in the Equilibrium (argument eq) if not given.
        - measured_Isol : float
            Measured solenoid current from the tokamak. It defaults to the
            current value stored in the Equilibrium (argument eq) if not given.
        - eq : Equilibrium
            An equilibrium object from which we get information about the
            plasma or solenoid current when Ip_obs or measured_Isol are not
            given.
        - solenoid_name : str
            A string to denote the solenoid current ("Solenoid", "P1", etc).
            Defaults to "Solenoid" if not given.

        Returns
        -------
        - Vsol : float
            Output control voltage of PCS to apply on the solenoid.

        """

        if Ip_obs is None or measured_Isol is None:
            if eq is None:
                raise Exception("An Equilibrium object should be provided to "
                                "ip_control() if Ip_req or measured_Isol are "
                                "not provided.")

            if Ip_obs is None:
                print("Ip_obs is not provided, using the equilibrium given to "
                      "estimate it.")
                Ip_obs = eq.plasmaCurrent()
                print(f"    The plasma current observed: {Ip_obs}")

            if measured_Isol is None:
                print("measured_Isol is not provided, using the equilibrium "
                      "given to estimate it.")

                if solenoid_name is None:
                    print("solenoid_name is not provided, using 'Solenoid' "
                          "as label for the solenoid current.")
                    solenoid_name = 'Solenoid'

                measured_Isol = eq.tokamak[solenoid_name].current
                print(f"  The measured solenoid current: {measured_Isol}")

        # Implement the plasma category. First, the relevant entities should be
        # retrieved from the scheduler
        controlled_targets = self.scheduler.desired_target_values(time_stamp)
        if not controlled_targets:
            print(f"  The plasma current is not controlled at t: {time_stamp}")
            return None
        Ip_req = controlled_targets[0]
        print(f"  The requested Ip: {Ip_req}")
        Vloop_req = self.scheduler.retrieve_parameter(time_stamp, "Vloop")
        print(f"  The requested Vloop: {Vloop_req}")
        gain_p = self.scheduler.retrieve_parameter(time_stamp, "Kp")
        print(f"  The plasma gain: {gain_p}")
        blend = self.scheduler.retrieve_parameter(time_stamp, "blend")
        print(f"  The blend value: {blend}")
        vc_vector = self.scheduler.retrieve_parameter(time_stamp, "vc")
        print(f"  The virtual circuit vector: {vc_vector}")
        dIsol = self.calculate_solenoid_delta(inductances=inductances,
                                              Ip_obs=Ip_obs,
                                              Ip_req=Ip_req,
                                              Vloop_ff=Vloop_req,
                                              gain=gain_p,
                                              blend=blend,
                                              vc_vector=vc_vector)
        print(f"  The delta solenoid current: {dIsol}")

        # Implement the estimation of the predicted solenoid current in the
        # circuit category
        Isol = self.compute_accumulated_currents(measured_Isol)
        print(f"  The estimated solenoid current: {Isol}")

        # Implement the system category
        approved_dIsol, approved_Isol = self.check_currents(dIsol, Isol)
        print(
            f"  The approved solenoid currents, (apr_dIsol, apr_Isol): "
            f"({approved_dIsol}, {approved_Isol})"
        )

        # Implement the PF category. First the relevant entities should be
        # retrieved
        gain_s = self.scheduler.retrieve_parameter(time_stamp, "Ks")
        print(f"  The solenoid gain: {gain_s}")
        Vsol = self.calculate_solenoid_voltage(
            Rp=Rp,
            inductances=inductances,
            gain=gain_s,
            approved_dIsol=approved_dIsol,
            approved_Isol=approved_Isol,
            measured_Isol=measured_Isol,
        )

        return Vsol


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

    def __init__(self,
                 target_waveform_path,
                 target_schedule_path,
                 control_params_path):
        """
        Initialise the Solenoid scheduler.

        Arguments
        ---------
        - target_waveform_path : str
            path to the file containing target sequence.
        - target_schedule_path : str
            path to the file containing target schedule.
        - control_params_path : str
            path to the file containing the control parameters sequence.

        Returns
        -------
        None

        """

        # Execute the parent __init__()
        super().__init__(target_waveform_path, target_schedule_path)

        # Load the control parameters into a dictionary
        self.control_params = self.load_pickle_dict(control_params_path)

    def retrieve_parameter(self, time_stamp, query):
        """
        Retrieves the value of the queried control parameter at time_stamp.

        Arguments
        ---------
        - time_stamp : float
            Time at which the gains are queried.
        - query : str
            Control parameter requested.

        Returns
        -------
        - requested_parameter : float
            Value of the requested parameter at time_stamp.

        """

        # Compute the position in the list for query that is the closest lower
        # time to time_stamp
        closest_pos = max(
            (key for key in self.control_params[query].keys()
             if key <= time_stamp),
            default=None,
        )
        if closest_pos is None:
            print("time requested is before first control parameter time")

        # Retrieved the value for this parameter at the chosen position
        requested_parameter = self.control_params[query][closest_pos]

        return requested_parameter
