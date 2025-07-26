import pickle
from copy import deepcopy
from pprint import pprint

import numpy as np


class TargetScheduler:
    """
    Generic Scheduling class used for control scheduling in all control categories.
    This provides a description, and associated methods for retrieval, of the various control parameters (described below).
    Quantities that 'vary with time' are provided as waveforms - dictionary of time series' (arrays for times and corresponding values)
    Quantities that are fixed throughout a certain 'phase' are provided as a 'schedule' - a dictionary with phase start times as keys and the
    parameter values as corresponding items.


    Naming conventions :
    - ff : FeedForward. Waveforms to be used for feedforward control
    - fb : FeedBack - waveforms to be used for P(I) feedback control.
    - blends  - values in range (0,1) to control mixing ratio of ff and fb.
              - blend = 1 corresponds to fb only, and blend = 0 corresponds to ff only.
    """

    def __init__(
        self,
        waveform_dict: dict,
        schedule_dict: dict,
        controlled_targets_all: list,
    ):
        """
        Initialise the class

        Parameters
        ----------
        waveform_dict : dict
            dictionary containing target waveforms.
        schedule_dict : dict
            dictionary containing target schedule.
        controlled_targets_all : list
            list of all targets that will be available for control, either FB or FF.
            If controlling shape and divertor separately, this will need to include all shape and divertor targets in one list.

        Returns
        -------
        None

        """
        # load schedule and create a list of times for it
        self.schedule_dict_raw = schedule_dict
        self.schedule_times = sorted(list(self.schedule_dict_raw.keys()))
        self.control_targs_all = controlled_targets_all

        print("schedule times", self.schedule_times)

        self.ff_waves = self.convert_waveform_dictionary(waveform_dict["ff"])
        self.fb_waves = self.convert_waveform_dictionary(waveform_dict["fb"])
        self.blends = self.convert_waveform_dictionary(waveform_dict["blends"])

        # Compatiblitly check - MUST provide all waveforms and gains for all targets.
        # check targets in schedule targets and waveform targets are the same set of targets.
        ff_targets = list(self.ff_waves.keys())
        fb_targets = list(self.fb_waves.keys())
        target_blends = list(self.blends.keys())
        # Print warning or raise error ??
        if not set(ff_targets).issubset(set(controlled_targets_all)):
            print(
                "Warning : there are feedforward waveforms missing. These will be assumed to be zero "
            )
            print(ff_targets)
            print(self.control_targs_all)
            # raise ValueError(
            #     "Missing feedforward waveforms \n "
            #     f"ff waveform targets {ff_targets} \n All targets {controlled_targets_all}"
            # )
        elif not set(fb_targets).issubset(set(controlled_targets_all)):
            print(
                "Warning : there are feedback waveforms missing. These will be assumed to be zero "
            )
            print(ff_targets)
            print(self.control_targs_all)
            # raise ValueError(
            #     "Missing feedback waveforms\n "
            #     f"ff waveform targets {fb_targets} \n All targets {controlled_targets_all}"
            # )
        if not set(target_blends).issubset(set(controlled_targets_all)):
            print(
                "Warning : there are blend waveforms missing. These will be assumed to be zero "
            )
            print(ff_targets)
            print(self.control_targs_all)
            # raise ValueError(
            #     "Missing blend waveforms\n "
            #     f"ff waveform targets {target_blends} \n All targets {controlled_targets_all}"
            # )

        print("Input schedule")
        pprint(self.schedule_dict_raw)
        # Build gain vectors now
        Kprop_arr_schedule = {}
        damping_schedule = {}
        Kint_arr_schedule = {}
        for time in self.schedule_times:
            Kprop_arr = np.zeros(len(self.control_targs_all))
            Kint_arr = np.zeros(len(self.control_targs_all))

            for i, targ in enumerate(self.control_targs_all):
                if targ in self.schedule_dict_raw[time]["targets"]:
                    Kprop_arr[i] = self.schedule_dict_raw[time]["gains"][targ]["Kprop"]
                    Kint_arr[i] = self.schedule_dict_raw[time]["gains"][targ]["Kint"]
            Kprop_arr_schedule[time] = Kprop_arr
            Kint_arr_schedule[time] = Kint_arr
            if "Damping Factor" in self.schedule_dict_raw[time].keys():
                damping_schedule[time] = self.schedule_dict_raw
            else:
                damping_schedule[time] = 1  # damping is 1 if not present

        self.schedule = {
            "Kprop": Kprop_arr_schedule,
            "Kint": Kint_arr_schedule,
            "damping": damping_schedule,
        }
        print(f"Scheduled quantities for {self.control_targs_all}")
        print("proportional gains")
        pprint(Kprop_arr_schedule)
        print("Integral gains")
        pprint(Kint_arr_schedule)
        print("Damping ")
        pprint(damping_schedule)

    def get_all_targets(self):
        """return list of all controllable targets"""
        return self.control_targs_all

    def convert_units_single_waveform(self, waveform: dict):
        """convert units of any single waveform into standard units : (A, m, s)
        This could be of use when reading in measured data.

        Parameters
        ----------
        waveform : dict
            waveform dictionary:  {times : [], vals : [], untis : ""}

        Return
        ------
        waveform_new : dict
            updated waveform
        """
        waveform_new = deepcopy(waveform)
        try:
            unit = waveform["units"]
            # convert units : return everything in
            if unit == "kA":  # convert kA to A
                waveform_new["vals"] *= 1000
                waveform_new["units"] = "A"
            elif unit == "ms":  # convert milliseconds to seconds
                waveform_new["vals"] /= 1000
                waveform_new["units"] = "s"
            elif unit == "cm":  # convert cm to m
                waveform_new["vals"] /= 100
                waveform_new["units"] = "m"
            elif unit == "mm":  # mm to m
                waveform_new["vals"] /= 1000
                waveform_new["units"] = "m"
            # print(f"units converted from {unit} to standard (A, m, s)")
        except KeyError:
            # print("Warning - waveform doesn't have units key ")
            pass

        return waveform_new

    def convert_waveform_dictionary(self, waveform_dict: dict):
        """convert units of any waveform into standard units : (A, m, s)
        This will convert a set of waveforms such as all ff waveforms in one go

        Parameters
        ----------
        waveform : dict
            waveform dictionary: {quantity :  {times : [], vals : [], untis : ""},...}

        Return
        ------
        waveform_new : dict
            updated dictionary of waveforms
        """
        # Convert units
        waveform_dict_new = deepcopy(waveform_dict)
        for key, waveform in waveform_dict_new.items():
            waveform = self.convert_units_single_waveform(waveform=waveform)

        print("Waveforms converted to standard units")
        return waveform_dict_new

    def get_schedule_time(self, time_stamp):
        """
        Get phase start time for the phase in which time_stamp occurs.

        Parameters
        ----------
        time_stamp : float
            time of interest

        Returns
        -------
        time_sch : float
            start time in schedule of phase containing timestamp
        """

        time_pos = max(time for time in self.schedule_times if time <= time_stamp)

        return time_pos

    def get_scheduled_params(
        self,
        param_type: str,
        time_stamp: float,
    ):
        """Retrieve pre built quantity from schedule - gains array or damping

        Parameters
        ----------
        param_type : str
            type of parameter of interest. Options are "Kprop", "K_int" and "damping"
        time_stamp : float
            time stamp at which the param type is required

        Returns
        -------
        param_values : np.ndarray | float
            numpy array of gains or float of damping factor.

        """
        time_pos = self.get_schedule_time(time_stamp=time_stamp)
        param_sch = self.schedule[param_type]

        return param_sch[time_pos]

    def interpolate(
        self,
        time_stamp: float,
        waveform: dict,
    ):
        """
        Interpolate the target value at time_stamp, from the information in
        target_waveform_dict.

        Arguments
        ---------
        - time_stamp : float (4 decimal places)
            Time stamp of the target to be retrieved.
        - waveform : dict
            waveform dictionary for target of interest.

        Returns
        -------
        - interpolation : float
            The interpolated value of target at time_stamp.

        """
        interpolation = np.interp(
            time_stamp,
            waveform["times"],
            waveform["vals"],
        )

        return interpolation

    def get_fb_controlled_targets(
        self,
        time_stamp: float,
    ):
        """
        Find the targets that are controlled at time time_stamp.

        Arguments
        ---------
        time_stamp : float (4 decimal places)
            Time stamp of the target to be retrieved.

        Returns
        -------
        target_names : list[str]
            List of target names.

        """

        closest_key = max(
            (key for key in self.schedule_dict_raw if key <= time_stamp),
            default=None,
        )
        # print(closest_key)
        if closest_key is None:
            print(
                "time requested is before first target schedule time - return empty list"
            )

            return []

        target_names = self.schedule_dict_raw[closest_key]["targets"]

        return target_names

    def desired_target_values_fb(
        self,
        time_stamp: float,
        controlled_targets: list[str] = None,
        interpolate: bool = False,
    ):
        """
        Retrieve values for desired control targets as linear interpolation
        between values at two adjacent time stamps.

        Parameters
        ----------
        time_stamp : float (4 decimal places)
            Time stamp of the target to be retrieved.
        controlled_targets : list[str] (optional)
            list of targets that wish to be controlled (subset of all targets).
            If None, will be set to fb_controlled_targets
        interpolate : bool (optional)
            True/False as to whether to return interpolation between two points, or the point at closest previous time.
            Defaults to False

        Returns
        -------
        targets_required : array (float)
            Requested target values to be used by the control classes.
        """
        # get set of targets being controlled at this time
        if controlled_targets is None:
            controlled_targets = self.get_fb_controlled_targets(time_stamp)

        targets_required = np.zeros(len(self.control_targs_all))

        if interpolate == True:
            waveform_dict = self.fb_waves
            for i, targ in enumerate(self.control_targs_all):
                if targ in controlled_targets:
                    targets_required[i] = self.interpolate(
                        time_stamp, waveform_dict[targ]
                    )

        else:
            for i, targ in enumerate(self.control_targs_all):
                targ_val = self.get_waveform_value("fb", targ, time_stamp)
                if targ_val is not None:
                    targets_required[i] = targ_val
                else:
                    print(f"No fb value for target {targ} at time {time_stamp}")

        return targets_required

    def get_blends(
        self,
        time_stamp: float,
        targets: list[str] = None,
    ):
        """get blend values for targets at time_stamp

        Parameters
        ----------
        time_stamp : float (4 decimal places)
            time stamp of the target to be retrieved
        targets : list[str]
            list of targets to get gains for

        Returns
        -------
            blends : np.array
                blends for the targets specified
        """
        if targets is None:
            targets = self.control_targs_all

        blends = np.zeros(len(self.control_targs_all))
        for i, target in enumerate(self.control_targs_all):
            try:
                blends[i] = self.get_waveform_value(
                    param_type="blends", param=target, time_stamp=time_stamp
                )
            except:
                print(f"warning - No blend for {target} : setting to zero")

        return blends

    def feed_forward_gradient(
        self,
        time_stamp: float,
        targets: list[str] = None,
    ):
        """
        Compute the feed forward gradient of the control voltages.
        uses np.diff to compute gradient

        Parameters
        ----------
        time_stamp : float
            time stamp of the target to be retrieved
        targets  : list[str] (optional)
            list of targets to compute gradients of. Defaults to all if none provided.
        Returns
        -------
        gradient : np.array
        """
        if targets is None:
            targets = self.control_targs_all
        waveform_dict = self.ff_waves

        # initialise array of zeros for all control targets
        grad_arr = np.zeros(len(self.control_targs_all))
        for i, target in enumerate(self.control_targs_all):
            if target in targets:
                slope = np.diff(waveform_dict[target]["vals"]) / np.diff(
                    waveform_dict[target]["times"]
                )
                position = np.searchsorted(
                    waveform_dict[target]["times"][1:], time_stamp, side="right"
                )
                # print(f"position index {position}")
                if time_stamp > waveform_dict[target]["times"][-1]:
                    print(
                        "time_stamp is greater than the last waveform time stamp "
                        f"for target {target}"
                    )
                    gradient = 0
                else:
                    gradient = slope[position]
                    # print(f"slope at {time_stamp}: {slope[position]}")

                grad_arr[i] = gradient

        return grad_arr

    # def retrieve_timeseries_param(self, param, time_stamp):
    def get_waveform_value(
        self,
        param_type: str,
        param: str,
        time_stamp: float,
    ):
        """
        Retrieves the value of the queried control parameter at time_stamp.

        Assumes timeseries waveform format - use for blends, vloop, ip, shapes

        Parameters
        ----------
        - param_type : str
            type of parameter of interest. Choose "ff", "fb" or "blends" or "Ip".
        - param : str
            Control parameter requested - name of target. for which the above type is required.
        - time_stamp : float (4 decimal places)
            Time stamp of the target to be retrieved.

        Returns
        -------
        requested_parameter : float
            value of that parameter
        """

        if param_type == "ff":
            waveform_dict = self.ff_waves
        elif param_type == "fb":
            waveform_dict = self.fb_waves
        elif param_type == "blends":
            waveform_dict = self.blends
        elif param_type == "Ip":
            waveform_dict = self.ips

        if param not in waveform_dict.keys():
            print(
                f"{param} is not present in waveform_dict, returning None "
                "from retrieve_control_param()"
            )
            requested_parameter = None
        else:
            # find time position
            arr = waveform_dict[param]["times"]
            # print(f"waveform for target {param} at time {time_stamp} : {arr}")
            eps = 1e-8
            t_vals_temp = [time for time in arr if time <= time_stamp + eps]
            if len(t_vals_temp) == 0:
                raise ValueError(
                    f"No time values in array less than or equal to {time_stamp}"
                )
            t_val = max(t_vals_temp)
            # get index corresponding to the time position
            if isinstance(arr, list):
                pos = waveform_dict[param]["times"].index(t_val)
            elif isinstance(arr, np.ndarray):
                pos = np.where(arr == t_val)[0][0]

            if pos is None:
                print(
                    "time requested is before first control parameter time, "
                    "returning None from retrieve_control_parameter()"
                )
                requested_parameter = None
            else:
                requested_parameter = waveform_dict[param]["vals"][pos]
                prev_value = waveform_dict[param]["vals"][pos - 1]
            # print(requested_parameter)

        # return requested_parameter, prev_value
        return requested_parameter

    # def get_gains(
    #     self,
    #     time_stamp: float,
    #     targets: list[str] = None,
    #     K_type: str = "Kprop",
    # ):
    #     """
    #     Retrieves the shape gains for the target at time_stamp, given the target schedule.
    #     # Gains provided as time_periods - assume units of milliseconds (ms)
    #     Gains provided as numbers with units 1/time (Not provided as tau in s or ms)
    #     Parameters
    #     ----------
    #     time_stamp : float (4 decimal places)
    #         time stamp of the target to be retrieved
    #     targets : list[str] (optional)
    #         list of targets to get gains for. defaults to fb controlled targets if None provided
    #     Returns
    #     -------
    #     shape_gains : np.array
    #         shape gains
    #     """
    #     gains_arr = np.zeros(len(self.control_targs_all))
    #     time_pos = max(time for time in self.schedule_times if time <= time_stamp)
    #     if targets is None:
    #         targets = self.get_fb_controlled_targets(time_stamp=time_stamp)
    #     if time_pos is None:
    #         print(
    #             "time requested is before first control parameter time, "
    #             "returning None f"
    #         )
    #     else:
    #         for i, target in enumerate(self.control_targs_all):
    #             if target in targets:
    #                 gains_arr[i] = self.schedule_dict_raw[time_pos]["gains"][
    #                     target
    #                 ][K_type]

    #     return gains_arr

    def get_gains(self, time_stamp, K_type="Kprop"):
        """
        Retrieves the shape gains for the target at time_stamp, given the target schedule.
        Gains provided as numbers with units 1/time (Not provided as tau in s or ms)

        Parameters
        ----------
        time_stamp : float
            time stamp at which gains are required

        Returns
        -------
        gains_arr : np.ndarray
            numpy array of gains for all controlled targets
        """
        return self.get_scheduled_params(time_stamp=time_stamp, param_type=K_type)

    def get_damping(
        self,
        time_stamp: float,
    ):
        """
        Get damping factor (if present)
        Returns 1 if none provided, corresponding to no damping being applied.

        Parameters
        ----------
        time_stamp : float
            time of retrieval

        Returns
        -------
        damp_factor : float
            damping factor for the current  P(ID) phase
        """

        return self.get_scheduled_params(time_stamp=time_stamp, param_type="damping")

    # def get_damping(
    #     self,
    #     time_stamp: float,
    # ):
    #     """
    #     Get damping factor (if present)
    #     Returns 1 if none provided, corresponding to no damping being applied.

    #     Parameters
    #     ----------
    #     time_stamp : float
    #         time of retrieval

    #     Returns
    #     -------
    #     damp_factor : float
    #         damping factor for the current  P(ID) phase
    #     """
    #     time_pos = max(time for time in self.schedule_times if time <= time_stamp)
    #     gain_dict = self.schedule_dict_raw[time_pos]["gains"]
    #     if "Damping Factor" in gain_dict.keys():
    #         # print("damping", gain_dict["Damping Factor"])
    #         return gain_dict["Damping Factor"]
    #     else:
    #         print("there is no damping factor : return 1")
    #         # No damping corresponds to damp_factor = 1
    #         return 1.0
