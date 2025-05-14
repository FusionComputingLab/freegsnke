import numpy as np
import pickle
from pprint import pprint


class TargetScheduler:
    """
    Generic target scheduler, used for scheduling shape targets and plasma
    current.

    """

    def __init__(
        self,
        waveform_dict,
        schedule_dict,
    ):
        """
        Initialise the class

        Parameters
        ----------
        waveform_dict : dict
            dictionary containing target waveforms.
        schedule_dict : dict
            dictionary containing target schedule.

        Returns
        -------
        None

        """
        # load schedule and create a list of times for it
        self.target_schedule_dict = schedule_dict
        schedule_times = sorted(list(self.target_schedule_dict.keys()))

        print("schedule times", schedule_times)
        print("schedule dictionary")
        pprint(self.target_schedule_dict)

        print("waveform dictionary")
        pprint(waveform_dict)
        # load target waveform
        # target waveform  dict ~ {target_name : {"times":[time list],
        #                                         "vals": [vals list] , }

        self.target_waveform_dict = waveform_dict
        for quantity in self.target_waveform_dict.keys():
            print("timeseries waveform type", quantity)
            if "times" in self.target_waveform_dict[quantity].keys():
                item = self.target_waveform_dict[quantity]
                if len(item["times"]) != len(item["vals"]):
                    raise ValueError("times and vals arrays must be same length")
            else:
                print("further nested dict")
                for key, item in self.target_waveform_dict[quantity].items():
                    print(key, item)
                    if len(item["times"]) != len(item["vals"]):
                        print("Error in waveform dict", key)
                        raise ValueError("times and vals arrays must be same length")

        #### TODO MODIFY this check to be more robust
        # check compatibility of target schedule and target waveform
        # # checks that ...
        # for time in schedule_times:
        #     # ### do this with set check...
        #     targ_names = self.target_schedule_dict[time]
        #     for targ in targ_names:
        #         # check 1 : check if all targets in target schedule are in
        #         # target waveform
        #         if targ not in self.target_waveform_dict.keys():
        #             raise ValueError(
        #                 f"Target {targ} is in schedule but is not defined in"
        #                 "target waveform"
        #             )
        #         # check 2 : check if targets in each interval lie within time
        #         # ranges of target waveform
        #         if len(self.target_waveform_dict[targ]["times"]) > 1:
        #             time_start = self.target_waveform_dict[targ]["times"][0]
        #             time_end = self.target_waveform_dict[targ]["times"][-1]
        #             if time_start > time or time_end < time:
        #                 print(f"time range for {targ}: ({time_start}, {time_end})")
        #                 print(f"target schedule time: {time}")
        #                 raise ValueError(
        #                     f"Range of defined values for Target {targ} not "
        #                     "compatible with schedule"
        #                 )

    def interpolate(self, time_stamp, waveform):
        """
        Interpolate the target value at time_stamp, from the information in
        target_waveform_dict.

        Arguments
        ---------
        - time_stamp : float (4 decimal places)
            Time stamp of the target to be retrieved.
        - target : str
            Target queried.

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

    def retrieve_controlled_targets(self, time_stamp):
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
            (key for key in self.target_schedule_dict if key <= time_stamp),
            default=None,
        )
        # print(closest_key)
        if closest_key is None:
            print(
                "time requested is before first target schedule time - return empty list"
            )

            return []

        target_names = self.target_schedule_dict[closest_key]["targets"]

        return target_names

    def desired_target_values(self, time_stamp, controlled_targets=None):
        """
        Retrieve values for desired control targets as linear interpolation
        between values at two adjacent time stamps.

        Parameters
        ----------
        time_stamp : float (4 decimal places)
            Time stamp of the target to be retrieved.

        Returns
        -------
        targets_required : array (float)
            Requested target values to be used by the control classes.
        """
        # get set of targets being controlled at this time
        if controlled_targets is None:
            controlled_targets = self.retrieve_controlled_targets(time_stamp)

        # retrieve correct waveform dict - for shape or ip
        if "shape_fb" in self.target_waveform_dict.keys():
            waveform_dict = self.target_waveform_dict["shape_fb"]
        elif "coil_pert" in self.target_waveform_dict.keys():
            waveform_dict = self.target_waveform_dict["coil_pert"]
        else:
            waveform_dict = self.target_waveform_dict

        targets_required = np.array(
            [
                self.interpolate(time_stamp, waveform_dict[targ])
                for targ in controlled_targets
            ]
        )

        return targets_required

    def feed_forward_gradient(self, time_stamp, waveform_dict=None, targets=None):
        """
        Compute the feed forward gradient of the control voltages.

        Parameters
        ----------
        time_stamp : float
            time stamp of the target to be retrieved
        Returns
        -------
        gradient : np.array
        """
        if targets is None:
            targets = self.retrieve_controlled_targets(time_stamp)

        if waveform_dict is None:
            if "shape_ff" in self.target_waveform_dict.keys():
                waveform_dict = self.target_waveform_dict["shape_ff"]
            elif "coil_pert" in self.target_waveform_dict.keys():
                waveform_dict = self.target_waveform_dict["coil_pert"]

        grad_arr = np.zeros(len(targets))
        for i, target in enumerate(targets):
            slope = np.diff(waveform_dict[target]["vals"]) / np.diff(
                waveform_dict[target]["times"]
            )
            position = np.searchsorted(
                waveform_dict[target]["times"][1:], time_stamp, side="right"
            )
            # print(f"position index {position}")
            if time_stamp > self.target_waveform_dict[target]["times"][-1]:
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
    def retrieve_control_param(self, param_dict, param, time_stamp):
        """
        Retrieves the value of the queried control parameter at time_stamp.

        Assumes timeseries waveform format - use for blends, vloop, ip, shapes
        Arguments
        ---------
        - time_stamp : float (4 decimal places)
            Time stamp of the target to be retrieved.
        - param : str
            Control parameter requested.

        Returns
        -------
        requested_parameter : float
        """
        # print("retrieving time series control parameter", param)
        # print(param_dict[param]["vals"])
        if param not in param_dict.keys():
            print(
                f"{param} is not present in param_dict, returning None "
                "from retrieve_control_param()"
            )
            requested_parameter = None
        else:
            # find time position
            arr = param_dict[param]["times"]
            t_val = max(time for time in arr if time <= time_stamp)
            # print("param tval", t_val)
            # get index corresponding to the time position
            if isinstance(arr, list):
                pos = param_dict[param]["times"].index(t_val)
            elif isinstance(arr, np.ndarray):
                pos = np.where(arr == t_val)[0][0]

            if pos is None:
                print(
                    "time requested is before first control parameter time, "
                    "returning None from retrieve_control_parameter()"
                )
                requested_parameter = None
            else:
                requested_parameter = param_dict[param]["vals"][pos]
            # print(requested_parameter)

        return requested_parameter

    def get_gains(self, targets, time_stamp, type="Kprop"):
        """
        Retrieves the shape gains for the target at time_stamp, given the target schedule.
        # Gains provided as time_periods - assume units of milliseconds (ms)
        Gains provided as numbers
        Parameters
        ----------
        targets : list[str]
            list of targets to get gains for
        time_stamp : float (4 decimal places)
            time stamp of the target to be retrieved
        Returns
        -------
        shape_gains : np.array
            shape gains
        """
        # get set of targets being controlled at this time
        print("--- loading shape gains")
        gains = []
        # dict format is {time : {target : tau, target_2 : tau_2, ...}}
        # more likely this if single set of gains for all time.
        time_pos = max(
            time for time in self.target_schedule_dict.keys() if time <= time_stamp
        )
        if time_pos is None:
            print(
                "time requested is before first control parameter time, "
                "returning None from retrieve_parameter()"
            )
        else:
            for target in targets:
                gains.append(self.target_schedule_dict[time_pos]["gains"][target][type])
        gains_arr = np.array(gains)
        print("gains array ---- ", gains_arr)
        return gains_arr, np.diag(gains_arr)
