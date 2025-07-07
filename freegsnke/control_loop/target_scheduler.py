import pickle
from pprint import pprint

import numpy as np


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
        # print("schedule dictionary")
        # pprint(self.target_schedule_dict)

        # print("waveform dictionary")
        # pprint(waveform_dict)
        # load target waveform
        # target waveform  dict ~ {target_name : {"times":[time list],
        #                                         "vals": [vals list] , }

        self.ff_waves = waveform_dict["ff"]
        self.fb_waves = waveform_dict["fb"]
        self.blends = waveform_dict["blends"]

        # check targets in schedule targets and waveform targets are the same set of targets.
        ff_targets = list(self.ff_waves.keys())
        fb_targets = list(self.fb_waves.keys())
        blend_targets = list(self.blends.keys())

        # check targets in schedule targets and waveform targets are the same set of targets.
        # for t, val in self.target_schedule_dict.items():
        #     targs_in_sched = val["targets"]
        #     # check if subset of targets in schedule are in targets in waveform
        #     if not set(targs_in_sched).issubset(set(ff_targets)):
        #         print(
        #             f"Scheduling error at time {t} - missing ff waves for targets requested"
        #         )
        #         raise ValueError(
        #             "Targets in schedule not a subset of targets in ff waveform"
        #         )
        #     elif not set(targs_in_sched).issubset(set(fb_targets)):
        #         print(
        #             f"Scheduling error at time {t} - missing fb waves for targets requested"
        #         )
        #         raise ValueError(
        #             "Targets in schedule not a subset of targets in fb waveform"
        #         )
        #     elif not set(targs_in_sched).issubset(set(blend_targets)):
        #         print(
        #             f"Scheduling error at time {t} - missing blends for targets requested"
        #         )
        #         raise ValueError(
        #             "Targets in schedule not a subset of targets in blend waveform"
        #         )

        #  OR popualte ff waves etc with targets from schedule

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

    def get_fb_controlled_targets(self, time_stamp):
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

    def desired_target_values_fb(
        self, time_stamp, controlled_targets=None, interpolate=False
    ):
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
            controlled_targets = self.get_fb_controlled_targets(time_stamp)

        if interpolate == True:
            waveform_dict = self.fb_waves
            targets_required = np.array(
                [
                    self.interpolate(time_stamp, waveform_dict[targ])
                    for targ in controlled_targets
                ]
            )
        else:
            targets_required = np.zeros(len(controlled_targets))
            for i, targ in enumerate(controlled_targets):
                targ_val = self.get_waveform_value("fb", targ, time_stamp)
                if targ_val is not None:
                    targets_required[i] = targ_val
                else:
                    print(f"No fb value for target {targ} at time {time_stamp}")
            # targets_required = np.array(
            #     [
            #         self.get_waveform_value("fb", targ, time_stamp)
            #         for targ in controlled_targets
            #     ]
            # )

        return targets_required

    def get_blends(self, targets, time_stamp):
        """get blend values for targets at time_stamp

        parameters
        targets : list[str]
            list of targets to get gains for
        time_stamp : float (4 decimal places)
            time stamp of the target to be retrieved

        returns
            blends : np.array
                blends for the targets specified
        """

        blends = [
            self.get_waveform_value(
                param_type="blends", param=target, time_stamp=time_stamp
            )
            for target in targets
        ]

        return np.array(blends)

    def feed_forward_gradient(self, time_stamp, targets=None):
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
            targets = self.get_fb_controlled_targets(time_stamp)

        waveform_dict = self.ff_waves
        #
        # V1 - asusmes ff waveform has units of target, not dTarget/dt
        grad_arr = np.zeros(len(targets))
        for i, target in enumerate(targets):
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

        # # VERSION 2 - Assume waveform is for the gradient itself dTarget/dt
        # ff_vals = np.array(
        #     [
        #         self.get_waveform_value(
        #             param_type="ff", time_stamp=time_stamp, param=targ
        #         )
        #         for targ in targets
        #     ]
        # )
        # return ff_vals

    # def retrieve_timeseries_param(self, param, time_stamp):
    def get_waveform_value(self, param_type, param, time_stamp):
        """
        Retrieves the value of the queried control parameter at time_stamp.

        Assumes timeseries waveform format - use for blends, vloop, ip, shapes

        Parameters
        ----------
        - time_stamp : float (4 decimal places)
            Time stamp of the target to be retrieved.
        - param : str
            Control parameter requested.

        Returns
        -------
        requested_parameter : float
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
            print(f"waveform for target {param} at time {time_stamp} : {arr}")
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
            # print(requested_parameter)

            # Convert units
            try:
                unit = waveform_dict[param]["units"]
                # convert units : return everything in
                if unit == "kA":  # convert kA to A
                    requested_parameter *= 1000
                elif unit == "ms":  # convert milliseconds to seconds
                    requested_parameter /= 1000
                elif unit == "cm":  # convert cm to m
                    requested_parameter /= 100
                elif unit == "mm":  # mm to m
                    requested_parameter /= 1000
                print("units converted from {unit} to standard (A, m, s)")
            except KeyError:
                print("Warning - waveform doesn't have units key ")

        return requested_parameter

    def get_gains(self, targets, time_stamp, K_type="Kprop"):
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
        if targets[0] == "plasma":
            print("--- loading plasma gains")
        else:
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
                "returning None f"
            )
        else:
            print("gains at time ", time_stamp)
            for target in targets:
                print("target ", target)
                blend = self.get_blends(time_stamp=time_stamp, targets=[target])
                print("blend ", blend)
                if blend == 0.0:
                    gains.append(0)
                    print("blend is zero - FF only so set gain to zero")
                else:
                    gains.append(
                        self.target_schedule_dict[time_pos]["gains"][target][K_type]
                    )
        gains_arr = np.array(gains)
        # print("gains array ---- ", gains_arr)
        return gains_arr, np.diag(gains_arr)

    def get_damping(self, time_stamp):
        """get damping factor (if present)

        Parameters
        ----------
        time_stamp : float
            time of retrieval

        Returns
        damp_factor : float
            damping factor for the current phase
        """
        time_pos = max(
            time for time in self.target_schedule_dict.keys() if time <= time_stamp
        )
        gain_dict = self.target_schedule_dict[time_pos]
        if "Damping Factor" in gain_dict.keys():
            return gain_dict["Damping Factor"]
        else:
            print("there is no damping factor : return 1")
            # No damping corresponds to damp_factor = 1
            return 1.0
