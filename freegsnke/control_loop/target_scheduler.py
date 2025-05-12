import numpy as np
import pickle


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
            dictionary containing target waveform.
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
        print("schedule dictionary", self.target_schedule_dict)

        # load target waveform
        # target waveform  dict ~ {target_name : {"times":[time list],
        #                                         "vals": [vals list] , }

        self.target_waveform_dict = waveform_dict
        for key, item in self.target_waveform_dict.items():
            if len(item["times"]) != len(item["vals"]):
                raise ValueError("times and vals arrays must be same length")

        # check compatibility of target schedule and target waveform
        # checks that ...
        for time in schedule_times:
            # ### do this with set check...
            targ_names = self.target_schedule_dict[time]
            for targ in targ_names:
                # check 1 : check if all targets in target schedule are in
                # target waveform
                if targ not in self.target_waveform_dict.keys():
                    raise ValueError(
                        f"Target {targ} is in schedule but is not defined in"
                        "target waveform"
                    )
                # check 2 : check if targets in each interval lie within time
                # ranges of target waveform
                if len(self.target_waveform_dict[targ]["times"]) > 1:
                    time_start = self.target_waveform_dict[targ]["times"][0]
                    time_end = self.target_waveform_dict[targ]["times"][-1]
                    if time_start > time or time_end < time:
                        print(f"time range for {targ}: ({time_start}, {time_end})")
                        print(f"target schedule time: {time}")
                        raise ValueError(
                            f"Range of defined values for Target {targ} not "
                            "compatible with schedule"
                        )

    def load_pickle_dict(self, path):
        """
        Load the dictionary from the file.

        Parameters
        ----------
        - path : str
            dictionary containing target schedule.

        Returns
        -------
        - pickle_dict : dictionary
            Dictionary with the file contents.

        """
        file_ext = (path).split(".")[-1]
        if file_ext == ("pkl" or "pickle"):
            print(f"loading {path}")
            # load target waveform from pickle file
            with open(path, "rb") as fp:
                pickle_dict = pickle.load(fp)
        return pickle_dict

    def interpolate(self, time_stamp, target):
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
            self.target_waveform_dict[target]["times"],
            self.target_waveform_dict[target]["vals"],
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

        if closest_key is None:
            print("time requested is before first target schedule time")

        target_names = self.target_schedule_dict[closest_key]

        return target_names

    def desired_target_values(self, time_stamp):
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
        controlled_targets = self.retrieve_controlled_targets(time_stamp)

        targets_required = np.array(
            [self.interpolate(time_stamp, targ) for targ in controlled_targets]
        )

        return targets_required

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
            targets = self.retrieve_controlled_targets(time_stamp)

        grad_arr = np.zeros(len(targets))
        for i, target in enumerate(targets):
            slope = np.diff(self.target_waveform_dict[target]["vals"]) / np.diff(
                self.target_waveform_dict[target]["times"]
            )
            position = np.searchsorted(
                self.target_waveform_dict[target]["times"][1:], time_stamp, side="right"
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

    def retrieve_control_param(self, param_dict, param, time_stamp):
        """
        Retrieves the value of the queried control parameter at time_stamp.

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
        print("retrieving control parameter", param)
        if param not in param_dict.keys():
            print(
                f"{param} is not present in param_dict, returning None "
                "from retrieve_control_parameter()"
            )
            requested_parameter = None
        else:
            # find time position
            arr = param_dict[param]["times"]
            t_val = max(time for time in arr if time <= time_stamp)
            print("param tval", t_val)
            # get index corresponding to the time position
            if isinstance(arr, list):
                pos = param_dict[param]["times"].index(t_val)
            elif isinstance(arr, np.ndarray):
                pos = np.where(arr == t_val)[0][0]

            print(f"pos {pos}")
            if pos is None:
                print(
                    "time requested is before first control parameter time, "
                    "returning None from retrieve_control_parameter()"
                )
                requested_parameter = None
            else:
                requested_parameter = param_dict[param]["vals"][pos]

        return requested_parameter
