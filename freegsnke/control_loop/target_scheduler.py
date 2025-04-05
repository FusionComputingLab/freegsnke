import numpy as np
import pickle


class TargetScheduler():
    """
    Generic target scheduler, used for scheduling shape targets and plasma
    current.

    """

    def __init__(
        self,
        target_sequence_path,
        target_schedule_path,
    ):
        """
        Initialise the class

        Parameters
        ----------
        target_sequence_path : str
            path to the file containing target sequence.
        target_schedule_path : str
            path to the file containing target schedule.

        Returns
        -------
        None

        """
        # load schedule and create a list of times for it
        self.target_schedule_dict = self.load_pickle_dict(target_schedule_path)
        schedule_times = sorted(list(self.target_schedule_dict.keys()))

        print("target schedule times", schedule_times)
        print("target schedule dict", self.target_schedule_dict)

        # load target sequence
        # target sequence  dict ~ {target_name : {"times":[time list],
        #                                         "vals": [vals list] , }

        self.target_sequence_dict = self.load_pickle_dict(target_sequence_path)
        for key, item in self.target_sequence_dict.items():
            if len(item["times"]) != len(item["vals"]):
                raise ValueError("times and vals arrays must be same length")

        # check compatibility of target schedule and target sequence
        # checks that ...
        for time in schedule_times:
            # ### do this with set check...
            targ_names = self.target_schedule_dict[time]
            for targ in targ_names:
                # check 1 : check if all targets in target schedule are in
                # target sequence
                if targ not in self.target_sequence_dict.keys():
                    raise ValueError(
                        f"Target {targ} is in schedule but is not defined in"
                        "target sequence"
                    )
                # check 2 : check if targets in each interval lie within time
                # ranges of target sequence
                time_start = self.target_sequence_dict[targ]["times"][0]
                time_end = self.target_sequence_dict[targ]["times"][-1]
                if time_start > time or time_end < time:
                    print(f"time range for {targ}", time_start, time_end)
                    print(f"target schedule time, {time}")
                    raise ValueError(
                        f"Range of defined values for Target {targ} not"
                        "compatible with schedule"
                    )

    def load_pickle_dict(self, path):
        """
        Load the dictionary from the file.

        Parameters
        ----------
        - path : str
            Path to the file containing target schedule.

        Returns
        -------
        - pickle_dict : dictionary
            Dictionary with the file contents.

        """
        file_ext = (path).split(".")[-1]
        if file_ext == ("pkl" or "pickle"):
            print(f"loading {file_ext[0]}")
            # load target sequence from pickle file
            with open(path, "rb") as fp:
                pickle_dict = pickle.load(fp)
        return pickle_dict

    def interpolate(self, time_stamp, target):
        """
        Interpolate the target value at time_stamp, from the information in
        target_sequence_dict.

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
                self.target_sequence_dict[target]["times"],
                self.target_sequence_dict[target]["vals"]
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

        closest_key = max((key for key in self.target_schedule_dict
                           if key <= time_stamp), default=None)

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
            [
                self.interpolate(time_stamp, targ)
                for targ in controlled_targets
            ]
        )

        return targets_required
