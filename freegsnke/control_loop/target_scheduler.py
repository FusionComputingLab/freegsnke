import numpy as np
import pickle


class TargetScheduler:
    """

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
            path to the file containing target sequence
        target_schedule_path : str
            path to the file containing target schedule

        Returns
        -------
        None

        """
        # load schedule
        self.target_schedule_dict = self.load_pickle_dict(target_schedule_path)

        # schedule file should be a dictionary of lists of targets, indexed by
        # times at which to start controlling that set of targets

        times = sorted(list(self.target_schedule_dict.keys()))
        self.target_schedule_times = np.array(times)

        print("target schedule times", self.target_schedule_times)
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
        for time in self.target_schedule_times:
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
        path : str
            path to the file containing target schedule

        Returns
        -------
        dictionary
        """
        file_ext = (path).split(".")[-1]
        if file_ext == ("pkl" or "pickle"):
            print("loading target schedule from pickle file")
            # load target sequence from pickle file
            with open(path, "rb") as fp:
                pickle_dict = pickle.load(fp)
        return pickle_dict

    def retrieve_controlled_targets(self, time_stamp):
        """
        Retrieve the list of targets to be controlled at a given time,
        given the target schedule.

        Parameters
        ----------
        time_stamp : float (4 decimal places)
            time stamp of the target to be retrieved

        Returns
        -------
        target_names : list[str]
            list of target names
        """
        # find index for time stamp
        index = (
            np.sum(self.target_schedule_times <= time_stamp) - 1
        )  # subtract 1 to get index of t_start
        if index == -1:
            print("time requested is before first target schedule time")

        else:
            # print("index", index)
            target_names = self.target_schedule_dict[self.target_schedule_times[index]]
            # print("targets being controlled now are", target_names)
            return target_names

    def desired_target_values(self, time_stamp):
        """
        Retrieve values for desired control targets as linear interpolation
        between values at two adjacent time stamps.

        Parameters
        ----------
        time_stamp : float (4 decimal places)
            time stamp of the target to be retrieved

        Returns
        -------
        targets_required : array   (float)
            requested target values to be used by the control voltages class
        """
        # get set of targets being controlled at this time
        controlled_targets = self.retrieve_controlled_targets(time_stamp)

        targets_required = np.array(
            [
                np.interp(
                    time_stamp,
                    self.target_sequence_dict[targ]["times"],
                    self.target_sequence_dict[targ]["vals"],
                )
                for targ in controlled_targets
            ]
        )
        return targets_required
