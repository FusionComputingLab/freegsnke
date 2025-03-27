"""
Module for target and virtual circuit sequencing in control loop.

"""

import pickle
from copy import deepcopy

import numpy as np

from .. import machine_config
from .. import virtual_circuits as vc  # import the virtual circuit class
from ..equilibrium_update import Equilibrium
from ..nonlinear_solve import nl_solver
from ..virtual_circuits import VirtualCircuit

from fnkemu.virtual_circuits.virtual_circuit_generator import VC_Generator as VCG

# import h5py


class VirtualCircuitSequence:
    """
    Class to build a virtual circuit objects from file, and store the sequence
    of virtual circuits along with appropriate time stamps.

    """

    def __init__(self, target_sequence_path=None):
        """
        Initialise the class

        Parameters
        ----------
        target_sequence_path : str
            target_sequence_path to the file containing VC's. Include file extension either hdf5 or pkl.

        Returns
        -------
        None
        """
        self.vc_path = (
            target_sequence_path  # target_sequence_path to the virtual circuit file
        )

        self.vc_times_calc = []  # times at which vcs are calculated
        self.vc_times_stop = []  # times at which vcs are to be stopped using
        self.vc_index = []  #
        self.vc_schedule = []  # list of virtual circuit ojbects

        # input currents and profile parameters to recreate eq if needed
        self.input_currents = []  # list of input current dictionaries
        self.input_profile_pars = []  # list of input profile parameter dictionaries

        if target_sequence_path is not None:
            print("loading vcs from file")
            # populate the vc_schedule
            self.load_vcs_fromfile()
            # create dictionary of vc times and corresponding index (using stop times)
            self.vc_time_stop_dict = {
                time: ind for ind, time in enumerate(self.vc_times_stop)
            }
            n_vc = len(self.vc_times_calc)
            print(f"there are {n_vc} VC's loaded")
        else:
            print("No file target_sequence_path provided. Add VC's manually if desired")

    def load_vcs_fromfile(self):
        """
        Load the virtual circuit matrix, shape matrix, coils and targets from a file, and save a list of VC objects.

        Returns
        -------
        None :
            Modifies the attributes of the class.
        """
        # file extension - hdf5 or csv or ???
        file_ext = (self.vc_path).split(".")[-1]
        if file_ext == ("pkl" or "pickle"):
            print("loading VC's from pickle file")
            # load vcs from pickle file
            with open(self.vc_path, "rb") as fp:
                vcs_pkl = pickle.load(fp)

                for key, item in vcs_pkl.items():
                    index = item["index"]
                    time_calc = item["time_calc"]
                    time_stop = item["time_stop"]
                    vc_matrix = item["vc_matrix"]
                    shape_matrix = item["shape_matrix"]
                    targets = item["targets"]
                    coils = item["coils"]
                    targets_val = item["targets_val"]
                    input_currents = item["input_currents"]
                    input_profile_pars = item["input_profile_pars"]

                    vc_object = VirtualCircuit(
                        name=f"vc_{index}_time_upto_{time_stop:.4f}",
                        eq=None,
                        profiles=None,
                        shape_matrix=shape_matrix,
                        VCs_matrix=vc_matrix,
                        targets=targets,
                        coils=coils,
                        targets_val=targets_val,
                        targets_options=None,
                        non_standard_targets=None,
                    )

                    self.vc_schedule.append(vc_object)
                    self.vc_times_calc.append(time_calc)
                    self.vc_times_stop.append(time_stop)
                    self.vc_index.append(index)
                    self.input_currents.append(input_currents)
                    self.input_profile_pars.append(input_profile_pars)
        # convert times to numpy array
        self.vc_times_stop = np.array(self.vc_times_stop)
        self.vc_times_calc = np.array(self.vc_times_calc)

    def add_vc_to_sequence(self, virtual_circuit, time_stop):
        """
        Add virtual circuit to sequence.

        Parameters
        ----------
        virtual_circuit : object
            virtual circuit object
        time_stamp : float
            time stamp of the virtual circuit

        Returns
        -------
        None
            modifies object in place
        """
        print("adding vc to sequence")
        self.vc_times_stop.append(time_stop)
        self.vc_schedule.append(virtual_circuit)
        # update vc time dictionary
        self.vc_time_dict = {time: ind for ind, time in enumerate(self.vc_times_stop)}

    def retrieve_vc(self, time_stamp=None, time_index=None):
        """
        Retrieve appropriate virtual circuit object from the sequence of virtual circuits.

        Parameters
        ----------
        time_stamp : float (4 decimal places)
            time stamp of the virtual circuit to be retrieved
        time_index : int
            index in the sequence of virtual circuits to be retrieved. Start at zero

        Returns
        -------
        vc : object (VirtualCircuit)
            virtual circuit object to be used by the control voltages class
        """

        # get index for time stamp
        if time_stamp is not None and time_index is None:
            position = np.sum(self.vc_times_stop < time_stamp)
            if position >= len(self.vc_times_stop):
                # use last vc if time beyond range
                position = -1
        elif time_stamp is None and time_index is not None:
            position = time_index
        else:
            print("Please specify either a time stamp or a time step")
            return None

        virtual_circuit = self.vc_schedule[position]
        return virtual_circuit


class TargetSequencer:
    """
    Class to build a target sequences from file, and store the sequence of desired targets along with appropriate time stamps.
    Naming conventions:
    Targets - These refer to 'shape targets'.
    Target Schedule - This provides which targets are to be controlled at a given time.
    Target Sequence - This provides the actual desired/requested targets at a given time.
    VC Schedule - The schedule of VC's to be used up to a given time. Similar to Target Schedule


    """

    def __init__(
        self,
        target_sequence_path,
        target_schedule_path,
        vc_flag="file",
        vc_schedule_path=None,
        model_path=None,
        model_names=None,
        n_models=None,
    ):
        """
        Initialise the class

        Parameters
        ----------
        target_sequence_path : str
            path to the file containing target sequence
        target_schedule_path : str
            path to the file containing target schedule
        vc_flag : str   (optional)
            flag to indicate whether to load virtual circuit from file or NN emulator (default = "file")
            options = ["file", "Emulator"]
        vc_schedule_path : str (optional)
            path to the file containing virtual circuit sequence, if vc_flag = "file"

        Returns
        -------
        None
        """
        self.target_sequence_path = (
            target_sequence_path  # path to the target sequence file
        )
        self.target_schedule_path = target_schedule_path
        self.vc_flag = vc_flag
        self.vc_schedule_path = vc_schedule_path

        # load schedule and sequence
        self.load_target_schedule(self.target_schedule_path)
        self.load_target_sequence(self.target_sequence_path)

        # check compatibility of target schedule and target sequence
        for time in self.target_schedule_times:
            targ_names = self.target_schedule_dict[time]
            for targ in targ_names:
                # check 1 : check if all targets in target schedule are in target sequence
                if targ not in self.target_sequence.keys():
                    raise ValueError(
                        f"Target {targ} is in schedule but is not defined in target sequence"
                    )
                # check 2 : check if targets in each interval lie within time ranges of target sequence
                time_start = self.target_sequence[targ]["times"][0]
                time_end = self.target_sequence[targ]["times"][-1]
                if time_start > time or time_end < time:
                    print(f"time range for {targ}", time_start, time_end)
                    print(f"target schedule time", time)
                    raise ValueError(
                        f"Range of defined values for Target {targ} not compatible with schedule"
                    )

        # check if vc_flag is file and load VC's from file.
        if vc_flag == "file":
            # initilase a vc sequence object
            assert vc_schedule_path is not None, "Please provide a vc sequence path"
            self.vc_scheduler = VirtualCircuitSequence(self.vc_schedule_path)

            # add check to see if targets in VC's match targets in target schedule
            # merge the time sequence from both target and vc, and check the targets match at each midpiont.
            print("checking target schedule and vc sequence")
            change_times = np.sort(
                np.concatenate(
                    (
                        self.target_schedule_times,
                        self.vc_scheduler.vc_times_stop,
                    )
                )
            )
            midpoints = (change_times[:-1] + change_times[1:]) / 2
            for _, midpoint in enumerate(midpoints):
                print(
                    f"checking compatibility of target schedule and vc sequence at time {midpoint}"
                )
                vc_targs = self.vc_scheduler.retrieve_vc(time_stamp=midpoint).targets
                controlled_targs = self.retrieve_controlled_targets(time_stamp=midpoint)
                # if vc.targets != self.retrieve_controlled_targets(time_stamp=midpoint):
                #     print(
                #         "target schedule and vc sequence do not match at time", midpoint
                #     )
                #     print("target schedule", self.target_schedule_dict[midpoint])
                #     print(
                #         "vc sequence",
                #         self.vc_scheduler.retrieve_vc(time_stamp=midpoint).targets,
                #     )
                #     raise ValueError(
                #         "target schedule and vc sequence do not match at time", midpoint
                # )

                # check that the target schedule is a subset of the vc sequence
                if not set(controlled_targs).issubset(set(vc_targs)):
                    raise ValueError(
                        f"targets scheduled not a subset of vc computable targets at time {midpoint} ",
                    )
                else:
                    # check the order of the targets
                    print("checking order of targets")
                    print("controlled targets", controlled_targs)
                    print("VC available targets", vc_targs)

                    # if order is different, or not full set, then recompute VC from sensitivity???

        elif vc_flag == "emulator":
            # initilase an Emulator sequencer
            print("initilising an emulator sequencer")
            self.vc_scheduler = VCG(model_path, model_names=None, n_models=None)

    def load_target_schedule(self, path):
        """
        Load the target schedule from the file. File should be a dictionary of lists of targets, indexed by times at which to start controlling that set of targets
        dict ~ {t_1_start : [target_names], t_2_start : [target_names], ... }


        Parameters
        ----------
        path : str
            path to the file containing target schedule

        Returns
        -------
        None
            Modifies the attributes of the class.
        """
        self.target_schedule_dict = {}

        file_ext = (path).split(".")[-1]
        if file_ext == ("pkl" or "pickle"):
            print("loading target schedule from pickle file")
            # load target sequence from pickle file
            with open(path, "rb") as fp:
                target_schedule_pkl = pickle.load(fp)
                times = list(target_schedule_pkl.keys())
                times.sort()
                self.target_schedule_times = np.array(times)

                for key, item in target_schedule_pkl.items():
                    self.target_schedule_dict[key] = (
                        item  # add  list of targets to dictionary
                    )
        print("target schedule times", self.target_schedule_times)
        print("target schedule dict", self.target_schedule_dict)
        return self.target_schedule_dict

    def retrieve_controlled_targets(self, time_stamp):
        """
        Retrieve the list of targets to be controlled at a given time, given the target schedule.

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

    def load_target_sequence(self, path):
        """
        Load the target sequence from the file, and store it in a nested dictionary.
        dict ~ {target_name : {"times":[time list], "vals": [vals list] , }

        Parameters
        ----------
        path : str
            path to the file containing target sequence

        Returns
        -------
        None
            Modifies the attributes of the class.
        """

        file_ext = (path).split(".")[-1]
        if file_ext == ("pkl" or "pickle"):
            print("loading target sequence from pickle file")
            # load target sequence from pickle file
            with open(path, "rb") as fp:
                target_sequence_pkl = pickle.load(fp)
                for key, item in target_sequence_pkl.items():
                    if len(item["times"]) != len(item["vals"]):
                        raise ValueError(
                            "times array and vals array must be same length"
                        )
                self.target_sequence = target_sequence_pkl  # assign dictionary of target sequences to class

    def desired_target_values(self, time_stamp):
        """
        Retrieve values for desired control targets as linear interpolation between values at two adjacent time stamps.

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
                    self.target_sequence[targ]["times"],
                    self.target_sequence[targ]["vals"],
                )
                for targ in controlled_targets
            ]
        )
        return targets_required

    def get_vc(self, time_stamp, coils):
        """
        Get VC object given time stamp.
        - load from file if provided or compute with emulator
        """

        if self.vc_flag == "file":
            vc = self.vc_scheduler.retrieve_vc(time_stamp=time_stamp)
        elif self.vc_flag == "emulator":
            control_targs = self.retrieve_controlled_targets(time_stamp)
            vc = self.vc_scheduler.build_vc(
                self.eq, self.profiles, coils=coils, targets=control_targs
            )
        return vc
