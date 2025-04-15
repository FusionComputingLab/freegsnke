"""
Module for target and virtual circuit sequencing in control loop.

"""

import pickle
import numpy as np

from fnkemu.virtual_circuits.virtual_circuit_generator import VC_Generator as VCG
from freegsnke.virtual_circuits import VirtualCircuit
from .target_scheduler import TargetScheduler


class VirtualCircuitScheduler:
    """
    Class to build a virtual circuit objects from file, and store the sequence
    of virtual circuits along with appropriate time stamps.

    """

    def __init__(self, vc_schedule_path=None):
        """
        Initialise the class

        Parameters
        ----------
        vc_schedule_path : str
            vc_schedule_path to the file containing VC's. Include file
            extension either hdf5 or pkl.

        Returns
        -------
        None
        """

        self.vc_times_calc = []  # times at which vcs are calculated
        self.vc_times_stop = []  # times at which vcs are to be stopped using
        self.vc_index = []  #
        self.vc_schedule = []  # list of virtual circuit ojbects
        self.gains = []

        # input currents and profile parameters to recreate eq if needed
        self.input_currents = []  # list of input current dictionaries
        self.input_profile_pars = []  # list of input profile parameter dicts

        if vc_schedule_path is not None:
            print("loading vcs from file")
            # populate the vc_schedule
            self.load_vcs_fromfile(vc_schedule_path)
            # create dictionary of vc times and corresponding index
            # (using stop times)
            self.vc_time_stop_dict = {
                time: ind for ind, time in enumerate(self.vc_times_stop)
            }
            n_vc = len(self.vc_times_calc)
            print(f"{n_vc} VC's loaded")
        else:
            print("No file target_waveform_path provided. Add VC's manually if desired")

    def load_vcs_fromfile(self, path):
        """
        Load the virtual circuit matrix, shape matrix, coils and targets from a
        file, and save a list of VC objects.

        Returns
        -------
        None :
            Modifies the attributes of the class.
        """
        # file extension - hdf5 or csv or ???
        file_ext = (path).split(".")[-1]
        if file_ext == ("pkl" or "pickle"):
            # load vcs from pickle file
            with open(path, "rb") as fp:
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
                    try:
                        gains_arr = item["target_gains"]
                    except:
                        print("no gains provided - default to 1")
                        gains_arr = np.ones(np.shape(targets))

                    gains_dict = dict(zip(targets, gains_arr))

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
                    self.input_currents.append(input_currents)
                    self.input_profile_pars.append(input_profile_pars)
                    assert len(gains_arr) == len(
                        targets
                    ), "gains provided don't match with number of targets"
                    self.gains.append(gains_dict)
        # convert times to numpy array
        self.vc_times_stop = np.array(self.vc_times_stop)
        self.vc_times_calc = np.array(self.vc_times_calc)

    # ???? Do we need this???? Maybe delete this method.
    def add_vc_to_sequence(self, virtual_circuit, time_start, time_stop):
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
        self.vc_time_stop_dict = {
            time: ind for ind, time in enumerate(self.vc_times_stop)
        }

        # update other parts such as vc_index, input currents, profile pars etc

    def retrieve_vc(self, time_stamp=None, time_index=None):
        """
        Retrieve appropriate virtual circuit object from the sequence of
        virtual circuits.

        Parameters
        ----------
        time_stamp : float (4 decimal places)
            time stamp of the virtual circuit to be retrieved
        time_index : int
            index in the sequence of virtual circuits to be retrieved. Start at
            zero.

        Returns
        -------
        vc : object (VirtualCircuit)
            virtual circuit object to be used by the control voltages class
        """

        # get index for time stamp
        if ((time_stamp is None) ^ (time_index is None)) is False:
            print("Please specify either a time stamp or a time step")
            # Maybe raise error instead
            return None

        # if time_stamp is not None and time_index is None:
        if time_stamp is not None:
            position = np.sum(self.vc_times_stop < time_stamp)
            if position >= len(self.vc_times_stop):
                # use last vc if time beyond range
                position = -1
        # elif time_stamp is None and time_index is not None:
        elif time_index is not None:
            position = time_index

        virtual_circuit = self.vc_schedule[position]
        return virtual_circuit

    def retrieve_gains(self, targets, time_stamp=None, time_index=None):
        """
        Retrieve the list of targets to be controlled at a given time, given the target schedule.
        Parameters
        ----------
        time_stamp : float (4 decimal places)
            time stamp of the target to be retrieved
        Returns
        -------
        gains_matrix : np.array
            gains matrix
        """
        # get index for time stamp
        if ((time_stamp is None) ^ (time_index is None)) is False:
            print("Please specify either a time stamp or a time step")
            # Maybe raise error instead
            return None

        # if time_stamp is not None and time_index is None:
        if time_stamp is not None:
            position = np.sum(self.vc_times_stop < time_stamp)
            if position >= len(self.vc_times_stop):
                # use last vc if time beyond range
                position = -1
        # elif time_stamp is None and time_index is not None:
        elif time_index is not None:
            position = time_index

        gains_arr = np.array([self.gains[position][targ] for targ in targets])

        gains_matrix = np.diag(gains_arr)

        return gains_matrix


class ShapeTargetScheduler(TargetScheduler):
    """
    Class to build a target sequences from file, and store the sequence of
    desired targets along with appropriate time stamps.
    Naming conventions:
    Targets - These refer to 'shape targets'.
    Target Schedule - This provides which targets are to be controlled at a
    given time.
    Target waveform - This provides the actual desired/requested targets at a
    given time.
    VC Schedule - The schedule of VC's to be used up to a given time. Similar
    to Target Schedule

    """

    def __init__(
        self,
        target_waveform_path,
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
        target_waveform_path : str
            path to the file containing target sequence
        target_schedule_path : str
            path to the file containing target schedule
        vc_flag : str   (optional)
            flag to indicate whether to load virtual circuit from file or NN
            emulator (default = "file")
            options = ["file", "Emulator"]
        vc_schedule_path : str (optional)
            path to the file containing virtual circuit sequence, if
            vc_flag = "file"

        Returns
        -------
        None
        """

        super().__init__(target_waveform_path, target_schedule_path)

        self.vc_flag = vc_flag

        # check if vc_flag is file and load VC's from file.
        if vc_flag == "file":
            # initilase a vc sequence object
            assert vc_schedule_path is not None, "Please provide a vc sequence path"
            self.vc_scheduler = VirtualCircuitScheduler(vc_schedule_path)

            # add check to see if targets in VC's match targets in target
            # schedule
            # merge the time sequence from both target and vc, and check the
            # targets match at each midpiont.
            print("checking target schedule and vc sequence")
            change_times = np.sort(
                np.concatenate(
                    (
                        list(self.target_schedule_dict.keys()),
                        self.vc_scheduler.vc_times_stop,
                    )
                )
            )
            midpoints = (change_times[:-1] + change_times[1:]) / 2
            # for _, midpoint in enumerate(midpoints):
            for midpoint in midpoints:
                # print(
                #     "checking compatibility of target schedule and vc"
                #     f" sequence at time {midpoint}"
                # )
                vc_targs = self.vc_scheduler.retrieve_vc(time_stamp=midpoint).targets
                controlled_targs = self.retrieve_controlled_targets(time_stamp=midpoint)
                # check that the target schedule is a subset of the vc sequence
                if not set(controlled_targs).issubset(set(vc_targs)):
                    raise ValueError(
                        "targets scheduled for control not a subset of vc "
                        f"computable targets at time {midpoint} ",
                    )
                elif controlled_targs != vc_targs:
                    # check the order of the targets
                    print(
                        "targets requested and vc available targets do not match : vc's will be recomputed as necessary"
                    )
                    # print("controlled targets", controlled_targs)
                    # print("VC available targets", vc_targs)

        elif vc_flag == "emulator" or "emu" or "Emulator":
            # initilase an Emulator scheduler
            assert model_path is not None, "Please provide a model path"
            print("initialising an emulator scheduler")
            self.vc_scheduler = VCG(model_path, model_names=None, n_models=None)

    def get_vc(self, eq, profiles, time_stamp, coils):
        """
        Get VC object given time stamp.
        - load from file if provided or compute with emulator
        """

        if self.vc_flag == "file":
            print("loading VC from file")
            vc = self.vc_scheduler.retrieve_vc(time_stamp=time_stamp)
        elif self.vc_flag == "Emulator" or "emulator" or "emu":
            print("Computing VC from emulator")
            control_targs = self.retrieve_controlled_targets(time_stamp)
            vc = self.vc_scheduler.build_vc(
                eq, profiles, coils=coils, targets=control_targs
            )
        return vc
