"""
Module to obtain control voltages from virtual circuits.

"""

import numpy as np
from copy import deepcopy
import pickle

# import h5py

from . import virtual_circuits as vc  # import the virtual circuit class
from .virtual_circuits import VirtualCircuit
from . import machine_config
from .nonlinear_solve import nl_solver
from .equilibrium_update import Equilibrium


class VirtualCircuitSequence:
    """
    Class to build a virtual circuit objects from file, and store the sequence
    of virtual circuits along with appropriate time stamps.

    """

    def __init__(self, sequence_path=None):
        """
        Initialise the class

        Parameters
        ----------
        sequence_path : str
            sequence_path to the file containing VC's. Include file extension either hdf5 or pkl.

        Returns
        -------
        None
        """
        self.vc_path = sequence_path  # sequence_path to the virtual circuit file

        self.vc_times_calc = []  # times at which vcs are calculated
        self.vc_times_stop = []  # times at which vcs are to be stopped using
        self.vc_index = []  #
        self.vc_sequence = []  # list of virtual circuit ojbects

        # input currents and profile parameters to recreate eq if needed
        self.input_currents = []  # list of input current dictionaries
        self.input_profile_pars = []  # list of input profile parameter dictionaries

        if sequence_path is not None:
            print("loading vcs from file")
            # populate the vc_sequence
            self.load_vcs_fromfile()
            # create dictionary of vc times and corresponding index (using stop times)
            self.vc_time_stop_dict = {
                time: ind for ind, time in enumerate(self.vc_times_stop)
            }
            n_vc = len(self.vc_times_calc)
            print(f"there are {n_vc} VC's loaded")
        else:
            print("No file sequence_path provided. Add VC's manually if desired")

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

                    self.vc_sequence.append(vc_object)
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
        self.vc_sequence.append(virtual_circuit)
        # update vc time dictionary
        self.vc_time_dict = {time: ind for ind, time in enumerate(self.vc_times)}

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

        virtual_circuit = self.vc_sequence[position]
        return virtual_circuit


class TargetSequencer:
    """
    Class to build a target sequence from file, and store the sequence of desired targets along with appropriate time stamps.

    Method to return desired target via a linear interpolation of the target sequence.

    """

    def __init__(
        self, sequence_path, schedule_path, vc_flag="file", vc_sequence_path=None
    ):
        """
        Initialise the class

        Parameters
        ----------
        sequence_path : str
            path to the file containing target sequence
        schedule_path : str
            path to the file containing target schedule
        vc_flag : str   (optional)
            flag to indicate whether to load virtual circuit from file or NN emulator (default = "file")
            options = ["file", "Emulator"]
        vc_sequence_path : str (optional)
            path to the file containing virtual circuit sequence, if vc_flag = "file"

        Returns
        -------
        None
        """
        self.sequence_path = sequence_path  # path to the target sequence file
        self.schedule_path = schedule_path
        self.vc_flag = vc_flag
        self.vc_sequence_path = vc_sequence_path

        # load schedule
        self.target_schedule = self.load_target_schedule(self.schedule_path)
        # load target time series'
        self.load_target_sequence(self.sequence_path)
        if vc_flag == "file":
            # initilase a vc sequence object
            assert vc_sequence_path is not None, "Please provide a vc sequence path"
            self.vc_sequencer = VirtualCircuitSequence(self.vc_sequence_path)

            # add check to see if targets in VC's match targets in target schedule
            # merge the time sequence from both target and vc, and check the targets match at each midpiont.
            print("checking target schedule and vc sequence")
            change_times = np.sort(
                np.concatenate(
                    (
                        self.target_schedule_times,
                        self.vc_sequencer.vc_times_stop,
                    )
                )
            )
            midpoints = (change_times[:-1] + change_times[1:]) / 2
            for i, midpoint in enumerate(midpoints):
                vc_targs = self.vc_sequencer.retrieve_vc(time_stamp=midpoint).targets
                controlled_targs = self.retrieve_controlled_targets(time_stamp=midpoint)
                # if vc.targets != self.retrieve_controlled_targets(time_stamp=midpoint):
                #     print(
                #         "target schedule and vc sequence do not match at time", midpoint
                #     )
                #     print("target schedule", self.target_schedule_dict[midpoint])
                #     print(
                #         "vc sequence",
                #         self.vc_sequencer.retrieve_vc(time_stamp=midpoint).targets,
                #     )
                #     raise ValueError(
                #         "target schedule and vc sequence do not match at time", midpoint
                # )

                # check that the target schedule is a subset of the vc sequence
                if not set(controlled_targs).issubset(set(vc_targs)):
                    raise ValueError(
                        "targets scheduled not a subset of vc computable targets at time ",
                        midpoint,
                    )
                else:
                    # check the order of the targets
                    print("checking order of targets")
                    print("controlled targets")
                    print(controlled_targs)
                    print("VC available targets")
                    print(vc_targs)

        elif vc_flag == "emulator":
            # initilase an Emulator sequencer
            print("initilising an emulator sequencer")
            # self.vc_sequencer = EmulatorSequencer()
            pass

    def load_target_schedule(self, path):
        """
        Load the target schedule from the file. File should be a dictionary of lists of targets, indexed by times at which to start controlling that set of targets
        dict ~ {t_1_start : [targets], ..., }


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
        Retrieve the list of targets to be controlled at a given time, given target schedule.

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
            np.sum(self.target_schedule_times < time_stamp) - 1
        )  # subtract 1 to get index of t_start
        if index == -1:
            print("time requested is before first target schedule time")

        else:
            # print("index", index)
            target_names = self.target_schedule_dict[self.target_schedule_times[index]]
            # print("targets being controlled now are", target_names)
            return target_names
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
                self.target_sequence = target_sequence_pkl  # assign dictionary of target sequences to class

    def desired_target_values(self, time_stamp):
        """
        Retrieve appropriate target sequence from the sequence of target sequences, as linear interpolation between values at two adjacent time stamps.


        Parameters
        ----------
        time_stamp : float (4 decimal places)
            time stamp of the target to be retrieved

        Returns
        -------
        target_values : array   (float)
            requested target values to be used by the control voltages class
        """
        # get set of targets being controlled at this time
        controlled_targets = self.retrieve_controlled_targets(time_stamp)

        target_values = np.array(
            [
                np.interp(
                    time_stamp,
                    self.target_sequence[targ]["times"],
                    self.target_sequence[targ]["vals"],
                )
                for targ in controlled_targets
            ]
        )
        return target_values


class ControlVoltages:
    """
        Class to implement control voltages from virtual circuit, and given a set of observed target values, and a set of requested target values.

        Attributes :
        ???

        Methods :
        get_active_coils : retrieve the active coils from the equilibrium object.
        get_inductance : retrieve inductance matrix from machine config
        calc_vc_from_eq : retrieve from file or compute a virtual circuit object from freegsnke or NN emulator.
    def calculate_voltage_vc_feedback_proportional : compute feedback voltages from a virtual circuit object and a set of target shifts.
    """

    def __init__(
        self,
        eq: Equilibrium,
        profiles,
        stepping: nl_solver,
        vc_sequencer: VirtualCircuitSequence,
        targets=None,
        coils=None,
    ):
        """
        Initialize the control voltages class

        Parameters
        ----------
            eq : equilibrium object
                equilibrium object
            profiles : list of profiles
                list of profiles
            stepping : Non Linear Solver object
                Non Linear Solver object
            targets : list[str] (optional)
                list of target names, defaults to ["R_in", "R_out", "Rx_lower","Rs_lower_outer"]
            coils : list[str]   (optional)
                list of coil names, defaults to all active coils defined in get_active_coils.
        """
        # assign equi and profiles objects
        self.eq = eq
        self.profiles = profiles

        self.stepping = stepping
        self.n_active_coils = (
            self.stepping.n_active_coils
        )  # could also be eq.tokamak.n_active_coils
        print("number active coils", self.n_active_coils)
        # initialise targets with defaults or lists given
        if targets is None:
            targets = ["R_in", "R_out", "Rx_lower", "Rs_lower_outer"]
            self.targets = targets
        else:
            self.targets = targets

        # set coil lists and dictionary for all active coils
        self.active_coils = self.eq.tokamak.coils_list[: self.n_active_coils]

        # create a dictionary to map coil names to their order in the list
        order_dictionary = {coil: i for i, coil in enumerate(self.active_coils)}
        self.order_dictionary = order_dictionary

        # assign coils to default or
        if coils is None:
            print("initialising with default all active coils")
            self.coils = deepcopy(self.active_coils)
        else:
            print("initialising with custom coils")
            self.coils = coils

        print("Default targets and current's initialised")
        print(self.coils)
        print(self.targets)
        print(self.active_coils)

        # get inductance matrix (full with all active coils)
        # ??Machine config and inductance matrix will come from stepper function later??
        self.inductance_full = machine_config.coil_self_ind[
            : len(self.active_coils), : len(self.active_coils)
        ]
        # initialise a VC handling object
        self.VCH = vc.VirtualCircuitHandling()
        self.VCH.define_solver(self.stepping.NK, target_relative_tolerance=1e-7)

        self.VC_sequencer = vc_sequencer

    def get_inductance_reduced(self, coils=None):
        """
        Select appropriate inductance rows and columns from inductance matrix, given set of coils in the VC.

        parameters
        ----------
        coils : list[str]
            list of coil names

        Returns
        -------
        inductance_reduced : np.array
            inductance matrix of reduced set of coils. Also updates inductance matrix attribute


        """
        if coils is None:  # use default of all active coils from tokamak
            print("Inductance matrix for default of all active coils")
            coils = self.coils
        else:  # use coils provided and select apropriate part of inductance matrix
            print(f"Inductance matrix for coils provided {coils}")
            pass

        # create mask for selecting part of inductance matrix
        mask = [self.order_dictionary[coil] for coil in coils]
        print("coil ordering mask ", mask)
        inductance_reduced = self.inductance_full[np.ix_(mask, mask)]
        self.inductance_reduced = inductance_reduced

        return inductance_reduced

    ## this function will be replaced by instance of build virtual circuit class.
    def calc_vc_from_eq(self, eq=None, profiles=None, targets=None, coils=None):
        """
        Compute a VC using freegsnke VirtualCircuitHandling.

        Parameters
        ----------
        eq : object
            equilibrium object

        profiles : object
            profiles object

        targets : list[str]
            list of targets (optional)

        coils : list[str]
            list of coils (optional)


        Returns
        -------
        virtual_circuit : object
            virtual circuit object
        """
        if eq is None:
            eq = self.eq
        if profiles is None:
            profiles = self.profiles

        # if targets and coils are provided, update targets/coils attributes
        if targets is not None:
            self.targets = targets
        if coils is not None:
            self.coils = coils

        print("building virtual circuit from freegsnke")
        self.VCH.calculate_VC(
            eq,
            profiles,
            coils=self.coils,
            targets=self.targets,
            targets_options=None,
        )

        # get the virtual circuit object
        virtual_circuit = self.VCH.latest_VC

        return virtual_circuit

    def calculate_voltage_vc_feedback_proportional(
        self,
        eq,
        profiles,
        targets_req,
        targets_obs=None,
        target_names=None,
        coil_names=None,
        virtual_circuit: VirtualCircuit = None,
        gain_matrix=None,
    ):
        """
        Compute current given a set of target value shifts and vc matrix, at a given time.

        Assigns attributes in place for feedback voltages, and returns them.

        Parameters
        ----------
        eq : object
            equilibrium object

        profiles : object
            profiles object

        targets_req : array
            array function of target values for each control voltage at a given time

        targets_obs : array
            array function of target values for each control voltage at given  time
            Defaults to None, in which case the targets are computed from the equilibrium.

        targets_names : list
            list of target names. Defaults to None, in which case the targets taken to be all active.

        virtual_circuit : object
            virtual circuit object. Defaults to None, in which case the virtual circuit is computed from the equilibrium.
            with default currents of the Tokamak minus p6, and solenoid (these are determined differently)

        gain_matrix : array
            diagonal square matrix of target gains. Defaults to identity matrix

        Returns
        -------
        feedback_voltages : array
            feedback voltages
        """

        # set default gain matrix if not provided
        if gain_matrix is None:
            gain_matrix = np.identity(len(targets_req))
            print("Gain matrix not provided, using identity matrix")
            print(gain_matrix)

        assert gain_matrix.shape[0] == len(
            targets_req
        ), "The gain matrix is not the same length as the target vector"

        # check coils and targets and update attributes accordingly
        if target_names is not None:
            self.targets = target_names

        if coil_names is not None:
            print("updating coils to", coil_names)
            self.coils = coil_names
        # build VC object if not provided
        if virtual_circuit is None:
            print("No VC object passed, building one with ")
            # check coils in virtual circuit match those in the tokamak
            print("target names provided ", target_names)
            print("self targets", self.targets)
            print("self coils", self.coils)
            virtual_circuit = self.calc_vc_from_eq(
                eq=eq, profiles=profiles, targets=self.targets, coils=self.coils
            )
        else:
            print("Virtual circuit provided")
            print("targets", virtual_circuit.targets)
            print("coils", virtual_circuit.coils)

        # assign virtual circuit attribute to class
        self.virtual_circuit = virtual_circuit

        if not target_names == virtual_circuit.targets:
            print(
                "The virtual circuit targets do not match the targets requested \n targets being updated to those in the VC"
            )
            self.targets = virtual_circuit.targets

        if targets_obs is None:
            # get the targets from the equilibrium
            print("Observed targets not provided, calculating from equilibrium")
            _, targets_obs = self.VCH.calculate_targets(eq, self.targets)
            print(targets_obs)

            # check dimensions of target values
        assert len(targets_req) == len(
            targets_obs
        ), "The target required and observed vectors are not the same length"

        # shifts required
        target_deltas = targets_req - targets_obs
        print("target deltas", target_deltas)

        # do matrix multiplication VC @ G @ delta
        delta_currents = virtual_circuit.VCs_matrix @ gain_matrix @ target_deltas
        print("delta currents", delta_currents)

        # option 1 reorder currents, fill in zeros and multiply by inductance matrix
        reshaped_currents = np.zeros(len(self.active_coils))
        for i, coil in enumerate(virtual_circuit.coils):
            # voltages_v1[i] = np.dot(inductance_matrix[self.order_dictionary[coil],:], delta_currents[:])
            reshaped_currents[self.order_dictionary[coil]] = delta_currents[i]
        print("reshaped currents")
        print(reshaped_currents)
        voltages_v1 = np.dot(self.inductance_full, reshaped_currents)

        print("------------- \n compuiting voltages \n -------------")
        print(
            "voltages v1 : reorder currents, fill in zeros and multiply by full active coil inductance matrix"
        )
        print("voltages v1 : shape", voltages_v1.shape)
        print(voltages_v1)

        # option 2 reshape inductance matrix, multiply by currents and then fill in zeros
        print("doing option 2")
        print("delta currents", delta_currents)
        inductance_matrix_reduced = self.get_inductance_reduced(
            coils=virtual_circuit.coils
        )
        voltages_v2_temp = np.dot(inductance_matrix_reduced, delta_currents)
        # fill in zeros
        voltages_v2 = np.zeros(len(self.active_coils))
        for i, coil in enumerate(virtual_circuit.coils):
            voltages_v2[self.order_dictionary[coil]] = voltages_v2_temp[i]

        print(
            "voltages v2 : reshaped inductance matrix, then fill in zeros in voltage vector"
        )
        print("voltages v2 : shape", voltages_v2.shape)
        print(voltages_v2)

        self.feedback_voltages_v1 = voltages_v1
        self.feedback_voltages_v2 = voltages_v2

        return voltages_v1, voltages_v2


### TESTING ###
# if __name__ == "__main__":

#     pass
