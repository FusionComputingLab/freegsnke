"""
Module to obtain control voltages from virtual circuits.

"""

import numpy as np
from copy import deepcopy
import pickle
import h5py

from . import virtual_circuits as vc  # import the virtual circuit class
from .virtual_circuits import VirtualCircuit
from . import machine_config
from .nonlinear_solve import nl_solver
from .equilibrium_update import Equilibrium


class ControlVoltages:
    """
    Class to implement control voltages from virtual circuit, and given a set of observed target values, and a set of requested target values.

    Attributes :
    ???

    Methods :
    get_active_coils : retrieve the active coils from the equilibrium object.
    get_inductance : retrieve inductance matrix from machine config
    calc_vc : retrieve from file or compute a virtual circuit object from freegsnke or NN emulator.
    calculate_feedback_voltage_vector : compute feedback voltages from a virtual circuit object and a set of target shifts.
    """

    def __init__(
        self, eq: Equilibrium, profiles, stepping: nl_solver, targets=None, coils=None
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
                list of coil names defaults to all active coils defined in get_active_coils.
        """
        # assign equi and profiles objects
        self.eq = eq
        self.profiles = profiles

        self.stepping = stepping
        self.n_active_coils = (
            self.stepping.n_active_coils
        )  # could also be eq.tokamak.n_active_coils
        print("number active coils", self.n_active_coils)
        # initialse targets with defaults or lists given
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
            print("initilasing with default all active coils")
            self.coils = deepcopy(self.active_coils)
        else:
            print("initilasing with custom coils")
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
        # initialise a VC handling ojbect
        self.VCH = vc.VirtualCircuitHandling()
        self.VCH.define_solver(self.stepping.NK, target_relative_tolerance=1e-7)

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
        if coils is None:  # use default of all acitve coils from tokamak
            print("Inductance matrix for default of all acitve coils")
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

    ## this function will be repalced by instance of build virtual circuit class.
    def calc_vc(self, eq=None, profiles=None, targets=None, coils=None):
        """
        Compute a VC using freegsnke VirtualCircuitHandling if no vc is provided.

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

        # if targets and coils provided, update targets/coils attributes
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

    def calculate_feedback_voltage_vector(
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
        Compute current given a set of target value shifs and vc matrix, at a given time.

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
            print("gain matrix not provided, using identity matrix")
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
            print("No VC ojbect passed, building one with ")
            # check coils in virtual circuit match those in the tokamak
            print("target names provided ", target_names)
            print("self targets", self.targets)
            print("self coils", self.coils)
            virtual_circuit = self.calc_vc(
                eq=eq, profiles=profiles, targets=self.targets, coils=self.coils
            )
        else:
            print("virtual circuit provided")
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
            print("observed targets not provided, calculating from equilibrium")
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
            "volatges v1 : reorder currents, fill in zeros and multiply by full active coil inductance matrix"
        )
        print("voltages v1 : shape", voltages_v1.shape)
        print(voltages_v1)

        # option 2 reshape inductance matrix, muiltply by currents and then fill in zeros
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


class VirtualCircuitSequence:
    """
    Class to build a virtual circuit objects from file, and store the sequence of virtual circuits along with appropriate time stamsp.

    """

    def __init__(self, path=None):
        """
        Initialize the class

        Parameters
        ----------
        path : str
            path to the file containing VC's. Include file extension either hdf5 or pkl.

        Returns
        -------
        None
        """
        self.vc_path = path  # path to the virtual circuit file

        self.vc_times = []  # list of virtual circuit time stamps
        self.vc_index = []  # list of virtual circuit indices
        self.vc_sequence = []  # list of virtual circuit ojbects

        # input currents and profile parameters to recreate eq if needed
        self.input_currents = []  # list of input current dictionaries
        self.input_profile_pars = []  # list of input profile parameter dictionaries

        if path is not None:
            print("loading vcs from file")
            # populate the vc_sequence
            self.load_vcs_fromfile()
            # create dictionary of vc times and corresponding index
            self.vc_time_dict = {time: ind for ind, time in enumerate(self.vc_times)}
            n_vc = len(self.vc_times)
            print(f"there are {n_vc} VC's loaded")
        else:
            print("No file path provided. Add VC's manually if desired")

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
                    timestamp = item["time"]
                    vc_matrix = item["vc_matrix"]
                    shape_matrix = item["shape_matrix"]
                    targets = item["targets"]
                    coils = item["coils"]
                    targets_val = item["targets_val"]
                    input_currents = item["input_currents"]
                    input_profile_pars = item["input_profile_pars"]

                    vc_ojbect = VirtualCircuit(
                        name=f"vc_{index}_time_from_{timestamp:.4f}",
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

                    self.vc_sequence.append(vc_ojbect)
                    self.vc_times.append(timestamp)
                    self.vc_index.append(index)
                    self.input_currents.append(input_currents)
                    self.input_profile_pars.append(input_profile_pars)

        # elif file_ext == ("hdf5" or "h5"):
        #     # load vcs from hdf5 file
        #     print("loading VC's from hdf5 file")
        #     with h5py.File(self.vc_path, "r") as f:
        #         timestamps = f["timestamps"]
        #         timestamp_dict = {time: i for i, time in enumerate(timestamps)}

        #         # Iterate over stored iterations
        #         for iter_key in f.keys():
        #             if iter_key.startswith("time_index"):
        #                 group = f[iter_key]
        #                 timestamp = group.attrs["time"]
        #                 index = group.attrs["index"]
        #                 target_names = [name.decode() for name in group["targets"][:]]
        #                 coil_names = [name.decode() for name in group["coils"][:]]
        #                 shape_mat = group["shape_matrix"][:]
        #                 vc_mat = group["vc_matrix"][:]
        #                 targets_val = group["targets_val"][:]
        #                 input_currents = group["input_currents"][:]
        #                 input_profile_pars = group["input_profile_pars"][:]

        #                 # add vc data to sequence
        #                 vc_ojbect = vc.VirtualCircuit(
        #                     f"vc_{index}_from_time_{timestamp:.4f}",
        #                     eq=None,
        #                     profiles=None,
        #                     shape_matrix=shape_mat,
        #                     VCs_matrix=vc_mat,
        #                     targets=target_names,
        #                     targets_val=targets_val,
        #                     targets_options=None,
        #                     non_standard_targets=None,
        #                     coils=coil_names,
        #                 )
        #                 self.vc_sequence.append(vc_ojbect)
        #                 self.vc_times.append(timestamp)
        #                 self.input_currents.append(input_currents)
        #                 self.input_profile_pars.append(input_profile_pars)

    def add_vc_to_sequence(self, virtual_circuit, time_stamp):
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
        self.vc_times.append(time_stamp)
        self.vc_sequence.append(virtual_circuit)
        # update vc time dictionary
        self.vc_time_dict = {time: ind for ind, time in enumerate(self.vc_times)}

    def retrieve_vc(self, time_stamp=None, time_index=None):
        """
        Retrieve appropriate virtual circuit object from the sequence of virtual circuits.
        prr

        Parameters
        ----------
        time_stamp : float (4 decimal places)
            time stamp of the virtual circuit to be retrieved
        time_index : int
            index in the sequence of virtual circuits to be retrieved. start at zero

        Returns
        -------
        vc : object (VirtualCircuit)
            virtual circuit object to be used by the control voltages class
        """

        # select desired point in sequence
        def find_time_interval_index(times, target_time):
            """
            Finds the index of the interval in which the target_time falls.
            times array must be ordered.

            Parameters:
            -----------
            times (list of float): A sorted list of timestamps.
            target_time (float): The time to locate within the intervals.

            Returns:
            int: The index of the interval where target_time lies (between times[i] and times[i+1]).
                Returns -1 if target_time is out of range.
            """
            if target_time < times[0] or target_time > times[-1]:
                return -1  # Out of range

            for i in range(len(times) - 1):
                if times[i] <= target_time < times[i + 1]:
                    return i

            # Handle edge case where target_time matches the last timestamp exactly
            if target_time == times[-1]:
                return len(times) - 2

            return -1  # Should never reach this line

        if time_stamp is not None and time_index is None:
            postition = find_time_interval_index(
                times=self.vc_times, target_time=time_stamp
            )
        elif time_stamp is None and time_index is not None:
            postition = time_index
        else:
            print("Please specify either a time stamp or a time step")
            return None

        virual_circuit = self.vc_sequence[postition]
        return virual_circuit


class TargetSequence:
    """
    Class to build a target sequence from file, and store the sequence of desired targets along with appropriate time stamps.

    Method to return desired target via a linear interpolation of the target sequence.

    """

    def __init__(self, path=None):
        """
        Initialize the class

        Parameters
        ----------
        path : str
            path to the file containing target sequence

        Returns
        -------
        None
        """
        self.target_path = path  # path to the target sequence file

        self.target_name_dict = {}
        self.target_val_dict = {}

    def load_target_sequence(self):
        """
        Load the target sequence from the file.

        Returns
        -------
        None
            Modifies the attributes of the class.
        """
        file_ext = (self.target_path).split(".")[-1]
        if file_ext == ("pkl" or "pickle"):
            print("loading target sequence from pickle file")
            # load target sequence from pickle file
            with open(self.target_path, "rb") as fp:
                target_sequence_pkl = pickle.load(fp)

                n_times = len(target_sequence_pkl.keys())
                self.target_times = np.zeros(n_times)
                self.target_names = np.zeros(n_times)
                self.target_vals = np.zeros(n_times)

                i = 0
                for key, item in target_sequence_pkl.items():
                    timestamp = item["time"]
                    target_names = item["target_names"]
                    target_vals = item["target_vals"]
                    self.target_times[i] = timestamp
                    self.target_names[i] = target_names
                    self.target_vals[i] = target_vals
                    self.target_name_dict[timestamp] = target_names
                    self.target_val_dict[timestamp] = target_vals
                    i += 1

            # sort target times and rest of target sequence according to target times
            # needed for doing interpolation
            ind = np.argsort(self.target_times)
            self.target_times = np.array(self.target_times)[ind]
            self.target_names = np.array(self.target_names)[ind]
            self.target_vals = np.array(self.target_vals)[ind]

        # testing print statments
        print("target times", self.target_times)
        print("target names", self.target_names)
        print("target vals", self.target_vals)
        print("target name dict", self.target_name_dict)
        print("target val dict", self.target_val_dict)

    def retrieve_targets(self, time_stamp):
        """
        Retrieve appropriate target sequence from the sequence of target sequences, as linear interpolation between values at two adjacent time stamps.


        Parameters
        ----------
        time_stamp : float (4 decimal places)
            time stamp of the target to be retrieved

        Returns
        -------
        target_values : array
            requested target values to be used by the control voltages class
        target_name : list[str]
            list of target names
        """

        # linearly interpolate target sequence
        # check target names at time t_i and t_i+1  are the same
        index = np.sum(self.target_times < time_stamp)
        targets_left = self.target_names[index]
        targets_right = self.target_names[index + 1]
        if targets_left != targets_right:
            print("target names at time t_i and t_i+1 are not the same")
            print("targets at t_i", targets_left)
            print("targets at t_i+1", targets_right)

            print("filling in missing target values")
            ##### do this later - get from vc that is passed in  when this is used???

            return None

        target_values = np.interp(
            time_stamp, self.target_times, self.target_vals[index]
        )
        return target_values, targets_left


### TESTING ###
# if __name__ == "__main__":

#     pass
