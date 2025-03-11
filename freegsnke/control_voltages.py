"""
Module to obtain control voltages from virtual circuits.

"""

import numpy as np
from copy import deepcopy
import pickle
import h5py

from . import virtual_circuits as vc  # import the virtual circuit class
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

        # initialse targets with defaults or lists given
        if targets is None:
            targets = ["R_in", "R_out", "Rx_lower", "Rs_lower_outer"]
            self.targets = targets
        else:
            self.targets = targets

        # set coil lists and dictionary for all active coils
        self.active_coils = self.eq.tokamak.coils_list[self.n_active_coils]

        # create a dictionary to map coil names to their order in the list
        order_dictionary = {coil: i for i, coil in enumerate(self.active_coils)}
        self.order_dictionary = order_dictionary

        # assign coils to default or
        if coils is None:
            self.coils = np.copy(self.active_coils)
        else:
            self.coils = coils

        # get inductance matrix (full with all active coils)
        # ??Machine config and inductance matrix will come from stepper function later??
        self.inductance_full = machine_config.coil_self_ind[
            len(self.active_coils), len(self.active_coils)
        ]

        # initialise a VC handling ojbect
        self.VCH = vc.VirtualCircuitHandling()
        self.VCH.define_solver(self.stepping.NK, target_relative_tolerance=1e-7)

    # def get_active_coils(self, eq):
    #     """
    #     Retrieve the active coils from the equilibrium object.

    #     set default coils to be used and set the order according to that in the tokamak description
    #     get all active ones
    #     assigne reduced set of coils without solenoid and p6 (these voltages will be set via  different method)

    #     Parameters
    #     ----------
    #     eq : object
    #         equilibrium object

    #     Returns
    #     -------
    #     active_coils : list
    #         list of all active coils
    #     active_coils_reduced : list
    #         list of default reduced set of active coils with solenoid and p6 removed
    #     order_dictionary : dict
    #         dictionary of coil names and their order in the list of all active coils
    #     """

    #     active_coils = eq.tokamak.coils_list[self.n_active_coils]

    #     self.active_coils = active_coils

    #     print("all active coils", self.active_coils)

    #     # create a dictionary to map coil names to their order in the list
    #     order_dictionary = {coil: i for i, coil in enumerate(active_coils)}
    #     self.order_dictionary = order_dictionary
    #     print("order dictionary", self.order_dictionary)

    #     return active_coils, order_dictionary

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
        inductance_reduced = machine_config.coil_self_ind[np.ix_(mask, mask)]
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
        virtual_circuit=None,
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

        # build VC object if not provided
        if virtual_circuit is None:
            virtual_circuit = self.calc_vc(
                eq=eq, profiles=profiles, targets=self.targets, coils=self.coils
            )
        # assign virtual circuit attribute to class
        self.virtual_circuit = virtual_circuit

        # check coils in virtual circuit match those in the tokamak
        print("target names provided ", target_names)
        print("self targets", self.targets)
        print("vc targs, ", virtual_circuit.targets)
        assert (
            target_names == virtual_circuit.targets
        ), "The virtual circuit targets do not match the targets requested"

        if targets_obs is None:
            # get the targets from the equilibrium
            targets_obs = self.VCH.calculate_targets(eq, self.targets)
            # check dimensions of target values
        assert len(targets_req) == len(
            targets_obs
        ), "The target required and observed vectors are not the same length"

        # shifts required
        target_deltas = targets_req - targets_obs
        print("target deltas", target_deltas)

        # do matrix multiplication VC @ G @ delta
        delta_currents = virtual_circuit.VCs_matrix @ gain_matrix @ target_deltas

        # option 1 reorder currents, fill in zeros and multiply by inductance matrix
        reshaped_currents = np.zeros(len(self.active_coils))
        for i, coil in enumerate(virtual_circuit.coils):
            # voltages_v1[i] = np.dot(inductance_matrix[self.order_dictionary[coil],:], delta_currents[:])
            reshaped_currents[self.order_dictionary[coil]] = delta_currents[i]
        voltages_v1 = np.dot(self.inductance_full, reshaped_currents)

        print(
            "volatges v1 : reorder currents, fill in zeros and multiply by full active coil inductance matrix"
        )
        print("voltages v1 : shape", voltages_v1.shape)
        print(voltages_v1)

        # option 2 reshape inductance matrix, muiltply by currents and then fill in zeros
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


class BuildVirtualCircuits:
    """
    Class to build a virtual circuit objects from either a file, and store the sequence of virtual circuits along with apropriate time stamsp


    Q's:
        ?? what format file will vc be saved in (csv, pickle,hdf5)??
        ?? what data will be saved (e.g. shape matrix, vcs matrix, targets, coils, etc.)
        ??
    """

    def __init__(self, path):
        """
        Initialize the class

        Parameters
        ----------
        path : str
            path to the file containing VC's

        Returns
        -------
        None
        """
        self.vc_path = path  # path to the virtual circuit file

        self.vc_config = {}  # dictionary of vc coil/target configurations
        self.vc_targs = []  # list of virtual circuit targets
        self.vc_coils = []  # list of virtual circuit coils
        self.vc_times = []  # list of virtual circuit time stamps
        self.vc_time_dict = {time: ind for ind, time in enumerate(self.vc_times)}
        self.vc_matrix_sequence = []
        self.target_sequence = []
        self.shape_matrix_sequence = []

    def load_vcs_fromfile(self):
        """
        ?? what format file will vc be saved in (csv, pickle,hdf5)??
        ?? what data will be saved (e.g. shape matrix, vcs matrix, targets, coils, etc.)

        Load the virtual circuit, coils and targets from a file, and save as attributes.

        Returns
        -------
        virtual_circuit : object
            virtual circuit object
        """
        # file extension - hdf5 or csv or ???
        file_ext = (self.path).split(".")[-1]
        if file_ext == "hdf5" or "h5":
            # load vcs from hdf5 file

            with h5py.File(self.path, "r") as f:
                timestamps = f["timestamps"]
                timestamp_dict = {time: i for i, time in enumerate(timestamps)}
                target_names = [name.decode() for name in f["targets"][:]]
                coil_names = [name.decode() for name in f["coils"][:]]

                print("loading VC's from hdf5 file")
                print("VC's have the following targets and coils:")
                print(f"Targets: {target_names}")
                print(f"Coils: {coil_names}")

                # Iterate over stored iterations
                for iter_key in f.keys():
                    if iter_key.startswith("time_step"):
                        group = f[iter_key]
                        timestamp = group.attrs["time"]
                        shape_mat = group["shape_matrix"][:]
                        vc_mat = group["vc_matrix"][:]
                        targ_vals = group["target_values"][:]

                        # add vc data to sequence
                        self.vc_matrix_sequence.append(vc_mat)
                        self.shape_matrix_sequence.append(shape_mat)
                        self.vc_targs.append(targ_vals)
                        self.vc_coils.append(coil_names)
                        self.vc_times.append(timestamp)

        elif file_ext == "pkl":
            # load vcs from pickle file
            with open(self.path, "rb") as fp:
                vcs = pickle.load(fp)

        pass

    def add_vc_to_sequence(self, vc_mat, time_stamp):
        """
        Add virtual circuit to sequence

        Parameters
        ----------
        vc : np.array
            virtual circuit matrix
        time_stamp : float
            time stamp of the virtual circuit

        Returns
        -------
        None
            modifies object in place
        """
        self.vc_times.append(time_stamp)
        self.vc_sequence.append(vc_mat)

    def retrieve_vc(self, time_stamp=None, time_step=None):
        """
        Retrieve apropriate virtual circuit object from the sequence of virtual circuits

        Parameters
        ----------
        time_stamp : float
            time stamp of the virtual circuit to be retrieved

        Returns
        -------
        vc : object (VirtualCircuit)
            virtual circuit object to be used by the control voltages class
        """

        # retrieve vc targets and coils
        vc_targs = self.vc_targs
        vc_coils = self.vc_coils

        # select desired point in sequence
        ### maybe can use np.where for this???
        if time_stamp is not None and time_step is None:
            postition = self.vc_time_dict[time_stamp]
        elif time_stamp is None and time_step is not None:
            postition = time_step
        else:
            print("Please specify either a time stamp or a time step")
            return None

        # retrieve vc and shape atrix from the sequence
        vc_mat = self.vc_matrix_sequence[postition]
        shape_mat = self.shape_matrix_sequence[postition]

        # assign to VC object ??? needs eqi and profiles???
        # only assign coils, targets and vc matrix. We don't have other data to initalise like eqi, profiles.
        virual_circuit = vc.VirtualCircuit(
            name=f"VC_at_time{time_stamp}",
            eq=None,
            profiles=None,
            shape_matrix=shape_mat,
            VCs_matrix=vc_mat,
            targets=vc_targs,
            coils=vc_coils,
            targets_options=None,
            non_standard_targets=None,
        )
        return virual_circuit


### TESTING ###
# if __name__ == "__main__":

#     pass
