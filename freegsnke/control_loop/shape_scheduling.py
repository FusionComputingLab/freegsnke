"""
Module for target and virtual circuit sequencing in control loop.

"""

# import pickle

from copy import deepcopy

import numpy as np

from freegsnke.observable_registry import ObservableRegistry
from freegsnke.virtual_circuits import VirtualCircuit

from .target_scheduler import TargetScheduler
from .vc_provider import VirtualCircuitProvider

# from fnkemu.virtual_circuits.virtual_circuit_generator import VC_Generator as VCG


class VirtualCircuitScheduler(VirtualCircuitProvider, TargetScheduler):
    """
    Class to build a virtual circuit objects from file, and store the sequence
    of virtual circuits along with appropriate time stamps.

    """

    def __init__(
        self,
        vc_schedule_dict: dict,
    ):
        """
        Initialise the class

        Parameters
        ----------
        vc_schedule_dict : dict
            vc_schedule_dict to the file containing VC's. Include file
            extension either hdf5 or pkl.

        Returns
        -------
        None
        """
        self.vc_schedule_full = vc_schedule_dict  # full schedule with possibly more vc columns than necessary
        self.schedule_times = sorted(list(self.vc_schedule_full.keys()))
        print(f"VC schedule with {len(self.vc_schedule_full)} vc phases")

        # self.build_reshaped_schedule(
        #     coil_order=vc_coil_order, target_order=control_targs_all
        # )

    def _build_reshaped_schedule(self, coil_order, target_order):
        """
        Create matrices of the correct shape and order appropriate for the control schedule.
        This should be called inside the ShapeTargetScheduler class, as that is where the row
        / column ordering is provided

        Parameters
        ----------
        coil_order : list[str]
            coil order for vc rows
        target_order : list[str]
            target order for vc columns. This should match all_control_targs in the shape_scheduler

        Returns
        -------
        None : modified in place and assigns class attribute.
        """
        print("building vc's")
        vc_schedule = {}
        for time, vc_dict in self.vc_schedule_full.items():
            # configuration from inputed dictionary
            vc_mat_full = vc_dict["vc_matrix"]
            coil_order_old = vc_dict["coils"]
            target_order_old = vc_dict["targets"]

            print("constructing VC's")
            # coil reordering is optional.
            if coil_order is None:
                coil_order = coil_order_old
            print("starting orders", coil_order_old, target_order_old)
            print("new orders", coil_order, target_order)

            # create empty matrix of zeros of appropriate shape (ncoils,ntargs)
            vc_mat_temp = np.zeros((len(coil_order), len(target_order)))

            if target_order_old == [] or target_order_old is None:
                # no vc provided - assign matrix of zeros
                print("no targets provided - will set vc's to zero")
                vc_matrix = vc_mat_temp
            else:
                # fill columns of matrix
                print(f"creating vc matrix for {target_order}")
                for i, targ in enumerate(target_order):
                    targ_index_full = target_order_old.index(targ)
                    # print(f"targ {targ} moves from colum {targ_index_full} to {i}")
                    vc_mat_temp[:, i] = vc_mat_full[:, targ_index_full]

            # reorder rows (keeps same order if coil_order wasn't provided)
            row_indices = [coil_order_old.index(c) for c in coil_order]
            vc_matrix = vc_mat_temp[row_indices, :]

            # put into dictionary
            vc_schedule[time] = {}
            vc_schedule[time]["vc_matrix"] = vc_matrix
            vc_schedule[time]["coil_order"] = coil_order
            vc_schedule[time]["target_order"] = target_order

        self.schedule = {}
        self.schedule["vc"] = vc_schedule
        print(vc_schedule)

    ### THIS IS OLD AND REDUNDANT
    # def unpack_vc_schedule(self, vcs_dict: dict):
    #     """
    #     Load the virtual circuit matrix, shape matrix, coils and targets from a
    #     dictionary, and save a list of VC objects and assocated data as class attributes.

    #     Returns
    #     -------
    #     None :
    #         Modifies the attributes of the class.
    #     """
    #     # set lists
    #     self.vc_times_start = []  # times at which vcs are to be stopped using
    #     self.vc_objects = []  # list of virtual circuit ojbects
    #     self.phase_names = []  # list of phase names
    #     for key, item in vcs_dict.items():
    #         phase_name = item["phase_name"]
    #         time_start = item["time_start"]
    #         print("time start", time_start)
    #         if "vc_matrix" in item.keys():
    #             vc_matrix = item["vc_matrix"]
    #             shape_matrix = item["shape_matrix"]
    #             targets = item["targets"]
    #             coils = item["coils"]
    #             # create VC object
    #             print("buliding VC")
    #             print("targets", targets)
    #             print("matrix", vc_matrix)
    #             vc_object = VirtualCircuit(
    #                 name=f"vc_start_{time_start:.4f}",
    #                 eq=None,
    #                 profiles=None,
    #                 shape_matrix=shape_matrix,
    #                 VCs_matrix=vc_matrix,
    #                 targets=targets,
    #                 coils=coils,
    #                 targets_val=None,
    #                 targets_options=None,
    #                 non_standard_targets=None,
    #             )
    #         else:
    #             vc_matrix = None
    #             vc_object = None
    #             shape_matrix = None
    #             targets = None
    #             coils = None

    #         self.vc_objects.append(vc_object)
    #         self.vc_times_start.append(time_start)
    #         self.phase_names.append(phase_name)

    # # convert times to numpy array
    # self.vc_times_start = np.array(self.vc_times_start)

    # THIS IS OLD  - keeping temporarily in case refactoring the scheduling goes wrong
    # def get_vc(
    #     self,
    #     time_stamp: float,
    #     targets: list[str] = None,
    # ) -> VirtualCircuit | None:
    #     """
    #     Gets a Virtual Circuit for the given timestamp and observables requested from
    #     the registry.

    #     Parameters
    #     ----------
    #     timestamp : float (4 decimal places)
    #         timestamp at which the virtual circuit should be retrieved
    #     targets : list[str]
    #         list of targets to get a virtual circuit for

    #     Returns
    #     -------
    #     vc : VirtualCircuit | None
    #         virtual circuit object to be used by the control voltages class or None if
    #         no virtual circuit could be obtained or constructed.
    #     """

    #     # find time position corresponding to vc to be used.
    #     t_vc = max(time for time in self.vc_times_start if time <= time_stamp)
    #     # get index corresponding to the time position
    #     pos = np.where(self.vc_times_start == t_vc)[0][0]

    #     virtual_circuit = self.vc_objects[pos]
    #     if virtual_circuit is None:
    #         return None
    #     # print("vc object matrix", virtual_circuit.VCs_matrix)
    #     if targets is not None:
    #         # check that the target schedule is a subset of the vc sequence
    #         if not set(targets).issubset(set(virtual_circuit.targets)):
    #             raise ValueError(
    #                 "targets scheduled for control not a subset of vc "
    #                 f"computable targets at time {time_stamp} ",
    #             )
    #         elif targets != virtual_circuit.targets:
    #             # check the order of the targets and select columns corresponding to the targets
    #             print(
    #                 "targets and vc targets do not match : Selecting columns corresponding to the targets"
    #             )
    #             targ_order_dict = dict(
    #                 zip(
    #                     virtual_circuit.targets, np.arange(len(virtual_circuit.targets))
    #                 )
    #             )
    #             mask = [targ_order_dict[targ] for targ in targets]
    #             print("coil ordering mask ", mask)
    #             # print("coil ordering mask ", mask)
    #             vc_mat_reduced = deepcopy(virtual_circuit.VCs_matrix[:, mask])
    #             # vc_mat_reduced = virtual_circuit.VCs_matrix[:, np.ix_(mask)]
    #             targs_reduced = [
    #                 virtual_circuit.targets[i]
    #                 for i in np.array([targ_order_dict[targ] for targ in targets])
    #             ]
    #             # reassign to VC object
    #             virtual_circuit_copy = VirtualCircuit(
    #                 name="reduced_vc",
    #                 eq=None,
    #                 profiles=None,
    #                 shape_matrix=None,
    #                 targets_options=None,
    #                 non_standard_targets=None,
    #                 targets_val=None,
    #                 coils=virtual_circuit.coils,
    #                 VCs_matrix=vc_mat_reduced,
    #                 targets=targs_reduced,
    #             )
    #     return virtual_circuit_copy

    def get_vc(self, time_stamp):
        """Retrieve VC from pre built matrices.
        Assumes columns/rows have already been reordered apropriately

        Parameters
        ----------
        time_stamp : float
            time at which VC matrix is required

        Returns
        -------
        vc_mat : np.array
            2dimensional numpy array for the VC. Columns correspond to targets and rows to coils.
        """

        # time_sch = self.get_schedule_time(time_stamp=time_stamp)
        vc_data = self.get_scheduled_params(param_type="vc", time_stamp=time_stamp)
        vc_mat = vc_data["vc_matrix"]
        return vc_mat

    def get_vc_2(
        self,
        time_stamp: float,
        target: str,
    ):
        """
        Alternative version of "get_vc" to work with vc's provided as a set of vc columns (rather than as matrix)

        Parameters
        --------
        time_stamp : float
            time in simulation to get virtual circuit
        target : name of vc needed

        Returns
        -------
        vc_col : np.array
            numpy array of single column of VC matrix
        """
        t_vc = max(
            time for time in list(self.vc_schedule_full.keys()) if time <= time_stamp
        )
        vc_phase = self.vc_schedule_full[t_vc]
        vc_col = vc_phase["vc_columns"][target]
        return vc_col

    def _validate_observable_registry(
        self, observable_registry: ObservableRegistry
    ) -> bool:
        """
        Determine if the provided observable registry satisfies the necessary
        requirements for get_vc to be executed correctly. E.g. does it provide access to
        all the physical parameters of an equilibrium needed by a model.

        This class does not require any observable registry as the VC matrices are
        already built and do not need to be computed.

        Parameters
        ----------
        observable_registry : ObservableRegistry
            The observable registry to validate.
        """
        return True

    def _validate_observable_registry(
        self, observable_registry: ObservableRegistry
    ) -> bool:
        """
        Determine if the provided observable registry satisfies the necessary
        requirements for get_vc to be executed correctly. E.g. does it provide access to
        all the physical parameters of an equilibrium needed by a model.

        This class does not require any observable registry as the VC matrices are
        already built and do not need to be computed.

        Parameters
        ----------
        observable_registry : ObservableRegistry
            The observable registry to validate.
        """
        return True


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
        waveform_dict: dict,
        schedule_dict: dict,
        vc_scheduler: VirtualCircuitProvider,
        controlled_targets_all: list[str],
        control_coils=None,
        vc_flag="file",
    ):
        """
        Initialise the class

        Parameters
        ----------
        waveform_dict : dict
            dictionary containing target waveform
        schedule_dict : dict
            dictionary containing target feedback schedule (gains, damping, fb targets)
        controlled_targs_all : list[str]
            list of all controllable targets (FB and FF). This is ideally the same as the targets in VCscheduler.
        vc_scheduler : object :VC_provider
            A VC provider, instance of VirtualCircuitProvider - either from file, emu or rtvc
        vc_flag : str   (optional)
            flag to indicate whether to load virtual circuit from file or NN
            emulator (default = "file")
            options = ["file", "Emulator"]
        Returns
        -------
        None
        """

        super().__init__(waveform_dict, schedule_dict, controlled_targets_all)

        self.vc_flag = vc_flag
        assert vc_scheduler is not None, "Please provide a vc schedule"

        # check if vc_flag is file .
        self.vc_scheduler = vc_scheduler

        if vc_flag == "file":
            # create vc schedule with reshaped vcs
            self.vc_scheduler._build_reshaped_schedule(
                coil_order=control_coils, target_order=controlled_targets_all
            )

        elif vc_flag == "emulator" or "emu" or "Emulator":
            print("Initialising an emulator scheduler")
            print("please run pre_run_emulators now")

    def get_vc(
        self,
        time_stamp: float,
        eq=None,
        profiles=None,
        coils: list[str] = None,
        targets: list[str] = None,
    ):
        """
        Get VC object given time stamp.
        - load from file if provided or compute with emulator
        All optional aguments only necessary if using Emulators to compute VC

        Parameters
        ----------
        times_stamp : float
            time at which VC is needed
        eq : FreeGSNKE equilibrium object (optional)
            equilibirum if doing dynamic simulation. provides input parameters for emulators
            if emulators being used to provide VC's
        profiles : FreeGSNKE profile object (optional)
            profiles if doing dynamic simulation. Provides input parameters for emulators
            if emulators being used to provide VC's
        coils  : list[str] (optional)
            list of coils to use for control. Provides coils to use when computing VC using emulators.
        coils  : list[str] (optional)
            list of coils to use for control. Provides coils to use when computing VC using emulators.

        Returns
        -------
        vc : VirtualCircuit object
            instance of FreeGSNKE virtual circuit object

        """

        if self.vc_flag == "file":
            print("loading VC from file")
            vc = self.vc_scheduler.get_vc(time_stamp=time_stamp)

        elif self.vc_flag == "Emulator" or "emulator" or "emu":
            assert (
                eq is not None and profiles is not None and coils is not None
            ), "Need eq, profiles and coils to compute VC"
            print("Computing VC from emulator")
            control_targs = self.get_fb_controlled_targets(time_stamp)
            vc = self.vc_scheduler_emu.build_vc(
                eq, profiles, coils=coils, targets=control_targs
            )
        return vc
