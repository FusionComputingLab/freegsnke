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

    def __init__(self, vc_schedule_dict):
        """
        Initialise the class

        Parameters
        ----------
        vc_schedule_dict : str
            vc_schedule_dict to the file containing VC's. Include file
            extension either hdf5 or pkl.

        Returns
        -------
        None
        """

        self.vc_schedule_full = vc_schedule_dict
        print(f"VC schedule with {len(self.vc_schedule_full)} vc phases")

        # unpack vc_schedule_dict
        self.unpack_vc_schedule(vc_schedule_dict)

    def unpack_vc_schedule(self, vcs_dict):
        """
        Load the virtual circuit matrix, shape matrix, coils and targets from a
        dictionary, and save a list of VC objects and assocated data as class attributes.

        Returns
        -------
        None :
            Modifies the attributes of the class.
        """
        # set lists
        self.vc_times_start = []  # times at which vcs are to be stopped using
        self.vc_ojbects = []  # list of virtual circuit ojbects
        self.phase_names = []  # list of phase names

        for key, item in vcs_dict.items():
            phase_name = item["phase_name"]
            time_start = item["time_start"]
            if "vc_matrix" in item.keys():
                vc_matrix = item["vc_matrix"]
                shape_matrix = item["shape_matrix"]
                targets = item["targets"]
                coils = item["coils"]
                # create VC object
                vc_object = VirtualCircuit(
                    name=f"vc_start_{time_start:.4f}",
                    eq=None,
                    profiles=None,
                    shape_matrix=shape_matrix,
                    VCs_matrix=vc_matrix,
                    targets=targets,
                    coils=coils,
                    targets_val=None,
                    targets_options=None,
                    non_standard_targets=None,
                )
            else:
                vc_matrix = None
                vc_object = None
                shape_matrix = None
                targets = None
                coils = None

            self.vc_ojbects.append(vc_object)
            self.vc_times_start.append(time_start)
            self.phase_names.append(phase_name)

        # convert times to numpy array
        self.vc_times_start = np.array(self.vc_times_start)

    # ???? Do we need this???? Maybe delete this method.
    # def add_vc_to_sequence(self, virtual_circuit, time_start, time_start):
    #     """
    #     Add virtual circuit to sequence.

    #     Parameters
    #     ----------
    #     virtual_circuit : object
    #         virtual circuit object
    #     time_stamp : float
    #         time stamp of the virtual circuit

    #     Returns
    #     -------
    #     None
    #         modifies object in place
    #     """
    #     print("adding vc to sequence")
    #     self.vc_times_start.append(time_start)
    #     self.vc_schedule.append(virtual_circuit)
    #     # update vc time dictionary
    #     self.vc_time_start_dict = {
    #         time: ind for ind, time in enumerate(self.vc_times_stop)
    #     }

    #     # update other parts such as vc_index, input currents, profile pars etc

    def get_vc(self, time_stamp):
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
        # find time position corresponding to vc to be used.
        t_vc = max(time for time in self.vc_times_start if time <= time_stamp)
        # get index corresponding to the time position
        pos = np.where(self.vc_times_start == t_vc)[0][0]

        virtual_circuit = self.vc_ojbects[pos]
        return virtual_circuit


def pre_run_emulators(vcg, stepping, targets, coils):
    """pre run emulators on given equilibrium and set of targets/coils for speed up later
        Run this after init.

    Inputs :
    --------
    vcg : object
        virtual circuit generator object (from freegsnke emu)
    stepping : object
        stepping object
    targets : list[str]
        list of targets to be used for shape control
    coils : list[str]
        list of coils to be used for shape control

    Returns
    -------
    None
    """

    vcg.build_vc(
        eq=stepping.eq1,
        profiles=stepping.profiles1,
        targets=targets,
        coils=coils,
    )


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
        waveform_dict,
        schedule_dict,
        vc_scheduler,
        vc_flag="file",
    ):
        """
        Initialise the class

        Parameters
        ----------
        waveform_dict : dict
            dictionary containing target waveform
        schedule_dict : dict
            dictionary containing target schedule
        vc_flag : str   (optional)
            flag to indicate whether to load virtual circuit from file or NN
            emulator (default = "file")
            options = ["file", "Emulator"]
        vc_scheduler : object
            virtual circuit scheduler object - either a VirtualCircuitScheduler (file)  or VCG (emulator)

        Returns
        -------
        None
        """

        super().__init__(waveform_dict, schedule_dict)

        self.vc_flag = vc_flag
        assert vc_scheduler is not None, "Please provide a vc schedule"

        # check if vc_flag is file .
        if vc_flag == "file":
            self.vc_scheduler = vc_scheduler
            # check consistency of target schedule and vc schedule
            # merge the time sequence from both target and vc, and check the
            # targets match at each midpiont.
            print("checking target schedule and vc sequence")
            print("vc schedule times ", self.vc_scheduler.vc_times_start)
            print("target schedule times ", self.target_schedule_dict.keys())
            change_times = np.sort(
                np.concatenate(
                    (
                        list(self.target_schedule_dict.keys()),
                        self.vc_scheduler.vc_times_start,
                    )
                )
            )
            midpoints = (change_times[:-1] + change_times[1:]) / 2
            # for _, midpoint in enumerate(midpoints):
            # ####### TODO MODIFY this compatibilty check to be more robust
            # for midpoint in midpoints:
            #     print(
            #         "checking compatibility of target schedule and vc"
            #         f" sequence at time {midpoint}"
            #     )
            #     # print("vc check at time", midpoint)
            #     vc_targs = self.vc_scheduler.retrieve_vc(time_stamp=midpoint).targets
            #     # print("vc_targs", vc_targs)
            #     controlled_targs = self.retrieve_controlled_targets(time_stamp=midpoint)
            #     # print("controlled_targs", controlled_targs)
            #     # check that the target schedule is a subset of the vc sequence
            #     if not set(controlled_targs).issubset(set(vc_targs)):
            #         raise ValueError(
            #             "targets scheduled for control not a subset of vc "
            #             f"computable targets at time {midpoint} ",
            #         )
            #     elif controlled_targs != vc_targs:
            #         # check the order of the targets
            #         print(
            #             "targets requested and vc available targets do not match : vc's will be recomputed as necessary"
            #         )
            #         # print("controlled targets", controlled_targs)
            #         # print("VC available targets", vc_targs)

        elif vc_flag == "emulator" or "emu" or "Emulator":
            print("Initialising an emulator scheduler")
            self.vc_scheduler = vc_scheduler
            print("please run pre_run_emulators now")

    # def get_shape_blends(self, targets, time_stamp):
    #     """
    #     Retrieves the blends for the target at time_stamp

    #     Parameters
    #     ----------
    #     targets : list[str]
    #     time_stamp : float
    #         time stamp of the target to be retrieved

    #     Returns
    #     -------
    #     blends : dict
    #         dictionary of blends for the target at time_stamp
    #     """
    #     blend_arr = []
    #     print("blend dict ", self.blends.keys())
    #     for target in targets:
    #         blend_arr.append(
    #             self.get_waveform_value(
    #                 param_dict=self.blends,
    #                 param=target,
    #                 time_stamp=time_stamp,
    #             )
    #         )
    #     print("blends", blend_arr)
    #     return np.array(blend_arr)

    def get_vc(self, time_stamp, eq=None, profiles=None, coils=None):
        """
        Get VC object given time stamp.
        - load from file if provided or compute with emulator
        """

        if self.vc_flag == "file":
            print("loading VC from file")
            vc = self.vc_scheduler.get_vc(time_stamp=time_stamp)

        elif self.vc_flag == "Emulator" or "emulator" or "emu":
            assert (
                eq is not None and profiles is not None and coils is not None
            ), "Need eq, profiles and coils to compute VC"
            print("Computing VC from emulator")
            control_targs = self.retrieve_controlled_targets(time_stamp)
            vc = self.vc_scheduler_emu.build_vc(
                eq, profiles, coils=coils, targets=control_targs
            )
        return vc
