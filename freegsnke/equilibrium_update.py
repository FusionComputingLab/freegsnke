import freegs
import numpy as np
import pickle


class Equilibrium(freegs.equilibrium.Equilibrium):
    """FreeGS equilibrium class with optional initialization."""

    def __init__(self, *args, **kwargs):
        """Instantiates the object."""
        super().__init__(*args, **kwargs)

        equilibrium_path = os.environ.get("EQUILIBRIUM_PATH", None)
        if equilibrium_path is not None:
            self.initialize_from_equilibrium()

    def initialize_from_equilibrium(
        self,
    ):
        """Initilizes the equilibrium with data from file"""
        with open(equilibrium_path, "rb") as f:
            equilibrium_data = pickle.load(f)

    def assign_profile_parameter(self, betap):
        """Assigns to the profile object a new value of the profile parameter betap"""
        self.betap = betap
        self.profile_parameter = betap

    def assign_profile_coefficients(self, alpha_m, alpha_n):
        """Assigns to the profile object new value of the coefficients (alpha_m, alpha_n)"""
        self.alpha_m = alpha_m
        self.alpha_n = alpha_n

    def _Jtor(self, R, Z, psi, psi_bndry=None, rel_psi_error=0):
        """Replaces the original FreeGS Jtor method if FreeGSfast is not available."""
        self.jtor = super().Jtor(R, Z, psi, psi_bndry)
        self.opt, self.xpt = critical.find_critical(R, Z, psi)

        self.diverted_core_mask = self.jtor > 0
        self.psi_bndry, mask, self.limiter_flag = (
            self.limiter_handler.core_mask_limiter(
                psi,
                self.xpt[0][2],
                self.diverted_core_mask,
                self.limiter_mask_out,
            )
        )
        self.jtor = super().Jtor(R, Z, psi, self.psi_bndry)
        return self.jtor

    def Jtor_fast(self, R, Z, psi, psi_bndry=None, rel_psi_error=0):
        """Used when FreeGSfast is available."""
        self.diverted_core_mask = super().Jtor_part1(R, Z, psi, psi_bndry)
        if self.diverted_core_mask is None:
            # print('no xpt')
            self.psi_bndry, self.limiter_core_mask, self.flag_limiter = (
                psi_bndry,
                None,
                False,
            )
        elif rel_psi_error < 0.02:
            self.psi_bndry, self.limiter_core_mask, self.flag_limiter = (
                self.limiter_handler.core_mask_limiter(
                    psi,
                    self.psi_bndry,
                    self.diverted_core_mask,
                    self.limiter_mask_out,
                )
            )
        else:
            self.limiter_core_mask = self.diverted_core_mask.copy()
        self.jtor = super().Jtor_part2(
            R, Z, psi, self.psi_bndry, self.limiter_core_mask
        )
        return self.jtor


class ConstrainPaxisIp(freegs.jtor.ConstrainPaxisIp):
    """FreeGS profile class with a few modifications, to:
    - retain memory of critical point calculation;
    - deal with limiter plasma configurations

    """

    def __init__(self, eq, limiter, *args, **kwargs):
        """Instantiates the object.

        Parameters
        ----------
        eq : freeGS Equilibrium object
            Specifies the domain properties
        limiter : freeGS.machine.Wall object
            Specifies the limiter contour points
        """
        super().__init__(*args, **kwargs)
        self.profile_parameter = self.paxis

        self.limiter_handler = limiter_func.Limiter_handler(eq, limiter)
        self.limiter_mask_out = plasma_grids.make_layer_mask(
            self.limiter_handler.mask_inside_limiter, layer_size=1
        )
        self.mask_inside_limiter = self.limiter_handler.mask_inside_limiter
        self.limiter_mask_for_plotting = (
            self.mask_inside_limiter + self.limiter_mask_out
        ) > 0

        if not hasattr(self, "fast"):
            self.Jtor = self._Jtor
        else:
            self.Jtor = self.Jtor_fast

    def assign_profile_parameter(self, paxis):
        """Assigns to the profile object a new value of the profile parameter paxis"""
        self.paxis = paxis
        self.profile_parameter = paxis

    def assign_profile_coefficients(self, alpha_m, alpha_n):
        """Assigns to the profile object new value of the coefficients (alpha_m, alpha_n)"""
        self.alpha_m = alpha_m
        self.alpha_n = alpha_n

    def _Jtor(self, R, Z, psi, psi_bndry=None, rel_psi_error=0):
        """Replaces the original FreeGS Jtor method if FreeGSfast is not available."""
        self.jtor = super().Jtor(R, Z, psi, psi_bndry)
        self.opt, self.xpt = critical.find_critical(R, Z, psi)

        self.diverted_core_mask = self.jtor > 0
        self.psi_bndry, mask, self.limiter_flag = (
            self.limiter_handler.core_mask_limiter(
                psi,
                self.xpt[0][2],
                self.diverted_core_mask,
                self.limiter_mask_out,
            )
        )
        self.jtor = super().Jtor(R, Z, psi, self.psi_bndry)
        return self.jtor

    def Jtor_fast(self, R, Z, psi, psi_bndry=None, rel_psi_error=0):
        """Used when FreeGSfast is available."""
        self.diverted_core_mask = super().Jtor_part1(R, Z, psi, psi_bndry)
        if self.diverted_core_mask is None:
            # print('no xpt')
            self.psi_bndry, self.limiter_core_mask, self.flag_limiter = (
                psi_bndry,
                None,
                False,
            )
        elif rel_psi_error < 0.02:
            self.psi_bndry, self.limiter_core_mask, self.flag_limiter = (
                self.limiter_handler.core_mask_limiter(
                    psi,
                    self.psi_bndry,
                    self.diverted_core_mask,
                    self.limiter_mask_out,
                )
            )
        else:
            self.limiter_core_mask = self.diverted_core_mask.copy()

        self.jtor = super().Jtor_part2(
            R, Z, psi, self.psi_bndry, self.limiter_core_mask
        )
        return self.jtor
