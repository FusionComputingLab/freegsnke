"""Specialised solver interface for up-down symmetry-reduced evolution."""

from .nonlinear_solve import nl_solver


class SymmetryReducedSolver(nl_solver):
    """Evolve an exactly up-down symmetric plasma using only even wall modes.

    This class provides a dedicated interface to the symmetry-reduced path in
    :class:`~freegsnke.nonlinear_solve.nl_solver`. It deliberately adds no new
    dynamics: the inherited solver remains the even-state evolution engine.
    The equilibrium properties defined here establish the interface that a
    later odd-state reconstruction can extend without changing the internal
    even equilibrium used for linearisation.

    Parameters
    ----------
    profiles, eq, GSStaticSolver
        Inputs accepted by :class:`~freegsnke.nonlinear_solve.nl_solver`.
    passive_reflection_operator : ndarray, optional
        Reflection map between the passive structures. It is required when
        passive structures are present.
    **kwargs
        Remaining keyword arguments passed to ``nl_solver``. Symmetric GS
        solves and removal of odd passive modes are enforced by this class.
    """

    def __init__(
        self,
        profiles,
        eq,
        GSStaticSolver,
        *,
        passive_reflection_operator=None,
        **kwargs,
    ):
        force_symmetric = kwargs.pop("force_up_down_symmetric", True)
        remove_odd_modes = kwargs.pop("remove_odd_passive_modes", True)
        if not force_symmetric:
            raise ValueError(
                "SymmetryReducedSolver requires 'force_up_down_symmetric=True'."
            )
        if not remove_odd_modes:
            raise ValueError(
                "SymmetryReducedSolver requires 'remove_odd_passive_modes=True'."
            )

        super().__init__(
            profiles=profiles,
            eq=eq,
            GSStaticSolver=GSStaticSolver,
            passive_reflection_operator=passive_reflection_operator,
            force_up_down_symmetric=True,
            remove_odd_passive_modes=True,
            **kwargs,
        )

    @property
    def even_equilibrium(self):
        """Return the internal even equilibrium used by the solver."""

        return self.eq1

    @property
    def even_profiles(self):
        """Return the internal profiles associated with ``even_equilibrium``."""

        return self.profiles1

    @property
    def observable_equilibrium(self):
        """Return the equilibrium intended for diagnostics and controllers.

        For purely even evolution this is the internal equilibrium itself. A
        later even-plus-odd implementation can override this property while
        leaving ``even_equilibrium`` untouched.
        """

        return self.even_equilibrium

    @property
    def observable_profiles(self):
        """Return profiles associated with ``observable_equilibrium``."""

        return self.even_profiles

    @property
    def measurement_equilibrium(self):
        """Return the equilibrium from which controller measurements are made."""

        return self.observable_equilibrium
