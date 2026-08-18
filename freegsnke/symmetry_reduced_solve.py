"""Specialised solver interface for up-down symmetry-reduced evolution."""

from .nonlinear_solve import nl_solver


class SymmetryReducedSolver(nl_solver):
    """Evolve an exactly up-down symmetric plasma using even passive modes.

    This class provides a dedicated interface to the symmetry-reduced path in
    :class:`~freegsnke.nonlinear_solve.nl_solver`. It adds no new dynamics: the
    inherited solver remains the even-state evolution engine.

    Around an even plasma in an even machine, the linearised evolution operator
    commutes with up-down reflection. Its even and odd state subspaces are
    therefore orthogonal and dynamically decoupled. This solver retains the
    even subspace and projects every Grad-Shafranov response back onto it.

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
