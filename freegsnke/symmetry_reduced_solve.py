"""
Specialised solver interface for up-down symmetry-reduced evolution.

Copyright 2025 UKAEA, UKRI-STFC, and The Authors, as per the COPYRIGHT and
README files.

This file is part of FreeGSNKE.

FreeGSNKE is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FreeGSNKE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with FreeGSNKE. If not, see <http://www.gnu.org/licenses/>.
"""

from __future__ import annotations

from typing import Any

import numpy as np

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
    During initialisation, ``eq._vgreen`` (and any un-mirrored passive filament
    refinements on ``eq.tokamak``) are reflection-symmetrised in place after
    validating the grid, limiter mask, and reflection operator.

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
        profiles: object,
        eq: object,
        GSStaticSolver: object,
        *,
        passive_reflection_operator: np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialise the symmetry-reduced evolutive solver.

        Parameters
        ----------
        profiles : object
            Profile object accepted by :class:`~freegsnke.nonlinear_solve.nl_solver`.
        eq : object
            Up-down symmetric equilibrium object.
        GSStaticSolver : object
            Static Grad-Shafranov solver instance.
        passive_reflection_operator : np.ndarray, optional
            Reflection permutation matrix between passive structures; required
            when passive structures are present.
        **kwargs : Any
            Additional keyword arguments forwarded to
            :class:`~freegsnke.nonlinear_solve.nl_solver`.
        """
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
