"""Lightweight checks for the backend surface consumed by FreeGSNKE.

Copyright 2026 Nicola C. Amorisco

Adapted from FreeGS4E.

SPDX-License-Identifier: LGPL-3.0-or-later

This file is part of FreeGSNKE.

FreeGSNKE is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FreeGSNKE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with FreeGSNKE.  If not, see <http://www.gnu.org/licenses/>.
"""

from freegsnke import (
    bilinear_interpolation,
    critical,
    gradshafranov,
    multigrid,
    plotting,
)
from freegsnke.coil import Coil
from freegsnke.equilibrium import Equilibrium
from freegsnke.jtor import (
    ConstrainBetapIp,
    ConstrainPaxisIp,
    Fiesta_Topeol,
    GeneralPprimeFFprime,
    Lao85,
    TensionSpline,
)
from freegsnke.machine import Circuit, Machine, Wall
from freegsnke.multi_coil import MultiCoil


def test_freegsnke_backend_contract():
    """Keep changes to the documented FreeGSNKE dependency surface visible."""
    required_callables = (
        bilinear_interpolation.biliint,
        critical.find_critical,
        critical.scan_for_crit,
        gradshafranov.Greens,
        gradshafranov.GreensBr,
        gradshafranov.GreensBz,
        gradshafranov.GreensdBrdz,
        gradshafranov.GSsparse,
        gradshafranov.GSsparse4thOrder,
        multigrid.createVcycle,
        plotting.plotConstraints,
        plotting.plotIOConstraints,
        plotting.plotProbes,
    )
    assert all(callable(item) for item in required_callables)
    assert gradshafranov.mu0 > 0

    required_types = (
        Coil,
        Circuit,
        Machine,
        MultiCoil,
        Wall,
        Equilibrium,
        ConstrainBetapIp,
        ConstrainPaxisIp,
        Fiesta_Topeol,
        GeneralPprimeFFprime,
        Lao85,
        TensionSpline,
    )
    assert all(isinstance(item, type) for item in required_types)

    required_machine_methods = (
        "calcPsiFromGreens",
        "createPsiGreens",
        "createPsiGreensVec",
        "getCurrents",
    )
    assert all(
        callable(getattr(Machine, name, None)) for name in required_machine_methods
    )

    required_equilibrium_methods = (
        "_updatePlasmaPsi",
        "Br",
        "Bz",
        "plasmaCurrent",
        "psi",
        "separatrix",
        "strikepoints",
    )
    assert all(
        callable(getattr(Equilibrium, name, None))
        for name in required_equilibrium_methods
    )
