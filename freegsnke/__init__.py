"""
FreeGSNKE package.

Provides tools for solving Grad-Shafranov equilibria and related
numerical methods for plasma physics modelling.

License & Attribution
---------------------
FreeGSNKE is distributed under the GNU Lesser General Public License version 3.
Copyright 2025 UKRI-STFC and UKAEA contributors.

The integrated equilibrium and numerical modules include contributions
(Copyright 2024-2025 Nicola C. Amorisco, George K. Holt, Adriano Agnello, and other contributors)
with solver and parallel-execution contributions by Tomas Rubio Cruz and Ubaid Qadri (2026),
and FreeGS (Copyright 2016-2021 Ben Dudson, University of York, and other contributors).
"""

import importlib.metadata

__version__ = importlib.metadata.version("freegsnke")
__author__ = "The FreeGSNKE Developers"

from . import (
    boundary,
    coil,
    control,
    critical,
    dump,
    equilibrium,
    gradshafranov,
    jtor,
    machine,
    multi_coil,
    multigrid,
    optimise,
    optimiser,
    picard,
    plotting,
    polygons,
    quadrature,
    shaped_coil,
)
from .dump import OutputFile
from .equilibrium import Equilibrium
from .picard import solve
