"""
Expose DivGeo equilibrium export through the historical import path.

Copyright 2018-2021 Ben Dudson, University of York. Email: benjamin.dudson@york.ac.uk

Derived from FreeGS through FreeGS4E.

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

from typing import TextIO

from . import _divgeo, geqdsk


def write(eq: object, fh: TextIO, oxpoints: object | None = None) -> None:
    """
    Write a DivGeo equilibrium file for an equilibrium.

    Parameters
    ----------
    eq : object
        FreeGSNKE Equilibrium object.
    fh : TextIO
        Open, writable text file handle.
    oxpoints : object, optional
        Pre-computed O- and X-points, passed to ``geqdsk.write``. If None,
        they are found from the equilibrium.
    """

    geqdsk.write(eq, fh, oxpoints=oxpoints, fileformat=_divgeo.write)
