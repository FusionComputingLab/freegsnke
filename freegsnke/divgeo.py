"""Expose DivGeo equilibrium export through the historical import path.

Copyright 2018-2021 Ben Dudson, University of York. Email: benjamin.dudson@york.ac.uk

Derived from FreeGS through FreeGS4E.

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

from . import _divgeo, geqdsk


def write(eq, fh, oxpoints=None):
    """
    Write a DivGeo equilibrium file, given a FreeGS Equilibrium object

    eq - Equilibrium object
    fh - file handle
    """

    geqdsk.write(eq, fh, oxpoints=oxpoints, fileformat=_divgeo.write)
