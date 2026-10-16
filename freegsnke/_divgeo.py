"""
Low level routines for reading and writing DivGeo files

Copyright 2018 Ben Dudson, University of York. Email: benjamin.dudson@york.ac.uk

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

from __future__ import annotations

from typing import TextIO

import numpy as np

from ._fileutils import ChunkOutput, write_1d, write_2d


def write(data: dict, fh: TextIO, label: str | None = None) -> None:
    """
    Write a DivGeo file, given a dictionary of data.

    Parameters
    ----------
    data : dict
        Equilibrium data with keys:

        - ``nx``, ``ny``: number of points in R (x) and Z (y)
        - ``rcentr``: reference value of R
        - ``bcentr``: vacuum toroidal magnetic field at ``rcentr``
        - ``sibdry``: poloidal flux psi at the plasma boundary
        - ``psi``: 2D array of poloidal flux, shape (nx, ny)
        - either ``r`` (1D array of R values, length nx) or ``rdim``
          (size of the R dimension) and ``rleft`` (innermost R point)
        - either ``z`` (1D array of Z values, length ny) or ``zdim``
          (size of the Z dimension) and ``zmid`` (middle of the Z dimension)
    fh : TextIO
        Open, writable text file handle.
    label : str, optional
        Unused; present for interface compatibility with the other writers.
    """

    # Write header
    fh.write(
        """    jm   :=  no. of grid points in radial direction;
    km   :=  no. of grid points in vertical direction;
    r    :=  radial   coordinates of grid points  [m];
    z    :=  vertical coordinates of grid points  [m];
    psi  :=  flux per radiant at grid points     [Wb/rad];
    psib :=  psi at plasma boundary              [Wb/rad];
    btf  :=  toroidal magnetic field                  [t];
    rtf  :=  major radius at which btf is specified   [m];


    jm    =          {nx:d} ;
    km    =          {ny:d} ;
    psib  =   {sibdry:1.15E}  Wb/rad;
    btf   =   {bcentr:1.14f}       t;
    rtf   =   {rcentr:1.14f}       m;

""".format(
            **data
        )
    )

    try:
        r = data["r"]
    except KeyError:
        # No "r" in the dictionary
        # use rdim, rleft and nx (from eqdsk)

        Rmin = data["rleft"]
        Rmax = data["rleft"] + data["rdim"]
        nx = data["nx"]

        dR = (Rmax - Rmin) / (nx - 1)
        r = np.arange(nx) * dR + Rmin

    try:
        z = data["z"]
    except KeyError:
        # No "z" in the dictionary
        # use zdim, zmid and ny (from eqdsk)

        Zmin = data["zmid"] - 0.5 * data["zdim"]
        Zmax = data["zmid"] + 0.5 * data["zdim"]
        ny = data["ny"]

        dZ = (Zmax - Zmin) / (ny - 1)
        z = np.arange(ny) * dZ + Zmin

    # Now write r and z
    fh.write("    r(1:jm);\n")
    co = ChunkOutput(fh, chunksize=5, extraspaces=3)
    write_1d(r, co)
    co.newline()

    fh.write(" \n    z(1:km);\n")
    write_1d(z, co)
    co.newline()

    fh.write(" \n      ((psi(j,k)-psib,j=1,jm),k=1,km)\n")

    write_2d(data["psi"] - data["sibdry"], co)
    co.newline()
