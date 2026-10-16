"""
Low level routines for reading and writing G-EQDSK files

Copyright 2016 Ben Dudson, University of York. Email: benjamin.dudson@york.ac.uk

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

from datetime import date
from typing import TextIO

import numpy as np
from numpy import pi, zeros

from ._fileutils import ChunkOutput, f2s, next_value, write_1d, write_2d


def write(
    data: dict,
    fh: TextIO,
    label: str | None = None,
    shot: int | str | None = None,
    time: int | str | None = None,
) -> None:
    """
    Write a GEQDSK equilibrium file, given a dictionary of data.

    Parameters
    ----------
    data : dict
        Equilibrium data with keys:

        - ``nx``, ``ny``: number of points in R (x) and Z (y)
        - ``rdim``, ``zdim``: sizes of the R and Z dimensions
        - ``rcentr``: reference value of R
        - ``bcentr``: vacuum toroidal magnetic field at ``rcentr``
        - ``rleft``: R at the left (inner) boundary
        - ``zmid``: Z at the middle of the domain
        - ``rmagx``, ``zmagx``: R, Z at the magnetic axis (O-point)
        - ``simagx``: poloidal flux psi at the magnetic axis
        - ``sibdry``: poloidal flux psi at the plasma boundary
        - ``cpasma``: plasma current [A]
        - ``fpol``: 1D array of f(psi)=R*Bt [m T]
        - ``pres``: 1D array of p(psi) [Pa]
        - ``qpsi``: 1D array of q(psi)
        - ``psi``: 2D array of poloidal flux, shape (nx, ny)
        - ``ffprime``, ``pprime`` (optional): 1D arrays of ff'(psi) and
          p'(psi); zeros are written if absent
        - ``rbdry``, ``zbdry`` (optional): plasma boundary points
        - ``rlim``, ``zlim`` (optional): limiter/wall points
    fh : TextIO
        Open, writable text file handle.
    label : str, optional
        Text label to put in the file (default ``"FREEGS"``); truncated if
        too long.
    shot : int or str, optional
        Shot number, or a ready-formatted string. Default 0.
    time : int or str, optional
        Time in ms, or a ready-formatted string. Default 0.
    """

    nx = data["nx"]
    ny = data["ny"]

    if not label:
        label = "FREEGS"
    if len(label) > 11:
        label = label[0:12]
        print("WARNING: label too long, it will be shortened to {}".format(label))

    creation_date = date.today().strftime("%d/%m/%Y")

    if not shot:
        shot = 0

    if isinstance(shot, int):
        shot = "# {:d}".format(shot)

    if not time:
        time = 0

    if isinstance(time, int):
        time = "  {:d}ms".format(time)

    # I have no idea what idum is, here it is set to 3
    idum = 3
    header = "{0:11s}{1:10s}   {2:>8s}{3:16s}{4:4d}{5:4d}{6:4d}\n".format(
        label, creation_date, shot, time, idum, nx, ny
    )

    # First line: Identification string, followed by resolution
    fh.write(header)

    # Second line
    fh.write(
        f2s(data["rdim"])
        + f2s(data["zdim"])
        + f2s(data["rcentr"])
        + f2s(data["rleft"])
        + f2s(data["zmid"])
        + "\n"
    )

    # Third line
    fh.write(
        f2s(data["rmagx"])
        + f2s(data["zmagx"])
        + f2s(data["simagx"])
        + f2s(data["sibdry"])
        + f2s(data["bcentr"])
        + "\n"
    )

    # 4th line
    fh.write(
        f2s(data["cpasma"])
        + f2s(data["simagx"])
        + f2s(0.0)
        + f2s(data["rmagx"])
        + f2s(0.0)
        + "\n"
    )

    # 5th line
    fh.write(
        f2s(data["zmagx"]) + f2s(0.0) + f2s(data["sibdry"]) + f2s(0.0) + f2s(0.0) + "\n"
    )

    # SCENE has actual ff' and p' data so can use that
    # fill arrays
    # Lukas Kripner (16/10/2018): uncommenting this, since you left there
    # check for data existence bellow. This seems to as safer variant.
    workk = zeros([nx])

    # Write arrays
    co = ChunkOutput(fh)

    write_1d(data["fpol"], co)
    write_1d(data["pres"], co)
    if "ffprime" in data:
        write_1d(data["ffprime"], co)
    else:
        write_1d(workk, co)
    if "pprime" in data:
        write_1d(data["pprime"], co)
    else:
        write_1d(workk, co)

    write_2d(data["psi"], co)
    write_1d(data["qpsi"], co)

    # Boundary / limiters
    nbdry = 0
    nlim = 0
    if "rbdry" in data:
        nbdry = len(data["rbdry"])
    if "rlim" in data:
        nlim = len(data["rlim"])

    co.newline()
    fh.write("{0:5d}{1:5d}\n".format(nbdry, nlim))

    if nbdry > 0:
        for r, z in zip(data["rbdry"], data["zbdry"]):
            co.write(r)
            co.write(z)
        co.newline()

    if nlim > 0:
        for r, z in zip(data["rlim"], data["zlim"]):
            co.write(r)
            co.write(z)
        co.newline()


def read(fh: TextIO, cocos: int = 1) -> dict:
    """
    Read a G-EQDSK formatted equilibrium file.

    Format is specified here: https://fusion.gat.com/theory/Efitgeqdsk

    Parameters
    ----------
    fh : TextIO
        Open, readable text file handle.
    cocos : int, optional
        COordinate COnventions. Not fully handled yet, only whether psi is
        divided by 2pi or not: if > 10 then psi is divided by 2pi (and
        ``pprime`` and ``ffprime`` multiplied by 2pi), otherwise not.

    Returns
    -------
    dict
        Equilibrium data with keys:

        - ``nx``, ``ny``: number of points in R (x) and Z (y)
        - ``rdim``, ``zdim``: sizes of the R and Z dimensions
        - ``rcentr``: reference value of R
        - ``bcentr``: vacuum toroidal magnetic field at ``rcentr``
        - ``rleft``: R at the left (inner) boundary
        - ``zmid``: Z at the middle of the domain
        - ``rmagx``, ``zmagx``: R, Z at the magnetic axis (O-point)
        - ``simagx``: poloidal flux psi at the magnetic axis
        - ``sibdry``: poloidal flux psi at the plasma boundary
        - ``cpasma``: plasma current [A]
        - ``fpol``: 1D array of f(psi)=R*Bt [m T]
        - ``pres``: 1D array of p(psi) [Pa]
        - ``qpsi``: 1D array of q(psi)
        - ``psi``: 2D array of poloidal flux, shape (nx, ny)
        - ``ffprime``, ``pprime``: 1D arrays of ff'(psi) and p'(psi)
        - ``rbdry``, ``zbdry``: plasma boundary points (if present)
        - ``rlim``, ``zlim``: limiter/wall points (if present)

    Raises
    ------
    ValueError
        If the first line contains fewer than 3 numbers.
    """

    # Read the first line
    header = fh.readline()
    words = header.split()  # Split on whitespace
    if len(words) < 3:
        raise ValueError("Expecting at least 3 numbers on first line")

    nx = int(words[-2])
    ny = int(words[-1])

    print("  nx = {0}, ny = {1}".format(nx, ny))

    # Dictionary to hold result
    data = {"nx": nx, "ny": ny}

    # List of fields to read. None means discard value
    fields = [
        "rdim",
        "zdim",
        "rcentr",
        "rleft",
        "zmid",
        "rmagx",
        "zmagx",
        "simagx",
        "sibdry",
        "bcentr",
        "cpasma",
        "simagx",
        None,
        "rmagx",
        None,
        "zmagx",
        None,
        "sibdry",
        None,
        None,
    ]

    values = next_value(fh)

    for f in fields:
        val = next(values)
        if f:
            data[f] = val

    # Read arrays

    def read_1d(n: int) -> np.ndarray:
        """
        Read a 1D array of length n.

        Parameters
        ----------
        n : int
            Number of values to read.

        Returns
        -------
        np.ndarray
            Array of shape (n,).
        """
        val = zeros(n)
        for i in range(n):
            val[i] = next(values)
        return val

    def read_2d(n: int, m: int) -> np.ndarray:
        """
        Read a 2D (n, m) array in Fortran order.

        Parameters
        ----------
        n : int
            Size of the first axis.
        m : int
            Size of the second axis.

        Returns
        -------
        np.ndarray
            Array of shape (n, m).
        """
        val = zeros([n, m])
        for y in range(m):
            for x in range(n):
                val[x, y] = next(values)
        return val

    data["fpol"] = read_1d(nx)
    data["pres"] = read_1d(nx)
    data["ffprime"] = read_1d(nx)
    data["pprime"] = read_1d(nx)

    data["psi"] = read_2d(nx, ny)

    data["qpsi"] = read_1d(nx)

    # Ensure that psi is divided by 2pi
    if cocos > 10:
        for var in ["psi", "simagx", "sibdry"]:
            data[var] /= 2 * pi
        # pprime and ffprime are derivatives with respect to psi. If the
        # flux coordinate is divided by 2pi, derivatives with respect to that
        # coordinate must be multiplied by 2pi.
        for var in ["pprime", "ffprime"]:
            data[var] *= 2 * pi

    nbdry = next(values)
    nlim = next(values)

    # print(nbdry, nlim)

    if nbdry > 0:
        # Read (R,Z) pairs
        print(nbdry)
        data["rbdry"] = zeros(nbdry)
        data["zbdry"] = zeros(nbdry)
        for i in range(nbdry):
            data["rbdry"][i] = next(values)
            data["zbdry"][i] = next(values)

    if nlim > 0:
        # Read (R,Z) pairs
        data["rlim"] = zeros(nlim)
        data["zlim"] = zeros(nlim)
        for i in range(nlim):
            data["rlim"][i] = next(values)
            data["zlim"][i] = next(values)

    return data
