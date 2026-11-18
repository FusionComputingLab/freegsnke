"""
Utilities for writing and reading files compatible with Fortran

Copyright 2018-2021 Ben Dudson, University of York. Email: benjamin.dudson@york.ac.uk
Copyright 2018 Peter Hill, Bhavin Patel
Copyright 2018, 2021 Lukas Kripner

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

import re
from types import TracebackType
from typing import Iterable, Iterator, Sequence, TextIO

import numpy as np


def f2s(f: float) -> str:
    """
    Format a float as a fixed-width Fortran-style scientific string.

    Parameters
    ----------
    f : float
        Value to format.

    Returns
    -------
    str
        String of the form ``" 1.234567890E+00"``; non-negative values are
        prefixed with a space so that columns line up with negative values.
    """
    s = ""
    if f >= 0.0:
        s += " "
    return s + "%1.9E" % f


class ChunkOutput:
    """
    Write values to a file in lines of a fixed number of values.

    Inserts newlines when needed.
    """

    def __init__(
        self, filehandle: TextIO, chunksize: int = 5, extraspaces: int = 0
    ) -> None:
        """
        Set up the chunked writer.

        Parameters
        ----------
        filehandle : TextIO
            Open, writable text file handle to write to.
        chunksize : int, optional
            Number of values on a line.
        extraspaces : int, optional
            Number of extra spaces written before each value.
        """
        self.fh = filehandle
        self.counter = 0
        self.chunk = chunksize
        self.extraspaces = extraspaces

    def write(self, value: int | float | list) -> None:
        """
        Write a value to the output, adding a newline if needed.

        Distinguishes between:

        - list  : iterates over the list and writes each element
        - int   : converts using ``str``
        - float : converts using ``f2s`` to a Fortran-formatted string

        Parameters
        ----------
        value : int or float or list
            Value (or list of values) to write.
        """
        if isinstance(value, list):
            for elt in value:
                self.write(elt)
            return

        self.fh.write(" " * self.extraspaces)

        if isinstance(value, int):
            self.fh.write("   " + str(value))
        else:
            self.fh.write(f2s(value))

        self.counter += 1
        if self.counter == self.chunk:
            self.fh.write("\n")
            self.counter = 0

    def newline(self) -> None:
        """Ensure that the file is at the start of a new line."""
        if self.counter != 0:
            self.fh.write("\n")
            self.counter = 0

    def endblock(self) -> None:
        """Make sure the next block of data starts on a new line."""
        self.fh.write("\n")
        self.counter = 0

    def __enter__(self) -> ChunkOutput:
        """
        Enter the context manager.

        Returns
        -------
        ChunkOutput
            This instance.
        """
        return self

    def __exit__(
        self,
        type: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """
        Ensure that the chunk finishes with a new line.

        Parameters
        ----------
        type : type[BaseException] or None
            Exception type, if one was raised in the ``with`` block.
        value : BaseException or None
            Exception instance, if one was raised.
        traceback : TracebackType or None
            Traceback, if an exception was raised.
        """
        if self.counter != 0:
            self.counter = 0
            self.fh.write("\n")


def write_1d(val: Sequence[float], out: ChunkOutput) -> None:
    """
    Write a 1D variable to a chunked output.

    Parameters
    ----------
    val : Sequence[float]
        Values to write.
    out : ChunkOutput
        Output to write to; a newline is ensured afterwards.
    """
    for i in range(len(val)):
        out.write(val[i])
    out.newline()


def write_2d(val: np.ndarray, out: ChunkOutput) -> None:
    """
    Write a 2D array, looping over the first index fastest.

    Note that this transposes the array on output.

    Parameters
    ----------
    val : np.ndarray
        Array of values, shape (nx, ny).
    out : ChunkOutput
        Output to write to; a newline is ensured afterwards.
    """
    nx, ny = val.shape
    for y in range(ny):
        for x in range(nx):
            out.write(val[x, y])
    out.newline()


def next_value(fh: Iterable[str]) -> Iterator[int | float]:
    """
    Yield the numeric values found in a file, one at a time.

    Checks if each value is a float or int, returning the correct type
    depending on whether '.' is in the matched string.

    Parameters
    ----------
    fh : Iterable[str]
        Open text file handle (or any iterable of lines).

    Yields
    ------
    int or float
        Next number found in the file.
    """
    pattern = re.compile(r"[ +\-]?\d+(?:\.\d+(?:[Ee][\+\-]\d\d)?)?")

    # Go through each line, extract values, then yield them one by one
    for line in fh:
        matches = pattern.findall(line)
        for match in matches:
            if "." in match:
                yield float(match)
            else:
                yield int(match)
