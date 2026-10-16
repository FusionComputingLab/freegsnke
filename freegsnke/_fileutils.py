"""
Utilities for writing and reading files compatible with Fortran

Copyright 2018-2021 Ben Dudson, University of York. Email: benjamin.dudson@york.ac.uk
Copyright 2018 Peter Hill, Bhavin Patel
Copyright 2018, 2021 Lukas Kripner

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

import re


def f2s(f):
    """
    Format a string containing a float
    """
    s = ""
    if f >= 0.0:
        s += " "
    return s + "%1.9E" % f


class ChunkOutput:
    """
    This outputs values in lines, inserting
    newlines when needed.
    """

    def __init__(self, filehandle, chunksize=5, extraspaces=0):
        """
        filehandle  output to write to
        chunksize   number of values on a line
        extraspaces  number of extra spaces between outputs
        """
        self.fh = filehandle
        self.counter = 0
        self.chunk = chunksize
        self.extraspaces = extraspaces

    def write(self, value):
        """ "
        Write a value to the output, adding a newline if needed

        Distinguishes between:
        - list  : Iterates over the list and writes each element
        - int   : Converts using str
        - float : Converts using f2s to Fortran-formatted string

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

    def newline(self):
        """
        Ensure that the file is at the start of a new line
        """
        if self.counter != 0:
            self.fh.write("\n")
            self.counter = 0

    def endblock(self):
        """
        Make sure next block of data is on new line
        """
        self.fh.write("\n")
        self.counter = 0

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        """Ensure that the chunk finishes with a new line"""
        if self.counter != 0:
            self.counter = 0
            self.fh.write("\n")


def write_1d(val, out):
    """
    Writes a 1D variable val to the file handle out
    """
    for i in range(len(val)):
        out.write(val[i])
    out.newline()


def write_2d(val, out):
    """
    Writes a 2D array. Note that this transposes
    the array, looping over the first index fastest
    """
    nx, ny = val.shape
    for y in range(ny):
        for x in range(nx):
            out.write(val[x, y])
    out.newline()


def next_value(fh):
    """
    A generator which yields values from a file handle

    Checks if the value is a float or int, returning
    the correct type depending on if '.' is in the string

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
