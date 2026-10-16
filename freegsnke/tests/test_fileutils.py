"""
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

from freegsnke import _fileutils

try:
    from io import StringIO
except:
    # Python 2
    from StringIO import StringIO


def test_f2s():
    assert _fileutils.f2s(0.0) == " 0.000000000E+00"
    assert _fileutils.f2s(1234) == " 1.234000000E+03"
    assert _fileutils.f2s(-1.65281e12) == "-1.652810000E+12"
    assert _fileutils.f2s(-1.65281e-2) == "-1.652810000E-02"


def test_ChunkOutput():
    output = StringIO()
    co = _fileutils.ChunkOutput(output)

    for val in [1.0, -3.2, 6.2e5, 8.7654e-12, 42.0, -76]:
        co.write(val)

    assert (
        output.getvalue()
        == """ 1.000000000E+00-3.200000000E+00 6.200000000E+05 8.765400000E-12 4.200000000E+01
   -76"""
    )
