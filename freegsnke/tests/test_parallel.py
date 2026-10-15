"""
Copyright 2026 Tomas Rubio Cruz, STFC - Hartree Centre

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

import numpy as np
import pytest
from numpy.exceptions import AxisError
from scipy.special import ellipe, ellipk
from threadpoolctl import threadpool_limits

from freegsnke import parallel

# All tests are written under the assumption that the parallel functions are implemented as wrappers, and should give
# EXACTLY equivalent results to the serial versions


@pytest.fixture
def make_test_array():
    return np.random.rand(100, 100, 100)


@pytest.mark.skipif(
    parallel.get_max_threads() == 1,
    reason="Maximum number of threads set at 1, can't set other values",
)
def test_set_threads():

    num_threads = parallel.get_num_threads()
    max_threads = parallel.get_max_threads()

    if num_threads < max_threads:
        new_num_threads = max_threads
    else:
        new_num_threads = 1

    parallel.set_num_threads(new_num_threads)

    assert parallel.get_num_threads() == new_num_threads

    parallel.set_num_threads(num_threads)


@pytest.mark.skipif(
    parallel.get_max_threads() == 1,
    reason="Maximum number of threads set at 1, can't set other values",
)
def test_threadpoolctl_support():

    num_threads = parallel.get_num_threads()
    max_threads = parallel.get_max_threads()

    if num_threads > 1:
        limit_threads = num_threads - 1
    elif max_threads > 1:
        limit_threads = max_threads

    with threadpool_limits(limits=limit_threads, user_api="fg_threads"):
        assert (
            parallel.get_num_threads() == limit_threads
        ), "Control within context manager failed"

    assert (
        parallel.get_num_threads() == num_threads
    ), "Catastrophic error, persistent change detected outside of context manager"


@pytest.mark.skipif(
    parallel.get_max_threads() == 1,
    reason="Maximum number of threads set at 1, can't set other values",
)
def test_take(make_test_array):

    idcs = np.random.randint(0, 100, size=(20, 3))
    reference = np.take(make_test_array, idcs)

    orig_num_threads = parallel.get_num_threads()
    parallel.set_num_threads(1)
    assert np.all(
        reference == parallel.threaded_take(make_test_array, idcs)
    ), "Thin wrapper fail (failed at 1 thread)"

    parallel.set_num_threads(2)
    assert np.all(
        reference == parallel.threaded_take(make_test_array, idcs)
    ), "Failed on 2 threads"

    parallel.set_num_threads(21)
    assert np.all(
        reference == parallel.threaded_take(make_test_array, idcs)
    ), "Failed n_threads>idcs.shape[0]"

    with pytest.raises(TypeError, match="Only numpy ndarray"):
        parallel.threaded_take(make_test_array, None)

    with pytest.raises(IndexError, match="out of bounds"):
        parallel.threaded_take(make_test_array, np.array(1e9, dtype=int))

    parallel.set_num_threads(orig_num_threads)


@pytest.mark.skipif(
    parallel.get_max_threads() == 1,
    reason="Maximum number of threads set at 1, can't set other values",
)
def test_take_axis(make_test_array):

    idcs = np.random.randint(0, 100, size=(20, 3))
    reference = np.take(make_test_array, idcs, axis=1)

    orig_num_threads = parallel.get_num_threads()
    parallel.set_num_threads(1)
    assert np.all(
        reference == parallel.threaded_take(make_test_array, idcs, axis=1)
    ), "Thin wrapper fail (failed at 1 thread)"

    parallel.set_num_threads(2)
    assert np.all(
        reference == parallel.threaded_take(make_test_array, idcs, axis=1)
    ), "Failed on 2 threads"

    parallel.set_num_threads(21)
    assert np.all(
        reference == parallel.threaded_take(make_test_array, idcs, axis=1)
    ), "Failed n_threads>idcs.shape[0]"

    with pytest.raises(AxisError, match="out of bounds"):
        parallel.threaded_take(make_test_array, idcs, axis=2000)

    parallel.set_num_threads(orig_num_threads)


@pytest.mark.skipif(
    parallel.get_max_threads() == 1,
    reason="Maximum number of threads set at 1, can't set other values",
)
def test_clip(make_test_array):

    amin, amax = 0.25, 0.75
    reference = np.clip(make_test_array, amin, amax)

    orig_num_threads = parallel.get_num_threads()
    parallel.set_num_threads(1)
    assert np.all(
        reference == parallel.threaded_clip(make_test_array, amin, amax)
    ), "Thin wrapper fail (failed at 1 thread)"

    parallel.set_num_threads(2)
    assert np.all(
        reference == parallel.threaded_clip(make_test_array, amin, amax)
    ), "Failed on 2 threads"

    max_threads = parallel.get_max_threads()
    small_array = np.random.rand(max_threads - 1)
    parallel.set_num_threads(max_threads)

    small_reference = np.clip(small_array, amin, amax)
    assert np.all(
        small_reference == parallel.threaded_clip(small_array, amin, amax)
    ), "Failed n_threads>array.shape[0]"

    with pytest.raises(ValueError, match="shape mismatch"):
        bad_out = np.empty((2, 2))
        parallel.threaded_clip(make_test_array, amin, amax, out=bad_out)

    with pytest.raises(Exception, match="Cannot cast"):
        bad_out = np.empty_like(make_test_array, dtype=np.int32)
        parallel.threaded_clip(make_test_array, amin, amax, out=bad_out)

    parallel.set_num_threads(orig_num_threads)


@pytest.mark.skipif(
    parallel.get_max_threads() == 1,
    reason="Maximum number of threads set at 1, can't set other values",
)
def test_elliptics(make_test_array):

    amin, amax = 0.25, 0.75
    refe = ellipe(make_test_array)
    refk = ellipk(make_test_array)

    orig_num_threads = parallel.get_num_threads()
    parallel.set_num_threads(1)
    rese, resk = parallel.threaded_elliptics_ek(make_test_array)
    assert np.all(
        (refe == rese) & (refk == resk)
    ), "Thin wrapper fail (failed at 1 thread)"

    parallel.set_num_threads(2)
    rese, resk = parallel.threaded_elliptics_ek(make_test_array)
    assert np.all((refe == rese) & (refk == resk)), "Failed on 2 threads"

    max_threads = parallel.get_max_threads()
    small_array = np.random.rand(max_threads - 1)
    parallel.set_num_threads(max_threads)

    refe = ellipe(small_array)
    refk = ellipk(small_array)
    rese, resk = parallel.threaded_elliptics_ek(small_array)
    assert np.all((refe == rese) & (refk == resk)), "Failed n_threads>idcs.shape[0]"

    with pytest.raises(TypeError, match="Only numpy ndarray"):
        parallel.threaded_elliptics_ek(None)

    parallel.set_num_threads(orig_num_threads)
