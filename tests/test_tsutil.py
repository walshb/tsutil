# Copyright 2016 Ben Walsh
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

import tsutil

def test_resample():
    times, values = np.array([1, 3, 5, 7]), np.array([1.0, 3.0, 5.0, 7.0])

    stimes = np.array([2, 3, 6])

    svalues = tsutil.resample(times, values, stimes)

    assert np.allclose(svalues, np.array([1.0, 3.0, 5.0]))


def test_resample_start():
    times, values = np.array([1, 3, 5, 7]), np.array([1.0, 3.0, 5.0, 7.0])

    stimes = np.array([0, 3, 6])

    svalues = tsutil.resample(times, values, stimes)

    assert np.allclose(svalues, np.array([0.0, 3.0, 5.0]))


def test_resample_interp():
    times, values = np.array([1, 3, 5, 7]), np.array([1.0, 3.0, 5.0, 7.0])

    stimes = np.array([2, 3, 6])

    svalues = tsutil.resample_interp(times, values, stimes)

    assert np.allclose(svalues, np.array([2.0, 3.0, 6.0]))


def test_resample_interp2():
    times, values = np.array([1, 3, 5, 7]), np.array([1.0, 3.0, 5.0, 10.0])

    stimes = np.array([2, 3, 6])

    svalues = tsutil.resample_interp(times, values, stimes)

    assert np.allclose(svalues, np.array([2.0, 3.0, 7.5]))


def test_maxdd():
    arr = np.array([10.0, 15.0, 20.0, 12.0, 18.0, 14.0, 5.0, 6.0, 7.0])

    res = tsutil.maxdd(arr)

    assert np.allclose(res, 15.0)
