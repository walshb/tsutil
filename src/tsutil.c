/*
 * Copyright 2016 Ben Walsh
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "Python.h"
#include "math.h"
#include "numpy/arrayobject.h"

static int
check_array_int64like(PyArrayObject *arr) {
    if (PyArray_TYPE(arr) != NPY_LONG && PyArray_TYPE(arr) != NPY_DATETIME) {
        PyErr_SetString(PyExc_ValueError, "array must be int64 or datetime64");
        return -1;
    }
    return 0;
}

static int
check_array_64bits(PyArrayObject *arr) {
    if (PyArray_ITEMSIZE(arr) != 8) {
        PyErr_SetString(PyExc_ValueError, "array must have 64-bit elements");
        return -1;
    }
    return 0;
}

static int
check_array_float64(PyArrayObject *arr) {
    if (PyArray_TYPE(arr) != NPY_FLOAT64) {
        PyErr_SetString(PyExc_ValueError, "array must be float64");
        return -1;
    }
    return 0;
}

static PyObject *
resample(PyArrayObject *times_arr, PyArrayObject *values_arr, PyArrayObject *stimes_arr)
{
    if (check_array_int64like(times_arr)
        || check_array_int64like(stimes_arr)
        || check_array_64bits(values_arr)) {
        return NULL;
    }

    int ndims = PyArray_NDIM(times_arr);
    if (ndims != 1) {
        PyErr_SetString(PyExc_ValueError, "ndim must be 1");
        return NULL;
    }

    npy_intp n = PyArray_DIM(times_arr, 0);
    npy_intp sn = PyArray_DIM(stimes_arr, 0);

    PyArrayObject *svalues_arr = (PyArrayObject *)PyArray_SimpleNew(1, &sn, PyArray_TYPE(values_arr));

    npy_int64 *times = PyArray_DATA(times_arr);
    npy_int64 *stimes = PyArray_DATA(stimes_arr);
    npy_uint64 *values = PyArray_DATA(values_arr);
    npy_uint64 *svalues = PyArray_DATA(svalues_arr);

    int i;
    for (i = 0; i < sn; i++) {
        if (stimes[i] >= times[0]) {
            break;
        }
        svalues[i] = 0;  // sample before start of series
    }
    int j = 0;
    for ( ; i < sn; i++) {
        while (j < n && times[j] <= stimes[i]) {
            j++;
        }
        // j now points after stimes[i]
        svalues[i] = values[j - 1];
    }

    return (PyObject *)svalues_arr;
}

static PyObject *
resample_interp(PyArrayObject *times_arr, PyArrayObject *values_arr, PyArrayObject *stimes_arr)
{
    if (check_array_int64like(times_arr)
        || check_array_int64like(stimes_arr)
        || check_array_float64(values_arr)) {
        return NULL;
    }

    int ndims = PyArray_NDIM(times_arr);
    if (ndims != 1) {
        PyErr_SetString(PyExc_ValueError, "ndim must be 1");
        return NULL;
    }

    npy_intp n = PyArray_DIM(times_arr, 0);
    npy_intp sn = PyArray_DIM(stimes_arr, 0);

    PyArrayObject *svalues_arr = (PyArrayObject *)PyArray_SimpleNew(1, &sn, PyArray_TYPE(values_arr));

    npy_int64 *times = PyArray_DATA(times_arr);
    npy_int64 *stimes = PyArray_DATA(stimes_arr);
    npy_float64 *values = PyArray_DATA(values_arr);
    npy_float64 *svalues = PyArray_DATA(svalues_arr);

    int i;
    for (i = 0; i < sn; i++) {
        if (stimes[i] >= times[0]) {
            break;
        }
        svalues[i] = 0.0;  // sample before start of series
    }
    int j = 0;
    for ( ; i < sn; i++) {
        while (j < n && times[j] <= stimes[i]) {
            j++;
        }
        // j now points after stimes[i]
        if (j >= n) {
            svalues[i] = values[j - 1];
        } else {
            svalues[i] = values[j - 1]
                + (values[j] - values[j - 1])
                * (npy_float64)(stimes[i] - times[j - 1])
                / (npy_float64)(times[j] - times[j - 1]);
        }
    }

    return (PyObject *)svalues_arr;
}

static npy_float64
maxdd(PyArrayObject *values_arr)
{
    if (check_array_float64(values_arr)) {
        return -1.0;
    }

    int ndims = PyArray_NDIM(values_arr);
    if (ndims != 1) {
        PyErr_SetString(PyExc_ValueError, "ndim must be 1");
        return -1.0;
    }

    npy_intp n = PyArray_DIM(values_arr, 0);

    npy_float64 res = 0.0;

    if (n == 0) {
        return res;
    }

    npy_float64 *values = (npy_float64 *)PyArray_DATA(values_arr);

    npy_float64 max_value = values[0];

    int i;
    for (i = 0; i < n; i++) {
        if (values[i] > max_value) {
            max_value = values[i];
        } else {
            npy_float64 dd = max_value - values[i];
            if (dd > res) {
                res = dd;
            }
        }
    }

    return res;
}

static PyObject *
resample_func(PyObject *self, PyObject *args)
{
    PyArrayObject *times_arr;
    PyArrayObject *values_arr;
    PyArrayObject *stimes_arr;

    if (!PyArg_ParseTuple(args, "OOO", &times_arr, &values_arr, &stimes_arr)) {
        return NULL;
    }

    return resample(times_arr, values_arr, stimes_arr);
}

static PyObject *
resample_interp_func(PyObject *self, PyObject *args)
{
    PyArrayObject *times_arr;
    PyArrayObject *values_arr;
    PyArrayObject *stimes_arr;

    if (!PyArg_ParseTuple(args, "OOO", &times_arr, &values_arr, &stimes_arr)) {
        return NULL;
    }

    return resample_interp(times_arr, values_arr, stimes_arr);
}

static PyObject *
maxdd_func(PyObject *self, PyObject *args)
{
    PyArrayObject *values_arr;

    if (!PyArg_ParseTuple(args, "O", &values_arr)) {
        return NULL;
    }

    npy_float64 res = maxdd(values_arr);

    if (res < 0.0) {
        return NULL;
    }

    return PyFloat_FromDouble(res);
}

static PyMethodDef mod_methods[] = {
    {"resample", (PyCFunction)resample_func, METH_VARARGS,
     "Resample"
    },
    {"resample_interp", (PyCFunction)resample_interp_func, METH_VARARGS,
     "Resample and interpolate"
    },
    {"maxdd", (PyCFunction)maxdd_func, METH_VARARGS,
     "Maximum drawdown"
    },
    {NULL}  /* Sentinel */
};

#ifndef PyMODINIT_FUNC  /* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
init_ctsutil(void)
{
    PyObject* m;

    import_array();

    m = Py_InitModule3("_ctsutil", mod_methods,
                       "Example module that creates an extension type.");

    (void)m;  /* avoid warning */
}
