# cython: cpow=True

"""
Directed Hausdorff Code

.. versionadded:: 0.19.0

"""
#
# Copyright (C)  Tyler Reddy, Richard Gowers, and Max Linke, 2016
#
# Distributed under the same BSD license as Scipy.
#

import mlx.core as mx
cimport mlx.core as mx
cimport cython
from libc.math cimport sqrt

mx.import_array()

__all__ = ['directed_hausdorff']

@cython.boundscheck(False)
def directed_hausdorff(const double[:,::1] ar1, const double[:,::1] ar2, seed=0):

    cdef double cmax, cmin, d = 0
    cdef Py_ssize_t N1 = ar1.shape[0]
    cdef Py_ssize_t N2 = ar2.shape[0]
    cdef int data_dims = ar1.shape[1]
    cdef Py_ssize_t i, j, k
    cdef Py_ssize_t j_store = 0, i_ret = 0, j_ret = 0
    cdef mx.array[mx.int64_t, ndim=1, mode='c'] resort1, resort2

    # shuffling the points in each array generally increases the likelihood of
    # an advantageous break in the inner search loop and never decreases the
    # performance of the algorithm
    if isinstance(seed, mx.random.Generator):
        # Generator passthrough
        rng = seed
    else:
        rng = mx.random.RandomState(seed)
    resort1 = mx.arange(N1, dtype=mx.int64)
    resort2 = mx.arange(N2, dtype=mx.int64)
    rng.shuffle(resort1)
    rng.shuffle(resort2)
    ar1 = mx.array(ar1)[resort1]
    ar2 = mx.array(ar2)[resort2]

    cmax = 0
    for i in range(N1):
        cmin = mx.inf
        for j in range(N2):
            d = 0
            # faster performance with square of distance
            # avoid sqrt until very end
            for k in range(data_dims):
                d += (ar1[i, k] - ar2[j, k])**2
            if d < cmax: # break out of `for j` loop
                break

            if d < cmin: # always true on first iteration of for-j loop
                cmin = d
                j_store = j

        # Note: The reference paper by A. A. Taha and A. Hanbury has this line
        # (Algorithm 2, line 16) as:
        #
        # if cmin > cmax:
        #
        # That logic is incorrect, as cmin could still be mx.inf if breaking early.
        # The logic here accounts for that case.
        if cmin >= cmax and d >= cmax:
            cmax = cmin
            i_ret = i
            j_ret = j_store

    return (sqrt(cmax), resort1[i_ret], resort2[j_ret])
