cimport cython
from libcpp.vector cimport vector
from libcpp cimport bool

import numpy as np
cimport numpy as cnp

cdef extern from "wavelets.hpp" namespace "wavelets":

    ctypedef long long unsigned int size_t
    ctypedef long long int stride_t
    ctypedef vector[size_t] size_v
    ctypedef vector[stride_t] stride_v

    bool vector_support
    size_t vector_byte_length

    cdef enum class Wavelet:
        Daubechies1
        Daubechies2
        Daubechies3
        Daubechies4
        Daubechies5
        Daubechies6
        BiorSpline3_1
        BiorSpline4_2
        CDF5_3
        CDF9_7

    cdef enum class BoundaryCondition:
        ZERO
        PERIODIC
        CONSTANT
        SYMMETRIC
        REFLECT

    void lwt_cpp "lwt" [T](
        Wavelet wvlt, BoundaryCondition bc, bool forward, bool adjoint,
        size_v& shape, stride_v& stride_in, stride_v& stride_out, size_v& axes, size_t level,
        const T* data_in, T* data_out, size_t n_threads
    )

    void lwt_sep_cpp "lwt" [T](
        Wavelet wvlt, BoundaryCondition bc, bool forward, bool adjoint,
        size_v& shape, stride_v& stride_in, stride_v& stride_out, size_v& axes, size_v& levels,
        const T* data_in, T* data_out, size_t n_threads
    )

    size_t max_level(Wavelet wvlt, size_t n)


def get_vectorization_info():
    return {"vectorized":vector_support, "vector length":vector_byte_length}

def lift_transform(
        cnp.ndarray x,
        str wavelet='db4',
        str mode='zero',
        bool forward=True,
        bool adjoint=False,
        axes=None,
        level=1,
        size_t n_threads = 1,
):
    cdef:
        size_t ndim = x.ndim
        size_v shape = size_v(ndim)
        stride_v stride_in = stride_v(ndim)
        size_t elem_size = x.itemsize
        Wavelet wvlt
        BoundaryCondition bc

        size_v axes_
        size_t level_
        size_v levels_

    if wavelet == 'db1' or wavelet == 'haar':
        wvlt = Wavelet.Daubechies1
    elif wavelet == 'db2':
        wvlt = Wavelet.Daubechies2
    elif wavelet == 'db3':
        wvlt = Wavelet.Daubechies3
    elif wavelet == 'db4':
        wvlt = Wavelet.Daubechies4
    elif wavelet == 'db5':
        wvlt = Wavelet.Daubechies5
    elif wavelet == 'db6':
        wvlt = Wavelet.Daubechies6
    elif wavelet == 'bior3.1':
        wvlt = Wavelet.BiorSpline3_1
    elif wavelet == 'bior4.2':
        wvlt = Wavelet.BiorSpline4_2
    elif wavelet == 'cdf5.3':
        wvlt = Wavelet.CDF5_3
    elif wavelet == 'cdf9.7':
        wvlt = Wavelet.CDF9_7
    else:
        raise ValueError(f"Unknown wavelet {wavelet}")

    if mode == 'zero':
        bc = BoundaryCondition.ZERO
    elif mode == 'periodic':
        bc = BoundaryCondition.PERIODIC
    elif mode == 'constant':
        bc = BoundaryCondition.CONSTANT
    elif mode == 'symmetric':
        bc = BoundaryCondition.SYMMETRIC
    elif mode == 'reflect':
        bc = BoundaryCondition.REFLECT
    else:
        raise ValueError(f"Unknown mode {mode}")

    if axes is not None:
        try:
            axes_ = axes
        except TypeError as err:
            try:
                axes_ = (axes, )
            except Exception as err2:
                raise TypeError("Axes must be a iterable or single number.")
    else:
        axes_ = np.arange(ndim)

    try:
        level_ = level
    except TypeError:
        try:
            levels_ = level
        except TypeError:
            raise TypeError("Level must be a single number, or a list of numbers")
        if levels_.size() > 0 and levels_.size() != axes_.size():
            raise TypeError("levels must be the same length as axes")

    cdef:
        cnp.ndarray out = np.empty_like(x)
        stride_v stride_out = stride_v(ndim)


    for i in range(ndim):
        shape[i] = x.shape[i]
        stride_in[i] = x.strides[i] // elem_size
        stride_out[i] = out.strides[i] // elem_size

    cdef const void* data_in = cnp.PyArray_DATA(x)
    cdef void* data_out = cnp.PyArray_DATA(out)

    if levels_.size() == 0:
        if x.dtype == np.float32:
            lwt_cpp[float](wvlt, bc, forward, adjoint, shape, stride_in, stride_out, axes_, level_, <const float*> data_in, <float*> data_out, n_threads)
        elif x.dtype == np.float64:
            lwt_cpp[double](wvlt, bc, forward, adjoint, shape, stride_in, stride_out, axes_, level_, <const double*> data_in, <double*> data_out, n_threads)
    else:
        if x.dtype == np.float32:
            lwt_sep_cpp[float](wvlt, bc, forward, adjoint, shape, stride_in, stride_out, axes_, levels_, <const float*> data_in, <float*> data_out, n_threads)
        elif x.dtype == np.float64:
            lwt_sep_cpp[double](wvlt, bc, forward, adjoint, shape, stride_in, stride_out, axes_, levels_, <const double*> data_in, <double*> data_out, n_threads)

    return out