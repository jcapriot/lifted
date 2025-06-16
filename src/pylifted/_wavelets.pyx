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

    cpdef enum class Wavelet:
        Daubechies1,
        Daubechies2,
        Daubechies3,
        Daubechies4,
        Daubechies5,
        Daubechies6,
        Daubechies7,
        Daubechies8,
        Daubechies9,
        Daubechies10,
        BiorSpline3_1,
        BiorSpline4_2,
        BiorSpline2_4,
        BiorSpline6_2,
        CDF5_3,
        CDF9_7

    cpdef enum class BoundaryCondition:
        ZERO
        PERIODIC
        CONSTANT
        SYMMETRIC
        REFLECT

    size_t max_level(Wavelet wvlt, size_t n)

cdef extern from "instantiations.hpp" namespace "lifted":

    # forward
    void lwt_float(
        const Wavelet wvlt, const BoundaryCondition bc,
        const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_t level,
        const float* data_in, float* data_out, size_t n_threads
    )

    void lwt_double(
        const Wavelet wvlt, const BoundaryCondition bc,
        const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_t level,
        const double* data_in, double* data_out, size_t n_threads
    )

    void lwt_sep_float(
        const Wavelet wvlt, const BoundaryCondition bc,
        const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_v& levels,
        const float* data_in, float* data_out, size_t n_threads
    )

    void lwt_sep_double(
        const Wavelet wvlt, const BoundaryCondition bc,
        const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_v& levels,
        const double* data_in, double* data_out, size_t n_threads
    );

    # inverse
    void ilwt_float(
        const Wavelet wvlt, const BoundaryCondition bc,
        const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_t level,
        const float* data_in, float* data_out, size_t n_threads
    )

    void ilwt_double(
        const Wavelet wvlt, const BoundaryCondition bc,
        const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_t level,
        const double* data_in, double* data_out, size_t n_threads
    )

    void ilwt_sep_float(
        const Wavelet wvlt, const BoundaryCondition bc,
        const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_v& levels,
        const float* data_in, float* data_out, size_t n_threads
    )

    void ilwt_sep_double(
        const Wavelet wvlt, const BoundaryCondition bc,
        const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_v& levels,
        const double* data_in, double* data_out, size_t n_threads
    )

    # forward adjoint
    void lwt_adjoint_float(
        const Wavelet wvlt, const BoundaryCondition bc,
        const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_t level,
        const float* data_in, float* data_out, size_t n_threads
    )

    void lwt_adjoint_double(
        const Wavelet wvlt, const BoundaryCondition bc,
        const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_t level,
        const double* data_in, double* data_out, size_t n_threads
    )

    void lwt_sep_adjoint_float(
        const Wavelet wvlt, const BoundaryCondition bc,
        const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_v& levels,
        const float* data_in, float* data_out, size_t n_threads
    )

    void lwt_sep_adjoint_double(
        const Wavelet wvlt, const BoundaryCondition bc,
        const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_v& levels,
        const double* data_in, double* data_out, size_t n_threads
    )

    # inverse adjoint
    void ilwt_adjoint_float(
        const Wavelet wvlt, const BoundaryCondition bc,
        const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_t level,
        const float* data_in, float* data_out, size_t n_threads
    )

    void ilwt_adjoint_double(
        const Wavelet wvlt, const BoundaryCondition bc,
        const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_t level,
        const double* data_in, double* data_out, size_t n_threads
    )

    void ilwt_sep_adjoint_float(
        const Wavelet wvlt, const BoundaryCondition bc,
        const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_v& levels,
        const float* data_in, float* data_out, size_t n_threads
    )

    void ilwt_sep_adjoint_double(
        const Wavelet wvlt, const BoundaryCondition bc,
        const size_v& shape, const stride_v& stride_in, const stride_v& stride_out, const size_v& axes, const size_v& levels,
        const double* data_in, double* data_out, size_t n_threads
    )

def get_vectorization_info():
    return {"vectorized":vector_support, "vector length":vector_byte_length}

cdef Wavelet get_wvlt(str wavelet):

    cdef Wavelet wvlt

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

    return wvlt

cdef BoundaryCondition get_bc(str mode):

    cdef BoundaryCondition bc

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

    return bc

cdef size_v get_axes(axes, ndim):
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
    return axes_

cdef (size_t, size_v) get_level(level, size_v axes):
    cdef:
        size_t level_
        size_v levels_

    try:
        level_ = level
    except TypeError:
        try:
            levels_ = level
        except TypeError:
            raise TypeError("Level must be a single number, or a list of numbers")
        if levels_.size() > 0 and levels_.size() != axes.size():
            raise TypeError("levels must be the same length as axes")

    return level_, levels_


def lwt(
        cnp.ndarray x,
        str wavelet='db4',
        str mode='zero',
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

    wvlt = get_wvlt(wavelet)
    bc = get_bc(mode)
    axes_ = get_axes(axes, ndim)
    level_, levels_ = get_level(level, axes_)

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
            lwt_float(wvlt, bc, shape, stride_in, stride_out, axes_, level_, <const float*> data_in, <float*> data_out, n_threads)
        elif x.dtype == np.float64:
            lwt_double(wvlt, bc, shape, stride_in, stride_out, axes_, level_, <const double*> data_in, <double*> data_out, n_threads)
    else:
        if x.dtype == np.float32:
            lwt_sep_float(wvlt, bc, shape, stride_in, stride_out, axes_, levels_, <const float*> data_in, <float*> data_out, n_threads)
        elif x.dtype == np.float64:
            lwt_sep_double(wvlt, bc, shape, stride_in, stride_out, axes_, levels_, <const double*> data_in, <double*> data_out, n_threads)

    return out


def ilwt(
        cnp.ndarray x,
        str wavelet='db4',
        str mode='zero',
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

    wvlt = get_wvlt(wavelet)
    bc = get_bc(mode)
    axes_ = get_axes(axes, ndim)
    level_, levels_ = get_level(level, axes_)

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
            ilwt_float(wvlt, bc, shape, stride_in, stride_out, axes_, level_, <const float*> data_in, <float*> data_out, n_threads)
        elif x.dtype == np.float64:
            ilwt_double(wvlt, bc, shape, stride_in, stride_out, axes_, level_, <const double*> data_in, <double*> data_out, n_threads)
    else:
        if x.dtype == np.float32:
            ilwt_sep_float(wvlt, bc, shape, stride_in, stride_out, axes_, levels_, <const float*> data_in, <float*> data_out, n_threads)
        elif x.dtype == np.float64:
            ilwt_sep_double(wvlt, bc, shape, stride_in, stride_out, axes_, levels_, <const double*> data_in, <double*> data_out, n_threads)

    return out



def lwt_adjoint(
        cnp.ndarray x,
        str wavelet='db4',
        str mode='zero',
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

    wvlt = get_wvlt(wavelet)
    bc = get_bc(mode)
    axes_ = get_axes(axes, ndim)
    level_, levels_ = get_level(level, axes_)

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
            lwt_adjoint_float(wvlt, bc, shape, stride_in, stride_out, axes_, level_, <const float*> data_in, <float*> data_out, n_threads)
        elif x.dtype == np.float64:
            lwt_adjoint_double(wvlt, bc, shape, stride_in, stride_out, axes_, level_, <const double*> data_in, <double*> data_out, n_threads)
    else:
        if x.dtype == np.float32:
            lwt_sep_adjoint_float(wvlt, bc, shape, stride_in, stride_out, axes_, levels_, <const float*> data_in, <float*> data_out, n_threads)
        elif x.dtype == np.float64:
            lwt_sep_adjoint_double(wvlt, bc, shape, stride_in, stride_out, axes_, levels_, <const double*> data_in, <double*> data_out, n_threads)

    return out


def ilwt_adjoint(
        cnp.ndarray x,
        str wavelet='db4',
        str mode='zero',
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

    wvlt = get_wvlt(wavelet)
    bc = get_bc(mode)
    axes_ = get_axes(axes, ndim)
    level_, levels_ = get_level(level, axes_)

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
            ilwt_adjoint_float(wvlt, bc, shape, stride_in, stride_out, axes_, level_, <const float*> data_in, <float*> data_out, n_threads)
        elif x.dtype == np.float64:
            ilwt_adjoint_double(wvlt, bc, shape, stride_in, stride_out, axes_, level_, <const double*> data_in, <double*> data_out, n_threads)
    else:
        if x.dtype == np.float32:
            ilwt_sep_adjoint_float(wvlt, bc, shape, stride_in, stride_out, axes_, levels_, <const float*> data_in, <float*> data_out, n_threads)
        elif x.dtype == np.float64:
            ilwt_sep_adjoint_double(wvlt, bc, shape, stride_in, stride_out, axes_, levels_, <const double*> data_in, <double*> data_out, n_threads)

    return out