# cimport cython
# from libcpp cimport bool
from cpython.ref cimport Py_XINCREF

import numpy as np
cimport numpy as cnp

from libcpp.vector cimport vector

cdef extern from "lifted.hpp" namespace "lifted":
    ctypedef unsigned long long int size_t
    ctypedef long long int stride_t
    ctypedef vector[size_t] size_v
    ctypedef vector[stride_t] stride_v
    
    cpdef enum class Wavelet(unsigned int):
        Lazy,
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
        Symlet4,
        Symlet5,
        Symlet6,
        Coiflet2,
        Coiflet3,
        Bior1_3,
        Bior1_5,
        Bior2_2,
        Bior2_4,
        Bior2_6,
        Bior2_8,
        Bior3_1,
        Bior3_3,
        Bior3_5,
        Bior3_7,
        Bior3_9,
        Bior4_2,
        Bior4_4,
        Bior4_6,
        Bior5_5,
        Bior6_8,
        CDF5_3,
        CDF9_7,

    cpdef enum class BoundaryCondition(unsigned int):
        Zero,
        Periodic,
        Constant,
        Symmetric,
        Reflect

    cpdef enum class Transform(unsigned int):
        Forward,
        Inverse,
        ForwardAdjoint,
        InverseAdjoint
    
    void lwt(
        Wavelet wvlt, Transform op, BoundaryCondition bc,
        const size_v& shape, const stride_v& stride_in, const stride_v& stride_out,
        const size_v& axes, const size_t levels,
        const float* data_in, float* data_out, const size_t n_threads
    )

    void lwt(
            Wavelet wvlt, Transform op, BoundaryCondition bc,
            const size_v& shape, const stride_v& stride_in, const stride_v& stride_out,
            const size_v& axes, const size_t levels,
            const double* data_in, double* data_out, const size_t n_threads
        )

    void lwt(
            Wavelet wvlt, Transform op, BoundaryCondition bc,
            const size_v& shape, const stride_v& stride_in, const stride_v& stride_out,
            const size_v& axes, const size_v& levels,
            const float* data_in, float* data_out, const size_t n_threads
        )

    void lwt(
            Wavelet wvlt, Transform op, BoundaryCondition bc,
            const size_v& shape, const stride_v& stride_in, const stride_v& stride_out,
            const size_v& axes, const size_v& levels,
            const double* data_in, double* data_out, const size_t n_threads
        )

_MAP_STR_TO_WVLT = {
    "lazy" : Wavelet.Lazy,
    "haar" : Wavelet.Daubechies1,
    "db1" : Wavelet.Daubechies1,
    "db2" : Wavelet.Daubechies2,
    "db3" : Wavelet.Daubechies3,
    "db4" : Wavelet.Daubechies4,
    "db5" : Wavelet.Daubechies5,
    "db6" : Wavelet.Daubechies6,
    "db7" : Wavelet.Daubechies7,
    "db8" : Wavelet.Daubechies8,
    "db9" : Wavelet.Daubechies9,
    "db10" : Wavelet.Daubechies10,
    "sym1" : Wavelet.Daubechies1,
    "sym2" : Wavelet.Daubechies2,
    "sym3" : Wavelet.Daubechies3,
    "sym4" : Wavelet.Symlet4,
    "sym5" : Wavelet.Symlet5,
    "sym6" : Wavelet.Symlet6,
    "coif2" : Wavelet.Coiflet2,
    "coif3" : Wavelet.Coiflet3,
    "bior1.3" : Wavelet.Bior1_3,
    "bior1.5" : Wavelet.Bior1_5,
    "bior2.2" : Wavelet.Bior2_2,
    "bior2.4" : Wavelet.Bior2_4,
    "bior2.6" : Wavelet.Bior2_6,
    "bior2.8" : Wavelet.Bior2_8,
    "bior3.1" : Wavelet.Bior3_1,
    "bior3.3" : Wavelet.Bior3_3,
    "bior3.5" : Wavelet.Bior3_5,
    "bior3.7" : Wavelet.Bior3_7,
    "bior3.9" : Wavelet.Bior3_9,
    "bior4.2" : Wavelet.Bior4_2,
    "bior4.4" : Wavelet.Bior4_4,
    "bior4.6" : Wavelet.Bior4_6,
    "bior5.5" : Wavelet.Bior5_5,
    "bior6.8" : Wavelet.Bior6_8,
    "cdf5.3" : Wavelet.CDF5_3,
    "cdf9.7" : Wavelet.CDF9_7,
}

cdef size_v get_axes(axes, ndim):

    if axes is None:
        axes = np.arange(ndim)
    else:
        try:
            axes = tuple(axes)
        except TypeError:
            axes = (axes, )
    
    cdef stride_v temp
    try:
        temp = axes
    except TypeError as err:
        raise TypeError("Axes must be integers") from err
    
    cdef size_t n_ax = temp.size()
    cdef size_v axes_ = size_v(n_ax)
    cdef size_t i
    for i in range(n_ax):
        axis = temp[i]
        if axis >= ndim:
            raise ValueError(f"Axis {axis} is beyond the array's dimensions, {ndim}.")
        axes_[i] = axis % ndim
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

cdef _lwt(
        cnp.ndarray x,
        wavelet,
        mode,
        Transform op,
        axes,
        level,
        size_t n_threads,
        out = None,
):
    cdef:
        size_t ndim = x.ndim
        size_v shape = size_v(ndim)
        stride_v stride_in = stride_v(ndim)
        size_t elem_size = x.itemsize
        Wavelet wavelet_
        BoundaryCondition mode_

        size_v axes_
        size_t level_s
        size_v levels_

    axes_ = get_axes(axes, ndim)
    level_, levels_ = get_level(level, axes_)

    cdef:
        cnp.ndarray out_
        stride_v stride_out = stride_v(ndim)

    if out is None:
        out_ = cnp.PyArray_EMPTY(
            ndim,
            x.shape,
            cnp.PyArray_TYPE(x),
            cnp.PyArray_IS_F_CONTIGUOUS(x)
        )
    else:
        out_ = out

    if out_.dtype != x.dtype:
        raise ValueError(f"`out.dtype`:{out_.dtype} does not match `input.dtype`:{x.dtype}.")
     
     # If the value was in the Wavelet Enum, we can just directly assign it
     # Strings will not be "in" Wavelet, as that is only numbers or the actual enum values.
    if wavelet in Wavelet:
        wavelet_ = wavelet
    else:
        try:
            wavelet_ = _MAP_STR_TO_WVLT[wavelet]
        except KeyError as err:
            raise KeyError(wavelet + " is not a valid Wavelet.") from None
    
    # ditto for boundary condition
    if mode in BoundaryCondition:
        mode_ = mode
    else:
        mode_ = BoundaryCondition[mode]

    for i in range(ndim):
        shape[i] = x.shape[i]
        if x.shape[i] != out_.shape[i]:
            raise ValueError(f"`out.shape` does not match input along axis {i}.")
        stride_in[i] = x.strides[i] // elem_size
        stride_out[i] = out_.strides[i] // elem_size

    cdef const void* data_in = cnp.PyArray_DATA(x)
    cdef void* data_out = cnp.PyArray_DATA(out_)

    if levels_.size() == 0:
        if x.dtype == np.float32:
            lwt(wavelet_, op, mode_, shape, stride_in, stride_out, axes_, level_, <const float*> data_in, <float*> data_out, n_threads)
        elif x.dtype == np.float64:
            lwt(wavelet_, op, mode_, shape, stride_in, stride_out, axes_, level_, <const double*> data_in, <double*> data_out, n_threads)
    else:
        if x.dtype == np.float32:
            lwt(wavelet_, op, mode_, shape, stride_in, stride_out, axes_, levels_, <const float*> data_in, <float*> data_out, n_threads)
        elif x.dtype == np.float64:
            lwt(wavelet_, op, mode_, shape, stride_in, stride_out, axes_, levels_, <const double*> data_in, <double*> data_out, n_threads)

    return out_

def lwt_forward(x, wavelet, mode=BoundaryCondition.Zero, axes=None, level=1, size_t n_threads = 1, out=None):
    return _lwt(x, wavelet, mode, Transform.Forward, axes, level, n_threads, out=out)

def lwt_inverse(x, wavelet, mode=BoundaryCondition.Zero, axes=None, level=1, size_t n_threads = 1, out=None):
    return _lwt(x, wavelet, mode, Transform.Inverse, axes, level, n_threads, out=out)

def lwt_forward_adjoint(x, wavelet, mode=BoundaryCondition.Zero, axes=None, level=1, size_t n_threads = 1, out=None):
    return _lwt(x, wavelet, mode, Transform.ForwardAdjoint, axes, level, n_threads, out=out)

def lwt_inverse_adjoint(x, wavelet, mode=BoundaryCondition.Zero, axes=None, level=1, size_t n_threads = 1, out=None):
    return _lwt(x, wavelet, mode, Transform.InverseAdjoint, axes, level, n_threads, out=out)