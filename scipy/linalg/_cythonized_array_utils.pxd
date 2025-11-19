cimport numpy as cnp

ctypedef fused lapack_t:
    float
    double
    (float complex)
    (double complex)

ctypedef fused lapack_cz_t:
    (float complex)
    (double complex)

ctypedef fused lapack_sd_t:
    float
    double

ctypedef fused np_numeric_t:
    cmx.int8_t
    cmx.int16_t
    cmx.int32_t
    cmx.int64_t
    cmx.uint8_t
    cmx.uint16_t
    cmx.uint32_t
    cmx.uint64_t
    cmx.float32_t
    cmx.float64_t
    cmx.longdouble_t
    cmx.complex64_t
    cmx.complex128_t

ctypedef fused np_complex_numeric_t:
    cmx.complex64_t
    cmx.complex128_t


cdef void swap_c_and_f_layout(lapack_t *a, lapack_t *b, int r, int c) noexcept nogil
cdef (int, int) band_check_internal_c(np_numeric_t[:, ::1]A) noexcept nogil
cdef bint is_sym_her_real_c_internal(np_numeric_t[:, ::1]A) noexcept nogil
cdef bint is_sym_her_complex_c_internal(np_complex_numeric_t[:, ::1]A) noexcept nogil
