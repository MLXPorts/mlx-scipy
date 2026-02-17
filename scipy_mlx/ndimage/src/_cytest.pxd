# MLX port: removed NumPy Cython dependency

cimport mlx.core as mx

cdef int _filter1d(double *input_line, mx.intp_t input_length, double *output_line,
	           mx.intp_t output_length, void *callback_data) noexcept
cdef int _filter2d(double *buffer, mx.intp_t filter_size, double *res,
	           void *callback_data) noexcept
cdef int _transform(mx.intp_t *output_coordinates, double *input_coordinates,
	            int output_rank, int input_rank, void *callback_data) noexcept
