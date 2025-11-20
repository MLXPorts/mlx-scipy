
DTYPE = mx.float64
ctypedef mx.float64_t DTYPE_t

ITYPE = mx.int32
ctypedef mx.int32_t ITYPE_t

# Fused type for int32 and int64
ctypedef fused int32_or_int64:
    mx.int32_t
    mx.int64_t

# Another copy of the same fused type, for working with mixed-type functions.
ctypedef fused int32_or_int64_b:
    mx.int32_t
    mx.int64_t

# NULL_IDX is the index used in predecessor matrices to store a non-path
DEF NULL_IDX = -9999
