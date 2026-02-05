import mlx.core as mx


def iscomplexobj(x):
    """MLX equivalent of numpy.iscomplexobj."""
    if isinstance(x, mx.array):
        return x.dtype in [mx.complex64]
    return False


# Test the function
x = mx.array([1+2j])
x_real = mx.array([1, 2, 3])
print(f"Complex array: {iscomplexobj(x)}")
print(f"Real array: {iscomplexobj(x_real)}")