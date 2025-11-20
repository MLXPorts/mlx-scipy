import mlx.core as mx
from scipy_mlx.io import loadmat

m = loadmat('test.mat', squeeze_me=True, struct_as_record=True,
        mat_dtype=True)
mx.savez('test.npz', **m)
