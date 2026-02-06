import importlib
import sys
from pathlib import Path

import mlx.core as mx


def _has_module(prefix: str) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for name in sys.modules)


def test_imports_do_not_load_numpy_or_scipy():
    # Ensure the project root is on sys.path for pytest environments that
    # don't automatically add it.
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)

    # Importing scipy_mlx should not *cause* upstream SciPy or NumPy imports.
    before = set(sys.modules)

    mods = [
        "scipy_mlx",
        "scipy_mlx.cluster",
        "scipy_mlx.constants",
        "scipy_mlx.datasets",
        "scipy_mlx.differentiate",
        "scipy_mlx.fft",
        "scipy_mlx.fftpack",
        "scipy_mlx.integrate",
        "scipy_mlx.interpolate",
        "scipy_mlx.io",
        "scipy_mlx.linalg",
        "scipy_mlx.ndimage",
        "scipy_mlx.odr",
        "scipy_mlx.optimize",
        "scipy_mlx.signal",
        "scipy_mlx.sparse",
        "scipy_mlx.spatial",
        "scipy_mlx.special",
        "scipy_mlx.stats",
    ]

    for name in mods:
        importlib.import_module(name)

    after = set(sys.modules)
    newly_loaded = after - before
    assert not any(m == "numpy" or m.startswith("numpy.") for m in newly_loaded)
    assert not any(m == "scipy" or m.startswith("scipy.") for m in newly_loaded)


def test_basic_fft_smoke():
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)

    from scipy_mlx.fft import fft, ifft

    x = mx.arange(8)
    y = fft(x)
    z = ifft(y)

    assert y.shape == x.shape
    assert z.shape == x.shape
