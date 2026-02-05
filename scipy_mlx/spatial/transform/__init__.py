"""
Spatial Transformations (:mod:`scipy.spatial.transform`)
========================================================

.. currentmodule:: scipy.spatial.transform

This package implements various spatial transformations. For now, rotations
and rigid transforms (rotations and translations) are supported.

Rotations in 3 dimensions
-------------------------
.. autosummary::
   :toctree: generated/

   RigidTransform
   Rotation
   Slerp
   RotationSpline
"""
from ._rotation import Rotation, Slerp
from ._rigid_transform import RigidTransform
try:
    from ._rotation_spline import RotationSpline
except Exception:  # pragma: no cover
    RotationSpline = None  # type: ignore[assignment]

# Deprecated namespaces, to be removed in v2.0.0
from . import rotation

__all__ = ['Rotation', 'Slerp', 'RigidTransform']
if RotationSpline is not None:
    __all__.append('RotationSpline')

from scipy_mlx._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester
