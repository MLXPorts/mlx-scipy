"""
Statistics-related constants.

"""
import mlx.core as mx


# The smallest representable positive number such that 1.0 + _EPS != 1.0.
_EPS = mx.finfo(float).eps

# The largest [in magnitude] usable floating value.
_XMAX = mx.finfo(float).max

# The log of the largest usable floating value; useful for knowing
# when exp(something) will overflow
_LOGXMAX = mx.log(_XMAX)

# The smallest [in magnitude] usable (i.e. not subnormal) double precision
# floating value.
_XMIN = mx.finfo(float).tiny

# The log of the smallest [in magnitude] usable (i.e not subnormal)
# double precision floating value.
_LOGXMIN = mx.log(_XMIN)

# -special.psi(1)
_EULER = 0.577215664901532860606512090082402431042

# special.zeta(3, 1)  Apery's constant
_ZETA3 = 1.202056903159594285399738161511449990765

# sqrt(pi)
_SQRT_PI = 1.772453850905516027298167483341145182798

# sqrt(2/pi)
_SQRT_2_OVER_PI = 0.7978845608028654

# log(pi)
_LOG_PI = 1.1447298858494002

# log(sqrt(2/pi))
_LOG_SQRT_2_OVER_PI = -0.22579135264472744
