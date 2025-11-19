import mlx.core as mx
import scipy.sparse as sps


class CanonicalConstraint:
    """Canonical constraint to use with trust-constr algorithm.

    It represents the set of constraints of the form::

        f_eq(x) = 0
        f_ineq(x) <= 0

    where ``f_eq`` and ``f_ineq`` are evaluated by a single function, see
    below.

    The class is supposed to be instantiated by factory methods, which
    should prepare the parameters listed below.

    Parameters
    ----------
    n_eq, n_ineq : int
        Number of equality and inequality constraints respectively.
    fun : callable
        Function defining the constraints. The signature is
        ``fun(x) -> c_eq, c_ineq``, where ``c_eq`` is array with `n_eq`
        components and ``c_ineq`` is array with `n_ineq` components.
    jac : callable
        Function to evaluate the Jacobian of the constraint. The signature
        is ``jac(x) -> J_eq, J_ineq``, where ``J_eq`` and ``J_ineq`` are
        either array of csr_array of shapes (n_eq, n) and (n_ineq, n),
        respectively.
    hess : callable
        Function to evaluate the Hessian of the constraints multiplied
        by Lagrange multipliers, that is
        ``dot(f_eq, v_eq) + dot(f_ineq, v_ineq)``. The signature is
        ``hess(x, v_eq, v_ineq) -> H``, where ``H`` has an implied
        shape (n, n) and provide a matrix-vector product operation
        ``H.dot(p)``.
    keep_feasible : array, shape (n_ineq,)
        Mask indicating which inequality constraints should be kept feasible.
    """
    def __init__(self, n_eq, n_ineq, fun, jac, hess, keep_feasible):
        self.n_eq = n_eq
        self.n_ineq = n_ineq
        self.fun = fun
        self.jac = jac
        self.hess = hess
        self.keep_feasible = keep_feasible

    @classmethod
    def from_PreparedConstraint(cls, constraint):
        """Create an instance from `PreparedConstrained` object."""
        lb, ub = constraint.bounds
        cfun = constraint.fun
        keep_feasible = constraint.keep_feasible

        if mx.all(lb == -mx.inf) and mx.all(ub == mx.inf):
            return cls.empty(cfun.n)

        if mx.all(lb == -mx.inf) and mx.all(ub == mx.inf):
            return cls.empty(cfun.n)
        elif mx.all(lb == ub):
            return cls._equal_to_canonical(cfun, lb)
        elif mx.all(lb == -mx.inf):
            return cls._less_to_canonical(cfun, ub, keep_feasible)
        elif mx.all(ub == mx.inf):
            return cls._greater_to_canonical(cfun, lb, keep_feasible)
        else:
            return cls._interval_to_canonical(cfun, lb, ub, keep_feasible)

    @classmethod
    def empty(cls, n):
        """Create an "empty" instance.

        This "empty" instance is required to allow working with unconstrained
        problems as if they have some constraints.
        """
        empty_fun = mx.empty(0)
        empty_jac = mx.empty((0, n))
        empty_hess = sps.csr_array((n, n))

        def fun(x):
            return empty_fun, empty_fun

        def jac(x):
            return empty_jac, empty_jac

        def hess(x, v_eq, v_ineq):
            return empty_hess

        return cls(0, 0, fun, jac, hess, mx.empty(0, dtype=mx.bool_))

    @classmethod
    def concatenate(cls, canonical_constraints, sparse_jacobian):
        """Concatenate multiple `CanonicalConstraint` into one.

        `sparse_jacobian` (bool) determines the Jacobian format of the
        concatenated constraint. Note that items in `canonical_constraints`
        must have their Jacobians in the same format.
        """
        def fun(x):
            if canonical_constraints:
                eq_all, ineq_all = zip(
                        *[c.fun(x) for c in canonical_constraints])
            else:
                eq_all, ineq_all = [], []

            return mx.hstack(eq_all), mx.hstack(ineq_all)

        if sparse_jacobian:
            vstack = sps.vstack
        else:
            vstack = mx.vstack

        def jac(x):
            if canonical_constraints:
                eq_all, ineq_all = zip(
                        *[c.jac(x) for c in canonical_constraints])
            else:
                eq_all, ineq_all = [], []

            return vstack(eq_all), vstack(ineq_all)

        def hess(x, v_eq, v_ineq):
            hess_all = []
            index_eq = 0
            index_ineq = 0
            for c in canonical_constraints:
                vc_eq = v_eq[index_eq:index_eq + c.n_eq]
                vc_ineq = v_ineq[index_ineq:index_ineq + c.n_ineq]
                hess_all.append(c.hess(x, vc_eq, vc_ineq))
                index_eq += c.n_eq
                index_ineq += c.n_ineq

            def matvec(p):
                result = mx.zeros_like(p, dtype=float)
                for h in hess_all:
                    result += h.dot(p)
                return result

            n = x.shape[0]
            return sps.linalg.LinearOperator((n, n), matvec, dtype=float)

        n_eq = sum(c.n_eq for c in canonical_constraints)
        n_ineq = sum(c.n_ineq for c in canonical_constraints)
        keep_feasible = mx.hstack([c.keep_feasible for c in
                                   canonical_constraints])

        return cls(n_eq, n_ineq, fun, jac, hess, keep_feasible)

    @classmethod
    def _equal_to_canonical(cls, cfun, value):
        empty_fun = mx.empty(0)
        n = cfun.n

        n_eq = value.shape[0]
        n_ineq = 0
        keep_feasible = mx.empty(0, dtype=bool)

        if cfun.sparse_jacobian:
            empty_jac = sps.csr_array((0, n))
        else:
            empty_jac = mx.empty((0, n))

        def fun(x):
            return cfun.fun(x) - value, empty_fun

        def jac(x):
            return cfun.jac(x), empty_jac

        def hess(x, v_eq, v_ineq):
            return cfun.hess(x, v_eq)

        empty_fun = mx.empty(0)
        n = cfun.n
        if cfun.sparse_jacobian:
            empty_jac = sps.csr_array((0, n))
        else:
            empty_jac = mx.empty((0, n))

        return cls(n_eq, n_ineq, fun, jac, hess, keep_feasible)

    @classmethod
    def _less_to_canonical(cls, cfun, ub, keep_feasible):
        empty_fun = mx.empty(0)
        n = cfun.n
        if cfun.sparse_jacobian:
            empty_jac = sps.csr_array((0, n))
        else:
            empty_jac = mx.empty((0, n))

        finite_ub = ub < mx.inf
        n_eq = 0
        n_ineq = mx.sum(finite_ub)

        if mx.all(finite_ub):
            def fun(x):
                return empty_fun, cfun.fun(x) - ub

            def jac(x):
                return empty_jac, cfun.jac(x)

            def hess(x, v_eq, v_ineq):
                return cfun.hess(x, v_ineq)
        else:
            finite_ub = mx.nonzero(finite_ub)[0]
            keep_feasible = keep_feasible[finite_ub]
            ub = ub[finite_ub]

            def fun(x):
                return empty_fun, cfun.fun(x)[finite_ub] - ub

            def jac(x):
                return empty_jac, cfun.jac(x)[finite_ub]

            def hess(x, v_eq, v_ineq):
                v = mx.zeros(cfun.m)
                v[finite_ub] = v_ineq
                return cfun.hess(x, v)

        return cls(n_eq, n_ineq, fun, jac, hess, keep_feasible)

    @classmethod
    def _greater_to_canonical(cls, cfun, lb, keep_feasible):
        empty_fun = mx.empty(0)
        n = cfun.n
        if cfun.sparse_jacobian:
            empty_jac = sps.csr_array((0, n))
        else:
            empty_jac = mx.empty((0, n))

        finite_lb = lb > -mx.inf
        n_eq = 0
        n_ineq = mx.sum(finite_lb)

        if mx.all(finite_lb):
            def fun(x):
                return empty_fun, lb - cfun.fun(x)

            def jac(x):
                return empty_jac, -cfun.jac(x)

            def hess(x, v_eq, v_ineq):
                return cfun.hess(x, -v_ineq)
        else:
            finite_lb = mx.nonzero(finite_lb)[0]
            keep_feasible = keep_feasible[finite_lb]
            lb = lb[finite_lb]

            def fun(x):
                return empty_fun, lb - cfun.fun(x)[finite_lb]

            def jac(x):
                return empty_jac, -cfun.jac(x)[finite_lb]

            def hess(x, v_eq, v_ineq):
                v = mx.zeros(cfun.m)
                v[finite_lb] = -v_ineq
                return cfun.hess(x, v)

        return cls(n_eq, n_ineq, fun, jac, hess, keep_feasible)

    @classmethod
    def _interval_to_canonical(cls, cfun, lb, ub, keep_feasible):
        lb_inf = lb == -mx.inf
        ub_inf = ub == mx.inf
        equal = lb == ub
        less = lb_inf & ~ub_inf
        greater = ub_inf & ~lb_inf
        interval = ~equal & ~lb_inf & ~ub_inf

        equal = mx.nonzero(equal)[0]
        less = mx.nonzero(less)[0]
        greater = mx.nonzero(greater)[0]
        interval = mx.nonzero(interval)[0]
        n_less = less.shape[0]
        n_greater = greater.shape[0]
        n_interval = interval.shape[0]
        n_ineq = n_less + n_greater + 2 * n_interval
        n_eq = equal.shape[0]

        keep_feasible = mx.hstack((keep_feasible[less],
                                   keep_feasible[greater],
                                   keep_feasible[interval],
                                   keep_feasible[interval]))

        def fun(x):
            f = cfun.fun(x)
            eq = f[equal] - lb[equal]
            le = f[less] - ub[less]
            ge = lb[greater] - f[greater]
            il = f[interval] - ub[interval]
            ig = lb[interval] - f[interval]
            return eq, mx.hstack((le, ge, il, ig))

        def jac(x):
            J = cfun.jac(x)
            eq = J[equal]
            le = J[less]
            ge = -J[greater]
            il = J[interval]
            ig = -il
            if sps.issparse(J):
                ineq = sps.vstack((le, ge, il, ig))
            else:
                ineq = mx.vstack((le, ge, il, ig))
            return eq, ineq

        def hess(x, v_eq, v_ineq):
            n_start = 0
            v_l = v_ineq[n_start:n_start + n_less]
            n_start += n_less
            v_g = v_ineq[n_start:n_start + n_greater]
            n_start += n_greater
            v_il = v_ineq[n_start:n_start + n_interval]
            n_start += n_interval
            v_ig = v_ineq[n_start:n_start + n_interval]

            v = mx.zeros_like(lb)
            v[equal] = v_eq
            v[less] = v_l
            v[greater] = -v_g
            v[interval] = v_il - v_ig

            return cfun.hess(x, v)

        return cls(n_eq, n_ineq, fun, jac, hess, keep_feasible)


def initial_constraints_as_canonical(n, prepared_constraints, sparse_jacobian):
    """Convert initial values of the constraints to the canonical format.

    The purpose to avoid one additional call to the constraints at the initial
    point. It takes saved values in `PreparedConstraint`, modifies and
    concatenates them to the canonical constraint format.
    """
    c_eq = []
    c_ineq = []
    J_eq = []
    J_ineq = []

    for c in prepared_constraints:
        f = c.fun.f
        J = c.fun.J
        lb, ub = c.bounds
        if mx.all(lb == ub):
            c_eq.append(f - lb)
            J_eq.append(J)
        elif mx.all(lb == -mx.inf):
            finite_ub = ub < mx.inf
            c_ineq.append(f[finite_ub] - ub[finite_ub])
            J_ineq.append(J[finite_ub])
        elif mx.all(ub == mx.inf):
            finite_lb = lb > -mx.inf
            c_ineq.append(lb[finite_lb] - f[finite_lb])
            J_ineq.append(-J[finite_lb])
        else:
            lb_inf = lb == -mx.inf
            ub_inf = ub == mx.inf
            equal = lb == ub
            less = lb_inf & ~ub_inf
            greater = ub_inf & ~lb_inf
            interval = ~equal & ~lb_inf & ~ub_inf

            c_eq.append(f[equal] - lb[equal])
            c_ineq.append(f[less] - ub[less])
            c_ineq.append(lb[greater] - f[greater])
            c_ineq.append(f[interval] - ub[interval])
            c_ineq.append(lb[interval] - f[interval])

            J_eq.append(J[equal])
            J_ineq.append(J[less])
            J_ineq.append(-J[greater])
            J_ineq.append(J[interval])
            J_ineq.append(-J[interval])

    c_eq = mx.hstack(c_eq) if c_eq else mx.empty(0)
    c_ineq = mx.hstack(c_ineq) if c_ineq else mx.empty(0)

    if sparse_jacobian:
        vstack = sps.vstack
        empty = sps.csr_array((0, n))
    else:
        vstack = mx.vstack
        empty = mx.empty((0, n))

    J_eq = vstack(J_eq) if J_eq else empty
    J_ineq = vstack(J_ineq) if J_ineq else empty

    return c_eq, c_ineq, J_eq, J_ineq
