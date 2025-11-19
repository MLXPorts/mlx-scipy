'''
This module contains subroutines concerning the geometry-improving of the interpolation set.

Translated from Zaikun Zhang's modern-Fortran reference implementation in PRIMA.

Dedicated to late Professor M. J. D. Powell FRS (1936--2015).

Python translation by Nickolai Belakovski.
'''

from ..common.consts import DEBUGGING
from ..common.linalg import isinv, matprod, inprod, norm, primasum, primapow2
import mlx.core as mx


def setdrop_tr(ximproved, d, delta, rho, sim, simi):
    '''
    This function finds (the index) of a current interpolation point to be replaced with
    the trust-region trial point. See (19)-(22) of the COBYLA paper.
    N.B.:
    1. If XIMPROVED == True, then JDROP > 0 so that D is included into XPT. Otherwise,
       it is a bug.
    2. COBYLA never sets JDROP = NUM_VARS
    TODO: Check whether it improves the performance if JDROP = NUM_VARS is allowed when
    XIMPROVED is True. Note that UPDATEXFC should be revised accordingly.
    '''

    # Local variables
    itol = 0.1

    # Sizes
    num_vars = mx.size(sim, 0)

    # Preconditions
    if DEBUGGING:
        assert num_vars >= 1
        assert mx.size(d) == num_vars and all(mx.isfinite(d))
        assert delta >= rho and rho > 0
        assert mx.size(sim, 0) == num_vars and mx.size(sim, 1) == num_vars + 1
        assert mx.isfinite(sim).all()
        assert all(mx.max(abs(sim[:, :num_vars]), axis=0) > 0)
        assert mx.size(simi, 0) == num_vars and mx.size(simi, 1) == num_vars
        assert mx.isfinite(simi).all()
        assert isinv(sim[:, :num_vars], simi, itol)

    #====================#
    # Calculation starts #
    #====================#

    # -------------------------------------------------------------------------------------------------- #
    #  The following code is Powell's scheme for defining JDROP.
    # -------------------------------------------------------------------------------------------------- #
    # ! JDROP = 0 by default. It cannot be removed, as JDROP may not be set below in some cases (e.g.,
    # ! when XIMPROVED == FALSE, MAXVAL(ABS(SIMID)) <= 1, and MAXVAL(VETA) <= EDGMAX).
    # jdrop = 0
    #
    # ! SIMID(J) is the value of the J-th Lagrange function at D. It is the counterpart of VLAG in UOBYQA
    # ! and DEN in NEWUOA/BOBYQA/LINCOA, but it excludes the value of the (N+1)-th Lagrange function.
    # simid = matprod(simi, d)
    # if (any(abs(simid) > 1) .or. (ximproved .and. any(.not. is_nan(simid)))) then
    #     jdrop = int(maxloc(abs(simid), mask=(.not. is_nan(simid)), dim=1), kind(jdrop))
    #     !!MATLAB: [~, jdrop] = max(simid, [], 'omitnan');
    # end if
    #
    # ! VETA(J) is the distance from the J-th vertex of the simplex to the best vertex, taking the trial
    # ! point SIM(:, N+1) + D into account.
    # if (ximproved) then
    #     veta = sqrt(sum((sim(:, 1:n) - spread(d, dim=2, ncopies=n))**2, dim=1))
    #     !!MATLAB: veta = sqrt(sum((sim(:, 1:n) - d).^2));  % d should be a column! Implicit expansion
    # else
    #     veta = sqrt(sum(sim(:, 1:n)**2, dim=1))
    # end if
    #
    # ! VSIG(J) (J=1, .., N) is the Euclidean distance from vertex J to the opposite face of the simplex.
    # vsig = ONE / sqrt(sum(simi**2, dim=2))
    # sigbar = abs(simid) * vsig
    #
    # ! The following JDROP will overwrite the previous one if its premise holds.
    # mask = (veta > factor_delta * delta .and. (sigbar >= factor_alpha * delta .or. sigbar >= vsig))
    # if (any(mask)) then
    #     jdrop = int(maxloc(veta, mask=mask, dim=1), kind(jdrop))
    #     !!MATLAB: etamax = max(veta(mask)); jdrop = find(mask & ~(veta < etamax), 1, 'first');
    # end if
    #
    # ! Powell's code does not include the following instructions. With Powell's code, if SIMID consists
    # ! of only NaN, then JDROP can be 0 even when XIMPROVED == TRUE (i.e., D reduces the merit function).
    # ! With the following code, JDROP cannot be 0 when XIMPROVED == TRUE, unless VETA is all NaN, which
    # ! should not happen if X0 does not contain NaN, the trust-region/geometry steps never contain NaN,
    # ! and we exit once encountering an iterate containing Inf (due to overflow).
    # if (ximproved .and. jdrop <= 0) then  ! Write JDROP <= 0 instead of JDROP == 0 for robustness.
    #     jdrop = int(maxloc(veta, mask=(.not. is_nan(veta)), dim=1), kind(jdrop))
    #     !!MATLAB: [~, jdrop] = max(veta, [], 'omitnan');
    # end if
    # -------------------------------------------------------------------------------------------------- #
    #  Powell's scheme ends here.
    # -------------------------------------------------------------------------------------------------- #

    # The following definition of JDROP is inspired by SETDROP_TR in UOBYQA/NEWUOA/BOBYQA/LINCOA.
    # It is simpler and works better than Powell's scheme. Note that we allow JDROP to be NUM_VARS+1 if
    # XIMPROVED is True, whereas Powell's code does not.
    # See also (4.1) of Scheinberg-Toint-2010: Self-Correcting Geometry in Model-Based Algorithms for
    # Derivative-Free Unconstrained Optimization, which refers to the strategy here as the "combined
    # distance/poisedness criteria".

    # DISTSQ[j] is the square of the distance from the jth vertex of the simplex to get "best" point so
    # far, taking the trial point SIM[:, NUM_VARS] + D into account.
    distsq = mx.zeros(mx.size(sim, 1))
    if ximproved:
        distsq[:num_vars] = primasum(primapow2(sim[:, :num_vars] - mx.tile(d, (num_vars, 1)).T), axis=0)
        distsq[num_vars] = primasum(d*d)
    else:
        distsq[:num_vars] = primasum(primapow2(sim[:, :num_vars]), axis=0)
        distsq[num_vars] = 0

    weight = mx.maximum(1, distsq / primapow2(mx.maximum(rho, delta/10)))  # Similar to Powell's NEWUOA code.

    # Other possible definitions of weight. They work almost the same as the one above.
    # weight = distsq  # Similar to Powell's LINCOA code, but WRONG. See comments in LINCOA/geometry.f90.
    # weight = max(1, max(25 * distsq / delta**2))  # Similar to Powell's BOBYQA code, works well.
    # weight = max(1, max(10 * distsq / delta**2))
    # weight = max(1, max(1e2 * distsq / delta**2))
    # weight = max(1, max(distsq / rho**2))  ! Similar to Powell's UOBYQA

    # If 0 <= j < NUM_VARS, SIMID[j] is the value of the jth Lagrange function at D; the value of the
    # (NUM_VARS+1)th Lagrange function is 1 - sum(SIMID). [SIMID, 1 - sum(SIMID)] is the counterpart of
    # VLAG in UOBYQA and DEN in NEWUOA/BOBYQA/LINCOA.
    simid = matprod(simi, d)
    score = weight * abs(mx.array([*simid, 1 - primasum(simid)]))

    # If XIMPROVED = False (D does not render a better X), set SCORE[NUM_VARS] = -1 to avoid JDROP = NUM_VARS.
    if not ximproved:
        score[num_vars] = -1

    # score[j] is NaN implies SIMID[j] is NaN, but we want abs(SIMID) to be big. So we
    # exclude such j.
    score[mx.isnan(score)] = -1

    jdrop = None
    # The following if statement works a bit better than
    # `if any(score > 1) or (any(score > 0) and ximproved)` from Powell's UOBYQA and
    # NEWUOA code.
    if any(score > 0):  # Powell's BOBYQA and LINCOA code.
        jdrop = mx.argmax(score)

    if (ximproved and jdrop is None):
        jdrop = mx.argmax(distsq)

    #==================#
    # Calculation ends #
    #==================#

    # Postconditions
    if DEBUGGING:
        assert jdrop is None or (0 <= jdrop < num_vars + 1)
        assert jdrop <= num_vars or ximproved
        assert jdrop >= 0 or not ximproved
        # JDROP >= 1 when XIMPROVED = TRUE unless NaN occurs in DISTSQ, which should not happen if the
        # starting point does not contain NaN and the trust-region/geometry steps never contain NaN.

    return jdrop




def geostep(jdrop, amat, bvec, conmat, cpen, cval, delbar, fval, simi):
    '''
    This function calculates a geometry step so that the geometry of the interpolation set is improved
    when SIM[: JDROP_GEO] is replaced with SIM[:, NUM_VARS] + D. See (15)--(17) of the COBYLA paper.
    '''

    # Sizes
    m_lcon = mx.size(bvec, 0) if bvec is not None else 0
    num_constraints = mx.size(conmat, 0)
    num_vars = mx.size(simi, 0)

    # Preconditions
    if DEBUGGING:
        assert num_constraints >= m_lcon >= 0
        assert num_vars >= 1
        assert delbar > 0
        assert cpen > 0
        assert mx.size(simi, 0) == num_vars and mx.size(simi, 1) == num_vars
        assert mx.isfinite(simi).all()
        assert mx.size(fval) == num_vars + 1 and not any(mx.isnan(fval) | mx.isposinf(fval))
        assert mx.size(conmat, 0) == num_constraints and mx.size(conmat, 1) == num_vars + 1
        assert not mx.any(mx.isnan(conmat) | mx.isposinf(conmat))
        assert mx.size(cval) == num_vars + 1 and not any(cval < 0 | mx.isnan(cval) | mx.isposinf(cval))
        assert 0 <= jdrop < num_vars

    #====================#
    # Calculation starts #
    #====================#

    # SIMI[JDROP, :] is a vector perpendicular to the face of the simplex to the opposite of vertex
    # JDROP. Set D to the vector in this direction and with length DELBAR.
    d = simi[jdrop, :]
    d = delbar * (d / norm(d))

    # The code below chooses the direction of D according to an approximation of the merit function.
    # See (17) of the COBYLA paper and  line 225 of Powell's cobylb.f.

    # Calculate the coefficients of the linear approximations to the objective and constraint functions.
    # N.B.: CONMAT and SIMI have been updated after the last trust-region step, but G and A have not.
    # So we cannot pass G and A from outside.
    g = matprod(fval[:num_vars] - fval[num_vars], simi)
    A = mx.zeros((num_vars, num_constraints))
    A[:, :m_lcon] = amat.T if amat is not None else amat
    A[:, m_lcon:] = matprod((conmat[m_lcon:, :num_vars] -
                          mx.tile(conmat[m_lcon:, num_vars], (num_vars, 1)).T), simi).T
    # CVPD and CVND are the predicted constraint violation of D and -D by the linear models.
    cvpd = mx.max(mx.append(0, conmat[:, num_vars] + matprod(d, A)))
    cvnd = mx.max(mx.append(0, conmat[:, num_vars] - matprod(d, A)))
    if -inprod(d, g) + cpen * cvnd < inprod(d, g) + cpen * cvpd:
        d *= -1

    #==================#
    # Calculation ends #
    #==================#

    # Postconditions
    if DEBUGGING:
        assert mx.size(d) == num_vars and all(mx.isfinite(d))
        # In theory, ||S|| == DELBAR, which may be false due to rounding, but not too far.
        # It is crucial to ensure that the geometry step is nonzero, which holds in theory.
        assert 0.9 * delbar < mx.linalg.norm(d) <= 1.1 * delbar
    return d
