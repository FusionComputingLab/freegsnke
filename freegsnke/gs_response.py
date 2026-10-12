"""Implicit odd susceptibility of a symmetric, smoothly tapered equilibrium."""

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres


def solve_odd_response(
    inverse, coefficient, vacuum_response, reflection, rtol=1e-9, maxiter=20
):
    """Solve (I - Po G C Po)v = Po G C g, then dj = Po C(v+g).

    inverse(j) is the linear free-boundary plasma-flux solve G, coefficient is
    dj/dpsi in A m^-2 per (Wb/rad), and vacuum_response is Wb/rad per ampere.
    The returned plasma_response and current_response are per forcing ampere.
    Inputs must describe the SAME frozen equilibrium, mesh and coil geometry.

    Valid only for an even base, odd forcing, an equivariant G and the smooth
    odd source derivative. Scalar critical-flux/normalization terms project out
    here; this is NOT their omission from a general Jacobian. A finite-current
    sharp interface or double-null switching needs separate treatment/checks.
    Caller must verify the full equilibrium residual and operator symmetry.
    """
    # C is the local source derivative at one frozen equilibrium. The caller
    # supplies its profile physics; this solver only couples C to the inverse G.
    c = np.asarray(coefficient, dtype=float)
    g = np.asarray(vacuum_response, dtype=float)
    p = np.asarray(reflection)
    if c.ndim != 1 or g.shape != c.shape or p.shape != c.shape:
        raise ValueError("Expected equal-length one-dimensional arrays")
    # Applying reflection twice must recover the original vertex ordering.
    # Otherwise Po below is not a projection onto odd fields.
    n = len(c)
    if (
        not np.issubdtype(p.dtype, np.integer)
        or np.any(p < 0)
        or np.any(p >= n)
        or not np.array_equal(p[p], np.arange(n))
    ):
        raise ValueError("Reflection must be an involutive permutation")
    if not np.all(np.isfinite(c)) or not np.all(np.isfinite(g)):
        raise ValueError("Nonfinite response input")
    if not np.allclose(c, c[p], rtol=1e-12, atol=0):
        raise ValueError("Source coefficient must be even")
    if np.linalg.norm(g + g[p]) > 1e-12 * max(np.linalg.norm(g), 1e-300):
        raise ValueError("Vacuum forcing must be odd")
    # Po removes even roundoff components on every inverse application. For
    # even C and a reflection-equivariant G, the odd subspace is closed.
    odd = lambda x: (x - x[p]) / 2
    # Linearise v = G[C(v+g)] and move the unknown plasma term to the left.
    # Matrix-free actions avoid storing a dense susceptibility matrix.
    operator = LinearOperator(
        (n, n), matvec=lambda x: x - odd(inverse(c * odd(x))), dtype=float
    )
    rhs = odd(inverse(c * g))
    # Restart limits the Krylov basis storage; maxiter counts restart cycles.
    # Check the equation ourselves as well as the library convergence flag.
    v, info = gmres(
        operator, rhs, rtol=rtol, atol=0, restart=min(40, n), maxiter=maxiter
    )
    v = odd(v)
    residual = np.linalg.norm(operator @ v - rhs) / max(np.linalg.norm(rhs), 1e-300)
    if info or not np.isfinite(residual) or residual > 10 * rtol:
        raise RuntimeError(
            f"Odd tangent did not converge: info={info}, residual={residual}"
        )
    # v is plasma flux only; the current responds to total flux v+g. Both
    # responses are per applied ampere, with no circuit model imposed here.
    return dict(
        plasma_response=v,
        current_response=odd(c * (v + g)),
        relative_residual=float(residual),
    )
