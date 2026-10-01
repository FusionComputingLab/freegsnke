Example Documentation
=====================


Lorem ipsum

Optional Anderson recovery for forward equilibria
------------------------------------------------

For difficult fixed-current equilibria, the static solver supports opt-in
SciPy limited-memory Anderson acceleration of Picard steps and recovery after a
Newton--Krylov step is rejected or stagnates::

    from freegsnke.anderson import AndersonOptions

    solver.solve(
        eq, profiles,
        target_relative_tolerance=1e-7,
        anderson_options=AndersonOptions(history=8, damping=0.15),
    )

The same keyword is available on ``forward_solve``. Its default, ``None``,
retains the existing solver path. The initial implementation supports forward
solves only; supplying magnetic constraints together with this option raises
an error before changing the equilibrium.

The opt-in path uses accelerated Picard above ``Picard_handover`` and
Newton--Krylov below it. Rejected Newton steps, recoverable geometry failures,
or repeated slow residual reduction trigger a limited recovery window.
Newton backtracking and each Anderson phase are bounded. Anderson uses
``scipy.optimize.root(method="anderson")`` with Armijo line search. SciPy may
accept a full step when its line search fails, so intermediate residuals can
increase while history is built. This is necessary for some noncontractive
fixed points. An unsuccessful phase restores its best full-residual-norm
iterate; if it made no progress, bounded damped Picard is tried before returning
``stalled``. No full Jacobian is assembled. ``max_rel_update_size`` limits
Newton and fallback Picard updates; SciPy manages its own Anderson steps.

SciPy owns history within each phase; it is recreated after a method switch
and for each new solve. ``history``, ``damping`` and ``regularization`` map to
SciPy's ``M``, ``alpha`` and ``w0`` (defaults 8, 0.15 and 0.01).
``recovery_iterations`` defaults to 50 and caps one Anderson phase; the overall
``max_solving_iterations`` budget includes both methods. Keep mesh,
profiles and external currents fixed within a forward solve. Derived profile
state is recomputed at the accepted field after rejected evaluations and on
exit. Symmetry projection is applied only when explicitly requested with
``force_up_down_symmetric``; the convergence check still uses the full residual.

Inspect ``solver.anderson_diagnostics`` for ``reason`` (``converged``,
``stalled`` or ``iteration_limit``), residual histories, evaluation count and
accepted-step/recovery events. As with the existing forward API, the accepted
field is written into the equilibrium even when the iteration limit is reached;
check ``solver.relative_change`` against the requested tolerance. Anderson
parameters are configurable experiment settings, not guarantees of convergence.
Small nonlinear residuals do not establish spatial mesh convergence.

Geometry retries in the standard solver
---------------------------------------

The standard forward solver and inverse topology-preservation check retain
their existing 0.75 step reduction, with at most 32 trials. If no valid step
is found, or further shrinking cannot change the floating-point flux, they
restore the accepted equilibrium/profile state and raise ``RuntimeError``
with the underlying numerical error chained. Unexpected exceptions and user
interrupts propagate instead of triggering repeated step reduction.

For the standard forward path, ``best_psi`` and ``best_relative_change`` track
the lowest range-relative residual across all accepted iterates. They are
diagnostics, not a new return-selection policy: the established reduced-step
rollback rule is retained because inverse solves rely on short forward runs.
``eq.plasma_psi`` and ``relative_change`` always describe the returned field.
