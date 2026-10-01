"""Opt-in SciPy Anderson phases and matrix-free Newton recovery.

Callers keep the mesh and physical controls fixed during each solve. SciPy
owns the Anderson history; FreeGSNKE owns switching and final certification.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import root


@dataclass(frozen=True)
class AndersonOptions:
    """Settings for SciPy Anderson acceleration and Newton recovery.

    ``history``, ``damping`` and ``regularization`` map to SciPy's ``M``,
    ``alpha`` and ``w0``. For F(x) = x - G(x), SciPy receives -F so its initial
    step is damped Picard. ``max_backtracks`` bounds Newton/Picard trials;
    Anderson uses SciPy's Armijo search, which can accept uphill steps when
    the search fails. ``recovery_iterations`` bounds each Anderson phase.
    """

    history: int = 8
    damping: float = 0.15
    regularization: float = 0.01
    max_backtracks: int = 12
    stagnation_iterations: int = 3
    stagnation_ratio: float = 0.95
    recovery_iterations: int = 50

    def __post_init__(self):
        """Reject invalid settings before any residual evaluation."""
        for name in (
            "history",
            "max_backtracks",
            "stagnation_iterations",
            "recovery_iterations",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(self.damping) or not 0 < self.damping <= 1:
            raise ValueError("damping must be in (0, 1]")
        if not np.isfinite(self.regularization) or self.regularization < 0:
            raise ValueError("regularization must be finite and nonnegative")
        if not np.isfinite(self.stagnation_ratio) or not 0 < self.stagnation_ratio <= 1:
            raise ValueError("stagnation_ratio must be in (0, 1]")


class _Converged(Exception):
    """Stop a SciPy phase when the full FreeGSNKE criterion is met."""


def safeguarded_solve(
    x,
    residual,
    newton_step,
    relative_error,
    *,
    options,
    tolerance,
    max_iterations,
    handover,
    max_rel_update_size,
    project=lambda x: x,
):
    """Switch between Newton and bounded SciPy Anderson phases.

    SciPy success alone never certifies convergence: the full residual and
    caller's tolerance are checked independently, including under projection.
    Valid callback iterates can temporarily increase the residual to build
    Anderson history. An unsuccessful phase restores its best full-norm
    iterate. Geometry errors abort that phase; bounded damped Picard is then
    attempted if no progress was made. Unexpected errors propagate after
    restoring residual-derived state at the retained field.

    ``max_iterations`` counts both Newton and Anderson iterations. The update
    size cap applies to Newton and fallback Picard; SciPy manages its own steps.
    """
    if not isinstance(options, AndersonOptions):
        raise TypeError("options must be AndersonOptions")
    x = project(np.array(x, dtype=float, copy=True))
    errors, norms, relative_norms, events = [], [], [], []
    slow = evaluations = iteration = 0
    reason = "iteration_limit"
    recoverable = (ValueError, RuntimeError, IndexError, np.linalg.LinAlgError)

    def evaluate(point):
        """Evaluate a finite full residual, retaining its derived state."""
        nonlocal evaluations
        evaluations += 1
        value = np.asarray(residual(point), dtype=float)
        if value.shape != point.shape or not np.all(np.isfinite(value)):
            raise ValueError("Residual must have the field shape and finite values")
        return value

    def record(value, point):
        """Record comparable full-residual diagnostics for both methods."""
        errors.append(float(relative_error(value, point)))
        norms.append(float(np.linalg.norm(value)))
        relative_norms.append(
            float(
                np.linalg.norm(value) / max(np.linalg.norm(point), np.finfo(float).tiny)
            )
        )

    def trial(step, f):
        """Backtrack a projected, size-limited Newton or Picard step."""
        step = project(np.asarray(step, dtype=float))
        if step.shape != x.shape or not np.all(np.isfinite(step)):
            evaluate(x)
            return None
        span = max(np.ptp(x), np.finfo(float).eps * max(1.0, np.max(abs(x))))
        size = np.max(abs(step))
        if size > max_rel_update_size * span:
            step = step * (max_rel_update_size * span / size)
        for k in range(options.max_backtracks):
            fraction = 0.5**k
            point = project(x + fraction * step)
            try:
                value = evaluate(point)
                if np.linalg.norm(value) <= (1 - 1e-4 * fraction) * np.linalg.norm(f):
                    return point, value, k
            except recoverable:
                pass
            evaluate(x)
        return None

    f = evaluate(x)
    record(f, x)
    try:
        while iteration < max_iterations:
            error = float(relative_error(f, x))
            if error <= tolerance:
                reason = "converged"
                break
            use_anderson = error > handover or slow >= options.stagnation_iterations
            accepted = None
            if not use_anderson:
                try:
                    proposal = newton_step(x.copy(), f.copy())
                    evaluate(x)
                    accepted = trial(proposal, f)
                except recoverable as exc:
                    events.append(
                        dict(
                            iteration=iteration,
                            event="newton_failure",
                            message=str(exc),
                        )
                    )
                    evaluate(x)
                if accepted is None:
                    events.append(dict(iteration=iteration, event="anderson_recovery"))
            if accepted is None:
                # Keep the best physical iterate outside SciPy's private state.
                best_x, best_f = x.copy(), f.copy()
                initial_norm = np.linalg.norm(f)
                phase_start = iteration
                phase_converged = False

                def scipy_residual(point):
                    """Use SciPy's sign convention for a Picard initial step."""
                    return -project(evaluate(project(point)))

                def callback(point, value):
                    """Retain valid iterates and certify the full residual."""
                    nonlocal x, f, best_x, best_f, iteration, phase_converged
                    point = project(point.copy())
                    full = evaluate(point)
                    iteration += 1
                    record(full, point)
                    events.append(
                        dict(
                            iteration=iteration - 1,
                            event="accepted",
                            method="scipy_anderson",
                        )
                    )
                    if np.linalg.norm(full) < np.linalg.norm(best_f):
                        best_x, best_f = point.copy(), full.copy()
                    # This also ensures restoration if an unexpected error follows.
                    x, f = best_x.copy(), best_f.copy()
                    if relative_error(full, point) <= tolerance:
                        x, f = point, full
                        phase_converged = True
                        raise _Converged()

                try:
                    result = root(
                        scipy_residual,
                        x.copy(),
                        method="anderson",
                        callback=callback,
                        options=dict(
                            maxiter=min(
                                options.recovery_iterations, max_iterations - iteration
                            ),
                            fatol=0.0,
                            line_search="armijo",
                            jac_options=dict(
                                alpha=options.damping,
                                M=options.history,
                                w0=options.regularization,
                            ),
                        ),
                    )
                    events.append(
                        dict(
                            iteration=iteration,
                            event="anderson_end",
                            message=str(result.message),
                        )
                    )
                except _Converged:
                    pass
                except recoverable as exc:
                    events.append(
                        dict(
                            iteration=iteration,
                            event="anderson_failure",
                            message=str(exc),
                        )
                    )
                finally:
                    if not phase_converged:
                        x, f = best_x, best_f
                    f = evaluate(x)
                if phase_converged:
                    reason = "converged"
                    break
                # Failed startup still consumes one attempt in the global budget.
                if iteration == phase_start:
                    iteration += 1
                slow = 0
                if np.linalg.norm(f) < initial_norm:
                    continue  # retry NK below handover, otherwise another phase
                if iteration >= max_iterations:
                    break
                accepted = trial(-options.damping * project(f), f)
                method = "damped_picard"
            else:
                method = "newton"
            if accepted is None:
                reason = "stalled"
                events.append(dict(iteration=iteration, event="stalled"))
                break
            point, value, backtracks = accepted
            slow = (
                slow + 1
                if np.linalg.norm(value) >= options.stagnation_ratio * np.linalg.norm(f)
                else 0
            )
            events.append(
                dict(
                    iteration=iteration,
                    event="accepted",
                    method=method,
                    backtracks=backtracks,
                )
            )
            x, f = point, value
            iteration += 1
            record(f, x)
    finally:
        f = evaluate(x)
    error = float(relative_error(f, x))
    if error <= tolerance:
        reason = "converged"
    return (
        x,
        f,
        dict(
            reason=reason,
            relative_error=error,
            residual_history=errors,
            norm_history=norms,
            relative_norm_history=relative_norms,
            evaluations=evaluations,
            iterations=iteration,
            events=events,
        ),
    )
