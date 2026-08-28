"""Reproducible SciPy BFGS runner for the fixed-anchor objective.

The baseline policy is intentionally strict: if SciPy evaluates a point where
the manuscript objective/gradient is undefined, including a trial point during
the Wolfe line search, that run fails immediately.  No ``+inf`` sentinel or
dummy gradient is supplied to the line search.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
from scipy.optimize import minimize

from library.scipy_objective import (
    FixedAnchorObjective,
    InvalidObjectiveEvaluation,
)


# The analytic objective is piecewise smooth at direction-sector boundaries.
# A 1e-3 infinity-norm tolerance avoids false precision-loss failures while
# leaving the accepted solution unchanged at the reported metric precision.
DEFAULT_BFGS_GTOL = 1e-3
DEFAULT_BFGS_MAXITER_PER_DIMENSION = 200


def objective_and_gradient(
    y: Sequence[float], problem: FixedAnchorObjective
) -> tuple[float, np.ndarray]:
    """Return a finite float64 ``(value, gradient)`` pair for SciPy."""

    vector = np.asarray(y, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError("BFGS variable y must be a 1-D vector.")
    if vector.shape != (problem.dimension,):
        raise ValueError(
            f"BFGS variable y must have shape {(problem.dimension,)}, "
            f"got {vector.shape}."
        )
    if not np.all(np.isfinite(vector)):
        raise InvalidObjectiveEvaluation("Non-finite BFGS variable.")

    value, gradient = problem.fun_and_jac(vector)
    value = float(value)
    gradient = np.asarray(gradient, dtype=np.float64).reshape(-1)

    if not np.isfinite(value):
        raise InvalidObjectiveEvaluation("Non-finite objective value.")
    if gradient.shape != vector.shape:
        raise InvalidObjectiveEvaluation(
            f"Analytic gradient has shape {gradient.shape}, expected {vector.shape}."
        )
    if not np.all(np.isfinite(gradient)):
        raise InvalidObjectiveEvaluation("Non-finite analytic gradient.")
    return value, gradient


def _invalid_run_result(reason: str) -> dict[str, Any]:
    return {
        "success": False,
        "failure_reason": f"invalid_objective_evaluation: {reason}",
        "scipy_success": False,
        "scipy_status": None,
        "scipy_message": None,
        "y_final": None,
        "objective_final": None,
        "gradient_norm": None,
        "gradient_tolerance": None,
        "iterations": None,
        "function_evaluations": None,
        "gradient_evaluations": None,
    }


def run_bfgs(
    y0: Sequence[float],
    problem: FixedAnchorObjective,
    *,
    gtol: float = DEFAULT_BFGS_GTOL,
    maxiter: int | None = None,
    callback: Callable[[np.ndarray], None] | None = None,
) -> dict[str, Any]:
    """Run unconstrained analytic-gradient BFGS with a strict domain policy.

    ``xrtol`` is fixed at zero.  A run is reported successful only when SciPy
    reports success *and* the returned infinity norm of the gradient is no
    larger than ``gtol``.
    """

    initial = np.asarray(y0, dtype=np.float64)
    if initial.ndim != 1:
        raise ValueError("BFGS initial variable y0 must be a 1-D vector.")
    if initial.shape != (problem.dimension,):
        raise ValueError(
            f"BFGS initial variable y0 must have shape {(problem.dimension,)}, "
            f"got {initial.shape}."
        )
    if not np.all(np.isfinite(initial)):
        return _invalid_run_result("Non-finite BFGS initial variable.")
    if not np.isfinite(gtol) or gtol <= 0.0:
        raise ValueError("gtol must be finite and strictly positive.")

    resolved_maxiter = (
        DEFAULT_BFGS_MAXITER_PER_DIMENSION * initial.size
        if maxiter is None
        else int(maxiter)
    )
    if resolved_maxiter <= 0:
        raise ValueError("maxiter must be a strictly positive integer.")

    try:
        result = minimize(
            objective_and_gradient,
            initial,
            args=(problem,),
            method="BFGS",
            jac=True,
            callback=callback,
            options={
                "gtol": float(gtol),
                "norm": np.inf,
                "xrtol": 0.0,
                "maxiter": resolved_maxiter,
                "c1": 1e-4,
                "c2": 0.9,
                "disp": False,
                "return_all": False,
            },
        )
    except InvalidObjectiveEvaluation as exc:
        return _invalid_run_result(str(exc))

    final_y = np.asarray(result.x, dtype=np.float64).reshape(initial.shape)
    final_gradient = np.asarray(result.jac, dtype=np.float64).reshape(initial.shape)
    final_value = float(result.fun)
    if (
        not np.all(np.isfinite(final_y))
        or not np.isfinite(final_value)
        or not np.all(np.isfinite(final_gradient))
    ):
        return _invalid_run_result("SciPy returned a non-finite final result.")

    gradient_norm = float(np.linalg.norm(final_gradient, ord=np.inf))
    gradient_converged = gradient_norm <= float(gtol)
    success = bool(result.success) and gradient_converged
    if success:
        failure_reason = None
    elif result.success:
        failure_reason = (
            "gradient_tolerance_not_met: "
            f"||gradient||_inf={gradient_norm:.17g} > gtol={gtol:.17g}"
        )
    else:
        failure_reason = str(result.message)

    return {
        "success": success,
        "failure_reason": failure_reason,
        "scipy_success": bool(result.success),
        "scipy_status": int(result.status),
        "scipy_message": str(result.message),
        "y_final": final_y,
        "objective_final": final_value,
        "gradient_norm": gradient_norm,
        "gradient_tolerance": float(gtol),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "gradient_evaluations": int(result.njev),
    }


__all__ = [
    "DEFAULT_BFGS_GTOL",
    "DEFAULT_BFGS_MAXITER_PER_DIMENSION",
    "objective_and_gradient",
    "run_bfgs",
]
