import math
from types import SimpleNamespace

import numpy as np
import pytest

import library.scipy_minimizer as scipy_minimizer
from library.scipy_minimizer import objective_and_gradient, run_bfgs
from library.scipy_objective import (
    FixedAnchorObjective,
    InvalidObjectiveEvaluation,
    ObjectiveDomainError,
    ObjectiveWeights,
)


def _smooth_problem(weights: ObjectiveWeights) -> FixedAnchorObjective:
    return FixedAnchorObjective(
        vertices=("A", "B", "C"),
        distance_pairs=((0, 1),),
        distance_targets=(1.3,),
        direction_pairs=((0, 2),),
        direction_vectors=((1.0, 0.0),),
        direction_half_widths=(math.pi / 4.0,),
        anchor_positions={0: (0.0, 0.0)},
        weights=weights,
        epsilon=0.1,
    )


def _single_free_vertex_problem(
    *,
    distance: bool = False,
    direction: bool = False,
    repulsion_weight: float = 0.0,
) -> FixedAnchorObjective:
    return FixedAnchorObjective(
        vertices=("anchor", "free"),
        distance_pairs=((0, 1),) if distance else (),
        distance_targets=(3.0,) if distance else (),
        direction_pairs=((0, 1),) if direction else (),
        direction_vectors=((1.0, 0.0),) if direction else (),
        direction_half_widths=(math.pi / 4.0,) if direction else (),
        anchor_positions={0: (7.0, -2.0)},
        weights=ObjectiveWeights(
            distance=1.0 if distance else 0.0,
            direction=1.0 if direction else 0.0,
            repulsion=repulsion_weight,
        ),
        epsilon=0.1,
    )


def test_objective_and_gradient_enforces_float64_scalar_vector_contract():
    problem = _smooth_problem(ObjectiveWeights(1.7, 2.3, 0.9))
    y32 = np.asarray([2.0, 0.5, 0.4, 2.0], dtype=np.float32)

    value, gradient = objective_and_gradient(y32, problem)

    assert isinstance(value, float)
    assert gradient.dtype == np.float64
    assert gradient.shape == y32.shape
    assert np.isfinite(value)
    assert np.all(np.isfinite(gradient))


def test_objective_and_gradient_uses_one_combined_problem_evaluation():
    class CountingProblem:
        dimension = 2

        def __init__(self):
            self.calls = 0

        def fun_and_jac(self, y):
            self.calls += 1
            return np.dot(y, y), 2.0 * y

    problem = CountingProblem()
    value, gradient = objective_and_gradient(np.asarray([1.0, -2.0]), problem)

    assert problem.calls == 1
    assert value == pytest.approx(5.0)
    assert gradient == pytest.approx((2.0, -4.0))


@pytest.mark.parametrize(
    "weights",
    [
        ObjectiveWeights(1.0, 0.0, 0.0),
        ObjectiveWeights(0.0, 1.0, 0.0),
        ObjectiveWeights(0.0, 0.0, 1.0),
        ObjectiveWeights(1.7, 2.3, 0.9),
    ],
    ids=("distance", "direction", "repulsion", "total"),
)
def test_directional_derivative_converges_in_smooth_region(weights):
    problem = _smooth_problem(weights)
    y = np.asarray([2.0, 0.5, 0.4, 2.0], dtype=np.float64)
    _value, gradient = objective_and_gradient(y, problem)
    rng = np.random.default_rng(20260821)

    for _ in range(5):
        direction = rng.normal(size=y.size)
        direction /= np.linalg.norm(direction)
        analytic = float(np.dot(gradient, direction))
        errors = []
        for step in (1e-2, 1e-3, 1e-4, 1e-5):
            plus = problem.fun(y + step * direction)
            minus = problem.fun(y - step * direction)
            numeric = (plus - minus) / (2.0 * step)
            scale = max(1.0, abs(analytic), abs(numeric))
            errors.append(abs(numeric - analytic) / scale)

        assert errors[2] < errors[0]
        assert min(errors[1:]) < 2e-7


def test_bfgs_converges_without_constraints_and_keeps_anchor_exact():
    problem = _single_free_vertex_problem(distance=True)
    result = run_bfgs(np.asarray([11.0, -2.0], dtype=np.float32), problem)

    assert result["success"]
    assert result["failure_reason"] is None
    assert result["y_final"].dtype == np.float64
    assert result["gradient_norm"] <= result["gradient_tolerance"]
    full = problem.unpack(result["y_final"])
    assert np.array_equal(full[problem.anchor_indices], problem.anchor_coordinates)
    assert np.linalg.norm(full[1] - full[0]) == pytest.approx(3.0, abs=1e-8)


@pytest.mark.parametrize(
    "problem",
    [
        _single_free_vertex_problem(distance=True),
        _single_free_vertex_problem(direction=True),
    ],
    ids=("distance_collision", "direction_collision"),
)
def test_undefined_collision_immediately_fails_run(problem):
    result = run_bfgs(np.asarray([7.0, -2.0]), problem)

    assert not result["success"]
    assert result["failure_reason"].startswith("invalid_objective_evaluation:")
    assert result["y_final"] is None


def test_invalid_line_search_trial_point_immediately_fails_run(monkeypatch):
    problem = _single_free_vertex_problem(direction=True)

    def fake_minimize(fun, x0, args, **_kwargs):
        fun(x0, *args)
        # Simulate a Wolfe line-search trial that places the free endpoint on
        # the anchor, where the direction angle and gradient are undefined.
        fun(np.asarray([7.0, -2.0]), *args)
        raise AssertionError("The invalid trial evaluation must propagate.")

    monkeypatch.setattr(scipy_minimizer, "minimize", fake_minimize)
    result = run_bfgs(np.asarray([8.0, -2.0]), problem)

    assert not result["success"]
    assert result["failure_reason"].startswith("invalid_objective_evaluation:")
    assert result["y_final"] is None


def test_repulsion_collision_is_finite_and_not_rejected():
    problem = _single_free_vertex_problem(repulsion_weight=1.0)
    y = np.asarray([7.0, -2.0])

    value, gradient = objective_and_gradient(y, problem)

    assert np.isfinite(value)
    assert np.array_equal(gradient, np.zeros_like(gradient))


def test_nonfinite_initial_vector_is_reported_as_invalid_run():
    problem = _single_free_vertex_problem(distance=True)
    result = run_bfgs(np.asarray([np.nan, -2.0]), problem)

    assert not result["success"]
    assert result["failure_reason"] == (
        "invalid_objective_evaluation: Non-finite BFGS initial variable."
    )


def test_wrong_vector_rank_is_a_programming_error():
    problem = _single_free_vertex_problem(distance=True)
    with pytest.raises(ValueError, match="1-D"):
        objective_and_gradient(np.asarray([[11.0, -2.0]]), problem)


def test_reported_success_still_requires_gradient_tolerance(monkeypatch):
    problem = _single_free_vertex_problem(distance=True)
    captured = {}

    def fake_minimize(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            success=True,
            status=0,
            message="synthetic small-step success",
            x=np.asarray([10.0, -2.0]),
            fun=0.0,
            jac=np.asarray([1e-3, 0.0]),
            nit=1,
            nfev=2,
            njev=2,
        )

    monkeypatch.setattr(scipy_minimizer, "minimize", fake_minimize)
    result = run_bfgs(np.asarray([11.0, -2.0]), problem, gtol=1e-6)

    assert captured["method"] == "BFGS"
    assert captured["jac"] is True
    assert captured["options"]["gtol"] == pytest.approx(1e-6)
    assert captured["options"]["norm"] == np.inf
    assert captured["options"]["xrtol"] == 0.0
    assert captured["options"]["maxiter"] == 200 * problem.dimension
    assert captured["options"]["c1"] == pytest.approx(1e-4)
    assert captured["options"]["c2"] == pytest.approx(0.9)
    assert captured["options"]["disp"] is False
    assert captured["options"]["return_all"] is False
    assert not result["success"]
    assert result["scipy_success"]
    assert result["failure_reason"].startswith("gradient_tolerance_not_met:")


def test_bfgs_forwards_iteration_callback(monkeypatch):
    problem = _single_free_vertex_problem(distance=True)
    received = []

    def callback(xk):
        received.append(np.asarray(xk).copy())

    def fake_minimize(*args, **kwargs):
        kwargs["callback"](np.asarray([10.0, -2.0]))
        return SimpleNamespace(
            success=True,
            status=0,
            message="synthetic convergence",
            x=np.asarray([10.0, -2.0]),
            fun=0.0,
            jac=np.asarray([0.0, 0.0]),
            nit=1,
            nfev=2,
            njev=2,
        )

    monkeypatch.setattr(scipy_minimizer, "minimize", fake_minimize)
    result = run_bfgs(
        np.asarray([11.0, -2.0]), problem, callback=callback
    )

    assert result["success"]
    assert len(received) == 1
    assert received[0] == pytest.approx((10.0, -2.0))


def test_domain_error_is_an_invalid_objective_evaluation():
    assert issubclass(ObjectiveDomainError, InvalidObjectiveEvaluation)
    assert issubclass(ObjectiveDomainError, ValueError)
