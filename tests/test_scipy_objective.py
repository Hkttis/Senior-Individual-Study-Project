import math

import numpy as np
import pytest
from scipy.optimize import check_grad, minimize

from library.scipy_objective import (
    FIXED_ANCHORS_SIM,
    FixedAnchorObjective,
    ObjectiveDomainError,
    ObjectiveWeights,
    PHYSICS_HPO_SELECTED_ALPHA,
    PHYSICS_HPO_SELECTED_BETA,
    PHYSICS_HPO_SELECTED_W_DIS,
    build_current_objective,
)


def _synthetic_problem(*, weights=None):
    return FixedAnchorObjective(
        vertices=("A", "B", "C"),
        distance_pairs=((0, 1),),
        distance_targets=(1.0,),
        direction_pairs=((0, 2),),
        direction_vectors=((1.0, 0.0),),
        direction_half_widths=(math.pi / 4.0,),
        anchor_positions={0: (0.0, 0.0)},
        weights=weights or ObjectiveWeights(2.0, 3.0, 5.0),
        epsilon=0.1,
    )


def test_current_dataset_contract_and_reduced_dimension():
    problem = build_current_objective()

    assert problem.n_vertices == 35
    assert len(problem.distance_pairs) == 44
    assert len(problem.direction_pairs) == 44
    assert len(problem.repulsion_pairs) == math.comb(35, 2)
    assert problem.n_free_vertices == 32
    assert problem.dimension == 64

    anchor_by_name = {
        problem.vertices[index]: tuple(position)
        for index, position in zip(
            problem.anchor_indices, problem.anchor_coordinates, strict=True
        )
    }
    assert anchor_by_name == dict(FIXED_ANCHORS_SIM)


def test_default_weights_reproduce_selected_existing_physics_experiment():
    expected = ObjectiveWeights.from_physics_hpo(
        alpha=PHYSICS_HPO_SELECTED_ALPHA,
        beta=PHYSICS_HPO_SELECTED_BETA,
        w_dis=PHYSICS_HPO_SELECTED_W_DIS,
    )
    defaults = ObjectiveWeights()
    problem = build_current_objective()

    assert defaults == expected
    assert problem.weights == expected
    assert expected.distance == pytest.approx(1500.0)
    assert expected.direction == pytest.approx(10_000_000.0)
    assert expected.repulsion == pytest.approx(158.11388300841898)


def test_unpack_inserts_fixed_numeric_anchors_and_pack_excludes_them():
    problem = build_current_objective()
    free = np.arange(problem.dimension, dtype=float) + 1000.0
    full = problem.unpack(free)

    for index, expected in zip(
        problem.anchor_indices, problem.anchor_coordinates, strict=True
    ):
        assert full[index] == pytest.approx(expected, abs=0.0)
    assert problem.pack(full) == pytest.approx(free, abs=0.0)


def test_components_match_sections_4_3_formula_term_by_term():
    problem = _synthetic_problem()
    # B=(2,0): distance residual is 1.  C=(0,2): eastward direction is
    # violated by pi/2-pi/4 = pi/4.
    free = np.asarray([2.0, 0.0, 0.0, 2.0])
    components = problem.components(free)

    expected_distance = 0.5
    expected_direction = 0.5 * (math.pi / 4.0) ** 2
    lengths = np.asarray([2.0, 2.0, math.sqrt(8.0)])
    expected_repulsion = float(np.sum(-np.log(lengths + 0.1) - 0.1 / (lengths + 0.1)))

    assert components.distance == pytest.approx(expected_distance)
    assert components.direction == pytest.approx(expected_direction)
    assert components.repulsion == pytest.approx(expected_repulsion)
    assert components.weighted_distance == pytest.approx(2.0 * expected_distance)
    assert components.weighted_direction == pytest.approx(3.0 * expected_direction)
    assert components.weighted_repulsion == pytest.approx(5.0 * expected_repulsion)
    assert components.total == pytest.approx(
        2.0 * expected_distance
        + 3.0 * expected_direction
        + 5.0 * expected_repulsion
    )


def test_analytic_gradient_matches_central_finite_difference():
    problem = build_current_objective(
        weights=ObjectiveWeights(distance=1.7, direction=2.3, repulsion=0.9)
    )
    rng = np.random.default_rng(20260821)
    full = rng.normal(loc=0.0, scale=80.0, size=(problem.n_vertices, 2))
    full[problem.anchor_indices] = problem.anchor_coordinates
    free = problem.pack(full)

    analytic = problem.jac(free)
    step = 1e-6
    numeric = np.empty_like(analytic)
    for index in range(free.size):
        plus = free.copy()
        minus = free.copy()
        plus[index] += step
        minus[index] -= step
        numeric[index] = (problem.fun(plus) - problem.fun(minus)) / (2.0 * step)

    scale = max(1.0, float(np.linalg.norm(analytic)), float(np.linalg.norm(numeric)))
    relative_error = float(np.linalg.norm(analytic - numeric)) / scale
    assert relative_error < 2e-7


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
def test_check_grad_validates_each_term_in_a_smooth_region(weights):
    problem = _synthetic_problem(weights=weights)
    # This point is separated from both collisions, the direction hinge
    # boundary, and the antipodal angular branch boundary.
    free = np.asarray([2.0, 0.5, 0.4, 2.0], dtype=np.float64)

    error = float(check_grad(problem.fun, problem.jac, free, epsilon=1e-7))
    scale = max(1.0, float(np.linalg.norm(problem.jac(free))))

    assert error / scale < 2e-7


def test_scipy_minimize_needs_no_constraints_and_cannot_move_anchor():
    problem = FixedAnchorObjective(
        vertices=("anchor", "free"),
        distance_pairs=((0, 1),),
        distance_targets=(3.0,),
        direction_pairs=(),
        direction_vectors=(),
        direction_half_widths=(),
        anchor_positions={0: (7.0, -2.0)},
        weights=ObjectiveWeights(distance=1.0, direction=0.0, repulsion=0.0),
    )

    result = minimize(problem.fun_and_jac, np.asarray([11.0, -2.0]), jac=True, method="BFGS")
    full = problem.unpack(result.x)

    assert result.success
    assert full[0] == pytest.approx((7.0, -2.0), abs=0.0)
    assert np.linalg.norm(full[1] - full[0]) == pytest.approx(3.0, abs=1e-8)
    assert result.fun == pytest.approx(0.0, abs=1e-12)


def test_current_35_node_problem_runs_unconstrained_scipy_and_decreases_objective():
    problem = build_current_objective()
    rng = np.random.default_rng(623)
    full = rng.normal(loc=0.0, scale=80.0, size=(problem.n_vertices, 2))
    full[problem.anchor_indices] = problem.anchor_coordinates
    initial = problem.pack(full)
    initial_value = problem.fun(initial)

    result = minimize(
        problem.fun_and_jac,
        initial,
        jac=True,
        method="BFGS",
        options={"maxiter": 5},
    )
    reconstructed = problem.unpack(result.x)

    assert result.fun < initial_value
    assert np.array_equal(
        reconstructed[problem.anchor_indices], problem.anchor_coordinates
    )


def test_manuscript_singular_direction_configuration_is_rejected():
    problem = _synthetic_problem()
    # C lies exactly opposite the prescribed eastward vector.  This is the
    # excluded antipodal atan2 branch of the differentiable formulation.
    with pytest.raises(ObjectiveDomainError, match="antipodal"):
        problem.fun(np.asarray([2.0, 0.0, -2.0, 0.0]))
