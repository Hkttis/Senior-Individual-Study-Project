import math

import numpy as np
import pytest

from library.directional_objectives import (
    DIRECTIONAL_OBJECTIVE_SOFT_EXACT,
    normalize_directional_objective,
    softened_exact_direction_pair_energy_force,
)
from library.scipy_objective import ObjectiveDomainError, ObjectiveWeights
from library.scipy_soft_exact_objective import SoftenedExactDirectionObjective
from run_paper_script.ch5_sector_soft_exact_control import (
    SOFT_EXACT_FORMAL_GRID,
    _validate_soft_exact_formal_protocol,
)


def _energy(r, q, weight, delta):
    return softened_exact_direction_pair_energy_force(
        r, q, weight, delta=delta
    )[0]


def test_softened_exact_alias_and_parameter_validation():
    assert normalize_directional_objective("softened-exact") == DIRECTIONAL_OBJECTIVE_SOFT_EXACT
    with pytest.raises(ValueError, match="delta"):
        softened_exact_direction_pair_energy_force([1, 0], [1, 0], 1, delta=0)


def test_softened_exact_pair_matches_expanded_objective():
    r = np.asarray([1.7, -0.4])
    q = np.asarray([math.cos(0.8), math.sin(0.8)])
    weight = 2.3
    delta = 0.1
    value, force_u, force_v, phi = softened_exact_direction_pair_energy_force(
        r, q, weight, delta=delta
    )
    s2 = float(np.dot(r, r))
    expected = 0.5 * weight * (
        s2 * phi**2 + delta**2 * math.pi**2
    ) / (s2 + delta**2)
    assert value == pytest.approx(expected)
    np.testing.assert_allclose(force_v, -force_u)


def test_softened_exact_force_matches_central_difference_gradient():
    r = np.asarray([1.7, -0.4], dtype=float)
    q = np.asarray([math.cos(0.8), math.sin(0.8)], dtype=float)
    weight = 2.3
    delta = 0.1
    _, force_u, _, _ = softened_exact_direction_pair_energy_force(
        r, q, weight, delta=delta
    )
    numeric = np.empty(2)
    h = 1e-6
    for axis in range(2):
        step = np.zeros(2)
        step[axis] = h
        numeric[axis] = (
            _energy(r + step, q, weight, delta)
            - _energy(r - step, q, weight, delta)
        ) / (2 * h)
    np.testing.assert_allclose(force_u, numeric, rtol=1e-6, atol=1e-7)


def test_collision_extension_is_finite_and_has_zero_force():
    weight = 3.0
    value, force_u, force_v, phi = softened_exact_direction_pair_energy_force(
        [0, 0], [1, 0], weight, delta=0.1
    )
    assert value == pytest.approx(0.5 * weight * math.pi**2)
    np.testing.assert_array_equal(force_u, [0, 0])
    np.testing.assert_array_equal(force_v, [0, 0])
    assert math.isnan(phi)


def test_aligned_finite_edge_retains_radial_separation_term():
    value, force_u, force_v, phi = softened_exact_direction_pair_energy_force(
        [2, 0], [1, 0], 4.0, delta=0.1
    )
    assert phi == pytest.approx(0.0)
    assert value > 0.0
    assert force_u[0] < 0.0
    assert force_v[0] > 0.0


def test_softened_bfgs_objective_accepts_direction_collision_and_matches_pair_kernel():
    problem = SoftenedExactDirectionObjective(
        vertices=["a", "b"],
        distance_pairs=[],
        distance_targets=[],
        direction_pairs=[[0, 1]],
        direction_vectors=[[1, 0]],
        direction_half_widths=[math.pi / 8],
        anchor_positions={0: [0, 0]},
        weights=ObjectiveWeights(0, 3, 0),
        direction_softening_delta=0.1,
    )
    collision = np.asarray([0.0, 0.0])
    assert problem.fun(collision) == pytest.approx(1.5 * math.pi**2)
    np.testing.assert_array_equal(problem.jac(collision), [0, 0])

    y = np.asarray([2.0, 1.0])
    value, _force_u, force_v, _phi = softened_exact_direction_pair_energy_force(
        y, [1, 0], 3, delta=0.1
    )
    assert problem.fun(y) == pytest.approx(value)
    np.testing.assert_allclose(problem.jac(y), -force_v, rtol=1e-12, atol=1e-12)

    with pytest.raises(ObjectiveDomainError):
        problem.fun([-2.0, 0.0])


def test_softened_bfgs_full_gradient_matches_central_difference():
    problem = SoftenedExactDirectionObjective(
        vertices=["a", "b", "c"],
        distance_pairs=[[0, 1]],
        distance_targets=[2.0],
        direction_pairs=[[0, 2]],
        direction_vectors=[[1, 0]],
        direction_half_widths=[math.pi / 8],
        anchor_positions={0: [0, 0]},
        weights=ObjectiveWeights(1.7, 2.3, 0.9),
        direction_softening_delta=0.1,
    )
    y = np.asarray([1.5, 0.4, 0.8, 1.3])
    analytic = problem.jac(y)
    numeric = np.empty_like(y)
    h = 1e-6
    for index in range(len(y)):
        step = np.zeros_like(y)
        step[index] = h
        numeric[index] = (problem.fun(y + step) - problem.fun(y - step)) / (2 * h)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-6)


def test_softened_exact_formal_grid_is_expanded_only_on_lower_beta_boundary():
    alphas, betas = _validate_soft_exact_formal_protocol(
        hpo_seeds=tuple(range(10)),
        final_seeds=tuple(range(100)),
        **SOFT_EXACT_FORMAL_GRID,
        allow_smoke=False,
    )
    np.testing.assert_allclose(alphas, [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    np.testing.assert_allclose(
        betas, [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5]
    )


def test_softened_exact_formal_grid_rejects_the_previous_boundary_grid():
    with pytest.raises(ValueError, match="Formal softened-exact HPO grid"):
        _validate_soft_exact_formal_protocol(
            hpo_seeds=tuple(range(10)),
            final_seeds=tuple(range(100)),
            alpha_min=-1.0,
            alpha_max=1.5,
            alpha_step=0.5,
            beta_min=-2.0,
            beta_max=0.5,
            beta_step=0.5,
            allow_smoke=False,
        )
