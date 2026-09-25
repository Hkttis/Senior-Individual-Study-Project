import math

import numpy as np
import pytest

from library.directional_objectives import (
    DIRECTIONAL_OBJECTIVE_EXACT,
    DIRECTIONAL_OBJECTIVE_SECTOR,
    exact_direction_pair_energy_force,
    mean_absolute_nominal_angular_deviation,
    normalize_directional_objective,
    oriented_angular_error,
)


def _energy(r, q, weight):
    return exact_direction_pair_energy_force(r, q, weight)[0]


def test_directional_objective_aliases_and_validation():
    assert normalize_directional_objective("sector-based") == DIRECTIONAL_OBJECTIVE_SECTOR
    assert normalize_directional_objective("zero-tolerance") == DIRECTIONAL_OBJECTIVE_EXACT
    with pytest.raises(ValueError, match="Unknown directional objective"):
        normalize_directional_objective("vector_magic")


def test_exact_direction_zero_error_has_zero_energy_and_force():
    energy, force_u, force_v, phi = exact_direction_pair_energy_force(
        [2.0, 0.0], [1.0, 0.0], 7.0
    )
    assert phi == pytest.approx(0.0)
    assert energy == pytest.approx(0.0)
    assert force_u == pytest.approx([0.0, 0.0])
    assert force_v == pytest.approx([0.0, 0.0])


def test_exact_direction_forces_are_equal_opposite_and_angle_symmetric():
    q_plus = [math.cos(math.pi / 6), math.sin(math.pi / 6)]
    q_minus = [math.cos(-math.pi / 6), math.sin(-math.pi / 6)]
    plus = exact_direction_pair_energy_force([1.0, 0.0], q_plus, 3.0)
    minus = exact_direction_pair_energy_force([1.0, 0.0], q_minus, 3.0)
    assert plus[2] == pytest.approx(-plus[1])
    assert minus[2] == pytest.approx(-minus[1])
    assert plus[0] == pytest.approx(minus[0])
    assert plus[1] == pytest.approx(-minus[1])


def test_exact_direction_force_matches_finite_difference_gradient_in_r():
    r = np.asarray([1.7, -0.4], dtype=float)
    q = np.asarray([math.cos(0.8), math.sin(0.8)], dtype=float)
    weight = 2.3
    _value, force_u, _force_v, _phi = exact_direction_pair_energy_force(r, q, weight)
    h = 1e-6
    finite_difference = np.empty(2, dtype=float)
    for axis in range(2):
        step = np.zeros(2, dtype=float)
        step[axis] = h
        finite_difference[axis] = (
            _energy(r + step, q, weight) - _energy(r - step, q, weight)
        ) / (2.0 * h)
    # For r = x_v - x_u, force_u equals d(psi)/d(r).
    assert force_u == pytest.approx(finite_difference, rel=1e-6, abs=1e-7)


def test_exact_direction_matches_zero_tolerance_sector_coefficient():
    r = np.asarray([1.3, 0.7], dtype=float)
    q = np.asarray([0.0, 1.0], dtype=float)
    weight = 4.0
    _energy_value, exact_force_u, _force_v, phi = exact_direction_pair_energy_force(r, q, weight)
    distance = float(np.linalg.norm(r))
    j_r = np.asarray([-r[1], r[0]], dtype=float)
    zero_tolerance_hinge = abs(phi) * (1.0 if phi >= 0.0 else -1.0)
    sector_force_u = -weight * zero_tolerance_hinge * j_r / (distance * distance)
    assert exact_force_u == pytest.approx(sector_force_u)


def test_exact_direction_energy_is_scale_invariant_and_force_scales_inverse_distance():
    r = np.asarray([1.0, 2.0], dtype=float)
    q = np.asarray([-1.0, 1.0], dtype=float)
    first = exact_direction_pair_energy_force(r, q, 5.0)
    second = exact_direction_pair_energy_force(4.0 * r, q, 5.0)
    assert second[0] == pytest.approx(first[0])
    assert second[1] == pytest.approx(first[1] / 4.0)


def test_direction_collision_is_rejected():
    with pytest.raises(ValueError, match="collision"):
        exact_direction_pair_energy_force([0.0, 0.0], [1.0, 0.0], 1.0)


def test_nominal_angular_deviation_uses_each_direction_observation():
    points = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=float)
    dni = {"A": 0, "B": 1, "C": 2}
    observations = [["A", "B", "東"], ["A", "C", "北"]]
    result = mean_absolute_nominal_angular_deviation(points, observations, dni)
    assert result == pytest.approx(math.pi / 8.0)
    assert oriented_angular_error([1.0, 0.0], [0.0, 1.0]) == pytest.approx(math.pi / 2.0)
