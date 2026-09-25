import math
import numpy as np
import pytest
from library.scipy_exact_objective import ExactDirectionObjective, build_current_exact_objective
from library.scipy_objective import ObjectiveWeights, ObjectiveDomainError, build_current_objective
from library.directional_objectives import exact_direction_pair_energy_force


def test_exact_inside_sector_has_penalty_and_matches_physics_force():
    p = ExactDirectionObjective(vertices=['a','b'], distance_pairs=[], distance_targets=[],
        direction_pairs=[[0,1]], direction_vectors=[[1,0]], direction_half_widths=[math.pi/2],
        anchor_positions={0:[0,0]}, weights=ObjectiveWeights(0,3,0))
    y = np.array([2.,1.])
    energy, _, force = exact_direction_pair_energy_force(y,[1,0],3)[:3]
    assert p.fun(y) == pytest.approx(energy)
    np.testing.assert_allclose(p.jac(y), -force)
    assert p.fun(y) > 0
    assert p.fun([2,0]) == 0
    np.testing.assert_array_equal(p.jac([2,0]), [0,0])
    with pytest.raises(ObjectiveDomainError):
        p.fun([-2,0])


def test_full_exact_gradient_central_difference_and_fixed_anchors():
    p = build_current_exact_objective(weights=ObjectiveWeights(1.7,2.3,.9))
    rng = np.random.default_rng(18)
    y = rng.normal(0,80,p.dimension)
    analytic = p.jac(y)
    numeric = np.empty_like(y)
    for i in range(len(y)):
        h=1e-4
        delta=np.zeros_like(y);delta[i]=h
        numeric[i]=(p.fun(y+delta)-p.fun(y-delta))/(2*h)
    assert np.linalg.norm(analytic-numeric)/np.linalg.norm(analytic) < 1e-7
    np.testing.assert_array_equal(p.unpack(y)[p.anchor_indices],p.anchor_coordinates)


def test_sector_unchanged_and_only_direction_component_differs():
    sector=build_current_objective()
    exact=build_current_exact_objective()
    y=np.random.default_rng(7).normal(0,80,sector.dimension)
    a,b=sector.components(y),exact.components(y)
    assert a.distance == b.distance
    assert a.repulsion == b.repulsion
    assert b.direction >= a.direction
    assert exact.weights == sector.weights
    np.testing.assert_array_equal(exact.unpack(y),sector.unpack(y))
