import numpy as np
import pandas as pd
import pytest

from library.config import refer_pos_sim
from library.scipy_objective import ObjectiveWeights, build_current_objective
from scripts.analyze_anchor_objective_rmse import _components, _correlation, _split_problem


def test_split_objective_uses_requested_anchors_and_weights():
    base = build_current_objective()
    anchors = {0: np.asarray([1.0, 2.0]), 1: np.asarray([3.0, 4.0]), 2: np.asarray([5.0, 6.0])}
    weights = ObjectiveWeights.from_physics_hpo(alpha=0.0, beta=-1.0)

    problem = _split_problem(base, anchors, weights)

    assert problem.anchor_indices.tolist() == [0, 1, 2]
    assert problem.weights == weights
    assert np.array_equal(problem.distance_pairs, base.distance_pairs)


def test_components_reinsert_exact_anchors_before_objective_evaluation():
    problem = build_current_objective()
    rng = np.random.default_rng(5)
    centered = rng.uniform(0.0, 100.0, size=(problem.n_vertices, 2))
    centered[problem.anchor_indices] = problem.anchor_coordinates
    expected = problem.components(problem.pack(centered)).total
    drifted = centered + np.asarray(refer_pos_sim)
    drifted[problem.anchor_indices] += 0.25

    result, drift = _components(problem, drifted)

    assert drift == pytest.approx(0.25)
    assert result.total == pytest.approx(expected)


def test_spearman_correlation_detects_inverse_objective_rmse_order():
    result = _correlation(pd.Series([1.0, 2.0, 3.0]), pd.Series([3.0, 2.0, 1.0]))

    assert result["n"] == 3
    assert result["spearman_rho"] == pytest.approx(-1.0)
