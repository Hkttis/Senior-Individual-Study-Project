import numpy as np
import pandas as pd
import pytest

from library.config import refer_pos_sim
from library.scipy_diagnostics import (
    assign_objective_strata,
    graph_node_diagnostics,
    reinsert_exact_anchors,
    test_site_radial_errors as compute_test_site_radial_errors,
    weighted_gradient_components,
)
from library.scipy_objective import build_current_objective
from run_paper_script.ch5_physics_bfgs_polishing import (
    _reject_external_polishing_hpo,
    _sample_steps,
    _source_objective_weights,
)


def test_reinsert_exact_anchors_preserves_free_coordinates_and_removes_anchor_drift():
    problem = build_current_objective()
    centered = np.arange(problem.n_vertices * 2, dtype=float).reshape(-1, 2)
    centered[problem.anchor_indices] = problem.anchor_coordinates + 0.25
    y_up = centered + np.asarray(refer_pos_sim, dtype=float)

    corrected, drift = reinsert_exact_anchors(y_up, problem)

    assert drift == pytest.approx(0.25)
    assert np.array_equal(corrected[problem.free_indices], centered[problem.free_indices])
    assert np.array_equal(corrected[problem.anchor_indices], problem.anchor_coordinates)


def test_weighted_gradient_components_sum_to_formal_total_gradient():
    problem = build_current_objective()
    rng = np.random.default_rng(10)
    y = rng.uniform(10.0, 100.0, size=problem.dimension)
    _value, reduced_total = problem.fun_and_jac(y)

    components = weighted_gradient_components(problem, y)

    expected = np.zeros((problem.n_vertices, 2), dtype=float)
    expected[problem.free_indices] = reduced_total.reshape(problem.n_free_vertices, 2)
    assert np.allclose(components["total"], expected, rtol=1e-12, atol=1e-8)


def test_assign_objective_strata_uses_largest_reference_gaps():
    reference = [-10, -9, -8, 10, 11, 30, 50, 51]

    strata, thresholds = assign_objective_strata([-20, 0, 20, 60], reference, 4)

    assert strata == [1, 1, 2, 4]
    assert thresholds == pytest.approx([1.0, 20.5, 40.0])


def test_sample_steps_includes_initial_and_final_once():
    assert _sample_steps(10, 4) == [0, 4, 8, 10]
    assert _sample_steps(10, 5) == [0, 5, 10]


def test_polishing_recovers_exact_formal_as_weights():
    runs = pd.DataFrame(
        {
            "spring_stiffness": [1500.0, 1500.0],
            "directional_force": [10_000_000.0, 10_000_000.0],
            "repulsion_strength": [158.11388300841898, 158.11388300841898],
        }
    )

    weights, alpha, beta, w_dis = _source_objective_weights(
        runs, {"alpha": 1.0, "beta": -0.5}
    )

    assert weights.distance == pytest.approx(1500.0)
    assert weights.direction == pytest.approx(10_000_000.0)
    assert weights.repulsion == pytest.approx(158.11388300841898)
    assert (alpha, beta, w_dis) == pytest.approx((1.0, -0.5, 1.0))


def test_polishing_rejects_independent_bfgs_hpo():
    with pytest.raises(ValueError, match="Independent BFGS HPO is not permitted"):
        _reject_external_polishing_hpo("outputs/bfgs_hpo", False)


def test_polishing_rejects_as_config_weight_mismatch():
    runs = pd.DataFrame(
        {
            "spring_stiffness": [1500.0],
            "directional_force": [10_000_000.0],
            "repulsion_strength": [158.11388300841898],
        }
    )

    with pytest.raises(ValueError, match="do not reproduce"):
        _source_objective_weights(runs, {"alpha": 0.5, "beta": -0.5})


def test_radial_error_is_positive_for_outward_displacement():
    problem = build_current_objective()
    label = problem.vertices[int(problem.free_indices[0])]
    index = problem.vertices.index(label)
    centroid = problem.anchor_coordinates.mean(axis=0)
    target = centroid + np.array([20.0, 0.0])
    centered = np.zeros((problem.n_vertices, 2), dtype=float)
    centered[index] = target + np.array([10.0, 0.0])

    errors = compute_test_site_radial_errors(
        centered_positions=centered,
        target_centered={label: target},
        test_labels=[label],
        dni={label: index},
        problem=problem,
    )

    from library.config import km2pix

    assert errors[label] == pytest.approx(10.0 / km2pix)


def test_graph_diagnostics_count_observations_and_anchor_hops():
    result = graph_node_diagnostics(
        vertices=["A", "B", "C", "D"],
        dni={"A": 0, "B": 1, "C": 2, "D": 3},
        distance_rows=[["A", "B", "1"], ["B", "C", "1"]],
        direction_rows=[["A", "B", "E"], ["A", "B", "NE"], ["C", "D", "S"]],
        anchor_labels=["A"],
    )

    assert result["A"] == {
        "distance_edge_degree": 1,
        "direction_edge_degree": 2,
        "distance_graph_hops_to_nearest_anchor": 0,
    }
    assert result["C"]["distance_graph_hops_to_nearest_anchor"] == 2
    assert result["D"]["distance_graph_hops_to_nearest_anchor"] == -1
