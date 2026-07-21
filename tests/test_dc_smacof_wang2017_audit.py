import numpy as np

from experiments.dc_smacof_wang2017_audit.run_audit import (
    build_problem,
    compute_targets,
    objective_components,
    run_iterations,
)


def _two_node_direction_problem(direction="北"):
    labels = ["A", "B"]
    dni = {"A": 0, "B": 1}
    return build_problem(
        labels,
        dni,
        [["A", "B", 100.0]],
        [["A", "B", direction]],
        distance_weight=1e-12,
        direction_weight=1.0,
    )


def test_wang_direction_target_uses_current_length_and_incidence_sign():
    problem = _two_node_direction_problem()
    positions = np.asarray([[0.0, 0.0], [30.0, 40.0]])
    _dw, dv = compute_targets(problem, positions, "wang_current")
    assert np.allclose(dv[0], [0.0, -50.0])


def test_production_proxy_uses_text_distance_instead_of_current_length():
    problem = _two_node_direction_problem()
    positions = np.asarray([[0.0, 0.0], [30.0, 40.0]])
    _dw, dv = compute_targets(problem, positions, "production_proxy")
    assert np.allclose(dv[0], [0.0, -100.0])


def test_wang_one_step_places_target_in_requested_direction_without_changing_length():
    problem = _two_node_direction_problem()
    positions, trace, status, reason = run_iterations(
        problem, mode="wang_current", seed=4, n_iterations=1, damping=1.0
    )
    initial = np.random.RandomState(4).rand(2, 2)
    initial_length = np.linalg.norm(initial[1] - initial[0])
    final_vector = positions[1] - positions[0]
    assert status == "ok", reason
    assert np.isclose(np.linalg.norm(final_vector), initial_length, rtol=1e-8, atol=1e-8)
    assert final_vector[1] > 0
    assert abs(final_vector[0]) < 1e-8
    assert all(row["finite"] for row in trace)


def test_direction_objective_is_zero_for_aligned_edge():
    problem = _two_node_direction_problem()
    positions = np.asarray([[0.0, 0.0], [0.0, 75.0]])
    _distance, direction = objective_components(problem, positions)
    assert direction == 0.0


def test_small_wang_run_stays_finite():
    labels = ["A", "B", "C", "D"]
    dni = {label: index for index, label in enumerate(labels)}
    distances = [
        ["A", "B", 100.0],
        ["B", "C", 100.0],
        ["C", "D", 100.0],
        ["D", "A", 100.0],
        ["A", "C", 2**0.5 * 100.0],
    ]
    directions = [["A", "B", "東"], ["B", "C", "北"], ["C", "D", "西"], ["D", "A", "南"]]
    problem = build_problem(labels, dni, distances, directions, direction_weight=0.31622776601683794)
    _positions, trace, status, reason = run_iterations(
        problem, mode="wang_current", seed=0, n_iterations=100, damping=1.0
    )
    assert status == "ok", reason
    assert all(row["finite"] for row in trace)
