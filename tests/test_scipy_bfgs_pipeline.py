import numpy as np
import pytest

from library.config import FILE_PATHS, refer_pos_sim
from library.data_io import (
    get_anchor_align_label,
    get_anchor_labels,
    load_ini_data_from_csv,
    uploading_ground_truth,
)
from library.scipy_objective import build_current_objective
from run_paper_script.ch5_scipy_bfgs import (
    _initial_free_vector,
    _parse_seeds,
    _positions_y_up,
    _snapshot_indices,
)
from scripts.check_scipy_bfgs_results import _problem_from_config


def test_parse_seeds_rejects_empty_and_duplicates():
    assert _parse_seeds("0, 2,5") == [0, 2, 5]
    with pytest.raises(ValueError, match="empty"):
        _parse_seeds("")
    with pytest.raises(ValueError, match="duplicate"):
        _parse_seeds("0,0")


def test_initialization_matches_formal_graph_and_keeps_all_anchors_exact():
    problem = build_current_objective()
    _graph, vertice, dni, _edges, _data = load_ini_data_from_csv(FILE_PATHS)
    gt_lonlat = uploading_ground_truth(vertice, dni)
    labels = get_anchor_labels()
    lonlat = [tuple(gt_lonlat[dni[label]]) for label in labels]
    initial = _initial_free_vector(
        0,
        problem,
        vertice,
        dni,
        labels,
        lonlat,
        get_anchor_align_label(),
    )
    centered = _positions_y_up(problem, initial) - np.asarray(refer_pos_sim)

    assert initial.shape == (64,)
    assert np.all(np.isfinite(initial))
    assert centered[problem.anchor_indices] == pytest.approx(
        problem.anchor_coordinates, abs=1e-10
    )


def test_snapshot_selection_includes_both_endpoints_and_is_bounded():
    selected = _snapshot_indices(101, count=6)
    assert selected[0] == 0
    assert selected[-1] == 100
    assert len(selected) == 6
    assert _snapshot_indices(3, count=6) == [0, 1, 2]


def test_result_checker_reconstructs_hpo_selected_objective_weights():
    problem = _problem_from_config({"alpha": 0.5, "beta": -0.5, "w_dis": 1.0})

    assert problem.weights.distance == pytest.approx(1500.0)
    assert problem.weights.direction == pytest.approx(10_000_000.0 * 10**-0.5)
    assert problem.weights.repulsion == pytest.approx(500.0 * 10**-0.5)
