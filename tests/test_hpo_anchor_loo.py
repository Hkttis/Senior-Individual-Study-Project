import math

import numpy as np
import pandas as pd
import pytest

from library.initialization import construct_Chen_graph
from run_paper_script.ch5_hparam_kfold_gridsearch_pareto import (
    _build_anchor_loo_folds,
    _euclidean_rmse_km,
    _non_dominated_mask,
    _select_one_se_balanced_candidate,
)


def test_build_anchor_loo_folds_uses_two_train_one_heldout():
    labels = ["A", "B", "C"]
    lonlat = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]

    folds = _build_anchor_loo_folds(labels, lonlat)

    assert len(folds) == 3
    assert folds[0].heldout_label == "A"
    assert folds[0].train_labels == ["B", "C"]
    assert folds[0].train_lonlat == [(2.0, 2.0), (3.0, 3.0)]
    assert folds[0].train_anchor_label == "B"

    assert folds[1].heldout_label == "B"
    assert folds[1].train_labels == ["A", "C"]

    assert folds[2].heldout_label == "C"
    assert folds[2].train_labels == ["A", "B"]


def test_build_anchor_loo_folds_requires_exactly_three_anchors():
    with pytest.raises(ValueError):
        _build_anchor_loo_folds(["A", "B"], [(1.0, 1.0), (2.0, 2.0)])


def test_construct_chen_graph_preserves_first_seen_node_order():
    data = [["B", "A", "1"], ["C", "B", "2"], ["A", "D", "3"]]

    _graph, vertice, dni, edges = construct_Chen_graph(data)

    assert vertice == ["B", "A", "C", "D"]
    assert dni == {"B": 0, "A": 1, "C": 2, "D": 3}
    assert edges == [("B", "A"), ("C", "B"), ("A", "D")]


def test_non_dominated_mask_minimizes_all_objectives():
    points = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [1.0, 2.0, 0.5],
            [0.5, 2.0, 2.0],
            [math.nan, 1.0, 1.0],
        ]
    )

    mask = _non_dominated_mask(points)

    assert mask.tolist() == [True, False, True, True, False]


def test_euclidean_rmse_single_label_is_distance_error():
    dni = {"A": 0}
    pred_km = [(3.0, 4.0)]
    gt_km = [(0.0, 0.0)]

    rmse = _euclidean_rmse_km(pred_km=pred_km, gt_km=gt_km, eval_labels=["A"], dni=dni)

    assert rmse == pytest.approx(5.0)


def test_euclidean_rmse_multiple_labels_uses_mean_square_distance():
    dni = {"A": 0, "B": 1}
    pred_km = [(3.0, 4.0), (0.0, 0.0)]
    gt_km = [(0.0, 0.0), (0.0, 0.0)]

    rmse = _euclidean_rmse_km(pred_km=pred_km, gt_km=gt_km, eval_labels=["A", "B"], dni=dni)

    assert rmse == pytest.approx(math.sqrt(25.0 / 2.0))


def test_euclidean_rmse_raises_on_missing_ground_truth():
    dni = {"A": 0}
    pred_km = [(3.0, 4.0)]
    gt_km = [(None, None)]

    with pytest.raises(ValueError):
        _euclidean_rmse_km(pred_km=pred_km, gt_km=gt_km, eval_labels=["A"], dni=dni)


def test_select_one_se_balanced_candidate_prefers_balanced_candidate_within_threshold():
    df = pd.DataFrame(
        [
            {
                "alpha": 1.0,
                "beta": -1.0,
                "n_folds": 3,
                "E_distance_stress_mean": 0.030,
                "E_direction_vr_mean": 0.016,
                "RMSE_anchor_LOO_mean_km": 147.9,
                "RMSE_anchor_LOO_std_km": 9.9,
            },
            {
                "alpha": 1.0,
                "beta": -0.5,
                "n_folds": 3,
                "E_distance_stress_mean": 0.027,
                "E_direction_vr_mean": 0.017,
                "RMSE_anchor_LOO_mean_km": 148.4,
                "RMSE_anchor_LOO_std_km": 9.8,
            },
            {
                "alpha": 1.0,
                "beta": 0.5,
                "n_folds": 3,
                "E_distance_stress_mean": 0.021,
                "E_direction_vr_mean": 0.035,
                "RMSE_anchor_LOO_mean_km": 161.7,
                "RMSE_anchor_LOO_std_km": 49.5,
            },
        ]
    )

    selected, meta = _select_one_se_balanced_candidate(
        df,
        ["E_distance_stress_mean", "E_direction_vr_mean", "RMSE_anchor_LOO_mean_km"],
    )

    assert selected["alpha"] == pytest.approx(1.0)
    assert selected["beta"] == pytest.approx(-0.5)
    assert meta["selection_rule"] == "pareto_one_se_balanced"
    assert meta["one_se_candidate_count"] == 2
    assert meta["one_se_threshold_km"] == pytest.approx(147.9 + 9.9 / math.sqrt(3))
