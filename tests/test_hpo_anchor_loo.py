import math

import numpy as np
import pytest

from library.initialization import construct_Chen_graph
from run_paper_script.ch5_hparam_kfold_gridsearch_pareto import (
    _build_anchor_loo_folds,
    _euclidean_rmse_km,
    _non_dominated_mask,
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
