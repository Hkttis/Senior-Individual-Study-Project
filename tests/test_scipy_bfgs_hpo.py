import numpy as np
import pandas as pd
import pytest

from library.config import FILE_PATHS
from library.data_io import load_ini_data_from_csv
from library.scipy_hpo_objective import build_bfgs_hpo_fold_objective
from library.scipy_objective import (
    FIXED_ANCHORS_SIM,
    ObjectiveWeights,
    build_current_objective,
)
from run_paper_script.ch5_hparam_kfold_gridsearch_pareto import _build_anchor_loo_folds
from run_paper_script.ch5_scipy_bfgs_hpo import (
    _anchor_inputs,
    _eligible_grid_points,
    _fold_targets_centered,
    load_selected_bfgs_hpo_params,
)


def test_hpo_fold_objective_has_two_anchors_without_mutating_formal_objective():
    labels = list(FIXED_ANCHORS_SIM)
    fold = build_bfgs_hpo_fold_objective(
        fixed_anchor_positions_sim={
            labels[0]: (0.0, 0.0),
            labels[1]: (10.0, 20.0),
        },
        weights=ObjectiveWeights.from_physics_hpo(alpha=0.0, beta=0.0),
    )

    assert len(fold.anchor_indices) == 2
    assert fold.dimension == 66

    formal = build_current_objective()
    assert len(formal.anchor_indices) == 3
    assert formal.dimension == 64


def test_hpo_fold_objective_rejects_test_site_or_any_non_calibration_label():
    with pytest.raises(ValueError, match="calibration anchors only"):
        build_bfgs_hpo_fold_objective(
            fixed_anchor_positions_sim={
                "鄯善": np.zeros(2),
                "龜茲": np.ones(2),
            },
            weights=ObjectiveWeights(),
        )


def test_hpo_fold_objective_requires_exactly_two_anchors():
    with pytest.raises(ValueError, match="exactly two"):
        build_bfgs_hpo_fold_objective(
            fixed_anchor_positions_sim={"鄯善": np.zeros(2)},
            weights=ObjectiveWeights(),
        )


def test_every_real_hpo_fold_holds_out_one_anchor_and_has_66_variables():
    _graph, vertices, dni, _edges, _data = load_ini_data_from_csv(FILE_PATHS)
    labels, lonlat, test_labels = _anchor_inputs(vertices, dni)
    assert len(test_labels) == 8
    folds = _build_anchor_loo_folds(labels, lonlat)

    for fold in folds:
        targets = _fold_targets_centered(
            vertices=vertices,
            dni=dni,
            anchor_labels=labels,
            anchor_lonlat=lonlat,
            frame_label=fold.train_anchor_label,
        )
        problem = build_bfgs_hpo_fold_objective(
            fixed_anchor_positions_sim={label: targets[label] for label in fold.train_labels},
            weights=ObjectiveWeights(),
        )
        fixed_indices = set(int(index) for index in problem.anchor_indices)
        assert fixed_indices == {dni[label] for label in fold.train_labels}
        assert dni[fold.heldout_label] not in fixed_indices
        assert problem.dimension == 66


def test_load_selected_bfgs_hpo_params_reads_only_weight_parameters(tmp_path):
    pd.DataFrame([{"alpha": -0.5, "beta": 0.25, "w_dis": 1.0, "selected_on_grid_boundary": False, "ignored": 99}]).to_csv(
        tmp_path / "bfgs_hpo_selected_candidate.csv", index=False
    )

    selected = load_selected_bfgs_hpo_params(tmp_path)

    assert selected == {"alpha": -0.5, "beta": 0.25, "w_dis": 1.0}


def test_load_selected_bfgs_hpo_params_rejects_boundary_choice_by_default(tmp_path):
    pd.DataFrame([{"alpha": -1.0, "beta": 0.0, "w_dis": 1.0, "selected_on_grid_boundary": True}]).to_csv(
        tmp_path / "bfgs_hpo_selected_candidate.csv", index=False
    )

    with pytest.raises(ValueError, match="search-grid boundary"):
        load_selected_bfgs_hpo_params(tmp_path)

    assert load_selected_bfgs_hpo_params(tmp_path, allow_boundary=True)["alpha"] == -1.0


def test_hpo_pareto_population_allows_partial_runs_when_every_fold_is_represented():
    grid = pd.DataFrame(
        [
            {
                "alpha": 0.0,
                "beta": 0.0,
                "is_complete": False,
                "all_folds_have_success": True,
                "E_distance_stress_mean": 0.1,
                "E_direction_vr_mean": 0.2,
                "RMSE_anchor_LOO_mean_km": 100.0,
            },
            {
                "alpha": 1.0,
                "beta": 0.0,
                "is_complete": False,
                "all_folds_have_success": False,
                "E_distance_stress_mean": 0.2,
                "E_direction_vr_mean": 0.1,
                "RMSE_anchor_LOO_mean_km": 90.0,
            },
        ]
    )

    eligible = _eligible_grid_points(grid)

    assert eligible[["alpha", "beta"]].to_dict("records") == [
        {"alpha": 0.0, "beta": 0.0}
    ]
