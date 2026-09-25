from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from library.directional_objectives import DIRECTIONAL_OBJECTIVE_EXACT
from run_paper_script import ch5_hparam_kfold_gridsearch_pareto as hpo
from run_paper_script import ch5_sector_exact_control as control


def _synthetic_combined() -> pd.DataFrame:
    rows = []
    for variant, offset in ((control.SECTOR_VARIANT, 0.0), (control.EXACT_VARIANT, 1.0)):
        for seed in (0, 1, 2):
            row = {"variant": variant, "seed": seed, "status": "ok", "error": ""}
            for metric_index, metric in enumerate(control.OFFICIAL_METRICS):
                row[metric] = float(metric_index + seed + offset)
            rows.append(row)
    return pd.DataFrame(rows)


def test_summary_and_paired_results_use_all_eight_metrics_and_matched_seeds():
    combined = _synthetic_combined()
    summary = control.summarize_runs(combined)
    paired = control.paired_comparison(combined)

    assert len(summary) == 2 * len(control.OFFICIAL_METRICS)
    assert set(summary["metric"]) == set(control.OFFICIAL_METRICS)
    assert len(paired) == len(control.OFFICIAL_METRICS)
    assert (paired["n_pairs"] == 3).all()
    assert np.allclose(paired["paired_diff_mean"], 1.0)
    assert np.allclose(paired["paired_diff_ci95_lo"], 1.0)
    assert np.allclose(paired["paired_diff_ci95_hi"], 1.0)
    assert (paired["diff_definition"] == "exact_minus_sector").all()


def test_formal_protocol_rejects_reduced_seed_lists():
    with pytest.raises(ValueError, match="HPO requires seeds 0-9"):
        control._validate_formal_protocol(
            hpo_seeds=[0],
            final_seeds=list(range(100)),
            alpha_min=-1,
            alpha_max=1.5,
            alpha_step=0.5,
            beta_min=-2,
            beta_max=0.5,
            beta_step=0.5,
            allow_smoke=False,
        )


def test_physics_eval_forwards_exact_objective_without_changing_evaluation(monkeypatch):
    captured = {}
    directional_data = [["A", "B", "東"]]
    distance_data = [["A", "B", "100"]]
    monkeypatch.setattr(hpo, "uploading_directional_data", lambda: directional_data)
    monkeypatch.setattr(
        hpo,
        "load_ini_data_from_csv",
        lambda _paths: ([], ["A", "B"], {"A": 0, "B": 1}, [], distance_data),
    )
    monkeypatch.setattr(
        hpo,
        "generate_CHEN_initial_positions",
        lambda *_args, **_kwargs: (["A", "B"], {"A": 0, "B": 1}, distance_data, [], []),
    )

    def fake_simulation(_vertices, _dni, _data, *_args, **kwargs):
        captured["directional_objective"] = kwargs["directional_objective"]
        return [], [0.0], [], [(0.0, 0.0), (1.0, 0.0)]

    monkeypatch.setattr(hpo, "main_physics_simulation", fake_simulation)
    monkeypatch.setattr(hpo, "calculate_kruskals_stress", lambda *_args: 0.1)
    monkeypatch.setattr(hpo, "direction_violation_rate", lambda *_args, **_kwargs: 0.2)
    monkeypatch.setattr(
        hpo, "mean_angular_error_violations", lambda *_args, **_kwargs: 0.3
    )
    monkeypatch.setattr(hpo, "_rmse_labels_km", lambda **_kwargs: 4.0)

    metrics, *_ = hpo._run_physics_eval(
        seed=0,
        fixed_labels=["A"],
        fixed_lonlat=[(80.0, 40.0)],
        eval_labels=["B"],
        rmse_gt_labels=["A", "B"],
        rmse_gt_lonlat=[(80.0, 40.0), (81.0, 40.0)],
        anchor_label_for_frame="A",
        spring_stiffness=1500.0,
        repulsion_strength=500.0,
        directional_force_magnitude=1_000_000.0,
        refer_pos_sim=[600, 500],
        directional_objective=DIRECTIONAL_OBJECTIVE_EXACT,
    )

    assert captured["directional_objective"] == DIRECTIONAL_OBJECTIVE_EXACT
    assert metrics["E_distance_stress"] == pytest.approx(0.1)
    assert metrics["E_direction_vr"] == pytest.approx(0.2)
    assert metrics["E_direction_mae"] == pytest.approx(0.3)
    assert metrics["RMSE_km"] == pytest.approx(4.0)
    assert metrics["E_direction_nominal_mae_rad"] == pytest.approx(0.0)
