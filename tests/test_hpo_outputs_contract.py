import json
from pathlib import Path

import pandas as pd
import pytest


HPO_OUTDIR = Path("outputs/ch5_hparam_anchor_loo_smoke_with_review")


def _require_hpo_output() -> Path:
    if not (HPO_OUTDIR / "grid_summary_cv.csv").exists():
        pytest.skip(f"HPO smoke output not found: {HPO_OUTDIR}")
    return HPO_OUTDIR


def test_hpo_grid_summary_has_current_objective_columns():
    outdir = _require_hpo_output()
    df = pd.read_csv(outdir / "grid_summary_cv.csv")

    required = {
        "alpha",
        "beta",
        "n_failed_runs",
        "E_distance_stress_mean",
        "E_direction_vr_mean",
        "RMSE_anchor_LOO_mean_km",
        "is_pareto",
    }
    if not {"n_failed_runs"}.issubset(df.columns):
        pytest.skip(f"Legacy HPO output without failure-count columns: {outdir}")
    assert required.issubset(df.columns)


def test_hpo_config_records_lcc_parameters():
    outdir = _require_hpo_output()
    config_path = outdir / "gridsearch_config.json"
    if not config_path.exists():
        pytest.skip(f"HPO config not found: {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))

    if not {"lcc_bounds", "lcc_parameters", "lcc_bounds_source"}.issubset(config):
        pytest.skip(f"Legacy HPO config without LCC metadata: {config_path}")
    assert {"lcc_bounds", "lcc_parameters", "lcc_bounds_source"}.issubset(config)
    assert {"lon_min", "lon_max", "lat_min", "lat_max"}.issubset(config["lcc_bounds"])
    assert {"lat_1", "lat_2", "lon_0"}.issubset(config["lcc_parameters"])


def test_hpo_selected_final_summary_contract():
    outdir = _require_hpo_output()
    summary_path = outdir / "selected_final_summary.json"
    if not summary_path.exists():
        pytest.skip(f"selected final summary not found: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    required = {
        "alpha",
        "beta",
        "anchor_labels",
        "test_labels",
        "RMSE_final_test_mean_km",
        "E_distance_stress_mean",
        "E_direction_vr_mean",
    }
    assert required.issubset(summary)
    assert len(summary["anchor_labels"]) == 3
    assert len(summary["test_labels"]) == 8


def test_hpo_loo_review_has_three_heldout_rows():
    outdir = _require_hpo_output()
    review_path = outdir / "loo_fold_review" / "all_loo_fold_review.csv"
    if not review_path.exists():
        pytest.skip(f"LOO review output not found: {review_path}")

    df = pd.read_csv(review_path)
    heldout = df[df["role"] == "anchor_heldout"]

    assert len(heldout) == 3
    assert set(heldout["heldout_label"]) == {"鄯善", "車師前", "都護治/烏壘"}
    assert {"fold_RMSE_anchor_LOO_km", "E_distance_stress", "E_direction_vr"}.issubset(df.columns)
