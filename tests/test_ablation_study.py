import json
import shutil
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from library.data_io import get_anchor_labels, get_test_site_labels
from run_paper_script.ch5_ablation_study import (
    PHYSICS_VARIANTS,
    _build_paired_comparisons,
    _load_selected_hpo_params,
    _pairwise_distance_stats_km,
    _series_stats,
    _variant_forces,
    run_ablation_study,
)
from scripts.export_ablation_review import _assert_lcc_matches_ablation_config


def _local_tmp_dir() -> Path:
    path = Path("outputs") / "test_tmp" / f"ablation_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def test_ablation_variants_define_expected_force_switches():
    assert set(PHYSICS_VARIANTS) == {
        "PhysicsSim-Full",
        "PhysicsSim-NoRep",
        "PhysicsSim-NoDir",
        "PhysicsSim-DistOnly",
    }
    assert PHYSICS_VARIANTS["PhysicsSim-Full"] == {"use_direction": True, "use_repulsion": True}
    assert PHYSICS_VARIANTS["PhysicsSim-NoRep"] == {"use_direction": True, "use_repulsion": False}
    assert PHYSICS_VARIANTS["PhysicsSim-NoDir"] == {"use_direction": False, "use_repulsion": True}
    assert PHYSICS_VARIANTS["PhysicsSim-DistOnly"] == {"use_direction": False, "use_repulsion": False}


def test_variant_forces_disable_expected_terms():
    common = {
        "alpha": 0.0,
        "beta": 0.0,
        "w_dis": 1.0,
        "base_spring_stiffness": 1500.0,
        "base_directional_force": 10000.0,
        "base_repulsion_strength": 500.0,
    }

    _w_dir, _w_reg, spring, directional, repulsion = _variant_forces("PhysicsSim-Full", **common)
    assert spring == pytest.approx(1500.0)
    assert directional == pytest.approx(10000.0)
    assert repulsion == pytest.approx(500.0)

    _w_dir, _w_reg, _spring, directional, repulsion = _variant_forces("PhysicsSim-NoRep", **common)
    assert directional == pytest.approx(10000.0)
    assert repulsion == pytest.approx(0.0)

    _w_dir, _w_reg, _spring, directional, repulsion = _variant_forces("PhysicsSim-NoDir", **common)
    assert directional == pytest.approx(0.0)
    assert repulsion == pytest.approx(500.0)

    _w_dir, _w_reg, _spring, directional, repulsion = _variant_forces("PhysicsSim-DistOnly", **common)
    assert directional == pytest.approx(0.0)
    assert repulsion == pytest.approx(0.0)


def test_load_selected_hpo_params():
    tmp_dir = _local_tmp_dir()
    try:
        (tmp_dir / "selected_final_summary.json").write_text(
            json.dumps({"alpha": 1.5, "beta": -0.5}), encoding="utf-8"
        )

        assert _load_selected_hpo_params(tmp_dir) == (1.5, -0.5)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_load_selected_hpo_params_prefers_selected_candidate_csv():
    tmp_dir = _local_tmp_dir()
    try:
        (tmp_dir / "selected_final_summary.json").write_text(
            json.dumps({"alpha": 9.0, "beta": 9.0}), encoding="utf-8"
        )
        pd.DataFrame([{"alpha": 1.0, "beta": -1.5}]).to_csv(
            tmp_dir / "selected_candidate_summary.csv", index=False, encoding="utf-8-sig"
        )

        assert _load_selected_hpo_params(tmp_dir) == (1.0, -1.5)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_ablation_uses_current_anchor_and_test_contract():
    assert get_anchor_labels() == ["鄯善", "車師前", "都護治/烏壘"]
    assert len(get_test_site_labels()) == 8


def test_series_stats_include_bootstrap_and_robust_statistics():
    stats = _series_stats([1.0, 2.0, 3.0, 4.0])

    assert stats["n"] == 4
    assert stats["mean"] == pytest.approx(2.5)
    assert stats["median"] == pytest.approx(2.5)
    assert stats["iqr"] == pytest.approx(1.5)
    assert stats["se"] > 0
    assert stats["ci95_lo"] <= stats["mean"] <= stats["ci95_hi"]


def test_pairwise_distance_stats_km_are_positive():
    pos_y_up_sim = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 20.0]])

    min_dist, median_dist = _pairwise_distance_stats_km(pos_y_up_sim)

    assert min_dist > 0
    assert median_dist >= min_dist


def test_build_paired_comparisons_uses_same_seed_differences():
    rows = []
    for seed, full_rmse, norep_rmse in [(0, 10.0, 12.0), (1, 20.0, 18.0)]:
        rows.append(
            {
                "variant": "PhysicsSim-Full",
                "model": "PhysicsSim",
                "seed": seed,
                "status": "ok",
                "E_distance_stress": 1.0,
                "E_direction_vr": 1.0,
                "E_direction_mae": 1.0,
                "RMSE_test_km": full_rmse,
                "min_pairwise_distance_km": 1.0,
                "median_pairwise_distance_km": 2.0,
            }
        )
        rows.append(
            {
                "variant": "PhysicsSim-NoRep",
                "model": "PhysicsSim",
                "seed": seed,
                "status": "ok",
                "E_distance_stress": 1.0,
                "E_direction_vr": 1.0,
                "E_direction_mae": 1.0,
                "RMSE_test_km": norep_rmse,
                "min_pairwise_distance_km": 1.0,
                "median_pairwise_distance_km": 2.0,
            }
        )

    paired = _build_paired_comparisons(pd.DataFrame(rows))
    row = paired[
        (paired["comparison"] == "repulsion_given_direction") & (paired["metric"] == "RMSE_test_km")
    ].iloc[0]

    assert row["n_pairs"] == 2
    assert row["paired_diff_mean"] == pytest.approx(0.0)
    assert row["left_better_win_rate"] == pytest.approx(0.5)


def test_ablation_smoke_output_contract_if_available():
    outdir = Path("outputs/ch5_ablation_smoke")
    runs_path = outdir / "ablation_runs_by_seed.csv"
    summary_path = outdir / "ablation_summary.csv"
    positions_path = outdir / "ablation_final_positions_y_up_sim.csv"
    if not runs_path.exists():
        pytest.skip(f"ablation smoke output not found: {runs_path}")

    runs = pd.read_csv(runs_path)
    required_run_cols = {
        "model",
        "variant",
        "seed",
        "status",
        "E_distance_stress",
        "E_direction_vr",
        "E_direction_mae",
        "RMSE_test_km",
        "min_pairwise_distance_km",
        "median_pairwise_distance_km",
    }
    assert required_run_cols.issubset(runs.columns)

    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        assert {"model", "variant", "RMSE_test_km_mean", "RMSE_test_km_ci95_lo", "RMSE_test_km_ci95_hi"}.issubset(
            summary.columns
        )

    if positions_path.exists():
        positions = pd.read_csv(positions_path)
        assert {"model", "variant", "seed", "label", "x_y_up_sim", "y_y_up_sim"}.issubset(positions.columns)

    paired_path = outdir / "ablation_paired_comparisons.csv"
    if paired_path.exists():
        paired = pd.read_csv(paired_path)
        assert {"comparison", "metric", "paired_diff_mean", "left_better_win_rate"}.issubset(paired.columns)


def test_ablation_refuses_nonempty_outdir_before_loading_hpo():
    hpo_dir = _local_tmp_dir()
    outdir = _local_tmp_dir()
    try:
        (outdir / "existing.txt").write_text("keep", encoding="utf-8")
        with pytest.raises(FileExistsError):
            run_ablation_study(hpo_outdir=hpo_dir, seeds=[0], outdir=outdir)
    finally:
        shutil.rmtree(hpo_dir, ignore_errors=True)
        shutil.rmtree(outdir, ignore_errors=True)


def test_ablation_review_rejects_missing_lcc_metadata():
    outdir = _local_tmp_dir()
    try:
        (outdir / "ablation_config.json").write_text(json.dumps({"alpha": 1.0}), encoding="utf-8")
        with pytest.raises(ValueError):
            _assert_lcc_matches_ablation_config(outdir)
    finally:
        shutil.rmtree(outdir, ignore_errors=True)
