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
    _load_selected_dc_smacof_params,
    _load_selected_hpo_params,
    _pairwise_distance_stats_km,
    _run_directed_mds_baseline,
    _run_smacof_baseline,
    _series_stats,
    _variant_forces,
    run_ablation_study,
)
import run_paper_script.ch5_ablation_study as ablation_module
import run_paper_script.ch6_visualize_representative as representative_module
import library.model_cmp as model_cmp
from MDS_model.directed_mds_model import revise_direction
from scripts.export_ablation_review import _assert_lcc_matches_ablation_config
from run_paper_script.ch6_visualize_representative import (
    _rerun_as_variant_history,
    _select_representative_seed_from_as,
    _verify_rerun_matches_as_metrics,
)


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


def test_smacof_baseline_applies_procrustes(monkeypatch):
    calls = {"procrustes": 0}

    def fake_stress_majorization(graph, dni, vertice, edges):
        return np.array([[1.0, 1.0], [2.0, 2.0]]), [], []

    def fake_alignment(pos, vertice, dni, refer_pos, y_down=False, anchor_label=None):
        return np.asarray(pos, dtype=float)

    def fake_procrustes(pos, fixed_labels, fixed_lonlat, dni, refer_pos=None, anchor_label=None):
        calls["procrustes"] += 1
        assert anchor_label == "A"
        return np.asarray(pos, dtype=float)

    monkeypatch.setattr(ablation_module, "stress_majorization", fake_stress_majorization)
    monkeypatch.setattr(ablation_module, "alignment_and_scaling", fake_alignment)
    monkeypatch.setattr(ablation_module, "procrustes_align_by_fixed_points", fake_procrustes)

    _run_smacof_baseline(
        seed=0,
        graph=None,
        vertice=["A", "B"],
        dni={"A": 0, "B": 1},
        edges=[],
        anchor_labels=["A", "B"],
        anchor_lonlat=[(1.0, 1.0), (2.0, 2.0)],
        refer_pos_sim=[600.0, 250.0],
    )

    assert calls["procrustes"] == 1


def test_dc_smacof_baseline_does_not_apply_procrustes(monkeypatch):
    def fake_run_directed_mds(vis=False, w_weight_value=None, v_weight_value=None):
        return [np.array([[1.0, 1.0], [2.0, 2.0]])]

    def fake_alignment(pos, vertice, dni, refer_pos, y_down=False, anchor_label=None):
        assert anchor_label == "A"
        return np.asarray(pos, dtype=float)

    def fail_procrustes(*args, **kwargs):
        raise AssertionError("DC-SMACOF must not use Procrustes alignment")

    monkeypatch.setattr(ablation_module, "run_directed_MDS", fake_run_directed_mds)
    monkeypatch.setattr(ablation_module, "alignment_and_scaling", fake_alignment)
    monkeypatch.setattr(ablation_module, "procrustes_align_by_fixed_points", fail_procrustes)

    pos = _run_directed_mds_baseline(
        seed=0,
        vertice=["A", "B"],
        dni={"A": 0, "B": 1},
        anchor_labels=["A", "B"],
        anchor_lonlat=[(1.0, 1.0), (2.0, 2.0)],
        refer_pos_sim=[600.0, 250.0],
    )

    assert pos.shape == (2, 2)


def test_dc_smacof_baseline_passes_hpo_weights(monkeypatch):
    captured = {}

    def fake_run_directed_mds(vis=False, w_weight_value=None, v_weight_value=None):
        captured["w_weight_value"] = w_weight_value
        captured["v_weight_value"] = v_weight_value
        return [np.array([[1.0, 1.0], [2.0, 2.0]])]

    def fake_alignment(pos, vertice, dni, refer_pos, y_down=False, anchor_label=None):
        return np.asarray(pos, dtype=float)

    monkeypatch.setattr(ablation_module, "run_directed_MDS", fake_run_directed_mds)
    monkeypatch.setattr(ablation_module, "alignment_and_scaling", fake_alignment)

    _run_directed_mds_baseline(
        seed=0,
        vertice=["A", "B"],
        dni={"A": 0, "B": 1},
        anchor_labels=["A", "B"],
        anchor_lonlat=[(1.0, 1.0), (2.0, 2.0)],
        refer_pos_sim=[600.0, 250.0],
        dc_w_weight=1.0,
        dc_v_weight=0.1,
    )

    assert captured["w_weight_value"] == pytest.approx(1.0)
    assert captured["v_weight_value"] == pytest.approx(0.1)


def test_run_directed_mds_uses_verified_direction_data(monkeypatch):
    captured = {}

    def fail_read_csvfile(*args, **kwargs):
        raise AssertionError("DC-SMACOF must not read legacy hard-coded GPT CSV files")

    def fake_uploading_directional_data():
        return [["A", "B", "東"], ["B", "C", "西北"]]

    def fake_load_ini_data_from_csv(file_paths):
        return [], ["A", "B", "C"], {"A": 0, "B": 1, "C": 2}, [], []

    def fake_directed_mds(c_data, data, graph, vertice, dni, edges, distance_weight=None, direction_weight=None):
        captured["c_data"] = c_data
        captured["distance_weight"] = distance_weight
        captured["direction_weight"] = direction_weight
        return np.zeros((3, 2)), [], [np.zeros((3, 2))]

    monkeypatch.setattr(model_cmp, "read_csvfile", fail_read_csvfile, raising=False)
    monkeypatch.setattr(model_cmp, "uploading_directional_data", fake_uploading_directional_data)
    monkeypatch.setattr(model_cmp, "load_ini_data_from_csv", fake_load_ini_data_from_csv)
    monkeypatch.setattr(model_cmp, "directed_MDS", fake_directed_mds)

    history = model_cmp.run_directed_MDS(vis=False)

    assert len(history) == 1
    assert captured["c_data"] == [[], [], [["A", "B", "東"], ["B", "C", "西北"]], []]
    assert captured["distance_weight"] is None
    assert captured["direction_weight"] is None


def test_revise_direction_keeps_verified_dir8_names():
    rows = [["A", "B", "東"], ["B", "C", "西北"]]

    assert revise_direction(rows) == rows


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


def test_load_selected_dc_smacof_params_from_selected_candidate_csv():
    tmp_dir = _local_tmp_dir()
    try:
        pd.DataFrame([{"alpha": -1.5, "w_weight": 1.0, "v_weight": 0.0316227766}]).to_csv(
            tmp_dir / "dc_smacof_selected_candidate.csv", index=False, encoding="utf-8-sig"
        )

        params = _load_selected_dc_smacof_params(tmp_dir)

        assert params["source"] == "dc_hpo_selected_candidate"
        assert params["alpha"] == pytest.approx(-1.5)
        assert params["w_weight"] == pytest.approx(1.0)
        assert params["v_weight"] == pytest.approx(0.0316227766)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_load_selected_dc_smacof_params_manual_alpha():
    params = _load_selected_dc_smacof_params(dc_alpha=-2.0)

    assert params["source"] == "manual_cli_alpha"
    assert params["w_weight"] == pytest.approx(1.0)
    assert params["v_weight"] == pytest.approx(0.01)


def test_ch6_representative_rerun_uses_ablation_dc_weights(monkeypatch):
    captured = {}

    def fake_run_directed_mds(vis=False, w_weight_value=None, v_weight_value=None):
        captured["w_weight_value"] = w_weight_value
        captured["v_weight_value"] = v_weight_value
        return [np.array([[1.0, 1.0], [2.0, 2.0]])]

    def fake_align_history(pos_history_li, vertice, dni, anchor_labels, rp_sim):
        return [[[float(x), float(y)] for x, y in pos_history_li[-1]]]

    monkeypatch.setattr(representative_module, "run_directed_MDS", fake_run_directed_mds)
    monkeypatch.setattr(representative_module, "_align_dc_smacof_history", fake_align_history)

    history = _rerun_as_variant_history(
        variant="DC-SMACOF",
        seed=0,
        config={"refer_pos_sim": [600.0, 250.0], "dc_smacof_hpo": {"w_weight": 1.0, "v_weight": 0.1}},
        graph=[],
        vertice=["A", "B"],
        dni={"A": 0, "B": 1},
        edges=[],
        anchor_labels=["A", "B"],
        anchor_lonlat=[(1.0, 1.0), (2.0, 2.0)],
    )

    assert history
    assert captured["w_weight_value"] == pytest.approx(1.0)
    assert captured["v_weight_value"] == pytest.approx(0.1)


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


def test_select_representative_seed_from_ablation_metrics_uses_median_distance():
    group = pd.DataFrame(
        [
            {
                "variant": "PhysicsSim-Full",
                "seed": 0,
                "status": "ok",
                "E_distance_stress": 1.0,
                "E_direction_vr": 1.0,
                "E_direction_mae": 1.0,
                "RMSE_test_km": 1.0,
            },
            {
                "variant": "PhysicsSim-Full",
                "seed": 1,
                "status": "ok",
                "E_distance_stress": 2.0,
                "E_direction_vr": 2.0,
                "E_direction_mae": 2.0,
                "RMSE_test_km": 2.0,
            },
            {
                "variant": "PhysicsSim-Full",
                "seed": 2,
                "status": "ok",
                "E_distance_stress": 20.0,
                "E_direction_vr": 20.0,
                "E_direction_mae": 20.0,
                "RMSE_test_km": 20.0,
            },
        ]
    )

    selected = _select_representative_seed_from_as(group)

    assert selected["seed"] == 1
    assert selected["selection_scope"] == "one representative seed per AS variant"
    assert selected["metrics"]["RMSE_test_km"] == pytest.approx(2.0)


def test_verify_rerun_matches_as_metrics_accepts_close_values():
    rep = {
        "variant": "PhysicsSim-Full",
        "seed": 0,
        "metrics": {
            "E_distance_stress": 1.0,
            "E_direction_vr": 0.1,
            "E_direction_mae": 0.2,
            "RMSE_test_km": 100.0,
        },
    }
    rerun = {
        "E_distance_stress": 1.0 + 1e-8,
        "E_direction_vr": 0.1,
        "E_direction_mae": 0.2,
        "RMSE_test_km": 100.000001,
    }

    diffs = _verify_rerun_matches_as_metrics(rep, rerun, abs_tol=1e-5, rel_tol=1e-5)

    assert all(item["ok"] for item in diffs.values())


def test_verify_rerun_matches_as_metrics_rejects_large_drift():
    rep = {
        "variant": "DC-SMACOF",
        "seed": 0,
        "metrics": {
            "E_distance_stress": 1.0,
            "E_direction_vr": 0.1,
            "E_direction_mae": 0.2,
            "RMSE_test_km": 100.0,
        },
    }
    rerun = {
        "E_distance_stress": 1.0,
        "E_direction_vr": 0.1,
        "E_direction_mae": 0.2,
        "RMSE_test_km": 110.0,
    }

    with pytest.raises(ValueError):
        _verify_rerun_matches_as_metrics(rep, rerun, abs_tol=1e-5, rel_tol=1e-5)


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
    assert "min_pairwise_distance_km" not in set(paired["metric"])
    assert "median_pairwise_distance_km" not in set(paired["metric"])


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
