import json

import numpy as np
import pandas as pd
import pytest

import run_paper_script.ch5_dc_smacof_hparam as hpo_module
from library.model_cmp import run_directed_MDS
from MDS_model.directed_mds_model import v_weight, w_weight
from run_paper_script.ch5_dc_smacof_hparam import _anchor_split, _dc_weights_from_alpha


def test_dc_weights_from_alpha_uses_log10_ratio():
    distance_weight, direction_weight = _dc_weights_from_alpha(-2.0, w_weight=1.0)
    assert distance_weight == pytest.approx(1.0)
    assert direction_weight == pytest.approx(0.01)

    distance_weight, direction_weight = _dc_weights_from_alpha(0.0, w_weight=2.0)
    assert distance_weight == pytest.approx(2.0)
    assert direction_weight == pytest.approx(2.0)


def test_dc_anchor_split_excludes_frame_anchor_from_rmse_anchors():
    anchor_labels, anchor_align_label, rmse_anchor_labels = _anchor_split()

    assert anchor_align_label in anchor_labels
    assert anchor_align_label not in rmse_anchor_labels
    assert set(rmse_anchor_labels) == set(anchor_labels) - {anchor_align_label}


def test_run_directed_mds_explicit_default_weights_match_legacy_default():
    np.random.seed(0)
    default_history = run_directed_MDS(vis=False)

    np.random.seed(0)
    explicit_history = run_directed_MDS(
        vis=False,
        w_weight_value=w_weight,
        v_weight_value=v_weight,
    )

    assert len(default_history) == len(explicit_history)
    assert np.allclose(np.asarray(default_history[-1], dtype=float), np.asarray(explicit_history[-1], dtype=float))


def test_dc_hpo_writes_direction_method_metadata_and_preprocessing_csv(tmp_path, monkeypatch):
    def fake_evaluate(**kwargs):
        distance_weight, direction_weight = hpo_module._dc_weights_from_alpha(
            kwargs["alpha"],
            w_weight=kwargs["w_weight"],
        )
        return {
            "alpha": float(kwargs["alpha"]),
            "seed": int(kwargs["seed"]),
            "w_weight": distance_weight,
            "v_weight": direction_weight,
            "status": "ok",
            "error": "",
            "n_iterations": 1,
            "E_distance_stress": 0.1,
            "E_direction_vr": 0.2,
            "RMSE_anc_km": 100.0,
        }

    monkeypatch.setattr(hpo_module, "_evaluate_dc_smacof_run", fake_evaluate)
    monkeypatch.setattr(hpo_module, "_plot_metric_lines", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(hpo_module, "_plot_pareto_3d", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(hpo_module, "_plot_pareto_2d", lambda *_args, **_kwargs: None)

    outdir = tmp_path / "dc_hpo_metadata"
    hpo_module.run_dc_smacof_hparam(
        seeds=[0, 1],
        alpha_min=-0.5,
        alpha_max=-0.5,
        alpha_step=0.5,
        outdir=outdir,
    )

    config = json.loads((outdir / "dc_smacof_hparam_config.json").read_text(encoding="utf-8"))
    assert config["direction_target_rule"] == "wang2017_current_pair_distance"
    assert config["direction_preprocessing"] == "vector_consensus_by_undirected_pair"
    assert config["raw_direction_observation_count"] == 44
    assert config["effective_direction_constraint_count"] == 43
    assert config["direction_evaluation_source"] == "raw_verified_observations"
    assert config["direction_preprocessing_file"] == "dc_smacof_direction_preprocessing.csv"

    preprocessing = pd.read_csv(outdir / "dc_smacof_direction_preprocessing.csv")
    assert len(preprocessing) == 43
    repeated = preprocessing[preprocessing["n_observations"] == 2]
    assert len(repeated) == 1
    assert repeated.iloc[0]["effective_source"] == "莎車"
    assert repeated.iloc[0]["effective_target"] == "疏勒"
    assert repeated.iloc[0]["effective_direction"] == "西北"
