import pandas as pd
import pytest

import run_paper_script.ch5_anchor_split_robustness as robustness
import run_paper_script.ch5_hparam_kfold_gridsearch_pareto as hpo
from library.geometry import lcc_transformation_with_anchor


def _candidate_frame():
    frame = pd.DataFrame(
        [
            {"model_name": "S1", "region_class": "southern_route", "anchor_eligible": True},
            {"model_name": "S2", "region_class": "southern_route", "anchor_eligible": True},
            {"model_name": "N1", "region_class": "northern_route", "anchor_eligible": True},
            {"model_name": "M1", "region_class": "north_of_mountains", "anchor_eligible": True},
            {"model_name": "M2", "region_class": "north_of_mountains", "anchor_eligible": True},
            {"model_name": "T1", "region_class": "", "anchor_eligible": False},
            {"model_name": "T2", "region_class": "", "anchor_eligible": False},
            {"model_name": "T3", "region_class": "", "anchor_eligible": False},
            {"model_name": "T4", "region_class": "", "anchor_eligible": False},
            {"model_name": "T5", "region_class": "", "anchor_eligible": False},
            {"model_name": "T6", "region_class": "", "anchor_eligible": False},
        ]
    )
    frame["lon"] = [float(i) for i in range(len(frame))]
    frame["lat"] = [float(i + 1) for i in range(len(frame))]
    frame["current_use_role"] = "test"
    return frame


def test_build_anchor_splits_is_stratified_deterministic_and_keeps_eight_tests(monkeypatch):
    monkeypatch.setattr(robustness, "get_anchor_labels", lambda: ["S1", "N1", "M1"])

    splits = robustness.build_anchor_splits(_candidate_frame())

    assert len(splits) == 4
    assert splits[0].split_id == "split_001"
    assert splits[0].anchor_labels == ("S1", "N1", "M1")
    assert splits[0].final_frame_anchor == "S1"
    assert splits[0].is_original_split is True
    assert all(len(split.test_labels) == 8 for split in splits)
    assert all(not (set(split.anchor_labels) & set(split.test_labels)) for split in splits)


def test_build_anchor_splits_never_selects_low_rmse_or_reorders_original(monkeypatch):
    monkeypatch.setattr(robustness, "get_anchor_labels", lambda: ["S2", "N1", "M2"])

    splits = robustness.build_anchor_splits(_candidate_frame())

    assert [split.split_id for split in splits] == ["split_001", "split_002", "split_003", "split_004"]
    assert [split.split_id for split in splits if split.is_original_split] == ["split_004"]


def test_load_candidate_table_rejects_eligible_site_without_region(monkeypatch):
    frame = _candidate_frame()
    frame.loc[frame["model_name"] == "S1", "region_class"] = ""
    monkeypatch.setattr(pd, "read_excel", lambda *args, **kwargs: frame)
    monkeypatch.setattr(
        robustness,
        "load_site_points",
        lambda: [
            {
                "name": row.model_name,
                "lon": row.lon,
                "lat": row.lat,
                "use_role": row.current_use_role,
            }
            for row in frame.itertuples(index=False)
        ],
    )
    monkeypatch.setattr(robustness.Path, "exists", lambda self: True)

    with pytest.raises(ValueError, match="still need region_class"):
        robustness.load_anchor_candidate_table("candidates.xlsx")


def test_override_inputs_require_exact_disjoint_three_plus_eight(monkeypatch):
    labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
    dni = {label: index for index, label in enumerate(labels)}
    monkeypatch.setattr(
        hpo,
        "_load_site_lonlat_by_label",
        lambda _dni: {label: (float(i), float(i + 1)) for i, label in enumerate(labels)},
    )

    anchors, _anchor_xy, tests, _test_xy = hpo._resolve_anchor_and_test_inputs(
        dni,
        anchor_labels_override=["A", "B", "C"],
        test_labels_override=labels[3:],
    )

    assert anchors == ["A", "B", "C"]
    assert tests == labels[3:]
    with pytest.raises(ValueError, match="overlap"):
        hpo._resolve_anchor_and_test_inputs(
            dni,
            anchor_labels_override=["A", "B", "C"],
            test_labels_override=["C", "D", "E", "F", "G", "H", "I", "J"],
        )


def test_anchor_loo_never_uses_heldout_anchor_as_fixed_or_frame():
    folds = hpo._build_anchor_loo_folds(
        ["South", "North", "Mountain"],
        [(80.0, 38.0), (84.0, 41.0), (88.0, 44.0)],
    )

    for fold in folds:
        assert fold.heldout_label not in fold.train_labels
        assert fold.train_anchor_label in fold.train_labels
        assert fold.train_anchor_label != fold.heldout_label


def test_split_summary_uses_sample_sd_and_bootstrap_ci(monkeypatch):
    monkeypatch.setattr(robustness, "get_anchor_labels", lambda: ["S1", "N1", "M1"])
    split = robustness.build_anchor_splits(_candidate_frame())[0]
    runs = pd.DataFrame(
        {
            "RMSE_final_test_km": [100.0, 200.0],
            "E_distance_stress": [0.1, 0.2],
            "E_direction_vr": [0.0, 0.1],
        }
    )

    summary = robustness._summarize_split(
        split,
        runs,
        {
            "alpha": 1.0,
            "beta": -0.5,
            "selected_on_alpha_boundary": True,
            "selected_on_beta_boundary": False,
            "selected_on_grid_boundary": True,
        },
    )

    assert summary["RMSE_final_test_mean_km"] == pytest.approx(150.0)
    assert summary["RMSE_final_test_std_km"] == pytest.approx(70.71067811865476)
    assert summary["RMSE_final_test_ci95_low_km"] <= 150.0 <= summary["RMSE_final_test_ci95_high_km"]
    assert summary["selected_on_grid_boundary"] is True


def test_lcc_frame_anchor_changes_only_coordinate_origin():
    dni = {"A": 0, "B": 1, "C": 2}
    lonlat = [(80.0, 38.0), (84.0, 41.0), (88.0, 44.0)]
    frame_a = pd.DataFrame(lcc_transformation_with_anchor(dni, lonlat, anchor_label="A"))
    frame_b = pd.DataFrame(lcc_transformation_with_anchor(dni, lonlat, anchor_label="B"))

    for i in range(3):
        for j in range(3):
            assert (frame_a.iloc[i] - frame_a.iloc[j]).to_numpy() == pytest.approx(
                (frame_b.iloc[i] - frame_b.iloc[j]).to_numpy(), abs=1e-10
            )


def test_numeric_grid_requires_exact_endpoint_and_positive_step():
    assert robustness._numeric_grid(-1.0, 1.5, 0.5, name="alpha") == [
        -1.0,
        -0.5,
        0.0,
        0.5,
        1.0,
        1.5,
    ]
    with pytest.raises(ValueError, match="positive"):
        robustness._numeric_grid(-1.0, 1.0, 0.0, name="alpha")
    with pytest.raises(ValueError, match="end exactly"):
        robustness._numeric_grid(-1.0, 1.0, 0.3, name="alpha")


def test_archive_incomplete_split_preserves_every_file(tmp_path):
    split_dir = tmp_path / "splits" / "split_001"
    split_dir.mkdir(parents=True)
    (split_dir / "partial.csv").write_text("unfinished", encoding="utf-8")

    archived = robustness._archive_incomplete_split(split_dir, tmp_path / "interrupted_attempts")

    assert not split_dir.exists()
    assert (archived / "partial.csv").read_text(encoding="utf-8") == "unfinished"
    assert archived.name.startswith("split_001_")


def test_resume_config_rejects_mixed_experiment_settings():
    existing = {key: "same" for key in robustness.RESUME_CONFIG_KEYS}
    requested = dict(existing)
    robustness._assert_resume_config_compatible(existing, requested)

    requested["final_evaluation_seeds"] = [0, 1, 2]
    with pytest.raises(ValueError, match="Resume configuration differs"):
        robustness._assert_resume_config_compatible(existing, requested)


def test_completed_split_requires_every_final_artifact(tmp_path):
    (tmp_path / "gridsearch_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "selected_final_summary.json").write_text('{"alpha": 1, "beta": -0.5}', encoding="utf-8")
    pd.DataFrame(
        {"RMSE_final_test_km": [1.0], "E_distance_stress": [0.1], "E_direction_vr": [0.0]}
    ).to_csv(tmp_path / "selected_final_runs_by_seed.csv", index=False)
    assert robustness._completed_split(tmp_path) is False

    pd.DataFrame({"site_label": ["T1"], "error_km": [1.0], "squared_error_km2": [1.0]}).to_csv(
        tmp_path / "selected_final_site_errors.csv", index=False
    )
    assert robustness._completed_split(tmp_path) is True


def test_event_log_is_append_only_and_valid_jsonl(tmp_path):
    path = tmp_path / "experiment_events.jsonl"
    robustness._append_event(path, {"event": "first"})
    robustness._append_event(path, {"event": "second"})

    rows = [robustness.json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["event"] for row in rows] == ["first", "second"]
    assert all("timestamp_utc" in row for row in rows)
