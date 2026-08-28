import json
from pathlib import Path

import numpy as np
import pandas as pd
import pymunk
import pytest

from library.config import FILE_PATHS, Li2sim, km2sim
from library.data_io import load_ini_data_from_csv
from library.metrics import calculate_kruskals_stress, raw_distance_stress_from_sim_data
from library.physics import create_nodes_and_springs
from library.units import data_Li2sim
import run_paper_script.ch5_detour_factor_sensitivity as detour
import run_paper_script.ch5_hparam_kfold_gridsearch_pareto as hpo
from scripts.export_hpo_loo_review import _selected_distance_scale
from scripts.verify_detour_sensitivity import _verify_distance_audit, _verify_fixed_hyperparameter_rows


def _write_fake_scenario(path: Path, *, kappa: float, final_seeds: list[int]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "gridsearch_config.json").write_text(
        json.dumps({"distance_scale": kappa}), encoding="utf-8"
    )
    (path / "selected_final_summary.json").write_text(
        json.dumps(
            {
                "distance_scale": kappa,
                "alpha": 1.0,
                "beta": -0.5,
                "selected_on_grid_boundary": False,
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {
            "seed": seed,
            "distance_scale": kappa,
            "alpha": 1.0,
            "beta": -0.5,
            "RMSE_final_test_km": 100.0 * kappa + seed,
            "E_distance_stress": 0.1 * kappa,
            "E_direction_vr": 0.05,
            "E_direction_mae": 0.02,
        }
        for seed in final_seeds
    ]
    pd.DataFrame(rows).to_csv(path / "selected_final_runs_by_seed.csv", index=False)
    pd.DataFrame(
        [
            {"seed": seed, "site_label": f"T{index}", "error_km": 1.0, "squared_error_km2": 1.0}
            for seed in final_seeds
            for index in range(8)
        ]
    ).to_csv(path / "selected_final_site_errors.csv", index=False)
    pd.DataFrame(
        [
            {
                "seed": seed,
                "node_idx": index,
                "label": f"N{index}",
                "x_y_up_sim": float(index),
                "y_y_up_sim": float(index),
            }
            for seed in final_seeds
            for index in range(35)
        ]
    ).to_csv(path / "selected_final_positions_y_up_sim.csv", index=False)
    pd.DataFrame({"RMSE_anchor_LOO_km": [10.0, 20.0]}).to_csv(path / "grid_runs_by_seed.csv", index=False)
    grid = pd.DataFrame(
        {
            "alpha": [1.0],
            "beta": [-0.5],
            "RMSE_anchor_LOO_mean_km": [15.0],
            "RMSE_anchor_LOO_std_km": [2.0],
        }
    )
    grid.to_csv(path / "grid_summary_cv.csv", index=False)
    grid.to_csv(path / "pareto_front_3d.csv", index=False)
    pd.DataFrame(
        [
            {
                "edge_index": index,
                "source": f"N{index % 35}",
                "target": f"N{(index + 1) % 35}",
                "original_distance_li": 100.0,
                "unscaled_target_sim": 10.0,
                "scaled_target_sim": 10.0 * kappa,
                "unscaled_target_km": 10.0 / km2sim,
                "scaled_target_km": 10.0 * kappa / km2sim,
                "distance_scale": kappa,
                "applied_ratio": kappa,
            }
            for index in range(44)
        ]
    ).to_csv(path / "distance_targets_audit.csv", index=False)


def _runner_kwargs(outdir: Path) -> dict:
    return {
        "seeds": [0],
        "final_seeds": [0, 1],
        "kappa_min": 0.975,
        "kappa_max": 1.0,
        "kappa_step": 0.025,
        "alpha_min": 1.0,
        "alpha_max": 1.0,
        "alpha_step": 1.0,
        "beta_min": -0.5,
        "beta_max": -0.5,
        "beta_step": 1.0,
        "w_dis": 1.0,
        "base_spring_stiffness": 1500.0,
        "base_directional_force": 1_000_000.0,
        "base_repulsion_strength": 500.0,
        "outdir": outdir,
        "generate_scenario_plots": False,
    }


def _write_fake_fixed_final(
    *,
    outdir: Path,
    selected: pd.Series,
    seeds: list[int],
    distance_scale: float,
    selection_rule: str,
    **_kwargs,
) -> None:
    alpha = float(selected["alpha"])
    beta = float(selected["beta"])
    rows = [
        {
            "selection_rule": selection_rule,
            "distance_scale": distance_scale,
            "alpha": alpha,
            "beta": beta,
            "seed": seed,
            "E_distance_stress": 0.1 * distance_scale,
            "E_direction_vr": 0.05,
            "E_direction_mae": 0.02,
            "RMSE_final_test_km": 100.0 * distance_scale + seed,
        }
        for seed in seeds
    ]
    pd.DataFrame(rows).to_csv(outdir / "selected_final_runs_by_seed.csv", index=False)
    pd.DataFrame(
        [
            {"seed": seed, "site_label": f"T{index}", "error_km": 1.0, "squared_error_km2": 1.0}
            for seed in seeds
            for index in range(8)
        ]
    ).to_csv(outdir / "selected_final_site_errors.csv", index=False)
    pd.DataFrame(
        [
            {
                "seed": seed,
                "node_idx": index,
                "label": f"N{index}",
                "x_y_up_sim": float(index),
                "y_y_up_sim": float(index),
            }
            for seed in seeds
            for index in range(35)
        ]
    ).to_csv(outdir / "selected_final_positions_y_up_sim.csv", index=False)
    (outdir / "selected_final_summary.json").write_text(
        json.dumps(
            {
                "selection_rule": selection_rule,
                "distance_scale": distance_scale,
                "alpha": alpha,
                "beta": beta,
                "n_seeds": len(seeds),
            }
        ),
        encoding="utf-8",
    )


def test_scale_sim_distance_data_preserves_original_and_scales_all_distance_fields():
    original = [["A", "B", 100.0, 120.0], ["B", "C", 20.0]]

    scaled = hpo._scale_sim_distance_data(original, 0.8)

    assert scaled == [["A", "B", 80.0, 96.0], ["B", "C", 16.0]]
    assert original == [["A", "B", 100.0, 120.0], ["B", "C", 20.0]]
    assert hpo._scale_sim_distance_data(original, 1.0) == original


def test_real_350_li_edge_is_scaled_once_with_correct_sim_and_km_units():
    data_li = [["A", "B", "350"]]
    scaled = hpo._scale_sim_distance_data(data_Li2sim(data_li), 0.8)
    audit = detour._distance_target_audit_frame(data_li, 0.8).iloc[0]

    assert scaled[0][2] == pytest.approx(28.0)
    assert audit["unscaled_target_sim"] == pytest.approx(35.0)
    assert audit["scaled_target_sim"] == pytest.approx(28.0)
    assert audit["unscaled_target_km"] == pytest.approx(145.25)
    assert audit["scaled_target_km"] == pytest.approx(116.2)
    assert audit["applied_ratio"] == pytest.approx(0.8)


def test_actual_pymunk_spring_receives_distance_scale_exactly_once():
    scaled = hpo._scale_sim_distance_data(data_Li2sim([["A", "B", "350"]]), 0.8)
    _nodes, space = create_nodes_and_springs(
        2,
        10.0,
        5.0,
        1000.0,
        1500.0,
        50.0,
        1e7,
        pymunk.Space(),
        scaled,
        {"A": 0, "B": 1},
        [[0.0, 0.0], [40.0, 0.0]],
        [],
    )
    springs = [constraint for constraint in space.constraints if isinstance(constraint, pymunk.DampedSpring)]

    assert len(springs) == 1
    assert springs[0].rest_length == pytest.approx(28.0)
    assert springs[0].rest_length != pytest.approx(35.0)
    assert springs[0].rest_length != pytest.approx(22.4)


@pytest.mark.parametrize("kappa", [1.0, 0.825, 0.7])
def test_all_44_real_pymunk_springs_receive_distance_scale_exactly_once(kappa):
    _graph, vertices, dni, _edges, data_li = load_ini_data_from_csv(FILE_PATHS)
    scaled = hpo._scale_sim_distance_data(data_Li2sim(data_li), kappa)
    initial_positions = [[float(index * 10), float(index * 7)] for index in range(len(vertices))]
    _nodes, space = create_nodes_and_springs(
        len(vertices),
        10.0,
        5.0,
        1000.0,
        1500.0,
        50.0,
        1e7,
        pymunk.Space(),
        scaled,
        dni,
        initial_positions,
        [],
    )
    actual = sorted(
        constraint.rest_length
        for constraint in space.constraints
        if isinstance(constraint, pymunk.DampedSpring)
    )
    expected = sorted(float(row[2]) * Li2sim * kappa for row in data_li)

    assert len(actual) == len(expected) == 44
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_independent_verifier_detects_distance_scale_applied_twice(tmp_path):
    data_li = [["A", "B", "350"]]
    audit = detour._distance_target_audit_frame(data_li, 0.8)
    audit.to_csv(tmp_path / "distance_targets_audit.csv", index=False)
    failures = []
    _verify_distance_audit(
        scenario_dir=tmp_path,
        data_li=data_li,
        kappa=0.8,
        failures=failures,
        atol=1e-8,
        rtol=1e-9,
    )
    assert failures == []

    audit.loc[0, "scaled_target_sim"] *= 0.8
    audit.loc[0, "scaled_target_km"] *= 0.8
    audit.to_csv(tmp_path / "distance_targets_audit.csv", index=False)
    _verify_distance_audit(
        scenario_dir=tmp_path,
        data_li=data_li,
        kappa=0.8,
        failures=failures,
        atol=1e-8,
        rtol=1e-9,
    )

    assert any("scaled exactly once" in failure for failure in failures)
    assert any("independently reconstructed distance" in failure for failure in failures)


def test_stress_and_stress_trace_use_scaled_target_exactly_once():
    scaled = hpo._scale_sim_distance_data(data_Li2sim([["A", "B", "350"]]), 0.8)
    dni = {"A": 0, "B": 1}
    correctly_scaled_positions_km = [[0.0, 0.0], [116.2, 0.0]]
    double_scaled_positions_km = [[0.0, 0.0], [92.96, 0.0]]

    assert calculate_kruskals_stress(dni, correctly_scaled_positions_km, scaled) == pytest.approx(0.0)
    assert raw_distance_stress_from_sim_data(scaled, dni, correctly_scaled_positions_km) == pytest.approx(0.0)
    assert calculate_kruskals_stress(dni, double_scaled_positions_km, scaled) > 0.0


def test_distance_source_consistency_rejects_stale_ini_data():
    detour._assert_distance_sources_consistent([["A", "B", "350"]], [["A", "B", "350"]])
    with pytest.raises(ValueError, match="numbers of distance edges"):
        detour._assert_distance_sources_consistent([["A", "B", "350"]], [])
    with pytest.raises(ValueError, match="endpoints differ"):
        detour._assert_distance_sources_consistent([["A", "B", "350"]], [["A", "C", "350"]])
    with pytest.raises(ValueError, match="values differ"):
        detour._assert_distance_sources_consistent([["A", "B", "350"]], [["A", "B", "351"]])


@pytest.mark.parametrize("scale", [0.0, -0.5, float("nan"), float("inf")])
def test_scale_sim_distance_data_rejects_invalid_scale(scale):
    with pytest.raises(ValueError, match="strictly positive"):
        hpo._scale_sim_distance_data([["A", "B", 1.0]], scale)


@pytest.mark.parametrize("distance", [0.0, -1.0, float("nan"), float("inf")])
def test_scale_sim_distance_data_rejects_invalid_distance(distance):
    with pytest.raises(ValueError, match="strictly positive"):
        hpo._scale_sim_distance_data([["A", "B", distance]], 0.8)


def test_formal_scenario_grid_has_thirteen_scales_and_reference_first():
    values = detour._scenario_grid(0.70, 1.0, 0.025)

    assert len(values) == 13
    assert values[0] == pytest.approx(1.0)
    assert values[-1] == pytest.approx(0.70)
    assert 0.825 in values
    assert len({detour._scenario_name(value) for value in values}) == 13


def test_scenario_grid_requires_reference_and_valid_range():
    with pytest.raises(ValueError, match="reference"):
        detour._scenario_grid(0.7, 0.9, 0.1)
    with pytest.raises(ValueError, match="0 < kappa <= 1"):
        detour._scenario_grid(0.8, 1.2, 0.1)
    with pytest.raises(ValueError, match="end exactly"):
        detour._scenario_grid(0.7, 1.0, 0.04)


def test_physics_eval_uses_same_scaled_distances_for_simulation_and_stress(monkeypatch):
    captured = {}
    directional_data = [["A", "B", "東"]]
    original_data = [["A", "B", "100"]]
    monkeypatch.setattr(hpo, "uploading_directional_data", lambda: directional_data)
    monkeypatch.setattr(
        hpo,
        "load_ini_data_from_csv",
        lambda _paths: ([], ["A", "B"], {"A": 0, "B": 1}, [], original_data),
    )
    monkeypatch.setattr(
        hpo,
        "generate_CHEN_initial_positions",
        lambda *_args, **_kwargs: (["A", "B"], {"A": 0, "B": 1}, original_data, [], []),
    )

    def fake_simulation(_vertices, _dni, data, *_args, **_kwargs):
        captured["simulation_data"] = data
        return [], [0.0], [], [(0.0, 0.0), (1.0, 0.0)]

    def fake_stress(_dni, _positions, data):
        captured["stress_data"] = data
        return 0.123

    monkeypatch.setattr(hpo, "main_physics_simulation", fake_simulation)
    monkeypatch.setattr(hpo, "calculate_kruskals_stress", fake_stress)
    monkeypatch.setattr(hpo, "direction_violation_rate", lambda *_args: 0.25)
    monkeypatch.setattr(hpo, "mean_angular_error_violations", lambda *_args: 0.125)
    monkeypatch.setattr(hpo, "_rmse_labels_km", lambda **_kwargs: 10.0)

    result, _positions, _vertices, _dni = hpo._run_physics_eval(
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
        distance_scale=0.8,
    )

    expected_distance = float(hpo.data_Li2sim(original_data)[0][2]) * 0.8
    assert captured["simulation_data"][0][2] == pytest.approx(expected_distance)
    assert captured["stress_data"] is captured["simulation_data"]
    assert result["E_direction_mae"] == pytest.approx(0.125)
    assert original_data == [["A", "B", "100"]]


def test_same_seed_keeps_initial_positions_and_fixed_anchors_across_scales(monkeypatch):
    snapshots = []

    def fake_simulation(_vertices, _dni, data, positions, _directional, fixed, *_args, **_kwargs):
        snapshots.append((np.asarray(positions, dtype=float).copy(), [list(row) for row in fixed], float(data[0][2])))
        return [], [0.0], [], positions

    monkeypatch.setattr(hpo, "main_physics_simulation", fake_simulation)
    monkeypatch.setattr(hpo, "_rmse_labels_km", lambda **_kwargs: 1.0)
    labels = detour.get_anchor_labels()
    from library.data_io import load_site_points

    site = {row["name"]: (float(row["lon"]), float(row["lat"])) for row in load_site_points()}
    for kappa in (1.0, 0.8):
        hpo._run_physics_eval(
            seed=7,
            fixed_labels=labels,
            fixed_lonlat=[site[label] for label in labels],
            eval_labels=detour.get_test_site_labels(),
            rmse_gt_labels=labels + detour.get_test_site_labels(),
            rmse_gt_lonlat=[site[label] for label in labels + detour.get_test_site_labels()],
            anchor_label_for_frame=detour.get_default_frame_anchor_label(),
            spring_stiffness=1500.0,
            repulsion_strength=500.0,
            directional_force_magnitude=1_000_000.0,
            refer_pos_sim=[600, 250],
            distance_scale=kappa,
        )

    np.testing.assert_allclose(snapshots[0][0], snapshots[1][0], rtol=0.0, atol=0.0)
    assert snapshots[0][1] == snapshots[1][1]
    assert snapshots[1][2] == pytest.approx(snapshots[0][2] * 0.8)


def test_paired_comparison_matches_seeds_instead_of_row_order():
    rows = []
    for kappa, seeds, offset in ((1.0, [0, 1, 2], 0.0), (0.8, [2, 0, 1], -10.0)):
        for seed in seeds:
            rows.append(
                {
                    "kappa": kappa,
                    "seed": seed,
                    "RMSE_final_test_km": 100.0 + seed + offset,
                    "E_distance_stress": 0.1 + seed,
                    "E_direction_vr": 0.2 + seed,
                    "E_direction_mae": 0.3 + seed,
                }
            )

    result = detour._paired_comparisons(pd.DataFrame(rows))
    rmse = result[result["metric"] == "RMSE_final_test_km"].iloc[0]

    assert rmse["n_pairs"] == 3
    assert rmse["difference_mean"] == pytest.approx(-10.0)
    assert rmse["difference_ci95_low"] == pytest.approx(-10.0)
    assert rmse["difference_ci95_high"] == pytest.approx(-10.0)


def test_paired_comparison_rejects_missing_seed():
    rows = pd.DataFrame(
        [
            {"kappa": 1.0, "seed": 0, **{metric: 1.0 for metric in detour.METRIC_COLUMNS}},
            {"kappa": 1.0, "seed": 1, **{metric: 2.0 for metric in detour.METRIC_COLUMNS}},
            {"kappa": 0.8, "seed": 0, **{metric: 1.0 for metric in detour.METRIC_COLUMNS}},
        ]
    )

    with pytest.raises(ValueError, match="cannot be paired"):
        detour._paired_comparisons(rows)


def test_completed_scenario_validates_scale_seeds_and_positions(tmp_path):
    _write_fake_scenario(tmp_path, kappa=0.8, final_seeds=[0, 1])

    assert detour._completed_scenario(tmp_path, kappa=0.8, final_seeds=[0, 1]) is True
    assert detour._completed_scenario(tmp_path, kappa=0.9, final_seeds=[0, 1]) is False
    assert detour._completed_scenario(tmp_path, kappa=0.8, final_seeds=[0]) is False


def test_completed_scenario_rejects_missing_or_double_scaled_distance_audit(tmp_path):
    _write_fake_scenario(tmp_path, kappa=0.8, final_seeds=[0, 1])
    path = tmp_path / "distance_targets_audit.csv"
    audit = pd.read_csv(path)
    audit.loc[0, "applied_ratio"] = 0.64
    audit.to_csv(path, index=False)

    assert detour._completed_scenario(tmp_path, kappa=0.8, final_seeds=[0, 1]) is False


def test_real_distance_source_and_ini_data_are_identical():
    _graph, _vertices, _dni, _edges, data_li = load_ini_data_from_csv(FILE_PATHS)
    detour._assert_distance_sources_consistent(detour.read_CHEN_csvfile(), data_li)
    assert len(data_li) == 44


def test_preflight_formal_configuration_counts_all_15340_runs(tmp_path):
    report = detour.preflight_detour_sensitivity(
        seeds=list(range(10)),
        final_seeds=list(range(100)),
        kappa_min=0.70,
        kappa_max=1.0,
        kappa_step=0.025,
        alpha_min=-1.0,
        alpha_max=1.5,
        alpha_step=0.5,
        beta_min=-2.0,
        beta_max=0.5,
        beta_step=0.5,
        outdir=tmp_path / "formal",
    )

    assert report["n_scenarios"] == 13
    assert report["hpo_runs_per_scenario"] == 1080
    assert report["final_runs_per_scenario"] == 100
    assert report["expected_total_model_runs"] == 15340
    assert len(report["anchor_labels"]) == 3
    assert len(report["test_labels"]) == 8
    assert not set(report["anchor_labels"]) & set(report["test_labels"])


def test_preflight_fixed_hyperparameters_counts_only_1300_final_runs(tmp_path):
    report = detour.preflight_detour_sensitivity(
        seeds=list(range(10)),
        final_seeds=list(range(100)),
        kappa_min=0.70,
        kappa_max=1.0,
        kappa_step=0.025,
        alpha_min=-1.0,
        alpha_max=1.5,
        alpha_step=0.5,
        beta_min=-2.0,
        beta_max=0.5,
        beta_step=0.5,
        fixed_alpha=1.0,
        fixed_beta=-0.5,
        outdir=tmp_path / "fixed",
    )

    assert report["hyperparameter_policy"] == "fixed"
    assert report["hpo_seeds"] == []
    assert report["hpo_runs_per_scenario"] == 0
    assert report["expected_total_model_runs"] == 1300
    assert report["fixed_alpha"] == pytest.approx(1.0)
    assert report["fixed_beta"] == pytest.approx(-0.5)


def test_preflight_fixed_hyperparameters_requires_alpha_and_beta_together(tmp_path):
    kwargs = _runner_kwargs(tmp_path / "fixed")
    kwargs.pop("generate_scenario_plots")
    with pytest.raises(ValueError, match="provided together"):
        detour.preflight_detour_sensitivity(**kwargs, fixed_alpha=1.0)


def test_preflight_fixed_reference_skips_one_hpo_scenario(tmp_path):
    kwargs = _runner_kwargs(tmp_path / "fixed_reference")
    kwargs.pop("generate_scenario_plots")

    report = detour.preflight_detour_sensitivity(
        **kwargs,
        reference_alpha=1.0,
        reference_beta=-0.5,
    )

    assert report["hyperparameter_policy"] == "scenario_specific_hpo_with_fixed_reference"
    assert report["reference_alpha"] == pytest.approx(1.0)
    assert report["reference_beta"] == pytest.approx(-0.5)
    assert report["expected_total_model_runs"] == 7


def test_runner_passes_scale_to_hpo_preserves_inputs_and_supports_resume(monkeypatch, tmp_path):
    calls = []
    hashes_before = detour._input_hashes()

    def fake_hpo(**kwargs):
        calls.append(kwargs)
        _write_fake_scenario(Path(kwargs["outdir"]), kappa=kwargs["distance_scale"], final_seeds=kwargs["final_seeds"])

    monkeypatch.setattr(detour, "run_anchor_loo_gridsearch_pareto", fake_hpo)
    monkeypatch.setattr(detour, "_save_plots", lambda *_args: None)
    kwargs = _runner_kwargs(tmp_path / "experiment")

    first = detour.run_detour_factor_sensitivity(**kwargs)

    assert first["n_completed_scenarios"] == 2
    assert [call["distance_scale"] for call in calls] == [1.0, 0.975]
    assert all(call["save_final_positions"] for call in calls)
    assert detour._input_hashes() == hashes_before
    summary = pd.read_csv(kwargs["outdir"] / "detour_scenario_summary.csv")
    assert summary["kappa"].tolist() == [1.0, 0.975]
    paired = pd.read_csv(kwargs["outdir"] / "detour_paired_comparisons.csv")
    assert set(paired["metric"]) == set(detour.METRIC_COLUMNS)

    resumed = detour.run_detour_factor_sensitivity(**kwargs, resume=True)

    assert resumed["n_completed_scenarios"] == 2
    assert len(calls) == 2


def test_fixed_runner_skips_hpo_and_keeps_identical_alpha_beta(monkeypatch, tmp_path):
    fixed_calls = []

    def fail_hpo(**_kwargs):
        raise AssertionError("Fixed-hyperparameter mode must not run HPO.")

    def fake_final(**kwargs):
        fixed_calls.append(kwargs)
        _write_fake_fixed_final(**kwargs)

    monkeypatch.setattr(detour, "run_anchor_loo_gridsearch_pareto", fail_hpo)
    monkeypatch.setattr(detour, "_run_final_selected_model", fake_final)
    monkeypatch.setattr(detour, "_save_plots", lambda *_args: None)
    kwargs = _runner_kwargs(tmp_path / "fixed_experiment")
    result = detour.run_detour_factor_sensitivity(
        **kwargs,
        fixed_alpha=1.0,
        fixed_beta=-0.5,
    )

    assert result["n_completed_scenarios"] == 2
    assert [call["distance_scale"] for call in fixed_calls] == [1.0, 0.975]
    assert all(float(call["selected"]["alpha"]) == 1.0 for call in fixed_calls)
    assert all(float(call["selected"]["beta"]) == -0.5 for call in fixed_calls)
    summary = pd.read_csv(kwargs["outdir"] / "detour_scenario_summary.csv")
    assert set(summary["hyperparameter_policy"]) == {"fixed"}
    assert set(summary["selected_alpha"]) == {1.0}
    assert set(summary["selected_beta"]) == {-0.5}
    assert set(summary["n_hpo_runs"]) == {0}


def test_runner_uses_formal_weights_only_for_reference(monkeypatch, tmp_path):
    hpo_calls = []
    reference_calls = []

    def fake_hpo(**kwargs):
        hpo_calls.append(kwargs["distance_scale"])
        _write_fake_scenario(Path(kwargs["outdir"]), kappa=kwargs["distance_scale"], final_seeds=kwargs["final_seeds"])

    def fake_reference(**kwargs):
        reference_calls.append(kwargs)
        _write_fake_fixed_final(**kwargs)

    monkeypatch.setattr(detour, "run_anchor_loo_gridsearch_pareto", fake_hpo)
    monkeypatch.setattr(detour, "_run_final_selected_model", fake_reference)
    monkeypatch.setattr(detour, "_save_plots", lambda *_args: None)
    kwargs = _runner_kwargs(tmp_path / "fixed_reference")

    result = detour.run_detour_factor_sensitivity(
        **kwargs,
        reference_alpha=1.0,
        reference_beta=-0.5,
    )

    assert result["n_completed_scenarios"] == 2
    assert hpo_calls == [0.975]
    assert len(reference_calls) == 1
    assert reference_calls[0]["distance_scale"] == pytest.approx(1.0)
    assert float(reference_calls[0]["selected"]["alpha"]) == pytest.approx(1.0)
    assert float(reference_calls[0]["selected"]["beta"]) == pytest.approx(-0.5)
    summary = pd.read_csv(kwargs["outdir"] / "detour_scenario_summary.csv")
    reference = summary[np.isclose(summary["kappa"], 1.0)].iloc[0]
    assert reference["hyperparameter_policy"] == "fixed_reference"
    assert reference["n_hpo_runs"] == 0

    failures = []
    _verify_fixed_hyperparameter_rows(
        scenario_dir=kwargs["outdir"] / "scenarios" / detour._scenario_name(1.0),
        kappa=1.0,
        fixed_alpha=1.0,
        fixed_beta=-0.5,
        expected_policy="fixed_reference",
        expected_selection_rule="predefined_formal_reference_hyperparameters",
        failures=failures,
        atol=1e-12,
        rtol=0.0,
    )
    assert failures == []


def test_resume_archives_only_incomplete_scenario(monkeypatch, tmp_path):
    calls = []

    def fake_hpo(**kwargs):
        calls.append(kwargs["distance_scale"])
        _write_fake_scenario(Path(kwargs["outdir"]), kappa=kwargs["distance_scale"], final_seeds=kwargs["final_seeds"])

    monkeypatch.setattr(detour, "run_anchor_loo_gridsearch_pareto", fake_hpo)
    monkeypatch.setattr(detour, "_save_plots", lambda *_args: None)
    kwargs = _runner_kwargs(tmp_path / "experiment")
    detour.run_detour_factor_sensitivity(**kwargs)
    interrupted = kwargs["outdir"] / "scenarios" / detour._scenario_name(0.975)
    (interrupted / "selected_final_positions_y_up_sim.csv").unlink()
    (interrupted / "partial_diagnostic.txt").write_text("preserve me", encoding="utf-8")

    detour.run_detour_factor_sensitivity(**kwargs, resume=True)

    assert calls == [1.0, 0.975, 0.975]
    archived = list((kwargs["outdir"] / "interrupted_attempts").glob("scenario_kappa_0p975_*"))
    assert len(archived) == 1
    assert (archived[0] / "partial_diagnostic.txt").read_text(encoding="utf-8") == "preserve me"


def test_resume_rejects_changed_experiment_configuration():
    existing = {key: "same" for key in detour.RESUME_CONFIG_KEYS}
    requested = dict(existing)
    detour._assert_resume_config_compatible(existing, requested)
    requested["final_evaluation_seeds"] = [0, 1]

    with pytest.raises(ValueError, match="Resume configuration differs"):
        detour._assert_resume_config_compatible(existing, requested)


def test_loo_visualization_uses_hpo_distance_scale_and_supports_legacy_outputs(tmp_path):
    assert _selected_distance_scale(tmp_path) == pytest.approx(1.0)
    config = tmp_path / "gridsearch_config.json"
    config.write_text(json.dumps({"distance_scale": 0.825}), encoding="utf-8")
    assert _selected_distance_scale(tmp_path) == pytest.approx(0.825)
    config.write_text(json.dumps({"distance_scale": -1.0}), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid distance_scale"):
        _selected_distance_scale(tmp_path)
