from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pymunk
import pytest

from library.metrics import direction_violation_rate, mean_angular_error_violations
from library.physics import apply_forces
from run_paper_script import ch5_direction_tolerance_sensitivity as sensitivity
from run_paper_script import ch5_hparam_kfold_gridsearch_pareto as hpo


def _body(x: float, y: float) -> pymunk.Body:
    body = pymunk.Body(1.0, 1.0)
    body.position = (x, y)
    return body


def test_registered_tolerance_grid_has_expected_cardinal_and_diagonal_angles():
    assert sensitivity.FORMAL_NUMERATORS == (6, 5, 4, 3, 2, 1, 0)
    expected = {
        0: (0.0, 0.0),
        1: (15.0, 7.5),
        2: (30.0, 15.0),
        3: (45.0, 22.5),
        4: (60.0, 30.0),
        5: (75.0, 37.5),
        6: (90.0, 45.0),
    }
    for numerator, (cardinal, diagonal) in expected.items():
        metadata = sensitivity.angle_metadata(numerator)
        assert metadata["direction_tolerance_scale"] == pytest.approx(numerator / 6.0)
        assert metadata["cardinal_half_width_deg"] == pytest.approx(cardinal)
        assert metadata["diagonal_half_width_deg"] == pytest.approx(diagonal)


def test_direction_metrics_use_native_tolerance_and_preserve_default_behavior():
    angle = math.radians(30.0)
    points = np.asarray([[0.0, 0.0], [math.cos(angle), math.sin(angle)]])
    observations = [["A", "B", "東"]]
    dni = {"A": 0, "B": 1}

    assert direction_violation_rate(points, observations, dni) == pytest.approx(0.0)
    assert direction_violation_rate(
        points, observations, dni, direction_tolerance_scale=1.0
    ) == pytest.approx(0.0)
    assert direction_violation_rate(
        points, observations, dni, direction_tolerance_scale=1.0 / 6.0
    ) == pytest.approx(1.0)
    assert mean_angular_error_violations(
        points, observations, dni, direction_tolerance_scale=1.0 / 6.0
    ) == pytest.approx(math.radians(15.0))
    assert mean_angular_error_violations(
        points, observations, dni, direction_tolerance_scale=0.0
    ) == pytest.approx(math.radians(30.0))


def test_sector_force_changes_only_when_angle_exceeds_scaled_half_width():
    angle = math.radians(30.0)
    observations = [["A", "B", "東"]]
    dni = {"A": 0, "B": 1}

    wide = [_body(0.0, 0.0), _body(math.cos(angle), math.sin(angle))]
    apply_forces(
        0.1,
        0.0,
        0.0,
        10.0,
        wide,
        observations,
        dni,
        direction_tolerance_scale=1.0,
    )
    assert np.asarray(wide[0].force) == pytest.approx([0.0, 0.0])
    assert np.asarray(wide[1].force) == pytest.approx([0.0, 0.0])

    narrow = [_body(0.0, 0.0), _body(math.cos(angle), math.sin(angle))]
    _nodes, violations, _wrong = apply_forces(
        0.1,
        0.0,
        0.0,
        10.0,
        narrow,
        observations,
        dni,
        direction_tolerance_scale=1.0 / 6.0,
    )
    assert violations == 1
    assert np.linalg.norm(np.asarray(narrow[0].force, dtype=float)) > 0.0
    np.testing.assert_allclose(
        np.asarray(narrow[0].force, dtype=float),
        -np.asarray(narrow[1].force, dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )


def test_physics_eval_forwards_tolerance_to_simulation_and_both_metric_policies(monkeypatch):
    captured = {"metric_scales": []}
    directions = [["A", "B", "東"]]
    distances = [["A", "B", "100"]]
    monkeypatch.setattr(hpo, "uploading_directional_data", lambda: directions)
    monkeypatch.setattr(
        hpo,
        "load_ini_data_from_csv",
        lambda _paths: ([], ["A", "B"], {"A": 0, "B": 1}, [], distances),
    )
    monkeypatch.setattr(
        hpo,
        "generate_CHEN_initial_positions",
        lambda *_args, **_kwargs: (["A", "B"], {"A": 0, "B": 1}, distances, [], []),
    )

    def fake_simulation(_vertices, _dni, _data, *_args, **kwargs):
        captured["simulation_scale"] = kwargs["direction_tolerance_scale"]
        return [], [0.0], [], [(0.0, 0.0), (1.0, 0.0)]

    def fake_vr(*_args, **kwargs):
        scale = kwargs.get("direction_tolerance_scale", 1.0)
        captured["metric_scales"].append(("vr", scale))
        return scale

    def fake_mae(*_args, **kwargs):
        scale = kwargs.get("direction_tolerance_scale", 1.0)
        captured["metric_scales"].append(("mae", scale))
        return scale * 2.0

    monkeypatch.setattr(hpo, "main_physics_simulation", fake_simulation)
    monkeypatch.setattr(hpo, "calculate_kruskals_stress", lambda *_args: 0.1)
    monkeypatch.setattr(hpo, "direction_violation_rate", fake_vr)
    monkeypatch.setattr(hpo, "mean_angular_error_violations", fake_mae)
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
        direction_tolerance_scale=0.5,
    )

    assert captured["simulation_scale"] == pytest.approx(0.5)
    assert captured["metric_scales"] == [
        ("vr", 0.5),
        ("mae", 0.5),
        ("vr", 1.0),
        ("mae", 1.0),
    ]
    assert metrics["E_direction_vr"] == pytest.approx(0.5)
    assert metrics["E_direction_mae"] == pytest.approx(1.0)
    assert metrics["E_direction_vr_reference"] == pytest.approx(1.0)
    assert metrics["E_direction_mae_reference"] == pytest.approx(2.0)


def test_paired_comparison_matches_seed_and_includes_zero_reference():
    rows = []
    for numerator, seeds, offset in ((6, [0, 1, 2], 0.0), (3, [2, 0, 1], 7.0)):
        for seed in seeds:
            row = {"tolerance_numerator": numerator, "seed": seed}
            for metric_index, metric in enumerate(sensitivity.METRICS):
                row[metric] = 100.0 + metric_index + seed + offset
            rows.append(row)

    paired = sensitivity.paired_comparisons(pd.DataFrame(rows))
    reference = paired[paired["tolerance_numerator"].eq(6)]
    comparison = paired[paired["tolerance_numerator"].eq(3)]

    assert len(reference) == len(sensitivity.METRICS)
    assert np.allclose(reference["difference_mean"], 0.0)
    assert np.allclose(reference["difference_ci95_low"], 0.0)
    assert np.allclose(reference["difference_ci95_high"], 0.0)
    assert np.allclose(comparison["difference_mean"], 7.0)
    assert np.allclose(comparison["difference_ci95_low"], 7.0)
    assert np.allclose(comparison["difference_ci95_high"], 7.0)


def test_formal_preflight_reports_registered_model_run_budget(tmp_path):
    report = sensitivity.preflight_direction_tolerance_sensitivity(
        numerators=sensitivity.FORMAL_NUMERATORS,
        hpo_seeds=sensitivity.FORMAL_HPO_SEEDS,
        final_seeds=sensitivity.FORMAL_FINAL_SEEDS,
        **sensitivity.FORMAL_GRID,
        reference_alpha=sensitivity.REFERENCE_ALPHA,
        reference_beta=sensitivity.REFERENCE_BETA,
        outdir=tmp_path / "formal",
    )

    assert report["hpo_runs_per_nonreference_scenario"] == 1080
    assert report["final_runs_per_scenario"] == 100
    assert report["expected_total_model_runs"] == 7180
