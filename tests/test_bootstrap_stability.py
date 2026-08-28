import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import library.bootstrap_and_visualization as bootstrap_lib
import run_paper_script.ch5_bootstrap_stability as bootstrap_script
import scripts.update_paper_results as paper_results_updater
from library.geometry import get_lcc_bounds, get_lcc_parameters


def test_relative_layout_scale_uses_mean_pairwise_separation():
    rows = [
        {"median_x_y_up_sim": 0.0, "median_y_y_up_sim": 0.0, "radial_q50_sim": 1.0,
         "radial_q95_sim": 2.0, "radial_max_sim": 3.0},
        {"median_x_y_up_sim": 2.0, "median_y_y_up_sim": 0.0, "radial_q50_sim": 1.0,
         "radial_q95_sim": 2.0, "radial_max_sim": 3.0},
        {"median_x_y_up_sim": 0.0, "median_y_y_up_sim": 6.0, "radial_q50_sim": 1.0,
         "radial_q95_sim": 2.0, "radial_max_sim": 3.0},
    ]

    layout_scale = bootstrap_lib._add_relative_layout_scale(rows)

    expected = np.mean([2.0, 6.0, np.sqrt(40.0)])
    assert layout_scale == pytest.approx(expected)
    assert rows[0]["radial_q95_layout_pct"] == pytest.approx(200.0 / expected)


def test_appendix_hdr_panels_respects_explicit_manual_selection(tmp_path):
    rng = np.random.default_rng(42)
    samples = np.stack(
        [
            rng.normal(loc=[0.0, 0.0], scale=[2.0, 1.0], size=(40, 2)),
            rng.normal(loc=[20.0, 10.0], scale=[1.0, 2.0], size=(40, 2)),
            rng.normal(loc=[40.0, 20.0], scale=[3.0, 1.0], size=(40, 2)),
        ],
        axis=1,
    )

    rows = bootstrap_lib.plot_appendix_hdr_panels(
        samples,
        ["A", "B", "C"],
        tmp_path / "manual.png",
        selected_labels=["C", "A"],
        ncols=2,
        grid_size=30,
    )

    assert [row["label"] for row in rows] == ["C", "A"]
    assert (tmp_path / "manual.png").stat().st_size > 0


def test_hdr_single_panel_includes_both_coordinate_axis_labels(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    samples = rng.normal(size=(40, 2, 2))
    samples[:, 1, :] += 10.0
    captured = {}
    original_savefig = matplotlib.figure.Figure.savefig

    def capture_labels(figure, *args, **kwargs):
        captured["x"] = figure.axes[0].get_xlabel()
        captured["y"] = figure.axes[0].get_ylabel()
        return original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", capture_labels)
    bootstrap_lib.plot_appendix_hdr_panels(
        samples,
        ["A", "B"],
        tmp_path / "single.png",
        selected_labels=["A"],
        ncols=1,
        grid_size=30,
    )

    assert captured == {"x": "x (simulation units)", "y": "y (simulation units)"}


def _write_formal_bootstrap_fixture(outdir, *, n_bootstrap=300):
    outdir.mkdir()
    hpo_dir = outdir.parent / "hpo"
    hpo_dir.mkdir(exist_ok=True)
    selected_hpo = hpo_dir / "selected_candidate_summary.csv"
    selected_hpo.write_text("alpha,beta\n1.0,-0.5\n", encoding="utf-8")
    hpo_config = hpo_dir / "gridsearch_config.json"
    hpo_config.write_text('{"test": true}', encoding="utf-8")
    labels = ["A", "B", "C", "D"]
    anchors = labels[:3]
    runs = pd.DataFrame(
        {
            "bootstrap_index": np.arange(n_bootstrap),
            "simulation_seed": np.arange(1000, 1000 + n_bootstrap),
            "alpha": np.full(n_bootstrap, 1.0),
            "beta": np.full(n_bootstrap, -0.5),
            "alpha_noise": np.zeros(n_bootstrap),
            "beta_noise": np.zeros(n_bootstrap),
            "w_dis": np.ones(n_bootstrap),
            "w_dir": np.full(n_bootstrap, 10.0),
            "w_reg": np.full(n_bootstrap, 10.0 ** -0.5),
            "spring_stiffness": np.full(n_bootstrap, 1500.0),
            "directional_force": np.full(n_bootstrap, 10_000_000.0),
            "repulsion_strength": np.full(n_bootstrap, 158.11388300841898),
        }
    )
    positions = pd.DataFrame(
        [
            {
                "bootstrap_index": run,
                "label": label,
                "x_y_up_sim": float(index + (run * 0.001 if label == "D" else 0.0)),
                "y_y_up_sim": float(index + (run * 0.002 if label == "D" else 0.0)),
            }
            for run in range(n_bootstrap)
            for index, label in enumerate(labels)
        ]
    )
    lonlat = positions.rename(columns={"x_y_up_sim": "lon", "y_y_up_sim": "lat"})
    ellipses = pd.DataFrame(
        [
            {
                "label": label,
                "confidence_level": confidence,
                "mean_x_y_up_sim": float(index),
                "mean_y_y_up_sim": float(index),
                "cov_xx": 0.0,
                "cov_xy": 0.0,
                "cov_yy": 0.0,
                "ellipse_width_sim": 0.0,
                "ellipse_height_sim": 0.0,
                "ellipse_angle_deg": 0.0,
            }
            for index, label in enumerate(labels)
            for confidence in (0.85, 0.90, 0.95)
        ]
    )
    kde = pd.DataFrame(
        {
            "label": labels,
            "kde_status": ["degenerate", "degenerate", "degenerate", "ok"],
            "n_samples": [n_bootstrap] * 4,
        }
    )
    drift = pd.DataFrame(
        {
            "label": anchors,
            "max_anchor_drift_sim": [0.0, 0.0, 0.0],
            "mean_anchor_drift_sim": [0.0, 0.0, 0.0],
        }
    )
    frames = {
        "bootstrap_run_parameters.csv": runs,
        "bootstrap_samples_y_up_sim.csv": positions,
        "bootstrap_samples_lonlat.csv": lonlat,
        "bootstrap_ellipse_summary.csv": ellipses,
        "bootstrap_kde_status.csv": kde,
        "bootstrap_anchor_drift.csv": drift,
    }
    for name, frame in frames.items():
        frame.to_csv(outdir / name, index=False)
    for name in (
        "confidence_ellipses.png",
        "confidence_ellipses.svg",
        "combined_kde_density.png",
        "combined_kde_density.svg",
    ):
        (outdir / name).write_bytes(f"test:{name}".encode())
    hashes = {
        name: paper_results_updater._sha256(outdir / name).lower()
        for name in paper_results_updater.BOOTSTRAP_FORMAL_FILES
        if name != "bootstrap_config.json"
    }
    config = {
        "method_classification": "parameter_perturbation_repeated_simulation_not_observation_resampling",
        "failure_count": 0,
        "n_bootstrap": n_bootstrap,
        "hpo_selection": {
            "alpha": 1.0,
            "beta": -0.5,
            "w_dis": 1.0,
            "spring_stiffness": 1500.0,
            "directional_force": 10_000_000.0,
            "repulsion_strength": 158.11388300841898,
            "selected_result_file": str(selected_hpo),
            "selected_result_sha256": paper_results_updater._sha256(selected_hpo).lower(),
        },
        "input_validation": {
            "lcc_matches_current_data": True,
            "site_roles_match_current_data": True,
            "physics_hpo_config": str(hpo_config),
            "physics_hpo_config_sha256": paper_results_updater._sha256(hpo_config).lower(),
        },
        "calibration_labels": anchors,
        "anchor_align_label": "A",
        "perturbation_space": "HPO log10 alpha/beta with w_dis fixed at 1",
        "artifact_sha256": hashes,
    }
    (outdir / "bootstrap_config.json").write_text(json.dumps(config), encoding="utf-8")
    return config


def test_bootstrap_visualization_uses_headless_backend():
    assert "agg" in matplotlib.get_backend().lower()


def test_bootstrap_dynamics_is_reproducible_and_uses_selected_forces(monkeypatch):
    calls = []

    def fake_run_once(
        seed,
        fixed_labels,
        fixed_lonlat,
        *,
        spring_stiffness,
        repulsion_strength,
        directional_force_magnitude,
        anchor_label,
    ):
        calls.append(
            (
                seed,
                spring_stiffness,
                repulsion_strength,
                directional_force_magnitude,
                anchor_label,
            )
        )
        return (
            np.asarray([[seed, spring_stiffness], [repulsion_strength, directional_force_magnitude]], dtype=float),
            ["A", "B"],
            {"A": 0, "B": 1},
        )

    monkeypatch.setattr(bootstrap_lib, "_run_once", fake_run_once)
    kwargs = dict(
        N_BOOTSTRAP=4,
        ALPHA_JITTER=0.10,
        BETA_JITTER=0.10,
        fixed_point_labels=["A"],
        fixed_points_lonlat=[(1.0, 2.0)],
        alpha=1.0,
        beta=-0.5,
        w_dis=1.0,
        spring_stiffness=1500.0,
        repulsion_strength=158.11388300841898,
        directional_force_magnitude=10_000_000.0,
        anchor_label="A",
        seed_start=20,
        jitter_seed=7,
        return_run_metadata=True,
    )
    first = bootstrap_lib.bootstrap_dynamics(**kwargs)
    first_calls = list(calls)
    calls.clear()
    second = bootstrap_lib.bootstrap_dynamics(**kwargs)

    np.testing.assert_allclose(first[0], second[0])
    assert first[3] == second[3]
    assert first_calls == calls
    assert first[3][0]["alpha"] == pytest.approx(1.0)
    assert first[3][0]["beta"] == pytest.approx(-0.5)
    assert first[3][0]["alpha_noise"] == pytest.approx(0.0)
    assert first[3][0]["beta_noise"] == pytest.approx(0.0)
    assert [row["simulation_seed"] for row in first[3]] == [20, 21, 22, 23]
    assert {row["spring_stiffness"] for row in first[3]} == {1500.0}
    assert len({row["directional_force"] for row in first[3]}) > 1
    assert len({row["repulsion_strength"] for row in first[3]}) > 1
    for row in first[3]:
        assert row["w_dir"] == pytest.approx(10.0 ** row["alpha"])
        assert row["w_reg"] == pytest.approx(10.0 ** row["beta"])
        assert row["directional_force"] == pytest.approx(10_000_000.0 * 10.0 ** row["alpha_noise"])
        assert row["repulsion_strength"] == pytest.approx(158.11388300841898 * 10.0 ** row["beta_noise"])


def test_confidence_ellipse_summary_matches_sample_covariance():
    samples = np.asarray(
        [
            [[0.0, 0.0], [5.0, 5.0]],
            [[2.0, 0.0], [5.0, 5.0]],
            [[0.0, 2.0], [5.0, 5.0]],
            [[2.0, 2.0], [5.0, 5.0]],
        ]
    )
    rows = bootstrap_lib.confidence_ellipse_summary(samples, ["moving", "fixed"])
    assert len(rows) == 6
    moving_95 = next(row for row in rows if row["label"] == "moving" and row["confidence_level"] == 0.95)
    fixed_95 = next(row for row in rows if row["label"] == "fixed" and row["confidence_level"] == 0.95)
    assert moving_95["mean_x_y_up_sim"] == pytest.approx(1.0)
    assert moving_95["mean_y_y_up_sim"] == pytest.approx(1.0)
    assert moving_95["cov_xx"] == pytest.approx(np.var([0.0, 2.0, 0.0, 2.0], ddof=1))
    assert moving_95["ellipse_width_sim"] > 0.0
    assert fixed_95["ellipse_width_sim"] == pytest.approx(0.0)
    assert fixed_95["ellipse_height_sim"] == pytest.approx(0.0)


def test_positional_stability_summary_uses_sample_covariance():
    samples = np.asarray(
        [
            [[0.0, 0.0]],
            [[2.0, 0.0]],
            [[0.0, 2.0]],
            [[2.0, 2.0]],
        ]
    )
    row = bootstrap_lib.positional_stability_summary(samples, ["moving"])[0]
    expected_radial_sd = np.sqrt(
        np.var([0.0, 2.0, 0.0, 2.0], ddof=1)
        + np.var([0.0, 0.0, 2.0, 2.0], ddof=1)
    )
    assert row["radial_sd_sim"] == pytest.approx(expected_radial_sd)
    assert row["radial_sd_km"] == pytest.approx(expected_radial_sd / bootstrap_lib.km2pix)
    assert row["ellipse_area_95_sim2"] > 0.0


def test_empirical_stability_summary_uses_radial_quantiles_from_spatial_median():
    samples = np.asarray([[[0.0, 0.0]], [[1.0, 0.0]], [[2.0, 0.0]], [[10.0, 0.0]]])
    row = bootstrap_lib.empirical_positional_stability_summary(samples, ["moving"])[0]
    radial = np.asarray([1.5, 0.5, 0.5, 8.5])
    assert row["median_x_y_up_sim"] == pytest.approx(1.5)
    assert row["radial_q50_sim"] == pytest.approx(np.quantile(radial, 0.50))
    assert row["radial_q95_sim"] == pytest.approx(np.quantile(radial, 0.95))
    assert row["radial_q95_km"] == pytest.approx(np.quantile(radial, 0.95) / bootstrap_lib.km2pix)


def test_highest_density_threshold_orders_nested_regions():
    density = np.asarray([[0.01, 0.02], [0.10, 0.87]])
    threshold_50 = bootstrap_lib._highest_density_threshold(density, 0.50)
    threshold_95 = bootstrap_lib._highest_density_threshold(density, 0.95)
    assert threshold_50 > threshold_95


def test_ellipse_and_kde_plots_create_png_and_svg(tmp_path):
    rng = np.random.default_rng(4)
    moving = rng.normal(loc=[400.0, 300.0], scale=[20.0, 10.0], size=(30, 2))
    fixed = np.repeat([[600.0, 250.0]], repeats=30, axis=0)
    fixed[:, 0] += np.linspace(0.0, 1e-9, 30)
    samples = np.stack([moving, fixed], axis=1)
    ellipse_paths = [tmp_path / "ellipses.png", tmp_path / "ellipses.svg"]
    kde_paths = [tmp_path / "kde.png", tmp_path / "kde.svg"]
    relative_paths = [tmp_path / "relative.png", tmp_path / "relative.svg"]
    outline_paths = [tmp_path / "outline.png", tmp_path / "outline.svg"]
    hdr_paths = [tmp_path / "hdr.png", tmp_path / "hdr.svg"]
    appendix_paths = [tmp_path / "appendix.png", tmp_path / "appendix.svg"]
    appendix_hdr_paths = [tmp_path / "appendix_hdr.png", tmp_path / "appendix_hdr.svg"]

    ellipse_rows = bootstrap_lib.plot_multi_ellipses(samples, ["moving", "fixed"], ellipse_paths)
    kde_rows = bootstrap_lib.plot_kde_combined(samples, ["moving", "fixed"], kde_paths, grid_size=30)
    stability_rows = bootstrap_lib.plot_relative_stability_map(
        samples, ["moving", "fixed"], relative_paths
    )
    bootstrap_lib.plot_ellipse_outline_map(samples, ["moving", "fixed"], outline_paths)
    hdr_rows = bootstrap_lib.plot_hdr_small_multiples(
        samples, ["moving", "fixed"], hdr_paths, grid_size=30
    )
    empirical_rows = bootstrap_lib.plot_appendix_stability_overview(
        samples, ["moving", "fixed"], appendix_paths, anchor_labels=["fixed"], test_labels=["moving"]
    )
    appendix_hdr_rows = bootstrap_lib.plot_appendix_hdr_panels(
        samples, ["moving", "fixed"], appendix_hdr_paths, anchor_labels=["fixed"], grid_size=30
    )

    assert len(ellipse_rows) == 6
    assert {row["label"]: row["kde_status"] for row in kde_rows} == {
        "moving": "ok",
        "fixed": "degenerate",
    }
    assert len(stability_rows) == 2
    assert [row["label"] for row in hdr_rows] == ["moving"]
    assert len(empirical_rows) == 2
    assert all(row["layout_scale_sim"] > 0.0 for row in empirical_rows)
    assert all(row["radial_q95_layout_pct"] >= row["radial_q50_layout_pct"] for row in empirical_rows)
    assert [row["label"] for row in appendix_hdr_rows] == ["moving"]
    for path in ellipse_paths + kde_paths + relative_paths + outline_paths + hdr_paths + appendix_paths + appendix_hdr_paths:
        assert path.exists()
        assert path.stat().st_size > 0


def test_load_physics_hpo_selection_checks_recorded_force_values(tmp_path):
    candidate = pd.DataFrame(
        [
            {
                "alpha": 1.0,
                "beta": -0.5,
                "spring_stiffness": 1500.0,
                "directional_force": 10_000_000.0,
                "repulsion_strength": 158.11388300841898,
            }
        ]
    )
    candidate.to_csv(tmp_path / "selected_candidate_summary.csv", index=False)
    result = bootstrap_script._load_physics_hpo_selection(tmp_path)
    assert result["alpha"] == pytest.approx(1.0)
    assert result["beta"] == pytest.approx(-0.5)
    assert result["directional_force"] == pytest.approx(10_000_000.0)
    assert result["repulsion_strength"] == pytest.approx(158.11388300841898)
    assert len(result["selected_result_sha256"]) == 64

    candidate.loc[0, "directional_force"] = 123.0
    candidate.to_csv(tmp_path / "selected_candidate_summary.csv", index=False)
    with pytest.raises(ValueError, match="does not match alpha/beta conversion"):
        bootstrap_script._load_physics_hpo_selection(tmp_path)


def test_validate_hpo_provenance_checks_lcc_and_site_roles(tmp_path):
    bounds = get_lcc_bounds()
    params = get_lcc_parameters()
    config = {
        "lcc_bounds": bounds,
        "lcc_parameters": params,
        "anchor_labels": ["A", "B", "C"],
        "test_labels": ["D"],
    }
    (tmp_path / "gridsearch_config.json").write_text(json.dumps(config), encoding="utf-8")
    result = bootstrap_script._validate_hpo_provenance(tmp_path, ["A", "B", "C"], ["D"])
    assert result["lcc_matches_current_data"] is True
    assert result["site_roles_match_current_data"] is True

    with pytest.raises(ValueError, match="anchor labels"):
        bootstrap_script._validate_hpo_provenance(tmp_path, ["A", "C", "B"], ["D"])


def test_run_bootstrap_stability_writes_formal_outputs(monkeypatch, tmp_path):
    vertice = ["A", "B", "C", "D"]
    dni = {label: index for index, label in enumerate(vertice)}
    samples = np.asarray(
        [
            [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0], [70.0, 80.0]],
            [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0], [75.0, 82.0]],
            [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0], [68.0, 85.0]],
        ]
    )
    run_metadata = [
        {
            "bootstrap_index": index,
            "simulation_seed": index,
            "alpha": 1.0,
            "beta": -0.5,
            "alpha_noise": 0.0,
            "beta_noise": 0.0,
            "w_dis": 1.0,
            "w_dir": 10.0,
            "w_reg": 10.0 ** -0.5,
            "spring_stiffness": 1500.0,
            "directional_force": 10_000_000.0,
            "repulsion_strength": 158.11388300841898,
            "status": "ok",
        }
        for index in range(3)
    ]
    gt = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)]
    selection = {
        "alpha": 1.0,
        "beta": -0.5,
        "w_dis": 1.0,
        "spring_stiffness": 1500.0,
        "directional_force": 10_000_000.0,
        "repulsion_strength": 158.11388300841898,
    }

    monkeypatch.setattr(bootstrap_script, "_load_physics_hpo_selection", lambda _path: selection)
    monkeypatch.setattr(bootstrap_script, "_validate_hpo_provenance", lambda *_args: {"validated": True})
    monkeypatch.setattr(
        bootstrap_script,
        "load_ini_data_from_csv",
        lambda _paths: (None, vertice, dni, None, None),
    )
    monkeypatch.setattr(bootstrap_script, "uploading_ground_truth", lambda *_args: gt)
    monkeypatch.setattr(bootstrap_script, "get_anchor_labels", lambda: ["A", "B", "C"])
    monkeypatch.setattr(bootstrap_script, "get_anchor_align_label", lambda: "A")
    monkeypatch.setattr(bootstrap_script, "get_test_site_labels", lambda: ["D"])
    monkeypatch.setattr(
        bootstrap_script,
        "bootstrap_dynamics",
        lambda *_args, **_kwargs: (samples, vertice, dni, run_metadata),
    )
    monkeypatch.setattr(
        bootstrap_script,
        "_target_positions_sim",
        lambda *_args: {"A": samples[0, 0], "B": samples[0, 1], "C": samples[0, 2]},
    )
    monkeypatch.setattr(bootstrap_script, "_samples_to_lonlat", lambda *_args: samples / 100.0)

    def fake_ellipses(_samples, labels, paths):
        for path in paths:
            Path(path).write_text("ellipse", encoding="utf-8")
        return [
            {
                "label": label,
                "confidence_level": 0.95,
                "mean_x_y_up_sim": 0.0,
                "mean_y_y_up_sim": 0.0,
                "cov_xx": 0.0,
                "cov_xy": 0.0,
                "cov_yy": 0.0,
                "ellipse_width_sim": 0.0,
                "ellipse_height_sim": 0.0,
                "ellipse_angle_deg": 0.0,
            }
            for label in labels
        ]

    def fake_kde(_samples, labels, paths, grid_size):
        assert grid_size == 30
        for path in paths:
            Path(path).write_text("kde", encoding="utf-8")
        return [{"label": label, "kde_status": "ok", "n_samples": 3} for label in labels]

    monkeypatch.setattr(bootstrap_script, "plot_multi_ellipses", fake_ellipses)
    monkeypatch.setattr(bootstrap_script, "plot_kde_combined", fake_kde)
    outdir = tmp_path / "formal_output"
    result = bootstrap_script.run_bootstrap_stability(
        hpo_outdir=tmp_path / "hpo",
        outdir=outdir,
        n_bootstrap=3,
        kde_grid_size=30,
    )

    expected = {
        "bootstrap_config.json",
        "bootstrap_run_parameters.csv",
        "bootstrap_samples_y_up_sim.csv",
        "bootstrap_samples_lonlat.csv",
        "bootstrap_ellipse_summary.csv",
        "bootstrap_kde_status.csv",
        "bootstrap_anchor_drift.csv",
        "confidence_ellipses.png",
        "confidence_ellipses.svg",
        "combined_kde_density.png",
        "combined_kde_density.svg",
        "bootstrap_stability_summary.csv",
        "bootstrap_hdr_selection.csv",
        "relative_stability_map.png",
        "relative_stability_map.svg",
        "dispersion_ellipses_95_outline.png",
        "dispersion_ellipses_95_outline.svg",
        "stability_hdr_small_multiples.png",
        "stability_hdr_small_multiples.svg",
        "bootstrap_empirical_stability_summary.csv",
        "bootstrap_appendix_hdr_selection.csv",
        "appendix_stability_overview.png",
        "appendix_stability_overview.svg",
        "appendix_stability_hdr.png",
        "appendix_stability_hdr.svg",
    }
    assert expected == {path.name for path in outdir.iterdir()}
    config = json.loads((outdir / "bootstrap_config.json").read_text(encoding="utf-8"))
    assert config["hpo_selection"] == selection
    assert config["method_classification"] == "parameter_perturbation_repeated_simulation_not_observation_resampling"
    assert config["perturbation_space"] == "HPO log10 alpha/beta with w_dis fixed at 1"
    assert config["failure_count"] == 0
    assert config["max_anchor_drift_sim"] == pytest.approx(0.0)
    assert len(config["artifact_sha256"]) == 24
    assert len(pd.read_csv(outdir / "bootstrap_samples_y_up_sim.csv")) == 12
    assert result["samples"].shape == (3, 4, 2)

    with pytest.raises(FileExistsError):
        bootstrap_script.run_bootstrap_stability(
            hpo_outdir=tmp_path / "hpo",
            outdir=outdir,
            n_bootstrap=3,
        )


def test_run_bootstrap_stability_rejects_anchor_drift(monkeypatch, tmp_path):
    vertice = ["A", "B", "C", "D"]
    dni = {label: index for index, label in enumerate(vertice)}
    samples = np.zeros((2, 4, 2), dtype=float)
    samples[1, 0, 0] = 0.01
    monkeypatch.setattr(
        bootstrap_script,
        "_load_physics_hpo_selection",
        lambda _path: {
            "alpha": 0.0,
            "beta": 0.0,
            "w_dis": 1.0,
            "spring_stiffness": 1.0,
            "directional_force": 1.0,
            "repulsion_strength": 1.0,
        },
    )
    monkeypatch.setattr(bootstrap_script, "_validate_hpo_provenance", lambda *_args: {})
    monkeypatch.setattr(bootstrap_script, "load_ini_data_from_csv", lambda _paths: (None, vertice, dni, None, None))
    monkeypatch.setattr(bootstrap_script, "uploading_ground_truth", lambda *_args: [(1.0, 1.0)] * 4)
    monkeypatch.setattr(bootstrap_script, "get_anchor_labels", lambda: ["A", "B", "C"])
    monkeypatch.setattr(bootstrap_script, "get_anchor_align_label", lambda: "A")
    monkeypatch.setattr(bootstrap_script, "get_test_site_labels", lambda: ["D"])
    monkeypatch.setattr(
        bootstrap_script,
        "bootstrap_dynamics",
        lambda *_args, **_kwargs: (
            samples,
            vertice,
            dni,
            [{"simulation_seed": 0}, {"simulation_seed": 1}],
        ),
    )
    monkeypatch.setattr(
        bootstrap_script,
        "_target_positions_sim",
        lambda *_args: {"A": np.zeros(2), "B": np.zeros(2), "C": np.zeros(2)},
    )
    with pytest.raises(ValueError, match="anchor drift"):
        bootstrap_script.run_bootstrap_stability(
            hpo_outdir=tmp_path / "hpo",
            outdir=tmp_path / "out",
            n_bootstrap=2,
        )


def test_paper_results_accepts_validated_300_run_bootstrap(tmp_path):
    outdir = tmp_path / "bootstrap"
    expected = _write_formal_bootstrap_fixture(outdir)
    actual = paper_results_updater._validate_bootstrap_source(
        outdir,
        {"alpha": 1.0, "beta": -0.5},
    )
    assert actual == expected


def test_paper_results_rejects_bootstrap_smoke_output(tmp_path):
    outdir = tmp_path / "bootstrap"
    _write_formal_bootstrap_fixture(outdir, n_bootstrap=3)
    with pytest.raises(ValueError, match="at least 300 runs"):
        paper_results_updater._validate_bootstrap_source(
            outdir,
            {"alpha": 1.0, "beta": -0.5},
        )
