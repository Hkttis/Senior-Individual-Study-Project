"""Independently verify detour scales, saved configurations, metrics, and summaries."""

from __future__ import annotations

import argparse
import json
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from library.config import FILE_PATHS, Li2km, Li2sim, km2sim
from library.data_io import (
    get_anchor_labels,
    get_default_frame_anchor_label,
    get_test_site_labels,
    load_ini_data_from_csv,
    load_site_points,
    read_CHEN_csvfile,
    uploading_directional_data,
)
from library.metrics import (
    calculate_kruskals_stress,
    direction_violation_rate,
    mean_angular_error_violations,
)
from library.units import data_Li2sim, pos_matrix_sim2km
from run_paper_script.ch5_detour_factor_sensitivity import (
    METRIC_COLUMNS,
    _assert_distance_sources_consistent,
    _completed_scenario,
    _distance_target_audit_frame,
    _input_hashes,
    _paired_comparisons,
    _scenario_name,
    _summarize_scenario,
)
from run_paper_script.ch5_hparam_kfold_gridsearch_pareto import (
    _rmse_labels_km,
    _scale_sim_distance_data,
    _site_errors_km,
)


def _failure(failures: list[str], label: str, actual: object, expected: object) -> None:
    failures.append(f"{label}: actual={actual!r}; expected={expected!r}")


def _assert_close(
    failures: list[str],
    label: str,
    actual: float,
    expected: float,
    *,
    atol: float,
    rtol: float,
) -> None:
    if not np.isclose(float(actual), float(expected), atol=atol, rtol=rtol, equal_nan=True):
        _failure(failures, label, actual, expected)


def _verify_visual(path: Path, failures: list[str]) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        failures.append(f"Visualization is missing or empty: {path}")
        return
    if path.suffix == ".svg":
        try:
            if not ET.parse(path).getroot().tag.lower().endswith("svg"):
                failures.append(f"Invalid SVG root element: {path}")
        except ET.ParseError as exc:
            failures.append(f"Invalid SVG document {path}: {exc}")
    elif path.suffix == ".png":
        data = path.read_bytes()[:24]
        if len(data) != 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
            failures.append(f"Invalid PNG signature: {path}")
        elif min(struct.unpack(">II", data[16:24])) <= 0:
            failures.append(f"Invalid PNG dimensions: {path}")


def _verify_distance_audit(
    *,
    scenario_dir: Path,
    data_li: list[list[object]],
    kappa: float,
    failures: list[str],
    atol: float,
    rtol: float,
) -> None:
    actual = pd.read_csv(scenario_dir / "distance_targets_audit.csv")
    expected = _distance_target_audit_frame(data_li, kappa)
    if len(actual) != len(expected):
        _failure(failures, f"{scenario_dir.name} distance edge count", len(actual), len(expected))
        return
    for column in ("edge_index", "source", "target"):
        if actual[column].tolist() != expected[column].tolist():
            failures.append(f"{scenario_dir.name} distance-audit column differs: {column}")
    for column in (
        "original_distance_li",
        "unscaled_target_sim",
        "scaled_target_sim",
        "unscaled_target_km",
        "scaled_target_km",
        "distance_scale",
        "applied_ratio",
    ):
        if not np.allclose(actual[column].to_numpy(float), expected[column].to_numpy(float), atol=atol, rtol=rtol):
            failures.append(f"{scenario_dir.name} distance-audit values differ: {column}")
    source_distances_li = np.asarray([float(row[2]) for row in data_li], dtype=float)
    independently_expected = {
        "unscaled_target_sim": source_distances_li * Li2sim,
        "scaled_target_sim": source_distances_li * Li2sim * kappa,
        "unscaled_target_km": source_distances_li * Li2km,
        "scaled_target_km": source_distances_li * Li2km * kappa,
    }
    for column, values in independently_expected.items():
        if not np.allclose(actual[column].to_numpy(float), values, atol=atol, rtol=rtol):
            failures.append(f"{scenario_dir.name} independently reconstructed distance is incorrect: {column}")
    ratios = actual["scaled_target_sim"].to_numpy(float) / actual["unscaled_target_sim"].to_numpy(float)
    if not np.allclose(ratios, kappa, atol=atol, rtol=rtol):
        failures.append(f"{scenario_dir.name} distance targets were not scaled exactly once by kappa={kappa}.")
    km_ratios = actual["scaled_target_km"].to_numpy(float) * km2sim / actual["scaled_target_sim"].to_numpy(float)
    if not np.allclose(km_ratios, 1.0, atol=atol, rtol=rtol):
        failures.append(f"{scenario_dir.name} simulation-to-km conversion is inconsistent.")


def _verify_hpo_rows(
    *,
    scenario_dir: Path,
    kappa: float,
    anchors: list[str],
    tests: list[str],
    hpo_seeds: list[int],
    alpha_count: int,
    beta_count: int,
    failures: list[str],
    atol: float,
    rtol: float,
) -> None:
    config = json.loads((scenario_dir / "gridsearch_config.json").read_text(encoding="utf-8"))
    _assert_close(failures, f"{scenario_dir.name} HPO distance_scale", config["distance_scale"], kappa, atol=atol, rtol=rtol)
    if list(config["anchor_labels"]) != anchors or list(config["test_labels"]) != tests:
        failures.append(f"{scenario_dir.name} HPO anchor/test labels do not match experiment configuration.")
    runs = pd.read_csv(scenario_dir / "grid_runs_by_seed.csv")
    expected_count = alpha_count * beta_count * 3 * len(hpo_seeds)
    if len(runs) != expected_count:
        _failure(failures, f"{scenario_dir.name} HPO run count", len(runs), expected_count)
    if not np.allclose(runs["distance_scale"].to_numpy(float), kappa, atol=atol, rtol=rtol):
        failures.append(f"{scenario_dir.name} HPO rows do not use the scenario distance scale.")
    for filename in ("grid_folds_mean_std.csv", "grid_summary_cv.csv", "pareto_front_3d.csv"):
        path = scenario_dir / filename
        if not path.is_file():
            failures.append(f"{scenario_dir.name} missing HPO output: {filename}")
            continue
        summary = pd.read_csv(path)
        if "distance_scale" not in summary or not np.allclose(
            summary["distance_scale"].to_numpy(float), kappa, atol=atol, rtol=rtol
        ):
            failures.append(f"{scenario_dir.name} {filename} does not preserve the scenario distance scale.")
    for _, row in runs.iterrows():
        train = set(str(row["train_labels"]).split("|"))
        heldout = str(row["heldout_label"])
        frame = str(row["train_anchor_label"])
        if len(train) != 2 or heldout not in anchors or heldout in train or frame not in train:
            failures.append(f"{scenario_dir.name} invalid three-anchor LOO partition in fold {row['fold_id']}.")
            break
        if train & set(tests) or heldout in tests:
            failures.append(f"{scenario_dir.name} held-out test sites leaked into HPO fold {row['fold_id']}.")
            break


def _verify_fixed_hyperparameter_rows(
    *,
    scenario_dir: Path,
    kappa: float,
    fixed_alpha: float,
    fixed_beta: float,
    expected_policy: str = "fixed",
    expected_selection_rule: str = "predefined_fixed_hyperparameters",
    failures: list[str],
    atol: float,
    rtol: float,
) -> None:
    config = json.loads((scenario_dir / "gridsearch_config.json").read_text(encoding="utf-8"))
    summary = json.loads((scenario_dir / "selected_final_summary.json").read_text(encoding="utf-8"))
    runs = pd.read_csv(scenario_dir / "selected_final_runs_by_seed.csv")
    if config.get("hyperparameter_policy") != expected_policy:
        failures.append(f"{scenario_dir.name} is not marked with policy {expected_policy}.")
    if config.get("selection_rule") != expected_selection_rule:
        failures.append(f"{scenario_dir.name} has an incorrect fixed-parameter selection rule.")
    for source_name, source in (("config", config), ("summary", summary)):
        _assert_close(failures, f"{scenario_dir.name} {source_name} distance_scale", source["distance_scale"], kappa, atol=atol, rtol=rtol)
        _assert_close(failures, f"{scenario_dir.name} {source_name} alpha", source["alpha"], fixed_alpha, atol=atol, rtol=rtol)
        _assert_close(failures, f"{scenario_dir.name} {source_name} beta", source["beta"], fixed_beta, atol=atol, rtol=rtol)
    if not np.allclose(runs["alpha"].to_numpy(float), fixed_alpha, atol=atol, rtol=rtol):
        failures.append(f"{scenario_dir.name} final runs do not all use fixed alpha={fixed_alpha}.")
    if not np.allclose(runs["beta"].to_numpy(float), fixed_beta, atol=atol, rtol=rtol):
        failures.append(f"{scenario_dir.name} final runs do not all use fixed beta={fixed_beta}.")
    if not np.allclose(runs["distance_scale"].to_numpy(float), kappa, atol=atol, rtol=rtol):
        failures.append(f"{scenario_dir.name} final runs do not all use kappa={kappa}.")
    if set(runs["selection_rule"].astype(str)) != {expected_selection_rule}:
        failures.append(f"{scenario_dir.name} final rows have an incorrect selection rule.")
    unexpected = [
        name
        for name in ("grid_runs_by_seed.csv", "grid_summary_cv.csv", "pareto_front_3d.csv")
        if (scenario_dir / name).exists()
    ]
    if unexpected:
        failures.append(f"{scenario_dir.name} fixed mode unexpectedly contains HPO artifacts: {unexpected}")


def _verify_position_metrics(
    *,
    scenario_dir: Path,
    kappa: float,
    vertices: list[str],
    dni: dict[str, int],
    data_li: list[list[object]],
    anchors: list[str],
    test_labels: list[str],
    site_lonlat: dict[str, tuple[float, float]],
    directional_data: list[list[object]],
    failures: list[str],
    atol: float,
    rtol: float,
) -> int:
    runs = pd.read_csv(scenario_dir / "selected_final_runs_by_seed.csv")
    positions = pd.read_csv(scenario_dir / "selected_final_positions_y_up_sim.csv")
    site_errors = pd.read_csv(scenario_dir / "selected_final_site_errors.csv")
    grid_config = json.loads((scenario_dir / "gridsearch_config.json").read_text(encoding="utf-8"))
    frame_anchor = grid_config["final_frame_anchor_label"]
    reference = grid_config["refer_pos_sim"]
    labels = anchors + test_labels
    lonlat = [site_lonlat[label] for label in labels]
    scaled_data = _scale_sim_distance_data(data_Li2sim(data_li), kappa)
    checked = 0
    for _, row in runs.iterrows():
        seed = int(row["seed"])
        frame = positions[positions["seed"] == seed].sort_values("node_idx")
        if frame["node_idx"].astype(int).tolist() != list(range(len(vertices))):
            failures.append(f"{scenario_dir.name} seed {seed} has missing or duplicate node positions.")
            continue
        if frame["label"].astype(str).tolist() != vertices:
            failures.append(f"{scenario_dir.name} seed {seed} node ordering differs from dni.")
            continue
        pos_sim = frame[["x_y_up_sim", "y_y_up_sim"]].to_numpy(float)
        pos_km = pos_matrix_sim2km(pos_sim.tolist())
        recomputed = {
            "E_distance_stress": calculate_kruskals_stress(dni, pos_km, scaled_data),
            "E_direction_vr": direction_violation_rate(pos_sim, directional_data, dni),
            "E_direction_mae": mean_angular_error_violations(pos_sim, directional_data, dni),
            "RMSE_final_test_km": _rmse_labels_km(
                pos_y_up_sim=pos_sim,
                dni=dni,
                refer_pos_sim=reference,
                gt_labels=labels,
                gt_lonlat=lonlat,
                eval_labels=test_labels,
                anchor_label_for_frame=frame_anchor,
            ),
        }
        for metric, expected in recomputed.items():
            _assert_close(
                failures,
                f"{scenario_dir.name} seed={seed} {metric}",
                row[metric],
                expected,
                atol=atol,
                rtol=rtol,
            )
        expected_errors = _site_errors_km(
            pos_y_up_sim=pos_sim,
            dni=dni,
            refer_pos_sim=reference,
            gt_labels=labels,
            gt_lonlat=lonlat,
            eval_labels=test_labels,
            anchor_label_for_frame=frame_anchor,
        )
        observed = site_errors[site_errors["seed"] == seed]
        if set(observed["site_label"].astype(str)) != set(test_labels) or len(observed) != len(test_labels):
            failures.append(f"{scenario_dir.name} seed {seed} site-error labels differ from held-out sites.")
        else:
            for _, error_row in observed.iterrows():
                label = str(error_row["site_label"])
                _assert_close(
                    failures,
                    f"{scenario_dir.name} seed={seed} site={label}",
                    error_row["error_km"],
                    expected_errors[label],
                    atol=atol,
                    rtol=rtol,
                )
        checked += 1
    return checked


def _verify_loo_review(
    *,
    scenario_dir: Path,
    kappa: float,
    failures: list[str],
    atol: float,
    rtol: float,
) -> None:
    review_path = scenario_dir / "loo_fold_review" / "all_loo_fold_review.csv"
    if not review_path.is_file():
        return
    review = pd.read_csv(review_path)
    if "distance_scale" not in review or not np.allclose(review["distance_scale"].to_numpy(float), kappa, atol=atol, rtol=rtol):
        failures.append(f"{scenario_dir.name} LOO visualization does not use the scenario distance scale.")
        return
    runs = pd.read_csv(scenario_dir / "grid_runs_by_seed.csv")
    heldout = review[review["role"] == "anchor_heldout"]
    for _, row in heldout.iterrows():
        matches = runs[
            (runs["fold_id"].astype(int) == int(row["fold_id"]))
            & (runs["seed"].astype(int) == int(row["seed"]))
            & np.isclose(runs["alpha"].to_numpy(float), float(row["alpha"]))
            & np.isclose(runs["beta"].to_numpy(float), float(row["beta"]))
        ]
        if len(matches) != 1:
            failures.append(f"{scenario_dir.name} LOO visualization does not identify one HPO source row.")
            continue
        _assert_close(
            failures,
            f"{scenario_dir.name} fold={row['fold_id']} visualization RMSE",
            row["fold_RMSE_anchor_LOO_km"],
            matches.iloc[0]["RMSE_anchor_LOO_km"],
            atol=atol,
            rtol=rtol,
        )


def verify_detour_sensitivity(
    *,
    outdir: Path,
    allow_incomplete: bool = False,
    atol: float = 1e-8,
    rtol: float = 1e-9,
) -> tuple[list[str], dict]:
    outdir = outdir.resolve()
    failures: list[str] = []
    config_path = outdir / "detour_experiment_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing detour experiment configuration: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    current_hashes = _input_hashes()
    if current_hashes != config.get("input_sha256"):
        failures.append("One or more original input files differ from the recorded SHA-256 values.")
    graph, vertices, dni, _edges, data_li = load_ini_data_from_csv(FILE_PATHS)
    del graph
    _assert_distance_sources_consistent(read_CHEN_csvfile(), data_li)
    anchors = list(config["anchor_labels"])
    tests = list(config["test_labels"])
    if anchors != get_anchor_labels() or tests != get_test_site_labels():
        failures.append("Calibration anchors or held-out test sites differ from current site roles.")
    if config["final_frame_anchor_label"] != get_default_frame_anchor_label():
        failures.append("Frame anchor differs from the current calibration anchor.")
    site_lonlat = {
        row["name"]: (float(row["lon"]), float(row["lat"]))
        for row in load_site_points()
    }
    directional_data = uploading_directional_data()
    hyperparameter_policy = config.get("hyperparameter_policy", "scenario_specific_hpo")
    alpha_min, alpha_max, alpha_step = config["alpha_range"]
    beta_min, beta_max, beta_step = config["beta_range"]
    alpha_count = int(round((alpha_max - alpha_min) / alpha_step)) + 1
    beta_count = int(round((beta_max - beta_min) / beta_step)) + 1
    expected_summaries: list[dict] = []
    expected_runs: list[pd.DataFrame] = []
    expected_errors: list[pd.DataFrame] = []
    checked_scenarios = 0
    checked_runs = 0

    for kappa in config["scenario_scales"]:
        scenario_dir = outdir / "scenarios" / _scenario_name(float(kappa))
        if not _completed_scenario(scenario_dir, kappa=float(kappa), final_seeds=config["final_evaluation_seeds"]):
            if not allow_incomplete:
                failures.append(f"Scenario is incomplete or contains invalid output: {scenario_dir}")
            continue
        _verify_distance_audit(
            scenario_dir=scenario_dir,
            data_li=data_li,
            kappa=float(kappa),
            failures=failures,
            atol=atol,
            rtol=rtol,
        )
        scenario_config = json.loads((scenario_dir / "gridsearch_config.json").read_text(encoding="utf-8"))
        scenario_policy = scenario_config.get("hyperparameter_policy", "scenario_specific_hpo")
        if scenario_policy == "fixed":
            _verify_fixed_hyperparameter_rows(
                scenario_dir=scenario_dir,
                kappa=float(kappa),
                fixed_alpha=float(config["fixed_alpha"]),
                fixed_beta=float(config["fixed_beta"]),
                failures=failures,
                atol=atol,
                rtol=rtol,
            )
        elif scenario_policy == "fixed_reference":
            reference = config.get("reference_hyperparameters") or {}
            _verify_fixed_hyperparameter_rows(
                scenario_dir=scenario_dir,
                kappa=float(kappa),
                fixed_alpha=float(reference["alpha"]),
                fixed_beta=float(reference["beta"]),
                expected_policy="fixed_reference",
                expected_selection_rule="predefined_formal_reference_hyperparameters",
                failures=failures,
                atol=atol,
                rtol=rtol,
            )
        else:
            _verify_hpo_rows(
                scenario_dir=scenario_dir,
                kappa=float(kappa),
                anchors=anchors,
                tests=tests,
                hpo_seeds=list(config["hpo_seeds"]),
                alpha_count=alpha_count,
                beta_count=beta_count,
                failures=failures,
                atol=atol,
                rtol=rtol,
            )
        checked_runs += _verify_position_metrics(
            scenario_dir=scenario_dir,
            kappa=float(kappa),
            vertices=vertices,
            dni=dni,
            data_li=data_li,
            anchors=anchors,
            test_labels=tests,
            site_lonlat=site_lonlat,
            directional_data=directional_data,
            failures=failures,
            atol=atol,
            rtol=rtol,
        )
        if scenario_policy == "scenario_specific_hpo":
            _verify_loo_review(scenario_dir=scenario_dir, kappa=float(kappa), failures=failures, atol=atol, rtol=rtol)
        summary, runs, errors = _summarize_scenario(scenario_dir, kappa=float(kappa))
        expected_summaries.append(summary)
        expected_runs.append(runs)
        expected_errors.append(errors)
        checked_scenarios += 1

    if checked_scenarios:
        actual_summary = pd.read_csv(outdir / "detour_scenario_summary.csv")
        expected_summary = pd.DataFrame(expected_summaries).sort_values("kappa", ascending=False).reset_index(drop=True)
        try:
            pd.testing.assert_frame_equal(
                actual_summary.sort_values("kappa", ascending=False).reset_index(drop=True),
                expected_summary,
                check_dtype=False,
                check_exact=False,
                atol=atol,
                rtol=rtol,
            )
        except AssertionError as exc:
            failures.append(f"Aggregate scenario summary differs from the independent reconstruction: {exc}")
        combined_runs = pd.concat(expected_runs, ignore_index=True)
        actual_runs = pd.read_csv(outdir / "detour_final_runs.csv")
        try:
            pd.testing.assert_frame_equal(
                actual_runs.sort_values(["kappa", "seed"]).reset_index(drop=True),
                combined_runs.sort_values(["kappa", "seed"]).reset_index(drop=True),
                check_dtype=False,
                check_exact=False,
                atol=atol,
                rtol=rtol,
            )
        except AssertionError as exc:
            failures.append(f"Aggregate final runs differ from scenario outputs: {exc}")
        actual_site_errors = pd.read_csv(outdir / "detour_site_errors.csv")
        combined_site_errors = pd.concat(expected_errors, ignore_index=True)
        try:
            pd.testing.assert_frame_equal(
                actual_site_errors.sort_values(["kappa", "seed", "site_label"]).reset_index(drop=True),
                combined_site_errors.sort_values(["kappa", "seed", "site_label"]).reset_index(drop=True),
                check_dtype=False,
                check_exact=False,
                atol=atol,
                rtol=rtol,
            )
        except AssertionError as exc:
            failures.append(f"Aggregate held-out site errors differ from scenario outputs: {exc}")
        expected_pairs = _paired_comparisons(combined_runs)
        pairs_path = outdir / "detour_paired_comparisons.csv"
        if expected_pairs.empty:
            if pairs_path.stat().st_size > 4:
                failures.append("Paired comparisons should be empty with reference scenario only.")
        else:
            actual_pairs = pd.read_csv(pairs_path)
            try:
                pd.testing.assert_frame_equal(
                    actual_pairs.sort_values(["kappa", "metric"]).reset_index(drop=True),
                    expected_pairs.sort_values(["kappa", "metric"]).reset_index(drop=True),
                    check_dtype=False,
                    check_exact=False,
                    atol=atol,
                    rtol=rtol,
                )
            except AssertionError as exc:
                failures.append(f"Paired bootstrap comparison differs from recomputed values: {exc}")
        for stem in ("detour_rmse_sensitivity", "detour_selected_hyperparameters", "detour_secondary_metrics"):
            for suffix in (".png", ".svg"):
                _verify_visual(outdir / f"{stem}{suffix}", failures)

    report = {
        "outdir": str(outdir),
        "checked_scenarios": checked_scenarios,
        "expected_scenarios": len(config["scenario_scales"]),
        "checked_final_runs": checked_runs,
        "distance_edges_per_scenario": len(data_li),
        "metrics_per_run": list(METRIC_COLUMNS),
        "hyperparameter_policy": hyperparameter_policy,
        "input_sha256_unchanged": current_hashes == config.get("input_sha256"),
        "allow_incomplete": allow_incomplete,
    }
    return failures, report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--rtol", type=float, default=1e-9)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    failures, report = verify_detour_sensitivity(
        outdir=Path(args.outdir),
        allow_incomplete=args.allow_incomplete,
        atol=args.atol,
        rtol=args.rtol,
    )
    if failures:
        print("[FAIL] Detour sensitivity verification failed")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("[OK] Detour sensitivity outputs are consistent with source data and saved positions")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
