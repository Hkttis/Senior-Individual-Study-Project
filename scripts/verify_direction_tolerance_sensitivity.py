"""Recompute and verify direction-tolerance sensitivity outputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pandas as pd

from library.config import FILE_PATHS, refer_pos_sim
from library.data_io import (
    get_anchor_labels,
    get_default_frame_anchor_label,
    get_test_site_labels,
    load_ini_data_from_csv,
    uploading_directional_data,
    uploading_ground_truth,
)
from library.metrics import (
    calculate_kruskals_stress,
    direction_violation_rate,
    mean_angular_error_violations,
)
from library.units import data_Li2sim, pos_matrix_sim2km
from run_paper_script.ch5_ablation_progressive import _layout_metrics
from run_paper_script.ch5_direction_tolerance_sensitivity import (
    FORMAL_FINAL_SEEDS,
    FORMAL_HPO_SEEDS,
    FORMAL_NUMERATORS,
    METRICS,
    _input_hashes,
    angle_metadata,
    paired_comparisons,
    scenario_name,
    tolerance_scale,
)
from run_paper_script.ch5_hparam_kfold_gridsearch_pareto import _rmse_labels_km
from run_paper_script.ch5_sector_exact_control import DEFAULT_SECTOR_RUNS


ATOL = 1e-9
RTOL = 1e-10


def _max_error(actual: pd.Series, expected: np.ndarray, label: str) -> float:
    values = actual.to_numpy(float)
    expected = np.asarray(expected, dtype=float)
    if not np.allclose(values, expected, rtol=RTOL, atol=ATOL, equal_nan=True):
        index = int(np.nanargmax(np.abs(values - expected)))
        raise AssertionError(
            f"{label} mismatch at row {index}: saved={values[index]}, recomputed={expected[index]}"
        )
    differences = np.abs(values - expected)
    finite = differences[np.isfinite(differences)]
    return float(finite.max()) if len(finite) else 0.0


def verify(outdir: str | Path, *, formal: bool = False) -> dict:
    out = Path(outdir)
    config = json.loads(
        (out / "direction_tolerance_experiment_config.json").read_text(encoding="utf-8")
    )
    if config["directional_objective"] != "sector" or config.get("softening") is not False:
        raise AssertionError("Experiment must use the original unsoftened sector objective.")
    if config["input_sha256"] != _input_hashes():
        raise AssertionError("Current input files differ from the recorded experiment inputs.")
    numerators = [int(value) for value in config["scenario_numerators"]]
    final_seeds = [int(value) for value in config["final_evaluation_seeds"]]
    if formal:
        if tuple(numerators) != FORMAL_NUMERATORS:
            raise AssertionError("Formal experiment requires all seven registered scenarios.")
        if config["hpo_seeds"] != list(FORMAL_HPO_SEEDS):
            raise AssertionError("Formal HPO requires seeds 0-9.")
        if final_seeds != list(FORMAL_FINAL_SEEDS):
            raise AssertionError("Formal final evaluation requires seeds 0-99.")

    _graph, vertices, dni, _edges, distance_rows = load_ini_data_from_csv(FILE_PATHS)
    data_sim = data_Li2sim(distance_rows)
    directions = uploading_directional_data()
    gt_lonlat = uploading_ground_truth(vertices, dni)
    anchors = get_anchor_labels()
    tests = get_test_site_labels()
    frame_anchor = get_default_frame_anchor_label()
    gt_labels = anchors + tests
    gt_values = [tuple(gt_lonlat[dni[label]]) for label in gt_labels]
    max_metric_error = 0.0
    all_runs = []
    scenario_checks = {}

    for numerator in numerators:
        scenario = out / "scenarios" / scenario_name(numerator)
        audit = json.loads((scenario / "angle_tolerance_audit.json").read_text(encoding="utf-8"))
        expected_meta = angle_metadata(numerator)
        for key, expected in expected_meta.items():
            if not np.isclose(float(audit[key]), float(expected), rtol=0.0, atol=1e-12):
                raise AssertionError(f"Angle audit mismatch for {numerator}/6: {key}")
        runs = pd.read_csv(scenario / "scenario_runs_with_layout.csv").sort_values("seed")
        positions = pd.read_csv(
            scenario / "selected_final_positions_y_up_sim.csv"
        )
        if runs["seed"].astype(int).tolist() != final_seeds:
            raise AssertionError(f"Seed mismatch in scenario {numerator}/6.")
        if formal and numerator != 6:
            hpo_runs = pd.read_csv(scenario / "grid_runs_by_seed.csv")
            hpo_grid = pd.read_csv(scenario / "grid_summary_cv.csv")
            if len(hpo_runs) != 1080 or len(hpo_grid) != 36:
                raise AssertionError(f"Scenario {numerator}/6 has incomplete formal HPO output.")

        recomputed = {metric: [] for metric in METRICS}
        scale = tolerance_scale(numerator)
        for seed in final_seeds:
            group = positions[positions["seed"].astype(int).eq(seed)].sort_values("node_idx")
            if len(group) != len(vertices) or group["label"].tolist() != list(vertices):
                raise AssertionError(f"Position rows/order mismatch for {numerator}/6 seed {seed}.")
            points = group[["x_y_up_sim", "y_y_up_sim"]].to_numpy(float)
            recomputed["RMSE_final_test_km"].append(
                _rmse_labels_km(
                    pos_y_up_sim=points,
                    dni=dni,
                    refer_pos_sim=refer_pos_sim,
                    gt_labels=gt_labels,
                    gt_lonlat=gt_values,
                    eval_labels=tests,
                    anchor_label_for_frame=frame_anchor,
                )
            )
            recomputed["E_distance_stress"].append(
                calculate_kruskals_stress(dni, pos_matrix_sim2km(points.tolist()), data_sim)
            )
            recomputed["E_direction_vr"].append(
                direction_violation_rate(
                    points, directions, dni, direction_tolerance_scale=scale
                )
            )
            recomputed["E_direction_mae"].append(
                mean_angular_error_violations(
                    points, directions, dni, direction_tolerance_scale=scale
                )
            )
            recomputed["E_direction_vr_reference"].append(
                direction_violation_rate(points, directions, dni)
            )
            recomputed["E_direction_mae_reference"].append(
                mean_angular_error_violations(points, directions, dni)
            )
            layout = _layout_metrics(points, vertices, dni, distance_rows)
            for metric in METRICS[6:]:
                recomputed[metric].append(layout[metric])
        for metric in METRICS:
            max_metric_error = max(
                max_metric_error,
                _max_error(runs[metric], recomputed[metric], f"{numerator}/6 {metric}"),
            )
        with_ids = runs.copy()
        with_ids.insert(0, "tolerance_numerator", numerator)
        if "direction_tolerance_scale" not in with_ids.columns:
            with_ids.insert(1, "direction_tolerance_scale", scale)
        all_runs.append(with_ids)
        scenario_checks[scenario_name(numerator)] = {
            "n_seeds": len(runs),
            "direction_tolerance_scale": scale,
            "status": "passed",
        }

    combined = pd.concat(all_runs, ignore_index=True)
    paired_saved = pd.read_csv(out / "direction_tolerance_paired_comparisons.csv").sort_values(
        ["tolerance_numerator", "metric"]
    )
    paired_expected = paired_comparisons(combined).sort_values(
        ["tolerance_numerator", "metric"]
    )
    max_paired_error = 0.0
    for column in (
        "difference_mean",
        "difference_std",
        "difference_ci95_low",
        "difference_ci95_high",
        "win_rate_lower",
    ):
        max_paired_error = max(
            max_paired_error,
            _max_error(paired_saved[column], paired_expected[column], f"paired {column}"),
        )

    reference = combined[combined["tolerance_numerator"].eq(6)].copy()
    official = pd.read_csv(DEFAULT_SECTOR_RUNS)
    official = official[
        official["variant"].eq("PhysicsSim-Full") & official["status"].eq("ok")
    ].copy()
    if reference["seed"].duplicated().any() or official["seed"].duplicated().any():
        raise AssertionError("Reference comparison contains duplicate seeds.")
    mapping = {
        "RMSE_final_test_km": "RMSE_test_km",
        "E_distance_stress": "E_distance_stress",
        "E_direction_vr": "E_direction_vr",
        "E_direction_mae": "E_direction_mae",
        "crowding_violation_rate_tau_0p1": "crowding_violation_rate_tau_0p1",
        "collapse_node_rate_tau_0p1": "collapse_node_rate_tau_0p1",
        "nnd_q05_km": "nnd_q05_km",
        "distance_edge_crossing_rate": "distance_edge_crossing_rate",
    }
    current_columns = {name: f"current__{name}" for name in mapping}
    primary_columns = {name: f"primary__{name}" for name in mapping.values()}
    reference = reference[["seed", *mapping]].rename(columns=current_columns).merge(
        official[["seed", *mapping.values()]].rename(columns=primary_columns),
        on="seed",
        how="inner",
        validate="one_to_one",
    ).sort_values("seed")
    if reference["seed"].astype(int).tolist() != final_seeds:
        raise AssertionError("Primary PhysicsSim-Full results do not cover the same final seeds.")
    reference_errors = {}
    for current, prior in mapping.items():
        reference_errors[current] = _max_error(
            reference[f"current__{current}"],
            reference[f"primary__{prior}"].to_numpy(float),
            f"reference {current}",
        )

    result = {
        "status": "passed",
        "formal": bool(formal),
        "max_recomputed_metric_error": max_metric_error,
        "max_recomputed_paired_error": max_paired_error,
        "reference_max_error_vs_primary_physics_full": max(reference_errors.values()),
        "input_hashes_match": True,
        "scenario_checks": scenario_checks,
    }
    (out / "direction_tolerance_verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--formal", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.outdir, formal=args.formal), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
