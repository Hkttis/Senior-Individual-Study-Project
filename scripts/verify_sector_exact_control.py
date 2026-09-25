"""Independently recompute and verify the sector/exact control outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from library.config import FILE_PATHS, refer_pos_sim as DEFAULT_REFER_POS_SIM
from library.data_io import load_ini_data_from_csv, uploading_directional_data
from library.directional_objectives import (
    DIRECTIONAL_OBJECTIVE_EXACT,
    mean_absolute_nominal_angular_deviation,
)
from library.metrics import (
    calculate_kruskals_stress,
    direction_violation_rate,
    mean_angular_error_violations,
)
from library.units import data_Li2sim, pos_matrix_sim2km
from run_paper_script.ch5_ablation_progressive import _layout_metrics
from run_paper_script.ch5_hparam_kfold_gridsearch_pareto import (
    _resolve_anchor_and_test_inputs,
    _rmse_labels_km,
)
from run_paper_script.ch5_sector_exact_control import (
    BOOTSTRAP_REPLICATES,
    EXACT_VARIANT,
    FORMAL_FINAL_SEEDS,
    FORMAL_HPO_SEEDS,
    OFFICIAL_METRICS,
    SECTOR_VARIANT,
    _sha256,
    paired_comparison,
    summarize_runs,
)


ATOL = 1e-9
RTOL = 1e-10


def _assert_close(actual: float, expected: float, label: str) -> float:
    error = abs(float(actual) - float(expected))
    if not np.isclose(actual, expected, rtol=RTOL, atol=ATOL, equal_nan=True):
        raise AssertionError(f"{label}: saved={actual!r}, recomputed={expected!r}, abs_error={error!r}")
    return error


def _compare_frames(saved: pd.DataFrame, recomputed: pd.DataFrame, keys: list[str], values: list[str]) -> float:
    left = saved.sort_values(keys).reset_index(drop=True)
    right = recomputed.sort_values(keys).reset_index(drop=True)
    if left[keys].astype(str).to_dict("records") != right[keys].astype(str).to_dict("records"):
        raise AssertionError(f"Row keys differ for {keys}.")
    maximum = 0.0
    for row_index in range(len(left)):
        key_text = ", ".join(f"{key}={left.loc[row_index, key]}" for key in keys)
        for value in values:
            maximum = max(
                maximum,
                _assert_close(left.loc[row_index, value], right.loc[row_index, value], f"{key_text}, {value}"),
            )
    return maximum


def verify(outdir: str | Path, *, formal: bool = False) -> dict:
    root = Path(outdir)
    required = {
        "config": root / "gridsearch_config.json",
        "protocol": root / "sector_exact_preflight_and_protocol.json",
        "positions": root / "selected_final_positions_y_up_sim.csv",
        "exact": root / "exact_direction_final_runs_by_seed.csv",
        "combined": root / "sector_exact_runs_by_seed.csv",
        "summary": root / "sector_exact_summary.csv",
        "paired": root / "sector_exact_paired_comparison.csv",
        "hpo_runs": root / "grid_runs_by_seed.csv",
        "hpo_summary": root / "grid_summary_cv.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing control-experiment outputs: {missing}")

    config = json.loads(required["config"].read_text(encoding="utf-8"))
    protocol = json.loads(required["protocol"].read_text(encoding="utf-8"))
    if config.get("directional_objective") != DIRECTIONAL_OBJECTIVE_EXACT:
        raise AssertionError("gridsearch_config.json is not marked as exact-direction.")
    if float(config.get("distance_scale", np.nan)) != 1.0:
        raise AssertionError("Control experiment must use distance_scale=1.0.")

    hpo_runs = pd.read_csv(required["hpo_runs"])
    hpo_summary = pd.read_csv(required["hpo_summary"])
    if set(hpo_runs["directional_objective"].dropna().astype(str)) != {DIRECTIONAL_OBJECTIVE_EXACT}:
        raise AssertionError("At least one HPO run is not marked exact-direction.")
    if set(hpo_summary["directional_objective"].dropna().astype(str)) != {DIRECTIONAL_OBJECTIVE_EXACT}:
        raise AssertionError("At least one HPO grid point is not marked exact-direction.")

    _graph, vertice, dni, _edges, distance_data = load_ini_data_from_csv(FILE_PATHS)
    directional_data = uploading_directional_data()
    data_sim = data_Li2sim(distance_data)
    anchor_labels, anchor_lonlat, test_labels, test_lonlat = _resolve_anchor_and_test_inputs(dni)
    if anchor_labels != list(config["anchor_labels"]) or test_labels != list(config["test_labels"]):
        raise AssertionError("Current anchor/test split differs from the experiment config.")
    frame_anchor = str(config["final_frame_anchor_label"])
    gt_labels = anchor_labels + test_labels
    gt_lonlat = anchor_lonlat + test_lonlat

    positions = pd.read_csv(required["positions"])
    recomputed_rows: list[dict] = []
    for seed, group in positions.groupby("seed", sort=True):
        ordered = group.sort_values("node_idx")
        if ordered["label"].tolist() != list(vertice):
            raise AssertionError(f"Position label order mismatch for seed {seed}.")
        points = ordered[["x_y_up_sim", "y_y_up_sim"]].to_numpy(float)
        row = {
            "variant": EXACT_VARIANT,
            "seed": int(seed),
            "RMSE_test_km": _rmse_labels_km(
                pos_y_up_sim=points,
                dni=dni,
                refer_pos_sim=DEFAULT_REFER_POS_SIM,
                gt_labels=gt_labels,
                gt_lonlat=gt_lonlat,
                eval_labels=test_labels,
                anchor_label_for_frame=frame_anchor,
            ),
            "E_distance_stress": float(
                calculate_kruskals_stress(dni, pos_matrix_sim2km(points.tolist()), data_sim)
            ),
            "E_direction_vr": float(direction_violation_rate(points, directional_data, dni)),
            "E_direction_mae": float(mean_angular_error_violations(points, directional_data, dni)),
            "E_direction_nominal_mae_rad": float(
                mean_absolute_nominal_angular_deviation(points, directional_data, dni)
            ),
        }
        row.update(_layout_metrics(points, vertice, dni, distance_data))
        recomputed_rows.append(row)
    recomputed_exact = pd.DataFrame(recomputed_rows)
    saved_exact = pd.read_csv(required["exact"])
    exact_values = [*OFFICIAL_METRICS, "E_direction_nominal_mae_rad"]
    max_exact_error = _compare_frames(saved_exact, recomputed_exact, ["variant", "seed"], exact_values)

    combined = pd.read_csv(required["combined"])
    expected_exact = combined[combined["variant"] == EXACT_VARIANT]
    max_combined_exact_error = _compare_frames(
        expected_exact, recomputed_exact, ["variant", "seed"], list(OFFICIAL_METRICS)
    )
    if set(combined["variant"]) != {SECTOR_VARIANT, EXACT_VARIANT}:
        raise AssertionError("Combined results do not contain exactly the two intended variants.")
    sector_seeds = sorted(combined.loc[combined["variant"] == SECTOR_VARIANT, "seed"].astype(int))
    exact_seeds = sorted(combined.loc[combined["variant"] == EXACT_VARIANT, "seed"].astype(int))
    if sector_seeds != exact_seeds:
        raise AssertionError("Sector and exact results are not paired on the same seeds.")

    recomputed_summary = summarize_runs(combined)
    saved_summary = pd.read_csv(required["summary"])
    max_summary_error = _compare_frames(
        saved_summary, recomputed_summary, ["variant", "metric"], ["n", "mean", "sd"]
    )
    recomputed_paired = paired_comparison(combined)
    saved_paired = pd.read_csv(required["paired"])
    max_paired_error = _compare_frames(
        saved_paired,
        recomputed_paired,
        ["comparison", "metric"],
        [
            "n_pairs", "paired_diff_mean", "paired_diff_sd",
            "paired_diff_ci95_lo", "paired_diff_ci95_hi",
            "bootstrap_replicates", "bootstrap_seed",
        ],
    )

    current_hashes = {
        "distance_edges": _sha256(FILE_PATHS["chen_data"]),
        "direction_edges": _sha256(FILE_PATHS["directional_data"]),
        "site_points": _sha256(FILE_PATHS["ground_truth_path"]),
        "sector_runs": _sha256(protocol["sector_source"]),
    }
    if current_hashes != protocol["input_sha256"]:
        raise AssertionError("One or more experiment inputs changed after the control run.")

    if formal:
        if config["hpo_seeds"] != list(FORMAL_HPO_SEEDS):
            raise AssertionError("Formal verification requires HPO seeds 0-9.")
        if config["final_evaluation_seeds"] != list(FORMAL_FINAL_SEEDS):
            raise AssertionError("Formal verification requires final seeds 0-99.")
        if len(hpo_summary) != 36:
            raise AssertionError("Formal verification requires all 36 HPO grid points.")
        if int(protocol["expected_total_model_runs"]) != 1180:
            raise AssertionError("Formal protocol must contain 1,180 model runs.")

    report = {
        "status": "passed",
        "formal_verification": bool(formal),
        "directional_objective": DIRECTIONAL_OBJECTIVE_EXACT,
        "n_hpo_grid_points": int(len(hpo_summary)),
        "n_hpo_run_records": int(len(hpo_runs)),
        "n_paired_final_seeds": int(len(exact_seeds)),
        "official_metric_count": int(len(OFFICIAL_METRICS)),
        "paired_ci_method": f"paired percentile bootstrap, {BOOTSTRAP_REPLICATES} replicates",
        "max_abs_error_exact_metric_recomputation": max_exact_error,
        "max_abs_error_combined_exact_metrics": max_combined_exact_error,
        "max_abs_error_summary": max_summary_error,
        "max_abs_error_paired": max_paired_error,
        "input_hashes_match": True,
    }
    (root / "sector_exact_verification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify sector/exact PhysicsSim control outputs.")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--formal", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    verify(args.outdir, formal=args.formal)


if __name__ == "__main__":
    main()
