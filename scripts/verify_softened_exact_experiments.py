"""Verify softened-exact Full-HPO or matched-control experiment outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from library.config import FILE_PATHS, refer_pos_sim
from library.data_io import (
    get_test_site_labels,
    load_ini_data_from_csv,
    uploading_directional_data,
    uploading_ground_truth,
)
from library.directional_objectives import DIRECTIONAL_OBJECTIVE_SOFT_EXACT
from library.metrics import (
    calculate_kruskals_stress,
    direction_violation_rate,
    mean_angular_error_violations,
)
from library.units import data_Li2sim, pos_matrix_sim2km
from run_paper_script.ch5_ablation_progressive import (
    _evaluate,
    _layout_metrics,
    _target_positions_sim,
)
from run_paper_script.ch5_hparam_kfold_gridsearch_pareto import (
    _resolve_anchor_and_test_inputs,
    _rmse_labels_km,
)
from run_paper_script.ch5_sector_exact_control import (
    FORMAL_FINAL_SEEDS,
    FORMAL_HPO_SEEDS,
    OFFICIAL_METRICS,
    SECTOR_VARIANT,
)
from run_paper_script.ch5_sector_soft_exact_control import (
    SOFT_EXACT_FORMAL_GRID,
    SOFT_EXACT_VARIANT,
    paired_comparison,
    summarize_runs,
)


ATOL = 1e-9
RTOL = 1e-10


def _hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _close(actual, expected, label: str) -> float:
    error = abs(float(actual) - float(expected))
    if not np.isclose(actual, expected, rtol=RTOL, atol=ATOL, equal_nan=True):
        raise AssertionError(f"{label}: saved={actual}, recomputed={expected}")
    return error


def _frame_close(saved, recomputed, keys, values) -> float:
    left = saved.sort_values(keys).reset_index(drop=True)
    right = recomputed.sort_values(keys).reset_index(drop=True)
    if left[keys].astype(str).to_dict("records") != right[keys].astype(str).to_dict("records"):
        raise AssertionError(f"Row keys differ: {keys}")
    maximum = 0.0
    for index in range(len(left)):
        for value in values:
            maximum = max(
                maximum,
                _close(left.loc[index, value], right.loc[index, value], f"row={index}, {value}"),
            )
    return maximum


def verify_full(out: Path, formal: bool) -> dict:
    paths = {
        "config": out / "gridsearch_config.json",
        "protocol": out / "soft_exact_preflight_and_protocol.json",
        "hpo_runs": out / "grid_runs_by_seed.csv",
        "hpo_grid": out / "grid_summary_cv.csv",
        "positions": out / "selected_final_positions_y_up_sim.csv",
        "soft": out / "soft_exact_direction_final_runs_by_seed.csv",
        "combined": out / "sector_soft_exact_runs_by_seed.csv",
        "summary": out / "sector_soft_exact_summary.csv",
        "paired": out / "sector_soft_exact_paired_comparison.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing outputs: {missing}")
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    protocol = json.loads(paths["protocol"].read_text(encoding="utf-8"))
    delta = float(protocol["direction_softening_delta"])
    if config["directional_objective"] != DIRECTIONAL_OBJECTIVE_SOFT_EXACT:
        raise AssertionError("HPO config is not softened-exact.")
    _close(config["direction_softening_delta"], delta, "HPO delta")
    hpo_runs = pd.read_csv(paths["hpo_runs"])
    hpo_grid = pd.read_csv(paths["hpo_grid"])
    if set(hpo_runs["directional_objective"]) != {DIRECTIONAL_OBJECTIVE_SOFT_EXACT}:
        raise AssertionError("HPO rows mix directional objectives.")
    if not np.allclose(hpo_runs["direction_softening_delta"], delta):
        raise AssertionError("HPO rows mix softening delta values.")

    _, vertices, dni, _, distance_data = load_ini_data_from_csv(FILE_PATHS)
    directions = uploading_directional_data()
    data_sim = data_Li2sim(distance_data)
    anchors, anchor_lonlat, tests, test_lonlat = _resolve_anchor_and_test_inputs(dni)
    frame_anchor = config["final_frame_anchor_label"]
    positions = pd.read_csv(paths["positions"])
    rows = []
    for seed, group in positions.groupby("seed", sort=True):
        ordered = group.sort_values("node_idx")
        if ordered["label"].tolist() != list(vertices):
            raise AssertionError(f"Node order mismatch for seed {seed}")
        points = ordered[["x_y_up_sim", "y_y_up_sim"]].to_numpy(float)
        row = {
            "variant": SOFT_EXACT_VARIANT,
            "seed": int(seed),
            "RMSE_test_km": _rmse_labels_km(
                pos_y_up_sim=points,
                dni=dni,
                refer_pos_sim=refer_pos_sim,
                gt_labels=anchors + tests,
                gt_lonlat=anchor_lonlat + test_lonlat,
                eval_labels=tests,
                anchor_label_for_frame=frame_anchor,
            ),
            "E_distance_stress": calculate_kruskals_stress(
                dni, pos_matrix_sim2km(points.tolist()), data_sim
            ),
            "E_direction_vr": direction_violation_rate(points, directions, dni),
            "E_direction_mae": mean_angular_error_violations(points, directions, dni),
        }
        row.update(_layout_metrics(points, vertices, dni, distance_data))
        rows.append(row)
    recomputed = pd.DataFrame(rows)
    saved_soft = pd.read_csv(paths["soft"])
    metric_error = _frame_close(
        saved_soft, recomputed, ["variant", "seed"], list(OFFICIAL_METRICS)
    )
    combined = pd.read_csv(paths["combined"])
    if set(combined["variant"]) != {SECTOR_VARIANT, SOFT_EXACT_VARIANT}:
        raise AssertionError("Combined Full results contain unexpected variants.")
    summary_error = _frame_close(
        pd.read_csv(paths["summary"]),
        summarize_runs(combined),
        ["variant", "metric"],
        ["n", "mean", "sd"],
    )
    paired_error = _frame_close(
        pd.read_csv(paths["paired"]),
        paired_comparison(combined),
        ["comparison", "metric"],
        [
            "n_pairs",
            "paired_diff_mean",
            "paired_diff_sd",
            "paired_diff_ci95_lo",
            "paired_diff_ci95_hi",
        ],
    )
    current_hashes = {
        "distance_edges": _hash(FILE_PATHS["chen_data"]),
        "direction_edges": _hash(FILE_PATHS["directional_data"]),
        "site_points": _hash(FILE_PATHS["ground_truth_path"]),
        "sector_runs": _hash(protocol["sector_source"]),
    }
    if current_hashes != protocol["input_sha256"]:
        raise AssertionError("Current inputs differ from the recorded Full experiment inputs.")
    if formal:
        if config["hpo_seeds"] != list(FORMAL_HPO_SEEDS):
            raise AssertionError("Formal Full HPO requires seeds 0-9.")
        if config["final_evaluation_seeds"] != list(FORMAL_FINAL_SEEDS):
            raise AssertionError("Formal Full evaluation requires seeds 0-99.")
        expected_alpha_range = [
            SOFT_EXACT_FORMAL_GRID["alpha_min"],
            SOFT_EXACT_FORMAL_GRID["alpha_max"],
            SOFT_EXACT_FORMAL_GRID["alpha_step"],
        ]
        expected_beta_range = [
            SOFT_EXACT_FORMAL_GRID["beta_min"],
            SOFT_EXACT_FORMAL_GRID["beta_max"],
            SOFT_EXACT_FORMAL_GRID["beta_step"],
        ]
        if not np.allclose(config["alpha_range"], expected_alpha_range):
            raise AssertionError("Formal Full HPO alpha range does not match the registered grid.")
        if not np.allclose(config["beta_range"], expected_beta_range):
            raise AssertionError("Formal Full HPO beta range does not match the registered grid.")
        alpha_count = int(round(
            (SOFT_EXACT_FORMAL_GRID["alpha_max"] - SOFT_EXACT_FORMAL_GRID["alpha_min"])
            / SOFT_EXACT_FORMAL_GRID["alpha_step"]
        )) + 1
        beta_count = int(round(
            (SOFT_EXACT_FORMAL_GRID["beta_max"] - SOFT_EXACT_FORMAL_GRID["beta_min"])
            / SOFT_EXACT_FORMAL_GRID["beta_step"]
        )) + 1
        expected_candidates = alpha_count * beta_count
        expected_runs = expected_candidates * 3 * len(FORMAL_HPO_SEEDS)
        if len(hpo_grid) != expected_candidates or len(hpo_runs) != expected_runs:
            raise AssertionError(
                "Formal Full HPO output count does not match the registered grid and seed protocol."
            )
    return {
        "kind": "full",
        "status": "passed",
        "direction_softening_delta": delta,
        "max_metric_error": metric_error,
        "max_summary_error": summary_error,
        "max_paired_error": paired_error,
        "input_hashes_match": True,
    }


def verify_matched(out: Path, formal: bool) -> dict:
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    delta = float(protocol["direction_softening_delta"])
    root = Path(__file__).resolve().parents[1]
    snapshot = out / "executed_source_snapshot"
    for raw, expected in protocol["source_sha256"].items():
        original = Path(raw)
        saved = snapshot / original.relative_to(root)
        if _hash(saved) != expected:
            raise AssertionError(f"Source snapshot hash mismatch: {saved}")
    _, vertices, dni, _, distances = load_ini_data_from_csv(FILE_PATHS)
    directions = uploading_directional_data()
    tests = get_test_site_labels()
    targets = _target_positions_sim(
        dni, uploading_ground_truth(vertices, dni), "鄯善", refer_pos_sim
    )
    checks = {}
    for block in protocol["blocks"]:
        files = sorted((out / block).glob("seed_*.json"))
        if formal and len(files) != 100:
            raise AssertionError(f"Formal {block} requires 100 seed files.")
        max_error = 0.0
        counts = {}
        for path in files:
            data = json.loads(path.read_text(encoding="utf-8"))
            initial = np.asarray(data["initial_positions"], dtype=float)
            initial_hash = hashlib.sha256(initial.tobytes()).hexdigest()
            if {row["variant"] for row in data["runs"]} != {
                DIRECTIONAL_OBJECTIVE_SOFT_EXACT,
                "sector",
            }:
                raise AssertionError(f"Unexpected variants in {path}")
            for row in data["runs"]:
                if row["initial_sha256"] != initial_hash:
                    raise AssertionError(f"Initialization hash mismatch in {path}")
                counts[row["variant"] + "/" + row["status"]] = (
                    counts.get(row["variant"] + "/" + row["status"], 0) + 1
                )
                if "positions" not in row:
                    continue
                points = np.asarray(row["positions"], dtype=float)
                metrics = _evaluate(
                    row["variant"],
                    row["seed"],
                    points,
                    vertices,
                    dni,
                    data_Li2sim(distances),
                    directions,
                    tests,
                    targets,
                    distances,
                )
                for metric in OFFICIAL_METRICS:
                    max_error = max(
                        max_error,
                        _close(row[metric], metrics[metric], f"{path.name}, {metric}"),
                    )
                if row["variant"] == DIRECTIONAL_OBJECTIVE_SOFT_EXACT:
                    _close(row["direction_softening_delta"], delta, f"{path.name}, delta")
        checks[block] = {
            "n_seed_files": len(files),
            "completion_counts": counts,
            "max_metric_error": max_error,
        }
    return {
        "kind": "matched",
        "status": "passed",
        "direction_softening_delta": delta,
        "checks": checks,
        "executed_snapshot_hashes_match": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["full", "matched"], required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--formal", action="store_true")
    args = parser.parse_args()
    out = Path(args.outdir)
    result = verify_full(out, args.formal) if args.kind == "full" else verify_matched(out, args.formal)
    (out / "softened_exact_verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
