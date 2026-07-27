"""Independently verify final manuscript tables against raw Progressive AS runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AS_OUTDIR = (
    PROJECT_ROOT
    / "outputs"
    / "ch5_progressive_as_physics_alpha_1_beta_-0.5_dc_alpha_-2_wang_current_100seeds_random1000_20260721"
)
DEFAULT_TABLE_DIR = PROJECT_ROOT / "paper_results" / "current" / "05_paper_tables"
DEFAULT_REPORT = PROJECT_ROOT / "outputs" / "manuscript_table_consistency_report_20260722.json"

METRICS = {
    "RMSE (km)": "RMSE_test_km",
    "Stress": "E_distance_stress",
    "Violation Rate": "E_direction_vr",
    "Mean Angular Error (rad)": "E_direction_mae",
    "Crowding Violation Rate (τ = 0.10)": "crowding_violation_rate_tau_0p1",
    "Collapse Node Rate (τ = 0.10)": "collapse_node_rate_tau_0p1",
    "Nearest-Neighbor Distance, 5th Quantile (km)": "nnd_q05_km",
    "Crossing-edge rate": "distance_edge_crossing_rate",
}
ALL_SUMMARY_METRICS = [
    "RMSE_test_km",
    "MAE_test_km",
    "median_error_km",
    "E_distance_stress",
    "E_direction_vr",
    "E_direction_mae",
    "crowding_violation_rate_tau_0p1",
    "collapse_node_rate_tau_0p1",
    "nnd_q05_km",
    "distance_edge_crossing_rate",
]
MODEL_SEEDS = set(range(100))
RANDOM_SEEDS = set(range(1000))
VARIANT_COUNTS = {
    "PhysicsSim-DistOnly": 100,
    "PhysicsSim-DistDir": 100,
    "PhysicsSim-DistDirAnch": 100,
    "PhysicsSim-Full": 100,
    "SMACOF": 100,
    "DC-SMACOF": 100,
    "Random+Align": 1000,
}
CORE_COLUMNS = [
    "Model",
    "RMSE (km)",
    "Stress",
    "Violation Rate",
    "Mean Angular Error (rad)",
    "Crowding Violation Rate (τ = 0.10)",
    "Collapse Node Rate (τ = 0.10)",
    "Nearest-Neighbor Distance, 5th Quantile (km)",
    "Crossing-edge rate",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _mean_sd(values: np.ndarray) -> str:
    return f"{float(np.mean(values)):.6g} ± {float(np.std(values, ddof=1)):.6g}"


def _bootstrap_ci(values: np.ndarray, *, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[indices].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _mean_ci(values: np.ndarray) -> str:
    lo, hi = _bootstrap_ci(values)
    return f"{float(np.mean(values)):.6g} [{lo:.6g}, {hi:.6g}]"


def _model_row(runs: pd.DataFrame, variant: str, labels: list[str]) -> dict[str, str]:
    group = runs[(runs["variant"] == variant) & (runs["status"] == "ok")]
    row = {"Model": variant}
    for label in labels:
        row[label] = _mean_sd(group[METRICS[label]].to_numpy(float))
    return row


def _paired_row(
    runs: pd.DataFrame,
    *,
    left: str,
    right: str,
    display_label: str,
    labels: list[str],
) -> dict[str, str]:
    left_rows = runs[(runs["variant"] == left) & (runs["status"] == "ok")].set_index("seed")
    right_rows = runs[(runs["variant"] == right) & (runs["status"] == "ok")].set_index("seed")
    seeds = sorted(set(left_rows.index).intersection(right_rows.index))
    row = {"Model": display_label}
    for label in labels:
        metric = METRICS[label]
        diff = left_rows.loc[seeds, metric].to_numpy(float) - right_rows.loc[seeds, metric].to_numpy(float)
        row[label] = _mean_ci(diff)
    return row


def _expected_tables(runs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    labels = CORE_COLUMNS[1:]
    random_mean = float(runs.loc[runs["variant"] == "Random+Align", "RMSE_test_km"].mean())
    rmse_models = [
        "Random+Align",
        "PhysicsSim-DistOnly",
        "SMACOF",
        "DC-SMACOF",
        "PhysicsSim-DistDir",
        "PhysicsSim-DistDirAnch",
        "PhysicsSim-Full",
    ]
    rmse_rows = []
    for variant in rmse_models:
        values = runs.loc[(runs["variant"] == variant) & (runs["status"] == "ok"), "RMSE_test_km"].to_numpy(float)
        reduction = "Reference" if variant == "Random+Align" else f"{(random_mean - values.mean()) / random_mean * 100.0:.2f}%"
        rmse_rows.append(
            {
                "Model": variant,
                "RMSE, mean ± SD (km)": _mean_sd(values),
                "RMSE reduction vs Random+Align": reduction,
            }
        )

    short_labels = ["RMSE (km)", "Stress", "Violation Rate", "Mean Angular Error (rad)", "Crossing-edge rate"]
    tables = {
        "table_6_1_rmse_reduction_vs_random": pd.DataFrame(rmse_rows),
        "table_6_1_random_vs_physics_full": pd.DataFrame(
            [_model_row(runs, "Random+Align", short_labels), _model_row(runs, "PhysicsSim-Full", short_labels)]
        ),
        "table_6_2_1_distonly_vs_distdir": pd.DataFrame(
            [
                _model_row(runs, "PhysicsSim-DistOnly", labels),
                _model_row(runs, "PhysicsSim-DistDir", labels),
                _paired_row(runs, left="PhysicsSim-DistDir", right="PhysicsSim-DistOnly", display_label="Paired difference: DistDir − DistOnly", labels=labels),
            ]
        ),
        "table_6_2_2_distdir_vs_distdiranch": pd.DataFrame(
            [
                _model_row(runs, "PhysicsSim-DistDir", labels),
                _model_row(runs, "PhysicsSim-DistDirAnch", labels),
                _paired_row(runs, left="PhysicsSim-DistDirAnch", right="PhysicsSim-DistDir", display_label="Paired difference: DistDirAnch − DistDir", labels=labels),
            ]
        ),
        "table_6_2_3_distdiranch_vs_full": pd.DataFrame(
            [
                _model_row(runs, "PhysicsSim-DistDirAnch", labels),
                _model_row(runs, "PhysicsSim-Full", labels),
                _paired_row(runs, left="PhysicsSim-Full", right="PhysicsSim-DistDirAnch", display_label="Paired difference: Full − DistDirAnch", labels=labels),
            ]
        ),
        "table_6_3_smacof_vs_distonly": pd.DataFrame(
            [_model_row(runs, "SMACOF", labels), _model_row(runs, "PhysicsSim-DistOnly", labels)]
        ),
        "table_6_3_dc_smacof_vs_distdir": pd.DataFrame(
            [_model_row(runs, "DC-SMACOF", labels), _model_row(runs, "PhysicsSim-DistDir", labels)]
        ),
        "table_6_4_overall_model_comparison": pd.DataFrame(
            [_model_row(runs, variant, labels) for variant in ("SMACOF", "DC-SMACOF", "PhysicsSim-Full")]
        ),
    }
    return tables


def _latex_cell(value: object) -> str:
    return str(value).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


def _verify_rendered_formats(stem: str, table: pd.DataFrame, table_dir: Path, failures: list[str]) -> None:
    markdown = (table_dir / f"{stem}.md").read_text(encoding="utf-8")
    latex = (table_dir / f"{stem}.tex").read_text(encoding="utf-8")
    for value in [*table.columns, *table.astype(str).to_numpy().ravel().tolist()]:
        if str(value) not in markdown:
            failures.append(f"Markdown is missing CSV value in {stem}: {value}")
    data_sections = re.findall(r"% DATA_ROWS_BEGIN(.*?)% DATA_ROWS_END", latex, flags=re.DOTALL)
    if data_sections:
        pattern = r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?"
        expected_numbers = Counter(
            token
            for value in table.astype(str).to_numpy().ravel().tolist()
            for token in re.findall(pattern, value, flags=re.IGNORECASE)
        )
        tex_data = re.sub(r"\\addlinespace\[\d+(?:\.\d+)?pt\]", "", "\n".join(data_sections))
        actual_numbers = Counter(re.findall(pattern, tex_data, flags=re.IGNORECASE))
        if actual_numbers != expected_numbers:
            failures.append(f"TeX data-row numbers differ from CSV values in {stem}")
    else:
        for value in [*table.columns, *table.astype(str).to_numpy().ravel().tolist()]:
            if _latex_cell(value) not in latex:
                failures.append(f"TeX is missing CSV value in {stem}: {value}")


def verify_manuscript_tables(*, as_outdir: Path, table_dir: Path) -> tuple[list[str], dict]:
    failures: list[str] = []
    runs_path = as_outdir / "progressive_runs_by_seed.csv"
    positions_path = as_outdir / "progressive_final_positions_y_up_sim.csv"
    config_path = as_outdir / "progressive_config.json"
    runs = pd.read_csv(runs_path, encoding="utf-8-sig")
    positions = pd.read_csv(positions_path, encoding="utf-8-sig")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    if runs.duplicated(["variant", "seed"]).any():
        failures.append("raw runs contain duplicate variant/seed rows")
    if (runs["status"] != "ok").any():
        failures.append("raw runs contain non-ok status rows")
    if not np.isfinite(runs[ALL_SUMMARY_METRICS].to_numpy(float)).all():
        failures.append("raw runs contain non-finite reported metrics")
    counts = runs.groupby("variant").size().to_dict()
    if counts != VARIANT_COUNTS:
        failures.append(f"raw run counts mismatch: {counts}")
    for variant in VARIANT_COUNTS:
        actual_seeds = set(runs.loc[runs["variant"] == variant, "seed"].astype(int))
        expected_seeds = RANDOM_SEEDS if variant == "Random+Align" else MODEL_SEEDS
        if actual_seeds != expected_seeds:
            failures.append(f"seed set mismatch for {variant}")

    run_keys = set(map(tuple, runs[["variant", "seed"]].itertuples(index=False, name=None)))
    position_keys = set(map(tuple, positions[["variant", "seed"]].drop_duplicates().itertuples(index=False, name=None)))
    if run_keys != position_keys:
        failures.append("final-position variant/seed keys do not match raw runs")
    if positions.duplicated(["variant", "seed", "label"]).any():
        failures.append("final positions contain duplicate variant/seed/label rows")
    group_sizes = positions.groupby(["variant", "seed"]).size()
    n_labels = int(positions["label"].nunique())
    if n_labels != 35 or not (group_sizes == n_labels).all() or len(positions) != len(runs) * n_labels:
        failures.append(
            f"final-position completeness mismatch: labels={n_labels}, rows={len(positions)}, expected={len(runs) * n_labels}"
        )
    if not np.isfinite(positions[["x_y_up_sim", "y_y_up_sim"]].to_numpy(float)).all():
        failures.append("final positions contain NaN or infinity")

    if int(config.get("failure_count", -1)) != 0:
        failures.append(f"config failure_count is not zero: {config.get('failure_count')}")
    expected_physics = {"alpha": 1.0, "beta": -0.5}
    for key, expected in expected_physics.items():
        if not np.isclose(float(config.get(key, np.nan)), expected, atol=1e-12, rtol=0.0):
            failures.append(f"PhysicsSim config mismatch for {key}: {config.get(key)}")
    dc = config.get("dc_smacof_hpo", {})
    expected_dc = {"alpha": -2.0, "w_weight": 1.0, "v_weight": 0.01}
    for key, expected in expected_dc.items():
        if not np.isclose(float(dc.get(key, np.nan)), expected, atol=1e-12, rtol=0.0):
            failures.append(f"DC-SMACOF config mismatch for {key}: {dc.get(key)}")
    expected_direction_method = {
        "direction_target_rule": "wang2017_current_pair_distance",
        "direction_preprocessing": "vector_consensus_by_undirected_pair",
        "raw_direction_observation_count": 44,
        "effective_direction_constraint_count": 43,
        "direction_evaluation_source": "raw_verified_observations",
    }
    direction_method = config.get("dc_smacof_direction_method", {})
    for key, expected in expected_direction_method.items():
        if direction_method.get(key) != expected:
            failures.append(f"DC-SMACOF direction-method mismatch for {key}: {direction_method.get(key)}")
    if config.get("anchor_align_label") != "鄯善":
        failures.append(f"anchor_align_label mismatch: {config.get('anchor_align_label')}")
    calibration_labels = config.get("calibration_labels", [])
    test_labels = config.get("test_labels", [])
    if calibration_labels != ["鄯善", "車師前", "都護治/烏壘"]:
        failures.append(f"calibration labels mismatch: {calibration_labels}")
    if len(test_labels) != 8 or set(test_labels).intersection(calibration_labels):
        failures.append(f"test-label partition mismatch: {test_labels}")
    expected_alignment = {
        "Random+Align": "anchor_frame+rotation_reflection_scaling",
        "PhysicsSim-DistOnly": "anchor_frame+rotation_reflection",
        "SMACOF": "anchor_frame+rotation_reflection",
        "PhysicsSim-DistDir": "anchor_frame",
        "PhysicsSim-DistDirAnch": "anchor_frame",
        "PhysicsSim-Full": "anchor_frame",
        "DC-SMACOF": "anchor_frame",
    }
    if config.get("alignment_protocol") != expected_alignment:
        failures.append(f"alignment protocol mismatch: {config.get('alignment_protocol')}")

    expected_tables = _expected_tables(runs)
    expected_files = {f"{stem}{suffix}" for stem in expected_tables for suffix in (".csv", ".md", ".tex")}
    actual_files = {path.name for path in table_dir.iterdir() if path.is_file()}
    if actual_files != expected_files:
        failures.append(f"manuscript table file-set mismatch: {sorted(actual_files ^ expected_files)}")
    for stem, expected in expected_tables.items():
        actual = pd.read_csv(table_dir / f"{stem}.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)
        expected = expected.astype(str)
        try:
            pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
        except AssertionError as exc:
            failures.append(f"CSV differs from independently recomputed raw-run values for {stem}: {exc}")
            continue
        _verify_rendered_formats(stem, actual, table_dir, failures)

    details = {
        "raw_run_count": int(len(runs)),
        "final_position_count": int(len(positions)),
        "node_count": n_labels,
        "table_count": len(expected_tables),
        "format_file_count": len(expected_files),
        "physics_parameters": expected_physics,
        "dc_smacof_parameters": expected_dc,
        "dc_smacof_direction_method": expected_direction_method,
        "calibration_labels": calibration_labels,
        "test_labels": test_labels,
        "alignment_protocol": expected_alignment,
        "source_sha256": {
            runs_path.name: _sha256(runs_path),
            positions_path.name: _sha256(positions_path),
            config_path.name: _sha256(config_path),
        },
    }
    return failures, details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-outdir", default=str(DEFAULT_AS_OUTDIR))
    parser.add_argument("--table-dir", default=str(DEFAULT_TABLE_DIR))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    args = parser.parse_args()
    failures, details = verify_manuscript_tables(
        as_outdir=Path(args.as_outdir),
        table_dir=Path(args.table_dir),
    )
    report = {"status": "ok" if not failures else "failed", "failures": failures, "details": details}
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if failures:
        print("[FAIL] Manuscript table audit failed:")
        for failure in failures:
            print(f"  - {failure}")
        print(f"[Saved] {report_path}")
        raise SystemExit(1)
    print("[OK] All manuscript table values independently match the raw experiment runs.")
    print(f"[Saved] {report_path}")


if __name__ == "__main__":
    main()
