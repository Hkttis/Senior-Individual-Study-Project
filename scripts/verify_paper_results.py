"""Verify that paper_results/current matches the numerical experiment outputs.

The verifier rebuilds publication tables and metric figures into a temporary
directory, then byte-compares them with the files stored in paper_results.
It also validates the snapshot manifest and rejects obsolete metric labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.export_paper_figures import export_figures
from scripts.export_paper_tables import (
    export_dc_smacof_vs_distdir_comparison,
    export_distdir_vs_distdiranch_comparison,
    export_distdiranch_vs_full_comparison,
    export_distonly_vs_distdir_comparison,
    export_overall_model_comparison,
    export_progressive_chain_summary,
    export_random_layout_summary,
    export_smacof_dc_smacof_baseline_full_statistics,
    export_smacof_vs_distonly_comparison,
)
from scripts.export_result_chapter_tables import export_result_chapter_tables
from scripts.update_paper_results import (
    AS_MAIN_FILES,
    AS_SUPPLEMENTARY_FILES,
    DEFAULT_AS_OUTDIR,
    DEFAULT_DC_HPO_OUTDIR,
)
from run_paper_script.ch5_ablation_progressive import _paired, _random_percentiles, _summary


DEFAULT_PAPER_RESULTS = PROJECT_ROOT / "paper_results" / "current"

FORBIDDEN_TEXT = [
    "Mean [95% CI]",
    "mean_ci",
    "Direction MAE",
    "Direction violation rate",
    "Direction VR",
    "Distance stress",
    "Test RMSE",
    "Test MAE",
    "Median test error",
    "CNR@0.10",
    "CVR@0.10",
    "NND-q05",
    "Distance-edge crossing",
    "Crossing-Edge Rate",
    "MAE (km)",
]

TABLE_STEMS = {
    "table_6_1_rmse_reduction_vs_random",
    "table_6_1_random_vs_physics_full",
    "table_6_2_1_distonly_vs_distdir",
    "table_6_2_2_distdir_vs_distdiranch",
    "table_6_2_3_distdiranch_vs_full",
    "table_6_3_smacof_vs_distonly",
    "table_6_3_dc_smacof_vs_distdir",
    "table_6_4_overall_model_comparison",
}
TABLE_FILES = {f"{stem}{suffix}" for stem in TABLE_STEMS for suffix in (".csv", ".md", ".tex")}

FIGURE_FILES = {
    "random_align_core_metrics_table.svg",
    "random_align_core_metrics_table_4800w.png",
    "progressive_chain_core_metrics_table.svg",
    "progressive_chain_core_metrics_table_4800w.png",
    "paired_comparison_direction_core_metrics.svg",
    "paired_comparison_direction_core_metrics_4800w.png",
    "paired_comparison_anchor_core_metrics.svg",
    "paired_comparison_anchor_core_metrics_4800w.png",
    "paired_comparison_rep_accuracy_metrics.svg",
    "paired_comparison_rep_accuracy_metrics_4800w.png",
    "paired_comparison_rep_layout_metrics.svg",
    "paired_comparison_rep_layout_metrics_4800w.png",
    "information_matched_overall_comparison_core_metrics.svg",
    "information_matched_overall_comparison_core_metrics_4800w.png",
    "information_matched_overall_comparison_core_metrics_dc.svg",
    "information_matched_overall_comparison_core_metrics_dc_4800w.png",
    "overall_model_comparison_core_metrics.svg",
    "overall_model_comparison_core_metrics_4800w.png",
}

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _compare_file(expected: Path, actual: Path, failures: list[str]) -> None:
    if not actual.exists():
        failures.append(f"missing paper result: {actual}")
        return
    if _sha256(expected) != _sha256(actual):
        failures.append(f"content mismatch: {actual}")


def _compare_frames(
    expected: pd.DataFrame,
    actual: pd.DataFrame,
    *,
    sort_by: list[str],
    label: str,
    failures: list[str],
) -> None:
    try:
        expected_sorted = expected.sort_values(sort_by).reset_index(drop=True)
        actual_sorted = actual.sort_values(sort_by).reset_index(drop=True)
        pd.testing.assert_frame_equal(
            expected_sorted,
            actual_sorted,
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except (AssertionError, KeyError, ValueError) as exc:
        failures.append(f"numerical mismatch in {label}: {exc}")


def _verify_numerical_experiment(as_outdir: Path, failures: list[str]) -> None:
    required = [
        "progressive_runs_by_seed.csv",
        "progressive_final_positions_y_up_sim.csv",
        "progressive_summary.csv",
        "progressive_paired_comparisons.csv",
        "random_align_percentiles.csv",
        "random_align_runs.csv",
        "random_align_summary.csv",
        "progressive_run_status.csv",
        "progressive_config.json",
    ]
    missing = [name for name in required if not (as_outdir / name).exists()]
    if missing:
        failures.append(f"formal AS is missing numerical files: {missing}")
        return

    runs = pd.read_csv(as_outdir / "progressive_runs_by_seed.csv")
    positions = pd.read_csv(as_outdir / "progressive_final_positions_y_up_sim.csv")
    config = json.loads((as_outdir / "progressive_config.json").read_text(encoding="utf-8"))
    core_metrics = ["E_distance_stress", "E_direction_vr", "E_direction_mae", "RMSE_test_km"]
    if (runs["status"] != "ok").any():
        failures.append("formal AS runs contain non-ok status rows")
    if not np.isfinite(runs[core_metrics].to_numpy(float)).all():
        failures.append("formal AS core metrics contain NaN or infinity")
    if not np.isfinite(positions[["x_y_up_sim", "y_y_up_sim"]].to_numpy(float)).all():
        failures.append("formal AS final positions contain NaN or infinity")
    if int(config.get("failure_count", -1)) != 0:
        failures.append(f"formal AS config failure_count is not zero: {config.get('failure_count')}")
    dc_params = config.get("dc_smacof_hpo", {})
    for key, expected in {"alpha": -2.0, "w_weight": 1.0, "v_weight": 0.01}.items():
        if not np.isclose(float(dc_params.get(key, np.nan)), expected, rtol=0.0, atol=1e-12):
            failures.append(f"formal AS DC-SMACOF {key} mismatch: {dc_params.get(key)}")

    actual_counts = runs[runs["status"] == "ok"].groupby("variant").size().to_dict()
    expected_counts = {
        "PhysicsSim-DistOnly": 100,
        "PhysicsSim-DistDir": 100,
        "PhysicsSim-DistDirAnch": 100,
        "PhysicsSim-Full": 100,
        "SMACOF": 100,
        "DC-SMACOF": 100,
        "Random+Align": 1000,
    }
    if actual_counts != expected_counts:
        failures.append(f"formal AS run-count mismatch: {actual_counts}")

    expected_summary = _summary(runs)
    expected_paired = _paired(runs)
    expected_percentiles = _random_percentiles(runs)
    expected_random_runs = runs[runs["variant"] == "Random+Align"].copy()
    expected_random_summary = expected_summary[expected_summary["variant"] == "Random+Align"].copy()
    expected_status = runs.groupby(["variant", "status"], dropna=False).size().reset_index(name="n_runs")

    _compare_frames(
        expected_summary,
        pd.read_csv(as_outdir / "progressive_summary.csv"),
        sort_by=["variant", "metric"],
        label="progressive_summary.csv recomputed from runs",
        failures=failures,
    )
    _compare_frames(
        expected_paired,
        pd.read_csv(as_outdir / "progressive_paired_comparisons.csv"),
        sort_by=["comparison", "metric"],
        label="progressive_paired_comparisons.csv recomputed from runs",
        failures=failures,
    )
    _compare_frames(
        expected_percentiles,
        pd.read_csv(as_outdir / "random_align_percentiles.csv"),
        sort_by=["variant", "metric"],
        label="random_align_percentiles.csv recomputed from runs",
        failures=failures,
    )
    _compare_frames(
        expected_random_runs,
        pd.read_csv(as_outdir / "random_align_runs.csv"),
        sort_by=["seed"],
        label="random_align_runs.csv filtered from runs",
        failures=failures,
    )
    _compare_frames(
        expected_random_summary,
        pd.read_csv(as_outdir / "random_align_summary.csv"),
        sort_by=["metric"],
        label="random_align_summary.csv recomputed from runs",
        failures=failures,
    )
    _compare_frames(
        expected_status,
        pd.read_csv(as_outdir / "progressive_run_status.csv"),
        sort_by=["variant", "status"],
        label="progressive_run_status.csv recomputed from runs",
        failures=failures,
    )


def _verify_snapshot_copies(
    *,
    as_outdir: Path,
    dc_hpo_outdir: Path,
    paper_results: Path,
    failures: list[str],
) -> None:
    main_dir = paper_results / "03_progressive_as_main"
    supplementary_dir = paper_results / "04_progressive_as_supplementary"
    dc_dir = paper_results / "02_dc_smacof_hpo"
    expected_main_names = set(AS_MAIN_FILES) | {"summary_statistics_metadata.json"}
    actual_main_names = {path.name for path in main_dir.iterdir() if path.is_file()}
    if actual_main_names != expected_main_names:
        failures.append(f"03_progressive_as_main file-set mismatch: {sorted(actual_main_names ^ expected_main_names)}")
    actual_supplementary_names = {path.name for path in supplementary_dir.iterdir() if path.is_file()}
    if actual_supplementary_names != set(AS_SUPPLEMENTARY_FILES):
        failures.append(
            "04_progressive_as_supplementary file-set mismatch: "
            f"{sorted(actual_supplementary_names ^ set(AS_SUPPLEMENTARY_FILES))}"
        )
    source_dc_names = {path.name for path in dc_hpo_outdir.iterdir() if path.is_file()}
    actual_dc_names = {path.name for path in dc_dir.iterdir() if path.is_file()}
    if actual_dc_names != source_dc_names:
        failures.append(f"02_dc_smacof_hpo file-set mismatch: {sorted(actual_dc_names ^ source_dc_names)}")
    for name in AS_MAIN_FILES:
        _compare_file(as_outdir / name, main_dir / name, failures)
    for name in AS_SUPPLEMENTARY_FILES:
        _compare_file(as_outdir / name, supplementary_dir / name, failures)
    for name in sorted(source_dc_names):
        _compare_file(dc_hpo_outdir / name, dc_dir / name, failures)


def _export_tables(as_outdir: Path, outdir: Path) -> None:
    export_random_layout_summary(as_outdir=as_outdir, outdir=outdir, overwrite=True)
    export_progressive_chain_summary(as_outdir=as_outdir, outdir=outdir, overwrite=True)
    export_distonly_vs_distdir_comparison(as_outdir=as_outdir, outdir=outdir, overwrite=True)
    export_distdir_vs_distdiranch_comparison(as_outdir=as_outdir, outdir=outdir, overwrite=True)
    export_distdiranch_vs_full_comparison(as_outdir=as_outdir, outdir=outdir, overwrite=True)
    export_smacof_vs_distonly_comparison(as_outdir=as_outdir, outdir=outdir, overwrite=True)
    export_dc_smacof_vs_distdir_comparison(as_outdir=as_outdir, outdir=outdir, overwrite=True)
    export_overall_model_comparison(as_outdir=as_outdir, outdir=outdir, overwrite=True)
    export_smacof_dc_smacof_baseline_full_statistics(as_outdir=as_outdir, outdir=outdir, overwrite=True)


def _verify_tables(as_outdir: Path, paper_results: Path, failures: list[str]) -> None:
    actual_dir = paper_results / "05_paper_tables"
    with tempfile.TemporaryDirectory(prefix="paper_tables_verify_") as tmp:
        supporting_dir = Path(tmp) / "supporting"
        expected_dir = Path(tmp) / "curated"
        supporting_dir.mkdir()
        _export_tables(as_outdir, supporting_dir)
        export_result_chapter_tables(
            paper_table_dir=supporting_dir,
            outdir=expected_dir,
            overwrite=True,
        )
        expected_names = {path.name for path in expected_dir.iterdir() if path.is_file()}
        actual_names = {
            path.name
            for path in actual_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".csv", ".md", ".tex"}
        }
        if expected_names != TABLE_FILES:
            failures.append(f"verifier expected-table set is stale: {sorted(expected_names.symmetric_difference(TABLE_FILES))}")
        unexpected = sorted(actual_names - TABLE_FILES)
        missing = sorted(TABLE_FILES - actual_names)
        if unexpected:
            failures.append(f"unexpected paper table files: {unexpected}")
        if missing:
            failures.append(f"missing paper table files: {missing}")
        for name in sorted(TABLE_FILES & expected_names):
            _compare_file(expected_dir / name, actual_dir / name, failures)


def _verify_figures(as_outdir: Path, paper_results: Path, failures: list[str]) -> None:
    actual_dir = paper_results / "06_paper_figures"
    with tempfile.TemporaryDirectory(prefix="paper_figures_verify_") as tmp:
        supporting_dir = Path(tmp) / "supporting"
        expected_dir = Path(tmp) / "figures"
        supporting_dir.mkdir()
        _export_tables(as_outdir, supporting_dir)
        export_figures(table_dir=supporting_dir, outdir=expected_dir)
        expected_names = {path.name for path in expected_dir.iterdir() if path.is_file()}
        if expected_names != FIGURE_FILES:
            failures.append(f"verifier expected-figure set is stale: {sorted(expected_names.symmetric_difference(FIGURE_FILES))}")
        for name in sorted(FIGURE_FILES & expected_names):
            _compare_file(expected_dir / name, actual_dir / name, failures)


def _verify_result_chapter_table_removed(paper_results: Path, failures: list[str]) -> None:
    legacy_dir = paper_results / "Result chapter table"
    if legacy_dir.exists():
        failures.append(f"legacy Result chapter table directory should be removed: {legacy_dir}")


def _verify_manifest(paper_results: Path, failures: list[str]) -> None:
    manifest = paper_results / "manifest_sha256.csv"
    if not manifest.exists():
        failures.append(f"missing manifest: {manifest}")
        return
    with manifest.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    manifest_paths = {row["snapshot_path"].replace("\\", "/") for row in rows}
    actual_paths = {
        path.relative_to(paper_results).as_posix()
        for path in paper_results.rglob("*")
        if path.is_file() and path != manifest
    }
    missing = sorted(actual_paths - manifest_paths)
    stale = sorted(manifest_paths - actual_paths)
    if missing:
        failures.append(f"files missing from manifest: {missing[:20]}")
    if stale:
        failures.append(f"manifest entries without files: {stale[:20]}")
    for row in rows:
        path = paper_results / row["snapshot_path"]
        if not path.exists():
            continue
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest().upper()
        if digest != row["sha256"] or str(len(data)) != row["size_bytes"]:
            failures.append(f"manifest hash/size mismatch: {row['snapshot_path']}")


def _scan_forbidden_labels(paper_results: Path, failures: list[str]) -> None:
    scan_dirs = [paper_results / "05_paper_tables", paper_results / "06_paper_figures"]
    suffixes = {".csv", ".md", ".svg", ".tex"}
    for folder in scan_dirs:
        for path in folder.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            text = path.read_text(encoding="utf-8-sig", errors="replace")
            hits = [needle for needle in FORBIDDEN_TEXT if needle in text]
            if hits:
                failures.append(f"obsolete label(s) {hits} in {path}")


def verify_paper_results(*, as_outdir: Path, dc_hpo_outdir: Path, paper_results: Path) -> list[str]:
    failures: list[str] = []
    if not as_outdir.exists():
        failures.append(f"missing AS output directory: {as_outdir}")
        return failures
    if not paper_results.exists():
        failures.append(f"missing paper_results directory: {paper_results}")
        return failures
    if not dc_hpo_outdir.exists():
        failures.append(f"missing DC-SMACOF HPO output directory: {dc_hpo_outdir}")
        return failures
    _verify_numerical_experiment(as_outdir, failures)
    _verify_snapshot_copies(
        as_outdir=as_outdir,
        dc_hpo_outdir=dc_hpo_outdir,
        paper_results=paper_results,
        failures=failures,
    )
    _verify_tables(as_outdir, paper_results, failures)
    _verify_figures(as_outdir, paper_results, failures)
    _verify_result_chapter_table_removed(paper_results, failures)
    _verify_manifest(paper_results, failures)
    _scan_forbidden_labels(paper_results, failures)
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify paper_results/current against formal numerical experiment outputs.")
    parser.add_argument("--as-outdir", default=str(DEFAULT_AS_OUTDIR), help="Formal progressive AS output directory.")
    parser.add_argument(
        "--dc-hpo-outdir",
        default=str(DEFAULT_DC_HPO_OUTDIR),
        help="Selected DC-SMACOF HPO output directory.",
    )
    parser.add_argument("--paper-results", default=str(DEFAULT_PAPER_RESULTS), help="paper_results/current directory.")
    args = parser.parse_args()
    failures = verify_paper_results(
        as_outdir=Path(args.as_outdir),
        dc_hpo_outdir=Path(args.dc_hpo_outdir),
        paper_results=Path(args.paper_results),
    )
    if failures:
        print("[FAIL] paper_results verification failed:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("[OK] paper_results/current is consistent with the formal numerical experiment outputs.")


if __name__ == "__main__":
    main()
