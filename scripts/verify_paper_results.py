"""Verify that paper_results/current matches the numerical experiment outputs.

The verifier rebuilds publication tables and metric figures into a temporary
directory, then byte-compares them with the files stored in paper_results.
It also validates the snapshot manifest and rejects obsolete metric labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

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


DEFAULT_AS_OUTDIR = PROJECT_ROOT / "outputs" / "ch5_progressive_as_physics_alpha_1_beta_-0.5_dc_alpha_-0.5_100seeds_random1000"
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

TABLE_FILES = {
    "table_random_layout_summary.csv",
    "table_random_layout_summary.md",
    "table_random_align_baseline_full_statistics.csv",
    "table_random_align_baseline_full_statistics.md",
    "table_random_layout_mean_sd.csv",
    "table_random_layout_mean_sd.md",
    "table_progressive_chain_summary.csv",
    "table_progressive_chain_summary.md",
    "table_progressive_chain_mean_sd.csv",
    "table_progressive_chain_mean_sd.md",
    "table_distonly_vs_distdir_paired_comparison.csv",
    "table_distonly_vs_distdir_paired_comparison.md",
    "table_distdir_vs_distdiranch_paired_comparison.csv",
    "table_distdir_vs_distdiranch_paired_comparison.md",
    "table_distdiranch_vs_full_paired_comparison.csv",
    "table_distdiranch_vs_full_paired_comparison.md",
    "table_smacof_vs_distonly_information_matched_comparison.csv",
    "table_smacof_vs_distonly_information_matched_comparison.md",
    "table_dc_smacof_vs_distdir_information_matched_comparison.csv",
    "table_dc_smacof_vs_distdir_information_matched_comparison.md",
    "table_overall_model_comparison_comparison.csv",
    "table_overall_model_comparison_comparison.md",
    "table_smacof_dc_smacof_baselines_full_statistics.csv",
    "table_smacof_dc_smacof_baselines_full_statistics.md",
}

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

RESULT_CHAPTER_TABLE_FILES = {
    "table_6_1_random_vs_physics_full.csv",
    "table_6_1_random_vs_physics_full.md",
    "table_6_1_rmse_reduction_vs_random.csv",
    "table_6_1_rmse_reduction_vs_random.md",
    "table_6_2_1_distonly_vs_distdir.csv",
    "table_6_2_1_distonly_vs_distdir.md",
    "table_6_2_2_distdir_vs_distdiranch.csv",
    "table_6_2_2_distdir_vs_distdiranch.md",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _compare_file(expected: Path, actual: Path, failures: list[str]) -> None:
    if not actual.exists():
        failures.append(f"missing paper result: {actual}")
        return
    if _sha256(expected) != _sha256(actual):
        failures.append(f"content mismatch: {actual}")


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
        expected_dir = Path(tmp)
        _export_tables(as_outdir, expected_dir)
        expected_names = {path.name for path in expected_dir.iterdir() if path.is_file()}
        actual_names = {path.name for path in actual_dir.iterdir() if path.is_file() and path.suffix.lower() in {".csv", ".md"}}
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


def _verify_figures(paper_results: Path, failures: list[str]) -> None:
    table_dir = paper_results / "05_paper_tables"
    actual_dir = paper_results / "06_paper_figures"
    with tempfile.TemporaryDirectory(prefix="paper_figures_verify_") as tmp:
        expected_dir = Path(tmp)
        export_figures(table_dir=table_dir, outdir=expected_dir)
        expected_names = {path.name for path in expected_dir.iterdir() if path.is_file()}
        if expected_names != FIGURE_FILES:
            failures.append(f"verifier expected-figure set is stale: {sorted(expected_names.symmetric_difference(FIGURE_FILES))}")
        for name in sorted(FIGURE_FILES & expected_names):
            _compare_file(expected_dir / name, actual_dir / name, failures)


def _verify_result_chapter_tables(paper_results: Path, failures: list[str]) -> None:
    paper_table_dir = paper_results / "05_paper_tables"
    actual_dir = paper_results / "Result chapter table"
    with tempfile.TemporaryDirectory(prefix="result_chapter_tables_verify_") as tmp:
        expected_dir = Path(tmp)
        export_result_chapter_tables(paper_table_dir=paper_table_dir, outdir=expected_dir, overwrite=True)
        expected_names = {path.name for path in expected_dir.iterdir() if path.is_file()}
        if expected_names != RESULT_CHAPTER_TABLE_FILES:
            failures.append(
                "verifier expected Result chapter table set is stale: "
                f"{sorted(expected_names.symmetric_difference(RESULT_CHAPTER_TABLE_FILES))}"
            )
        if not actual_dir.exists():
            failures.append(f"missing Result chapter table directory: {actual_dir}")
            return
        actual_names = {path.name for path in actual_dir.iterdir() if path.is_file() and path.suffix.lower() in {".csv", ".md"}}
        unexpected = sorted(actual_names - RESULT_CHAPTER_TABLE_FILES)
        missing = sorted(RESULT_CHAPTER_TABLE_FILES - actual_names)
        if unexpected:
            failures.append(f"unexpected Result chapter table files: {unexpected}")
        if missing:
            failures.append(f"missing Result chapter table files: {missing}")
        for name in sorted(RESULT_CHAPTER_TABLE_FILES & expected_names):
            _compare_file(expected_dir / name, actual_dir / name, failures)


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
    scan_dirs = [paper_results / "05_paper_tables", paper_results / "06_paper_figures", paper_results / "Result chapter table"]
    suffixes = {".csv", ".md", ".svg"}
    for folder in scan_dirs:
        for path in folder.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            text = path.read_text(encoding="utf-8-sig", errors="replace")
            hits = [needle for needle in FORBIDDEN_TEXT if needle in text]
            if hits:
                failures.append(f"obsolete label(s) {hits} in {path}")


def verify_paper_results(*, as_outdir: Path, paper_results: Path) -> list[str]:
    failures: list[str] = []
    if not as_outdir.exists():
        failures.append(f"missing AS output directory: {as_outdir}")
        return failures
    if not paper_results.exists():
        failures.append(f"missing paper_results directory: {paper_results}")
        return failures
    _verify_tables(as_outdir, paper_results, failures)
    _verify_figures(paper_results, failures)
    _verify_result_chapter_tables(paper_results, failures)
    _verify_manifest(paper_results, failures)
    _scan_forbidden_labels(paper_results, failures)
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify paper_results/current against formal numerical experiment outputs.")
    parser.add_argument("--as-outdir", default=str(DEFAULT_AS_OUTDIR), help="Formal progressive AS output directory.")
    parser.add_argument("--paper-results", default=str(DEFAULT_PAPER_RESULTS), help="paper_results/current directory.")
    args = parser.parse_args()
    failures = verify_paper_results(as_outdir=Path(args.as_outdir), paper_results=Path(args.paper_results))
    if failures:
        print("[FAIL] paper_results verification failed:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("[OK] paper_results/current is consistent with the formal numerical experiment outputs.")


if __name__ == "__main__":
    main()
