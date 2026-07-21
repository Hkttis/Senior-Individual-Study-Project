"""Rebuild progressive AS aggregate summaries from existing per-seed results.

This performs no model simulation. It exists to correct descriptive-statistic
conventions or regenerate summaries from ``progressive_runs_by_seed.csv``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from run_paper_script.ch5_ablation_progressive import _summary


def recompute_progressive_statistics(*, as_outdir: str | Path, backup_dir: str | Path) -> dict[str, Path]:
    source_dir = Path(as_outdir)
    backup_dir = Path(backup_dir)
    runs_path = source_dir / "progressive_runs_by_seed.csv"
    summary_path = source_dir / "progressive_summary.csv"
    random_summary_path = source_dir / "random_align_summary.csv"
    if not runs_path.exists() or not summary_path.exists() or not random_summary_path.exists():
        raise FileNotFoundError("Expected progressive runs, summary, and Random+Align summary CSV files.")
    if backup_dir.exists() and any(backup_dir.iterdir()):
        raise FileExistsError(f"Backup directory is not empty: {backup_dir}")
    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(summary_path, backup_dir / summary_path.name)
    shutil.copy2(random_summary_path, backup_dir / random_summary_path.name)

    runs = pd.read_csv(runs_path)
    summary = _summary(runs)
    random_summary = summary[summary["variant"] == "Random+Align"].copy()
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    random_summary.to_csv(random_summary_path, index=False, encoding="utf-8-sig")
    metadata_path = source_dir / "summary_statistics_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "source_runs": runs_path.name,
                "summary_files": [summary_path.name, random_summary_path.name],
                "std_definition": "sample standard deviation (ddof=1; n=1 uses 0.0)",
                "se_definition": "sample SD / sqrt(n)",
                "ci95_definition": "2,000-resample percentile bootstrap confidence interval for the mean",
                "model_simulation_rerun": False,
                "original_summary_backup": str(backup_dir),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"summary": summary_path, "random_summary": random_summary_path, "metadata": metadata_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild progressive AS summaries without rerunning models.")
    parser.add_argument("--as-outdir", required=True)
    parser.add_argument("--backup-dir", required=True)
    args = parser.parse_args()
    paths = recompute_progressive_statistics(as_outdir=args.as_outdir, backup_dir=args.backup_dir)
    for label, path in paths.items():
        print(f"[Saved] {label}: {path}")


if __name__ == "__main__":
    main()
