"""Recompute and export the formal anchor-split RMSE ranking and SD audit."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_OUTDIR = Path(
    "outputs/ch5_anchor_split_robustness_formal_45splits_hpo3_final10_20260824"
)


def build_ranking(outdir: Path) -> Path:
    runs_path = outdir / "anchor_split_final_runs.csv"
    summary_path = outdir / "anchor_split_summary.csv"
    output_path = outdir / "anchor_split_rmse_ranking_and_sd.csv"

    runs = pd.read_csv(runs_path, encoding="utf-8-sig")
    summary = pd.read_csv(summary_path, encoding="utf-8-sig")

    required_run_columns = {"split_id", "seed", "RMSE_final_test_km"}
    required_summary_columns = {
        "split_id",
        "is_original_split",
        "southern_route_anchor",
        "northern_route_anchor",
        "north_of_mountains_anchor",
        "final_frame_anchor",
        "test_labels",
        "selected_alpha",
        "selected_beta",
        "n_seeds",
        "RMSE_final_test_mean_km",
        "RMSE_final_test_std_km",
    }
    if missing := required_run_columns.difference(runs.columns):
        raise ValueError(f"Run-level input is missing columns: {sorted(missing)}")
    if missing := required_summary_columns.difference(summary.columns):
        raise ValueError(f"Summary input is missing columns: {sorted(missing)}")
    if runs.duplicated(["split_id", "seed"]).any():
        raise ValueError("Run-level input contains duplicate split/seed pairs.")

    recomputed = (
        runs.groupby("split_id", as_index=False)["RMSE_final_test_km"]
        .agg(
            recomputed_n_seeds="count",
            recomputed_mean_rmse_km="mean",
            recomputed_sample_sd_km=lambda values: values.std(ddof=1),
        )
    )
    if len(recomputed) != 45 or not (recomputed["recomputed_n_seeds"] == 10).all():
        raise ValueError("Formal ranking requires 45 splits with 10 seeds per split.")

    audited = summary.merge(recomputed, on="split_id", how="inner", validate="one_to_one")
    if len(audited) != 45:
        raise ValueError("Run-level and summary files do not contain the same 45 splits.")
    if not np.array_equal(audited["n_seeds"].to_numpy(int), audited["recomputed_n_seeds"].to_numpy(int)):
        raise ValueError("Saved and recomputed seed counts differ.")
    if not np.allclose(
        audited["RMSE_final_test_mean_km"], audited["recomputed_mean_rmse_km"], atol=1e-12, rtol=1e-12
    ):
        raise ValueError("Saved and recomputed split means differ.")
    if not np.allclose(
        audited["RMSE_final_test_std_km"], audited["recomputed_sample_sd_km"], atol=1e-12, rtol=1e-12
    ):
        raise ValueError("Saved and recomputed sample SDs differ.")

    ranked = audited.sort_values(["recomputed_mean_rmse_km", "split_id"]).reset_index(drop=True)
    n_splits = len(ranked)
    ranked.insert(0, "rank_low_to_high_mean_rmse", np.arange(1, n_splits + 1))
    ranked.insert(1, "empirical_rank_percent_rank_over_n", ranked["rank_low_to_high_mean_rmse"] / n_splits * 100.0)
    ranked.insert(2, "n_splits_with_lower_mean_rmse", ranked["rank_low_to_high_mean_rmse"] - 1)
    ranked.insert(3, "percent_splits_with_lower_mean_rmse", ranked["n_splits_with_lower_mean_rmse"] / n_splits * 100.0)

    minimum = float(ranked["recomputed_mean_rmse_km"].min())
    maximum = float(ranked["recomputed_mean_rmse_km"].max())
    ranked["all_splits_min_mean_rmse_km"] = minimum
    ranked["all_splits_max_mean_rmse_km"] = maximum
    ranked["all_splits_mean_rmse_range_width_km"] = maximum - minimum
    ranked["average_within_split_sample_sd_km"] = float(ranked["recomputed_sample_sd_km"].mean())
    ranked["sd_definition"] = "sample SD across the 10 final seeds within this split (ddof=1)"
    ranked["average_sd_definition"] = "arithmetic mean of the 45 within-split sample SD values"
    ranked["difference_definition"] = "ranking uses split-level mean held-out test RMSE; lower rank is lower RMSE"
    ranked["run_level_source"] = runs_path.as_posix()
    ranked["summary_source"] = summary_path.as_posix()

    columns = [
        "rank_low_to_high_mean_rmse",
        "empirical_rank_percent_rank_over_n",
        "n_splits_with_lower_mean_rmse",
        "percent_splits_with_lower_mean_rmse",
        "split_id",
        "is_original_split",
        "southern_route_anchor",
        "northern_route_anchor",
        "north_of_mountains_anchor",
        "final_frame_anchor",
        "test_labels",
        "selected_alpha",
        "selected_beta",
        "recomputed_n_seeds",
        "recomputed_mean_rmse_km",
        "recomputed_sample_sd_km",
        "all_splits_min_mean_rmse_km",
        "all_splits_max_mean_rmse_km",
        "all_splits_mean_rmse_range_width_km",
        "average_within_split_sample_sd_km",
        "sd_definition",
        "average_sd_definition",
        "difference_definition",
        "run_level_source",
        "summary_source",
    ]
    ranked[columns].to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()
    output = build_ranking(args.outdir)
    print(f"[OK] Saved verified anchor-split ranking: {output}")


if __name__ == "__main__":
    main()
