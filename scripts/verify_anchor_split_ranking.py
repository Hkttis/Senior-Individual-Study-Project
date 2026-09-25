"""Independently verify the exported formal anchor-split ranking and SD audit."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_OUTDIR = Path(
    "outputs/ch5_anchor_split_robustness_formal_45splits_hpo3_final10_20260824"
)


def verify(outdir: Path) -> dict[str, float | int | str]:
    runs = pd.read_csv(outdir / "anchor_split_final_runs.csv", encoding="utf-8-sig")
    summary = pd.read_csv(outdir / "anchor_split_summary.csv", encoding="utf-8-sig")
    ranking = pd.read_csv(outdir / "anchor_split_rmse_ranking_and_sd.csv", encoding="utf-8-sig")

    if len(ranking) != 45 or ranking["split_id"].nunique() != 45:
        raise ValueError("Ranking file must contain exactly 45 unique splits.")
    if ranking["rank_low_to_high_mean_rmse"].tolist() != list(range(1, 46)):
        raise ValueError("Ranking column is not the exact sequence 1..45.")

    grouped = runs.groupby("split_id")["RMSE_final_test_km"]
    expected = pd.DataFrame(
        {
            "split_id": grouped.size().index,
            "expected_n": grouped.size().to_numpy(int),
            "expected_mean": grouped.mean().to_numpy(float),
            "expected_sd": grouped.std(ddof=1).to_numpy(float),
        }
    ).sort_values(["expected_mean", "split_id"]).reset_index(drop=True)
    expected["expected_rank"] = np.arange(1, len(expected) + 1)

    if not ranking["split_id"].equals(expected["split_id"]):
        raise ValueError("Split ordering differs from ascending recomputed mean RMSE.")
    checks = {
        "seed counts": np.array_equal(ranking["recomputed_n_seeds"].to_numpy(int), expected["expected_n"]),
        "mean RMSE": np.allclose(ranking["recomputed_mean_rmse_km"], expected["expected_mean"], atol=1e-12, rtol=1e-12),
        "sample SD": np.allclose(ranking["recomputed_sample_sd_km"], expected["expected_sd"], atol=1e-12, rtol=1e-12),
        "rank": np.array_equal(ranking["rank_low_to_high_mean_rmse"].to_numpy(int), expected["expected_rank"]),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Ranking verification failed for: {failed}")

    summary_indexed = summary.set_index("split_id")
    ordered_summary = summary_indexed.loc[ranking["split_id"]]
    if not np.allclose(ordered_summary["RMSE_final_test_mean_km"], expected["expected_mean"], atol=1e-12, rtol=1e-12):
        raise ValueError("Saved summary means differ from independently recomputed run-level means.")
    if not np.allclose(ordered_summary["RMSE_final_test_std_km"], expected["expected_sd"], atol=1e-12, rtol=1e-12):
        raise ValueError("Saved summary SDs differ from independently recomputed run-level sample SDs.")

    minimum = float(expected["expected_mean"].min())
    maximum = float(expected["expected_mean"].max())
    range_width = maximum - minimum
    average_sd = float(expected["expected_sd"].mean())
    repeated_values = {
        "all_splits_min_mean_rmse_km": minimum,
        "all_splits_max_mean_rmse_km": maximum,
        "all_splits_mean_rmse_range_width_km": range_width,
        "average_within_split_sample_sd_km": average_sd,
    }
    for column, expected_value in repeated_values.items():
        if not np.allclose(ranking[column], expected_value, atol=1e-12, rtol=1e-12):
            raise ValueError(f"Repeated global statistic is incorrect: {column}")

    original = ranking.loc[ranking["is_original_split"].astype(bool)]
    if len(original) != 1:
        raise ValueError("Ranking must identify exactly one original split.")
    return {
        "n_splits": 45,
        "seeds_per_split": 10,
        "minimum_mean_rmse_km": minimum,
        "maximum_mean_rmse_km": maximum,
        "range_width_km": range_width,
        "average_within_split_sample_sd_km": average_sd,
        "original_split_id": str(original.iloc[0]["split_id"]),
        "original_split_rank": int(original.iloc[0]["rank_low_to_high_mean_rmse"]),
        "original_split_empirical_rank_percent": float(original.iloc[0]["empirical_rank_percent_rank_over_n"]),
        "original_split_n_lower": int(original.iloc[0]["n_splits_with_lower_mean_rmse"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()
    report = verify(args.outdir)
    print("[OK] Anchor-split ranking and SD audit matches run-level and summary data")
    for key, value in report.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
