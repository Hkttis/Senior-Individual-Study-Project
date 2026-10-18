"""Export paired percentile-bootstrap CIs from formal polishing run records.

The comparison is BFGS-polished endpoint minus its PhysicsSim-Full starting
endpoint for the same seed. Held-out RMSE is used only for posthoc reporting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_paper_script.ch5_ablation_study import _bootstrap_ci_mean


METRICS = (
    ("objective_total", "objective value", "objective units"),
    ("RMSE_test_km_posthoc", "held-out test RMSE", "km"),
)
N_BOOT = 2_000
BOOTSTRAP_SEED = 0


def calculate_paired_ci(runs: pd.DataFrame, configured_seeds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"seed"}
    for key, _label, _unit in METRICS:
        required.update((f"before_{key}", f"after_{key}", f"delta_{key}"))
    missing = required.difference(runs.columns)
    if missing:
        raise ValueError(f"Missing polishing columns: {sorted(missing)}")
    if runs["seed"].isna().any() or runs["seed"].duplicated().any():
        raise ValueError("Each polishing seed must appear exactly once.")
    if len(configured_seeds) != 100 or len(set(configured_seeds)) != 100:
        raise ValueError("Formal polishing requires 100 distinct configured seeds.")
    if set(runs["seed"].tolist()) != set(configured_seeds):
        raise ValueError("Polishing rows do not match configured seeds.")

    paired = runs[["seed"]].copy().sort_values("seed").reset_index(drop=True)
    ordered = runs.set_index("seed").loc[paired["seed"]]
    summary = []
    for key, label, unit in METRICS:
        before = ordered[f"before_{key}"].to_numpy(dtype=np.float64)
        after = ordered[f"after_{key}"].to_numpy(dtype=np.float64)
        recorded_delta = ordered[f"delta_{key}"].to_numpy(dtype=np.float64)
        if not np.isfinite(np.column_stack((before, after, recorded_delta))).all():
            raise ValueError(f"Non-finite polishing values for {key}.")
        delta = after - before
        if not np.allclose(delta, recorded_delta, atol=1e-7, rtol=1e-12):
            raise ValueError(f"Recorded delta does not equal after minus before for {key}.")
        paired[f"before_{key}"] = before
        paired[f"after_{key}"] = after
        paired[f"delta_{key}"] = delta
        lo, hi = _bootstrap_ci_mean(delta)
        summary.append(
            {
                "metric": label,
                "source_column": key,
                "unit": unit,
                "difference_definition": "after_minus_before",
                "n_pairs": len(delta),
                "before_mean": float(before.mean()),
                "after_mean": float(after.mean()),
                "paired_mean_difference": float(delta.mean()),
                "paired_difference_sample_sd": float(delta.std(ddof=1)),
                "paired_mean_ci95_lower": lo,
                "paired_mean_ci95_upper": hi,
                "ci_method": "paired percentile bootstrap of seed-level differences",
                "bootstrap_resamples": N_BOOT,
                "bootstrap_seed": BOOTSTRAP_SEED,
            }
        )
    return paired, pd.DataFrame(summary)


def export(outdir: str | Path) -> tuple[Path, Path]:
    outdir = Path(outdir)
    config = json.loads((outdir / "polishing_config.json").read_text(encoding="utf-8"))
    if config.get("method_version") != "source_weight_locked_fixed_anchor_bfgs_polishing_v2":
        raise ValueError("Expected formal source-weight-locked polishing results, not another BFGS experiment.")
    runs = pd.read_csv(outdir / "polishing_runs.csv")
    paired, summary = calculate_paired_ci(runs, config["seeds"])
    paired_path = outdir / "polishing_paired_seed_differences.csv"
    summary_path = outdir / "polishing_paired_95ci.csv"
    paired.to_csv(paired_path, index=False, encoding="utf-8-sig", float_format="%.17g")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig", float_format="%.17g")
    return paired_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True, help="Formal polishing output directory")
    args = parser.parse_args()
    for path in export(args.outdir):
        print(f"[Saved] {path}")


if __name__ == "__main__":
    main()
