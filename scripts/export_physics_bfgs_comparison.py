"""Export a checked PhysicsSim-Full versus HPO-selected SciPy-BFGS table."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from run_paper_script.ch5_ablation_study import _bootstrap_ci_mean
from scripts.export_result_chapter_tables import _latex_longtable


VARIANT_PHYSICS = "PhysicsSim-Full"
VARIANT_BFGS = "SciPy-BFGS"
METRICS = (
    ("RMSE_test_km", "RMSE (km)"),
    ("E_distance_stress", "Stress"),
    ("E_direction_vr", "Violation Rate"),
    ("E_direction_mae", "Mean Angular Error (rad)"),
)
OUTPUT_STEM = "table_bfgs_vs_physics_full"


def _format_mean_sd(values: np.ndarray) -> str:
    return f"{float(np.mean(values)):.6g} ± {float(np.std(values, ddof=1)):.6g}"


def _format_mean_ci(values: np.ndarray) -> str:
    lo, hi = _bootstrap_ci_mean(values, n_boot=10_000, seed=0)
    return f"{float(np.mean(values)):.6g} [{lo:.6g}, {hi:.6g}]"


def build_physics_bfgs_comparison(
    *, as_runs: pd.DataFrame, bfgs_runs: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the display table and its fully numeric audit table."""

    metric_names = [metric for metric, _label in METRICS]
    required_as = {"variant", "seed", "status", *metric_names}
    required_bfgs = {"seed", "status", *metric_names}
    missing_as = required_as.difference(as_runs.columns)
    missing_bfgs = required_bfgs.difference(bfgs_runs.columns)
    if missing_as or missing_bfgs:
        raise ValueError(
            f"Missing columns: AS={sorted(missing_as)}, BFGS={sorted(missing_bfgs)}"
        )
    physics = as_runs.loc[
        (as_runs["variant"] == VARIANT_PHYSICS) & (as_runs["status"] == "ok"),
        ["seed", *metric_names],
    ].copy()
    bfgs = bfgs_runs.loc[
        bfgs_runs["status"] == "ok", ["seed", *metric_names]
    ].copy()
    paired = physics.merge(bfgs, on="seed", suffixes=("_physics", "_bfgs"), validate="one_to_one")
    if len(physics) != 100 or len(bfgs) != 100 or len(paired) != 100:
        raise ValueError(
            "Formal comparison requires exactly 100 successful runs from each model with matching seeds."
        )
    if set(physics["seed"]) != set(bfgs["seed"]):
        raise ValueError("PhysicsSim and BFGS seed sets differ; paired comparison is invalid.")

    display_rows = [
        {"Model": VARIANT_PHYSICS},
        {"Model": VARIANT_BFGS},
        {"Model": "Paired difference: BFGS − PhysicsSim-Full"},
    ]
    audit_rows: list[dict] = []
    for metric, label in METRICS:
        physics_values = paired[f"{metric}_physics"].to_numpy(float)
        bfgs_values = paired[f"{metric}_bfgs"].to_numpy(float)
        difference = bfgs_values - physics_values
        if not np.isfinite(np.concatenate([physics_values, bfgs_values])).all():
            raise ValueError(f"Non-finite values found for {metric}.")
        display_rows[0][label] = _format_mean_sd(physics_values)
        display_rows[1][label] = _format_mean_sd(bfgs_values)
        display_rows[2][label] = _format_mean_ci(difference)
        lo, hi = _bootstrap_ci_mean(difference, n_boot=10_000, seed=0)
        audit_rows.extend(
            [
                {
                    "row": VARIANT_PHYSICS,
                    "metric": metric,
                    "n": len(physics_values),
                    "mean": float(np.mean(physics_values)),
                    "sample_sd": float(np.std(physics_values, ddof=1)),
                    "ci95_lo": np.nan,
                    "ci95_hi": np.nan,
                },
                {
                    "row": VARIANT_BFGS,
                    "metric": metric,
                    "n": len(bfgs_values),
                    "mean": float(np.mean(bfgs_values)),
                    "sample_sd": float(np.std(bfgs_values, ddof=1)),
                    "ci95_lo": np.nan,
                    "ci95_hi": np.nan,
                },
                {
                    "row": "BFGS − PhysicsSim-Full",
                    "metric": metric,
                    "n": len(difference),
                    "mean": float(np.mean(difference)),
                    "sample_sd": float(np.std(difference, ddof=1)),
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                },
            ]
        )
    return pd.DataFrame(display_rows), pd.DataFrame(audit_rows)


def _markdown_table(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join(frame.columns) + " |"
    separator = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def export_physics_bfgs_comparison(
    *,
    as_outdir: str | Path,
    bfgs_outdir: str | Path,
    outdir: str | Path,
    overwrite: bool = False,
) -> dict[str, Path]:
    as_outdir = Path(as_outdir)
    bfgs_outdir = Path(bfgs_outdir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {
        "csv": outdir / f"{OUTPUT_STEM}.csv",
        "markdown": outdir / f"{OUTPUT_STEM}.md",
        "latex": outdir / f"{OUTPUT_STEM}.tex",
    }
    if not overwrite and any(path.exists() for path in paths.values()):
        raise FileExistsError(f"PhysicsSim-BFGS comparison already exists in {outdir}.")
    table, _audit = build_physics_bfgs_comparison(
        as_runs=pd.read_csv(as_outdir / "progressive_runs_by_seed.csv"),
        bfgs_runs=pd.read_csv(bfgs_outdir / "bfgs_runs_by_seed.csv"),
    )
    table.to_csv(paths["csv"], index=False, encoding="utf-8-sig")
    paths["markdown"].write_text(
        "\n".join(
            [
                "# PhysicsSim-Full vs HPO-selected SciPy-BFGS",
                "",
                "Model rows report mean ± sample SD. The paired-difference row is BFGS minus "
                "PhysicsSim-Full and reports mean [95% percentile-bootstrap CI]; n_pairs = 100.",
                "Held-out test coordinates are used only for post-hoc RMSE evaluation.",
                "",
                _markdown_table(table),
                "",
            ]
        ),
        encoding="utf-8",
    )
    paths["latex"].write_text(
        _latex_longtable(table, equal_width_columns=True), encoding="utf-8"
    )
    return paths


def verify_exported_comparison(
    *, as_outdir: str | Path, bfgs_outdir: str | Path, table_csv: str | Path
) -> None:
    expected, _audit = build_physics_bfgs_comparison(
        as_runs=pd.read_csv(Path(as_outdir) / "progressive_runs_by_seed.csv"),
        bfgs_runs=pd.read_csv(Path(bfgs_outdir) / "bfgs_runs_by_seed.csv"),
    )
    actual = pd.read_csv(table_csv, dtype=str, keep_default_na=False)
    expected_strings = expected.astype(str)
    if list(actual.columns) != list(expected_strings.columns) or not actual.equals(expected_strings):
        raise ValueError("Exported PhysicsSim-BFGS table differs from recomputed source values.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-outdir", required=True)
    parser.add_argument("--bfgs-outdir", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    paths = export_physics_bfgs_comparison(
        as_outdir=args.as_outdir,
        bfgs_outdir=args.bfgs_outdir,
        outdir=args.outdir,
        overwrite=args.overwrite,
    )
    verify_exported_comparison(
        as_outdir=args.as_outdir,
        bfgs_outdir=args.bfgs_outdir,
        table_csv=paths["csv"],
    )
    print(f"[Verified] {paths['csv']}")


if __name__ == "__main__":
    main()
