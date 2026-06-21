"""Export publication-ready tables from progressive AS results.

Currently supported:
  - Random+Align aggregate statistics (no per-run positions or coordinates).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


RANDOM_VARIANT = "Random+Align"
PROGRESSIVE_CHAIN_VARIANTS = [
    "PhysicsSim-DistOnly",
    "PhysicsSim-DistDir",
    "PhysicsSim-DistDirAnch",
    "PhysicsSim-Full",
]
SUMMARY_COLUMNS = [
    "metric",
    "n",
    "mean",
    "std",
    "se",
    "median",
    "iqr",
    "ci95_lo",
    "ci95_hi",
    "q05",
    "q25",
    "q75",
    "q95",
]


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(_format_value(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def _mean_ci_table(table: pd.DataFrame, leading_columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            **{column: table[column] for column in leading_columns},
            "Mean [95% CI]": [
                f"{mean:.6g} [{low:.6g}, {high:.6g}]"
                for mean, low, high in zip(table["mean"], table["ci95_lo"], table["ci95_hi"])
            ],
        }
    )


def export_random_layout_summary(*, as_outdir: str | Path, outdir: str | Path, overwrite: bool = False) -> dict[str, Path]:
    """Export all aggregate Random+Align metrics without per-run layout data."""
    source_dir = Path(as_outdir)
    source_path = source_dir / "random_align_summary.csv"
    if not source_path.exists():
        raise FileNotFoundError(f"Random-layout summary not found: {source_path}")

    summary = pd.read_csv(source_path)
    if "variant" not in summary.columns:
        raise ValueError(f"Missing variant column in {source_path}")
    summary = summary[summary["variant"] == RANDOM_VARIANT].copy()
    if summary.empty:
        raise ValueError(f"No {RANDOM_VARIANT} rows found in {source_path}")
    missing = [column for column in SUMMARY_COLUMNS if column not in summary.columns]
    if missing:
        raise ValueError(f"Missing Random+Align summary columns: {missing}")

    table = summary.loc[:, SUMMARY_COLUMNS].sort_values("metric").reset_index(drop=True)
    destination = Path(outdir)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "table_random_layout_summary.csv"
    markdown_path = destination / "table_random_layout_summary.md"
    mean_ci_csv_path = destination / "table_random_layout_mean_ci.csv"
    mean_ci_markdown_path = destination / "table_random_layout_mean_ci.md"
    output_paths = (csv_path, markdown_path, mean_ci_csv_path, mean_ci_markdown_path)
    if not overwrite and any(path.exists() for path in output_paths):
        raise FileExistsError(f"Paper table already exists in {destination}; use --overwrite to replace it.")

    table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    central = table.loc[:, ["metric", "n", "mean", "std", "se", "ci95_lo", "ci95_hi", "median", "iqr"]]
    quantiles = table.loc[:, ["metric", "q05", "q25", "q75", "q95"]]
    markdown = "\n".join(
        [
            "# Random+Align Aggregate Results",
            "",
            "All values summarize the 1,000 Random+Align runs. Per-run positions are intentionally excluded.",
            "",
            "## Central Statistics",
            "",
            _markdown_table(central),
            "",
            "## Distribution Quantiles",
            "",
            _markdown_table(quantiles),
            "",
        ]
    )
    markdown_path.write_text(markdown, encoding="utf-8")
    mean_ci_table = _mean_ci_table(table, ["metric"])
    mean_ci_table.to_csv(mean_ci_csv_path, index=False, encoding="utf-8-sig")
    mean_ci_markdown_path.write_text(
        "\n".join(
            [
                "# Random+Align Mean [95% CI]",
                "",
                "All values summarize the 1,000 Random+Align runs.",
                "",
                _markdown_table(mean_ci_table),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "csv": csv_path,
        "markdown": markdown_path,
        "mean_ci_csv": mean_ci_csv_path,
        "mean_ci_markdown": mean_ci_markdown_path,
    }


def export_progressive_chain_summary(*, as_outdir: str | Path, outdir: str | Path, overwrite: bool = False) -> dict[str, Path]:
    """Export all aggregate statistics for the four progressive PhysicsSim variants."""
    source_path = Path(as_outdir) / "progressive_summary.csv"
    if not source_path.exists():
        raise FileNotFoundError(f"Progressive summary not found: {source_path}")

    summary = pd.read_csv(source_path)
    missing = [column for column in ["variant", *SUMMARY_COLUMNS] if column not in summary.columns]
    if missing:
        raise ValueError(f"Missing progressive-summary columns: {missing}")
    table = summary[summary["variant"].isin(PROGRESSIVE_CHAIN_VARIANTS)].copy()
    variants_present = set(table["variant"])
    missing_variants = [variant for variant in PROGRESSIVE_CHAIN_VARIANTS if variant not in variants_present]
    if missing_variants:
        raise ValueError(f"Missing progressive-chain variants: {missing_variants}")
    table["variant"] = pd.Categorical(table["variant"], categories=PROGRESSIVE_CHAIN_VARIANTS, ordered=True)
    table = table.loc[:, ["variant", *SUMMARY_COLUMNS]].sort_values(["variant", "metric"]).reset_index(drop=True)

    destination = Path(outdir)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "table_progressive_chain_summary.csv"
    markdown_path = destination / "table_progressive_chain_summary.md"
    mean_ci_csv_path = destination / "table_progressive_chain_mean_ci.csv"
    mean_ci_markdown_path = destination / "table_progressive_chain_mean_ci.md"
    output_paths = (csv_path, markdown_path, mean_ci_csv_path, mean_ci_markdown_path)
    if not overwrite and any(path.exists() for path in output_paths):
        raise FileExistsError(f"Paper table already exists in {destination}; use --overwrite to replace it.")

    table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    central = table.loc[:, ["variant", "metric", "n", "mean", "std", "se", "ci95_lo", "ci95_hi", "median", "iqr"]]
    quantiles = table.loc[:, ["variant", "metric", "q05", "q25", "q75", "q95"]]
    markdown_path.write_text(
        "\n".join(
            [
                "# Progressive Information Chain Aggregate Results",
                "",
                "All values summarize 100 seeds per model. Per-run positions are intentionally excluded.",
                "",
                "## Central Statistics",
                "",
                _markdown_table(central),
                "",
                "## Distribution Quantiles",
                "",
                _markdown_table(quantiles),
                "",
            ]
        ),
        encoding="utf-8",
    )
    mean_ci_long = _mean_ci_table(table, ["variant", "metric"])
    metric_columns = list(dict.fromkeys(table["metric"].tolist()))
    mean_ci_table = (
        mean_ci_long.pivot(index="variant", columns="metric", values="Mean [95% CI]")
        .reindex(PROGRESSIVE_CHAIN_VARIANTS)
        .reindex(columns=metric_columns)
        .reset_index()
    )
    mean_ci_table.to_csv(mean_ci_csv_path, index=False, encoding="utf-8-sig")
    mean_ci_markdown_path.write_text(
        "\n".join(
            [
                "# Progressive Information Chain Mean [95% CI]",
                "",
                "All values summarize 100 seeds per model.",
                "",
                _markdown_table(mean_ci_table),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "csv": csv_path,
        "markdown": markdown_path,
        "mean_ci_csv": mean_ci_csv_path,
        "mean_ci_markdown": mean_ci_markdown_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export publication-ready tables from progressive AS results.")
    parser.add_argument("--as-outdir", required=True, help="Formal progressive AS output directory.")
    parser.add_argument("--outdir", required=True, help="New directory for exported paper tables.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing tables in --outdir.")
    args = parser.parse_args()
    random_paths = export_random_layout_summary(as_outdir=args.as_outdir, outdir=args.outdir, overwrite=args.overwrite)
    chain_paths = export_progressive_chain_summary(as_outdir=args.as_outdir, outdir=args.outdir, overwrite=args.overwrite)
    for label, path in {**random_paths, **{f"progressive_chain_{key}": value for key, value in chain_paths.items()}}.items():
        print(f"[Saved] {label}: {path}")


if __name__ == "__main__":
    main()
