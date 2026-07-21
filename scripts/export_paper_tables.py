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
DISTONLY_VS_DISTDIR_METRICS = [
    "RMSE_test_km",
    "E_distance_stress",
    "E_direction_vr",
    "E_direction_mae",
    "crowding_violation_rate_tau_0p1",
    "collapse_node_rate_tau_0p1",
    "nnd_q05_km",
    "distance_edge_crossing_rate",
]
DISTDIR_VS_DISTDIRANCH_METRICS = [
    "RMSE_test_km",
    "E_distance_stress",
    "E_direction_vr",
    "E_direction_mae",
    "crowding_violation_rate_tau_0p1",
    "collapse_node_rate_tau_0p1",
    "nnd_q05_km",
    "distance_edge_crossing_rate",
]
DISTDIRANCH_VS_FULL_METRICS = [
    "RMSE_test_km",
    "E_distance_stress",
    "E_direction_vr",
    "E_direction_mae",
    "crowding_violation_rate_tau_0p1",
    "collapse_node_rate_tau_0p1",
    "nnd_q05_km",
    "distance_edge_crossing_rate",
]
PAPER_METRIC_ORDER = [
    "RMSE_test_km",
    "E_distance_stress",
    "E_direction_vr",
    "E_direction_mae",
    "crowding_violation_rate_tau_0p1",
    "collapse_node_rate_tau_0p1",
    "nnd_q05_km",
    "distance_edge_crossing_rate",
]
METRIC_LABELS = {
    "E_direction_mae": "Mean Angular Error (rad)",
    "E_direction_vr": "Violation Rate",
    "E_distance_stress": "Stress",
    "RMSE_test_km": "RMSE (km)",
    "MAE_test_km": "Mean Absolute Error (km)",
    "median_error_km": "Median Error (km)",
    "collapse_node_rate_tau_0p1": "Collapse Node Rate (τ = 0.10)",
    "crowding_violation_rate_tau_0p1": "Crowding Violation Rate (τ = 0.10)",
    "nnd_q05_km": "Nearest-Neighbor Distance, 5th Quantile (km)",
    "distance_edge_crossing_rate": "Crossing-edge rate",
}
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


DISPLAY_STAT_COLUMN = "Mean ± SD"


def _mean_sd_table(table: pd.DataFrame, leading_columns: list[str]) -> pd.DataFrame:
    leading_values = {}
    for column in leading_columns:
        if column == "metric":
            leading_values[column] = table[column].map(METRIC_LABELS).fillna(table[column])
        else:
            leading_values[column] = table[column]
    return pd.DataFrame(
        {
            **leading_values,
            DISPLAY_STAT_COLUMN: [
                f"{mean:.6g} ± {std:.6g}"
                for mean, std in zip(table["mean"], table["std"])
            ],
        }
    )


def _mean_sd_from_summary_row(row: pd.Series) -> str:
    return f"{float(row['mean']):.6g} ± {float(row['std']):.6g}"


def _mean_ci_from_paired_row(row: pd.Series) -> str:
    return (
        f"{float(row['paired_diff_mean']):.6g} "
        f"[{float(row['paired_diff_ci95_lo']):.6g}, {float(row['paired_diff_ci95_hi']):.6g}]"
    )


def _sample_std(values: pd.Series) -> float:
    return float(pd.to_numeric(values, errors="raise").std(ddof=1))


def _load_paired_diff_stds(
    *,
    as_outdir: str | Path,
    metrics: list[str],
    left_variant: str,
    right_variant: str,
) -> dict[str, float]:
    source_path = Path(as_outdir) / "progressive_runs_by_seed.csv"
    if not source_path.exists():
        raise FileNotFoundError(f"Progressive runs-by-seed file is required for paired SD: {source_path}")
    runs = pd.read_csv(source_path)
    required = {"variant", "seed", *metrics}
    missing = required.difference(runs.columns)
    if missing:
        raise ValueError(f"Missing progressive-runs columns for paired SD: {sorted(missing)}")
    paired_stds: dict[str, float] = {}
    for metric in metrics:
        left = runs.loc[runs["variant"] == left_variant, ["seed", metric]].rename(columns={metric: "left"})
        right = runs.loc[runs["variant"] == right_variant, ["seed", metric]].rename(columns={metric: "right"})
        merged = left.merge(right, on="seed", how="inner")
        if merged.empty:
            raise ValueError(f"No paired seeds found for {left_variant} vs {right_variant}, metric={metric}")
        paired_stds[metric] = _sample_std(merged["left"] - merged["right"])
    return paired_stds


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
    legacy_csv_path = destination / "table_random_align_baseline_full_statistics.csv"
    legacy_markdown_path = destination / "table_random_align_baseline_full_statistics.md"
    mean_sd_csv_path = destination / "table_random_layout_mean_sd.csv"
    mean_sd_markdown_path = destination / "table_random_layout_mean_sd.md"
    output_paths = (csv_path, markdown_path, legacy_csv_path, legacy_markdown_path, mean_sd_csv_path, mean_sd_markdown_path)
    if not overwrite and any(path.exists() for path in output_paths):
        raise FileExistsError(f"Paper table already exists in {destination}; use --overwrite to replace it.")

    table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    table.to_csv(legacy_csv_path, index=False, encoding="utf-8-sig")
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
    legacy_markdown_path.write_text(
        markdown.replace("# Random+Align Aggregate Results", "# Random+Align Lower-Bound Baseline: Full Statistics"),
        encoding="utf-8",
    )
    mean_sd_table = _mean_sd_table(table, ["metric"])
    mean_sd_table.to_csv(mean_sd_csv_path, index=False, encoding="utf-8-sig")
    mean_sd_markdown_path.write_text(
        "\n".join(
            [
                "# Random+Align Mean ± SD",
                "",
                "All values summarize the 1,000 Random+Align runs.",
                "",
                _markdown_table(mean_sd_table),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "csv": csv_path,
        "markdown": markdown_path,
        "legacy_full_statistics_csv": legacy_csv_path,
        "legacy_full_statistics_markdown": legacy_markdown_path,
        "mean_sd_csv": mean_sd_csv_path,
        "mean_sd_markdown": mean_sd_markdown_path,
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
    mean_sd_csv_path = destination / "table_progressive_chain_mean_sd.csv"
    mean_sd_markdown_path = destination / "table_progressive_chain_mean_sd.md"
    output_paths = (csv_path, markdown_path, mean_sd_csv_path, mean_sd_markdown_path)
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
    mean_sd_long = _mean_sd_table(table, ["variant", "metric"])
    metric_columns = list(dict.fromkeys(table["metric"].tolist()))
    mean_sd_table = (
        mean_sd_long.pivot(index="variant", columns="metric", values=DISPLAY_STAT_COLUMN)
        .reindex(PROGRESSIVE_CHAIN_VARIANTS)
        .reindex(columns=[METRIC_LABELS.get(metric, metric) for metric in metric_columns])
        .reset_index()
    )
    mean_sd_table.to_csv(mean_sd_csv_path, index=False, encoding="utf-8-sig")
    mean_sd_markdown_path.write_text(
        "\n".join(
            [
                "# Progressive Information Chain Mean ± SD",
                "",
                "All values summarize 100 seeds per model.",
                "",
                _markdown_table(mean_sd_table),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "csv": csv_path,
        "markdown": markdown_path,
        "mean_sd_csv": mean_sd_csv_path,
        "mean_sd_markdown": mean_sd_markdown_path,
    }


def _export_progressive_paired_comparison(
    *,
    as_outdir: str | Path,
    outdir: str | Path,
    overwrite: bool,
    metrics: list[str],
    comparison: str,
    left_variant: str,
    right_variant: str,
    paired_label: str,
    title: str,
    filename_stem: str,
) -> dict[str, Path]:
    """Export one planned progressive-chain paired comparison as a wide table."""
    source_dir = Path(as_outdir)
    summary_path = source_dir / "progressive_summary.csv"
    paired_path = source_dir / "progressive_paired_comparisons.csv"
    if not summary_path.exists() or not paired_path.exists():
        raise FileNotFoundError("Progressive summary and paired-comparison CSV files are both required.")
    summary = pd.read_csv(summary_path)
    paired = pd.read_csv(paired_path)
    rows = []
    for metric in metrics:
        variant_rows = summary[
            (summary["metric"] == metric)
            & summary["variant"].isin([right_variant, left_variant])
        ].set_index("variant")
        required_variants = {right_variant, left_variant}
        if set(variant_rows.index) != required_variants:
            raise ValueError(f"Missing comparison summary rows for {metric}: {required_variants}")
        paired_rows = paired[
            (paired["comparison"] == comparison)
            & (paired["left_variant"] == left_variant)
            & (paired["right_variant"] == right_variant)
            & (paired["metric"] == metric)
        ]
        if len(paired_rows) != 1:
            raise ValueError(f"Expected one paired row for {comparison}/{metric}, found {len(paired_rows)}")
        pair = paired_rows.iloc[0]
        metric_label = METRIC_LABELS[metric]
        rows.extend(
            [
                {
                    "model": right_variant,
                    "metric_label": metric_label,
                    DISPLAY_STAT_COLUMN: _mean_sd_from_summary_row(variant_rows.loc[right_variant]),
                },
                {
                    "model": left_variant,
                    "metric_label": metric_label,
                    DISPLAY_STAT_COLUMN: _mean_sd_from_summary_row(variant_rows.loc[left_variant]),
                },
                {
                    "model": paired_label,
                    "metric_label": metric_label,
                    DISPLAY_STAT_COLUMN: _mean_ci_from_paired_row(pair),
                },
            ]
        )
    model_order = [right_variant, left_variant, paired_label]
    metric_order = [METRIC_LABELS[metric] for metric in metrics]
    table = (
        pd.DataFrame(rows)
        .pivot(index="model", columns="metric_label", values=DISPLAY_STAT_COLUMN)
        .reindex(model_order)
        .reindex(columns=metric_order)
        .reset_index()
    )
    destination = Path(outdir)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / f"table_{filename_stem}_paired_comparison.csv"
    markdown_path = destination / f"table_{filename_stem}_paired_comparison.md"
    if not overwrite and any(path.exists() for path in (csv_path, markdown_path)):
        raise FileExistsError(f"Paper table already exists in {destination}; use --overwrite to replace it.")
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    markdown = "\n".join(
        [
            f"# {title}: Paired Comparison",
            "",
            f"Paired difference is defined as {left_variant} minus {right_variant}; n_pairs = 100.",
            "",
            _markdown_table(table),
            "",
        ]
    )
    markdown_path.write_text(markdown, encoding="utf-8")
    return {"csv": csv_path, "markdown": markdown_path}


def export_distonly_vs_distdir_comparison(*, as_outdir: str | Path, outdir: str | Path, overwrite: bool = False) -> dict[str, Path]:
    """Export the planned paired comparison for adding directional information."""
    return _export_progressive_paired_comparison(
        as_outdir=as_outdir,
        outdir=outdir,
        overwrite=overwrite,
        metrics=DISTONLY_VS_DISTDIR_METRICS,
        comparison="direction_given_distance",
        left_variant="PhysicsSim-DistDir",
        right_variant="PhysicsSim-DistOnly",
        paired_label="DistDir - DistOnly paired",
        title="DistOnly vs DistDir",
        filename_stem="distonly_vs_distdir",
    )


def export_distdir_vs_distdiranch_comparison(*, as_outdir: str | Path, outdir: str | Path, overwrite: bool = False) -> dict[str, Path]:
    """Export the planned paired comparison for adding optimization-time anchors."""
    return _export_progressive_paired_comparison(
        as_outdir=as_outdir,
        outdir=outdir,
        overwrite=overwrite,
        metrics=DISTDIR_VS_DISTDIRANCH_METRICS,
        comparison="optimization_anchors_given_distance_direction",
        left_variant="PhysicsSim-DistDirAnch",
        right_variant="PhysicsSim-DistDir",
        paired_label="DistDirAnch - DistDir paired",
        title="DistDir vs DistDirAnch",
        filename_stem="distdir_vs_distdiranch",
    )


def export_distdiranch_vs_full_comparison(*, as_outdir: str | Path, outdir: str | Path, overwrite: bool = False) -> dict[str, Path]:
    """Export the planned paired comparison for adding repulsion regularization."""
    return _export_progressive_paired_comparison(
        as_outdir=as_outdir,
        outdir=outdir,
        overwrite=overwrite,
        metrics=DISTDIRANCH_VS_FULL_METRICS,
        comparison="repulsion_given_distance_direction_anchors",
        left_variant="PhysicsSim-Full",
        right_variant="PhysicsSim-DistDirAnch",
        paired_label="Full - DistDirAnch paired",
        title="DistDirAnch vs Full",
        filename_stem="distdiranch_vs_full",
    )


def _export_unpaired_model_comparison(
    *,
    as_outdir: str | Path,
    outdir: str | Path,
    overwrite: bool,
    variants: list[str],
    title: str,
    filename_stem: str,
) -> dict[str, Path]:
    """Export model-level aggregate statistics without paired inference."""
    source_path = Path(as_outdir) / "progressive_summary.csv"
    if not source_path.exists():
        raise FileNotFoundError(f"Progressive summary not found: {source_path}")
    summary = pd.read_csv(source_path)
    required_columns = {"variant", "metric", "mean", "std"}
    missing_columns = required_columns.difference(summary.columns)
    if missing_columns:
        raise ValueError(f"Missing progressive-summary columns: {sorted(missing_columns)}")
    selected = summary[summary["variant"].isin(variants)].copy()
    if set(selected["variant"]) != set(variants):
        raise ValueError(f"Missing variants for {title}: {variants}")
    metric_sets = {variant: set(selected.loc[selected["variant"] == variant, "metric"]) for variant in variants}
    if len({frozenset(metrics) for metrics in metric_sets.values()}) != 1:
        raise ValueError(f"Variants for {title} do not contain the same metric set.")
    available_metrics = next(iter(metric_sets.values()))
    ordered_metrics = [metric for metric in PAPER_METRIC_ORDER if metric in available_metrics]
    unknown_labels = [metric for metric in ordered_metrics if metric not in METRIC_LABELS]
    if unknown_labels:
        raise ValueError(f"No paper label defined for metrics: {unknown_labels}")
    selected[DISPLAY_STAT_COLUMN] = selected.apply(_mean_sd_from_summary_row, axis=1)
    selected["metric_label"] = selected["metric"].map(METRIC_LABELS)
    metric_labels = [METRIC_LABELS[metric] for metric in ordered_metrics]
    table = (
        selected.pivot(index="variant", columns="metric_label", values=DISPLAY_STAT_COLUMN)
        .reindex(variants)
        .reindex(columns=metric_labels)
        .reset_index()
        .rename(columns={"variant": "model"})
    )
    destination = Path(outdir)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / f"table_{filename_stem}_comparison.csv"
    markdown_path = destination / f"table_{filename_stem}_comparison.md"
    if not overwrite and any(path.exists() for path in (csv_path, markdown_path)):
        raise FileExistsError(f"Paper table already exists in {destination}; use --overwrite to replace it.")
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    markdown_path.write_text(
        "\n".join(
            [
                f"# {title}",
                "",
                "All values summarize 100 independent seeds per model. No paired comparison is performed.",
                "",
                _markdown_table(table),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {"csv": csv_path, "markdown": markdown_path}


def export_smacof_vs_distonly_comparison(*, as_outdir: str | Path, outdir: str | Path, overwrite: bool = False) -> dict[str, Path]:
    """Export the information-matched SMACOF vs PhysicsSim-DistOnly table."""
    return _export_unpaired_model_comparison(
        as_outdir=as_outdir,
        outdir=outdir,
        overwrite=overwrite,
        variants=["SMACOF", "PhysicsSim-DistOnly"],
        title="SMACOF vs PhysicsSim-DistOnly: Information-Matched Comparison",
        filename_stem="smacof_vs_distonly_information_matched",
    )


def export_dc_smacof_vs_distdir_comparison(*, as_outdir: str | Path, outdir: str | Path, overwrite: bool = False) -> dict[str, Path]:
    """Export the information-matched DC-SMACOF vs PhysicsSim-DistDir table."""
    return _export_unpaired_model_comparison(
        as_outdir=as_outdir,
        outdir=outdir,
        overwrite=overwrite,
        variants=["DC-SMACOF", "PhysicsSim-DistDir"],
        title="DC-SMACOF vs PhysicsSim-DistDir: Information-Matched Comparison",
        filename_stem="dc_smacof_vs_distdir_information_matched",
    )


def export_overall_model_comparison(*, as_outdir: str | Path, outdir: str | Path, overwrite: bool = False) -> dict[str, Path]:
    """Export the overall SMACOF, DC-SMACOF, and Full PhysicsSim comparison."""
    return _export_unpaired_model_comparison(
        as_outdir=as_outdir,
        outdir=outdir,
        overwrite=overwrite,
        variants=["SMACOF", "DC-SMACOF", "PhysicsSim-Full"],
        title="Overall Model Comparison",
        filename_stem="overall_model_comparison",
    )


def export_smacof_dc_smacof_baseline_full_statistics(*, as_outdir: str | Path, outdir: str | Path, overwrite: bool = False) -> dict[str, Path]:
    """Export all aggregate statistics for the two algorithmic baseline models."""
    source_path = Path(as_outdir) / "progressive_summary.csv"
    if not source_path.exists():
        raise FileNotFoundError(f"Progressive summary not found: {source_path}")
    summary = pd.read_csv(source_path)
    missing = [column for column in ["variant", *SUMMARY_COLUMNS] if column not in summary.columns]
    if missing:
        raise ValueError(f"Missing baseline summary columns: {missing}")
    variants = ["SMACOF", "DC-SMACOF"]
    table = summary[summary["variant"].isin(variants)].loc[:, ["variant", *SUMMARY_COLUMNS]].copy()
    if set(table["variant"]) != set(variants):
        raise ValueError(f"Missing SMACOF/DC-SMACOF baseline rows in {source_path}")
    table["variant"] = pd.Categorical(table["variant"], categories=variants, ordered=True)
    table = table.sort_values(["variant", "metric"]).reset_index(drop=True)
    destination = Path(outdir)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "table_smacof_dc_smacof_baselines_full_statistics.csv"
    markdown_path = destination / "table_smacof_dc_smacof_baselines_full_statistics.md"
    if not overwrite and any(path.exists() for path in (csv_path, markdown_path)):
        raise FileExistsError(f"Paper table already exists in {destination}; use --overwrite to replace it.")
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    markdown_path.write_text(
        "\n".join(
            [
                "# SMACOF and DC-SMACOF Baselines: Full Statistics",
                "",
                "All values summarize 100 independent seeds per model. No paired comparison is performed.",
                "",
                _markdown_table(table),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {"csv": csv_path, "markdown": markdown_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export publication-ready tables from progressive AS results.")
    parser.add_argument("--as-outdir", required=True, help="Formal progressive AS output directory.")
    parser.add_argument("--outdir", required=True, help="New directory for exported paper tables.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing tables in --outdir.")
    args = parser.parse_args()
    random_paths = export_random_layout_summary(as_outdir=args.as_outdir, outdir=args.outdir, overwrite=args.overwrite)
    chain_paths = export_progressive_chain_summary(as_outdir=args.as_outdir, outdir=args.outdir, overwrite=args.overwrite)
    paired_paths = export_distonly_vs_distdir_comparison(as_outdir=args.as_outdir, outdir=args.outdir, overwrite=args.overwrite)
    anchor_paths = export_distdir_vs_distdiranch_comparison(as_outdir=args.as_outdir, outdir=args.outdir, overwrite=args.overwrite)
    full_paths = export_distdiranch_vs_full_comparison(as_outdir=args.as_outdir, outdir=args.outdir, overwrite=args.overwrite)
    smacof_paths = export_smacof_vs_distonly_comparison(as_outdir=args.as_outdir, outdir=args.outdir, overwrite=args.overwrite)
    dc_smacof_paths = export_dc_smacof_vs_distdir_comparison(as_outdir=args.as_outdir, outdir=args.outdir, overwrite=args.overwrite)
    overall_paths = export_overall_model_comparison(as_outdir=args.as_outdir, outdir=args.outdir, overwrite=args.overwrite)
    algorithmic_baseline_paths = export_smacof_dc_smacof_baseline_full_statistics(as_outdir=args.as_outdir, outdir=args.outdir, overwrite=args.overwrite)
    paths = {
        **random_paths,
        **{f"progressive_chain_{key}": value for key, value in chain_paths.items()},
        **{f"distonly_vs_distdir_{key}": value for key, value in paired_paths.items()},
        **{f"distdir_vs_distdiranch_{key}": value for key, value in anchor_paths.items()},
        **{f"distdiranch_vs_full_{key}": value for key, value in full_paths.items()},
        **{f"smacof_vs_distonly_{key}": value for key, value in smacof_paths.items()},
        **{f"dc_smacof_vs_distdir_{key}": value for key, value in dc_smacof_paths.items()},
        **{f"overall_model_{key}": value for key, value in overall_paths.items()},
        **{f"algorithmic_baselines_{key}": value for key, value in algorithmic_baseline_paths.items()},
    }
    for label, path in paths.items():
        print(f"[Saved] {label}: {path}")


if __name__ == "__main__":
    main()
