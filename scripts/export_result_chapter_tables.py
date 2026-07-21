"""Export manuscript Result chapter tables from verified paper result tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SECTION_6_1_METRICS = [
    "RMSE (km)",
    "Stress",
    "Violation Rate",
    "Mean Angular Error (rad)",
    "Crossing-edge rate",
]
SECTION_6_2_1_DIRECTION_METRICS = [
    "RMSE (km)",
    "Stress",
    "Violation Rate",
    "Mean Angular Error (rad)",
    "Crowding Violation Rate (τ = 0.10)",
    "Collapse Node Rate (τ = 0.10)",
    "Nearest-Neighbor Distance, 5th Quantile (km)",
    "Crossing-edge rate",
]
SECTION_6_2_2_ANCHOR_METRICS = SECTION_6_2_1_DIRECTION_METRICS
MODEL_ORDER_FOR_RANDOM_RMSE = [
    "Random+Align",
    "PhysicsSim-DistOnly",
    "SMACOF",
    "DC-SMACOF",
    "PhysicsSim-DistDir",
    "PhysicsSim-DistDirAnch",
    "PhysicsSim-Full",
]
DISPLAY_STAT_COLUMN = "Mean ± SD"


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def _mean_from_mean_sd(value: object) -> float:
    text = str(value)
    for sep in ["±", "¡Ó", "簣"]:
        if sep in text:
            return float(text.split(sep)[0].strip())
    return float(text.split()[0])


def _load_sources(paper_table_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paper_table_dir = Path(paper_table_dir)
    random_path = paper_table_dir / "table_random_layout_mean_sd.csv"
    progressive_path = paper_table_dir / "table_progressive_chain_mean_sd.csv"
    smacof_path = paper_table_dir / "table_smacof_vs_distonly_information_matched_comparison.csv"
    dc_path = paper_table_dir / "table_dc_smacof_vs_distdir_information_matched_comparison.csv"
    overall_path = paper_table_dir / "table_overall_model_comparison_comparison.csv"
    for path in [random_path, progressive_path, smacof_path, dc_path, overall_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing source paper table: {path}")
    return (
        pd.read_csv(random_path, encoding="utf-8-sig"),
        pd.read_csv(progressive_path, encoding="utf-8-sig"),
        pd.read_csv(smacof_path, encoding="utf-8-sig"),
        pd.read_csv(dc_path, encoding="utf-8-sig"),
        pd.read_csv(overall_path, encoding="utf-8-sig"),
    )


def _random_values(random: pd.DataFrame) -> pd.Series:
    if "metric" not in random.columns:
        raise ValueError("Random+Align source table is missing the metric column.")
    value_column = [column for column in random.columns if column != "metric"]
    if len(value_column) != 1:
        raise ValueError(f"Expected one Random+Align value column, found {value_column}")
    return random.set_index("metric")[value_column[0]]


def export_section_6_1_random_vs_physics_full(
    *,
    paper_table_dir: str | Path,
    outdir: str | Path,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Export the original Section 6.1 Random+Align versus PhysicsSim-Full table."""
    random, _progressive, _smacof, _dc, overall = _load_sources(paper_table_dir)
    random_values = _random_values(random)
    missing = sorted(set(SECTION_6_1_METRICS).difference(random_values.index))
    if missing:
        raise ValueError(f"Missing Section 6.1 metrics in Random+Align table: {missing}")
    full_rows = overall[overall["model"] == "PhysicsSim-Full"]
    if len(full_rows) != 1:
        raise ValueError(f"Expected one PhysicsSim-Full row, found {len(full_rows)}")
    full = full_rows.iloc[0]

    table = pd.DataFrame(
        [
            {
                "Model": "Random+Align",
                **{metric: random_values.loc[metric] for metric in SECTION_6_1_METRICS},
            },
            {
                "Model": "PhysicsSim-Full",
                **{metric: full[metric] for metric in SECTION_6_1_METRICS},
            },
        ]
    )

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "table_6_1_random_vs_physics_full.csv"
    markdown_path = outdir / "table_6_1_random_vs_physics_full.md"
    if not overwrite and any(path.exists() for path in [csv_path, markdown_path]):
        raise FileExistsError(f"Result chapter table already exists in {outdir}; use --overwrite to replace it.")
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    markdown_path.write_text(
        "\n".join(
            [
                "# Table 6.1 Random+Align Null Model Versus PhysicsSim-Full",
                "",
                "Values are reported as mean ± sample SD.",
                "",
                _markdown_table(table),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {"csv": csv_path, "markdown": markdown_path}


def export_section_6_1_rmse_reduction_vs_random(
    *,
    paper_table_dir: str | Path,
    outdir: str | Path,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Export all-model RMSE comparison against Random+Align."""
    random, progressive, smacof, dc, overall = _load_sources(paper_table_dir)
    random_values = _random_values(random)
    random_rmse = random_values.loc["RMSE (km)"]
    random_mean = _mean_from_mean_sd(random_rmse)

    values: dict[str, str] = {"Random+Align": random_rmse}
    for _, row in progressive.iterrows():
        values[row["variant"]] = row["RMSE (km)"]
    for frame in [smacof, dc, overall]:
        for _, row in frame.iterrows():
            values.setdefault(row["model"], row["RMSE (km)"])

    rows = []
    for model in MODEL_ORDER_FOR_RANDOM_RMSE:
        if model not in values:
            raise ValueError(f"Missing RMSE value for model: {model}")
        rmse_value = values[model]
        if model == "Random+Align":
            reduction = "Reference"
        else:
            reduction = f"{(random_mean - _mean_from_mean_sd(rmse_value)) / random_mean * 100.0:.2f}%"
        rows.append(
            {
                "Model": model,
                "RMSE, mean ± SD (km)": rmse_value,
                "RMSE reduction vs Random+Align": reduction,
            }
        )
    table = pd.DataFrame(rows)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "table_6_1_rmse_reduction_vs_random.csv"
    markdown_path = outdir / "table_6_1_rmse_reduction_vs_random.md"
    if not overwrite and any(path.exists() for path in [csv_path, markdown_path]):
        raise FileExistsError(f"Result chapter table already exists in {outdir}; use --overwrite to replace it.")
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    markdown_path.write_text(
        "\n".join(
            [
                "# Table 6.1 RMSE Reduction Versus Random+Align",
                "",
                "Values are reported as mean ± sample SD.",
                "",
                _markdown_table(table),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {"csv": csv_path, "markdown": markdown_path}


def export_section_6_2_1_distonly_vs_distdir(
    *,
    paper_table_dir: str | Path,
    outdir: str | Path,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Export Section 6.2.1 direction-constraint paired comparison table."""
    paper_table_dir = Path(paper_table_dir)
    source_path = paper_table_dir / "table_distonly_vs_distdir_paired_comparison.csv"
    if not source_path.exists():
        raise FileNotFoundError(f"Missing DistOnly vs DistDir source table: {source_path}")
    source = pd.read_csv(source_path, encoding="utf-8-sig")
    required_columns = ["model", *SECTION_6_2_1_DIRECTION_METRICS]
    missing = [column for column in required_columns if column not in source.columns]
    if missing:
        raise ValueError(f"Missing Section 6.2.1 columns: {missing}")
    table = source.loc[:, required_columns].copy()

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "table_6_2_1_distonly_vs_distdir.csv"
    markdown_path = outdir / "table_6_2_1_distonly_vs_distdir.md"
    if not overwrite and any(path.exists() for path in [csv_path, markdown_path]):
        raise FileExistsError(f"Result chapter table already exists in {outdir}; use --overwrite to replace it.")
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    markdown_path.write_text(
        "\n".join(
            [
                "# Table 6.2.1 DistOnly vs DistDir: Paired Comparison",
                "",
                "Paired difference is defined as PhysicsSim-DistDir minus PhysicsSim-DistOnly; n_pairs = 100.",
                "",
                _markdown_table(table),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {"csv": csv_path, "markdown": markdown_path}


def export_section_6_2_2_distdir_vs_distdiranch(
    *,
    paper_table_dir: str | Path,
    outdir: str | Path,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Export Section 6.2.2 anchor-effect paired comparison table."""
    paper_table_dir = Path(paper_table_dir)
    source_path = paper_table_dir / "table_distdir_vs_distdiranch_paired_comparison.csv"
    if not source_path.exists():
        raise FileNotFoundError(f"Missing DistDir vs DistDirAnch source table: {source_path}")
    source = pd.read_csv(source_path, encoding="utf-8-sig")
    required_columns = ["model", *SECTION_6_2_2_ANCHOR_METRICS]
    missing = [column for column in required_columns if column not in source.columns]
    if missing:
        raise ValueError(f"Missing Section 6.2.2 columns: {missing}")
    table = source.loc[:, required_columns].copy()

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "table_6_2_2_distdir_vs_distdiranch.csv"
    markdown_path = outdir / "table_6_2_2_distdir_vs_distdiranch.md"
    if not overwrite and any(path.exists() for path in [csv_path, markdown_path]):
        raise FileExistsError(f"Result chapter table already exists in {outdir}; use --overwrite to replace it.")
    table.to_csv(csv_path, index=False, encoding="utf-8-sig")
    markdown_path.write_text(
        "\n".join(
            [
                "# Table 6.2.2 DistDir vs DistDirAnch: Paired Comparison",
                "",
                "Paired difference is defined as PhysicsSim-DistDirAnch minus PhysicsSim-DistDir; n_pairs = 100.",
                "",
                _markdown_table(table),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {"csv": csv_path, "markdown": markdown_path}


def export_result_chapter_tables(
    *,
    paper_table_dir: str | Path,
    outdir: str | Path,
    overwrite: bool = False,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    section_6_1 = export_section_6_1_random_vs_physics_full(
        paper_table_dir=paper_table_dir,
        outdir=outdir,
        overwrite=overwrite,
    )
    rmse_reduction = export_section_6_1_rmse_reduction_vs_random(
        paper_table_dir=paper_table_dir,
        outdir=outdir,
        overwrite=overwrite,
    )
    section_6_2_1 = export_section_6_2_1_distonly_vs_distdir(
        paper_table_dir=paper_table_dir,
        outdir=outdir,
        overwrite=overwrite,
    )
    section_6_2_2 = export_section_6_2_2_distdir_vs_distdiranch(
        paper_table_dir=paper_table_dir,
        outdir=outdir,
        overwrite=overwrite,
    )
    paths.update({f"section_6_1_{key}": value for key, value in section_6_1.items()})
    paths.update({f"section_6_1_rmse_reduction_{key}": value for key, value in rmse_reduction.items()})
    paths.update({f"section_6_2_1_{key}": value for key, value in section_6_2_1.items()})
    paths.update({f"section_6_2_2_{key}": value for key, value in section_6_2_2.items()})
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Export manuscript Result chapter tables.")
    parser.add_argument("--paper-table-dir", required=True, help="Directory containing verified paper result tables.")
    parser.add_argument("--outdir", required=True, help="Output directory for Result chapter tables.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing Result chapter tables.")
    args = parser.parse_args()
    for label, path in export_result_chapter_tables(
        paper_table_dir=args.paper_table_dir,
        outdir=args.outdir,
        overwrite=args.overwrite,
    ).items():
        print(f"[Saved] {label}: {path}")


if __name__ == "__main__":
    main()
