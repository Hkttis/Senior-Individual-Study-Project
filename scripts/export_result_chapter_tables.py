"""Export manuscript Result chapter tables from verified paper result tables."""

from __future__ import annotations

import argparse
import re
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


def _latex_cell(value: object) -> str:
    return (
        str(value)
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


_LATEX_HEADERS = {
    "RMSE (km)": r"\shortstack{RMSE\\(km)}",
    f"RMSE, mean {chr(177)} SD (km)": r"\shortstack{RMSE, mean\\$\pm$ SD (km)}",
    "RMSE reduction vs Random+Align": r"\shortstack{RMSE reduction vs\\Random+Align}",
    "Violation Rate": r"\shortstack{Violation\\Rate}",
    "Mean Angular Error (rad)": r"\shortstack{Mean\\Angular\\Error\\(rad)}",
    "Crowding Violation Rate (τ = 0.10)": r"\shortstack{Crowding\\Violation Rate\\($\tau = 0.10$)}",
    "Collapse Node Rate (τ = 0.10)": r"\shortstack{Collapse\\Node Rate\\($\tau = 0.10$)}",
    "Nearest-Neighbor Distance, 5th Quantile (km)": r"\shortstack{Nearest-\\Neighbor\\Distance, 5th\\Quantile\\(km)}",
    "Crossing-edge rate": r"\shortstack{Crossing-\\edge rate}",
}


def _latex_header(value: object) -> str:
    return _LATEX_HEADERS.get(str(value), _latex_cell(value))


def _latex_model(value: object) -> str:
    text = str(value)
    replacements = {
        "PhysicsSim-DistOnly": r"\shortstack[l]{PhysicsSim-\\DistOnly}",
        "PhysicsSim-DistDir": r"\shortstack[l]{PhysicsSim-\\DistDir}",
        "PhysicsSim-DistDirAnch": r"\shortstack[l]{PhysicsSim-\\DistDirAnch}",
        "PhysicsSim-Full": r"\shortstack[l]{PhysicsSim-\\Full}",
        "Paired difference: DistDir − DistOnly": r"\shortstack[l]{Paired difference:\\DistDir $-$\\DistOnly}",
        "Paired difference: DistDirAnch − DistDir": r"\shortstack[l]{Paired difference:\\DistDirAnch $-$\\DistDir}",
        "Paired difference: Full − DistDirAnch": r"\shortstack[l]{Paired difference:\\Full $-$\\DistDirAnch}",
    }
    return replacements.get(text, _latex_cell(text))


def _latex_stat(value: object, *, multiline: bool = True) -> str:
    text = str(value)
    if " ± " in text:
        mean, sd = text.split(" ± ", 1)
        if not multiline:
            return rf"{_latex_cell(mean)} $\pm$ {_latex_cell(sd)}"
        return rf"\shortstack{{{_latex_cell(mean)}\\$\pm$ {_latex_cell(sd)}}}"
    match = re.fullmatch(r"(.+?) \[([^,]+), ([^\]]+)\]", text)
    if match:
        mean, lo, hi = match.groups()
        return rf"\shortstack{{{_latex_cell(mean)}\\{{}}[{_latex_cell(lo)},\\{_latex_cell(hi)}]}}"
    return _latex_cell(text)


def _latex_panel_table(frame: pd.DataFrame, *, panel_title: str | None = None) -> str:
    n_columns = len(frame.columns)
    array_stretch = "1.30" if n_columns == 3 else "1.45"
    if n_columns == 6:
        widths = ["0.165", *(["0.145"] * 5)]
    elif n_columns == 5:
        widths = ["0.220", *(["0.168"] * 4)]
    elif n_columns == 3:
        widths = ["0.340", "0.270", "0.280"]
    else:
        widths = [f"{0.92 / n_columns:.4f}"] * n_columns
    column_spec = "\n".join(
        (
            rf">{{\raggedright\arraybackslash}}p{{{width}\linewidth}}"
            if index == 0
            else rf">{{\centering\arraybackslash}}p{{{width}\linewidth}}"
        )
        for index, width in enumerate(widths)
    )
    header = " & ".join(_latex_header(column) for column in frame.columns) + r" \\"
    rows = []
    for row_index, row in enumerate(frame.itertuples(index=False, name=None)):
        model_cell = _latex_cell(row[0]) if n_columns == 3 else _latex_model(row[0])
        cells = [
            model_cell,
            *(_latex_stat(value, multiline=n_columns != 3) for value in row[1:]),
        ]
        if str(row[0]).startswith("Paired difference:"):
            rows.append(r"\addlinespace[4pt]")
        rows.append(" & ".join(cells) + r" \\")
        if row_index < len(frame) - 1:
            rows.append(r"\addlinespace[2pt]")
    title_row = []
    if panel_title:
        title_row = [rf"\multicolumn{{{n_columns}}}{{@{{}}l}}{{\textit{{{panel_title}}}}} \\[2pt]"]
    return "\n".join(
        [
            r"\begingroup",
            r"\small",
            r"\setlength{\tabcolsep}{3pt}",
            rf"\renewcommand{{\arraystretch}}{{{array_stretch}}}",
            r"\begin{longtable}[]{@{}" + column_spec + r"@{}}",
            *title_row,
            r"\toprule\noalign{}",
            header,
            r"\midrule\noalign{}",
            r"\endhead",
            r"\bottomrule\noalign{}",
            r"\endlastfoot",
            r"% DATA_ROWS_BEGIN",
            *rows,
            r"% DATA_ROWS_END",
            r"\end{longtable}",
            r"\endgroup",
        ]
    )


def _latex_longtable(frame: pd.DataFrame, *, equal_width_columns: bool) -> str:
    if equal_width_columns and len(frame.columns) == 9:
        accuracy = frame.iloc[:, :5]
        diagnostics = frame.loc[:, [frame.columns[0], *frame.columns[5:]]]
        body = "\n\n".join(
            [
                _latex_panel_table(
                    accuracy,
                    panel_title="Panel A. Site accuracy and constraint satisfaction",
                ),
                r"\vspace{0.7\baselineskip}",
                _latex_panel_table(
                    diagnostics,
                    panel_title="Panel B. Layout and topology diagnostics",
                ),
            ]
        )
        return "\n".join(
            [
                r"{\def\LTcaptype{none} % do not increment counter",
                body,
                "}",
                "",
            ]
        )
    if equal_width_columns and len(frame.columns) in {3, 6}:
        return "\n".join(
            [
                r"{\def\LTcaptype{none} % do not increment counter",
                _latex_panel_table(frame),
                "}",
                "",
            ]
        )
    if equal_width_columns:
        n_columns = len(frame.columns)
        width = f"{1.0 / n_columns:.4f}"
        column_spec = "\n".join(
            rf"  >{{\raggedright\arraybackslash}}p{{(\linewidth - {2 * (n_columns - 1)}\tabcolsep) * \real{{{width}}}}}"
            for _ in frame.columns
        )
        begin = "\n".join([r"\begin{longtable}[]{@{}", column_spec + r"@{}}"])
    else:
        begin = rf"\begin{{longtable}}[]{{@{{}}{'l' * len(frame.columns)}@{{}}}}"
    header = " & ".join(_latex_cell(column) for column in frame.columns) + r" \\"
    rows = [
        " & ".join(_latex_cell(value) for value in row) + r" \\"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join(
        [
            r"{\def\LTcaptype{none} % do not increment counter",
            begin,
            r"\toprule\noalign{}",
            header,
            r"\midrule\noalign{}",
            r"\endhead",
            r"\bottomrule\noalign{}",
            r"\endlastfoot",
            *rows,
            r"\end{longtable}",
            "}",
            "",
        ]
    )


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
    latex_path = outdir / "table_6_1_random_vs_physics_full.tex"
    if not overwrite and any(path.exists() for path in [csv_path, markdown_path, latex_path]):
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
    latex_path.write_text(_latex_longtable(table, equal_width_columns=True), encoding="utf-8")
    return {"csv": csv_path, "markdown": markdown_path, "latex": latex_path}


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
    latex_path = outdir / "table_6_1_rmse_reduction_vs_random.tex"
    if not overwrite and any(path.exists() for path in [csv_path, markdown_path, latex_path]):
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
    latex_path.write_text(_latex_longtable(table, equal_width_columns=True), encoding="utf-8")
    return {"csv": csv_path, "markdown": markdown_path, "latex": latex_path}


def export_section_6_2_1_distonly_vs_distdir(
    *,
    paper_table_dir: str | Path,
    outdir: str | Path,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Export Section 6.2.1 direction-constraint paired comparison table."""
    return _export_curated_comparison(
        paper_table_dir=paper_table_dir,
        outdir=outdir,
        overwrite=overwrite,
        source_name="table_distonly_vs_distdir_paired_comparison.csv",
        output_stem="table_6_2_1_distonly_vs_distdir",
        paired_label_map={"DistDir - DistOnly paired": "Paired difference: DistDir − DistOnly"},
        title="Table 6.2.1 DistOnly vs DistDir: Paired Comparison",
        note="Paired difference is defined as PhysicsSim-DistDir minus PhysicsSim-DistOnly; n_pairs = 100.",
    )


def export_section_6_2_2_distdir_vs_distdiranch(
    *,
    paper_table_dir: str | Path,
    outdir: str | Path,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Export Section 6.2.2 anchor-effect paired comparison table."""
    return _export_curated_comparison(
        paper_table_dir=paper_table_dir,
        outdir=outdir,
        overwrite=overwrite,
        source_name="table_distdir_vs_distdiranch_paired_comparison.csv",
        output_stem="table_6_2_2_distdir_vs_distdiranch",
        paired_label_map={"DistDirAnch - DistDir paired": "Paired difference: DistDirAnch − DistDir"},
        title="Table 6.2.2 DistDir vs DistDirAnch: Paired Comparison",
        note="Paired difference is defined as PhysicsSim-DistDirAnch minus PhysicsSim-DistDir; n_pairs = 100.",
    )


def _export_curated_comparison(
    *,
    paper_table_dir: str | Path,
    outdir: str | Path,
    overwrite: bool,
    source_name: str,
    output_stem: str,
    paired_label_map: dict[str, str] | None,
    title: str,
    note: str,
    page_break_before: bool = False,
) -> dict[str, Path]:
    source_path = Path(paper_table_dir) / source_name
    if not source_path.exists():
        raise FileNotFoundError(f"Missing source paper table: {source_path}")
    source = pd.read_csv(source_path, encoding="utf-8-sig")
    required_columns = ["model", *SECTION_6_2_1_DIRECTION_METRICS]
    missing = [column for column in required_columns if column not in source.columns]
    if missing:
        raise ValueError(f"Missing manuscript table columns in {source_name}: {missing}")
    table = source.loc[:, required_columns].copy().rename(columns={"model": "Model"})
    if paired_label_map:
        table["Model"] = table["Model"].replace(paired_label_map)
    return _write_table_bundle(
        table=table,
        outdir=outdir,
        output_stem=output_stem,
        title=title,
        note=note,
        overwrite=overwrite,
        page_break_before=page_break_before,
    )


def _write_table_bundle(
    *,
    table: pd.DataFrame,
    outdir: str | Path,
    output_stem: str,
    title: str,
    note: str,
    overwrite: bool,
    page_break_before: bool = False,
) -> dict[str, Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {
        "csv": outdir / f"{output_stem}.csv",
        "markdown": outdir / f"{output_stem}.md",
        "latex": outdir / f"{output_stem}.tex",
    }
    if not overwrite and any(path.exists() for path in paths.values()):
        raise FileExistsError(f"Result chapter table already exists in {outdir}; use --overwrite to replace it.")
    table.to_csv(paths["csv"], index=False, encoding="utf-8-sig")
    paths["markdown"].write_text(
        "\n".join([f"# {title}", "", note, "", _markdown_table(table), ""]),
        encoding="utf-8",
    )
    latex = _latex_longtable(table, equal_width_columns=True)
    if page_break_before:
        latex = "\\newpage\n" + latex
    paths["latex"].write_text(latex, encoding="utf-8")
    return paths


def export_section_6_2_3_distdiranch_vs_full(*, paper_table_dir, outdir, overwrite=False):
    return _export_curated_comparison(
        paper_table_dir=paper_table_dir,
        outdir=outdir,
        overwrite=overwrite,
        source_name="table_distdiranch_vs_full_paired_comparison.csv",
        output_stem="table_6_2_3_distdiranch_vs_full",
        paired_label_map={"Full - DistDirAnch paired": "Paired difference: Full − DistDirAnch"},
        title="Table 6.2.3 DistDirAnch vs Full: Paired Comparison",
        note="Paired difference is defined as PhysicsSim-Full minus PhysicsSim-DistDirAnch; n_pairs = 100.",
    )


def export_section_6_3_smacof_vs_distonly(*, paper_table_dir, outdir, overwrite=False):
    return _export_curated_comparison(
        paper_table_dir=paper_table_dir,
        outdir=outdir,
        overwrite=overwrite,
        source_name="table_smacof_vs_distonly_information_matched_comparison.csv",
        output_stem="table_6_3_smacof_vs_distonly",
        paired_label_map=None,
        title="Table 6.3 SMACOF vs PhysicsSim-DistOnly",
        note="Values are reported as mean ± sample SD; no paired comparison is performed.",
    )


def export_section_6_3_dc_smacof_vs_distdir(*, paper_table_dir, outdir, overwrite=False):
    return _export_curated_comparison(
        paper_table_dir=paper_table_dir,
        outdir=outdir,
        overwrite=overwrite,
        source_name="table_dc_smacof_vs_distdir_information_matched_comparison.csv",
        output_stem="table_6_3_dc_smacof_vs_distdir",
        paired_label_map=None,
        title="Table 6.3 DC-SMACOF vs PhysicsSim-DistDir",
        note="Values are reported as mean ± sample SD; no paired comparison is performed.",
    )


def export_section_6_4_overall_model_comparison(*, paper_table_dir, outdir, overwrite=False):
    return _export_curated_comparison(
        paper_table_dir=paper_table_dir,
        outdir=outdir,
        overwrite=overwrite,
        source_name="table_overall_model_comparison_comparison.csv",
        output_stem="table_6_4_overall_model_comparison",
        page_break_before=True,
        paired_label_map=None,
        title="Table 6.4 Overall Model Comparison",
        note="Values are reported as mean ± sample SD; no paired comparison is performed.",
    )


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
    section_6_2_3 = export_section_6_2_3_distdiranch_vs_full(
        paper_table_dir=paper_table_dir, outdir=outdir, overwrite=overwrite,
    )
    section_6_3_smacof = export_section_6_3_smacof_vs_distonly(
        paper_table_dir=paper_table_dir, outdir=outdir, overwrite=overwrite,
    )
    section_6_3_dc = export_section_6_3_dc_smacof_vs_distdir(
        paper_table_dir=paper_table_dir, outdir=outdir, overwrite=overwrite,
    )
    section_6_4 = export_section_6_4_overall_model_comparison(
        paper_table_dir=paper_table_dir, outdir=outdir, overwrite=overwrite,
    )
    paths.update({f"section_6_1_{key}": value for key, value in section_6_1.items()})
    paths.update({f"section_6_1_rmse_reduction_{key}": value for key, value in rmse_reduction.items()})
    paths.update({f"section_6_2_1_{key}": value for key, value in section_6_2_1.items()})
    paths.update({f"section_6_2_2_{key}": value for key, value in section_6_2_2.items()})
    paths.update({f"section_6_2_3_{key}": value for key, value in section_6_2_3.items()})
    paths.update({f"section_6_3_smacof_{key}": value for key, value in section_6_3_smacof.items()})
    paths.update({f"section_6_3_dc_{key}": value for key, value in section_6_3_dc.items()})
    paths.update({f"section_6_4_{key}": value for key, value in section_6_4.items()})
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
