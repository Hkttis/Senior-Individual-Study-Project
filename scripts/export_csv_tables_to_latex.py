"""Export every CSV table in a directory as a matching LaTeX longtable."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _latex_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    return (
        str(value)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
        .replace("$", r"\$")
    )


def dataframe_to_longtable(frame: pd.DataFrame) -> str:
    if frame.empty and len(frame.columns) == 0:
        raise ValueError("Cannot export a CSV with no columns to LaTeX.")
    n_columns = len(frame.columns)
    width = f"{1.0 / n_columns:.6f}"
    column_spec = "\n".join(
        rf"  >{{\raggedright\arraybackslash}}p{{(\linewidth - {2 * (n_columns - 1)}\tabcolsep) * \real{{{width}}}}}"
        for _ in frame.columns
    )
    header = " & ".join(_latex_cell(column) for column in frame.columns) + r" \\"
    rows = [
        " & ".join(_latex_cell(value) for value in row) + r" \\"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join(
        [
            r"{\def\LTcaptype{none} % do not increment counter",
            r"\begin{longtable}[]{@{}",
            column_spec + r"@{}}",
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


def export_csv_tables_to_latex(
    table_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    table_dir = Path(table_dir)
    if not table_dir.exists():
        raise FileNotFoundError(f"Table directory does not exist: {table_dir}")
    csv_paths = sorted(table_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV tables found in: {table_dir}")

    outputs: dict[str, Path] = {}
    for csv_path in csv_paths:
        latex_path = csv_path.with_suffix(".tex")
        if latex_path.exists() and not overwrite:
            outputs[csv_path.stem] = latex_path
            continue
        frame = pd.read_csv(csv_path, encoding="utf-8-sig")
        latex_path.write_text(dataframe_to_longtable(frame), encoding="utf-8")
        outputs[csv_path.stem] = latex_path
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for name, path in export_csv_tables_to_latex(args.table_dir, overwrite=args.overwrite).items():
        print(f"[Saved] {name}: {path}")


if __name__ == "__main__":
    main()

