"""Regenerate slide-ready paper figures from exported paper tables."""

from __future__ import annotations

import argparse
import html
from pathlib import Path

import pandas as pd


CORE_METRICS = ["RMSE (km)", "Stress", "Violation Rate", "Mean Angular Error (rad)"]
PROGRESSIVE_METRICS = [
    "RMSE (km)",
    "Stress",
    "Violation Rate",
    "Mean Angular Error (rad)",
    "Collapse Node Rate (τ = 0.10)",
    "Crossing-edge rate",
]
REP_ACCURACY_METRICS = ["RMSE (km)", "Stress", "Violation Rate", "Mean Angular Error (rad)"]
REP_LAYOUT_METRICS = [
    "Collapse Node Rate (τ = 0.10)",
    "Crowding Violation Rate (τ = 0.10)",
    "Nearest-Neighbor Distance, 5th Quantile (km)",
    "Crossing-edge rate",
]

SVG_NS = "http://www.w3.org/2000/svg"
RAW_TO_LABEL = {
    "E_distance_stress": "Stress",
    "E_direction_mae": "Mean Angular Error (rad)",
    "E_direction_vr": "Violation Rate",
    "RMSE_test_km": "RMSE (km)",
    "MAE_test_km": "Mean Absolute Error (km)",
    "collapse_node_rate_tau_0p1": "Collapse Node Rate (τ = 0.10)",
    "crowding_violation_rate_tau_0p1": "Crowding Violation Rate (τ = 0.10)",
    "distance_edge_crossing_rate": "Crossing-edge rate",
    "median_error_km": "Median Error (km)",
    "nnd_q05_km": "Nearest-Neighbor Distance, 5th Quantile (km)",
}


def _fmt(value: object) -> str:
    text = str(value)
    if "±" not in text:
        return text
    left, right = [part.strip() for part in text.split("±", 1)]
    try:
        return f"{_fmt_number(float(left))} ± {_fmt_number(float(right))}"
    except ValueError:
        return text


def _fmt_number(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 100:
        return f"{value:.1f}"
    if abs_value >= 10:
        return f"{value:.2f}"
    if abs_value >= 1:
        return f"{value:.3f}"
    return f"{value:.4f}"


def _wrap_label(label: str, max_chars: int = 22) -> list[str]:
    text = _fmt(label)
    if len(text) <= max_chars:
        return [text]
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines[:3]


def _text(x: float, y: float, text: str, cls: str, *, anchor: str = "middle") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}">{html.escape(text)}</text>'


def _multiline_text(x: float, y: float, lines: list[str], cls: str, *, anchor: str = "middle", line_gap: float = 36) -> str:
    if len(lines) == 1:
        return _text(x, y, lines[0], cls, anchor=anchor)
    start = y - (len(lines) - 1) * line_gap / 2
    return "\n".join(_text(x, start + i * line_gap, line, cls, anchor=anchor) for i, line in enumerate(lines))


def _write_svg(path: Path, body: str, *, title: str, desc: str, width: int = 2400, height: int = 1350) -> None:
    svg = f'''<svg xmlns="{SVG_NS}" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">{html.escape(title)}</title>
  <desc id="desc">{html.escape(desc)}</desc>
  <style>
    .title {{ font: 700 64px Arial, sans-serif; fill: #172033; }}
    .subtitle {{ font: 400 28px Arial, sans-serif; fill: #52627A; }}
    .header {{ font: 700 25px Arial, sans-serif; fill: #FFFFFF; }}
    .rowhead {{ font: 700 27px Arial, sans-serif; fill: #172033; }}
    .cell {{ font: 400 23px Arial, sans-serif; fill: #172033; }}
    .cellstrong {{ font: 700 23px Arial, sans-serif; fill: #172033; }}
    .note {{ font: 400 23px Arial, sans-serif; fill: #64748B; }}
  </style>
  <rect width="{width}" height="{height}" fill="#F8FAFC"/>
{body}
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def _font(size: int, *, bold: bool = False):
    from PIL import ImageFont

    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _draw_center(draw, xy, text: str, font, fill="#172033"):
    bbox = draw.textbbox((0, 0), text, font=font)
    x, y = xy
    draw.text((x - (bbox[2] - bbox[0]) / 2, y - (bbox[3] - bbox[1]) / 2), text, font=font, fill=fill)


def _draw_multiline_center(draw, x: float, y: float, lines: list[str], font, fill="#172033", line_gap: int = 64):
    if len(lines) == 1:
        _draw_center(draw, (x, y), lines[0], font, fill)
        return
    start = y - (len(lines) - 1) * line_gap / 2
    for i, line in enumerate(lines):
        _draw_center(draw, (x, start + i * line_gap), line, font, fill)


def _draw_multiline_left(draw, x: float, y: float, lines: list[str], font, fill="#172033", line_gap: int = 64):
    start = y - (len(lines) - 1) * line_gap / 2
    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        draw.text((x, start + i * line_gap - (bbox[3] - bbox[1]) / 2), line, font=font, fill=fill)


def _save_png_table(
    *,
    path: Path,
    table: pd.DataFrame,
    row_col: str,
    value_cols: list[str],
    title: str,
    subtitle: str,
    note: str,
    strong_last_row: bool,
) -> None:
    from PIL import Image, ImageDraw

    scale = 2
    width, height = 2400 * scale, 1350 * scale
    img = Image.new("RGB", (width, height), "#F8FAFC")
    draw = ImageDraw.Draw(img)
    title_font = _font(64 * scale, bold=True)
    subtitle_font = _font(28 * scale)
    header_font = _font(25 * scale, bold=True)
    row_font = _font(27 * scale, bold=True)
    cell_font = _font(23 * scale)
    cell_bold = _font(23 * scale, bold=True)
    note_font = _font(23 * scale)

    draw.text((width / 2, 118 * scale), title, font=title_font, fill="#172033", anchor="mm")
    draw.text((width / 2, 166 * scale), subtitle, font=subtitle_font, fill="#52627A", anchor="mm")

    left, top = 70 * scale, 250 * scale
    table_w, table_h = 2260 * scale, 850 * scale
    row_h = table_h / (len(table) + 1)
    rowhead_w = (450 if len(value_cols) <= 4 else 390) * scale
    col_w = (table_w - rowhead_w) / len(value_cols)

    draw.rounded_rectangle((left, top, left + table_w, top + table_h), radius=14 * scale, fill="#FFFFFF", outline="#CBD5E1", width=3 * scale)
    draw.rectangle((left, top, left + table_w, top + row_h), fill="#1E3A5F")
    draw.line((left + rowhead_w, top, left + rowhead_w, top + table_h), fill="#CBD5E1", width=3 * scale)
    for i in range(1, len(value_cols)):
        x = left + rowhead_w + i * col_w
        draw.line((x, top, x, top + table_h), fill="#CBD5E1", width=3 * scale)
    _draw_center(draw, (left + rowhead_w / 2, top + row_h * 0.58), row_col, header_font, "#FFFFFF")
    for c, col in enumerate(value_cols):
        x = left + rowhead_w + col_w * (c + 0.5)
        _draw_multiline_center(draw, x, top + row_h * 0.58, _wrap_label(col, 17), header_font, "#FFFFFF", line_gap=30 * scale)

    for r, (_, row) in enumerate(table.iterrows()):
        y = top + row_h * (r + 1)
        fill = "#F8FAFC" if r % 2 == 0 else "#FFFFFF"
        draw.rectangle((left, y, left + table_w, y + row_h), fill=fill)
        _draw_multiline_left(draw, left + 36 * scale, y + row_h * 0.58, _wrap_label(row[row_col], 25), row_font, line_gap=32 * scale)
        font = cell_bold if strong_last_row and r == len(table) - 1 else cell_font
        for c, col in enumerate(value_cols):
            x = left + rowhead_w + col_w * (c + 0.5)
            _draw_center(draw, (x, y + row_h * 0.58), _fmt(row[col]), font)
        draw.line((left, y + row_h, left + table_w, y + row_h), fill="#CBD5E1", width=2 * scale)
    draw.text((left, 1188 * scale), note, font=note_font, fill="#64748B")
    img.save(path)


def _table_figure(
    *,
    table: pd.DataFrame,
    row_col: str,
    value_cols: list[str],
    title: str,
    subtitle: str,
    note: str,
    out_svg: Path,
    desc: str,
    strong_last_row: bool = False,
) -> None:
    width, height = 2400, 1350
    left, top = 70, 250
    table_w, table_h = 2260, 850
    row_h = table_h / (len(table) + 1)
    rowhead_w = 450 if len(value_cols) <= 4 else 390
    col_w = (table_w - rowhead_w) / len(value_cols)
    parts = [
        _text(width / 2, 118, title, "title"),
        _text(width / 2, 166, subtitle, "subtitle"),
        f'<rect x="{left}" y="{top}" width="{table_w}" height="{table_h}" rx="14" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="3"/>',
        f'<path d="M{left} {top} H{left + table_w} V{top + row_h} H{left} Z" fill="#1E3A5F"/>',
        f'<line x1="{left + rowhead_w}" y1="{top}" x2="{left + rowhead_w}" y2="{top + table_h}" stroke="#CBD5E1" stroke-width="3"/>',
    ]
    for i in range(1, len(value_cols)):
        x = left + rowhead_w + i * col_w
        parts.append(f'<line x1="{x}" y1="{top}" x2="{x}" y2="{top + table_h}" stroke="#CBD5E1" stroke-width="3"/>')
    parts.append(_text(left + rowhead_w / 2, top + row_h * 0.58, row_col, "header"))
    for c, col in enumerate(value_cols):
        x = left + rowhead_w + col_w * (c + 0.5)
        parts.append(_multiline_text(x, top + row_h * 0.58, _wrap_label(col, 17), "header", line_gap=30))

    for r, (_, row) in enumerate(table.iterrows()):
        y = top + row_h * (r + 1)
        fill = "#F8FAFC" if r % 2 == 0 else "#FFFFFF"
        parts.append(f'<rect x="{left}" y="{y}" width="{table_w}" height="{row_h}" fill="{fill}"/>')
        parts.append(_multiline_text(left + 36, y + row_h * 0.58, _wrap_label(row[row_col], 25), "rowhead", anchor="start", line_gap=32))
        cell_cls = "cellstrong" if strong_last_row and r == len(table) - 1 else "cell"
        for c, col in enumerate(value_cols):
            x = left + rowhead_w + col_w * (c + 0.5)
            parts.append(_text(x, y + row_h * 0.58, _fmt(row[col]), cell_cls))
        parts.append(f'<line x1="{left}" y1="{y + row_h}" x2="{left + table_w}" y2="{y + row_h}" stroke="#CBD5E1" stroke-width="2"/>')
    parts.append(_text(left, 1188, note, "note", anchor="start"))
    _write_svg(out_svg, "\n".join(parts), title=title, desc=desc, width=width, height=height)
    _save_png_table(
        path=out_svg.with_name(out_svg.stem + "_4800w.png"),
        table=table,
        row_col=row_col,
        value_cols=value_cols,
        title=title,
        subtitle=subtitle,
        note=note,
        strong_last_row=strong_last_row,
    )


def _select_columns(table: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    existing = [col for col in cols if col in table.columns]
    missing = [col for col in cols if col not in table.columns]
    if missing:
        raise ValueError(f"Missing table columns: {missing}")
    return table.loc[:, existing]


def export_figures(*, table_dir: str | Path, outdir: str | Path) -> list[Path]:
    table_dir = Path(table_dir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    random_table = pd.read_csv(table_dir / "table_random_layout_mean_sd.csv")
    random_core = random_table[random_table["metric"].isin(["Stress", "Mean Angular Error (rad)", "Violation Rate", "RMSE (km)"])].copy()
    random_core["Metric"] = random_core["metric"]
    random_core = random_core.loc[:, ["Metric", "Mean ± SD"]]
    svg = outdir / "random_align_core_metrics_table.svg"
    _table_figure(
        table=random_core,
        row_col="Metric",
        value_cols=["Mean ± SD"],
        title="Random+Align Baseline Results",
        subtitle="Mean ± SD across 1,000 random layouts",
        note="Random+Align is reported as an empirical lower-bound reference condition.",
        out_svg=svg,
        desc="A slide-ready table of four core Random+Align metrics summarized across 1000 runs.",
    )
    written.extend([svg, svg.with_name(svg.stem + "_4800w.png")])

    progressive = pd.read_csv(table_dir / "table_progressive_chain_mean_sd.csv").rename(columns={"variant": "Metric", **RAW_TO_LABEL})
    svg = outdir / "progressive_chain_core_metrics_table.svg"
    _table_figure(
        table=_select_columns(progressive, ["Metric", *PROGRESSIVE_METRICS]),
        row_col="Metric",
        value_cols=PROGRESSIVE_METRICS,
        title="Progressive PhysicsSim Ablation: Core Metrics",
        subtitle="Mean ± SD across 100 seeds per variant; lower is better except nearest-neighbor distance",
        note="Values are reported as mean ± sample SD.",
        out_svg=svg,
        desc="A slide-ready table of core metrics for the four progressive PhysicsSim variants across 100 seeds each.",
        strong_last_row=True,
    )
    written.extend([svg, svg.with_name(svg.stem + "_4800w.png")])

    figure_specs = [
        (
            "table_distonly_vs_distdir_paired_comparison.csv",
            "paired_comparison_direction_core_metrics.svg",
            "Paired Comparison: Direction Constraints",
            "Model rows: mean ± sample SD; paired row: mean [95% bootstrap CI]",
            CORE_METRICS,
        ),
        (
            "table_distdir_vs_distdiranch_paired_comparison.csv",
            "paired_comparison_anchor_core_metrics.svg",
            "Paired Comparison: Anchor Constraints",
            "Model rows: mean ± sample SD; paired row: mean [95% bootstrap CI]",
            CORE_METRICS,
        ),
        (
            "table_distdiranch_vs_full_paired_comparison.csv",
            "paired_comparison_rep_accuracy_metrics.svg",
            "Paired Comparison: REP Accuracy Metrics",
            "Model rows: mean ± sample SD; paired row: mean [95% bootstrap CI]",
            REP_ACCURACY_METRICS,
        ),
        (
            "table_distdiranch_vs_full_paired_comparison.csv",
            "paired_comparison_rep_layout_metrics.svg",
            "Paired Comparison: REP Layout Diagnostics",
            "Model rows: mean ± sample SD; paired row: mean [95% bootstrap CI]",
            REP_LAYOUT_METRICS,
        ),
        (
            "table_smacof_vs_distonly_information_matched_comparison.csv",
            "information_matched_overall_comparison_core_metrics.svg",
            "Information-Matched Baseline Comparison",
            "SMACOF vs DistOnly and DC-SMACOF vs DistDir; values are mean ± sample SD",
            CORE_METRICS,
        ),
        (
            "table_dc_smacof_vs_distdir_information_matched_comparison.csv",
            "information_matched_overall_comparison_core_metrics_dc.svg",
            "Information-Matched Baseline Comparison: Directional Models",
            "DC-SMACOF vs PhysicsSim-DistDir; values are mean ± sample SD",
            CORE_METRICS,
        ),
        (
            "table_overall_model_comparison_comparison.csv",
            "overall_model_comparison_core_metrics.svg",
            "Overall Model Comparison",
            "Core metrics across final baselines and PhysicsSim-Full; values are mean ± sample SD",
            CORE_METRICS,
        ),
    ]
    for csv_name, svg_name, title, subtitle, cols in figure_specs:
        table = pd.read_csv(table_dir / csv_name)
        svg = outdir / svg_name
        _table_figure(
            table=_select_columns(table, ["model", *cols]),
            row_col="model",
            value_cols=cols,
            title=title,
            subtitle=subtitle,
            note="For paired rows, SD is computed from same-seed paired differences.",
            out_svg=svg,
            desc=title,
            strong_last_row="paired" in " ".join(str(x) for x in table["model"].tolist()).lower(),
        )
        written.extend([svg, svg.with_name(svg.stem + "_4800w.png")])

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate slide-ready paper figures from paper tables.")
    parser.add_argument("--table-dir", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    for path in export_figures(table_dir=args.table_dir, outdir=args.outdir):
        print(f"[Saved] {path}")


if __name__ == "__main__":
    main()
