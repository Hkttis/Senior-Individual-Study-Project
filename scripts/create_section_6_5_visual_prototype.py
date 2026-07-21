"""Create a compact Section 6.5 visualization prototype.

The figure uses the already verified representative runs from the formal
progressive AS pipeline. It does not rerun models or modify experiment outputs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))
(Path.cwd() / ".matplotlib").mkdir(exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, colors
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D

from library.config import FILE_PATHS, km2pix, refer_pos_sim
from library.data_io import get_anchor_labels, get_test_site_labels, load_ini_data_from_csv, uploading_directional_data, uploading_ground_truth
from library.units import data_Li2sim
from MDS_model.plot_node_link_diagram import wrong_directions_nonflip
from run_paper_script.ch5_ablation_progressive import _target_positions_sim


DEFAULT_VARIANTS = ("PhysicsSim-Full", "SMACOF", "DC-SMACOF")
DISPLAY_NAMES = {
    "PhysicsSim-Full": "PhysicsSim-Full",
    "SMACOF": "SMACOF",
    "DC-SMACOF": "DC-SMACOF",
}
PANEL_LETTERS = tuple("abcdefghijklmnopqrstuvwxyz")
LABEL_OFFSETS = (
    (4, 4),
    (4, -8),
    (-4, 4),
    (-4, -8),
    (8, 0),
    (-8, 0),
    (0, 8),
    (0, -12),
    (10, 6),
    (-10, 6),
    (10, -10),
    (-10, -10),
)
OVERLAY_HANDLES = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#222222", markersize=11, label="Ground-truth test site"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="#7b3294", markeredgecolor="#222222", markersize=11, label="Model test site"),
    Line2D([0], [0], color="#9c9c9c", lw=1.8, label="Test-site displacement"),
    Line2D([0], [0], marker="^", color="none", markerfacecolor="#222222", markeredgecolor="white", markersize=12, label="Calibration anchor"),
]
NODE_HANDLES = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor="#2ca02c", markeredgecolor="white", markersize=11, label="Direction satisfied"),
    Line2D([0], [0], marker="x", linestyle="none", color="#d62728", markerfacecolor="none", markeredgecolor="#d62728", markersize=11, markeredgewidth=2.2, label="Direction violation"),
]


def _cjk_font(size: float) -> FontProperties:
    for path in (
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\mingliu.ttc"),
        Path(r"C:\Windows\Fonts\msyhbd.ttc"),
    ):
        if path.exists():
            return FontProperties(fname=str(path), size=size)
    return FontProperties(size=size)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--progressive-outdir",
        default="outputs/ch5_progressive_as_physics_alpha_1_beta_-0.5_dc_alpha_-0.5_100seeds_random1000",
    )
    parser.add_argument(
        "--representative-dir",
        default="outputs/ch6_section_6_5_full_smacof_dc_representative",
        help="Directory containing representative_selection.json from verified ch6 visualizations.",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/ch6_section_6_5_visual_prototype",
    )
    parser.add_argument(
        "--variants",
        default=",".join(DEFAULT_VARIANTS),
        help="Comma-separated variants to include.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overlay-label-size", type=float, default=8.2)
    parser.add_argument("--map-label-size", type=float, default=7.4)
    parser.add_argument(
        "--shared-extent",
        action="store_true",
        help="Use a common spatial extent across all panels. Default uses tighter per-panel extents for readability.",
    )
    parser.add_argument(
        "--panel-label-start",
        type=int,
        default=0,
        help="Zero-based panel-letter offset for the overlay row. The error-map row starts after the overlay row.",
    )
    parser.add_argument(
        "--row-label-x",
        type=float,
        default=0.018,
        help="Figure-coordinate x position for vertical row labels in the integrated figure.",
    )
    return parser.parse_args()


def _load_selected_records(representative_dir: Path, variants: list[str]) -> list[dict]:
    path = representative_dir / "representative_selection.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing representative selection file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_variant = {row["variant"]: row for row in payload["selections"]}
    missing = sorted(set(variants).difference(by_variant))
    if missing:
        raise ValueError(f"Representative selection is missing variants: {missing}")
    return [by_variant[variant] for variant in variants]


def _load_position_matrix(progressive_outdir: Path, record: dict, vertice: list[str]) -> np.ndarray:
    path = progressive_outdir / "progressive_final_positions_y_up_sim.csv"
    positions = pd.read_csv(path, encoding="utf-8-sig")
    selected = positions[
        (positions["variant"] == record["variant"])
        & (positions["seed"] == int(record["seed"]))
    ]
    if len(selected) != len(vertice):
        raise ValueError(f"Expected {len(vertice)} positions for {record['variant']} seed {record['seed']}, got {len(selected)}")
    selected = selected.set_index("label").loc[vertice]
    return selected[["x_y_up_sim", "y_y_up_sim"]].to_numpy(float)


def _distance_edge_errors(points: np.ndarray, data_sim: list[list[object]], dni: dict[str, int]) -> list[tuple[int, int, float]]:
    rows = []
    for source, target, distance in data_sim:
        i, j = dni[source], dni[target]
        actual = float(np.linalg.norm(points[i] - points[j]))
        ideal = float(distance)
        error = abs(actual - ideal) / ideal if ideal > 0.0 else 0.0
        rows.append((i, j, error))
    return rows


def _wrong_direction_nodes(points: np.ndarray, vertice: list[str], dni: dict[str, int]) -> set[int]:
    wrong = wrong_directions_nonflip(points.tolist(), vertice, dni)
    nodes: set[int] = set()
    for source, target, *_rest in wrong:
        if source in dni:
            nodes.add(dni[source])
        if target in dni:
            nodes.add(dni[target])
    return nodes


def _combined_extent(model_points: dict[str, np.ndarray], target_positions: dict[str, np.ndarray], labels: list[str], pad_frac: float = 0.16):
    arrays = list(model_points.values()) + [np.asarray([target_positions[label] for label in labels], dtype=float)]
    pts = np.vstack(arrays)
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    span = max(xmax - xmin, ymax - ymin, 1.0)
    pad = span * pad_frac
    cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    half = span / 2.0 + pad
    return cx - half, cx + half, cy - half, cy + half


def _combined_overlay_extent(
    model_points: dict[str, np.ndarray],
    target_positions: dict[str, np.ndarray],
    dni: dict[str, int],
    anchors: list[str],
    tests: list[str],
    pad_frac: float = 0.075,
):
    arrays = []
    for points in model_points.values():
        labels = [label for label in tests + anchors if label in dni]
        arrays.append(points[[dni[label] for label in labels]])
    arrays.append(np.asarray([target_positions[label] for label in tests + anchors if label in target_positions], dtype=float))
    pts = np.vstack(arrays)
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    span = max(xmax - xmin, ymax - ymin, 1.0)
    pad = span * pad_frac
    cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    half = span / 2.0 + pad
    return cx - half, cx + half, cy - half, cy + half


def _panel_extent(points: np.ndarray, extra_points: list[np.ndarray] | None = None, pad_frac: float = 0.12):
    arrays = [np.asarray(points, dtype=float)]
    if extra_points:
        arrays.extend(np.asarray(point, dtype=float).reshape(1, 2) for point in extra_points)
    pts = np.vstack(arrays)
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    span_x = max(float(xmax - xmin), 1.0)
    span_y = max(float(ymax - ymin), 1.0)
    pad_x = span_x * pad_frac
    pad_y = span_y * pad_frac
    return xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y


def _annotate_entries(
    ax,
    entries,
    fontprops: FontProperties,
    *,
    color: str = "#222222",
    zorder: int = 10,
    clip_on: bool = True,
):
    annotations = []
    for idx, (label, x, y) in enumerate(entries):
        dx, dy = LABEL_OFFSETS[idx % len(LABEL_OFFSETS)]
        ha = "left" if dx >= 0 else "right"
        va = "bottom" if dy >= 0 else "top"
        annotations.append(ax.annotate(
            str(label),
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            color=color,
            fontproperties=fontprops,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.35},
            zorder=zorder,
            clip_on=clip_on,
        ))
    return annotations


def _relax_annotations(fig, ax, annotations, *, iterations: int = 120, step: float = 1.1, max_offset: float = 32.0):
    """Small dependency-free label relaxation in annotation-offset space."""
    if len(annotations) < 2:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for _ in range(iterations):
        moved = False
        boxes = [ann.get_window_extent(renderer=renderer).expanded(1.06, 1.12) for ann in annotations]
        shifts = [[0.0, 0.0] for _ in annotations]
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if not boxes[i].overlaps(boxes[j]):
                    continue
                ci = np.asarray([(boxes[i].x0 + boxes[i].x1) / 2.0, (boxes[i].y0 + boxes[i].y1) / 2.0])
                cj = np.asarray([(boxes[j].x0 + boxes[j].x1) / 2.0, (boxes[j].y0 + boxes[j].y1) / 2.0])
                delta = ci - cj
                norm = float(np.linalg.norm(delta))
                if norm < 1e-9:
                    delta = np.asarray([1.0, 0.6])
                    norm = float(np.linalg.norm(delta))
                unit = delta / norm
                shifts[i][0] += unit[0] * step
                shifts[i][1] += unit[1] * step
                shifts[j][0] -= unit[0] * step
                shifts[j][1] -= unit[1] * step
        for ann, (sx, sy) in zip(annotations, shifts):
            if sx == 0.0 and sy == 0.0:
                continue
            x, y = ann.get_position()
            nx = float(np.clip(x + sx, -max_offset, max_offset))
            ny = float(np.clip(y + sy, -max_offset, max_offset))
            ann.set_position((nx, ny))
            moved = True
        if not moved:
            break
        fig.canvas.draw()


def _format_metrics(record: dict) -> str:
    metrics = record["rerun_metrics"]
    return (
        f"RMSE={metrics['RMSE_test_km']:.1f} km, "
        f"Stress={metrics['E_distance_stress']:.3f}, "
        f"VR={metrics['E_direction_vr']:.3f}"
    )


def _panel_label_text(letter: str, variant: str) -> str:
    return f"({letter}) {DISPLAY_NAMES.get(variant, variant)}"


def _panel_letters(n_panels: int, *, start: int = 0) -> tuple[str, ...]:
    if start + n_panels > len(PANEL_LETTERS):
        raise ValueError(f"Too many panels for built-in labels: {n_panels=}, {start=}")
    return PANEL_LETTERS[start:start + n_panels]


def _draw_overlay(
    ax,
    points,
    targets,
    record,
    vertice,
    dni,
    anchors,
    tests,
    error_norm,
    error_cmap,
    label_font,
    *,
    title_fontsize: float = 10.0,
    title_y: float | None = None,
    draw_title: bool = True,
):
    ax.scatter(points[:, 0], points[:, 1], s=15, c="#c9c9c9", edgecolors="none", alpha=0.40, zorder=1)
    label_entries = []
    for label in tests:
        idx = dni[label]
        gt = np.asarray(targets[label], dtype=float)
        pred = points[idx]
        err = float(np.linalg.norm(pred - gt) / km2pix)
        ax.plot([pred[0], gt[0]], [pred[1], gt[1]], color="#808080", lw=1.65, alpha=0.90, zorder=2)
        ax.scatter(gt[0], gt[1], s=68, marker="o", facecolors="white", edgecolors="#222222", linewidths=1.35, zorder=4)
        ax.scatter(pred[0], pred[1], s=76, marker="o", color=error_cmap(error_norm(err)), edgecolors="#222222", linewidths=0.75, zorder=5)
        label_entries.append((label, pred[0], pred[1]))
    for k, label in enumerate(anchors, start=1):
        if label not in dni:
            continue
        p = points[dni[label]]
        ax.scatter(p[0], p[1], s=104, marker="^", color="#222222", edgecolors="white", linewidths=0.75, zorder=6)
        label_entries.append((label, p[0], p[1]))
    annotations = _annotate_entries(ax, label_entries, label_font, zorder=8)
    if draw_title:
        ax.set_title(
            f"{DISPLAY_NAMES.get(record['variant'], record['variant'])}\n{_format_metrics(record)}",
            fontsize=title_fontsize,
            pad=12,
            y=title_y,
        )
    return annotations


def _draw_error_map(ax, points, edge_errors, wrong_nodes, edge_norm, edge_cmap, vertice, label_font, *, clip_labels: bool = True):
    for i, j, err in edge_errors:
        ax.plot(
            [points[i, 0], points[j, 0]],
            [points[i, 1], points[j, 1]],
            color=edge_cmap(edge_norm(err)),
            lw=1.65,
            alpha=0.78,
            zorder=1,
        )
    satisfied_idx = [i for i in range(len(points)) if i not in wrong_nodes]
    wrong_idx = sorted(wrong_nodes)
    if satisfied_idx:
        ax.scatter(
            points[satisfied_idx, 0],
            points[satisfied_idx, 1],
            s=48,
            marker="o",
            c="#2ca02c",
            edgecolors="white",
            linewidths=0.7,
            zorder=3,
        )
    if wrong_idx:
        ax.scatter(
            points[wrong_idx, 0],
            points[wrong_idx, 1],
            s=80,
            marker="x",
            c="#d62728",
            linewidths=2.0,
            zorder=4,
        )
    return _annotate_entries(
        ax,
        [(label, float(points[i, 0]), float(points[i, 1])) for i, label in enumerate(vertice)],
        label_font,
        zorder=5,
        clip_on=clip_labels,
    )


def _style_axis(ax, extent):
    xmin, xmax, ymin, ymax = extent
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _save_large_overlay_figure(
    *,
    outdir,
    records,
    model_points,
    targets,
    vertice,
    dni,
    anchors,
    tests,
    overlay_norm,
    overlay_cmap,
    label_font,
    dpi,
):
    n_cols = len(records)
    fig, axes = plt.subplots(1, n_cols, figsize=(max(24.0, 7.8 * n_cols), 9.6), constrained_layout=True)
    axes = np.asarray(axes).reshape(1, -1)[0]
    shared_extent = _combined_overlay_extent(model_points, targets, dni, anchors, tests, pad_frac=0.075)
    for ax, record in zip(axes, records):
        variant = record["variant"]
        points = model_points[variant]
        annotations = _draw_overlay(
            ax,
            points,
            targets,
            record,
            vertice,
            dni,
            anchors,
            tests,
            overlay_norm,
            overlay_cmap,
            label_font,
            draw_title=False,
        )
        _style_axis(ax, shared_extent)
        _relax_annotations(fig, ax, annotations, iterations=150, max_offset=70.0)
    title_y = 0.915
    title_xs = np.linspace(0.13, 0.87, n_cols)
    for title_x, letter, record in zip(title_xs, _panel_letters(n_cols, start=0), records):
        fig.text(
            title_x,
            title_y,
            _panel_label_text(letter, record["variant"]),
            ha="center",
            va="top",
            fontsize=19,
            fontweight="bold",
        )
        fig.text(
            title_x,
            title_y - 0.032,
            _format_metrics(record),
            ha="center",
            va="top",
            fontsize=16,
        )
    sm_overlay = cm.ScalarMappable(norm=overlay_norm, cmap=overlay_cmap)
    sm_overlay.set_array([])
    cbar = fig.colorbar(sm_overlay, ax=axes, orientation="vertical", fraction=0.012, pad=0.018)
    cbar.set_label("Test-site error (km)", fontsize=13)
    cbar.ax.tick_params(labelsize=11)
    fig.legend(handles=OVERLAY_HANDLES, loc="lower center", ncol=4, frameon=False, fontsize=13, bbox_to_anchor=(0.5, 0.02))
    png_path = outdir / "section_6_5_overlay_large.png"
    svg_path = outdir / "section_6_5_overlay_large.svg"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path


def _save_large_error_map_figure(
    *,
    outdir,
    records,
    model_points,
    edge_errors,
    wrong_nodes,
    edge_norm,
    edge_cmap,
    vertice,
    label_font,
    dpi,
):
    n_cols = len(records)
    fig, axes = plt.subplots(1, n_cols, figsize=(max(24.0, 7.8 * n_cols), 9.6), constrained_layout=True)
    axes = np.asarray(axes).reshape(1, -1)[0]
    for ax, record in zip(axes, records):
        variant = record["variant"]
        points = model_points[variant]
        extent = _panel_extent(points, [], pad_frac=0.15)
        annotations = _draw_error_map(
            ax,
            points,
            edge_errors[variant],
            wrong_nodes[variant],
            edge_norm,
            edge_cmap,
            vertice,
            label_font,
            clip_labels=False,
        )
        _style_axis(ax, extent)
        _relax_annotations(fig, ax, annotations, iterations=260, max_offset=82.0)
    title_y = 0.890
    title_xs = np.linspace(0.13, 0.87, n_cols)
    for title_x, letter, record in zip(title_xs, _panel_letters(n_cols, start=n_cols), records):
        fig.text(
            title_x,
            title_y,
            _panel_label_text(letter, record["variant"]),
            ha="center",
            va="top",
            fontsize=21,
            fontweight="bold",
        )
    sm_edge = cm.ScalarMappable(norm=edge_norm, cmap=edge_cmap)
    sm_edge.set_array([])
    cbar = fig.colorbar(sm_edge, ax=axes, orientation="vertical", fraction=0.012, pad=0.028)
    cbar.set_label("Distance-edge relative error", fontsize=13)
    cbar.ax.tick_params(labelsize=11)
    fig.legend(handles=NODE_HANDLES, loc="lower center", ncol=2, frameon=False, fontsize=13, bbox_to_anchor=(0.5, 0.02))
    png_path = outdir / "section_6_5_error_map_large.png"
    svg_path = outdir / "section_6_5_error_map_large.svg"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path


def main() -> None:
    args = _parse_args()
    progressive_outdir = Path(args.progressive_outdir)
    representative_dir = Path(args.representative_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    records = _load_selected_records(representative_dir, variants)
    overlay_panel_labels = _panel_letters(len(records), start=args.panel_label_start)
    error_panel_labels = _panel_letters(len(records), start=args.panel_label_start + len(records))
    graph, vertice, dni, edges, distance_data = load_ini_data_from_csv(FILE_PATHS)
    gt_lonlat = uploading_ground_truth(vertice, dni)
    config = json.loads((progressive_outdir / "progressive_config.json").read_text(encoding="utf-8"))
    anchor_label = str(config["anchor_align_label"])
    anchors = list(config.get("calibration_labels") or get_anchor_labels())
    tests = list(config.get("test_labels") or get_test_site_labels())
    target_refer_pos = config.get("refer_pos_sim", refer_pos_sim)
    targets = _target_positions_sim(dni, gt_lonlat, anchor_label, target_refer_pos)
    data_sim = data_Li2sim(distance_data)

    model_points = {record["variant"]: _load_position_matrix(progressive_outdir, record, vertice) for record in records}
    edge_errors = {variant: _distance_edge_errors(points, data_sim, dni) for variant, points in model_points.items()}
    wrong_nodes = {variant: _wrong_direction_nodes(points, vertice, dni) for variant, points in model_points.items()}

    overlay_errors = []
    for variant, points in model_points.items():
        for label in tests:
            overlay_errors.append(float(np.linalg.norm(points[dni[label]] - np.asarray(targets[label], dtype=float)) / km2pix))
    edge_error_values = [err for rows in edge_errors.values() for *_ij, err in rows]
    overlay_norm = colors.Normalize(vmin=0.0, vmax=max(overlay_errors) if overlay_errors else 1.0)
    edge_vmax = max(float(np.quantile(edge_error_values, 0.95)), 0.03) if edge_error_values else 0.03
    edge_norm = colors.Normalize(vmin=0.0, vmax=edge_vmax)
    overlay_cmap = plt.get_cmap("plasma")
    edge_cmap = plt.get_cmap("RdYlGn_r")

    overlay_shared_extent = _combined_overlay_extent(model_points, targets, dni, anchors, tests, pad_frac=0.075)
    overlay_label_font = _cjk_font(args.overlay_label_size)
    map_label_font = _cjk_font(args.map_label_size)
    fig, axes = plt.subplots(2, len(records), figsize=(max(23.8, 7.6 * len(records)), 12.0), constrained_layout=True)
    if len(records) == 1:
        axes = np.asarray(axes).reshape(2, 1)

    for col, record in enumerate(records):
        variant = record["variant"]
        points = model_points[variant]
        overlay_extent = overlay_shared_extent
        map_extent = _panel_extent(points, [], pad_frac=0.17)
        overlay_annotations = _draw_overlay(
            axes[0, col],
            points,
            targets,
            record,
            vertice,
            dni,
            anchors,
            tests,
            overlay_norm,
            overlay_cmap,
            overlay_label_font,
            draw_title=False,
        )
        map_annotations = _draw_error_map(axes[1, col], points, edge_errors[variant], wrong_nodes[variant], edge_norm, edge_cmap, vertice, map_label_font)
        axes[0, col].set_title(
            f"{_panel_label_text(overlay_panel_labels[col], variant)}\n{_format_metrics(record)}",
            fontsize=13,
            fontweight="bold",
            pad=8,
            linespacing=1.25,
        )
        axes[1, col].set_title(_panel_label_text(error_panel_labels[col], variant), fontsize=14, fontweight="bold", pad=8)
        for row, extent in [(0, overlay_extent), (1, map_extent)]:
            ax = axes[row, col]
            _style_axis(ax, extent)
        _relax_annotations(fig, axes[0, col], overlay_annotations, iterations=95, max_offset=44.0)
        _relax_annotations(fig, axes[1, col], map_annotations, iterations=210, max_offset=56.0)

    fig.text(args.row_label_x, 0.705, "Ground-truth overlay", rotation=90, ha="center", va="center", fontsize=14, fontweight="bold")
    fig.text(args.row_label_x, 0.285, "Constraint-error visualization", rotation=90, ha="center", va="center", fontsize=14, fontweight="bold")

    fig.legend(handles=OVERLAY_HANDLES + NODE_HANDLES, loc="lower center", ncol=6, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.01))

    sm_overlay = cm.ScalarMappable(norm=overlay_norm, cmap=overlay_cmap)
    sm_overlay.set_array([])
    cbar1 = fig.colorbar(sm_overlay, ax=axes[0, :], orientation="vertical", fraction=0.010, pad=0.006)
    cbar1.set_label("Test-site error (km)", fontsize=9)
    cbar1.ax.tick_params(labelsize=8)

    sm_edge = cm.ScalarMappable(norm=edge_norm, cmap=edge_cmap)
    sm_edge.set_array([])
    cbar2 = fig.colorbar(sm_edge, ax=axes[1, :], orientation="vertical", fraction=0.010, pad=0.006)
    cbar2.set_label("Distance-edge relative error", fontsize=9)
    cbar2.ax.tick_params(labelsize=8)

    png_path = outdir / "section_6_5_three_model_visualization_prototype.png"
    svg_path = outdir / "section_6_5_three_model_visualization_prototype.svg"
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    large_overlay_png, large_overlay_svg = _save_large_overlay_figure(
        outdir=outdir,
        records=records,
        model_points=model_points,
        targets=targets,
        vertice=vertice,
        dni=dni,
        anchors=anchors,
        tests=tests,
        overlay_norm=overlay_norm,
        overlay_cmap=overlay_cmap,
        label_font=_cjk_font(max(args.overlay_label_size + 5.0, 13.0)),
        dpi=args.dpi,
    )
    large_error_png, large_error_svg = _save_large_error_map_figure(
        outdir=outdir,
        records=records,
        model_points=model_points,
        edge_errors=edge_errors,
        wrong_nodes=wrong_nodes,
        edge_norm=edge_norm,
        edge_cmap=edge_cmap,
        vertice=vertice,
        label_font=_cjk_font(max(args.map_label_size + 5.0, 12.4)),
        dpi=args.dpi,
    )

    metadata = {
        "source_progressive_as": str(progressive_outdir),
        "source_representative_dir": str(representative_dir),
        "variants": variants,
        "seeds": {record["variant"]: int(record["seed"]) for record in records},
        "metrics": {record["variant"]: record["rerun_metrics"] for record in records},
        "visual_design": {
            "rows": ["ground-truth overlay", "constraint-error visualization"],
            "columns": variants,
            "label_policy": "overlay row labels test sites and calibration anchors; constraint-error row labels all countries using small offset CJK labels",
            "extent_policy": "combined figure uses per-panel extents unless --shared-extent is set; large ground-truth overlay uses one shared coordinate extent and equal aspect ratio; large constraint-error panels use independent equal-aspect extents for topology/constraint readability.",
            "panel_labels": {
                "ground_truth_overlay": [f"({letter}) {DISPLAY_NAMES.get(record['variant'], record['variant'])}" for letter, record in zip(overlay_panel_labels, records)],
                "constraint_error_visualization": [f"({letter}) {DISPLAY_NAMES.get(record['variant'], record['variant'])}" for letter, record in zip(error_panel_labels, records)],
            },
            "global_title_policy": "No large in-figure global title; describe figure purpose in the manuscript caption.",
            "large_readable_outputs": [
                str(large_overlay_png),
                str(large_overlay_svg),
                str(large_error_png),
                str(large_error_svg),
            ],
        },
    }
    metadata_path = outdir / "section_6_5_three_model_visualization_prototype.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Saved] {png_path}")
    print(f"[Saved] {svg_path}")
    print(f"[Saved] {large_overlay_png}")
    print(f"[Saved] {large_overlay_svg}")
    print(f"[Saved] {large_error_png}")
    print(f"[Saved] {large_error_svg}")
    print(f"[Saved] {metadata_path}")


if __name__ == "__main__":
    main()
