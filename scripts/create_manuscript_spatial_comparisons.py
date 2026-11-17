"""Create two readable 2x2 spatial-comparison figures for the manuscript.

No model is rerun. Coordinates and metrics are loaded from the registered
formal 100-seed experiments. BFGS uses the same four-metric representative-run
rule as the progressive AS visualizations.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from textwrap import fill

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm, colors
from matplotlib.font_manager import FontProperties


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from library.config import FILE_PATHS, km2pix, refer_pos_sim
from library.data_io import load_ini_data_from_csv, uploading_ground_truth
from library.units import data_Li2sim
from run_paper_script.ch5_ablation_progressive import _target_positions_sim
from scripts.create_section_6_5_visual_prototype import (
    NODE_HANDLES,
    OVERLAY_HANDLES,
    _combined_overlay_extent,
    _distance_edge_errors,
    _draw_error_map,
    _draw_overlay,
    _panel_extent,
    _relax_annotations,
    _style_axis,
    _wrong_direction_nodes,
)


SELECTION_METRICS = (
    "E_distance_stress",
    "E_direction_vr",
    "E_direction_mae",
    "RMSE_test_km",
)
COUNTRY_NAMES_PATH = PROJECT_ROOT / "data" / "country_display_names_yu2006.csv"


def _load_country_display_names(path: Path, vertices: list[str]) -> dict[str, str]:
    """Load the model key and English spelling columns from the author's checklist."""
    names = {}
    with path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            key = row["model_name"]
            value = row["english_name"].strip()
            if not key or not value or key in names:
                raise ValueError(f"Empty or duplicate country display-name entry: {key!r}")
            names[key] = value
    if names.keys() != set(vertices):
        missing = set(vertices) - names.keys()
        extra = names.keys() - set(vertices)
        raise ValueError(f"Country display names do not match model nodes: missing={sorted(missing)}, extra={sorted(extra)}")
    return names


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def select_bfgs_representative(runs: pd.DataFrame) -> dict:
    required = {"variant", "seed", "status", *SELECTION_METRICS}
    missing = required.difference(runs.columns)
    if missing:
        raise ValueError(f"BFGS runs are missing representative-selection columns: {sorted(missing)}")
    ok = runs.loc[runs["status"].eq("ok")].copy()
    values = ok.loc[:, SELECTION_METRICS].astype(float)
    if len(values) != 100 or not np.isfinite(values.to_numpy()).all():
        raise ValueError("BFGS representative selection requires 100 successful finite runs.")
    median = values.median()
    mad = (values - median).abs().median().replace(0.0, 1.0)
    distance = np.sqrt((((values - median) / mad) ** 2).sum(axis=1))
    selected = ok.loc[distance.idxmin()]
    return {
        "variant": "BFGS",
        "source_variant": str(selected["variant"]),
        "seed": int(selected["seed"]),
        "selection_metrics": {metric: float(selected[metric]) for metric in SELECTION_METRICS},
        "median_vector": {metric: float(median[metric]) for metric in SELECTION_METRICS},
        "mad_vector": {metric: float(mad[metric]) for metric in SELECTION_METRICS},
        "standardized_distance": float(distance.loc[selected.name]),
        "rerun_metrics": {metric: float(selected[metric]) for metric in SELECTION_METRICS},
        "selection_rule": "minimum four-metric MAD-standardized distance to the model-specific median profile",
    }


def _load_positions(path: Path, *, variant: str, seed: int, vertices: list[str]) -> np.ndarray:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    selected = frame.loc[(frame["variant"].eq(variant)) & (frame["seed"].eq(seed))].copy()
    if len(selected) != len(vertices) or selected["label"].nunique() != len(vertices):
        raise ValueError(f"Expected {len(vertices)} unique positions for {variant} seed {seed}.")
    selected = selected.set_index("label").loc[vertices]
    points = selected[["x_y_up_sim", "y_y_up_sim"]].to_numpy(float)
    if not np.isfinite(points).all():
        raise ValueError(f"Non-finite positions for {variant} seed {seed}.")
    return points


def _record_metrics_text(record: dict) -> str:
    metrics = record["rerun_metrics"]
    return (
        f"RMSE={metrics['RMSE_test_km']:.1f} km, "
        f"Stress={metrics['E_distance_stress']:.3f}, "
        f"VR={metrics['E_direction_vr']:.3f}"
    )


def _keep_labels_left_of_colorbar(fig, annotations, colorbar_ax, *, gap_points: float = 6.0) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    points_per_pixel = 72.0 / fig.dpi
    right_edge = colorbar_ax.bbox.x0 - gap_points / points_per_pixel
    for annotation in annotations:
        overflow = annotation.get_window_extent(renderer).x1 - right_edge
        if overflow > 0:
            x, y = annotation.get_position()
            annotation.set_position((x - overflow * points_per_pixel, y))


def _save_group_figure(
    *,
    records: list[dict],
    points_by_variant: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    vertices: list[str],
    display_labels: dict[str, str],
    dni: dict[str, int],
    anchors: list[str],
    tests: list[str],
    edge_errors: dict[str, list[tuple[int, int, float]]],
    wrong_nodes: dict[str, set[int]],
    overlay_extent: tuple[float, float, float, float],
    overlay_norm: colors.Normalize,
    edge_norm: colors.Normalize,
    output_stem: Path,
) -> None:
    overlay_cmap = plt.get_cmap("plasma")
    edge_cmap = plt.get_cmap("cividis")
    fig, axes = plt.subplots(2, 2, figsize=(16.8, 12.2), constrained_layout=True)
    overlay_font = FontProperties(size=10.4)
    error_font = FontProperties(size=9.2)
    for col, record in enumerate(records):
        variant = record["variant"]
        points = points_by_variant[variant]
        overlay_annotations = _draw_overlay(
            axes[0, col], points, targets, record, vertices, dni, anchors, tests,
            overlay_norm, overlay_cmap, overlay_font, draw_title=False, display_labels=display_labels,
        )
        error_annotations = _draw_error_map(
            axes[1, col], points, edge_errors[variant], wrong_nodes[variant],
            edge_norm, edge_cmap, vertices, error_font, clip_labels=False, display_labels=display_labels,
        )
        _style_axis(axes[0, col], overlay_extent)
        _style_axis(axes[1, col], _panel_extent(points, [], pad_frac=0.17))
        _relax_annotations(fig, axes[0, col], overlay_annotations, iterations=145, max_offset=66.0)
        _relax_annotations(fig, axes[1, col], error_annotations, iterations=250, max_offset=76.0)

    fig.canvas.draw()
    bottom_title_y = max(axes[1, col].get_position().y1 for col in range(2)) + 0.008
    fig.set_layout_engine(None)
    bottom_row_shift = 0.025
    bottom_row_shift_x = 0.043 if "physics_full_vs_bfgs" in output_stem.name else 0.0
    for col in range(2):
        position = axes[1, col].get_position()
        axes[1, col].set_position(
            [
                position.x0 + bottom_row_shift_x,
                position.y0 + bottom_row_shift,
                position.width,
                position.height,
            ]
        )
    top_positions = [axes[0, col].get_position() for col in range(2)]
    bottom_positions = [axes[1, col].get_position() for col in range(2)]
    top_y = max(position.y1 for position in top_positions) + 0.014
    bottom_y = bottom_title_y
    column_centres = (0.285, 0.705)
    for col, (_position, record) in enumerate(zip(top_positions, records)):
        x = column_centres[col]
        fig.text(x, top_y, f"({chr(ord('a') + col)}) {record['variant']}", ha="center", va="bottom", fontsize=17, fontweight="bold")
        fig.text(x, top_y - 0.026, _record_metrics_text(record), ha="center", va="bottom", fontsize=13.5)
    for col, (_position, record) in enumerate(zip(bottom_positions, records)):
        x = column_centres[col]
        fig.text(x, bottom_y, f"({chr(ord('c') + col)}) {record['variant']}", ha="center", va="bottom", fontsize=17, fontweight="bold")
    fig.text(0.018, 0.70, "Ground-truth overlay", rotation=90, ha="center", va="center", fontsize=14, fontweight="bold")
    fig.text(0.018, 0.285, "Constraint-error visualization", rotation=90, ha="center", va="center", fontsize=14, fontweight="bold")

    overlay_map = cm.ScalarMappable(norm=overlay_norm, cmap=overlay_cmap)
    overlay_map.set_array([])
    cbar1 = fig.colorbar(overlay_map, ax=axes[0, :], orientation="vertical", fraction=0.015, pad=0.012)
    cbar1.set_label("Test-site error (km)", fontsize=11)
    cbar1.ax.tick_params(labelsize=9)
    edge_map = cm.ScalarMappable(norm=edge_norm, cmap=edge_cmap)
    edge_map.set_array([])
    cbar2 = fig.colorbar(edge_map, ax=axes[1, :], orientation="vertical", fraction=0.015, pad=0.012)
    cbar2.set_label("Distance-edge relative error", fontsize=11)
    cbar2.ax.tick_params(labelsize=9)
    _keep_labels_left_of_colorbar(fig, overlay_annotations, cbar1.ax)
    _keep_labels_left_of_colorbar(fig, error_annotations, cbar2.ax)
    fig.legend(
        handles=OVERLAY_HANDLES,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.025),
    )
    fig.legend(
        handles=NODE_HANDLES,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.002),
    )
    for suffix in ("png", "svg"):
        fig.savefig(output_stem.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_spatial_comparisons(
    *,
    as_dir: Path,
    bfgs_dir: Path,
    representative_dir: Path,
    outdir: Path,
) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    graph, vertices, dni, edges, distance_data = load_ini_data_from_csv(FILE_PATHS)
    del graph, edges
    display_names = _load_country_display_names(COUNTRY_NAMES_PATH, vertices)
    display_labels = {key: fill(label, width=22, break_long_words=False) for key, label in display_names.items()}
    as_config_path = as_dir / "progressive_config.json"
    bfgs_config_path = bfgs_dir / "bfgs_experiment_config.json"
    as_config = json.loads(as_config_path.read_text(encoding="utf-8"))
    bfgs_config = json.loads(bfgs_config_path.read_text(encoding="utf-8"))
    anchors = list(as_config["calibration_labels"])
    tests = list(as_config["test_labels"])
    anchor_align = str(as_config["anchor_align_label"])
    if anchors != list(bfgs_config["calibration_labels"]) or tests != list(bfgs_config["test_labels"]) or anchor_align != str(bfgs_config["anchor_align_label"]):
        raise ValueError("BFGS and progressive AS do not use the same calibration/test split.")

    progressive_selection_path = representative_dir / "representative_selection.json"
    progressive_selection = json.loads(progressive_selection_path.read_text(encoding="utf-8"))
    records = {row["variant"]: row for row in progressive_selection["selections"]}
    for variant in ("PhysicsSim-Full", "SMACOF", "DC-SMACOF"):
        if variant not in records:
            raise ValueError(f"Missing verified progressive representative: {variant}")
    bfgs_runs_path = bfgs_dir / "bfgs_runs_by_seed.csv"
    bfgs_positions_path = bfgs_dir / "bfgs_final_positions_y_up_sim.csv"
    bfgs_record = select_bfgs_representative(pd.read_csv(bfgs_runs_path, encoding="utf-8-sig"))
    records["BFGS"] = bfgs_record

    as_positions_path = as_dir / "progressive_final_positions_y_up_sim.csv"
    points_by_variant = {
        variant: _load_positions(as_positions_path, variant=variant, seed=int(records[variant]["seed"]), vertices=vertices)
        for variant in ("PhysicsSim-Full", "SMACOF", "DC-SMACOF")
    }
    points_by_variant["BFGS"] = _load_positions(
        bfgs_positions_path,
        variant=str(bfgs_record["source_variant"]),
        seed=int(bfgs_record["seed"]),
        vertices=vertices,
    )
    gt_lonlat = uploading_ground_truth(vertices, dni)
    targets = _target_positions_sim(dni, gt_lonlat, anchor_align, as_config.get("refer_pos_sim", refer_pos_sim))
    data_sim = data_Li2sim(distance_data)
    edge_errors = {variant: _distance_edge_errors(points, data_sim, dni) for variant, points in points_by_variant.items()}
    wrong_nodes = {variant: _wrong_direction_nodes(points, vertices, dni) for variant, points in points_by_variant.items()}
    overlay_errors = [
        float(np.linalg.norm(points[dni[label]] - np.asarray(targets[label], dtype=float)) / km2pix)
        for points in points_by_variant.values()
        for label in tests
    ]
    all_edge_errors = [error for rows in edge_errors.values() for *_pair, error in rows]
    overlay_norm = colors.Normalize(vmin=0.0, vmax=max(overlay_errors))
    edge_norm = colors.Normalize(vmin=0.0, vmax=max(float(np.quantile(all_edge_errors, 0.95)), 0.03))
    shared_overlay_extent = _combined_overlay_extent(points_by_variant, targets, dni, anchors, tests, pad_frac=0.075)

    groups = (
        ("figure_1a_physics_full_vs_bfgs_spatial_reconstruction", [records["PhysicsSim-Full"], records["BFGS"]]),
        ("figure_1b_smacof_vs_dc_smacof_spatial_reconstruction", [records["SMACOF"], records["DC-SMACOF"]]),
    )
    for stem, group_records in groups:
        _save_group_figure(
            records=group_records,
            points_by_variant=points_by_variant,
            targets=targets,
            vertices=vertices,
            display_labels=display_labels,
            dni=dni,
            anchors=anchors,
            tests=tests,
            edge_errors=edge_errors,
            wrong_nodes=wrong_nodes,
            overlay_extent=shared_overlay_extent,
            overlay_norm=overlay_norm,
            edge_norm=edge_norm,
            output_stem=outdir / stem,
        )

    selection_path = outdir / "figure_1_bfgs_representative_selection.json"
    selection_path.write_text(json.dumps(bfgs_record, ensure_ascii=False, indent=2), encoding="utf-8")
    plotted_positions = []
    for variant, points in points_by_variant.items():
        seed = int(records[variant]["seed"])
        plotted_positions.extend(
            {"variant": variant, "seed": seed, "label": label, "x_y_up_sim": float(points[index, 0]), "y_y_up_sim": float(points[index, 1])}
            for index, label in enumerate(vertices)
        )
    positions_path = outdir / "figure_1_plotted_positions_y_up_sim.csv"
    pd.DataFrame(plotted_positions).to_csv(positions_path, index=False, encoding="utf-8-sig")
    metadata = {
        "layout": "two separate 2x2 figures; overlay row above constraint-error row",
        "groups": [[record["variant"] for record in group] for _stem, group in groups],
        "representative_seeds": {variant: int(record["seed"]) for variant, record in records.items()},
        "representative_metrics": {variant: record["rerun_metrics"] for variant, record in records.items()},
        "overlay_extent_policy": "one shared equal-aspect extent across all four models",
        "error_map_extent_policy": "independent equal-aspect extent for topology readability",
        "coordinates": "formal y-up simulation coordinates; overlay errors reported in LCC kilometres",
        "country_display_names": {
            "source": "data/country_display_names_yu2006.csv",
            "sha256": _sha256(COUNTRY_NAMES_PATH),
        },
        "source_sha256": {
            str(path.resolve()): _sha256(path)
            for path in (as_config_path, bfgs_config_path, progressive_selection_path, as_positions_path, bfgs_runs_path, bfgs_positions_path)
        },
    }
    metadata_path = outdir / "figure_1_spatial_comparison_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"records": records, "points": points_by_variant, "metadata": metadata}


if __name__ == "__main__":
    from scripts.build_manuscript_ready_results import AS_DIR, BFGS_DIR, REPRESENTATIVE_DIR

    create_spatial_comparisons(
        as_dir=AS_DIR,
        bfgs_dir=BFGS_DIR,
        representative_dir=REPRESENTATIVE_DIR,
        outdir=PROJECT_ROOT / "outputs" / "manuscript_spatial_comparisons",
    )
