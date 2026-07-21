"""Run final PhysicsSim-Full reconstruction with all verified sites fixed.

This experiment is intended for the final Western Han country configuration:
all archaeologically verified locations in data/site_rmse_points.csv are used
as hard anchors during the PhysicsSim-Full simulation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
(PROJECT_ROOT / ".matplotlib").mkdir(exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors

from library.anchor_frame import px_list_to_km_list
from library.config import (
    DIRECTIONAL_FORCE_MAGNITUDE_BASE,
    FILE_PATHS,
    REPULSION_STRENGTH_BASE,
    SPRING_STIFFNESS_BASE,
    km2pix,
    refer_pos_sim,
)
from library.data_io import (
    get_anchor_align_label,
    load_ini_data_from_csv,
    load_site_points,
    uploading_directional_data,
    uploading_ground_truth,
)
from library.geometry import (
    get_lcc_bounds,
    get_lcc_parameters,
    inverse_lcc_transformation,
)
from library.initialization import generate_CHEN_initial_positions
from library.metrics import (
    calculate_kruskals_stress,
    direction_violation_rate,
    mean_angular_error_violations,
)
from library.physics import main_physics_simulation
from library.units import data_Li2sim, pos_matrix_sim2km
from MDS_model.plot_node_link_diagram import wrong_directions_nonflip
from run_paper_script.ch5_ablation_study import _load_selected_hpo_params
from run_paper_script.ch5_hparam_kfold_gridsearch_pareto import _weights_from_alpha_beta
from run_paper_script.ch5_ablation_progressive import _target_positions_sim
from scripts.create_section_6_5_visual_prototype import (
    NODE_HANDLES,
    OVERLAY_HANDLES,
    _annotate_entries,
    _cjk_font,
    _combined_overlay_extent,
    _distance_edge_errors,
    _draw_error_map,
    _draw_overlay,
    _format_metrics,
    _panel_extent,
    _relax_annotations,
    _style_axis,
    _wrong_direction_nodes,
)


DEFAULT_HPO_OUTDIR = PROJECT_ROOT / "outputs" / "ch5_hparam_anchor_loo_grid_lcc_sitebounds_36x10_run2_manual_alpha_1_beta_-0.5"
DEFAULT_OUTDIR = PROJECT_ROOT / "outputs" / "final_reconstruction_all_verified_sites_alpha_1_beta_-0.5_seed0"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hpo-outdir", default=str(DEFAULT_HPO_OUTDIR))
    parser.add_argument("--alpha", type=float, default=None, help="Override PhysicsSim alpha. Default reads --hpo-outdir.")
    parser.add_argument("--beta", type=float, default=None, help="Override PhysicsSim beta. Default reads --hpo-outdir.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def _all_verified_site_labels(dni: dict[str, int]) -> tuple[list[str], list[tuple[float, float]], list[dict]]:
    labels: list[str] = []
    lonlat: list[tuple[float, float]] = []
    rows_out: list[dict] = []
    seen: set[str] = set()
    for row in load_site_points():
        label = row["name"]
        if label in seen:
            raise ValueError(f"Duplicate verified site label: {label}")
        seen.add(label)
        if label not in dni:
            raise ValueError(f"Verified site is not in model graph: {label}")
        lon = float(row["lon"])
        lat = float(row["lat"])
        labels.append(label)
        lonlat.append((lon, lat))
        rows_out.append({"label": label, "lon": lon, "lat": lat, "use_role": row["use_role"]})
    if not labels:
        raise ValueError("No verified sites found in site_rmse_points.csv")
    return labels, lonlat, rows_out


def _record_for_metrics(metrics: dict) -> dict:
    return {
        "variant": "PhysicsSim-Full-AllVerifiedAnchors",
        "rerun_metrics": metrics,
    }


def _save_configuration(
    *,
    outdir: Path,
    vertice: list[str],
    dni: dict[str, int],
    points: np.ndarray,
    gt_lonlat,
    verified_rows: list[dict],
    anchor_label: str,
) -> Path:
    verified_by_label = {row["label"]: row for row in verified_rows}
    points_km = px_list_to_km_list(points.tolist(), tuple(refer_pos_sim), km2pix)
    anchor_lonlat = gt_lonlat[dni[anchor_label]]
    points_lonlat = inverse_lcc_transformation(points_km, anchor_lonlat)
    rows = []
    for i, label in enumerate(vertice):
        gt = gt_lonlat[dni[label]]
        is_verified = label in verified_by_label
        row = {
            "label": label,
            "x_y_up_sim": float(points[i, 0]),
            "y_y_up_sim": float(points[i, 1]),
            "x_anchor_km": float(points_km[i][0]),
            "y_anchor_km": float(points_km[i][1]),
            "lon_reconstructed": points_lonlat[i][0],
            "lat_reconstructed": points_lonlat[i][1],
            "is_verified_site_anchor": bool(is_verified),
            "use_role": verified_by_label.get(label, {}).get("use_role", ""),
            "ground_truth_lon": gt[0] if gt != [0, 0] else "",
            "ground_truth_lat": gt[1] if gt != [0, 0] else "",
        }
        rows.append(row)
    path = outdir / "final_country_configuration.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _save_verified_site_errors(
    *,
    outdir: Path,
    points: np.ndarray,
    dni: dict[str, int],
    targets: dict[str, np.ndarray],
    verified_labels: list[str],
) -> tuple[Path, dict]:
    rows = []
    errors = []
    for label in verified_labels:
        err = float(np.linalg.norm(points[dni[label]] - targets[label]) / km2pix)
        rows.append({"label": label, "fixed_site_error_km": err})
        errors.append(err)
    metrics = {
        "fixed_site_RMSE_km": float(np.sqrt(np.mean(np.square(errors)))),
        "fixed_site_MAE_km": float(np.mean(errors)),
        "fixed_site_max_error_km": float(np.max(errors)),
        "n_fixed_verified_sites": int(len(verified_labels)),
    }
    path = outdir / "verified_site_anchor_errors.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path, metrics


def _save_visualizations(
    *,
    outdir: Path,
    vertice: list[str],
    dni: dict[str, int],
    points: np.ndarray,
    targets: dict[str, np.ndarray],
    verified_labels: list[str],
    data_sim,
    distance_data,
    metrics: dict,
    dpi: int,
) -> dict:
    record = _record_for_metrics(metrics)
    overlay_errors = [
        float(np.linalg.norm(points[dni[label]] - np.asarray(targets[label], dtype=float)) / km2pix)
        for label in verified_labels
    ]
    overlay_norm = colors.Normalize(vmin=0.0, vmax=max(overlay_errors) if overlay_errors else 1.0)
    overlay_cmap = plt.get_cmap("plasma")
    edge_errors = _distance_edge_errors(points, data_sim, dni)
    edge_values = [err for *_ij, err in edge_errors]
    edge_norm = colors.Normalize(vmin=0.0, vmax=max(float(np.quantile(edge_values, 0.95)), 0.03))
    edge_cmap = plt.get_cmap("RdYlGn_r")
    wrong_nodes = _wrong_direction_nodes(points, vertice, dni)

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 13.2), constrained_layout=True)
    overlay_font = _cjk_font(9.2)
    map_font = _cjk_font(8.0)
    overlay_annotations = _draw_overlay(
        axes[0], points, targets, record, vertice, dni, [], verified_labels,
        overlay_norm, overlay_cmap, overlay_font, draw_title=False,
    )
    axes[0].set_title(f"(a) PhysicsSim-Full with all verified sites fixed\n{_format_metrics(record)}", fontsize=13, fontweight="bold")
    _style_axis(axes[0], _combined_overlay_extent({"final": points}, targets, dni, [], verified_labels, pad_frac=0.075))
    _relax_annotations(fig, axes[0], overlay_annotations, iterations=130, max_offset=52.0)

    map_annotations = _draw_error_map(
        axes[1], points, edge_errors, wrong_nodes, edge_norm, edge_cmap, vertice, map_font, clip_labels=False,
    )
    axes[1].set_title("(b) Constraint-error visualization", fontsize=13, fontweight="bold")
    _style_axis(axes[1], _panel_extent(points, [], pad_frac=0.16))
    _relax_annotations(fig, axes[1], map_annotations, iterations=260, max_offset=62.0)

    fig.text(0.018, 0.735, "Ground-truth overlay", rotation=90, ha="center", va="center", fontsize=12, fontweight="bold")
    fig.text(0.018, 0.285, "Constraint-error visualization", rotation=90, ha="center", va="center", fontsize=12, fontweight="bold")
    fig.legend(handles=OVERLAY_HANDLES[:3] + NODE_HANDLES, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.005))

    sm_overlay = cm.ScalarMappable(norm=overlay_norm, cmap=overlay_cmap)
    sm_overlay.set_array([])
    cbar1 = fig.colorbar(sm_overlay, ax=axes[0], orientation="vertical", fraction=0.025, pad=0.01)
    cbar1.set_label("Verified-site error (km)", fontsize=9)
    cbar1.ax.tick_params(labelsize=8)

    sm_edge = cm.ScalarMappable(norm=edge_norm, cmap=edge_cmap)
    sm_edge.set_array([])
    cbar2 = fig.colorbar(sm_edge, ax=axes[1], orientation="vertical", fraction=0.025, pad=0.01)
    cbar2.set_label("Distance-edge relative error", fontsize=9)
    cbar2.ax.tick_params(labelsize=8)

    integrated_png = outdir / "final_reconstruction_overlay_and_error_map.png"
    integrated_svg = outdir / "final_reconstruction_overlay_and_error_map.svg"
    fig.savefig(integrated_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(integrated_svg, bbox_inches="tight")
    plt.close(fig)

    # Also save the two visualization types separately for manuscript layout flexibility.
    separate_paths = {}
    for kind, row_idx, filename in (
        ("overlay", 0, "final_reconstruction_ground_truth_overlay"),
        ("error_map", 1, "final_reconstruction_constraint_error_map"),
    ):
        fig_single, ax_single = plt.subplots(1, 1, figsize=(10.5, 7.2), constrained_layout=True)
        if kind == "overlay":
            anns = _draw_overlay(
                ax_single, points, targets, record, vertice, dni, [], verified_labels,
                overlay_norm, overlay_cmap, overlay_font, draw_title=False,
            )
            ax_single.set_title(f"PhysicsSim-Full with all verified sites fixed\n{_format_metrics(record)}", fontsize=13, fontweight="bold")
            _style_axis(ax_single, _combined_overlay_extent({"final": points}, targets, dni, [], verified_labels, pad_frac=0.075))
            _relax_annotations(fig_single, ax_single, anns, iterations=130, max_offset=52.0)
            cbar = fig_single.colorbar(sm_overlay, ax=ax_single, orientation="vertical", fraction=0.025, pad=0.01)
            cbar.set_label("Verified-site error (km)", fontsize=9)
            fig_single.legend(handles=OVERLAY_HANDLES[:3], loc="lower center", ncol=3, frameon=False, fontsize=8)
        else:
            anns = _draw_error_map(ax_single, points, edge_errors, wrong_nodes, edge_norm, edge_cmap, vertice, map_font, clip_labels=False)
            ax_single.set_title("PhysicsSim-Full constraint-error visualization", fontsize=13, fontweight="bold")
            _style_axis(ax_single, _panel_extent(points, [], pad_frac=0.16))
            _relax_annotations(fig_single, ax_single, anns, iterations=260, max_offset=62.0)
            cbar = fig_single.colorbar(sm_edge, ax=ax_single, orientation="vertical", fraction=0.025, pad=0.01)
            cbar.set_label("Distance-edge relative error", fontsize=9)
            fig_single.legend(handles=NODE_HANDLES, loc="lower center", ncol=2, frameon=False, fontsize=8)
        png = outdir / f"{filename}.png"
        svg = outdir / f"{filename}.svg"
        fig_single.savefig(png, dpi=dpi, bbox_inches="tight")
        fig_single.savefig(svg, bbox_inches="tight")
        plt.close(fig_single)
        separate_paths[f"{kind}_png"] = str(png)
        separate_paths[f"{kind}_svg"] = str(svg)

    return {
        "integrated_png": str(integrated_png),
        "integrated_svg": str(integrated_svg),
        **separate_paths,
    }


def main() -> None:
    args = _parse_args()
    outdir = Path(args.outdir)
    if outdir.exists() and any(outdir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {outdir}. Use --overwrite intentionally.")
    outdir.mkdir(parents=True, exist_ok=True)

    alpha_beta = None
    if args.alpha is None or args.beta is None:
        alpha_beta = _load_selected_hpo_params(args.hpo_outdir)
    alpha = float(args.alpha if args.alpha is not None else alpha_beta[0])
    beta = float(args.beta if args.beta is not None else alpha_beta[1])

    np.random.seed(args.seed)
    graph, vertice, dni, edges, distance_data = load_ini_data_from_csv(FILE_PATHS)
    del graph, edges
    gt_lonlat = uploading_ground_truth(vertice, dni)
    anchor_label = get_anchor_align_label()
    verified_labels, verified_lonlat, verified_rows = _all_verified_site_labels(dni)
    if anchor_label not in verified_labels:
        raise ValueError(f"Frame anchor {anchor_label!r} is not in verified site labels.")

    vertice0, dni0, data_li, initial, fixed_positions = generate_CHEN_initial_positions(
        list(refer_pos_sim), verified_labels, verified_lonlat, anchor_label=anchor_label
    )
    if vertice0 != vertice or dni0 != dni:
        raise ValueError("Initialization graph order does not match load_ini_data_from_csv().")

    w_dir, w_reg, spring, directional, repulsion = _weights_from_alpha_beta(
        alpha,
        beta,
        1.0,
        SPRING_STIFFNESS_BASE,
        DIRECTIONAL_FORCE_MAGNITUDE_BASE,
        REPULSION_STRENGTH_BASE,
    )
    data_sim = data_Li2sim(data_li)
    directional_data = uploading_directional_data()
    wrong_dir, stress_history, pos_history, final = main_physics_simulation(
        vertice,
        dni,
        data_sim,
        initial,
        directional_data,
        fixed_positions,
        spring,
        repulsion,
        directional,
        plot=False,
    )
    points = np.asarray(final, dtype=float)
    targets = _target_positions_sim(dni, gt_lonlat, anchor_label, refer_pos_sim)
    fixed_error_path, fixed_metrics = _save_verified_site_errors(
        outdir=outdir,
        points=points,
        dni=dni,
        targets=targets,
        verified_labels=verified_labels,
    )
    metrics = {
        "RMSE_test_km": fixed_metrics["fixed_site_RMSE_km"],
        "E_distance_stress": float(calculate_kruskals_stress(dni, pos_matrix_sim2km(points.tolist()), data_sim)),
        "E_direction_vr": float(direction_violation_rate(points, directional_data, dni)),
        "E_direction_mae": float(mean_angular_error_violations(points, directional_data, dni)),
        **fixed_metrics,
    }
    config = {
        "experiment": "PhysicsSim-Full final reconstruction with all verified sites fixed",
        "seed": int(args.seed),
        "hpo_outdir": str(args.hpo_outdir),
        "alpha": alpha,
        "beta": beta,
        "w_dis": 1.0,
        "w_dir": w_dir,
        "w_reg": w_reg,
        "spring_stiffness": spring,
        "directional_force": directional,
        "repulsion_strength": repulsion,
        "anchor_align_label": anchor_label,
        "fixed_verified_site_labels": verified_labels,
        "lcc_bounds": dict(zip(("lon_min", "lon_max", "lat_min", "lat_max"), map(float, get_lcc_bounds()))),
        "lcc_parameters": dict(zip(("lat_1", "lat_2", "lon_0"), map(float, get_lcc_parameters()))),
        "metrics": metrics,
    }
    config_path = outdir / "final_reconstruction_config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics_path = outdir / "final_reconstruction_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    config_csv = _save_configuration(
        outdir=outdir,
        vertice=vertice,
        dni=dni,
        points=points,
        gt_lonlat=gt_lonlat,
        verified_rows=verified_rows,
        anchor_label=anchor_label,
    )
    visual_paths = _save_visualizations(
        outdir=outdir,
        vertice=vertice,
        dni=dni,
        points=points,
        targets=targets,
        verified_labels=verified_labels,
        data_sim=data_sim,
        distance_data=distance_data,
        metrics=metrics,
        dpi=args.dpi,
    )
    pd.DataFrame({
        "iteration": list(range(1, len(stress_history) + 1)),
        "stress_history": stress_history,
    }).to_csv(outdir / "stress_history.csv", index=False, encoding="utf-8-sig")

    print(f"[Saved] {config_csv}")
    print(f"[Saved] {fixed_error_path}")
    print(f"[Saved] {config_path}")
    print(f"[Saved] {metrics_path}")
    for path in visual_paths.values():
        print(f"[Saved] {path}")


if __name__ == "__main__":
    main()
