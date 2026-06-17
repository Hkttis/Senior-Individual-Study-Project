"""run_paper_script.ch6_visualize_single_model

Chapter 6 — Figure generation utilities.

This script produces the *single-model* visualizations used in the paper:

  1) Stress convergence curve                (plot_stress_convergence_log)
  # tmp not ot draw stress convergence curve here 
  2) Distance-error heatmap (full & zoomed)  (visualize_error_map_official)
  3) Ground-truth overlay & RMSE             (ground_truth_comparison)
  4) Force heatmap (scalar magnitude sum)    (plot_force_heatmap_scalar_sum)
  5) Node-link diagram                       (draw_node_link_pygame)

Supported models
----------------
  - PhysicsSim            : run the physics simulation once (seeded)
  - StressMajorization    : run baseline (vis=False) and postprocess
  - DirectedMDS           : run baseline (vis=False) and postprocess

Usage
-----
Run from the physics_simulation project root.
Default fixed anchors come from data/site_rmse_points.csv (use_role=anchor).
Use --no-wait for non-interactive smoke tests.

python -m run_paper_script.paper_run ch6-visualize --model PhysicsSim --seed 0 --no-wait
python -m run_paper_script.paper_run ch6-visualize --model SMACOF --no-wait
python -m run_paper_script.paper_run ch6-visualize --model DC-SMACOF --no-wait
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from typing import List

import numpy as np

from library.config import FILE_PATHS, refer_pos_screen, refer_pos_sim, refer_pos
from library.data_io import load_ini_data_from_csv, uploading_ground_truth, uploading_directional_data, save_vis_data, save_err_data, get_anchor_labels, get_anchor_align_label
from library.units import data_Li2sim
from library.coordinates import flipping_y
from library.visualization import (
    plot_stress_convergence_log,
    visualize_error_map_official,
    ground_truth_comparison,
    plot_force_heatmap_scalar_sum,
)

from library.metrics import alignment_and_scaling, procrustes_align_by_fixed_points
from MDS_model.plot_node_link_diagram import draw_node_link_pygame, wrong_directions_nonflip


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["PhysicsSim", "SMACOF", "DC-SMACOF"],
    )
    p.add_argument("--seed", type=int, default=0, help="Only used for PhysicsSim")
    p.add_argument(
        "--fixed",
        type=str,
        default="",
        help="Comma-separated anchor labels (must exist in ground truth).",
    )
    p.add_argument(
        "--zoom",
        type=str,
        default="500,325,800,400",
        help="Zoom window for error-map as 'x_min,y_min,x_max,y_max' (pixels).",
    )
    p.add_argument("--skip-heatmap", action="store_true", help="Do not display force heatmap")
    p.add_argument("--skip-nodelink", action="store_true", help="Do not display node-link diagram")
    p.add_argument("--no-wait", action="store_true", help="Save figures and exit without waiting for windows to close")
    return p.parse_args()


def _parse_zoom(s: str):
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) != 4:
        raise ValueError("--zoom must have 4 numbers: x_min,y_min,x_max,y_max")
    return tuple(parts)


def _run_physics(seed: int, fixed_point_labels: List[str]):
    from library.initialization import generate_CHEN_initial_positions
    from library.physics import main_physics_simulation
    from library.config import SPRING_STIFFNESS_BASE, REPULSION_STRENGTH_BASE, DIRECTIONAL_FORCE_MAGNITUDE_BASE

    _graph, vertice, dni, edges, data = load_ini_data_from_csv(FILE_PATHS)
    gt_lonlat = uploading_ground_truth(vertice, dni)
    directional_data = uploading_directional_data()
    fixed_points_lonlat = [tuple(gt_lonlat[dni[name]]) for name in fixed_point_labels]

    np.random.seed(seed)
    vertice, dni, data, pos_matrix, fixed_positions_list = generate_CHEN_initial_positions(
        deepcopy(refer_pos_sim), fixed_point_labels, fixed_points_lonlat
    )

    wrong_direction_lists, stress_history, pos_history, pos_final_y_up = main_physics_simulation(
        vertice,
        dni,
        data_Li2sim(data),
        pos_matrix,
        directional_data,
        fixed_positions_list,
        SPRING_STIFFNESS_BASE,
        REPULSION_STRENGTH_BASE,
        DIRECTIONAL_FORCE_MAGNITUDE_BASE,
        plot=False,
    )

    pos_final_px = flipping_y(pos_final_y_up)
    return vertice, dni, edges, data, gt_lonlat, directional_data, wrong_direction_lists, stress_history, pos_final_px


def _run_baseline(model: str, fixed_point_labels: List[str]):
    from library.model_cmp import run_stress_majorization, run_directed_MDS

    graph, vertice, dni, edges, data = load_ini_data_from_csv(FILE_PATHS)
    gt_lonlat = uploading_ground_truth(vertice, dni)
    fixed_points_lonlat = [tuple(gt_lonlat[dni[name]]) for name in fixed_point_labels]

    if model == "SMACOF":
        pos_hist_li = run_stress_majorization(vis=False)
        pos_last_li = deepcopy(pos_hist_li[-1])
        wrong_direction_lists = wrong_directions_nonflip(pos_last_li, vertice, dni)
        pos_px = alignment_and_scaling(pos_last_li, vertice, dni, refer_pos=list(refer_pos_sim), y_down=False)
        pos_px = procrustes_align_by_fixed_points(deepcopy(pos_px), fixed_point_labels, fixed_points_lonlat, dni, refer_pos = refer_pos_sim)
        stress_history = []
    else:
        pos_hist_li = run_directed_MDS(vis=False)
        pos_last_li = deepcopy(pos_hist_li[-1])
        wrong_direction_lists = wrong_directions_nonflip(pos_last_li, vertice, dni)
        pos_px = alignment_and_scaling(pos_last_li, vertice, dni, refer_pos=list(refer_pos_sim), y_down=False)
        stress_history = []

    directional_data = uploading_directional_data()
    return vertice, dni, edges, data, gt_lonlat, directional_data, wrong_direction_lists, stress_history, pos_px


def main() -> None:
    args = _parse_args()
    fixed_point_labels = [x.strip() for x in args.fixed.split(",") if x.strip()] or get_anchor_labels()
    anchor_align_label = get_anchor_align_label()
    zoom_area = _parse_zoom(args.zoom)

    if args.model == "PhysicsSim":
        vertice, dni, edges, data, gt_lonlat, directional_data, wrong_dir, stress_hist, pos_px = _run_physics(
            seed=args.seed, fixed_point_labels=fixed_point_labels
        )
        file_prefix = f"PhysicsSim_seed{args.seed}_"
        # plot_stress_convergence_log(stress_hist, file_name=file_prefix)
    else:
        vertice, dni, edges, data, gt_lonlat, directional_data, wrong_dir, _stress_hist, pos_px = _run_baseline(
            args.model, fixed_point_labels
        )
        pos_px = flipping_y(pos_px)
        file_prefix = f"{args.model}_"

    # 2) Error map (full + zoom)
    errors, edge_labels = visualize_error_map_official(
        deepcopy(pos_px), vertice, dni, data, wrong_dir, zoom_area=None, file_name=file_prefix, wait=not args.no_wait
    )
    visualize_error_map_official(
        deepcopy(pos_px), vertice, dni, data, wrong_dir, zoom_area=zoom_area, file_name=file_prefix, wait=not args.no_wait
    )

    # 3) GT overlay
    ground_truth_comparison(
        vertice,
        dni,
        data_Li2sim(data),
        deepcopy(gt_lonlat),
        pos_px[dni[anchor_align_label]],
        deepcopy(pos_px),
        file_name=file_prefix,
        wait=not args.no_wait,
    )
    # 4) Force heatmap (recommended for physics)
    if not args.skip_heatmap:
        plot_force_heatmap_scalar_sum(
            pos_matrix=deepcopy(pos_px),
            vertice=vertice,
            dni=dni,
            data=data,
            directional_data=[(r[0], r[1], r[2]) for r in directional_data],
            canvas_size=(1200, 750),
            sigma_px=28.0,
            show_points=True,
            window_caption=f"Force Heatmap ({args.model})",
            wait=not args.no_wait,
        )
    # 5) Node-link diagram 
    if not args.skip_nodelink:
        draw_node_link_pygame(
            pos=[(float(x), float(y)) for x, y in pos_px],
            vertice=vertice,
            edges=edges,
            directed=False,
            caption=f"Node-Link ({args.model})",
            wait=not args.no_wait,
        )

    # 6) Save data for interactive map
    # The map/export pipeline expects *north-up* (y-up) pixel coordinates.
    # Our publication visuals use pygame coords (y-down), so convert back.
    pos_y_up = flipping_y(deepcopy(pos_px))
    save_vis_data(vertice, dni, deepcopy(pos_y_up), deepcopy(gt_lonlat), deepcopy(refer_pos_sim))
    save_err_data(
        vertice,
        dni,
        deepcopy(pos_y_up),
        deepcopy(gt_lonlat),
        deepcopy(refer_pos_sim),
        errors,
        edge_labels,
    )


if __name__ == "__main__":
    main()
