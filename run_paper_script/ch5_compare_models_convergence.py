"""scripts.ch5_compare_models_convergence

Chapter 5 — Baseline comparison experiment.

Runs:
  1) Physics simulation (our method)
  2) DirectedMDS baseline
  3) StressMajorization baseline

Then visualizes convergence (stress + RMSE bands) using:
  library.visualization.plot_three_model_convergence_pygame_pixelaware

Usage
-----
python -m scripts.ch5_compare_models_convergence --seed 0
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from typing import List

import numpy as np

from library.config import (
    FILE_PATHS,
    height,
    SPRING_STIFFNESS_BASE,
    REPULSION_STRENGTH_BASE,
    DIRECTIONAL_FORCE_MAGNITUDE_BASE,
)
from library.data_io import load_ini_data_from_csv, uploading_ground_truth, uploading_directional_data
from library.initialization import generate_CHEN_initial_positions
from library.units import data_Li2sim
from library.physics import main_physics_simulation
from library.visualization import plot_three_model_convergence_pygame_pixelaware


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--fixed",
        type=str,
        default="鄯善,都護治/烏壘",
        help="Comma-separated anchor labels (must exist in ground truth).",
    )
    p.add_argument(
        "--refer-pos",
        type=str,
        default="600,500",
        help="Screen anchor pixel, as 'x,y' (y-down).",
    )
    return p.parse_args()


def _parse_xy(s: str) -> List[float]:
    xs, ys = s.split(",")
    return [float(xs.strip()), float(ys.strip())]


def main() -> None:
    args = _parse_args()
    fixed_point_labels = [x.strip() for x in args.fixed.split(",") if x.strip()]
    _refer_pos_screen = _parse_xy(args.refer_pos)

    graph, vertice, dni, edges, data = load_ini_data_from_csv(FILE_PATHS)
    gt_lonlat = uploading_ground_truth(vertice, dni)
    fixed_points_lonlat = [tuple(gt_lonlat[dni[name]]) for name in fixed_point_labels]

    # 1) Physics simulation history (px, y-up internally)
    np.random.seed(args.seed)
    directional_data = uploading_directional_data()
    # init expects y-up reference point
    _refer_pos_sim = [float(_refer_pos_screen[0]), float(height) - float(_refer_pos_screen[1])]
    _vertice, _dni, _data, pos_matrix, fixed_positions_list = generate_CHEN_initial_positions(
        deepcopy(_refer_pos_sim), fixed_point_labels, fixed_points_lonlat
    )
    _, _, pos_hist_ph_px, _ = main_physics_simulation(
        vertice,
        dni,
        data_Li2sim(data),
        pos_matrix,
        directional_data,
        fixed_positions_list,
        spring_stiffness=SPRING_STIFFNESS_BASE,
        repulsion_strength=REPULSION_STRENGTH_BASE,
        directional_force_magnitude=DIRECTIONAL_FORCE_MAGNITUDE_BASE,
        plot=False,
    )

    # 2) Baseline histories (Li units)
    from library.model_cmp import run_stress_majorization, run_directed_MDS

    pos_hist_sm_li = run_stress_majorization(vis=False)
    pos_hist_dm_li = run_directed_MDS(vis=False)

    # 3) Convergence comparison figure
    plot_three_model_convergence_pygame_pixelaware(
        pos_hist_ph_px,
        pos_hist_dm_li,
        pos_hist_sm_li,
        vertice=vertice,
        dni=dni,
        data=data,
        ground_truth_positions=gt_lonlat,
        fixed_point_labels=fixed_point_labels,
        fixed_point_lonlat=fixed_points_lonlat,
        refer_pos=tuple(_refer_pos_screen),
        orientation="north-up",
        pre_process=False,
    )


if __name__ == "__main__":
    main()
