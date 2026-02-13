"""scripts.paper_run

Run one end-to-end round of the workflow described in the paper.

Usage
-----
python -m scripts.paper_run
python -m scripts.paper_run --compare_models
"""

#TODO 再執行一次， y 軸不對、方向不對

from __future__ import annotations

import argparse
from copy import deepcopy
from typing import List
from math import sqrt

from library.config import (
    FILE_PATHS,
    refer_pos as DEFAULT_REFER_POS,
    refer_pos_sim, 
    refer_pos_screen,
    SPRING_STIFFNESS_BASE,
    REPULSION_STRENGTH_BASE,
    DIRECTIONAL_FORCE_MAGNITUDE_BASE,
)
from library.units import data_Li2sim
from library.data_io import (
    load_ini_data_from_csv,
    uploading_ground_truth,
    uploading_directional_data,
    save_vis_data,
    save_err_data,
)
from library.initialization import generate_CHEN_initial_positions
from library.physics import main_physics_simulation
from library.visualization import (
    plot_stress_convergence_log,
    visualize_error_map_official,
    ground_truth_comparison,
    plot_three_model_convergence_pygame_pixelaware,
)
from library.coordinates import flipping_y


def run_one_round(
    *,
    refer_pos: List[float],
    fixed_point_labels: List[str],
    seed: int = None,
    plot_simulation: bool = False,
    compare_models: bool = False,
) -> None:
    graph, vertice, dni, edges, data = load_ini_data_from_csv(FILE_PATHS)

    gt_lonlat = uploading_ground_truth(vertice, dni)
    fixed_points_lonlat = [tuple(gt_lonlat[dni[name]]) for name in fixed_point_labels]

    directional_data = uploading_directional_data()

    import numpy as np
    np.random.seed(seed)

    vertice, dni, data, pos_matrix, fixed_positions_list = generate_CHEN_initial_positions(
        deepcopy(refer_pos_sim), fixed_point_labels, fixed_points_lonlat
    )

    wrong_direction_lists, stress_history, pos_history, pos_final = main_physics_simulation(
        vertice, dni, data_Li2sim(data), pos_matrix, directional_data, fixed_positions_list,
        SPRING_STIFFNESS_BASE, REPULSION_STRENGTH_BASE, DIRECTIONAL_FORCE_MAGNITUDE_BASE,
        plot=plot_simulation,
    )

    pos_final = flipping_y(pos_matrix)

    #'''
    plot_stress_convergence_log(stress_history, file_name="PhysicsSim_")

    errors, edge_labels = visualize_error_map_official(
        deepcopy(pos_final), vertice, dni, data, wrong_direction_lists,
        zoom_area=None, file_name="PhysicsSim_",
    )

    visualize_error_map_official(
        deepcopy(pos_final), vertice, dni, data, wrong_direction_lists,
        zoom_area=(500, 325, 800, 400), file_name="PhysicsSim_",
    )

    ground_truth_comparison(
        vertice, dni, data_Li2sim(data), deepcopy(gt_lonlat),
        deepcopy(refer_pos_screen), deepcopy(pos_final),
        file_name="PhysicsSim_",
    )

    save_vis_data(vertice, dni, deepcopy(pos_final), deepcopy(gt_lonlat), deepcopy(refer_pos_sim))
    save_err_data(vertice, dni, deepcopy(pos_final), deepcopy(gt_lonlat), deepcopy(refer_pos_sim), errors, edge_labels)

    if compare_models:
        from library.model_cmp import run_stress_majorization, run_directed_MDS

        pos_hist_sm_li = run_stress_majorization(vis=False)
        pos_hist_dm_li = run_directed_MDS(vis=False)
        pos_hist_ph_px = pos_history

        plot_three_model_convergence_pygame_pixelaware(
            pos_hist_ph_px, pos_hist_dm_li, pos_hist_sm_li,
            vertice=vertice, dni=dni, data=data,
            ground_truth_positions=gt_lonlat,
            fixed_point_labels=fixed_point_labels,
            fixed_point_lonlat=fixed_points_lonlat,
            refer_pos=tuple(refer_pos_sim),
            pre_process=False,
        )
    #'''


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compare_models", action="store_true")
    p.add_argument("--plot", action="store_true", help="Show the physics simulation window")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_one_round(
        refer_pos=list(DEFAULT_REFER_POS),
        fixed_point_labels=["鄯善", "都護治/烏壘"],
        seed=args.seed,
        plot_simulation=args.plot,
        compare_models=args.compare_models,
    )
