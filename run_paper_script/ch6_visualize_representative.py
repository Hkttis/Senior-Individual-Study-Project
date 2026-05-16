"""run_paper_script.ch6_visualize_representative

Load 100-run CSV histories, select representative run per model,
then produce 6.2 convergence / 6.3 error-map / 6.4 overlay figures.

Usage
-----
python -m run_paper_script.paper_run ch6-representative or

python -m run_paper_script.paper_run ch6-representative `
   --csv-sm "C:\\Users\\hktti\\Desktop\\project\\results_data_copy\\all_pos_sm_px_data 100_runs_DCandDatafixed_0405.csv" `
   --csv-dm "C:\\Users\\hktti\\Desktop\\project\\results_data_copy\\all_pos_dm_px_data 100_runs_DCandDatafixed_0405.csv" `
   --csv-ph "C:\\Users\\hktti\\Desktop\\project\\results_data_copy\\all_pos_sm_ph_data 100_runs_DCandDatafixed_0405.csv"
"""

from __future__ import annotations
import argparse, json
from copy import deepcopy
from pathlib import Path
import numpy as np

from library.config import FILE_PATHS, refer_pos, refer_pos_sim, refer_pos_screen, OUTPUT_DIR
from library.data_io import (
    load_ini_data_from_csv, uploading_ground_truth,
    uploading_directional_data, _read_model_csv,
)
from library.units import data_Li2sim
from library.coordinates import flipping_y
from library.metrics import alignment_and_scaling, procrustes_align_by_fixed_points
from library.model_cmp import select_representative_run
from library.visualization import (
    plot_three_model_convergence_pygame_pixelaware,
    plot_three_model_direction_convergence,
    visualize_error_map_official,
    ground_truth_comparison,
)
from MDS_model.plot_node_link_diagram import wrong_directions_nonflip


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fixed", type=str, default="鄯善,都護治/烏壘")
    p.add_argument("--csv-sm", type=str, default=FILE_PATHS["save_all_pos_sm_px_data"])
    p.add_argument("--csv-dm", type=str, default=FILE_PATHS["save_all_pos_dm_px_data"])
    p.add_argument("--csv-ph", type=str, default=FILE_PATHS["save_all_pos_ph_px_data"])
    p.add_argument("--skip-convergence", action="store_true")
    p.add_argument("--skip-errormap", action="store_true")
    p.add_argument("--skip-overlay", action="store_true")
    return p.parse_args()

def _flip_for_display_keep_anchor(pos_yup, anchor_idx, target_anchor_xy):
    """
    將 y-up 座標翻成畫圖用座標，但保持 anchor 固定在 target_anchor_xy。
    只做翻轉與平移，不做任何縮放。
    """
    tx, ty = float(target_anchor_xy[0]), float(target_anchor_xy[1])

    # 先繞 y = ty 這條水平線翻轉，而不是繞整個 screen height
    flipped = [[float(x), 2.0 * ty - float(y)] for x, y in pos_yup]

    # 保險起見，再做一次平移，確保 anchor 精準落在 target_anchor_xy
    dx = tx - flipped[anchor_idx][0]
    dy = ty - flipped[anchor_idx][1]

    return [[x + dx, y + dy] for x, y in flipped]

def main():
    args = _parse_args()
    fixed_labels = [x.strip() for x in args.fixed.split(",") if x.strip()]

    graph, vertice, dni, edges, data = load_ini_data_from_csv(FILE_PATHS)
    gt_lonlat = uploading_ground_truth(vertice, dni)
    directional_data = uploading_directional_data()
    fixed_lonlat = [tuple(gt_lonlat[dni[n]]) for n in fixed_labels]
    data_sim = data_Li2sim(data)

    # --- Load CSV histories ---
    sm_all, _ = _read_model_csv(args.csv_sm)
    dm_all, _ = _read_model_csv(args.csv_dm)
    ph_all, _ = _read_model_csv(args.csv_ph)

    # --- Select representative runs ---
    rep = {}
    for name, all_hist in [("SMACOF", sm_all), ("DC-SMACOF", dm_all), ("PhysicsSim", ph_all)]:
        rep[name] = select_representative_run(
            all_hist, dni, gt_lonlat, directional_data,
            refer_pos_screen, data_sim,
            model_name=name, verbose=True,
        )

    # --- Save metadata ---
    outdir = Path(OUTPUT_DIR) / "ch6_representative"
    outdir.mkdir(parents=True, exist_ok=True)
    meta = {}
    for name in rep:
        r = rep[name]
        meta[name] = {
            "run_idx": r["run_idx"],
            "metrics": r["metrics"],
            "median_vector": r["median_vector"],
            "mad_vector": r["mad_vector"],
            "std_distance": r["std_distance"],
        }
    meta_path = outdir / "representative_runs_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[Saved] {meta_path}")

    # ===================================================================
    # 6.2 Convergence (Stress+RMSE and VR+MAE)
    # ===================================================================
    if not args.skip_convergence:
        hist_ph = rep["PhysicsSim"]["history"]
        hist_dm = rep["DC-SMACOF"]["history"]
        hist_sm = rep["SMACOF"]["history"]

        plot_three_model_convergence_pygame_pixelaware(
            hist_ph, hist_dm, hist_sm,
            vertice=vertice, dni=dni, data=data,
            ground_truth_positions=gt_lonlat,
            fixed_point_labels=fixed_labels,
            fixed_point_lonlat=fixed_lonlat,
            refer_pos=tuple(refer_pos_screen),
            orientation="north-up",
            pre_process=True,   # ← CSV data 已是 pixel/y-up
        )

        plot_three_model_direction_convergence(
            hist_ph, hist_dm, hist_sm,
            vertice=vertice, dni=dni, data=data,
            ground_truth_positions=gt_lonlat,
            directional_data=directional_data,
            fixed_point_labels=fixed_labels,
            fixed_point_lonlat=fixed_lonlat,
            refer_pos=tuple(refer_pos_screen),
            orientation="north-up",
            pre_process=True,
            bin_size_sm_vr=25,
            bin_size_sm_mae=25,
            bin_size_dm_vr=5,
            bin_size_dm_mae=5,
            bin_size_ph_vr=15,
            bin_size_ph_mae=15,
        )

    # ===================================================================
    # 6.3 Error map  &  6.4 Overlay  (per model)
    # ===================================================================
    for name in ["SMACOF", "DC-SMACOF", "PhysicsSim"]:
        final_yup = rep[name]["final_pos"]

        # wrong_directions needs y-up
        wrong_dir = wrong_directions_nonflip(deepcopy(final_yup), vertice, dni)

        final_ydown = _flip_for_display_keep_anchor(
            deepcopy(final_yup),
            anchor_idx=dni["鄯善"],
            target_anchor_xy=tuple(refer_pos_screen),
        )

        if not args.skip_errormap:
            visualize_error_map_official(
                deepcopy(final_ydown), vertice, dni, data, wrong_dir,
                zoom_area=None, file_name=f"{name}_",
            )

        if not args.skip_overlay:
            ground_truth_comparison(
                vertice, dni, data_sim, deepcopy(gt_lonlat),
                final_ydown[dni["鄯善"]],
                deepcopy(final_ydown),
                file_name=f"{name}_",
            )


if __name__ == "__main__":
    main()