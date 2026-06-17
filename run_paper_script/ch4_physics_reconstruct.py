"""run_paper_script.ch4_physics_reconstruct

Chapter 4 — Core method reproduction.

Run ONE physics simulation (ball–spring with damping + anchors + direction
correction) and optionally show the live Pygame window.

This script is intentionally minimal: it focuses on generating the recovered
layout (pos_matrix) and the optimization trace (stress_history, pos_history).
Chapter 6 scripts take care of publication figures.

Usage
-----
Run from the physics_simulation project root.
By default, fixed anchors are read from data/site_rmse_points.csv
(use_role=anchor: 鄯善, 車師前, 都護治/烏壘).

python -m run_paper_script.paper_run ch4 --seed 0 --plot
python -m run_paper_script.paper_run ch4 --seed 0 --no-save

Outputs
-------
By default it saves a compressed NPZ under:
  <OUTPUT_DIR>/ch4/physics_run_seed{seed}.npz

The NPZ contains:
  - pos_final_px      (n,2)   final layout in *screen pixel coordinates* (y-down)
  - pos_history_px    (T,n,2) layout trace in screen pixel coordinates
  - stress_history    (T,)    stress trace
  - wrong_dir_count   (1,)    number of violated direction constraints (final)

Note: We keep the "paper" convention that visualization functions expect
screen coordinates (y-down). Internally, Pymunk runs in y-up; we therefore flip
all positions before saving.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import List

import numpy as np

from library.config import (
    OUTPUT_DIR,
    FILE_PATHS,
    refer_pos as DEFAULT_REFER_POS_SCREEN,
    height,
    SPRING_STIFFNESS_BASE,
    REPULSION_STRENGTH_BASE,
    DIRECTIONAL_FORCE_MAGNITUDE_BASE,
)
from library.data_io import load_ini_data_from_csv, uploading_ground_truth, uploading_directional_data, get_anchor_labels
from library.initialization import generate_CHEN_initial_positions
from library.units import data_Li2sim
from library.physics import main_physics_simulation
from library.coordinates import flipping_y


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--fixed",
        type=str,
        default="",
        help="Comma-separated anchor labels (must exist in ground truth).",
    )
    p.add_argument(
        "--refer-pos",
        type=str,
        default="600,500",
        help="Screen anchor pixel, as 'x,y' (y-down).",
    )
    p.add_argument("--plot", action="store_true", help="Show live physics simulation window")
    p.add_argument("--no-save", action="store_true", help="Do not save NPZ/JSON artifacts")
    p.add_argument(
        "--outdir",
        type=str,
        default="",
        help="Output directory (default: <OUTPUT_DIR>/ch4)",
    )
    return p.parse_args()


def _parse_xy(s: str) -> List[float]:
    xs, ys = s.split(",")
    return [float(xs.strip()), float(ys.strip())]


def run(seed: int, fixed_point_labels: List[str], refer_pos_screen: List[float], *, plot: bool) -> dict:
    """Run the Chapter-4 physics reconstruction and return artifacts."""
    # Load static inputs
    _graph, vertice, dni, _edges, data = load_ini_data_from_csv(FILE_PATHS)
    gt_lonlat = uploading_ground_truth(vertice, dni)
    directional_data = uploading_directional_data()
    fixed_points_lonlat = [tuple(gt_lonlat[dni[name]]) for name in fixed_point_labels]

    # Seed
    np.random.seed(seed)

    # Initialization uses a y-up SIM anchor.
    # The user's CLI uses screen coords (y-down), so convert here.
    _refer_pos_sim = [float(refer_pos_screen[0]), float(height) - float(refer_pos_screen[1])]
    vertice, dni, data, pos_matrix, fixed_positions_list = generate_CHEN_initial_positions(
        _refer_pos_sim, fixed_point_labels, fixed_points_lonlat
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
        plot=plot,
    )

    # Convert to screen-y-down for later visuals & saving.
    pos_final_px = np.asarray(flipping_y(pos_final_y_up), dtype=np.float32)
    pos_history_px = np.asarray([flipping_y(frame) for frame in pos_history], dtype=np.float32)
    stress_history = np.asarray(stress_history, dtype=np.float64)

    return {
        "vertice": vertice,
        "dni": dni,
        "data": data,
        "gt_lonlat": gt_lonlat,
        "directional_data": directional_data,
        "fixed_point_labels": fixed_point_labels,
        "fixed_points_lonlat": fixed_points_lonlat,
        "refer_pos_screen": refer_pos_screen,
        "refer_pos_sim": _refer_pos_sim,
        "wrong_direction_lists": wrong_direction_lists,
        "stress_history": stress_history,
        "pos_history_px": pos_history_px,
        "pos_final_px": pos_final_px,
    }


def main() -> None:
    args = _parse_args()
    fixed_point_labels = [x.strip() for x in args.fixed.split(",") if x.strip()] or get_anchor_labels()
    refer_pos_screen = _parse_xy(args.refer_pos)

    artifacts = run(
        seed=args.seed,
        fixed_point_labels=fixed_point_labels,
        refer_pos_screen=refer_pos_screen,
        plot=args.plot,
    )

    if args.no_save:
        print("[ch4] Done (no-save).")
        return

    outdir = Path(args.outdir) if args.outdir else (Path(OUTPUT_DIR) / "ch4")
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"physics_run_seed{args.seed}"

    npz_path = outdir / f"{tag}.npz"
    np.savez_compressed(
        npz_path,
        pos_final_px=artifacts["pos_final_px"],
        pos_history_px=artifacts["pos_history_px"],
        stress_history=artifacts["stress_history"],
        wrong_dir_count=np.asarray([len(artifacts["wrong_direction_lists"])], dtype=np.int32),
    )

    meta = {
        "seed": args.seed,
        "fixed_point_labels": artifacts["fixed_point_labels"],
        "fixed_points_lonlat": artifacts["fixed_points_lonlat"],
        "refer_pos_screen": artifacts["refer_pos_screen"],
        "refer_pos_sim": artifacts["refer_pos_sim"],
        "spring_stiffness": SPRING_STIFFNESS_BASE,
        "repulsion_strength": REPULSION_STRENGTH_BASE,
        "directional_force_magnitude": DIRECTIONAL_FORCE_MAGNITUDE_BASE,
        "note": "pos_*_px are screen pixel coords (y-down).",
    }
    meta_path = outdir / f"{tag}.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # Save violated direction constraints list for debugging/paper appendix.
    wrong_path = outdir / f"{tag}_wrong_directions.json"
    wrong_path.write_text(
        json.dumps(artifacts["wrong_direction_lists"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[ch4] Saved: {npz_path}")
    print(f"[ch4] Saved: {meta_path}")
    print(f"[ch4] Saved: {wrong_path}")


if __name__ == "__main__":
    main()
