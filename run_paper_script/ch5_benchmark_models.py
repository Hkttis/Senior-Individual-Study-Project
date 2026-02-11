"""scripts.ch5_benchmark_models

Chapter 5 — Repeated-measurement benchmark.

Runs `library.model_cmp.multi_measurement_benchmark` to compare:
  - StressMajorization
  - DirectedMDS
  - PhysicsSim

It prints summary statistics (mean/SD/SE/95% CI) and optionally saves
all position histories to CSV (for later reuse in figure generation).

Usage
-----
python -m scripts.ch5_benchmark_models --n-runs 100
python -m scripts.ch5_benchmark_models --n-runs 200 --save-histories
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from library.config import OUTPUT_DIR, FILE_PATHS
from library.data_io import load_ini_data_from_csv, uploading_ground_truth, save_all_pos_histories_px_csv


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-runs", type=int, default=100)
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
    p.add_argument("--save-histories", action="store_true", help="Save all histories to CSV")
    p.add_argument(
        "--outdir",
        type=str,
        default="",
        help="Output directory for JSON summary (default: <OUTPUT_DIR>/ch5)",
    )
    return p.parse_args()


def _parse_xy(s: str):
    xs, ys = s.split(",")
    return (float(xs.strip()), float(ys.strip()))


def main() -> None:
    args = _parse_args()
    fixed_point_labels = [x.strip() for x in args.fixed.split(",") if x.strip()]
    refer_pos = _parse_xy(args.refer_pos)

    # Ground truth is needed to determine anchor lon/lat.
    _graph, vertice, dni, _edges, _data = load_ini_data_from_csv(FILE_PATHS)
    gt_lonlat = uploading_ground_truth(vertice, dni)
    fixed_point_lonlat = [tuple(gt_lonlat[dni[name]]) for name in fixed_point_labels]

    from library.model_cmp import multi_measurement_benchmark

    res = multi_measurement_benchmark(
        n_runs=int(args.n_runs),
        refer_pos=refer_pos,
        fixed_point_labels=fixed_point_labels,
        fixed_point_lonlat=fixed_point_lonlat,
        verbose=True,
    )

    # Persist summary stats as JSON (paper tables).
    outdir = Path(args.outdir) if args.outdir else (Path(OUTPUT_DIR) / "ch5")
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / f"benchmark_n{int(args.n_runs)}.json"
    json_path.write_text(json.dumps(res["stats"], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ch5-benchmark] Saved: {json_path}")

    if args.save_histories:
        save_all_pos_histories_px_csv(
            res["all_pos_history_px"]["StressMajorization"],
            res["all_pos_history_px"]["DirectedMDS"],
            res["all_pos_history_px"]["PhysicsSim"],
            vertice=vertice,
        )
        print("[ch5-benchmark] Saved: all_pos_histories_px_csv (paths from FILE_PATHS)")


if __name__ == "__main__":
    main()
