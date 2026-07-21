"""Run the minimally edited production clone and save finite-value diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments.dc_smacof_wang2017_audit.run_audit import (
    _formal_metrics,
    build_problem,
    objective_components,
)
from experiments.dc_smacof_wang2017_audit.wang_model_minimal_copy import directed_MDS
from library.config import FILE_PATHS
from library.data_io import load_ini_data_from_csv, uploading_directional_data, uploading_ground_truth


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the minimally edited Wang production clone.")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--distance-weight", type=float, default=1.0)
    parser.add_argument("--direction-weight", type=float, default=10.0 ** -0.5)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=False)

    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    graph, labels, dni, edges, data = load_ini_data_from_csv(FILE_PATHS)
    directions = uploading_directional_data()
    gt_lonlat = uploading_ground_truth(labels, dni)
    c_data = [[], [], [[row[0], row[1], row[2]] for row in directions], []]
    problem = build_problem(
        labels,
        dni,
        data,
        directions,
        distance_weight=args.distance_weight,
        direction_weight=args.direction_weight,
    )

    trace_rows: list[dict] = []
    summary_rows: list[dict] = []
    for seed in seeds:
        print(f"[minimal-copy] seed={seed}")
        np.random.seed(seed)
        status, failure = "ok", ""
        try:
            final, stress_history, position_history = directed_MDS(
                c_data,
                data,
                graph,
                labels,
                dni,
                edges,
                distance_weight=args.distance_weight,
                direction_weight=args.direction_weight,
            )
        except Exception as exc:
            status, failure = "failed", f"{type(exc).__name__}: {exc}"
            final, stress_history, position_history = np.full((len(labels), 2), np.nan), [], []

        for iteration, positions in enumerate(position_history):
            positions = np.asarray(positions, dtype=float)
            distance_obj, direction_obj = objective_components(problem, positions)
            trace_rows.append(
                {
                    "seed": seed,
                    "iteration": iteration,
                    "finite": bool(np.isfinite(positions).all()),
                    "max_abs_coordinate_li": (
                        float(np.max(np.abs(positions))) if np.isfinite(positions).all() else float("inf")
                    ),
                    "distance_objective": distance_obj,
                    "direction_objective": direction_obj,
                    "total_objective": distance_obj + direction_obj,
                    "recorded_model_stress": float(stress_history[iteration]),
                }
            )
        if position_history and not all(row["finite"] for row in trace_rows if row["seed"] == seed):
            status, failure = "failed", "non_finite_position_history"

        summary = {
            "seed": seed,
            "status": status,
            "failure_reason": failure,
            "history_frames": len(position_history),
            "final_max_abs_coordinate_li": (
                float(np.max(np.abs(final))) if np.isfinite(final).all() else float("inf")
            ),
        }
        if status == "ok":
            summary.update(_formal_metrics(np.asarray(final), labels, dni, data, directions, gt_lonlat))
        summary_rows.append(summary)

    _write_csv(args.outdir / "minimal_copy_iteration_trace.csv", trace_rows)
    _write_csv(args.outdir / "minimal_copy_run_summary.csv", summary_rows)
    config = {
        "implementation": "production clone with only DV changed to Wang current-distance target",
        "seeds": seeds,
        "distance_weight": args.distance_weight,
        "direction_weight": args.direction_weight,
        "n_failures": sum(row["status"] != "ok" for row in summary_rows),
    }
    (args.outdir / "minimal_copy_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[minimal-copy] saved to {args.outdir}; failures={config['n_failures']}")


if __name__ == "__main__":
    main()
