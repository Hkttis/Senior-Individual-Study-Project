"""Run SMACOF and DC-SMACOF on the advanced sparse non-rigid fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import MDS_model.directed_mds_model as dc_smacof
import MDS_model.stress_majorization_mds_model as smacof
from scripts.run_advanced_repulsion_synthetic import METRICS, evaluate_layout, load_dataset


def _build_graph(dataset: dict) -> tuple[list, list[tuple[str, str]]]:
    graph = [[] for _ in dataset["vertices"]]
    edges = []
    for source, target, distance in dataset["distance_data"]:
        edges.append((source, target))
        graph[dataset["dni"][source]].append([source, target, "", distance])
        graph[dataset["dni"][target]].append([target, source, "", distance])
    return graph, edges


def _rigid_align(points: np.ndarray, target: np.ndarray, fit_indices: list[int]) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    target = np.asarray(target, dtype=float)
    indices = np.asarray(fit_indices, dtype=int)
    point_center = points[indices].mean(axis=0)
    target_center = target[indices].mean(axis=0)
    u, _singular_values, vt = np.linalg.svd(
        (points[indices] - point_center).T @ (target[indices] - target_center)
    )
    return (points - point_center) @ (u @ vt) + target_center


def _summary(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in runs.groupby("variant", sort=False):
        for metric in METRICS:
            values = group[metric].to_numpy(float)
            rows.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "n": len(values),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "median": float(np.median(values)),
                }
            )
    return pd.DataFrame(rows)


def run_baselines(
    seeds: list[int],
    *,
    smacof_iterations: int = 1000,
    dc_iterations: int = 1000,
    dc_distance_weight: float = 1.0,
    dc_direction_weight: float = 0.01,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset = load_dataset()
    graph, edges = _build_graph(dataset)
    calibration_indices = [
        dataset["dni"][label]
        for label, role in dataset["roles"].items()
        if role in {"anchor", "anchor_align"}
    ]
    frame_index = dataset["dni"]["P00"]
    rows = []
    position_rows = []

    old_smacof_iterations = smacof.iteration_times
    old_dc_iterations = dc_smacof.stop_iteration_times
    smacof.iteration_times = int(smacof_iterations)
    dc_smacof.stop_iteration_times = int(dc_iterations)
    try:
        for seed in seeds:
            np.random.seed(seed)
            points, _stress, _history = smacof.stress_majorization(
                graph, dataset["dni"], dataset["vertices"], edges
            )
            points = _rigid_align(points, dataset["expected"], calibration_indices)
            rows.append(evaluate_layout("SMACOF", seed, points, dataset))
            for label, point in zip(dataset["vertices"], points):
                position_rows.append(
                    {"variant": "SMACOF", "seed": seed, "model_name": label, "x": point[0], "y": point[1]}
                )

            np.random.seed(seed)
            points, _stress, _history = dc_smacof.directed_MDS(
                [dataset["direction_data"], [], []],
                dataset["distance_data"],
                graph,
                dataset["vertices"],
                dataset["dni"],
                edges,
                dc_distance_weight,
                dc_direction_weight,
            )
            points = np.asarray(points, dtype=float)
            points = points - points[frame_index] + dataset["expected"][frame_index]
            rows.append(evaluate_layout("DC-SMACOF", seed, points, dataset))
            for label, point in zip(dataset["vertices"], points):
                position_rows.append(
                    {"variant": "DC-SMACOF", "seed": seed, "model_name": label, "x": point[0], "y": point[1]}
                )
    finally:
        smacof.iteration_times = old_smacof_iterations
        dc_smacof.stop_iteration_times = old_dc_iterations

    runs = pd.DataFrame(rows)
    return runs, _summary(runs), pd.DataFrame(position_rows)


def _parse_seeds(text: str) -> list[int]:
    seeds = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--smacof-iterations", type=int, default=1000)
    parser.add_argument("--dc-iterations", type=int, default=1000)
    parser.add_argument("--dc-distance-weight", type=float, default=1.0)
    parser.add_argument("--dc-direction-weight", type=float, default=0.01)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    seeds = _parse_seeds(args.seeds)
    runs, summary, positions = run_baselines(
        seeds,
        smacof_iterations=args.smacof_iterations,
        dc_iterations=args.dc_iterations,
        dc_distance_weight=args.dc_distance_weight,
        dc_direction_weight=args.dc_direction_weight,
    )
    runs.to_csv(outdir / "baseline_runs.csv", index=False)
    summary.to_csv(outdir / "baseline_summary.csv", index=False)
    positions.to_csv(outdir / "baseline_positions.csv", index=False)
    (outdir / "baseline_config.json").write_text(
        json.dumps(
            {
                "seeds": seeds,
                "smacof_iterations": args.smacof_iterations,
                "dc_iterations": args.dc_iterations,
                "dc_distance_weight": args.dc_distance_weight,
                "dc_direction_weight": args.dc_direction_weight,
                "smacof_alignment": "rigid_procrustes_P00_P01_P07",
                "dc_alignment": "translation_P00_only",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"\nSaved: {outdir}")


if __name__ == "__main__":
    main()
