"""Verify SciPy-BFGS outputs against the objective and formal AS metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from library.config import FILE_PATHS, refer_pos_sim
from library.data_io import (
    get_anchor_align_label,
    get_test_site_labels,
    load_ini_data_from_csv,
    uploading_directional_data,
    uploading_ground_truth,
)
from library.scipy_objective import ObjectiveWeights, build_current_objective
from library.units import data_Li2sim
from run_paper_script.ch5_ablation_progressive import METRICS, _evaluate, _target_positions_sim


def _assert_close(label: str, actual, expected, *, atol=1e-8, rtol=1e-8):
    if not np.allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=True):
        raise ValueError(f"{label} mismatch: actual={actual!r}, expected={expected!r}")


def _problem_from_config(config: dict):
    return build_current_objective(
        weights=ObjectiveWeights.from_physics_hpo(
            alpha=float(config["alpha"]),
            beta=float(config["beta"]),
            w_dis=float(config["w_dis"]),
        )
    )


def verify(outdir: str | Path) -> dict:
    outdir = Path(outdir)
    config_path = outdir / "bfgs_experiment_config.json"
    runs_path = outdir / "bfgs_runs_by_seed.csv"
    positions_path = outdir / "bfgs_final_positions_y_up_sim.csv"
    for path in (config_path, runs_path, positions_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing SciPy-BFGS output: {path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["method"] != "scipy.optimize.minimize(method='BFGS')":
        raise ValueError("Output was not produced by full-memory SciPy BFGS.")
    if config.get("uses_limited_memory_bfgs") is not False:
        raise ValueError("Output incorrectly identifies the solver as limited-memory BFGS.")

    problem = _problem_from_config(config)
    _graph, vertice, dni, _edges, distance_data = load_ini_data_from_csv(FILE_PATHS)
    gt_lonlat = uploading_ground_truth(vertice, dni)
    anchor_label = get_anchor_align_label()
    test_labels = get_test_site_labels()
    targets = _target_positions_sim(dni, gt_lonlat, anchor_label, refer_pos_sim)
    data_sim = data_Li2sim(distance_data)
    directional_data = uploading_directional_data()
    runs = pd.read_csv(runs_path)
    positions = pd.read_csv(positions_path)
    expected_seeds = [int(seed) for seed in config["seeds"]]
    if sorted(runs["seed"].astype(int).tolist()) != sorted(expected_seeds):
        raise ValueError("Run rows do not match configured seeds.")

    checked_history_states = 0
    for seed in expected_seeds:
        seed_dir = outdir / f"seed_{seed}"
        history = pd.read_csv(seed_dir / "bfgs_objective_history.csv")
        history_positions = pd.read_csv(seed_dir / "bfgs_position_history_y_up_sim.csv")
        final_positions = pd.read_csv(seed_dir / "bfgs_final_positions_y_up_sim.csv")
        seed_metrics = pd.read_csv(seed_dir / "bfgs_final_metrics.csv").iloc[0]
        run_summary = json.loads((seed_dir / "bfgs_run_summary.json").read_text(encoding="utf-8"))
        if len(history) != int(run_summary["accepted_states_including_initial"]):
            raise ValueError(f"History length mismatch for seed {seed}.")
        if history["iteration"].astype(int).tolist() != list(range(len(history))):
            raise ValueError(f"History iterations are not contiguous for seed {seed}.")
        objective_diff = np.diff(history["objective_total"].to_numpy(float))
        scale = max(1.0, float(np.max(np.abs(history["objective_total"]))))
        if np.any(objective_diff > 1e-10 * scale):
            raise ValueError(f"Accepted BFGS objective is not monotone for seed {seed}.")

        for iteration, group in history_positions.groupby("iteration", sort=True):
            ordered = group.set_index("label").loc[vertice]
            points = ordered[["x_y_up_sim", "y_y_up_sim"]].to_numpy(float)
            centered = points - np.asarray(refer_pos_sim, dtype=float)
            vector = problem.pack(centered)
            components = problem.components(vector)
            recorded = history.loc[history["iteration"] == iteration].iloc[0]
            _assert_close(
                f"seed {seed} iteration {iteration} objective",
                components.total,
                float(recorded["objective_total"]),
            )
            _assert_close(
                f"seed {seed} iteration {iteration} anchors",
                centered[problem.anchor_indices],
                problem.anchor_coordinates,
                atol=1e-9,
                rtol=0.0,
            )
            checked_history_states += 1

        final_ordered = final_positions.set_index("label").loc[vertice]
        final_points = final_ordered[["x_y_up_sim", "y_y_up_sim"]].to_numpy(float)
        aggregate = positions[positions["seed"] == seed].set_index("label").loc[vertice]
        aggregate_points = aggregate[["x_y_up_sim", "y_y_up_sim"]].to_numpy(float)
        _assert_close(f"seed {seed} aggregate positions", aggregate_points, final_points)

        recalculated = _evaluate(
            "SciPy-BFGS",
            seed,
            final_points,
            vertice,
            dni,
            data_sim,
            directional_data,
            test_labels,
            targets,
            distance_data,
        )
        for metric in METRICS:
            _assert_close(
                f"seed {seed} metric {metric}",
                float(seed_metrics[metric]),
                float(recalculated[metric]),
            )

    return {
        "verified": True,
        "n_seeds": len(expected_seeds),
        "n_history_states": checked_history_states,
        "n_metrics_per_seed": len(METRICS),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.outdir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
