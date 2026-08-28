"""Verify PhysicsSim-to-BFGS polishing outputs against their source experiment."""

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
from run_paper_script.ch5_ablation_progressive import _evaluate, _target_positions_sim


REQUIRED_FILES = (
    "polishing_config.json",
    "polishing_runs.csv",
    "polishing_summary.csv",
    "polishing_stratum_counts.csv",
    "polishing_trajectory.csv",
    "polishing_final_positions_y_up_sim.csv",
    "peripheral_force_by_seed_node.csv",
    "peripheral_node_summary.csv",
    "peripheral_spearman_node_means.csv",
    "peripheral_spearman_by_seed.csv",
)
PLOT_FILES = (
    "trajectory_objective_and_test_rmse.png",
    "objective_vs_test_rmse.png",
    "repulsion_vs_spread.png",
    "trajectory_components_and_gradient.png",
    "peripheral_force_balance_diagnostics.png",
)


def _assert_close(label: str, actual, expected, *, atol=1e-8, rtol=1e-8):
    if not np.allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=True):
        raise ValueError(f"{label} mismatch: actual={actual!r}, expected={expected!r}")


def _read_required(outdir: Path, name: str) -> Path:
    path = outdir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing polishing output: {path}")
    return path


def verify(outdir: str | Path) -> dict:
    outdir = Path(outdir)
    for name in REQUIRED_FILES:
        _read_required(outdir, name)

    config = json.loads((outdir / "polishing_config.json").read_text(encoding="utf-8"))
    if config.get("optimizer") != "scipy.optimize.minimize(method='BFGS', jac=True)":
        raise ValueError("Polishing output does not identify full-memory analytic-gradient BFGS.")
    if config.get("uses_limited_memory_bfgs") is not False:
        raise ValueError("Polishing output incorrectly identifies limited-memory BFGS.")
    if "posthoc diagnostics only" not in config.get("test_site_policy", ""):
        raise ValueError("The held-out test-site policy is missing or incorrect.")
    if config.get("plots_generated"):
        for name in PLOT_FILES:
            _read_required(outdir, name)

    weights = ObjectiveWeights.from_physics_hpo(
        alpha=float(config["alpha"]),
        beta=float(config["beta"]),
        w_dis=float(config["w_dis"]),
    )
    problem = build_current_objective(weights=weights)
    _graph, vertices, dni, _edges, distance_rows = load_ini_data_from_csv(FILE_PATHS)
    if tuple(vertices) != problem.vertices:
        raise ValueError("Polishing objective and current graph use different vertex orders.")

    gt_lonlat = uploading_ground_truth(vertices, dni)
    test_labels = get_test_site_labels()
    anchor_label = get_anchor_align_label()
    targets = _target_positions_sim(dni, gt_lonlat, anchor_label, refer_pos_sim)
    data_sim = data_Li2sim(distance_rows)
    directional_data = uploading_directional_data()

    runs = pd.read_csv(outdir / "polishing_runs.csv")
    trajectory = pd.read_csv(outdir / "polishing_trajectory.csv")
    positions = pd.read_csv(outdir / "polishing_final_positions_y_up_sim.csv")
    forces = pd.read_csv(outdir / "peripheral_force_by_seed_node.csv")
    expected_seeds = sorted(int(seed) for seed in config["seeds"])
    if sorted(runs["seed"].astype(int).tolist()) != expected_seeds:
        raise ValueError("Polishing run rows do not match configured seeds.")

    as_outdir = Path(config["physics_as_source"])
    source_positions = pd.read_csv(as_outdir / "progressive_final_positions_y_up_sim.csv")
    source_positions = source_positions[source_positions["variant"] == "PhysicsSim-Full"]

    checked_endpoints = 0
    checked_trajectory_rows = 0
    max_source_free_coordinate_difference = 0.0
    for seed in expected_seeds:
        run = runs[runs["seed"] == seed].iloc[0]
        seed_positions = positions[positions["seed"] == seed]
        if len(seed_positions) != 2 * len(vertices):
            raise ValueError(f"Seed {seed} does not contain two complete endpoint configurations.")

        source = source_positions[source_positions["seed"] == seed].set_index("label").loc[vertices]
        source_xy = source[["x_y_up_sim", "y_y_up_sim"]].to_numpy(float)
        for stage, prefix in (("PhysicsSim endpoint", "before"), ("BFGS polished", "after")):
            ordered = seed_positions[seed_positions["stage"] == stage].set_index("label").loc[vertices]
            points = ordered[["x_y_up_sim", "y_y_up_sim"]].to_numpy(float)
            centered = points - np.asarray(refer_pos_sim, dtype=float)
            _assert_close(
                f"seed {seed} {stage} exact anchors",
                centered[problem.anchor_indices],
                problem.anchor_coordinates,
                atol=1e-9,
                rtol=0.0,
            )
            components = problem.components(problem.pack(centered))
            _assert_close(f"seed {seed} {stage} objective total", components.total, run[f"{prefix}_objective_total"])
            _assert_close(f"seed {seed} {stage} distance objective", components.weighted_distance, run[f"{prefix}_objective_distance_weighted"])
            _assert_close(f"seed {seed} {stage} direction objective", components.weighted_direction, run[f"{prefix}_objective_direction_weighted"])
            _assert_close(f"seed {seed} {stage} repulsion objective", components.weighted_repulsion, run[f"{prefix}_objective_repulsion_weighted"])

            recalculated = _evaluate(
                stage,
                seed,
                points,
                vertices,
                dni,
                data_sim,
                directional_data,
                test_labels,
                targets,
                distance_rows,
            )
            metric_pairs = {
                "RMSE_test_km": "RMSE_test_km_posthoc",
                "E_distance_stress": "E_distance_stress",
                "E_direction_vr": "E_direction_vr",
                "E_direction_mae": "E_direction_mae",
            }
            for formal_name, polishing_name in metric_pairs.items():
                _assert_close(
                    f"seed {seed} {stage} metric {formal_name}",
                    recalculated[formal_name],
                    run[f"{prefix}_{polishing_name}"],
                )
            checked_endpoints += 1

            if stage == "PhysicsSim endpoint":
                source_centered = source_xy - np.asarray(refer_pos_sim, dtype=float)
                free_diff = float(
                    np.max(np.abs(centered[problem.free_indices] - source_centered[problem.free_indices]))
                )
                max_source_free_coordinate_difference = max(
                    max_source_free_coordinate_difference, free_diff
                )
                _assert_close(f"seed {seed} PhysicsSim free coordinates", free_diff, 0.0, atol=1e-10, rtol=0.0)

        bfgs_history = trajectory[
            (trajectory["seed"] == seed) & (trajectory["stage"] == "BFGS polishing")
        ].sort_values("stage_iteration")
        if bfgs_history.empty or int(bfgs_history.iloc[0]["stage_iteration"]) != 0:
            raise ValueError(f"Seed {seed} has no BFGS initial trajectory state.")
        _assert_close(f"seed {seed} BFGS trajectory start", bfgs_history.iloc[0]["objective_total"], run["before_objective_total"])
        _assert_close(f"seed {seed} BFGS trajectory end", bfgs_history.iloc[-1]["objective_total"], run["after_objective_total"])
        objective = bfgs_history["objective_total"].to_numpy(float)
        tolerance = max(1.0, float(np.max(np.abs(objective)))) * 1e-10
        if np.any(np.diff(objective) > tolerance):
            raise ValueError(f"Accepted BFGS objective is not monotone for seed {seed}.")
        for _, row in bfgs_history.iterrows():
            total = (
                row["objective_distance_weighted"]
                + row["objective_direction_weighted"]
                + row["objective_repulsion_weighted"]
            )
            _assert_close(f"seed {seed} trajectory component sum", total, row["objective_total"])
            if row["test_metric_policy"] != "posthoc_only_never_used_for_optimization_or_stopping":
                raise ValueError(f"Seed {seed} trajectory has an incorrect test metric policy.")
        checked_trajectory_rows += len(bfgs_history)

    expected_force_rows = len(expected_seeds) * 2 * len(test_labels)
    if len(forces) != expected_force_rows:
        raise ValueError(
            f"Peripheral diagnostics contain {len(forces)} rows; expected {expected_force_rows}."
        )
    for suffix in ("gradient_outward_component", "force_outward_component"):
        component_sum = sum(forces[f"{term}_{suffix}"] for term in ("distance", "direction", "repulsion"))
        _assert_close(f"peripheral total {suffix}", forces[f"total_{suffix}"], component_sum)

    stratum_counts = pd.read_csv(outdir / "polishing_stratum_counts.csv")
    observed_counts = runs.groupby("after_objective_stratum").size().rename("count").reset_index()
    pd.testing.assert_frame_equal(
        stratum_counts.sort_values("after_objective_stratum").reset_index(drop=True),
        observed_counts.sort_values("after_objective_stratum").reset_index(drop=True),
        check_dtype=False,
    )

    return {
        "verified": True,
        "n_seeds": len(expected_seeds),
        "n_endpoints": checked_endpoints,
        "n_bfgs_trajectory_states": checked_trajectory_rows,
        "n_peripheral_rows": len(forces),
        "max_source_free_coordinate_difference_sim": max_source_free_coordinate_difference,
        "test_site_policy": config["test_site_policy"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.outdir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
