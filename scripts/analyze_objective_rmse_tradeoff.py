"""Diagnose whether lowering the formal objective also lowers held-out RMSE.

The analysis is post hoc and never modifies the source BFGS experiment.  It
replaces held-out test positions with their archaeological target positions in
three increasingly conservative ways: direct substitution, interpolation, and
conditional re-optimization of every remaining free node.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from library.config import FILE_PATHS, km2pix, refer_pos_sim
from library.data_io import (
    get_anchor_align_label,
    get_test_site_labels,
    load_ini_data_from_csv,
    uploading_ground_truth,
)
from library.scipy_minimizer import run_bfgs
from library.scipy_objective import FixedAnchorObjective, build_current_objective
from run_paper_script.ch5_ablation_progressive import _target_positions_sim


DEFAULT_SOURCE = "outputs/ch5_scipy_bfgs_full_100seeds_20260821"
DEFAULT_OUTDIR = "outputs/ch5_scipy_bfgs_objective_rmse_tradeoff_20260822"
FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _load_positions(source: Path, seed: int, vertices: tuple[str, ...]) -> np.ndarray:
    frame = pd.read_csv(source / "bfgs_final_positions_y_up_sim.csv")
    selected = frame[frame["seed"] == seed].set_index("label")
    missing = [label for label in vertices if label not in selected.index]
    if missing:
        raise ValueError(f"Seed {seed} is missing position rows: {missing}")
    return selected.loc[list(vertices), ["x_y_up_sim", "y_y_up_sim"]].to_numpy(float)


def _rmse_km(
    positions_y_up_sim: np.ndarray,
    targets_y_up_sim: dict[str, np.ndarray],
    test_labels: list[str],
    vertex_index: dict[str, int],
) -> float:
    errors_sim = np.asarray(
        [
            np.linalg.norm(
                positions_y_up_sim[vertex_index[label]] - targets_y_up_sim[label]
            )
            for label in test_labels
        ],
        dtype=float,
    )
    return float(np.sqrt(np.mean((errors_sim / km2pix) ** 2)))


def _component_row(problem: FixedAnchorObjective, centered: np.ndarray) -> dict[str, float]:
    components = problem.components(problem.pack(centered))
    return {
        "objective_total": components.total,
        "objective_distance_weighted": components.weighted_distance,
        "objective_direction_weighted": components.weighted_direction,
        "objective_repulsion_weighted": components.weighted_repulsion,
    }


def _with_fixed_positions(
    base: FixedAnchorObjective,
    fixed_positions: dict[int, np.ndarray],
) -> FixedAnchorObjective:
    return FixedAnchorObjective(
        vertices=base.vertices,
        distance_pairs=base.distance_pairs,
        distance_targets=base.distance_targets,
        direction_pairs=base.direction_pairs,
        direction_vectors=base.direction_vectors,
        direction_half_widths=base.direction_half_widths,
        anchor_positions=fixed_positions,
        weights=base.weights,
        epsilon=base.epsilon,
        singularity_tolerance=base.singularity_tolerance,
    )


def _lowest_basin_seeds(runs: pd.DataFrame) -> list[int]:
    ok = runs[(runs["status"] == "ok") & runs["objective_final"].notna()].copy()
    ordered = ok.sort_values("objective_final").reset_index(drop=True)
    values = ordered["objective_final"].to_numpy(float)
    if len(values) < 4:
        return ordered["seed"].astype(int).tolist()
    gaps = np.diff(values)
    split_indices = np.sort(np.argsort(gaps)[-3:] + 1)
    return ordered.iloc[: int(split_indices[0])]["seed"].astype(int).tolist()


def analyze(source: Path, outdir: Path) -> dict:
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)

    problem = build_current_objective()
    _graph, vertices, dni, _edges, _distance_data = load_ini_data_from_csv(FILE_PATHS)
    if tuple(vertices) != problem.vertices:
        raise ValueError("Formal graph order differs from the SciPy objective order.")
    gt_lonlat = uploading_ground_truth(vertices, dni)
    anchor_label = get_anchor_align_label()
    test_labels = get_test_site_labels()
    targets = _target_positions_sim(dni, gt_lonlat, anchor_label, refer_pos_sim)
    target_centered = {
        label: np.asarray(targets[label], dtype=float) - np.asarray(refer_pos_sim, dtype=float)
        for label in test_labels
    }

    runs = pd.read_csv(source / "bfgs_runs_by_seed.csv")
    successful = runs[(runs["status"] == "ok") & runs["objective_final"].notna()]
    minimum_seed = int(successful.loc[successful["objective_final"].idxmin(), "seed"])
    minimum_y_up = _load_positions(source, minimum_seed, problem.vertices)
    minimum_centered = minimum_y_up - np.asarray(refer_pos_sim, dtype=float)
    baseline_components = _component_row(problem, minimum_centered)
    baseline_rmse = _rmse_km(minimum_y_up, targets, test_labels, dni)

    direct_rows = []
    for fraction in FRACTIONS:
        candidate = minimum_centered.copy()
        for label in test_labels:
            index = dni[label]
            candidate[index] = (
                (1.0 - fraction) * minimum_centered[index]
                + fraction * target_centered[label]
            )
        row = {"fraction_toward_ground_truth": fraction}
        row.update(_component_row(problem, candidate))
        row["RMSE_test_km"] = _rmse_km(
            candidate + np.asarray(refer_pos_sim, dtype=float), targets, test_labels, dni
        )
        row["delta_objective"] = row["objective_total"] - baseline_components["objective_total"]
        direct_rows.append(row)
    direct = pd.DataFrame(direct_rows)
    direct.to_csv(outdir / "direct_interpolation_path.csv", index=False, encoding="utf-8-sig")

    site_rows = []
    for label in test_labels:
        candidate = minimum_centered.copy()
        candidate[dni[label]] = target_centered[label]
        row = {"test_label": label}
        row.update(_component_row(problem, candidate))
        row["RMSE_test_km"] = _rmse_km(
            candidate + np.asarray(refer_pos_sim, dtype=float), targets, test_labels, dni
        )
        row["delta_objective"] = row["objective_total"] - baseline_components["objective_total"]
        row["delta_RMSE_test_km"] = row["RMSE_test_km"] - baseline_rmse
        site_rows.append(row)
    per_site = pd.DataFrame(site_rows).sort_values("delta_objective", ascending=False)
    per_site.to_csv(outdir / "single_site_substitution.csv", index=False, encoding="utf-8-sig")

    conditional_rows = []
    warm_full = minimum_centered.copy()
    original_anchors = {
        int(index): problem.anchor_coordinates[position].copy()
        for position, index in enumerate(problem.anchor_indices)
    }
    for fraction in FRACTIONS:
        fixed = dict(original_anchors)
        for label in test_labels:
            index = dni[label]
            fixed[index] = (
                (1.0 - fraction) * minimum_centered[index]
                + fraction * target_centered[label]
            )
        conditional_problem = _with_fixed_positions(problem, fixed)
        for index, position in fixed.items():
            warm_full[index] = position
        result = run_bfgs(conditional_problem.pack(warm_full), conditional_problem)
        if result["y_final"] is None:
            raise RuntimeError(
                f"Conditional BFGS failed at fraction={fraction}: {result['failure_reason']}"
            )
        warm_full = conditional_problem.unpack(result["y_final"])
        components = conditional_problem.components(result["y_final"])
        positions_y_up = warm_full + np.asarray(refer_pos_sim, dtype=float)
        conditional_rows.append(
            {
                "fraction_toward_ground_truth": fraction,
                "optimizer_success": result["success"],
                "optimizer_message": result["failure_reason"] or "",
                "objective_total": components.total,
                "objective_distance_weighted": components.weighted_distance,
                "objective_direction_weighted": components.weighted_direction,
                "objective_repulsion_weighted": components.weighted_repulsion,
                "RMSE_test_km": _rmse_km(positions_y_up, targets, test_labels, dni),
                "delta_objective": components.total - baseline_components["objective_total"],
                "iterations": result["iterations"],
                "gradient_norm_inf": result["gradient_norm"],
            }
        )
    conditional = pd.DataFrame(conditional_rows)
    conditional.to_csv(
        outdir / "conditional_reoptimization_path.csv", index=False, encoding="utf-8-sig"
    )

    basin_rows = []
    basin_seeds = _lowest_basin_seeds(runs)
    for seed in basin_seeds:
        original_y_up = _load_positions(source, seed, problem.vertices)
        original_centered = original_y_up - np.asarray(refer_pos_sim, dtype=float)
        replaced = original_centered.copy()
        for label in test_labels:
            replaced[dni[label]] = target_centered[label]
        before = _component_row(problem, original_centered)
        after = _component_row(problem, replaced)
        basin_rows.append(
            {
                "seed": seed,
                "RMSE_before_km": _rmse_km(original_y_up, targets, test_labels, dni),
                "RMSE_after_km": 0.0,
                "objective_before": before["objective_total"],
                "objective_after": after["objective_total"],
                "delta_objective": after["objective_total"] - before["objective_total"],
                "delta_distance_weighted": after["objective_distance_weighted"] - before["objective_distance_weighted"],
                "delta_direction_weighted": after["objective_direction_weighted"] - before["objective_direction_weighted"],
                "delta_repulsion_weighted": after["objective_repulsion_weighted"] - before["objective_repulsion_weighted"],
            }
        )
    basin = pd.DataFrame(basin_rows)
    basin.to_csv(outdir / "lowest_basin_all_test_substitution.csv", index=False, encoding="utf-8-sig")

    summary = {
        "source": str(source),
        "minimum_seed": minimum_seed,
        "minimum_objective": baseline_components["objective_total"],
        "minimum_seed_RMSE_test_km": baseline_rmse,
        "test_labels": test_labels,
        "lowest_basin_run_count": len(basin_seeds),
        "all_test_direct_delta_objective": float(direct.iloc[-1]["delta_objective"]),
        "all_test_conditional_delta_objective": float(conditional.iloc[-1]["delta_objective"]),
        "lowest_basin_direct_delta_objective_mean": float(basin["delta_objective"].mean()),
        "lowest_basin_direct_delta_objective_min": float(basin["delta_objective"].min()),
        "lowest_basin_direct_delta_objective_max": float(basin["delta_objective"].max()),
    }
    (outdir / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    args = parser.parse_args()
    summary = analyze(Path(args.source), Path(args.outdir))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
