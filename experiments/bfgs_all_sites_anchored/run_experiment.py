"""Run an isolated BFGS oracle diagnostic with all known sites fixed.

This module deliberately lives outside the formal paper pipeline.  It adds the
eight held-out test-site coordinates as hard constraints only to a newly built
diagnostic objective; ``build_current_objective()`` and the HPO/AS pipelines
remain unchanged and continue to use only the three calibration anchors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from library.config import FILE_PATHS, refer_pos_sim
from library.data_io import (
    get_anchor_align_label,
    get_anchor_labels,
    get_test_site_labels,
    load_ini_data_from_csv,
    uploading_ground_truth,
)
from library.scipy_minimizer import run_bfgs
from library.scipy_objective import FixedAnchorObjective, build_current_objective
from run_paper_script.ch5_ablation_progressive import _target_positions_sim
from run_paper_script.ch5_scipy_bfgs import _initial_free_vector


DEFAULT_OUTDIR = "outputs/diagnostic_bfgs_all_sites_anchored_seed0_9_20260822"


def build_all_sites_anchored_problem() -> tuple[
    FixedAnchorObjective,
    FixedAnchorObjective,
    list[str],
    dict[str, int],
    list[str],
    list[str],
]:
    """Return separate formal and oracle problems without mutating either one."""

    formal_problem = build_current_objective()
    _graph, vertices, dni, _edges, _distance_data = load_ini_data_from_csv(FILE_PATHS)
    if tuple(vertices) != formal_problem.vertices:
        raise ValueError("Formal graph order differs from the SciPy objective order.")

    calibration_labels = get_anchor_labels()
    test_labels = get_test_site_labels()
    anchor_label = get_anchor_align_label()
    gt_lonlat = uploading_ground_truth(vertices, dni)
    targets = _target_positions_sim(dni, gt_lonlat, anchor_label, refer_pos_sim)
    origin = np.asarray(refer_pos_sim, dtype=np.float64)

    fixed_positions: dict[int, np.ndarray] = {
        int(index): formal_problem.anchor_coordinates[position].copy()
        for position, index in enumerate(formal_problem.anchor_indices)
    }
    for label in test_labels:
        fixed_positions[dni[label]] = np.asarray(targets[label], dtype=np.float64) - origin

    oracle_problem = FixedAnchorObjective(
        vertices=formal_problem.vertices,
        distance_pairs=formal_problem.distance_pairs,
        distance_targets=formal_problem.distance_targets,
        direction_pairs=formal_problem.direction_pairs,
        direction_vectors=formal_problem.direction_vectors,
        direction_half_widths=formal_problem.direction_half_widths,
        anchor_positions=fixed_positions,
        weights=formal_problem.weights,
        epsilon=formal_problem.epsilon,
        singularity_tolerance=formal_problem.singularity_tolerance,
    )
    return (
        formal_problem,
        oracle_problem,
        vertices,
        dni,
        calibration_labels,
        test_labels,
    )


def run_diagnostic(*, seeds: list[int], outdir: Path) -> dict:
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)

    (
        formal_problem,
        oracle_problem,
        vertices,
        dni,
        calibration_labels,
        test_labels,
    ) = build_all_sites_anchored_problem()
    gt_lonlat = uploading_ground_truth(vertices, dni)
    calibration_lonlat = [tuple(gt_lonlat[dni[label]]) for label in calibration_labels]
    anchor_label = get_anchor_align_label()

    rows: list[dict] = []
    position_rows: list[dict] = []
    for seed in seeds:
        formal_initial = _initial_free_vector(
            seed,
            formal_problem,
            vertices,
            dni,
            calibration_labels,
            calibration_lonlat,
            anchor_label,
        )
        full_initial = formal_problem.unpack(formal_initial)
        full_initial[oracle_problem.anchor_indices] = oracle_problem.anchor_coordinates
        result = run_bfgs(oracle_problem.pack(full_initial), oracle_problem)
        selected = result.get("y_final")
        if selected is None:
            rows.append(
                {
                    "seed": seed,
                    "success": False,
                    "failure_reason": result["failure_reason"],
                }
            )
            continue

        components = oracle_problem.components(selected)
        full_final = oracle_problem.unpack(selected)
        rows.append(
            {
                "seed": seed,
                "success": bool(result["success"]),
                "failure_reason": result["failure_reason"] or "",
                "objective_total": components.total,
                "objective_distance_raw": components.distance,
                "objective_direction_raw": components.direction,
                "objective_repulsion_raw": components.repulsion,
                "objective_distance_weighted": components.weighted_distance,
                "objective_direction_weighted": components.weighted_direction,
                "objective_repulsion_weighted": components.weighted_repulsion,
                "gradient_norm_inf": result["gradient_norm"],
                "iterations": result["iterations"],
                "function_evaluations": result["function_evaluations"],
            }
        )
        position_rows.extend(
            {
                "seed": seed,
                "label": label,
                "x_centered_sim": float(full_final[index, 0]),
                "y_centered_sim": float(full_final[index, 1]),
                "is_calibration_anchor": label in calibration_labels,
                "is_test_site_oracle_anchor": label in test_labels,
            }
            for index, label in enumerate(vertices)
        )

    runs = pd.DataFrame(rows)
    runs.to_csv(outdir / "oracle_bfgs_runs.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(position_rows).to_csv(
        outdir / "oracle_bfgs_final_positions_centered_sim.csv",
        index=False,
        encoding="utf-8-sig",
    )

    finite = runs[runs.get("objective_total", pd.Series(index=runs.index)).notna()]
    successful = finite[finite["success"] == True]  # noqa: E712
    summary = {
        "diagnostic_only": True,
        "held_out_data_used_as_hard_constraints": True,
        "eligible_for_formal_test_rmse_reporting": False,
        "oracle_test_RMSE_km": 0.0,
        "oracle_test_RMSE_note": "Zero by construction; not an evaluation result.",
        "formal_anchor_count_unchanged": formal_problem.n_vertices - formal_problem.n_free_vertices,
        "oracle_anchor_count": oracle_problem.n_vertices - oracle_problem.n_free_vertices,
        "calibration_labels": calibration_labels,
        "oracle_test_labels": test_labels,
        "seeds": seeds,
        "run_count": len(runs),
        "successful_run_count": len(successful),
        "failure_count": int(len(runs) - len(successful)),
        "best_successful_objective": (
            None if successful.empty else float(successful["objective_total"].min())
        ),
        "mean_successful_objective": (
            None if successful.empty else float(successful["objective_total"].mean())
        ),
        "best_finite_objective_including_failed_status": (
            None if finite.empty else float(finite["objective_total"].min())
        ),
    }
    (outdir / "oracle_bfgs_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def _parse_seeds(raw: str) -> list[int]:
    seeds = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must contain unique integer values.")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    args = parser.parse_args()
    summary = run_diagnostic(seeds=_parse_seeds(args.seeds), outdir=Path(args.outdir))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
