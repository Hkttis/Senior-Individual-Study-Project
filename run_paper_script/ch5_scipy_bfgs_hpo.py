"""BFGS-specific alpha/beta HPO with three-anchor LOO validation.

Each fold fixes two calibration anchors and evaluates the third.  Archaeological
test sites are never fixed, evaluated, or used for candidate selection here.
The selected weights can later be supplied to ``ch5-scipy-bfgs`` through its
``--hpo-outdir`` option for an independent 100-seed final experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from library.config import FILE_PATHS, km2pix, refer_pos_sim
from library.data_io import (
    load_ini_data_from_csv,
    uploading_directional_data,
)
from library.geometry import (
    get_lcc_bounds,
    get_lcc_parameters,
    lcc_transformation_with_anchor,
)
from library.metrics import calculate_kruskals_stress, direction_violation_rate
from library.scipy_hpo_objective import build_bfgs_hpo_fold_objective
from library.scipy_minimizer import DEFAULT_BFGS_GTOL, run_bfgs
from library.scipy_objective import FIXED_ANCHORS_SIM, ObjectiveWeights
from library.units import data_Li2sim, pos_matrix_sim2km
from run_paper_script.ch5_hparam_kfold_gridsearch_pareto import (
    _build_anchor_loo_folds,
    _make_alpha_beta_grid,
    _non_dominated_mask,
    _plot_heatmap,
    _plot_pareto_2d_projections,
    _plot_pareto_3d,
    _select_one_se_balanced_candidate,
)
from run_paper_script.ch5_scipy_bfgs import _initial_free_vector


DEFAULT_OUTDIR = "outputs/ch5_scipy_bfgs_hpo_smoke"
OBJECTIVE_COLUMNS = (
    "E_distance_stress_mean",
    "E_direction_vr_mean",
    "RMSE_anchor_LOO_mean_km",
)


def _parse_seeds(raw: str) -> list[int]:
    seeds = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must contain unique integer values.")
    return seeds


def _eligible_grid_points(grid: pd.DataFrame) -> pd.DataFrame:
    """Keep grid points represented by every LOO fold, despite failed runs."""

    finite = np.isfinite(grid[list(OBJECTIVE_COLUMNS)].to_numpy(float)).all(axis=1)
    represented = grid["all_folds_have_success"].astype(bool).to_numpy()
    return grid.loc[finite & represented].copy().reset_index(drop=True)


def _anchor_inputs(vertices: Sequence[str], dni: dict[str, int]):
    labels: list[str] = []
    lonlat: list[tuple[float, float]] = []
    test_labels: list[str] = []
    with open(FILE_PATHS["ground_truth_path"], newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            name = (row.get("model_name") or row.get("節點名稱") or "").strip()
            role = (row.get("use_role") or "").strip()
            if role in {"anchor", "anchor_align"}:
                labels.append(name)
                lonlat.append((float(row["lon"]), float(row["lat"])))
            elif role == "test":
                # Record names for the leakage audit, but never parse test lon/lat.
                test_labels.append(name)
    if len(labels) != 3 or set(labels) != set(FIXED_ANCHORS_SIM):
        raise ValueError("BFGS HPO requires exactly the current three calibration anchors.")
    if len(test_labels) != 8 or len(set(test_labels)) != 8:
        raise ValueError("BFGS HPO leakage audit requires eight unique test-site labels.")
    if any(label not in dni for label in labels):
        raise ValueError("A calibration anchor is absent from the formal graph.")
    if any(label not in dni for label in test_labels):
        raise ValueError("A test-site audit label is absent from the formal graph.")
    return labels, lonlat, test_labels


def _fold_targets_centered(
    *,
    vertices: Sequence[str],
    dni: dict[str, int],
    anchor_labels: Sequence[str],
    anchor_lonlat: Sequence[tuple[float, float]],
    frame_label: str,
) -> dict[str, np.ndarray]:
    """Project calibration anchors only; all test-site coordinates stay masked."""

    masked = [[0.0, 0.0] for _ in vertices]
    for label, lonlat in zip(anchor_labels, anchor_lonlat, strict=True):
        masked[dni[label]] = [float(lonlat[0]), float(lonlat[1])]
    projected_km = lcc_transformation_with_anchor(dni, masked, anchor_label=frame_label)
    return {
        label: np.asarray(projected_km[dni[label]], dtype=np.float64) * km2pix
        for label in anchor_labels
    }


def _sample_std(values: pd.Series) -> float:
    return float(values.std(ddof=1)) if len(values) > 1 else float("nan")


def _evaluate_fold_state(
    *,
    problem,
    y,
    heldout_label: str,
    heldout_target_centered: np.ndarray,
    dni,
    data_sim,
    directional_data,
) -> dict[str, float]:
    centered = problem.unpack(y)
    points = centered + np.asarray(refer_pos_sim, dtype=np.float64)
    heldout_error_km = float(
        np.linalg.norm(centered[dni[heldout_label]] - heldout_target_centered) / km2pix
    )
    components = problem.components(y)
    return {
        "E_distance_stress": float(
            calculate_kruskals_stress(dni, pos_matrix_sim2km(points.tolist()), data_sim)
        ),
        "E_direction_vr": float(direction_violation_rate(points, directional_data, dni)),
        "RMSE_anchor_LOO_km": heldout_error_km,
        "objective_total": components.total,
        "objective_distance_weighted": components.weighted_distance,
        "objective_direction_weighted": components.weighted_direction,
        "objective_repulsion_weighted": components.weighted_repulsion,
    }


def load_selected_bfgs_hpo_params(
    hpo_outdir: str | Path, *, allow_boundary: bool = False
) -> dict[str, float]:
    path = Path(hpo_outdir) / "bfgs_hpo_selected_candidate.csv"
    if not path.exists():
        raise FileNotFoundError(f"BFGS HPO selected candidate is missing: {path}")
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise ValueError(f"Expected exactly one BFGS HPO selected candidate: {path}")
    row = frame.iloc[0]
    raw_boundary = row.get("selected_on_grid_boundary", False)
    on_boundary = (
        str(raw_boundary).strip().lower() == "true"
        if isinstance(raw_boundary, str)
        else bool(raw_boundary)
    )
    if on_boundary and not allow_boundary:
        raise ValueError(
            "The selected BFGS HPO candidate lies on the search-grid boundary. "
            "Expand the HPO range before the formal 100-run experiment, or pass "
            "--allow-boundary-hpo only for an intentional diagnostic run."
        )
    return {
        "alpha": float(row["alpha"]),
        "beta": float(row["beta"]),
        "w_dis": float(row["w_dis"]),
    }


def run_bfgs_hpo(
    *,
    seeds: Sequence[int],
    alpha_min: float,
    alpha_max: float,
    alpha_step: float,
    beta_min: float,
    beta_max: float,
    beta_step: float,
    outdir: str | Path,
    w_dis: float = 1.0,
    gtol: float = DEFAULT_BFGS_GTOL,
) -> dict:
    outdir = Path(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f"BFGS HPO output directory is not empty: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)

    _graph, vertices, dni, _edges, distance_data = load_ini_data_from_csv(FILE_PATHS)
    anchor_labels, anchor_lonlat, test_labels = _anchor_inputs(vertices, dni)
    if set(test_labels) & set(anchor_labels):
        raise ValueError("Calibration and held-out test labels overlap.")
    folds = _build_anchor_loo_folds(anchor_labels, anchor_lonlat)
    alphas, betas = _make_alpha_beta_grid(
        alpha_min, alpha_max, alpha_step, beta_min, beta_max, beta_step
    )
    data_sim = data_Li2sim(distance_data)
    directional_data = uploading_directional_data()
    expected_runs_per_combo = len(folds) * len(seeds)

    run_rows: list[dict] = []
    fold_rows: list[dict] = []
    grid_rows: list[dict] = []
    total_combinations = len(alphas) * len(betas)
    combination = 0
    for alpha in alphas:
        for beta in betas:
            combination += 1
            weights = ObjectiveWeights.from_physics_hpo(
                alpha=float(alpha), beta=float(beta), w_dis=float(w_dis)
            )
            print(
                f"[{combination}/{total_combinations}] BFGS HPO "
                f"alpha={alpha:g}, beta={beta:g}"
            )
            current_fold_rows: list[dict] = []
            for fold in folds:
                targets = _fold_targets_centered(
                    vertices=vertices,
                    dni=dni,
                    anchor_labels=anchor_labels,
                    anchor_lonlat=anchor_lonlat,
                    frame_label=fold.train_anchor_label,
                )
                problem = build_bfgs_hpo_fold_objective(
                    fixed_anchor_positions_sim={
                        label: targets[label] for label in fold.train_labels
                    },
                    weights=weights,
                )
                fold_run_rows: list[dict] = []
                for seed in seeds:
                    initial = _initial_free_vector(
                        int(seed),
                        problem,
                        vertices,
                        dni,
                        fold.train_labels,
                        fold.train_lonlat,
                        fold.train_anchor_label,
                    )
                    result = run_bfgs(initial, problem, gtol=gtol)
                    selected_y = result.get("y_final")
                    metrics = None
                    if selected_y is not None:
                        metrics = _evaluate_fold_state(
                            problem=problem,
                            y=selected_y,
                            heldout_label=fold.heldout_label,
                            heldout_target_centered=targets[fold.heldout_label],
                            dni=dni,
                            data_sim=data_sim,
                            directional_data=directional_data,
                        )
                    row = {
                        "alpha": float(alpha),
                        "beta": float(beta),
                        "w_dis": float(w_dis),
                        "w_dir": float(10.0 ** float(alpha)),
                        "w_reg": float(10.0 ** float(beta)),
                        "fold_id": int(fold.fold_id),
                        "train_labels": "|".join(fold.train_labels),
                        "frame_anchor_label": fold.train_anchor_label,
                        "heldout_anchor_label": fold.heldout_label,
                        "seed": int(seed),
                        "optimizer_success": bool(result["success"]),
                        "failure_reason": result["failure_reason"] or "",
                        "optimization_dimension": int(problem.dimension),
                        "iterations": result["iterations"],
                        "gradient_norm_inf": result["gradient_norm"],
                    }
                    for key in (
                        "E_distance_stress",
                        "E_direction_vr",
                        "RMSE_anchor_LOO_km",
                        "objective_total",
                        "objective_distance_weighted",
                        "objective_direction_weighted",
                        "objective_repulsion_weighted",
                    ):
                        row[key] = float("nan") if metrics is None else metrics[key]
                    run_rows.append(row)
                    fold_run_rows.append(row)

                fold_frame = pd.DataFrame(fold_run_rows)
                ok = fold_frame[fold_frame["optimizer_success"] == True]  # noqa: E712
                fold_summary = {
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "w_dis": float(w_dis),
                    "fold_id": int(fold.fold_id),
                    "train_labels": "|".join(fold.train_labels),
                    "frame_anchor_label": fold.train_anchor_label,
                    "heldout_anchor_label": fold.heldout_label,
                    "n_expected_runs": len(seeds),
                    "n_successful_runs": len(ok),
                    "n_failed_runs": len(seeds) - len(ok),
                }
                for source_col, output_prefix in (
                    ("E_distance_stress", "E_distance_stress"),
                    ("E_direction_vr", "E_direction_vr"),
                    ("RMSE_anchor_LOO_km", "RMSE_anchor_LOO"),
                ):
                    fold_summary[f"{output_prefix}_mean" + ("_km" if "RMSE" in output_prefix else "")] = (
                        float(ok[source_col].mean()) if not ok.empty else float("nan")
                    )
                    fold_summary[f"{output_prefix}_std" + ("_km" if "RMSE" in output_prefix else "")] = (
                        _sample_std(ok[source_col]) if not ok.empty else float("nan")
                    )
                fold_rows.append(fold_summary)
                current_fold_rows.append(fold_summary)

            fold_frame = pd.DataFrame(current_fold_rows)
            complete = int(fold_frame["n_successful_runs"].sum()) == expected_runs_per_combo
            min_successful_per_fold = int(fold_frame["n_successful_runs"].min())
            grid_rows.append(
                {
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "w_dis": float(w_dis),
                    "w_dir": float(10.0 ** float(alpha)),
                    "w_reg": float(10.0 ** float(beta)),
                    "effective_distance_weight": weights.distance,
                    "effective_direction_weight": weights.direction,
                    "effective_repulsion_weight": weights.repulsion,
                    "n_folds": len(folds),
                    "n_seeds_per_fold": len(seeds),
                    "n_expected_runs": expected_runs_per_combo,
                    "n_successful_runs": int(fold_frame["n_successful_runs"].sum()),
                    "n_failed_runs": int(fold_frame["n_failed_runs"].sum()),
                    "is_complete": complete,
                    "success_rate": float(
                        fold_frame["n_successful_runs"].sum() / expected_runs_per_combo
                    ),
                    "min_successful_runs_per_fold": min_successful_per_fold,
                    "all_folds_have_success": min_successful_per_fold > 0,
                    "E_distance_stress_mean": float(fold_frame["E_distance_stress_mean"].mean()),
                    "E_distance_stress_std": _sample_std(fold_frame["E_distance_stress_mean"]),
                    "E_direction_vr_mean": float(fold_frame["E_direction_vr_mean"].mean()),
                    "E_direction_vr_std": _sample_std(fold_frame["E_direction_vr_mean"]),
                    "RMSE_anchor_LOO_mean_km": float(fold_frame["RMSE_anchor_LOO_mean_km"].mean()),
                    "RMSE_anchor_LOO_std_km": _sample_std(fold_frame["RMSE_anchor_LOO_mean_km"]),
                }
            )

    runs = pd.DataFrame(run_rows)
    fold_summary = pd.DataFrame(fold_rows)
    grid = pd.DataFrame(grid_rows).sort_values(["alpha", "beta"]).reset_index(drop=True)
    eligible = _eligible_grid_points(grid)
    if eligible.empty:
        runs.to_csv(outdir / "bfgs_hpo_runs.csv", index=False, encoding="utf-8-sig")
        fold_summary.to_csv(outdir / "bfgs_hpo_fold_summary.csv", index=False, encoding="utf-8-sig")
        grid.to_csv(outdir / "bfgs_hpo_grid_summary.csv", index=False, encoding="utf-8-sig")
        raise RuntimeError(
            "No BFGS HPO grid point has finite metrics and at least one successful run "
            "in every anchor LOO fold."
        )

    pareto_mask = _non_dominated_mask(eligible[list(OBJECTIVE_COLUMNS)].to_numpy(float))
    eligible["is_pareto"] = pareto_mask
    pareto = eligible[eligible["is_pareto"]].copy()
    selected, selection_meta = _select_one_se_balanced_candidate(pareto, OBJECTIVE_COLUMNS)
    selected_on_alpha_boundary = bool(
        np.isclose(float(selected["alpha"]), float(alpha_min))
        or np.isclose(float(selected["alpha"]), float(alpha_max))
    )
    selected_on_beta_boundary = bool(
        np.isclose(float(selected["beta"]), float(beta_min))
        or np.isclose(float(selected["beta"]), float(beta_max))
    )
    selected_on_grid_boundary = selected_on_alpha_boundary or selected_on_beta_boundary
    boundary_meta = {
        "selected_on_alpha_boundary": selected_on_alpha_boundary,
        "selected_on_beta_boundary": selected_on_beta_boundary,
        "selected_on_grid_boundary": selected_on_grid_boundary,
        "boundary_action": (
            "expand_grid_before_formal_run" if selected_on_grid_boundary else "none"
        ),
    }
    selected_frame = pd.DataFrame(
        [{**selected.to_dict(), **selection_meta, **boundary_meta}]
    )

    runs.to_csv(outdir / "bfgs_hpo_runs.csv", index=False, encoding="utf-8-sig")
    fold_summary.to_csv(outdir / "bfgs_hpo_fold_summary.csv", index=False, encoding="utf-8-sig")
    grid.to_csv(outdir / "bfgs_hpo_grid_summary.csv", index=False, encoding="utf-8-sig")
    pareto.to_csv(outdir / "bfgs_hpo_pareto_front.csv", index=False, encoding="utf-8-sig")
    selected_frame.to_csv(
        outdir / "bfgs_hpo_selected_candidate.csv", index=False, encoding="utf-8-sig"
    )

    _plot_heatmap(eligible, "RMSE_anchor_LOO_mean_km", outdir / "sensitivity_rmse_anchor_loo.png", "BFGS HPO: Anchor LOO RMSE", selected)
    _plot_heatmap(eligible, "E_distance_stress_mean", outdir / "sensitivity_stress.png", "BFGS HPO: Stress", selected)
    _plot_heatmap(eligible, "E_direction_vr_mean", outdir / "sensitivity_violation_rate.png", "BFGS HPO: Violation Rate", selected)
    _plot_pareto_3d(eligible, pareto_mask, outdir / "pareto_front_3d.png", selected)
    _plot_pareto_2d_projections(eligible, pareto_mask, outdir / "pareto_front_2d.png", selected)

    config = {
        "method": "SciPy full-memory BFGS HPO",
        "validation": "three_anchor_leave_one_anchor_out",
        "optimization_dimension_per_fold": 66,
        "formal_final_dimension": 64,
        "fold_fixed_anchor_count": 2,
        "formal_final_anchor_count": 3,
        "selection_objectives": list(OBJECTIVE_COLUMNS),
        "selection_rule": selection_meta["selection_rule"],
        "selection_population_policy": (
            "include grid points with finite objectives and at least one successful "
            "run in every anchor LOO fold; retain failure counts"
        ),
        "test_site_policy": "not used in HPO optimization, validation, or selection",
        "anchor_labels": anchor_labels,
        "test_labels_recorded_for_leakage_audit_only": test_labels,
        "seeds": list(map(int, seeds)),
        "alpha_range": [alpha_min, alpha_max, alpha_step],
        "beta_range": [beta_min, beta_max, beta_step],
        "w_dis": float(w_dis),
        "gtol": float(gtol),
        "lcc_bounds": dict(zip(("lon_min", "lon_max", "lat_min", "lat_max"), map(float, get_lcc_bounds()))),
        "lcc_parameters": dict(zip(("lat_1", "lat_2", "lon_0"), map(float, get_lcc_parameters()))),
        "input_files": {
            "distance": str(FILE_PATHS["chen_data"]),
            "direction": str(FILE_PATHS["directional_data"]),
            "site_roles": str(FILE_PATHS["ground_truth_path"]),
        },
        "failure_count": int((~runs["optimizer_success"]).sum()),
        "eligible_grid_count": int(len(eligible)),
        "complete_grid_count": int(grid["is_complete"].sum()),
    }
    (outdir / "bfgs_hpo_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (outdir / "bfgs_hpo_selection.json").write_text(
        json.dumps(
            {
                "alpha": float(selected["alpha"]),
                "beta": float(selected["beta"]),
                "w_dis": float(selected["w_dis"]),
                **selection_meta,
                **boundary_meta,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "outdir": str(outdir),
        "selected_alpha": float(selected["alpha"]),
        "selected_beta": float(selected["beta"]),
        "pareto_count": int(len(pareto)),
        "eligible_grid_count": int(len(eligible)),
        "complete_grid_count": int(grid["is_complete"].sum()),
        "failure_count": int((~runs["optimizer_success"]).sum()),
        **boundary_meta,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--alpha-min", type=float, default=1.0)
    parser.add_argument("--alpha-max", type=float, default=1.0)
    parser.add_argument("--alpha-step", type=float, default=1.0)
    parser.add_argument("--beta-min", type=float, default=-0.5)
    parser.add_argument("--beta-max", type=float, default=-0.5)
    parser.add_argument("--beta-step", type=float, default=1.0)
    parser.add_argument("--w-dis", type=float, default=1.0)
    parser.add_argument("--gtol", type=float, default=DEFAULT_BFGS_GTOL)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    args = parser.parse_args()
    result = run_bfgs_hpo(
        seeds=_parse_seeds(args.seeds),
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        alpha_step=args.alpha_step,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        beta_step=args.beta_step,
        outdir=args.outdir,
        w_dis=args.w_dis,
        gtol=args.gtol,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
