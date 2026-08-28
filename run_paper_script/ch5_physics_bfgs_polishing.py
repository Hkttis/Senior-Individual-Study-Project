"""Polish PhysicsSim-Full endpoints with full-memory BFGS and audit trajectories.

Held-out test coordinates are used only after each accepted state is produced;
they never enter the objective, BFGS line search, callback return value, stopping
rule, or stratum definition.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.family"] = "Noto Sans TC"

from library.config import (
    DIRECTIONAL_FORCE_MAGNITUDE_BASE,
    FILE_PATHS,
    REPULSION_STRENGTH_BASE,
    SPRING_STIFFNESS_BASE,
    km2pix,
    refer_pos_sim,
)
from library.data_io import (
    get_anchor_align_label,
    get_anchor_labels,
    get_test_site_labels,
    load_ini_data_from_csv,
    uploading_directional_data,
    uploading_ground_truth,
)
from library.initialization import generate_CHEN_initial_positions
from library.metrics import (
    calculate_kruskals_stress,
    direction_violation_rate,
    mean_angular_error_violations,
)
from library.physics import main_physics_simulation
from library.progressive_alignment import place_in_anchor_frame
from library.scipy_diagnostics import (
    anchor_centroid_radius_rms_km,
    assign_objective_strata,
    graph_node_diagnostics,
    reinsert_exact_anchors,
    test_site_radial_errors,
    weighted_gradient_components,
)
from library.scipy_minimizer import DEFAULT_BFGS_GTOL, run_bfgs
from library.scipy_objective import (
    ObjectiveWeights,
    build_current_objective,
)
from library.units import data_Li2sim, pos_matrix_sim2km
from run_paper_script.ch5_ablation_progressive import (
    PHYSICS_VARIANTS,
    _physics_forces,
    _target_positions_sim,
)
DEFAULT_AS_OUTDIR = "outputs/ch5_progressive_as_physics_alpha_1_beta_-0.5_dc_alpha_-2_wang_current_100seeds_random1000_20260721"
DEFAULT_REFERENCE_BFGS = "outputs/ch5_scipy_bfgs_full_100seeds_20260821"
DEFAULT_OUTDIR = "outputs/ch5_physics_to_bfgs_polishing_smoke"
CORE_METRICS = (
    "RMSE_test_km_posthoc",
    "E_distance_stress",
    "E_direction_vr",
    "E_direction_mae",
    "objective_total",
    "objective_distance_weighted",
    "objective_direction_weighted",
    "objective_repulsion_weighted",
    "gradient_norm_inf",
    "anchor_centroid_radius_rms_km",
    "peripheral_radial_error_mean_km",
)


def _ascii_scientific_tick(value, _position):
    return "0" if value == 0 else f"{value:.0e}"


def _parse_seeds(raw: str) -> list[int]:
    seeds = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must contain unique integer values.")
    return seeds


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required provenance file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_source_endpoints(
    as_outdir: Path, seeds: Sequence[int], vertices: Sequence[str]
) -> tuple[dict[int, np.ndarray], pd.DataFrame, dict]:
    config = _load_json(as_outdir / "progressive_config.json")
    positions = pd.read_csv(as_outdir / "progressive_final_positions_y_up_sim.csv")
    runs = pd.read_csv(as_outdir / "progressive_runs_by_seed.csv")
    positions = positions[positions["variant"] == "PhysicsSim-Full"]
    runs = runs[(runs["variant"] == "PhysicsSim-Full") & (runs["status"] == "ok")]
    result: dict[int, np.ndarray] = {}
    for seed in seeds:
        selected = positions[positions["seed"] == seed].set_index("label")
        missing = [label for label in vertices if label not in selected.index]
        if missing:
            raise ValueError(f"PhysicsSim-Full seed {seed} is missing positions: {missing}")
        if not (runs["seed"] == seed).any():
            raise ValueError(f"PhysicsSim-Full seed {seed} has no successful AS metrics row.")
        result[int(seed)] = selected.loc[
            list(vertices), ["x_y_up_sim", "y_y_up_sim"]
        ].to_numpy(float)
    return result, runs, config


def _source_objective_weights(
    source_runs: pd.DataFrame, as_config: Mapping[str, object]
) -> tuple[ObjectiveWeights, float, float, float]:
    """Recover and verify the effective weights used by the formal AS runs."""
    columns = ("spring_stiffness", "directional_force", "repulsion_strength")
    missing = [column for column in columns if column not in source_runs.columns]
    if missing:
        raise ValueError(f"PhysicsSim AS runs are missing weight columns: {missing}")

    recorded: dict[str, float] = {}
    for column in columns:
        values = pd.to_numeric(source_runs[column], errors="coerce").to_numpy(float)
        if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError(f"PhysicsSim AS {column} must be finite and strictly positive.")
        if not np.allclose(values, values[0], rtol=0.0, atol=1e-12):
            raise ValueError(f"PhysicsSim AS runs do not share one fixed {column}.")
        recorded[column] = float(values[0])

    w_dis = recorded["spring_stiffness"] / float(SPRING_STIFFNESS_BASE)
    alpha = float(
        np.log10(
            recorded["directional_force"]
            / (float(DIRECTIONAL_FORCE_MAGNITUDE_BASE) * w_dis)
        )
    )
    beta = float(
        np.log10(
            recorded["repulsion_strength"]
            / (float(REPULSION_STRENGTH_BASE) * w_dis)
        )
    )
    if "alpha" not in as_config or "beta" not in as_config:
        raise ValueError("PhysicsSim AS config must record the manually selected alpha and beta.")
    configured = np.asarray([as_config["alpha"], as_config["beta"]], dtype=float)
    recovered = np.asarray([alpha, beta], dtype=float)
    if not np.all(np.isfinite(configured)) or not np.allclose(
        configured, recovered, rtol=0.0, atol=1e-12
    ):
        raise ValueError(
            "PhysicsSim AS config alpha/beta do not reproduce the effective weights "
            "recorded in the formal PhysicsSim-Full runs."
        )

    weights = ObjectiveWeights(
        distance=recorded["spring_stiffness"],
        direction=recorded["directional_force"],
        repulsion=recorded["repulsion_strength"],
    )
    return weights, alpha, beta, w_dis


def _reject_external_polishing_hpo(
    hpo_outdir: str | None, allow_boundary_hpo: bool
) -> None:
    if hpo_outdir is not None or allow_boundary_hpo:
        raise ValueError(
            "BFGS polishing must use the exact effective weights recorded by the "
            "formal PhysicsSim AS source. Independent BFGS HPO is not permitted."
        )


def _validate_reference_weights(reference_outdir: Path, alpha: float, beta: float, w_dis: float):
    config = _load_json(reference_outdir / "bfgs_experiment_config.json")
    expected = np.asarray([alpha, beta, w_dis], dtype=float)
    actual = np.asarray([config["alpha"], config["beta"], config["w_dis"]], dtype=float)
    if not np.allclose(actual, expected, rtol=0.0, atol=1e-12):
        raise ValueError(
            "Objective strata reference uses different BFGS weights. Run the random-start "
            "BFGS experiment with the source PhysicsSim weights before polishing."
        )
    runs = pd.read_csv(reference_outdir / "bfgs_runs_by_seed.csv")
    ok = runs[(runs["status"] == "ok") & runs["objective_final"].notna()]
    if len(ok) < 4:
        raise ValueError("The BFGS stratum reference has fewer than four successful runs.")
    return ok["objective_final"].to_numpy(float), config


def _state_metrics(
    *,
    centered: np.ndarray,
    problem,
    dni,
    data_sim,
    directional_data,
    test_labels,
    targets_centered,
) -> dict[str, float | str]:
    y = problem.pack(centered)
    components = problem.components(y)
    _objective_value, gradient = problem.fun_and_jac(y)
    points = centered + np.asarray(refer_pos_sim, dtype=float)
    errors_km = np.asarray(
        [
            np.linalg.norm(centered[dni[label]] - targets_centered[label]) / km2pix
            for label in test_labels
        ],
        dtype=float,
    )
    radial_errors = test_site_radial_errors(
        centered_positions=centered,
        target_centered=targets_centered,
        test_labels=test_labels,
        dni=dni,
        problem=problem,
    )
    return {
        "RMSE_test_km_posthoc": float(np.sqrt(np.mean(errors_km**2))),
        "E_distance_stress": float(
            calculate_kruskals_stress(dni, pos_matrix_sim2km(points.tolist()), data_sim)
        ),
        "E_direction_vr": float(direction_violation_rate(points, directional_data, dni)),
        "E_direction_mae": float(
            mean_angular_error_violations(points, directional_data, dni)
        ),
        "objective_total": components.total,
        "objective_distance_weighted": components.weighted_distance,
        "objective_direction_weighted": components.weighted_direction,
        "objective_repulsion_weighted": components.weighted_repulsion,
        "gradient_norm_inf": float(np.linalg.norm(gradient, ord=np.inf)),
        "anchor_centroid_radius_rms_km": anchor_centroid_radius_rms_km(centered, problem),
        "peripheral_radial_error_mean_km": float(np.mean(list(radial_errors.values()))),
        "peripheral_radial_error_rms_km": float(
            np.sqrt(np.mean(np.square(list(radial_errors.values()))))
        ),
        "calibration_anchor_LOOCV_RMSE_km": float("nan"),
        "calibration_anchor_metric_policy": "not_applicable_fixed_in_this_run",
        "test_metric_policy": "posthoc_only_never_used_for_optimization_or_stopping",
    }


def _peripheral_force_rows(
    *,
    seed: int,
    stage: str,
    centered: np.ndarray,
    problem,
    dni,
    test_labels,
    targets_centered,
    node_graph,
) -> list[dict]:
    y = problem.pack(centered)
    gradients = weighted_gradient_components(problem, y)
    centroid = problem.anchor_coordinates.mean(axis=0)
    radial_errors = test_site_radial_errors(
        centered_positions=centered,
        target_centered=targets_centered,
        test_labels=test_labels,
        dni=dni,
        problem=problem,
    )
    rows = []
    for label in test_labels:
        index = dni[label]
        target = targets_centered[label]
        radial_unit = (target - centroid) / np.linalg.norm(target - centroid)
        row = {
            "seed": int(seed),
            "stage": stage,
            "label": label,
            **node_graph[label],
            "ground_truth_radius_from_anchor_centroid_km": float(
                np.linalg.norm(target - centroid) / km2pix
            ),
            "predicted_radius_from_anchor_centroid_km": float(
                np.linalg.norm(centered[index] - centroid) / km2pix
            ),
            "radial_error_km": radial_errors[label],
        }
        for term, gradient in gradients.items():
            vector = gradient[index]
            row[f"{term}_gradient_magnitude"] = float(np.linalg.norm(vector))
            row[f"{term}_gradient_outward_component"] = float(np.dot(vector, radial_unit))
            row[f"{term}_force_outward_component"] = float(-np.dot(vector, radial_unit))
        rows.append(row)
    return rows


def _sample_steps(history_length: int, stride: int) -> list[int]:
    if stride <= 0:
        raise ValueError("Physics trajectory stride must be positive.")
    steps = list(range(stride, history_length + 1, stride))
    if history_length not in steps:
        steps.append(history_length)
    return [0, *steps]


def _rerun_physics_trajectory(
    *,
    seed: int,
    stride: int,
    source_endpoint: np.ndarray,
    problem,
    vertices,
    dni,
    calibration_labels,
    calibration_lonlat,
    anchor_label,
    data_sim,
    directional_data,
    targets_centered,
    test_labels,
    alpha: float,
    beta: float,
) -> tuple[list[dict], float]:
    np.random.seed(seed)
    generated_vertices, generated_dni, data_li, initial, fixed_positions = (
        generate_CHEN_initial_positions(
            list(refer_pos_sim),
            list(calibration_labels),
            list(calibration_lonlat),
            anchor_label=anchor_label,
        )
    )
    if generated_vertices != list(vertices) or generated_dni != dni:
        raise ValueError("Physics trajectory rerun uses a different vertex order.")
    initial_saved = np.asarray(initial, dtype=float).copy()
    _w_dir, _w_reg, spring, directional, repulsion = _physics_forces(
        PHYSICS_VARIANTS["PhysicsSim-Full"], alpha, beta
    )
    _wrong, _stress, history, final = main_physics_simulation(
        vertices,
        dni,
        data_Li2sim(data_li),
        initial,
        directional_data,
        fixed_positions,
        spring,
        repulsion,
        directional,
        plot=False,
    )
    aligned_final = np.asarray(
        place_in_anchor_frame(final, dni, anchor_label, refer_pos_sim), dtype=float
    )
    reproduction_max_abs_diff = float(np.max(np.abs(aligned_final - source_endpoint)))
    if reproduction_max_abs_diff > 1e-7:
        raise ValueError(
            f"PhysicsSim seed {seed} endpoint rerun differs from AS by "
            f"{reproduction_max_abs_diff:.6g} simulation units."
        )

    rows: list[dict] = []
    for step in _sample_steps(len(history), stride):
        raw = initial_saved if step == 0 else np.asarray(history[step - 1], dtype=float)
        framed = np.asarray(
            place_in_anchor_frame(raw, dni, anchor_label, refer_pos_sim), dtype=float
        )
        centered, anchor_drift = reinsert_exact_anchors(framed, problem)
        rows.append(
            {
                "seed": int(seed),
                "stage": "PhysicsSim",
                "stage_iteration": int(step),
                "accepted_state": int(step),
                "anchor_drift_max_abs_sim_before_reinsertion": anchor_drift,
                **_state_metrics(
                    centered=centered,
                    problem=problem,
                    dni=dni,
                    data_sim=data_sim,
                    directional_data=directional_data,
                    test_labels=test_labels,
                    targets_centered=targets_centered,
                ),
            }
        )
    return rows, reproduction_max_abs_diff


def _endpoint_summary_row(seed: int, before: dict, after: dict, result: dict, drift: float):
    row = {
        "seed": int(seed),
        "optimizer_success": bool(result["success"]),
        "failure_reason": result["failure_reason"] or "",
        "optimizer_iterations": result["iterations"],
        "optimizer_function_evaluations": result["function_evaluations"],
        "anchor_drift_max_abs_sim_before_reinsertion": drift,
    }
    for metric in CORE_METRICS:
        row[f"before_{metric}"] = before[metric]
        row[f"after_{metric}"] = after[metric]
        row[f"delta_{metric}"] = after[metric] - before[metric]
    return row


def _spearman_outputs(force_frame: pd.DataFrame, outdir: Path):
    summary = (
        force_frame.groupby(["stage", "label"], as_index=False)
        .mean(numeric_only=True)
    )
    summary.to_csv(outdir / "peripheral_node_summary.csv", index=False, encoding="utf-8-sig")
    predictors = [
        "distance_edge_degree",
        "direction_edge_degree",
        "distance_graph_hops_to_nearest_anchor",
        "ground_truth_radius_from_anchor_centroid_km",
        "repulsion_gradient_magnitude",
        "repulsion_force_outward_component",
        "distance_gradient_magnitude",
        "direction_gradient_magnitude",
    ]
    def correlation_row(group, predictor):
        if group[predictor].nunique() < 2 or group["radial_error_km"].nunique() < 2:
            return float("nan"), float("nan"), "undefined_constant_input"
        result = spearmanr(group[predictor], group["radial_error_km"])
        return float(result.statistic), float(result.pvalue), "ok"

    rows = []
    for stage, group in summary.groupby("stage"):
        for predictor in predictors:
            rho, p_value, status = correlation_row(group, predictor)
            rows.append(
                {
                    "stage": stage,
                    "analysis_unit": "eight_test_site_seed_means",
                    "predictor": predictor,
                    "outcome": "radial_error_km",
                    "n_nodes": len(group),
                    "spearman_rho": rho,
                    "p_value_diagnostic_only": p_value,
                    "status": status,
                }
            )
    pd.DataFrame(rows).to_csv(
        outdir / "peripheral_spearman_node_means.csv", index=False, encoding="utf-8-sig"
    )
    per_seed_rows = []
    for (stage, seed), group in force_frame.groupby(["stage", "seed"]):
        for predictor in predictors:
            rho, p_value, status = correlation_row(group, predictor)
            per_seed_rows.append(
                {
                    "stage": stage,
                    "seed": int(seed),
                    "analysis_unit": "eight_test_sites_within_seed",
                    "predictor": predictor,
                    "outcome": "radial_error_km",
                    "n_nodes": len(group),
                    "spearman_rho": rho,
                    "p_value_diagnostic_only": p_value,
                    "status": status,
                }
            )
    pd.DataFrame(per_seed_rows).to_csv(
        outdir / "peripheral_spearman_by_seed.csv", index=False, encoding="utf-8-sig"
    )
    return summary


def _save_plot(fig, outdir: Path, stem: str) -> None:
    fig.savefig(outdir / f"{stem}.png", dpi=300)
    fig.savefig(outdir / f"{stem}.svg")


def _make_plots(
    trajectory: pd.DataFrame,
    force_summary: pd.DataFrame,
    endpoints: pd.DataFrame,
    outdir: Path,
):
    show_legend = trajectory["seed"].nunique() <= 8
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), constrained_layout=True)
    for (seed, stage), group in trajectory.groupby(["seed", "stage"]):
        label = f"{stage} seed {seed}" if show_legend else None
        alpha = 0.65 if show_legend else 0.12
        axes[0].plot(group["stage_iteration"], group["objective_total"], alpha=alpha, label=label)
        axes[1].plot(group["stage_iteration"], group["RMSE_test_km_posthoc"], alpha=alpha, label=label)
    axes[0].set(ylabel="Total objective", title="Objective trajectory")
    axes[0].set_yscale("symlog", linthresh=1e5)
    axes[0].yaxis.set_major_formatter(FuncFormatter(_ascii_scientific_tick))
    axes[1].set(xlabel="Stage iteration", ylabel="Held-out test RMSE (km)", title="Post-hoc RMSE trajectory")
    axes[0].grid(alpha=0.25)
    axes[1].grid(alpha=0.25)
    if show_legend:
        axes[0].legend(fontsize=7, ncol=2)
    _save_plot(fig, outdir, "trajectory_objective_and_test_rmse")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True)
    for (seed, stage), group in trajectory.groupby(["seed", "stage"]):
        label = f"{stage} seed {seed}" if show_legend else None
        ax.plot(group["objective_total"], group["RMSE_test_km_posthoc"], marker=".", ms=2, alpha=0.6 if show_legend else 0.1, label=label)
    ax.set(xlabel="Total objective", ylabel="Held-out test RMSE (km; post-hoc only)", title="Objective versus external RMSE")
    ax.set_xscale("symlog", linthresh=1e5)
    ax.xaxis.set_major_formatter(FuncFormatter(_ascii_scientific_tick))
    ax.grid(alpha=0.25)
    if show_legend:
        ax.legend(fontsize=7)
    _save_plot(fig, outdir, "objective_vs_test_rmse")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 6.5), constrained_layout=True)
    for row in endpoints.itertuples(index=False):
        ax.plot(
            [row.before_objective_total, row.after_objective_total],
            [row.before_RMSE_test_km_posthoc, row.after_RMSE_test_km_posthoc],
            color="#7f7f7f",
            alpha=0.18,
            linewidth=0.8,
            zorder=1,
        )
    ax.scatter(
        endpoints["before_objective_total"],
        endpoints["before_RMSE_test_km_posthoc"],
        s=32,
        alpha=0.72,
        label="PhysicsSim endpoint",
        color="#1f77b4",
        zorder=2,
    )
    ax.scatter(
        endpoints["after_objective_total"],
        endpoints["after_RMSE_test_km_posthoc"],
        s=32,
        alpha=0.72,
        label="BFGS polished",
        color="#ff7f0e",
        zorder=3,
    )
    ax.set_xscale("symlog", linthresh=1e5)
    ax.set(
        xlabel="Total objective under the source PhysicsSim weights",
        ylabel="Held-out test RMSE (km; post-hoc only)",
        title="PhysicsSim to BFGS Polishing: Objective and External RMSE",
    )
    ax.grid(alpha=0.22)
    ax.legend()
    _save_plot(fig, outdir, "polishing_endpoint_objective_vs_test_rmse")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 6.5), constrained_layout=True)
    for (seed, stage), group in trajectory.groupby(["seed", "stage"]):
        label = f"{stage} seed {seed}" if show_legend else None
        ax.plot(group["anchor_centroid_radius_rms_km"], group["objective_repulsion_weighted"], marker=".", ms=2, alpha=0.6 if show_legend else 0.1, label=label)
    ax.set(xlabel="RMS radius from anchor centroid (km)", ylabel="Weighted repulsion objective", title="Repulsion and configuration spread")
    ax.yaxis.set_major_formatter(FuncFormatter(_ascii_scientific_tick))
    ax.grid(alpha=0.25)
    if show_legend:
        ax.legend(fontsize=7)
    _save_plot(fig, outdir, "repulsion_vs_spread")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 9), constrained_layout=True)
    component_columns = (
        ("objective_distance_weighted", "Distance"),
        ("objective_direction_weighted", "Direction"),
        ("objective_repulsion_weighted", "Repulsion"),
    )
    for (seed, stage), group in trajectory.groupby(["seed", "stage"]):
        alpha = 0.7 if show_legend else 0.1
        for column, component_label in component_columns:
            label = f"{stage} seed {seed}: {component_label}" if show_legend else None
            axes[0].plot(group["stage_iteration"], group[column], alpha=alpha, label=label)
        label = f"{stage} seed {seed}" if show_legend else None
        axes[1].plot(group["stage_iteration"], group["gradient_norm_inf"], alpha=alpha, label=label)
    axes[0].set(ylabel="Weighted objective component", title="Objective components")
    axes[0].set_yscale("symlog", linthresh=1e5)
    axes[0].yaxis.set_major_formatter(FuncFormatter(_ascii_scientific_tick))
    axes[1].set(
        xlabel="Stage iteration",
        ylabel="Gradient infinity norm",
        title="Gradient convergence diagnostic",
    )
    axes[1].set_yscale("log")
    axes[1].yaxis.set_major_formatter(FuncFormatter(_ascii_scientific_tick))
    axes[0].grid(alpha=0.25)
    axes[1].grid(alpha=0.25)
    if show_legend:
        axes[0].legend(fontsize=7, ncol=2)
        axes[1].legend(fontsize=7, ncol=2)
    _save_plot(fig, outdir, "trajectory_components_and_gradient")
    plt.close(fig)

    predictors = [
        ("distance_edge_degree", "Distance-edge degree"),
        ("distance_graph_hops_to_nearest_anchor", "Distance-graph hops to nearest anchor"),
        ("repulsion_force_outward_component", "Repulsion outward force component"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for ax, (column, label) in zip(axes, predictors):
        for stage, group in force_summary.groupby("stage"):
            ax.scatter(group[column], group["radial_error_km"], label=stage, s=45)
            for _, row in group.iterrows():
                ax.annotate(row["label"], (row[column], row["radial_error_km"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set(xlabel=label, ylabel="Radial error (km)")
        ax.yaxis.set_major_formatter(FuncFormatter(_ascii_scientific_tick))
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    _save_plot(fig, outdir, "peripheral_force_balance_diagnostics")
    plt.close(fig)


def run_polishing(
    *,
    seeds: Sequence[int],
    as_outdir: str | Path,
    reference_bfgs_outdir: str | Path,
    outdir: str | Path,
    hpo_outdir: str | None = None,
    allow_boundary_hpo: bool = False,
    gtol: float = DEFAULT_BFGS_GTOL,
    rerun_physics_trajectory: bool = False,
    physics_stride: int = 25,
    make_plots: bool = True,
) -> dict:
    _reject_external_polishing_hpo(hpo_outdir, allow_boundary_hpo)
    outdir = Path(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f"Polishing output directory is not empty: {outdir}")

    _graph, vertices, dni, _edges, distance_rows = load_ini_data_from_csv(FILE_PATHS)
    source_endpoints, source_runs, as_config = _load_source_endpoints(
        Path(as_outdir), seeds, vertices
    )
    weights, alpha, beta, w_dis = _source_objective_weights(source_runs, as_config)
    problem = build_current_objective(weights=weights)
    if tuple(vertices) != problem.vertices:
        raise ValueError("AS and objective vertex orders differ.")
    reference_values, reference_config = _validate_reference_weights(
        Path(reference_bfgs_outdir), alpha, beta, w_dis
    )
    outdir.mkdir(parents=True, exist_ok=True)

    gt_lonlat = uploading_ground_truth(vertices, dni)
    calibration_labels = get_anchor_labels()
    test_labels = get_test_site_labels()
    anchor_label = get_anchor_align_label()
    targets_y_up = _target_positions_sim(dni, gt_lonlat, anchor_label, refer_pos_sim)
    targets_centered = {
        label: np.asarray(targets_y_up[label], dtype=float) - np.asarray(refer_pos_sim, dtype=float)
        for label in test_labels
    }
    calibration_lonlat = [tuple(gt_lonlat[dni[label]]) for label in calibration_labels]
    data_sim = data_Li2sim(distance_rows)
    directional_data = uploading_directional_data()
    node_graph = graph_node_diagnostics(
        vertices=vertices,
        dni=dni,
        distance_rows=distance_rows,
        direction_rows=directional_data,
        anchor_labels=calibration_labels,
    )

    endpoint_rows: list[dict] = []
    trajectory_rows: list[dict] = []
    position_rows: list[dict] = []
    force_rows: list[dict] = []
    physics_reproduction: dict[int, float] = {}
    after_objectives: list[float] = []

    for seed in seeds:
        print(f"[PhysicsSim -> BFGS] seed={seed}")
        source = source_endpoints[int(seed)]
        before_centered, anchor_drift = reinsert_exact_anchors(source, problem)
        before_metrics = _state_metrics(
            centered=before_centered,
            problem=problem,
            dni=dni,
            data_sim=data_sim,
            directional_data=directional_data,
            test_labels=test_labels,
            targets_centered=targets_centered,
        )
        trajectory_rows.append(
            {"seed": int(seed), "stage": "BFGS polishing", "stage_iteration": 0, "accepted_state": 0, **before_metrics}
        )
        accepted = 0

        def callback(xk):
            nonlocal accepted
            accepted += 1
            centered = problem.unpack(xk)
            trajectory_rows.append(
                {"seed": int(seed), "stage": "BFGS polishing", "stage_iteration": accepted, "accepted_state": accepted, **_state_metrics(
                    centered=centered,
                    problem=problem,
                    dni=dni,
                    data_sim=data_sim,
                    directional_data=directional_data,
                    test_labels=test_labels,
                    targets_centered=targets_centered,
                )}
            )

        result = run_bfgs(problem.pack(before_centered), problem, gtol=gtol, callback=callback)
        if result.get("y_final") is None:
            raise RuntimeError(f"BFGS polishing seed {seed} produced no finite endpoint: {result['failure_reason']}")
        after_centered = problem.unpack(result["y_final"])
        after_metrics = _state_metrics(
            centered=after_centered,
            problem=problem,
            dni=dni,
            data_sim=data_sim,
            directional_data=directional_data,
            test_labels=test_labels,
            targets_centered=targets_centered,
        )
        if not trajectory_rows or not np.isclose(
            float(trajectory_rows[-1]["objective_total"]),
            float(after_metrics["objective_total"]),
            rtol=0.0,
            atol=1e-8,
        ):
            trajectory_rows.append(
                {"seed": int(seed), "stage": "BFGS polishing", "stage_iteration": accepted + 1, "accepted_state": accepted + 1, **after_metrics}
            )
        endpoint_rows.append(
            _endpoint_summary_row(seed, before_metrics, after_metrics, result, anchor_drift)
        )
        source_metric_row = source_runs[source_runs["seed"] == seed].iloc[0]
        endpoint_rows[-1].update(
            {
                "source_AS_RMSE_test_km": float(source_metric_row["RMSE_test_km"]),
                "source_AS_E_distance_stress": float(source_metric_row["E_distance_stress"]),
                "source_AS_E_direction_vr": float(source_metric_row["E_direction_vr"]),
                "source_AS_E_direction_mae": float(source_metric_row["E_direction_mae"]),
                "before_minus_source_RMSE_test_km": float(before_metrics["RMSE_test_km_posthoc"] - source_metric_row["RMSE_test_km"]),
                "before_minus_source_E_distance_stress": float(before_metrics["E_distance_stress"] - source_metric_row["E_distance_stress"]),
                "before_minus_source_E_direction_vr": float(before_metrics["E_direction_vr"] - source_metric_row["E_direction_vr"]),
                "before_minus_source_E_direction_mae": float(before_metrics["E_direction_mae"] - source_metric_row["E_direction_mae"]),
            }
        )
        after_objectives.append(float(after_metrics["objective_total"]))
        for stage, centered in (("PhysicsSim endpoint", before_centered), ("BFGS polished", after_centered)):
            position_rows.extend(
                {
                    "seed": int(seed),
                    "stage": stage,
                    "label": label,
                    "x_y_up_sim": float(centered[index, 0] + refer_pos_sim[0]),
                    "y_y_up_sim": float(centered[index, 1] + refer_pos_sim[1]),
                }
                for index, label in enumerate(vertices)
            )
            force_rows.extend(
                _peripheral_force_rows(
                    seed=seed,
                    stage=stage,
                    centered=centered,
                    problem=problem,
                    dni=dni,
                    test_labels=test_labels,
                    targets_centered=targets_centered,
                    node_graph=node_graph,
                )
            )

        if rerun_physics_trajectory:
            rows, max_diff = _rerun_physics_trajectory(
                seed=seed,
                stride=physics_stride,
                source_endpoint=source,
                problem=problem,
                vertices=vertices,
                dni=dni,
                calibration_labels=calibration_labels,
                calibration_lonlat=calibration_lonlat,
                anchor_label=anchor_label,
                data_sim=data_sim,
                directional_data=directional_data,
                targets_centered=targets_centered,
                test_labels=test_labels,
                alpha=float(as_config["alpha"]),
                beta=float(as_config["beta"]),
            )
            trajectory_rows.extend(rows)
            physics_reproduction[int(seed)] = max_diff

    strata, thresholds = assign_objective_strata(after_objectives, reference_values, 4)
    for row, stratum in zip(endpoint_rows, strata, strict=True):
        row["after_objective_stratum"] = int(stratum)
    endpoints = pd.DataFrame(endpoint_rows)
    trajectory = pd.DataFrame(trajectory_rows)
    positions = pd.DataFrame(position_rows)
    forces = pd.DataFrame(force_rows)
    force_summary = _spearman_outputs(forces, outdir)

    summary_rows = []
    for metric in CORE_METRICS:
        for prefix in ("before", "after", "delta"):
            values = endpoints[f"{prefix}_{metric}"].to_numpy(float)
            summary_rows.append(
                {
                    "stage": prefix,
                    "metric": metric,
                    "n": len(values),
                    "mean": float(np.mean(values)),
                    "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
                    "median": float(np.median(values)),
                }
            )
    pd.DataFrame(summary_rows).to_csv(
        outdir / "polishing_summary.csv", index=False, encoding="utf-8-sig"
    )
    endpoints.groupby("after_objective_stratum").size().rename("count").reset_index().to_csv(
        outdir / "polishing_stratum_counts.csv", index=False, encoding="utf-8-sig"
    )

    endpoints.to_csv(outdir / "polishing_runs.csv", index=False, encoding="utf-8-sig")
    trajectory.to_csv(outdir / "polishing_trajectory.csv", index=False, encoding="utf-8-sig")
    positions.to_csv(outdir / "polishing_final_positions_y_up_sim.csv", index=False, encoding="utf-8-sig")
    forces.to_csv(outdir / "peripheral_force_by_seed_node.csv", index=False, encoding="utf-8-sig")
    if make_plots:
        _make_plots(trajectory, force_summary, endpoints, outdir)

    config = {
        "experiment": "PhysicsSim-Full endpoint to full-memory BFGS polishing",
        "method_version": "source_weight_locked_fixed_anchor_bfgs_polishing_v2",
        "objective_implementation": "library.scipy_objective.FixedAnchorObjective",
        "optimizer": "scipy.optimize.minimize(method='BFGS', jac=True)",
        "uses_limited_memory_bfgs": False,
        "seeds": list(map(int, seeds)),
        "physics_as_source": str(as_outdir),
        "bfgs_stratum_reference": str(reference_bfgs_outdir),
        "bfgs_hpo_source": None,
        "weight_source": "effective weights recorded by the formal PhysicsSim-Full AS runs",
        "methodological_guard": "independent BFGS HPO is rejected for polishing",
        "alpha": float(alpha),
        "beta": float(beta),
        "w_dis": float(w_dis),
        "effective_weights": {
            "distance": problem.weights.distance,
            "direction": problem.weights.direction,
            "repulsion": problem.weights.repulsion,
        },
        "anchor_policy": "three exact anchors reinserted before packing free coordinates",
        "warm_start_policy": "PhysicsSim-Full formal AS endpoint; only free coordinates are optimized",
        "test_site_policy": "posthoc diagnostics only; never used by objective or stopping",
        "calibration_anchor_trajectory_policy": "fixed; LOO RMSE not applicable in this run",
        "trajectory_policy": "BFGS accepted iterates; optional PhysicsSim states sampled at a fixed step stride",
        "gtol": float(gtol),
        "stratum_thresholds": thresholds,
        "rerun_physics_trajectory": bool(rerun_physics_trajectory),
        "physics_stride": int(physics_stride),
        "plots_generated": bool(make_plots),
        "physics_endpoint_reproduction_max_abs_sim": physics_reproduction,
        "optimizer_failure_count": int((~endpoints["optimizer_success"]).sum()),
        "reference_config_alpha_beta_w_dis": [
            reference_config["alpha"],
            reference_config["beta"],
            reference_config["w_dis"],
        ],
    }
    (outdir / "polishing_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"runs": endpoints, "trajectory": trajectory, "config": config}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--as-outdir", default=DEFAULT_AS_OUTDIR)
    parser.add_argument("--reference-bfgs-outdir", default=DEFAULT_REFERENCE_BFGS)
    parser.add_argument(
        "--hpo-outdir",
        default=None,
        help="Deprecated safety trap: independent BFGS HPO is forbidden for polishing.",
    )
    parser.add_argument(
        "--allow-boundary-hpo",
        action="store_true",
        help="Deprecated safety trap: independent BFGS HPO is forbidden for polishing.",
    )
    parser.add_argument("--gtol", type=float, default=DEFAULT_BFGS_GTOL)
    parser.add_argument("--rerun-physics-trajectory", action="store_true")
    parser.add_argument("--physics-stride", type=int, default=25)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    args = parser.parse_args()
    result = run_polishing(
        seeds=_parse_seeds(args.seeds),
        as_outdir=args.as_outdir,
        reference_bfgs_outdir=args.reference_bfgs_outdir,
        outdir=args.outdir,
        hpo_outdir=args.hpo_outdir,
        allow_boundary_hpo=args.allow_boundary_hpo,
        gtol=args.gtol,
        rerun_physics_trajectory=args.rerun_physics_trajectory,
        physics_stride=args.physics_stride,
        make_plots=not args.no_plots,
    )
    print(result["runs"].to_string(index=False))
    print(f"[Saved] {Path(args.outdir)}")


if __name__ == "__main__":
    main()
