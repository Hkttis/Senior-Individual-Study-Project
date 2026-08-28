"""PhysicsSim route-distance sensitivity with scenario-specific or fixed hyperparameters."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from library.config import (
    DIRECTIONAL_FORCE_MAGNITUDE_BASE,
    FILE_PATHS,
    OUTPUT_DIR,
    REPULSION_STRENGTH_BASE,
    SPRING_STIFFNESS_BASE,
    km2sim,
    refer_pos_sim as DEFAULT_REFER_POS_SIM,
)
from library.data_io import (
    get_anchor_labels,
    get_default_frame_anchor_label,
    get_test_site_labels,
    load_ini_data_from_csv,
    read_CHEN_csvfile,
)
from library.geometry import get_lcc_bounds, get_lcc_parameters
from library.units import data_Li2sim
from run_paper_script.ch5_anchor_split_robustness import (
    _append_event,
    _archive_incomplete_split,
    _bootstrap_ci_mean,
    _numeric_grid,
    _parse_seed_list,
    _sha256,
    _utc_now,
    _write_json,
)
from run_paper_script.ch5_hparam_kfold_gridsearch_pareto import (
    _resolve_anchor_and_test_inputs,
    _run_final_selected_model,
    _scale_sim_distance_data,
    run_anchor_loo_gridsearch_pareto,
)


METRIC_COLUMNS = (
    "RMSE_final_test_km",
    "E_distance_stress",
    "E_direction_vr",
    "E_direction_mae",
)
RESUME_CONFIG_KEYS = (
    "experiment",
    "scenario_scales",
    "anchor_labels",
    "test_labels",
    "final_frame_anchor_label",
    "hpo_validation",
    "selection_rule",
    "boundary_policy",
    "hpo_seeds",
    "final_evaluation_seeds",
    "alpha_range",
    "beta_range",
    "w_dis",
    "base_spring_stiffness",
    "base_directional_force",
    "base_repulsion_strength",
    "reference_hyperparameters",
    "lcc_bounds",
    "lcc_parameters",
    "input_sha256",
)


def _scenario_name(kappa: float) -> str:
    return f"scenario_kappa_{float(kappa):.3f}".replace(".", "p")


def _scenario_grid(minimum: float, maximum: float, step: float) -> list[float]:
    values = _numeric_grid(minimum, maximum, step, name="kappa")
    if any(value <= 0.0 or value > 1.0 + 1e-10 for value in values):
        raise ValueError("Every kappa value must satisfy 0 < kappa <= 1.")
    if not any(np.isclose(value, 1.0, rtol=0.0, atol=1e-10) for value in values):
        raise ValueError("The scenario grid must include the kappa=1.0 reference.")
    values = [round(float(value), 10) for value in reversed(values)]
    names = [_scenario_name(value) for value in values]
    if len(names) != len(set(names)):
        raise ValueError("Kappa scenarios produce duplicate three-decimal output folder names.")
    return values


def _input_paths() -> dict[str, Path]:
    return {
        "site_points": Path(FILE_PATHS["ground_truth_path"]),
        "distance_edges": Path(FILE_PATHS["chen_data"]),
        "direction_edges": Path(FILE_PATHS["directional_data"]),
        "ini_data": Path(FILE_PATHS["ini_data"]),
    }


def _input_hashes() -> dict[str, str]:
    return {name: _sha256(path) for name, path in _input_paths().items()}


def _assert_distance_sources_consistent(raw_rows: Sequence[Sequence[object]], ini_rows: Sequence[Sequence[object]]) -> None:
    if len(raw_rows) != len(ini_rows):
        raise ValueError(
            "distance_edges_verified.csv and ini_data.csv contain different numbers of distance edges; "
            "rebuild ini_data before running the experiment."
        )
    for index, (raw, cached) in enumerate(zip(raw_rows, ini_rows), start=1):
        if len(raw) < 3 or len(cached) < 3:
            raise ValueError(f"Distance edge {index} has fewer than three required fields.")
        if str(raw[0]) != str(cached[0]) or str(raw[1]) != str(cached[1]):
            raise ValueError(f"Distance edge {index} endpoints differ between source CSV and ini_data.csv.")
        if not np.isclose(float(raw[2]), float(cached[2]), rtol=0.0, atol=1e-12):
            raise ValueError(f"Distance edge {index} values differ between source CSV and ini_data.csv.")


def _distance_target_audit_frame(data_li: Sequence[Sequence[object]], kappa: float) -> pd.DataFrame:
    base_sim = data_Li2sim(data_li)
    scaled_sim = _scale_sim_distance_data(base_sim, kappa)
    rows = []
    for index, (original, unscaled, scaled) in enumerate(zip(data_li, base_sim, scaled_sim)):
        original_li = float(original[2])
        unscaled_target_sim = float(unscaled[2])
        scaled_target_sim = float(scaled[2])
        rows.append(
            {
                "edge_index": index,
                "source": str(original[0]),
                "target": str(original[1]),
                "original_distance_li": original_li,
                "unscaled_target_sim": unscaled_target_sim,
                "scaled_target_sim": scaled_target_sim,
                "unscaled_target_km": unscaled_target_sim / km2sim,
                "scaled_target_km": scaled_target_sim / km2sim,
                "distance_scale": float(kappa),
                "applied_ratio": scaled_target_sim / unscaled_target_sim,
            }
        )
    return pd.DataFrame(rows)


def _completed_scenario(scenario_dir: Path, *, kappa: float, final_seeds: Sequence[int]) -> bool:
    base_required = (
        "gridsearch_config.json",
        "selected_final_summary.json",
        "selected_final_runs_by_seed.csv",
        "selected_final_site_errors.csv",
        "selected_final_positions_y_up_sim.csv",
        "distance_targets_audit.csv",
    )
    if not all((scenario_dir / filename).is_file() for filename in base_required):
        return False
    try:
        config = json.loads((scenario_dir / "gridsearch_config.json").read_text(encoding="utf-8"))
        if config.get("hyperparameter_policy", "scenario_specific_hpo") == "scenario_specific_hpo":
            hpo_required = ("grid_runs_by_seed.csv", "grid_summary_cv.csv", "pareto_front_3d.csv")
            if not all((scenario_dir / filename).is_file() for filename in hpo_required):
                return False
        summary = json.loads((scenario_dir / "selected_final_summary.json").read_text(encoding="utf-8"))
        runs = pd.read_csv(scenario_dir / "selected_final_runs_by_seed.csv")
        errors = pd.read_csv(scenario_dir / "selected_final_site_errors.csv")
        positions = pd.read_csv(scenario_dir / "selected_final_positions_y_up_sim.csv")
        audit = pd.read_csv(scenario_dir / "distance_targets_audit.csv")
    except (json.JSONDecodeError, OSError, ValueError, pd.errors.ParserError):
        return False
    if not np.isclose(float(config.get("distance_scale", np.nan)), kappa, rtol=0.0, atol=1e-10):
        return False
    if not np.isclose(float(summary.get("distance_scale", np.nan)), kappa, rtol=0.0, atol=1e-10):
        return False
    if not {"seed", "distance_scale", *METRIC_COLUMNS}.issubset(runs.columns):
        return False
    if len(runs) != len(final_seeds) or set(runs["seed"].astype(int)) != set(map(int, final_seeds)):
        return False
    if not np.allclose(runs["distance_scale"].to_numpy(float), kappa, rtol=0.0, atol=1e-10):
        return False
    if not np.isfinite(runs[list(METRIC_COLUMNS)].to_numpy(float)).all():
        return False
    if not {"original_distance_li", "unscaled_target_sim", "scaled_target_sim", "scaled_target_km", "applied_ratio"}.issubset(audit.columns):
        return False
    if len(audit) != 44 or not np.allclose(audit["applied_ratio"].to_numpy(float), kappa, rtol=0.0, atol=1e-10):
        return False
    return (
        {"site_label", "error_km", "squared_error_km2"}.issubset(errors.columns)
        and len(errors) == len(final_seeds) * 8
        and {"seed", "node_idx", "label", "x_y_up_sim", "y_y_up_sim"}.issubset(positions.columns)
        and len(positions) == len(final_seeds) * 35
    )


def _assert_resume_config_compatible(existing: dict, requested: dict) -> None:
    mismatches = [key for key in RESUME_CONFIG_KEYS if existing.get(key) != requested.get(key)]
    if mismatches:
        details = {key: {"existing": existing.get(key), "requested": requested.get(key)} for key in mismatches}
        raise ValueError(
            "Resume configuration differs from the existing detour experiment; use a new --outdir. "
            f"Mismatched fields: {json.dumps(details, ensure_ascii=False)}"
        )


def preflight_detour_sensitivity(
    *,
    seeds: Sequence[int],
    final_seeds: Sequence[int],
    kappa_min: float,
    kappa_max: float,
    kappa_step: float,
    alpha_min: float,
    alpha_max: float,
    alpha_step: float,
    beta_min: float,
    beta_max: float,
    beta_step: float,
    outdir: str | Path,
    fixed_alpha: float | None = None,
    fixed_beta: float | None = None,
    reference_alpha: float | None = None,
    reference_beta: float | None = None,
    w_dis: float = 1.0,
    base_spring_stiffness: float = SPRING_STIFFNESS_BASE,
    base_directional_force: float = DIRECTIONAL_FORCE_MAGNITUDE_BASE,
    base_repulsion_strength: float = REPULSION_STRENGTH_BASE,
    resume: bool = False,
) -> dict:
    scenarios = _scenario_grid(kappa_min, kappa_max, kappa_step)
    if (fixed_alpha is None) != (fixed_beta is None):
        raise ValueError("--fixed-alpha and --fixed-beta must be provided together.")
    if (reference_alpha is None) != (reference_beta is None):
        raise ValueError("--reference-alpha and --reference-beta must be provided together.")
    fixed_mode = fixed_alpha is not None
    reference_mode = reference_alpha is not None
    if fixed_mode and reference_mode:
        raise ValueError("Reference-only hyperparameters cannot be combined with fixed-hyperparameter mode.")
    if reference_mode and (
        not np.isfinite(float(reference_alpha)) or not np.isfinite(float(reference_beta))
    ):
        raise ValueError("Reference alpha and beta must be finite.")
    if fixed_mode:
        if not np.isfinite(float(fixed_alpha)) or not np.isfinite(float(fixed_beta)):
            raise ValueError("Fixed alpha and beta must be finite.")
        alpha_values = [float(fixed_alpha)]
        beta_values = [float(fixed_beta)]
        hpo_seeds: list[int] = []
    else:
        alpha_values = _numeric_grid(alpha_min, alpha_max, alpha_step, name="alpha")
        beta_values = _numeric_grid(beta_min, beta_max, beta_step, name="beta")
        hpo_seeds = [int(seed) for seed in seeds]
    final_seed_values = [int(seed) for seed in final_seeds]
    seed_groups = [("final-evaluation seeds", final_seed_values)]
    if not fixed_mode:
        seed_groups.insert(0, ("HPO seeds", hpo_seeds))
    for label, values in seed_groups:
        if not values or len(values) != len(set(values)) or any(seed < 0 for seed in values):
            raise ValueError(f"{label} must contain distinct, non-negative integers.")
    weights = {
        "w_dis": float(w_dis),
        "base_spring_stiffness": float(base_spring_stiffness),
        "base_directional_force": float(base_directional_force),
        "base_repulsion_strength": float(base_repulsion_strength),
    }
    if not all(np.isfinite(value) and value > 0.0 for value in weights.values()):
        raise ValueError(f"All base weights must be finite and strictly positive: {weights}")

    paths = _input_paths()
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required experiment inputs are missing: {missing}")
    _graph, vertices, dni, _edges, data_li = load_ini_data_from_csv(FILE_PATHS)
    if len(vertices) != 35 or len(dni) != 35 or not data_li:
        raise ValueError("Expected the current 35-node reconstruction graph with non-empty distance data.")
    raw_distance_rows = read_CHEN_csvfile()
    _assert_distance_sources_consistent(raw_distance_rows, data_li)
    _scale_sim_distance_data(data_Li2sim(data_li), min(scenarios))
    anchors = get_anchor_labels()
    test_labels = get_test_site_labels()
    frame_anchor = get_default_frame_anchor_label()
    if len(set(anchors)) != 3 or len(set(test_labels)) != 8 or set(anchors) & set(test_labels):
        raise ValueError("Expected three distinct calibration anchors and eight disjoint held-out sites.")
    if frame_anchor not in anchors:
        raise ValueError("The final frame anchor must be one of the calibration anchors.")

    output = Path(outdir)
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f"Output folder is non-empty: {output}. Use a new path or --resume.")
    completed = 0
    incomplete = 0
    if output.exists():
        for kappa in scenarios:
            scenario_dir = output / "scenarios" / _scenario_name(kappa)
            if _completed_scenario(scenario_dir, kappa=kappa, final_seeds=final_seed_values):
                completed += 1
            elif scenario_dir.exists() and any(scenario_dir.iterdir()):
                incomplete += 1
    disk_parent = output.parent
    while not disk_parent.exists() and disk_parent != disk_parent.parent:
        disk_parent = disk_parent.parent
    free_disk_bytes = int(shutil.disk_usage(disk_parent).free)
    if free_disk_bytes < 1_000_000_000:
        raise OSError(f"Less than 1 GB free on the output drive: {free_disk_bytes} bytes.")

    hpo_runs = 0 if fixed_mode else len(alpha_values) * len(beta_values) * 3 * len(hpo_seeds)
    n_hpo_scenarios = 0 if fixed_mode else len(scenarios) - int(reference_mode)
    return {
        "checked_at_utc": _utc_now(),
        "scenario_scales": scenarios,
        "n_scenarios": len(scenarios),
        "n_distance_edges": len(data_li),
        "distance_sources_consistent": True,
        "hpo_seeds": hpo_seeds,
        "hyperparameter_policy": (
            "fixed"
            if fixed_mode
            else "scenario_specific_hpo_with_fixed_reference"
            if reference_mode
            else "scenario_specific_hpo"
        ),
        "fixed_alpha": float(fixed_alpha) if fixed_mode else None,
        "fixed_beta": float(fixed_beta) if fixed_mode else None,
        "reference_alpha": float(reference_alpha) if reference_mode else None,
        "reference_beta": float(reference_beta) if reference_mode else None,
        "final_evaluation_seeds": final_seed_values,
        "alpha_values": alpha_values,
        "beta_values": beta_values,
        "hpo_runs_per_scenario": hpo_runs,
        "final_runs_per_scenario": len(final_seed_values),
        "expected_total_model_runs": n_hpo_scenarios * hpo_runs + len(scenarios) * len(final_seed_values),
        "anchor_labels": anchors,
        "test_labels": test_labels,
        "final_frame_anchor_label": frame_anchor,
        "existing_completed_scenarios": completed,
        "existing_incomplete_scenarios": incomplete,
        "free_disk_bytes": free_disk_bytes,
        "input_sha256": _input_hashes(),
        "base_weights": weights,
        "resume": bool(resume),
    }


def _summarize_scenario(scenario_dir: Path, *, kappa: float) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    runs = pd.read_csv(scenario_dir / "selected_final_runs_by_seed.csv")
    site_errors = pd.read_csv(scenario_dir / "selected_final_site_errors.csv")
    final_summary = json.loads((scenario_dir / "selected_final_summary.json").read_text(encoding="utf-8"))
    scenario_config = json.loads((scenario_dir / "gridsearch_config.json").read_text(encoding="utf-8"))
    policy = scenario_config.get("hyperparameter_policy", "scenario_specific_hpo")
    if policy in {"fixed", "fixed_reference"}:
        selected = None
        n_hpo_runs = n_hpo_successful = n_hpo_failed = 0
    else:
        grid_runs = pd.read_csv(scenario_dir / "grid_runs_by_seed.csv")
        grid_summary = pd.read_csv(scenario_dir / "grid_summary_cv.csv")
        match = grid_summary[
            np.isclose(grid_summary["alpha"], float(final_summary["alpha"]))
            & np.isclose(grid_summary["beta"], float(final_summary["beta"]))
        ]
        if len(match) != 1:
            raise ValueError(f"Selected alpha/beta do not identify one grid candidate: {scenario_dir}")
        selected = match.iloc[0]
        n_hpo_runs = int(len(grid_runs))
        n_hpo_successful = int(grid_runs["RMSE_anchor_LOO_km"].notna().sum())
        n_hpo_failed = int(grid_runs["RMSE_anchor_LOO_km"].isna().sum())
    summary = {
        "kappa": float(kappa),
        "detour_ratio": float(1.0 / kappa),
        "scenario_id": _scenario_name(kappa),
        "hyperparameter_policy": policy,
        "selected_alpha": float(final_summary["alpha"]),
        "selected_beta": float(final_summary["beta"]),
        "selected_on_alpha_boundary": bool(final_summary.get("selected_on_alpha_boundary", False)),
        "selected_on_beta_boundary": bool(final_summary.get("selected_on_beta_boundary", False)),
        "selected_on_grid_boundary": bool(final_summary.get("selected_on_grid_boundary", False)),
        "anchor_loo_rmse_mean_km": float(selected["RMSE_anchor_LOO_mean_km"]) if selected is not None else float("nan"),
        "anchor_loo_rmse_std_km": float(selected["RMSE_anchor_LOO_std_km"]) if selected is not None else float("nan"),
        "n_hpo_runs": n_hpo_runs,
        "n_hpo_successful": n_hpo_successful,
        "n_hpo_failed": n_hpo_failed,
        "n_final_runs": int(len(runs)),
    }
    for index, metric in enumerate(METRIC_COLUMNS):
        values = runs[metric].to_numpy(float)
        ci_low, ci_high = _bootstrap_ci_mean(values, seed=17_000 + round(kappa * 1000) + index)
        summary[f"{metric}_mean"] = float(np.mean(values))
        summary[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
        summary[f"{metric}_median"] = float(np.median(values))
        summary[f"{metric}_ci95_low"] = ci_low
        summary[f"{metric}_ci95_high"] = ci_high
    runs = runs.copy()
    runs.insert(0, "kappa", float(kappa))
    runs.insert(1, "detour_ratio", float(1.0 / kappa))
    site_errors = site_errors.copy()
    site_errors.insert(0, "kappa", float(kappa))
    return summary, runs, site_errors


def _paired_comparisons(all_runs: pd.DataFrame) -> pd.DataFrame:
    reference = all_runs[np.isclose(all_runs["kappa"].to_numpy(float), 1.0)].copy()
    if reference.empty:
        return pd.DataFrame()
    if reference["seed"].duplicated().any():
        raise ValueError("The reference scenario contains duplicate seeds.")
    rows: list[dict] = []
    for kappa, scenario in all_runs.groupby("kappa", sort=False):
        if np.isclose(float(kappa), 1.0):
            continue
        if scenario["seed"].duplicated().any() or set(scenario["seed"]) != set(reference["seed"]):
            raise ValueError(f"Scenario kappa={kappa} cannot be paired with the reference seeds.")
        merged = scenario.merge(reference, on="seed", suffixes=("_scenario", "_reference"), validate="one_to_one")
        for index, metric in enumerate(METRIC_COLUMNS):
            delta = merged[f"{metric}_scenario"].to_numpy(float) - merged[f"{metric}_reference"].to_numpy(float)
            low, high = _bootstrap_ci_mean(delta, seed=23_000 + round(float(kappa) * 1000) + index)
            rows.append(
                {
                    "kappa": float(kappa),
                    "detour_ratio": float(1.0 / float(kappa)),
                    "reference_kappa": 1.0,
                    "metric": metric,
                    "n_pairs": int(len(delta)),
                    "difference_mean": float(np.mean(delta)),
                    "difference_std": float(np.std(delta, ddof=1)) if len(delta) > 1 else float("nan"),
                    "difference_ci95_low": low,
                    "difference_ci95_high": high,
                    "win_rate_lower": float(np.mean(delta < 0.0)),
                }
            )
    return pd.DataFrame(rows)


def _save_plots(summary: pd.DataFrame, outdir: Path) -> None:
    if summary.empty:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = summary.sort_values("kappa")
    x = ordered["kappa"].to_numpy(float)
    fig, axis = plt.subplots(figsize=(9.2, 5.2))
    y = ordered["RMSE_final_test_km_mean"].to_numpy(float)
    axis.plot(x, y, marker="o", color="#165b9f", linewidth=2.0)
    axis.fill_between(
        x,
        ordered["RMSE_final_test_km_ci95_low"].to_numpy(float),
        ordered["RMSE_final_test_km_ci95_high"].to_numpy(float),
        color="#165b9f",
        alpha=0.18,
        label="95% bootstrap CI",
    )
    reference = ordered[np.isclose(ordered["kappa"], 1.0)]
    if not reference.empty:
        axis.axhline(float(reference.iloc[0]["RMSE_final_test_km_mean"]), linestyle="--", color="#777777")
    axis.set_xlabel("Distance scaling factor kappa")
    axis.set_ylabel("Held-out test RMSE (km)")
    axis.grid(alpha=0.22)
    axis.legend(loc="best")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(outdir / f"detour_rmse_sensitivity.{suffix}", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.6), sharex=True)
    for axis, column, label, color in (
        (axes[0], "selected_alpha", "Selected alpha", "#165b9f"),
        (axes[1], "selected_beta", "Selected beta", "#b45309"),
    ):
        axis.step(x, ordered[column].to_numpy(float), where="mid", color=color, linewidth=1.8)
        axis.scatter(x, ordered[column].to_numpy(float), color=color, zorder=3)
        axis.set_ylabel(label)
        axis.grid(alpha=0.22)
    axes[-1].set_xlabel("Distance scaling factor kappa")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(outdir / f"detour_selected_hyperparameters.{suffix}", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), sharex=True)
    panels = [
        ("E_distance_stress_mean", "Stress"),
        ("E_direction_vr_mean", "Violation Rate"),
        ("E_direction_mae_mean", "Mean Angular Error (rad)"),
    ]
    if summary["anchor_loo_rmse_mean_km"].notna().any():
        panels.append(("anchor_loo_rmse_mean_km", "Anchor LOO RMSE (km)"))
    else:
        panels.append(("RMSE_final_test_km_mean", "Held-out test RMSE (km)"))
    for axis, (column, label) in zip(axes.flat, panels):
        axis.plot(x, ordered[column].to_numpy(float), marker="o", linewidth=1.7)
        axis.set_ylabel(label)
        axis.grid(alpha=0.22)
    for axis in axes[-1]:
        axis.set_xlabel("Distance scaling factor kappa")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(outdir / f"detour_secondary_metrics.{suffix}", dpi=220)
    plt.close(fig)


def _write_aggregate(
    *,
    outdir: Path,
    summaries: Sequence[dict],
    all_runs: Sequence[pd.DataFrame],
    all_errors: Sequence[pd.DataFrame],
    expected_scenarios: int,
    hyperparameter_policy: str = "scenario_specific_hpo",
) -> None:
    if not summaries:
        return
    summary = pd.DataFrame(summaries).sort_values("kappa", ascending=False).reset_index(drop=True)
    runs = pd.concat(all_runs, ignore_index=True)
    errors = pd.concat(all_errors, ignore_index=True)
    paired = _paired_comparisons(runs)
    summary.to_csv(outdir / "detour_scenario_summary.csv", index=False, encoding="utf-8-sig")
    runs.to_csv(outdir / "detour_final_runs.csv", index=False, encoding="utf-8-sig")
    errors.to_csv(outdir / "detour_site_errors.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(outdir / "detour_paired_comparisons.csv", index=False, encoding="utf-8-sig")
    _write_json(
        outdir / "detour_global_summary.json",
        {
            "estimand": (
                "held-out RMSE sensitivity to route-distance scaling with fixed hyperparameters"
                if hyperparameter_policy == "fixed"
                else "held-out RMSE sensitivity to route-distance scaling with scenario-specific HPO"
            ),
            "hyperparameter_policy": hyperparameter_policy,
            "n_completed_scenarios": int(len(summary)),
            "n_expected_scenarios": int(expected_scenarios),
            "n_selected_on_grid_boundary": int(summary["selected_on_grid_boundary"].sum()),
            "scenario_scales": summary["kappa"].astype(float).tolist(),
            "reference_kappa": 1.0,
            "selection_warning": "Held-out RMSE is diagnostic only and must not select a detour factor.",
            "stress_warning": "Stress is evaluated against each scenario's scaled distance targets.",
        },
    )
    _save_plots(summary, outdir)


def run_detour_factor_sensitivity(
    *,
    seeds: Sequence[int],
    final_seeds: Sequence[int],
    kappa_min: float,
    kappa_max: float,
    kappa_step: float,
    alpha_min: float,
    alpha_max: float,
    alpha_step: float,
    beta_min: float,
    beta_max: float,
    beta_step: float,
    w_dis: float,
    base_spring_stiffness: float,
    base_directional_force: float,
    base_repulsion_strength: float,
    outdir: str | Path,
    fixed_alpha: float | None = None,
    fixed_beta: float | None = None,
    reference_alpha: float | None = None,
    reference_beta: float | None = None,
    resume: bool = False,
    generate_scenario_plots: bool = True,
) -> dict:
    preflight = preflight_detour_sensitivity(
        seeds=seeds,
        final_seeds=final_seeds,
        kappa_min=kappa_min,
        kappa_max=kappa_max,
        kappa_step=kappa_step,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        alpha_step=alpha_step,
        beta_min=beta_min,
        beta_max=beta_max,
        beta_step=beta_step,
        outdir=outdir,
        fixed_alpha=fixed_alpha,
        fixed_beta=fixed_beta,
        reference_alpha=reference_alpha,
        reference_beta=reference_beta,
        w_dis=w_dis,
        base_spring_stiffness=base_spring_stiffness,
        base_directional_force=base_directional_force,
        base_repulsion_strength=base_repulsion_strength,
        resume=resume,
    )
    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=True)
    scenarios_root = output / "scenarios"
    scenarios_root.mkdir(exist_ok=True)
    status_root = output / "scenario_status"
    archive_root = output / "interrupted_attempts"
    event_log = output / "experiment_events.jsonl"
    scenarios = preflight["scenario_scales"]
    fixed_mode = preflight["hyperparameter_policy"] == "fixed"
    reference_mode = preflight["hyperparameter_policy"] == "scenario_specific_hpo_with_fixed_reference"
    config = {
        "experiment": (
            "detour_factor_sensitivity_with_fixed_hyperparameters"
            if fixed_mode
            else "detour_factor_sensitivity_with_scenario_specific_hpo"
        ),
        "hyperparameter_policy": preflight["hyperparameter_policy"],
        "fixed_alpha": preflight["fixed_alpha"],
        "fixed_beta": preflight["fixed_beta"],
        "reference_hyperparameters": (
            {"alpha": preflight["reference_alpha"], "beta": preflight["reference_beta"]}
            if reference_mode
            else None
        ),
        "scenario_scales": scenarios,
        "scenario_order": "descending kappa; unscaled reference runs first",
        "detour_ratio_definition": "route_distance / straight_distance = 1 / kappa",
        "distance_scaling_policy": "all distance targets scaled in memory; source files remain unchanged",
        "anchor_labels": preflight["anchor_labels"],
        "test_labels": preflight["test_labels"],
        "final_frame_anchor_label": preflight["final_frame_anchor_label"],
        "hpo_validation": "not applicable; predefined fixed hyperparameters" if fixed_mode else "three-anchor leave-one-anchor-out; no held-out test sites",
        "selection_rule": "predefined_fixed_hyperparameters" if fixed_mode else "pareto_one_se_balanced",
        "boundary_policy": "not applicable" if fixed_mode else "fixed common grid; no scenario-specific expansion; report boundary frequency",
        "heldout_policy": "test RMSE is diagnostic and never selects kappa or HPO candidates",
        "stress_policy": "evaluate Stress against the scenario-specific scaled distance targets",
        "hpo_seeds": preflight["hpo_seeds"],
        "final_evaluation_seeds": list(map(int, final_seeds)),
        "alpha_range": [fixed_alpha, fixed_alpha, 1.0] if fixed_mode else [alpha_min, alpha_max, alpha_step],
        "beta_range": [fixed_beta, fixed_beta, 1.0] if fixed_mode else [beta_min, beta_max, beta_step],
        "w_dis": float(w_dis),
        "base_spring_stiffness": float(base_spring_stiffness),
        "base_directional_force": float(base_directional_force),
        "base_repulsion_strength": float(base_repulsion_strength),
        "lcc_bounds": dict(zip(["lon_min", "lon_max", "lat_min", "lat_max"], map(float, get_lcc_bounds()))),
        "lcc_parameters": dict(zip(["lat_1", "lat_2", "lon_0"], map(float, get_lcc_parameters()))),
        "input_sha256": preflight["input_sha256"],
    }
    config_path = output / "detour_experiment_config.json"
    if config_path.exists():
        if not resume:
            raise FileExistsError(f"Configuration already exists: {config_path}")
        _assert_resume_config_compatible(json.loads(config_path.read_text(encoding="utf-8")), config)
    else:
        _write_json(config_path, config)
    preflight_name = "preflight_report.json" if not resume else f"preflight_report_resume_{datetime.now():%Y%m%d_%H%M%S}.json"
    _write_json(output / preflight_name, preflight)
    _append_event(
        event_log,
        {
            "event": "preflight_passed",
            "resume": bool(resume),
            "n_scenarios": len(scenarios),
            "expected_total_model_runs": preflight["expected_total_model_runs"],
        },
    )

    summaries: list[dict] = []
    all_runs: list[pd.DataFrame] = []
    all_errors: list[pd.DataFrame] = []
    durations: list[float] = []
    for index, kappa in enumerate(scenarios, start=1):
        name = _scenario_name(kappa)
        scenario_dir = scenarios_root / name
        status_path = status_root / f"{name}.json"
        print(f"[{index}/{len(scenarios)}] kappa={kappa:.3f}; detour ratio={1.0 / kappa:.4f}", flush=True)
        if _completed_scenario(scenario_dir, kappa=kappa, final_seeds=final_seeds):
            if not resume:
                raise FileExistsError(f"Completed scenario output already exists: {scenario_dir}")
            print("  [Resume] using completed scenario output", flush=True)
            _append_event(event_log, {"event": "resume_completed_scenario", "scenario_id": name, "kappa": kappa})
        else:
            if scenario_dir.exists() and any(scenario_dir.iterdir()):
                if not resume:
                    raise RuntimeError(f"Incomplete scenario folder exists: {scenario_dir}. Use --resume.")
                archived = _archive_incomplete_split(scenario_dir, archive_root)
                print(f"  [Resume] archived incomplete attempt to {archived}", flush=True)
                _append_event(
                    event_log,
                    {"event": "archive_incomplete_scenario", "scenario_id": name, "archived_to": str(archived)},
                )
            started = time.perf_counter()
            _append_event(event_log, {"event": "scenario_started", "scenario_id": name, "kappa": kappa})
            _write_json(
                status_path,
                {"scenario_id": name, "kappa": kappa, "status": "running", "process_id": os.getpid()},
            )
            try:
                reference_scenario = reference_mode and np.isclose(kappa, 1.0, rtol=0.0, atol=1e-10)
                if fixed_mode or reference_scenario:
                    selected_alpha = float(fixed_alpha if fixed_mode else reference_alpha)
                    selected_beta = float(fixed_beta if fixed_mode else reference_beta)
                    scenario_dir.mkdir(parents=True, exist_ok=True)
                    _graph, _vertices, dni, _edges, _data = load_ini_data_from_csv(FILE_PATHS)
                    anchors, anchor_lonlat, test_labels, test_lonlat = _resolve_anchor_and_test_inputs(dni)
                    _write_json(
                        scenario_dir / "gridsearch_config.json",
                        {
                            "experiment": (
                                "detour_fixed_hyperparameters_final_evaluation"
                                if fixed_mode
                                else "detour_fixed_reference_final_evaluation"
                            ),
                            "hyperparameter_policy": "fixed" if fixed_mode else "fixed_reference",
                            "selection_rule": (
                                "predefined_fixed_hyperparameters"
                                if fixed_mode
                                else "predefined_formal_reference_hyperparameters"
                            ),
                            "distance_scale": float(kappa),
                            "alpha": selected_alpha,
                            "beta": selected_beta,
                            "anchor_labels": list(anchors),
                            "test_labels": list(test_labels),
                            "final_frame_anchor_label": preflight["final_frame_anchor_label"],
                            "refer_pos_sim": list(map(float, DEFAULT_REFER_POS_SIM)),
                            "final_evaluation_seeds": list(map(int, final_seeds)),
                            "w_dis": float(w_dis),
                            "base_spring_stiffness": float(base_spring_stiffness),
                            "base_directional_force": float(base_directional_force),
                            "base_repulsion_strength": float(base_repulsion_strength),
                        },
                    )
                    _run_final_selected_model(
                        selected=pd.Series({"alpha": selected_alpha, "beta": selected_beta}),
                        anchor_labels=anchors,
                        anchor_lonlat=anchor_lonlat,
                        test_labels=test_labels,
                        test_lonlat=test_lonlat,
                        seeds=final_seeds,
                        w_dis=w_dis,
                        base_spring_stiffness=base_spring_stiffness,
                        base_directional_force=base_directional_force,
                        base_repulsion_strength=base_repulsion_strength,
                        refer_pos_sim=DEFAULT_REFER_POS_SIM,
                        outdir=scenario_dir,
                        selection_rule=(
                            "predefined_fixed_hyperparameters"
                            if fixed_mode
                            else "predefined_formal_reference_hyperparameters"
                        ),
                        final_frame_anchor_label=preflight["final_frame_anchor_label"],
                        save_final_positions=True,
                        distance_scale=kappa,
                    )
                else:
                    run_anchor_loo_gridsearch_pareto(
                        seeds=seeds,
                        final_seeds=final_seeds,
                        alpha_min=alpha_min,
                        alpha_max=alpha_max,
                        alpha_step=alpha_step,
                        beta_min=beta_min,
                        beta_max=beta_max,
                        beta_step=beta_step,
                        w_dis=w_dis,
                        base_spring_stiffness=base_spring_stiffness,
                        base_directional_force=base_directional_force,
                        base_repulsion_strength=base_repulsion_strength,
                        refer_pos_sim=DEFAULT_REFER_POS_SIM,
                        outdir=scenario_dir,
                        final_frame_anchor_label=preflight["final_frame_anchor_label"],
                        generate_plots=generate_scenario_plots,
                        save_final_positions=True,
                        distance_scale=kappa,
                    )
                _graph, _vertices, _dni, _edges, data_li = load_ini_data_from_csv(FILE_PATHS)
                _distance_target_audit_frame(data_li, kappa).to_csv(
                    scenario_dir / "distance_targets_audit.csv", index=False, encoding="utf-8-sig"
                )
                if not _completed_scenario(scenario_dir, kappa=kappa, final_seeds=final_seeds):
                    raise RuntimeError(f"Scenario completed without all verified output artifacts: {scenario_dir}")
                current_hashes = _input_hashes()
                if current_hashes != config["input_sha256"]:
                    raise RuntimeError("Source data changed while the detour experiment was running.")
            except BaseException as exc:
                elapsed = time.perf_counter() - started
                _append_event(
                    event_log,
                    {
                        "event": "scenario_failed",
                        "scenario_id": name,
                        "kappa": kappa,
                        "elapsed_seconds": elapsed,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                _write_json(
                    status_path,
                    {
                        "scenario_id": name,
                        "kappa": kappa,
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                raise
            elapsed = time.perf_counter() - started
            durations.append(elapsed)
            remaining = float(np.mean(durations) * (len(scenarios) - index))
            _append_event(
                event_log,
                {
                    "event": "scenario_completed",
                    "scenario_id": name,
                    "kappa": kappa,
                    "elapsed_seconds": elapsed,
                    "estimated_remaining_seconds": remaining,
                },
            )
            _write_json(
                status_path,
                {"scenario_id": name, "kappa": kappa, "status": "completed", "elapsed_seconds": elapsed},
            )
            print(
                f"  [Completed] {name} in {elapsed / 60.0:.1f} min; "
                f"estimated remaining {remaining / 3600.0:.2f} h",
                flush=True,
            )
        summary, scenario_runs, scenario_errors = _summarize_scenario(scenario_dir, kappa=kappa)
        summaries.append(summary)
        all_runs.append(scenario_runs)
        all_errors.append(scenario_errors)
        _write_aggregate(
            outdir=output,
            summaries=summaries,
            all_runs=all_runs,
            all_errors=all_errors,
            expected_scenarios=len(scenarios),
            hyperparameter_policy=preflight["hyperparameter_policy"],
        )
    return {"outdir": output, "n_completed_scenarios": len(summaries), "n_expected_scenarios": len(scenarios)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default=",".join(map(str, range(10))))
    parser.add_argument("--final-seeds", default=",".join(map(str, range(100))))
    parser.add_argument("--kappa-min", type=float, default=0.70)
    parser.add_argument("--kappa-max", type=float, default=1.00)
    parser.add_argument("--kappa-step", type=float, default=0.025)
    parser.add_argument("--alpha-min", type=float, default=-1.0)
    parser.add_argument("--alpha-max", type=float, default=1.5)
    parser.add_argument("--alpha-step", type=float, default=0.5)
    parser.add_argument("--beta-min", type=float, default=-2.0)
    parser.add_argument("--beta-max", type=float, default=0.5)
    parser.add_argument("--beta-step", type=float, default=0.5)
    parser.add_argument("--fixed-alpha", type=float)
    parser.add_argument("--fixed-beta", type=float)
    parser.add_argument("--reference-alpha", type=float)
    parser.add_argument("--reference-beta", type=float)
    parser.add_argument("--w-dis", type=float, default=1.0)
    parser.add_argument("--base-spring", type=float, default=SPRING_STIFFNESS_BASE)
    parser.add_argument("--base-dir", type=float, default=DIRECTIONAL_FORCE_MAGNITUDE_BASE)
    parser.add_argument("--base-rep", type=float, default=REPULSION_STRENGTH_BASE)
    parser.add_argument("--outdir", default=str(Path(OUTPUT_DIR) / "ch5_detour_factor_sensitivity"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-scenario-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_detour_factor_sensitivity(
        seeds=_parse_seed_list(args.seeds),
        final_seeds=_parse_seed_list(args.final_seeds),
        kappa_min=args.kappa_min,
        kappa_max=args.kappa_max,
        kappa_step=args.kappa_step,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        alpha_step=args.alpha_step,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        beta_step=args.beta_step,
        fixed_alpha=args.fixed_alpha,
        fixed_beta=args.fixed_beta,
        reference_alpha=args.reference_alpha,
        reference_beta=args.reference_beta,
        w_dis=args.w_dis,
        base_spring_stiffness=args.base_spring,
        base_directional_force=args.base_dir,
        base_repulsion_strength=args.base_rep,
        outdir=args.outdir,
        resume=args.resume,
        generate_scenario_plots=not args.no_scenario_plots,
    )
    print(f"[Saved] {result['outdir']} ({result['n_completed_scenarios']} scenarios)")


if __name__ == "__main__":
    main()
