"""PhysicsSim-Full sensitivity to the angular-tolerance assumption.

Only the half-widths of the existing unsoftened sector hinge are scaled.
The lambda=1 reference uses the registered primary PhysicsSim hyperparameters;
the other six scenarios independently repeat the established anchor-LOOCV HPO.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Sequence

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pandas as pd

from library.config import (
    DIRECTIONAL_FORCE_MAGNITUDE_BASE,
    FILE_PATHS,
    REPULSION_STRENGTH_BASE,
    SPRING_STIFFNESS_BASE,
    theta_thr_4dir,
    theta_thr_8dir,
    refer_pos_sim as DEFAULT_REFER_POS_SIM,
)
from library.data_io import (
    get_anchor_labels,
    get_default_frame_anchor_label,
    get_test_site_labels,
    load_ini_data_from_csv,
    uploading_directional_data,
)
from library.directional_objectives import DIRECTIONAL_OBJECTIVE_SECTOR
from run_paper_script.ch5_ablation_progressive import _layout_metrics
from run_paper_script.ch5_anchor_split_robustness import (
    _append_event,
    _archive_incomplete_split,
    _bootstrap_ci_mean,
    _parse_seed_list,
    _sha256,
    _utc_now,
    _write_json,
)
from run_paper_script.ch5_hparam_kfold_gridsearch_pareto import (
    _resolve_anchor_and_test_inputs,
    _run_final_selected_model,
    run_anchor_loo_gridsearch_pareto,
)


FORMAL_NUMERATORS = tuple(range(6, -1, -1))
FORMAL_HPO_SEEDS = tuple(range(10))
FORMAL_FINAL_SEEDS = tuple(range(100))
FORMAL_GRID = {
    "alpha_min": -1.0,
    "alpha_max": 1.5,
    "alpha_step": 0.5,
    "beta_min": -2.0,
    "beta_max": 0.5,
    "beta_step": 0.5,
}
REFERENCE_ALPHA = 1.0
REFERENCE_BETA = -0.5
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260908

METRICS = (
    "RMSE_final_test_km",
    "E_distance_stress",
    "E_direction_vr",
    "E_direction_mae",
    "E_direction_vr_reference",
    "E_direction_mae_reference",
    "crowding_violation_rate_tau_0p1",
    "collapse_node_rate_tau_0p1",
    "nnd_q05_km",
    "distance_edge_crossing_rate",
)


def tolerance_scale(numerator: int) -> float:
    value = int(numerator)
    if value < 0 or value > 6:
        raise ValueError("Tolerance numerator must be an integer from 0 through 6.")
    return value / 6.0


def scenario_name(numerator: int) -> str:
    return f"scenario_theta_{int(numerator)}of6"


def angle_metadata(numerator: int) -> dict[str, float]:
    scale = tolerance_scale(numerator)
    return {
        "direction_tolerance_scale": scale,
        "cardinal_half_width_rad": float(theta_thr_4dir * scale),
        "diagonal_half_width_rad": float(theta_thr_8dir * scale),
        "cardinal_half_width_deg": float(math.degrees(theta_thr_4dir * scale)),
        "diagonal_half_width_deg": float(math.degrees(theta_thr_8dir * scale)),
    }


def _input_paths() -> dict[str, Path]:
    return {
        "site_points": Path(FILE_PATHS["ground_truth_path"]),
        "distance_edges": Path(FILE_PATHS["chen_data"]),
        "direction_edges": Path(FILE_PATHS["directional_data"]),
        "ini_data": Path(FILE_PATHS["ini_data"]),
    }


def _input_hashes() -> dict[str, str]:
    return {name: _sha256(path) for name, path in _input_paths().items()}


def _validate_formal_protocol(
    *,
    numerators: Sequence[int],
    hpo_seeds: Sequence[int],
    final_seeds: Sequence[int],
    alpha_min: float,
    alpha_max: float,
    alpha_step: float,
    beta_min: float,
    beta_max: float,
    beta_step: float,
    reference_alpha: float,
    reference_beta: float,
    allow_smoke: bool,
) -> None:
    if allow_smoke:
        return
    if tuple(numerators) != FORMAL_NUMERATORS:
        raise ValueError(f"Formal tolerance scenarios must be {FORMAL_NUMERATORS}.")
    if tuple(hpo_seeds) != FORMAL_HPO_SEEDS:
        raise ValueError("Formal HPO requires seeds 0-9.")
    if tuple(final_seeds) != FORMAL_FINAL_SEEDS:
        raise ValueError("Formal evaluation requires seeds 0-99.")
    actual = (alpha_min, alpha_max, alpha_step, beta_min, beta_max, beta_step)
    expected = tuple(
        FORMAL_GRID[key]
        for key in (
            "alpha_min",
            "alpha_max",
            "alpha_step",
            "beta_min",
            "beta_max",
            "beta_step",
        )
    )
    if not np.allclose(actual, expected, rtol=0.0, atol=1e-12):
        raise ValueError(f"Formal HPO grid must be {expected}; received {actual}.")
    if not np.isclose(reference_alpha, REFERENCE_ALPHA) or not np.isclose(
        reference_beta, REFERENCE_BETA
    ):
        raise ValueError("Formal reference must use alpha=1 and beta=-0.5.")


def _completed_scenario(
    scenario_dir: Path, *, numerator: int, final_seeds: Sequence[int]
) -> bool:
    required = (
        "gridsearch_config.json",
        "selected_final_summary.json",
        "selected_final_runs_by_seed.csv",
        "selected_final_site_errors.csv",
        "selected_final_positions_y_up_sim.csv",
        "scenario_runs_with_layout.csv",
        "angle_tolerance_audit.json",
    )
    if not all((scenario_dir / name).is_file() for name in required):
        return False
    try:
        config = json.loads((scenario_dir / "gridsearch_config.json").read_text(encoding="utf-8"))
        runs = pd.read_csv(scenario_dir / "scenario_runs_with_layout.csv")
        positions = pd.read_csv(scenario_dir / "selected_final_positions_y_up_sim.csv")
        audit = json.loads((scenario_dir / "angle_tolerance_audit.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError, pd.errors.ParserError):
        return False
    expected_scale = tolerance_scale(numerator)
    if not np.isclose(
        float(config.get("direction_tolerance_scale", np.nan)), expected_scale
    ):
        return False
    if not np.isclose(
        float(audit.get("direction_tolerance_scale", np.nan)), expected_scale
    ):
        return False
    expected_seeds = list(map(int, final_seeds))
    return (
        sorted(runs["seed"].astype(int).tolist()) == expected_seeds
        and len(positions) == len(expected_seeds) * 35
        and set(METRICS).issubset(runs.columns)
    )


def preflight_direction_tolerance_sensitivity(
    *,
    numerators: Sequence[int],
    hpo_seeds: Sequence[int],
    final_seeds: Sequence[int],
    alpha_min: float,
    alpha_max: float,
    alpha_step: float,
    beta_min: float,
    beta_max: float,
    beta_step: float,
    reference_alpha: float,
    reference_beta: float,
    outdir: str | Path,
    allow_smoke: bool = False,
    resume: bool = False,
) -> dict:
    numerators = [int(value) for value in numerators]
    if not numerators or len(numerators) != len(set(numerators)):
        raise ValueError("Tolerance scenario numerators must be non-empty and unique.")
    for value in numerators:
        tolerance_scale(value)
    hpo_seeds = [int(seed) for seed in hpo_seeds]
    final_seeds = [int(seed) for seed in final_seeds]
    for name, values in (("HPO seeds", hpo_seeds), ("final seeds", final_seeds)):
        if not values or len(values) != len(set(values)) or any(seed < 0 for seed in values):
            raise ValueError(f"{name} must be distinct non-negative integers.")
    _validate_formal_protocol(
        numerators=numerators,
        hpo_seeds=hpo_seeds,
        final_seeds=final_seeds,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        alpha_step=alpha_step,
        beta_min=beta_min,
        beta_max=beta_max,
        beta_step=beta_step,
        reference_alpha=reference_alpha,
        reference_beta=reference_beta,
        allow_smoke=allow_smoke,
    )
    missing = [str(path) for path in _input_paths().values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required inputs are missing: {missing}")
    _graph, vertices, dni, _edges, distances = load_ini_data_from_csv(FILE_PATHS)
    directions = uploading_directional_data()
    if len(vertices) != 35 or len(dni) != 35 or len(distances) != 44 or len(directions) != 44:
        raise ValueError("Expected the registered 35-node, 44-distance, 44-direction dataset.")
    anchors = get_anchor_labels()
    tests = get_test_site_labels()
    frame = get_default_frame_anchor_label()
    if len(anchors) != 3 or len(tests) != 8 or set(anchors) & set(tests) or frame not in anchors:
        raise ValueError("Registered anchor/test split is invalid.")
    output = Path(outdir)
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f"Output folder is non-empty: {output}. Use --resume or a new path.")
    completed = incomplete = 0
    for numerator in numerators:
        path = output / "scenarios" / scenario_name(numerator)
        if _completed_scenario(path, numerator=numerator, final_seeds=final_seeds):
            completed += 1
        elif path.exists() and any(path.iterdir()):
            incomplete += 1
    alpha_count = int(round((alpha_max - alpha_min) / alpha_step)) + 1
    beta_count = int(round((beta_max - beta_min) / beta_step)) + 1
    hpo_per_scenario = alpha_count * beta_count * 3 * len(hpo_seeds)
    n_hpo_scenarios = len([value for value in numerators if value != 6])
    disk_parent = output.parent
    while not disk_parent.exists() and disk_parent != disk_parent.parent:
        disk_parent = disk_parent.parent
    free_disk = int(shutil.disk_usage(disk_parent).free)
    if free_disk < 1_000_000_000:
        raise OSError("Less than 1 GB of free disk space remains.")
    return {
        "status": "passed",
        "checked_at_utc": _utc_now(),
        "scenario_numerators": numerators,
        "scenario_metadata": [
            {"numerator": value, **angle_metadata(value)} for value in numerators
        ],
        "hpo_seeds": hpo_seeds,
        "final_evaluation_seeds": final_seeds,
        "hpo_runs_per_nonreference_scenario": hpo_per_scenario,
        "final_runs_per_scenario": len(final_seeds),
        "expected_total_model_runs": (
            n_hpo_scenarios * hpo_per_scenario + len(numerators) * len(final_seeds)
        ),
        "reference_alpha": float(reference_alpha),
        "reference_beta": float(reference_beta),
        "anchor_labels": anchors,
        "test_labels": tests,
        "final_frame_anchor_label": frame,
        "existing_completed_scenarios": completed,
        "existing_incomplete_scenarios": incomplete,
        "free_disk_bytes": free_disk,
        "input_sha256": _input_hashes(),
        "allow_smoke": bool(allow_smoke),
        "resume": bool(resume),
    }


def _attach_layout_metrics(scenario_dir: Path) -> pd.DataFrame:
    runs = pd.read_csv(scenario_dir / "selected_final_runs_by_seed.csv")
    positions = pd.read_csv(scenario_dir / "selected_final_positions_y_up_sim.csv")
    _graph, vertices, dni, _edges, distances = load_ini_data_from_csv(FILE_PATHS)
    rows = []
    for seed, group in positions.groupby("seed", sort=True):
        ordered = group.sort_values("node_idx")
        if ordered["label"].tolist() != list(vertices):
            raise ValueError(f"Position order mismatch for seed {seed} in {scenario_dir}.")
        points = ordered[["x_y_up_sim", "y_y_up_sim"]].to_numpy(float)
        rows.append({"seed": int(seed), **_layout_metrics(points, vertices, dni, distances)})
    result = runs.merge(pd.DataFrame(rows), on="seed", how="left", validate="one_to_one")
    if not np.isfinite(result[list(METRICS)].to_numpy(float)).all():
        raise ValueError(f"Non-finite final metric in {scenario_dir}.")
    result.to_csv(scenario_dir / "scenario_runs_with_layout.csv", index=False, encoding="utf-8-sig")
    return result


def _summarize_scenario(scenario_dir: Path, numerator: int) -> tuple[dict, pd.DataFrame]:
    runs = pd.read_csv(scenario_dir / "scenario_runs_with_layout.csv")
    final = json.loads((scenario_dir / "selected_final_summary.json").read_text(encoding="utf-8"))
    config = json.loads((scenario_dir / "gridsearch_config.json").read_text(encoding="utf-8"))
    policy = config["hyperparameter_policy"]
    row = {
        "scenario_id": scenario_name(numerator),
        "tolerance_numerator": int(numerator),
        **angle_metadata(numerator),
        "hyperparameter_policy": policy,
        "selected_alpha": float(final["alpha"]),
        "selected_beta": float(final["beta"]),
        "selected_on_alpha_boundary": bool(final.get("selected_on_alpha_boundary", False)),
        "selected_on_beta_boundary": bool(final.get("selected_on_beta_boundary", False)),
        "selected_on_grid_boundary": bool(final.get("selected_on_grid_boundary", False)),
        "n_final_runs": int(len(runs)),
    }
    for index, metric in enumerate(METRICS):
        values = runs[metric].to_numpy(float)
        lo, hi = _bootstrap_ci_mean(
            values, seed=BOOTSTRAP_SEED + int(numerator) * 100 + index
        )
        row[f"{metric}_mean"] = float(values.mean())
        row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        row[f"{metric}_ci95_low"] = lo
        row[f"{metric}_ci95_high"] = hi
    runs.insert(0, "tolerance_numerator", int(numerator))
    expected_scale = tolerance_scale(numerator)
    if "direction_tolerance_scale" in runs.columns:
        if not np.allclose(
            runs["direction_tolerance_scale"].to_numpy(float),
            expected_scale,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"Tolerance scale mismatch in {scenario_dir}.")
    else:
        runs.insert(1, "direction_tolerance_scale", expected_scale)
    return row, runs


def paired_comparisons(all_runs: pd.DataFrame) -> pd.DataFrame:
    reference = all_runs[all_runs["tolerance_numerator"].eq(6)].copy()
    if reference.empty or reference["seed"].duplicated().any():
        raise ValueError("A unique lambda=1 reference is required for paired comparisons.")
    rows = []
    for numerator, scenario in all_runs.groupby("tolerance_numerator", sort=True):
        if scenario["seed"].duplicated().any() or set(scenario["seed"]) != set(reference["seed"]):
            raise ValueError(f"Scenario {numerator}/6 cannot be paired by seed with reference.")
        merged = scenario.merge(
            reference, on="seed", suffixes=("_scenario", "_reference"), validate="one_to_one"
        )
        for index, metric in enumerate(METRICS):
            differences = (
                merged[f"{metric}_scenario"].to_numpy(float)
                - merged[f"{metric}_reference"].to_numpy(float)
            )
            lo, hi = _bootstrap_ci_mean(
                differences,
                seed=BOOTSTRAP_SEED + 10_000 + int(numerator) * 100 + index,
            )
            rows.append(
                {
                    "tolerance_numerator": int(numerator),
                    **angle_metadata(int(numerator)),
                    "reference_numerator": 6,
                    "metric": metric,
                    "n_pairs": int(len(differences)),
                    "difference_mean": float(differences.mean()),
                    "difference_std": (
                        float(differences.std(ddof=1)) if len(differences) > 1 else float("nan")
                    ),
                    "difference_ci95_low": lo,
                    "difference_ci95_high": hi,
                    "win_rate_lower": float(np.mean(differences < 0.0)),
                }
            )
    return pd.DataFrame(rows)


def _save_plots(summary: pd.DataFrame, paired: pd.DataFrame, outdir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = summary.sort_values("direction_tolerance_scale")
    rmse_paired = paired[paired["metric"].eq("RMSE_final_test_km")].sort_values(
        "direction_tolerance_scale"
    )
    x = rmse_paired["direction_tolerance_scale"].to_numpy(float)
    y = rmse_paired["difference_mean"].to_numpy(float)
    lo = rmse_paired["difference_ci95_low"].to_numpy(float)
    hi = rmse_paired["difference_ci95_high"].to_numpy(float)
    labels = [f"{int(row.tolerance_numerator)}/6" for row in rmse_paired.itertuples()]
    angle_labels = [
        f"{row.cardinal_half_width_deg:g}/{row.diagonal_half_width_deg:g}°"
        for row in rmse_paired.itertuples()
    ]
    fig, ax = plt.subplots(figsize=(11.8, 7.0))
    ax.errorbar(
        x,
        y,
        yerr=np.vstack((y - lo, hi - y)),
        marker="o",
        markersize=5.0,
        linewidth=1.8,
        elinewidth=1.0,
        capsize=2.5,
        color="#1769aa",
        label="Paired mean difference (95% CI)",
    )
    ax.axhline(
        0.0,
        color="#D55E00",
        linestyle="--",
        linewidth=1.4,
        label="λ = 1.00 reference",
    )
    ax.set_xlim(-0.025, 1.025)
    ax.set_xticks(x, labels)
    ax.set_xlabel("Direction-tolerance multiplier, λ")
    ax.set_ylabel("Paired ΔRMSE vs λ = 1.00 (km)")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(True, alpha=0.2)

    secondary = ax.secondary_xaxis("top")
    secondary.set_xticks(x, angle_labels)
    secondary.set_xlabel("Cardinal/diagonal tolerance half-width (°)")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(outdir / f"direction_tolerance_paired_rmse.{suffix}", dpi=300)
    plt.close(fig)

    x = ordered["direction_tolerance_scale"].to_numpy(float)
    y = ordered["RMSE_final_test_km_mean"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.plot(x, y, marker="o", linewidth=2.0, color="#0072B2")
    ax.fill_between(
        x,
        ordered["RMSE_final_test_km_ci95_low"].to_numpy(float),
        ordered["RMSE_final_test_km_ci95_high"].to_numpy(float),
        color="#0072B2",
        alpha=0.18,
        label="95% bootstrap CI across seeds",
    )
    ax.set_xlabel("Direction-tolerance multiplier, lambda")
    ax.set_ylabel("Held-out test RMSE (km)")
    ax.grid(alpha=0.22)
    ax.legend(loc="best")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(outdir / f"direction_tolerance_raw_rmse.{suffix}", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.5), sharex=True)
    for ax, column, label, color in (
        (axes[0], "selected_alpha", "Selected alpha", "#0072B2"),
        (axes[1], "selected_beta", "Selected beta", "#E69F00"),
    ):
        ax.step(x, ordered[column].to_numpy(float), where="mid", color=color)
        ax.scatter(x, ordered[column].to_numpy(float), color=color)
        ax.set_ylabel(label)
        ax.grid(alpha=0.22)
    axes[-1].set_xlabel("Direction-tolerance multiplier, lambda")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(outdir / f"direction_tolerance_selected_hyperparameters.{suffix}", dpi=300)
    plt.close(fig)


def _write_aggregate(
    outdir: Path,
    summaries: Sequence[dict],
    run_frames: Sequence[pd.DataFrame],
    expected_scenarios: int,
) -> None:
    if not summaries:
        return
    summary = pd.DataFrame(summaries).sort_values(
        "direction_tolerance_scale", ascending=False
    )
    runs = pd.concat(run_frames, ignore_index=True)
    paired = paired_comparisons(runs) if summary["tolerance_numerator"].eq(6).any() else pd.DataFrame()
    summary.to_csv(outdir / "direction_tolerance_scenario_summary.csv", index=False, encoding="utf-8-sig")
    runs.to_csv(outdir / "direction_tolerance_final_runs.csv", index=False, encoding="utf-8-sig")
    if not paired.empty:
        paired.to_csv(outdir / "direction_tolerance_paired_comparisons.csv", index=False, encoding="utf-8-sig")
        _save_plots(summary, paired, outdir)
    _write_json(
        outdir / "direction_tolerance_global_summary.json",
        {
            "estimand": "held-out RMSE sensitivity to angular-tolerance assumptions",
            "n_completed_scenarios": int(len(summary)),
            "n_expected_scenarios": int(expected_scenarios),
            "reference_tolerance_scale": 1.0,
            "n_selected_on_grid_boundary": int(summary["selected_on_grid_boundary"].sum()),
            "heldout_policy": "Held-out RMSE is diagnostic and never selects a tolerance scenario.",
            "direction_metric_policy": (
                "Native metrics use scenario tolerance; reference metrics always use the original 90/45-degree thresholds."
            ),
        },
    )


def run_direction_tolerance_sensitivity(
    *,
    numerators: Sequence[int],
    hpo_seeds: Sequence[int],
    final_seeds: Sequence[int],
    alpha_min: float,
    alpha_max: float,
    alpha_step: float,
    beta_min: float,
    beta_max: float,
    beta_step: float,
    reference_alpha: float,
    reference_beta: float,
    outdir: str | Path,
    allow_smoke: bool = False,
    resume: bool = False,
    generate_scenario_plots: bool = True,
) -> dict:
    preflight = preflight_direction_tolerance_sensitivity(
        numerators=numerators,
        hpo_seeds=hpo_seeds,
        final_seeds=final_seeds,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        alpha_step=alpha_step,
        beta_min=beta_min,
        beta_max=beta_max,
        beta_step=beta_step,
        reference_alpha=reference_alpha,
        reference_beta=reference_beta,
        outdir=outdir,
        allow_smoke=allow_smoke,
        resume=resume,
    )
    output = Path(outdir)
    output.mkdir(parents=True, exist_ok=True)
    scenarios_root = output / "scenarios"
    scenarios_root.mkdir(exist_ok=True)
    archive_root = output / "interrupted_attempts"
    event_log = output / "experiment_events.jsonl"
    config = {
        "experiment": "direction_tolerance_sensitivity",
        "directional_objective": DIRECTIONAL_OBJECTIVE_SECTOR,
        "directional_objective_formula": "0.5*w_dir*max(0, abs(phi)-lambda*theta_h_base)^2",
        "softening": False,
        "scenario_numerators": list(map(int, numerators)),
        "scenario_metadata": preflight["scenario_metadata"],
        "hpo_policy": "independent HPO for lambda<1; registered primary hyperparameters for lambda=1",
        "hpo_seeds": list(map(int, hpo_seeds)),
        "final_evaluation_seeds": list(map(int, final_seeds)),
        "alpha_range": [alpha_min, alpha_max, alpha_step],
        "beta_range": [beta_min, beta_max, beta_step],
        "reference_alpha": float(reference_alpha),
        "reference_beta": float(reference_beta),
        "anchor_labels": preflight["anchor_labels"],
        "test_labels": preflight["test_labels"],
        "final_frame_anchor_label": preflight["final_frame_anchor_label"],
        "input_sha256": preflight["input_sha256"],
    }
    config_path = output / "direction_tolerance_experiment_config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != config:
            raise ValueError("Resume configuration differs; use a new output folder.")
    else:
        _write_json(config_path, config)
    _write_json(output / "preflight_report.json", preflight)

    summaries = []
    run_frames = []
    for index, numerator in enumerate(numerators, start=1):
        numerator = int(numerator)
        scale = tolerance_scale(numerator)
        name = scenario_name(numerator)
        scenario_dir = scenarios_root / name
        print(f"[{index}/{len(numerators)}] lambda={numerator}/6 ({scale:.6f})", flush=True)
        if _completed_scenario(scenario_dir, numerator=numerator, final_seeds=final_seeds):
            if not resume:
                raise FileExistsError(f"Completed scenario exists: {scenario_dir}")
            print("  [Resume] using completed scenario", flush=True)
        else:
            if scenario_dir.exists() and any(scenario_dir.iterdir()):
                if not resume:
                    raise RuntimeError(f"Incomplete scenario exists: {scenario_dir}. Use --resume.")
                archived = _archive_incomplete_split(scenario_dir, archive_root)
                print(f"  [Resume] archived incomplete attempt to {archived}", flush=True)
            started = time.perf_counter()
            if numerator == 6:
                scenario_dir.mkdir(parents=True, exist_ok=True)
                anchors, anchor_lonlat, tests, test_lonlat = _resolve_anchor_and_test_inputs(
                    load_ini_data_from_csv(FILE_PATHS)[2]
                )
                _write_json(
                    scenario_dir / "gridsearch_config.json",
                    {
                        "experiment": "direction_tolerance_fixed_reference",
                        "hyperparameter_policy": "fixed_reference",
                        "directional_objective": DIRECTIONAL_OBJECTIVE_SECTOR,
                        "direction_tolerance_scale": scale,
                        "alpha": float(reference_alpha),
                        "beta": float(reference_beta),
                        "hpo_seeds": [],
                        "final_evaluation_seeds": list(map(int, final_seeds)),
                    },
                )
                _run_final_selected_model(
                    selected=pd.Series({"alpha": reference_alpha, "beta": reference_beta}),
                    anchor_labels=anchors,
                    anchor_lonlat=anchor_lonlat,
                    test_labels=tests,
                    test_lonlat=test_lonlat,
                    seeds=final_seeds,
                    w_dis=1.0,
                    base_spring_stiffness=SPRING_STIFFNESS_BASE,
                    base_directional_force=DIRECTIONAL_FORCE_MAGNITUDE_BASE,
                    base_repulsion_strength=REPULSION_STRENGTH_BASE,
                    refer_pos_sim=DEFAULT_REFER_POS_SIM,
                    outdir=scenario_dir,
                    selection_rule="registered_primary_reference",
                    final_frame_anchor_label=get_default_frame_anchor_label(),
                    save_final_positions=True,
                    directional_objective=DIRECTIONAL_OBJECTIVE_SECTOR,
                    direction_tolerance_scale=scale,
                )
            else:
                run_anchor_loo_gridsearch_pareto(
                    seeds=hpo_seeds,
                    final_seeds=final_seeds,
                    alpha_min=alpha_min,
                    alpha_max=alpha_max,
                    alpha_step=alpha_step,
                    beta_min=beta_min,
                    beta_max=beta_max,
                    beta_step=beta_step,
                    w_dis=1.0,
                    base_spring_stiffness=SPRING_STIFFNESS_BASE,
                    base_directional_force=DIRECTIONAL_FORCE_MAGNITUDE_BASE,
                    base_repulsion_strength=REPULSION_STRENGTH_BASE,
                    refer_pos_sim=DEFAULT_REFER_POS_SIM,
                    outdir=scenario_dir,
                    generate_plots=generate_scenario_plots,
                    save_final_positions=True,
                    directional_objective=DIRECTIONAL_OBJECTIVE_SECTOR,
                    direction_tolerance_scale=scale,
                    fail_on_selected_boundary=False,
                )
                scenario_config = json.loads(
                    (scenario_dir / "gridsearch_config.json").read_text(encoding="utf-8")
                )
                scenario_config["hyperparameter_policy"] = "scenario_specific_hpo"
                _write_json(scenario_dir / "gridsearch_config.json", scenario_config)
            _write_json(
                scenario_dir / "angle_tolerance_audit.json",
                {
                    "tolerance_numerator": numerator,
                    **angle_metadata(numerator),
                    "base_cardinal_half_width_deg": math.degrees(theta_thr_4dir),
                    "base_diagonal_half_width_deg": math.degrees(theta_thr_8dir),
                    "directional_objective": DIRECTIONAL_OBJECTIVE_SECTOR,
                    "softening": False,
                },
            )
            _attach_layout_metrics(scenario_dir)
            _append_event(
                event_log,
                {
                    "event": "scenario_completed",
                    "scenario_id": name,
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )
        summary, runs = _summarize_scenario(scenario_dir, numerator)
        summaries.append(summary)
        run_frames.append(runs)
        _write_aggregate(output, summaries, run_frames, len(numerators))
    return {"outdir": output, "summary": pd.DataFrame(summaries)}


def _parse_numerators(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one scenario numerator is required.")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numerators", default="6,5,4,3,2,1,0")
    parser.add_argument("--seeds", default=",".join(map(str, FORMAL_HPO_SEEDS)))
    parser.add_argument("--final-seeds", default=",".join(map(str, FORMAL_FINAL_SEEDS)))
    for name, default in FORMAL_GRID.items():
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=default)
    parser.add_argument("--reference-alpha", type=float, default=REFERENCE_ALPHA)
    parser.add_argument("--reference-beta", type=float, default=REFERENCE_BETA)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--no-scenario-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_direction_tolerance_sensitivity(
        numerators=_parse_numerators(args.numerators),
        hpo_seeds=_parse_seed_list(args.seeds),
        final_seeds=_parse_seed_list(args.final_seeds),
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        alpha_step=args.alpha_step,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        beta_step=args.beta_step,
        reference_alpha=args.reference_alpha,
        reference_beta=args.reference_beta,
        outdir=args.outdir,
        allow_smoke=args.allow_smoke,
        resume=args.resume,
        generate_scenario_plots=not args.no_scenario_plots,
    )


if __name__ == "__main__":
    main()
