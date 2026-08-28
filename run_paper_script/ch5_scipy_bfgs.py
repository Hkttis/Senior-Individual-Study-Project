"""Optimize the PhysicsSim objective with SciPy's full-memory BFGS.

This independent numerical baseline uses the same objective coefficients,
fixed anchors, input data, initialization, metrics, and final visualization
conventions as PhysicsSim-Full. It never modifies HPO or AS output.
"""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Sequence

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import scipy

plt.rcParams["font.family"] = ["Microsoft JhengHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from library.config import FILE_PATHS, refer_pos_sim
from library.data_io import (
    get_anchor_align_label,
    get_anchor_labels,
    get_test_site_labels,
    load_ini_data_from_csv,
    uploading_directional_data,
    uploading_ground_truth,
)
from library.geometry import get_lcc_bounds, get_lcc_parameters
from library.initialization import generate_CHEN_initial_positions
from library.scipy_minimizer import (
    DEFAULT_BFGS_GTOL,
    DEFAULT_BFGS_MAXITER_PER_DIMENSION,
    objective_and_gradient,
    run_bfgs,
)
from library.scipy_objective import (
    PHYSICS_HPO_SELECTED_ALPHA,
    PHYSICS_HPO_SELECTED_BETA,
    PHYSICS_HPO_SELECTED_W_DIS,
    FixedAnchorObjective,
    ObjectiveWeights,
    build_current_objective,
)
from library.units import data_Li2sim
from library.visualization import ground_truth_comparison, visualize_error_map_official
from MDS_model.plot_node_link_diagram import wrong_directions_nonflip
from run_paper_script.ch5_ablation_progressive import (
    METRICS,
    _evaluate,
    _series_stats,
    _target_positions_sim,
)
from run_paper_script.ch6_visualize_progressive_representative import (
    _flip_y_up_for_display,
)


DEFAULT_OUTDIR = "outputs/ch5_scipy_bfgs_smoke"
CORE_METRICS = (
    "RMSE_test_km",
    "E_distance_stress",
    "E_direction_vr",
    "E_direction_mae",
)
LOG_TICK_FORMATTER = FuncFormatter(lambda value, _position: f"{value:.0e}")


def _parse_seeds(raw: str) -> list[int]:
    seeds = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not seeds:
        raise ValueError("--seeds cannot be empty")
    if len(set(seeds)) != len(seeds):
        raise ValueError("--seeds cannot contain duplicate values")
    return seeds


def _as_mapping(value, keys: Sequence[str]) -> dict[str, float]:
    if isinstance(value, dict):
        return {key: float(value[key]) for key in keys}
    return {key: float(item) for key, item in zip(keys, value, strict=True)}


def _initial_free_vector(
    seed: int,
    problem: FixedAnchorObjective,
    vertice,
    dni,
    calibration_labels,
    calibration_lonlat,
    anchor_label,
) -> np.ndarray:
    """Reproduce the PhysicsSim-Full initialization in the anchor frame."""

    np.random.seed(seed)
    generated_vertices, generated_dni, _data, points, _fixed = (
        generate_CHEN_initial_positions(
            list(refer_pos_sim),
            list(calibration_labels),
            list(calibration_lonlat),
            anchor_label=anchor_label,
        )
    )
    if list(generated_vertices) != list(vertice) or generated_dni != dni:
        raise ValueError("Initialization vertex order differs from the formal graph order.")
    if tuple(vertice) != problem.vertices:
        raise ValueError("SciPy objective vertex order differs from the formal graph order.")

    centered = np.asarray(points, dtype=np.float64) - np.asarray(
        refer_pos_sim, dtype=np.float64
    )
    if not np.allclose(
        centered[problem.anchor_indices],
        problem.anchor_coordinates,
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError("Initialized anchors do not match the SciPy objective anchors.")
    return problem.pack(centered)


def _history_record(problem: FixedAnchorObjective, y, iteration: int) -> dict:
    vector = np.asarray(y, dtype=np.float64).copy()
    components = problem.components(vector)
    _value, gradient = objective_and_gradient(vector, problem)
    return {
        "iteration": int(iteration),
        "y": vector,
        "objective_total": float(components.total),
        "objective_distance_raw": float(components.distance),
        "objective_direction_raw": float(components.direction),
        "objective_repulsion_raw": float(components.repulsion),
        "objective_distance_weighted": float(components.weighted_distance),
        "objective_direction_weighted": float(components.weighted_direction),
        "objective_repulsion_weighted": float(components.weighted_repulsion),
        "gradient_norm_inf": float(np.linalg.norm(gradient, ord=np.inf)),
    }


def _append_distinct_final(history: list[dict], problem, y) -> None:
    if y is None:
        return
    vector = np.asarray(y, dtype=np.float64)
    if history and np.array_equal(history[-1]["y"], vector):
        return
    history.append(_history_record(problem, vector, len(history)))


def _positions_y_up(problem: FixedAnchorObjective, y) -> np.ndarray:
    return problem.unpack(y) + np.asarray(refer_pos_sim, dtype=np.float64)


def _save_history(seed_dir: Path, seed: int, history, problem, vertice) -> None:
    scalar_rows = []
    position_rows = []
    for record in history:
        scalar_rows.append(
            {"seed": seed, **{key: value for key, value in record.items() if key != "y"}}
        )
        points = _positions_y_up(problem, record["y"])
        position_rows.extend(
            {
                "seed": seed,
                "iteration": record["iteration"],
                "label": label,
                "x_y_up_sim": float(points[index, 0]),
                "y_y_up_sim": float(points[index, 1]),
            }
            for index, label in enumerate(vertice)
        )
    pd.DataFrame(scalar_rows).to_csv(
        seed_dir / "bfgs_objective_history.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(position_rows).to_csv(
        seed_dir / "bfgs_position_history_y_up_sim.csv",
        index=False,
        encoding="utf-8-sig",
    )


def _plot_objective_history(seed_dir: Path, history, seed: int) -> None:
    frame = pd.DataFrame(
        [{key: value for key, value in row.items() if key != "y"} for row in history]
    )
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)
    final_objective = float(frame["objective_total"].iloc[-1])
    objective_gap = frame["objective_total"].to_numpy(float) - final_objective
    gap_floor = max(float(np.max(objective_gap)) * 1e-12, 1e-12)
    objective_gap = np.maximum(objective_gap, gap_floor)
    axes[0].plot(frame["iteration"], objective_gap, color="#222222", lw=2)
    axes[0].set(
        title=(
            f"SciPy BFGS objective history (seed {seed}; "
            f"final objective={final_objective:.6g})"
        ),
        ylabel="Objective gap to final",
    )
    axes[0].set_yscale("log")
    axes[0].yaxis.set_major_formatter(LOG_TICK_FORMATTER)
    for values, label, color in (
        (frame["objective_distance_weighted"], "Distance", "#2474b5"),
        (frame["objective_direction_weighted"], "Direction", "#d95f02"),
        (-frame["objective_repulsion_weighted"], "-Repulsion", "#238b45"),
    ):
        values = np.asarray(values, dtype=float)
        component_floor = max(float(np.max(values)) * 1e-12, 1e-12)
        axes[1].plot(
            frame["iteration"],
            np.maximum(values, component_floor),
            label=label,
            lw=1.7,
            color=color,
        )
    axes[1].set(xlabel="Accepted BFGS iteration", ylabel="Weighted objective component")
    axes[1].set_yscale("log")
    axes[1].yaxis.set_major_formatter(LOG_TICK_FORMATTER)
    axes[1].legend(frameon=False, ncol=3)
    axes[2].plot(
        frame["iteration"], frame["gradient_norm_inf"], color="#6a3d9a", lw=1.7
    )
    axes[2].set_yscale("log")
    axes[2].yaxis.set_major_formatter(LOG_TICK_FORMATTER)
    axes[2].set(
        xlabel="Accepted BFGS iteration",
        ylabel="Gradient infinity norm",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
    for extension in ("png", "svg"):
        fig.savefig(seed_dir / f"bfgs_objective_history.{extension}", dpi=300)
    plt.close(fig)


def _snapshot_indices(n_states: int, count: int = 6) -> list[int]:
    if n_states <= count:
        return list(range(n_states))
    return sorted(set(np.linspace(0, n_states - 1, count, dtype=int).tolist()))


def _draw_configuration(axis, points, distance_data, dni, labels, title):
    for row in distance_data:
        a, b = dni[row[0]], dni[row[1]]
        axis.plot(
            [points[a, 0], points[b, 0]],
            [points[a, 1], points[b, 1]],
            color="#b8c1c8",
            lw=0.7,
            zorder=1,
        )
    axis.scatter(points[:, 0], points[:, 1], s=18, color="#1f6f8b", zorder=2)
    for label in labels:
        index = dni[label]
        axis.scatter(points[index, 0], points[index, 1], s=42, color="#c62828", zorder=3)
        axis.annotate(
            label,
            points[index],
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            color="#8e0000",
        )
    axis.set_title(title, fontsize=11)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.18)


def _plot_configuration_evolution(
    seed_dir, history, problem, distance_data, dni, calibration_labels, seed
) -> None:
    selected = _snapshot_indices(len(history))
    states = [_positions_y_up(problem, history[index]["y"]) for index in selected]
    all_points = np.vstack(states)
    pad = max(float(np.ptp(all_points[:, 0])), float(np.ptp(all_points[:, 1]))) * 0.05
    pad = max(pad, 1.0)
    limits = (
        float(all_points[:, 0].min() - pad),
        float(all_points[:, 0].max() + pad),
        float(all_points[:, 1].min() - pad),
        float(all_points[:, 1].max() + pad),
    )
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    axes = axes.ravel()
    for panel, history_index, points in zip(axes, selected, states):
        iteration = int(history[history_index]["iteration"])
        _draw_configuration(
            panel,
            points,
            distance_data,
            dni,
            calibration_labels,
            f"Iteration {iteration}",
        )
        panel.set_xlim(limits[0], limits[1])
        panel.set_ylim(limits[2], limits[3])
    for panel in axes[len(selected):]:
        panel.set_visible(False)
    fig.suptitle(f"SciPy BFGS configuration evolution (seed {seed})", fontsize=16)
    for extension in ("png", "svg"):
        fig.savefig(seed_dir / f"bfgs_configuration_evolution.{extension}", dpi=300)
    plt.close(fig)


def _plot_metric_summary(seed_dir: Path, metrics: dict, seed: int) -> None:
    labels = ("Test RMSE (km)", "Stress", "Violation Rate", "Mean Angular Error (rad)")
    values = [float(metrics[key]) for key in CORE_METRICS]
    colors = ("#2c7fb8", "#4daf9c", "#f28e2b", "#d62728")
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    for axis, label, value, color in zip(axes.ravel(), labels, values, colors):
        axis.set_facecolor("#f7f7f7")
        axis.text(0.5, 0.62, label, ha="center", va="center", fontsize=13, color="#333333")
        axis.text(0.5, 0.34, f"{value:.6g}", ha="center", va="center", fontsize=22, color=color, weight="bold")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#cccccc")
    fig.suptitle(f"SciPy BFGS final evaluation metrics (seed {seed})", fontsize=15)
    for extension in ("png", "svg"):
        fig.savefig(seed_dir / f"bfgs_final_metrics.{extension}", dpi=300)
    plt.close(fig)


def _save_evolution_gif(seed_dir, history, problem, distance_data, dni, calibration_labels, seed):
    frame_indices = _snapshot_indices(len(history), count=60)
    states = [_positions_y_up(problem, history[index]["y"]) for index in frame_indices]
    all_points = np.vstack(states)
    pad = max(float(np.ptp(all_points[:, 0])), float(np.ptp(all_points[:, 1]))) * 0.05
    pad = max(pad, 1.0)
    fig, axis = plt.subplots(figsize=(9, 7), constrained_layout=True)

    def update(frame):
        axis.clear()
        history_index = frame_indices[frame]
        _draw_configuration(
            axis,
            states[frame],
            distance_data,
            dni,
            calibration_labels,
            f"SciPy BFGS seed {seed}, iteration {int(history[history_index]['iteration'])}",
        )
        axis.set_xlim(all_points[:, 0].min() - pad, all_points[:, 0].max() + pad)
        axis.set_ylim(all_points[:, 1].min() - pad, all_points[:, 1].max() + pad)

    animation = FuncAnimation(fig, update, frames=len(states), interval=180, repeat=False)
    animation.save(seed_dir / "bfgs_configuration_evolution.gif", writer=PillowWriter(fps=6))
    plt.close(fig)


def _summary_table(runs: pd.DataFrame) -> pd.DataFrame:
    ok = runs[runs["status"] == "ok"]
    rows = []
    for metric in METRICS:
        values = ok[metric].dropna().to_numpy(float)
        if len(values):
            rows.append({"variant": "SciPy-BFGS", "metric": metric, **_series_stats(values)})
    return pd.DataFrame(rows)


def run_experiment(
    *,
    seeds: Sequence[int],
    outdir: str | Path,
    gtol: float = DEFAULT_BFGS_GTOL,
    maxiter: int | None = None,
    visualize_seeds: Sequence[int] = (),
    make_gif: bool = True,
    alpha: float = PHYSICS_HPO_SELECTED_ALPHA,
    beta: float = PHYSICS_HPO_SELECTED_BETA,
    w_dis: float = PHYSICS_HPO_SELECTED_W_DIS,
    hpo_source: str | None = None,
) -> dict:
    outdir = Path(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)

    weights = ObjectiveWeights.from_physics_hpo(alpha=alpha, beta=beta, w_dis=w_dis)
    problem = build_current_objective(weights=weights)
    graph, vertice, dni, edges, distance_data = load_ini_data_from_csv(FILE_PATHS)
    del graph, edges
    gt_lonlat = uploading_ground_truth(vertice, dni)
    calibration_labels = get_anchor_labels()
    anchor_label = get_anchor_align_label()
    test_labels = get_test_site_labels()
    if len(calibration_labels) != 3 or anchor_label not in calibration_labels:
        raise ValueError("SciPy BFGS requires three calibration anchors including anchor_align.")
    calibration_lonlat = [tuple(gt_lonlat[dni[label]]) for label in calibration_labels]
    targets = _target_positions_sim(dni, gt_lonlat, anchor_label, refer_pos_sim)
    data_sim = data_Li2sim(distance_data)
    directional_data = uploading_directional_data()
    visualize_set = set(int(seed) for seed in visualize_seeds)

    run_rows = []
    final_positions_rows = []
    for seed in seeds:
        seed = int(seed)
        print(f"[SciPy-BFGS] seed={seed}")
        seed_dir = outdir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=False)
        initial = _initial_free_vector(
            seed,
            problem,
            vertice,
            dni,
            calibration_labels,
            calibration_lonlat,
            anchor_label,
        )
        history = [_history_record(problem, initial, 0)]

        def callback(xk):
            history.append(_history_record(problem, xk, len(history)))

        result = run_bfgs(initial, problem, gtol=gtol, maxiter=maxiter, callback=callback)
        _append_distinct_final(history, problem, result.get("y_final"))
        selected_y = result.get("y_final")
        evaluated_state = "scipy_final"
        if selected_y is None:
            selected_y = history[-1]["y"]
            evaluated_state = "last_valid_accepted"
        points = _positions_y_up(problem, selected_y)
        metrics = _evaluate(
            "SciPy-BFGS",
            seed,
            points,
            vertice,
            dni,
            data_sim,
            directional_data,
            test_labels,
            targets,
            distance_data,
        )
        metrics["status"] = "ok" if result["success"] else "optimizer_failed"
        metrics["error"] = "" if result["success"] else str(result["failure_reason"])
        metrics.update(
            {
                "optimizer": "SciPy minimize(method=BFGS)",
                "optimizer_success": bool(result["success"]),
                "optimizer_iterations": result["iterations"],
                "optimizer_function_evaluations": result["function_evaluations"],
                "optimizer_gradient_evaluations": result["gradient_evaluations"],
                "objective_final": result["objective_final"],
                "gradient_norm_inf": result["gradient_norm"],
                "evaluated_state": evaluated_state,
            }
        )
        run_rows.append(metrics)
        final_positions_rows.extend(
            {
                "variant": "SciPy-BFGS",
                "seed": seed,
                "label": label,
                "x_y_up_sim": float(points[index, 0]),
                "y_y_up_sim": float(points[index, 1]),
                "optimizer_success": bool(result["success"]),
                "evaluated_state": evaluated_state,
            }
            for index, label in enumerate(vertice)
        )
        _save_history(seed_dir, seed, history, problem, vertice)
        pd.DataFrame([metrics]).to_csv(
            seed_dir / "bfgs_final_metrics.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(
            {
                "label": vertice,
                "x_y_up_sim": points[:, 0],
                "y_y_up_sim": points[:, 1],
            }
        ).to_csv(seed_dir / "bfgs_final_positions_y_up_sim.csv", index=False, encoding="utf-8-sig")

        result_json = {key: value for key, value in result.items() if key != "y_final"}
        result_json.update(
            {
                "seed": seed,
                "evaluated_state": evaluated_state,
                "accepted_states_including_initial": len(history),
            }
        )
        (seed_dir / "bfgs_run_summary.json").write_text(
            json.dumps(result_json, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        if seed in visualize_set:
            _plot_objective_history(seed_dir, history, seed)
            _plot_configuration_evolution(
                seed_dir, history, problem, distance_data, dni, calibration_labels, seed
            )
            _plot_metric_summary(seed_dir, metrics, seed)
            if make_gif:
                _save_evolution_gif(
                    seed_dir, history, problem, distance_data, dni, calibration_labels, seed
                )
            points_y_down = _flip_y_up_for_display(points, dni[anchor_label])
            prefix = f"SciPy_BFGS_seed{seed}_"
            wrong_dir = wrong_directions_nonflip(deepcopy(points), vertice, dni)
            visualize_error_map_official(
                deepcopy(points_y_down),
                vertice,
                dni,
                distance_data,
                wrong_dir,
                file_name=prefix,
                wait=False,
                output_dir=seed_dir,
                title=f"Constraint-error Visualization: SciPy BFGS (seed {seed})",
            )
            overlay = ground_truth_comparison(
                vertice,
                dni,
                data_sim,
                deepcopy(gt_lonlat),
                points_y_down[dni[anchor_label]],
                deepcopy(points_y_down),
                prefix,
                wait=False,
                eval_labels=test_labels,
                output_dir=seed_dir,
                title=f"Ground-truth Overlay: SciPy BFGS (seed {seed})",
            )
            if not np.isclose(
                overlay["rmse_km"], metrics["RMSE_test_km"], rtol=1e-6, atol=1e-6
            ):
                raise ValueError(
                    "Ground-truth overlay RMSE differs from the formal metric: "
                    f"{overlay['rmse_km']} != {metrics['RMSE_test_km']}"
                )

    runs = pd.DataFrame(run_rows)
    positions = pd.DataFrame(final_positions_rows)
    summary = _summary_table(runs)
    runs.to_csv(outdir / "bfgs_runs_by_seed.csv", index=False, encoding="utf-8-sig")
    positions.to_csv(
        outdir / "bfgs_final_positions_y_up_sim.csv", index=False, encoding="utf-8-sig"
    )
    summary.to_csv(outdir / "bfgs_summary.csv", index=False, encoding="utf-8-sig")
    config = {
        "method": "scipy.optimize.minimize(method='BFGS')",
        "scipy_version": scipy.__version__,
        "uses_limited_memory_bfgs": False,
        "analytic_gradient": True,
        "gtol": float(gtol),
        "gradient_norm_order": "infinity",
        "xrtol": 0.0,
        "maxiter": (
            DEFAULT_BFGS_MAXITER_PER_DIMENSION * problem.dimension
            if maxiter is None
            else int(maxiter)
        ),
        "maxiter_policy": "200 * optimization dimension when CLI --maxiter is omitted",
        "c1": 1e-4,
        "c2": 0.9,
        "seeds": [int(seed) for seed in seeds],
        "visualize_seeds": sorted(visualize_set),
        "alpha": float(alpha),
        "beta": float(beta),
        "w_dis": float(w_dis),
        "hpo_source": hpo_source,
        "effective_weights": {
            "distance": problem.weights.distance,
            "direction": problem.weights.direction,
            "repulsion": problem.weights.repulsion,
        },
        "calibration_labels": calibration_labels,
        "anchor_align_label": anchor_label,
        "test_labels": test_labels,
        "lcc_bounds": _as_mapping(
            get_lcc_bounds(), ("lon_min", "lon_max", "lat_min", "lat_max")
        ),
        "lcc_parameters": _as_mapping(
            get_lcc_parameters(), ("lat_1", "lat_2", "lon_0")
        ),
        "input_files": {
            "distance": FILE_PATHS["chen_data"],
            "direction": FILE_PATHS["directional_data"],
            "site_points": FILE_PATHS["ground_truth_path"],
        },
        "failure_count": int((runs["status"] != "ok").sum()),
    }
    (outdir / "bfgs_experiment_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"runs": runs, "summary": summary, "positions": positions, "config": config}


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="0", help="Comma-separated random seeds.")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--gtol", type=float, default=DEFAULT_BFGS_GTOL)
    parser.add_argument(
        "--maxiter",
        type=int,
        default=None,
        help="Optional safety cap. Omit it to use the full BFGS 200*d policy.",
    )
    parser.add_argument(
        "--visualize-seeds",
        default=None,
        help="Seeds receiving detailed plots; defaults to the first requested seed.",
    )
    parser.add_argument("--no-visualizations", action="store_true")
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument(
        "--hpo-outdir",
        default=None,
        help="BFGS HPO folder containing bfgs_hpo_selected_candidate.csv.",
    )
    parser.add_argument(
        "--allow-boundary-hpo",
        action="store_true",
        help="Diagnostic override; formal runs should expand a boundary-selected HPO grid.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    seeds = _parse_seeds(args.seeds)
    if args.no_visualizations:
        visualize_seeds = []
    elif args.visualize_seeds is None:
        visualize_seeds = [seeds[0]]
    else:
        visualize_seeds = _parse_seeds(args.visualize_seeds)
    unknown = sorted(set(visualize_seeds) - set(seeds))
    if unknown:
        raise ValueError(f"--visualize-seeds must be included in --seeds: {unknown}")
    if args.hpo_outdir:
        from run_paper_script.ch5_scipy_bfgs_hpo import load_selected_bfgs_hpo_params

        selected = load_selected_bfgs_hpo_params(
            args.hpo_outdir, allow_boundary=args.allow_boundary_hpo
        )
        alpha, beta, w_dis = selected["alpha"], selected["beta"], selected["w_dis"]
    else:
        alpha = PHYSICS_HPO_SELECTED_ALPHA
        beta = PHYSICS_HPO_SELECTED_BETA
        w_dis = PHYSICS_HPO_SELECTED_W_DIS
    result = run_experiment(
        seeds=seeds,
        outdir=args.outdir,
        gtol=args.gtol,
        maxiter=args.maxiter,
        visualize_seeds=visualize_seeds,
        make_gif=not args.no_gif,
        alpha=alpha,
        beta=beta,
        w_dis=w_dis,
        hpo_source=args.hpo_outdir,
    )
    print(result["runs"].to_string(index=False))
    print(f"[Saved] {Path(args.outdir)}")


if __name__ == "__main__":
    main()
