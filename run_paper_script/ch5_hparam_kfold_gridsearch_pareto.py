"""run_paper_script.ch5_hparam_kfold_gridsearch_pareto

Chapter 5 HPO for the current site-point setup.

Current validation design:
  - data/site_rmse_points.csv must contain exactly 3 use_role=anchor sites.
  - Leave-one-anchor-out validation fixes 2 anchors and evaluates RMSE on the
    held-out anchor.
  - Pareto objectives are E_distance_stress, E_direction_vr, and
    RMSE_anchor_LOO.
  - After selecting a Pareto solution, the script reruns the model with all
    3 anchors fixed and reports final RMSE only on the 8 use_role=test sites.

Usage
-----
Run from the physics_simulation project root.

Small smoke test:
python -m run_paper_script.paper_run ch5-hparam-kfold --seeds 0 --alpha-min 0 --alpha-max 0 --alpha-step 1 --beta-min 0 --beta-max 0 --beta-step 1 --outdir outputs/ch5_hparam_anchor_loo_smoke

Example 3x3 grid:
python -m run_paper_script.paper_run ch5-hparam-kfold --seeds 0 --alpha-min -1 --alpha-max 1 --alpha-step 1 --beta-min -1 --beta-max 1 --beta-step 1 --outdir outputs/ch5_hparam_anchor_loo_grid_3x3_seed0

Larger run example:
python -m run_paper_script.paper_run ch5-hparam-kfold --seeds 0,1,2 --alpha-min -1 --alpha-max 2 --alpha-step 0.5 --beta-min -1 --beta-max 2 --beta-step 0.5 --outdir outputs/ch5_hparam_anchor_loo_grid_full

Export LOO fold review plots after HPO:
python -m run_paper_script.paper_run ch5-hparam-kfold --seeds 0 --alpha-min 0 --alpha-max 0 --alpha-step 1 --beta-min 0 --beta-max 0 --beta-step 1 --outdir outputs/ch5_hparam_anchor_loo_smoke --export-loo-review
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from library.anchor_frame import px_list_to_km_list
from library.config import (
    FILE_PATHS,
    DIRECTIONAL_FORCE_MAGNITUDE_BASE,
    OUTPUT_DIR,
    REPULSION_STRENGTH_BASE,
    SPRING_STIFFNESS_BASE,
    refer_pos_sim as DEFAULT_REFER_POS_SIM,
    km2pix,
)
from library.data_io import (
    get_anchor_labels,
    get_test_site_labels,
    load_ini_data_from_csv,
    load_site_points,
    uploading_directional_data,
    uploading_ground_truth,
)
from library.geometry import get_lcc_bounds, get_lcc_parameters, lcc_transformation_with_anchor
from library.initialization import generate_CHEN_initial_positions
from library.metrics import calculate_kruskals_stress, direction_violation_rate
from library.physics import main_physics_simulation
from library.units import data_Li2sim, pos_matrix_sim2km


_MPLCONFIGDIR = Path(OUTPUT_DIR) / ".matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

def _get_plt():
    """Load matplotlib only when an HPO figure is actually requested."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


@dataclass
class FoldSpec:
    fold_id: int
    train_labels: List[str]
    train_lonlat: List[Tuple[float, float]]
    heldout_label: str
    train_anchor_label: str


def _parse_seed_list(raw: str) -> List[int]:
    seeds = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not seeds:
        raise ValueError("--seeds cannot be empty")
    return seeds


def _make_alpha_beta_grid(
    alpha_min: float,
    alpha_max: float,
    alpha_step: float,
    beta_min: float,
    beta_max: float,
    beta_step: float,
) -> tuple[np.ndarray, np.ndarray]:
    alphas = np.arange(alpha_min, alpha_max + 1e-12, alpha_step, dtype=float)
    betas = np.arange(beta_min, beta_max + 1e-12, beta_step, dtype=float)
    return alphas, betas


def _load_site_lonlat_by_label(dni: Dict[str, int]) -> Dict[str, Tuple[float, float]]:
    out: Dict[str, Tuple[float, float]] = {}
    for row in load_site_points():
        name = row["name"]
        if name not in dni:
            raise ValueError(f"Site point not found in dni: {name}")
        out[name] = (float(row["lon"]), float(row["lat"]))
    return out


def _load_anchor_and_test_inputs(
    vertice: Sequence[str],
    dni: Dict[str, int],
) -> tuple[List[str], List[Tuple[float, float]], List[str], List[Tuple[float, float]]]:
    site_lonlat = _load_site_lonlat_by_label(dni)
    anchor_labels = get_anchor_labels()
    test_labels = get_test_site_labels()
    if len(anchor_labels) != 3:
        raise ValueError(f"Expected exactly 3 anchor sites for leave-one-anchor-out HPO, got {anchor_labels}")
    if len(test_labels) != 8:
        raise ValueError(f"Expected exactly 8 test sites for final RMSE, got {test_labels}")
    missing = [label for label in anchor_labels + test_labels if label not in site_lonlat]
    if missing:
        raise ValueError(f"Missing lon/lat for site labels: {missing}")
    return (
        anchor_labels,
        [site_lonlat[label] for label in anchor_labels],
        test_labels,
        [site_lonlat[label] for label in test_labels],
    )


def _build_anchor_loo_folds(
    anchor_labels: Sequence[str],
    anchor_lonlat: Sequence[Tuple[float, float]],
) -> List[FoldSpec]:
    if len(anchor_labels) != 3:
        raise ValueError(f"LOO HPO requires exactly 3 anchors, got {list(anchor_labels)}")
    folds: List[FoldSpec] = []
    for heldout_idx, heldout_label in enumerate(anchor_labels):
        train_idx = [i for i in range(len(anchor_labels)) if i != heldout_idx]
        train_labels = [anchor_labels[i] for i in train_idx]
        folds.append(
            FoldSpec(
                fold_id=heldout_idx,
                train_labels=train_labels,
                train_lonlat=[anchor_lonlat[i] for i in train_idx],
                heldout_label=heldout_label,
                train_anchor_label=train_labels[0],
            )
        )
    return folds


def _build_gt_lonlat_full(
    dni: Dict[str, int],
    labels: Sequence[str],
    lonlat: Sequence[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    gt_lonlat_full: List[Tuple[float, float]] = [(0.0, 0.0) for _ in range(len(dni))]
    for label, xy in zip(labels, lonlat):
        if label not in dni:
            raise KeyError(f"Site label not found in dni: {label}")
        gt_lonlat_full[dni[label]] = (float(xy[0]), float(xy[1]))
    return gt_lonlat_full


def _rmse_labels_km(
    *,
    pos_y_up_sim: Sequence[Sequence[float]],
    dni: Dict[str, int],
    refer_pos_sim: Sequence[float],
    gt_labels: Sequence[str],
    gt_lonlat: Sequence[Tuple[float, float]],
    eval_labels: Sequence[str],
    anchor_label_for_frame: str,
) -> float:
    pred_km = px_list_to_km_list(pos_y_up_sim, tuple(refer_pos_sim), km2pix)
    gt_lonlat_full = _build_gt_lonlat_full(dni, gt_labels, gt_lonlat)
    gt_km = lcc_transformation_with_anchor(dni, gt_lonlat_full, anchor_label=anchor_label_for_frame)
    return _euclidean_rmse_km(pred_km=pred_km, gt_km=gt_km, eval_labels=eval_labels, dni=dni)


def _euclidean_rmse_km(
    *,
    pred_km: Sequence[Sequence[float]],
    gt_km: Sequence[Sequence[float]],
    eval_labels: Sequence[str],
    dni: Dict[str, int],
) -> float:
    se: List[float] = []
    for label in eval_labels:
        idx = dni[label]
        gx, gy = gt_km[idx]
        if gx is None or gy is None:
            raise ValueError(f"Ground truth missing for RMSE label: {label}")
        px, py = pred_km[idx]
        se.append((float(px) - float(gx)) ** 2 + (float(py) - float(gy)) ** 2)
    return float(math.sqrt(sum(se) / len(se))) if se else float("nan")


def _non_dominated_mask(points: np.ndarray) -> np.ndarray:
    mask = np.ones(points.shape[0], dtype=bool)
    for i in range(points.shape[0]):
        if not np.isfinite(points[i]).all():
            mask[i] = False
            continue
        for j in range(points.shape[0]):
            if i == j or not np.isfinite(points[j]).all():
                continue
            if np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                mask[i] = False
                break
    return mask


def _weights_from_alpha_beta(
    alpha: float,
    beta: float,
    w_dis: float,
    base_spring_stiffness: float,
    base_directional_force: float,
    base_repulsion_strength: float,
) -> tuple[float, float, float, float, float]:
    w_dir = float(w_dis * math.pow(10, alpha))
    w_reg = float(w_dis * math.pow(10, beta))
    spring_stiffness = float(base_spring_stiffness * w_dis)
    directional_force = float(base_directional_force * w_dir)
    repulsion_strength = float(base_repulsion_strength * w_reg)
    return w_dir, w_reg, spring_stiffness, directional_force, repulsion_strength


def _run_physics_eval(
    *,
    seed: int,
    fixed_labels: Sequence[str],
    fixed_lonlat: Sequence[Tuple[float, float]],
    eval_labels: Sequence[str],
    rmse_gt_labels: Sequence[str],
    rmse_gt_lonlat: Sequence[Tuple[float, float]],
    anchor_label_for_frame: str,
    spring_stiffness: float,
    repulsion_strength: float,
    directional_force_magnitude: float,
    refer_pos_sim: Sequence[float],
) -> tuple[Dict[str, float], np.ndarray, List[str], Dict[str, int]]:
    np.random.seed(seed)
    directional_data = uploading_directional_data()
    _graph, _vertice0, _dni0, _edges0, data_li = load_ini_data_from_csv(FILE_PATHS)
    data_sim = data_Li2sim(data_li)

    vertice, dni, _data_li_again, pos_init, fixed_positions_list = generate_CHEN_initial_positions(
        list(refer_pos_sim),
        list(fixed_labels),
        list(fixed_lonlat),
        anchor_label=anchor_label_for_frame,
    )

    wrong_direction_lists, stress_history, _pos_history, pos_final_y_up = main_physics_simulation(
        vertice,
        dni,
        data_sim,
        pos_init,
        directional_data,
        fixed_positions_list,
        spring_stiffness,
        repulsion_strength,
        directional_force_magnitude,
        plot=False,
    )

    pos_final_y_up = np.asarray([(float(p[0]), float(p[1])) for p in pos_final_y_up], dtype=float)
    pos_final_km = pos_matrix_sim2km(pos_final_y_up.tolist())
    e_distance = float(calculate_kruskals_stress(dni, pos_final_km, data_sim))
    e_direction = float(direction_violation_rate(pos_final_y_up, directional_data, dni))
    rmse = _rmse_labels_km(
        pos_y_up_sim=pos_final_y_up,
        dni=dni,
        refer_pos_sim=refer_pos_sim,
        gt_labels=rmse_gt_labels,
        gt_lonlat=rmse_gt_lonlat,
        eval_labels=eval_labels,
        anchor_label_for_frame=anchor_label_for_frame,
    )

    metrics = {
        "E_distance_stress": e_distance,
        "E_direction_vr": e_direction,
        "RMSE_km": rmse,
        "wrong_dir_count": float(len(wrong_direction_lists)),
        "last_raw_stress_trace": float(stress_history[-1]) if len(stress_history) > 0 else float("nan"),
    }
    return metrics, pos_final_y_up, vertice, dni


def _selected_grid_row(df_grid: pd.DataFrame, selected: pd.Series | None) -> pd.Series | None:
    if selected is None:
        return None
    match = df_grid[
        np.isclose(df_grid["alpha"].to_numpy(float), float(selected["alpha"]))
        & np.isclose(df_grid["beta"].to_numpy(float), float(selected["beta"]))
    ]
    if len(match) != 1:
        raise ValueError("Selected alpha/beta does not identify exactly one HPO grid point.")
    return match.iloc[0]


def _plot_heatmap(
    df_grid: pd.DataFrame,
    value_col: str,
    out_png: Path,
    title: str,
    selected: pd.Series | None = None,
) -> None:
    plt = _get_plt()
    pivot = df_grid.pivot(index="beta", columns="alpha", values=value_col).sort_index(ascending=True)
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    im = ax.imshow(pivot.values, origin="lower", aspect="auto")
    fig.colorbar(im, ax=ax).set_label(value_col)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{x:g}" for x in pivot.columns], rotation=45)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{y:g}" for y in pivot.index])
    ax.set_xlabel("alpha = log10(w_dir / w_dis)")
    ax.set_ylabel("beta = log10(w_reg / w_dis)")
    ax.set_title(title)
    if selected is not None:
        alpha, beta = float(selected["alpha"]), float(selected["beta"])
        if alpha not in pivot.columns or beta not in pivot.index:
            raise ValueError("Selected alpha/beta is not present in the heatmap grid.")
        ax.scatter(
            [pivot.columns.get_loc(alpha)], [pivot.index.get_loc(beta)], marker="*", s=220,
            c="#d62728", edgecolors="black", linewidths=0.8,
            label=f"Selected (alpha={alpha:g}, beta={beta:g})",
            zorder=5,
        )
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _plot_pareto_3d(
    df_grid: pd.DataFrame,
    pareto_mask: np.ndarray,
    out_png: Path,
    selected: pd.Series | None = None,
) -> None:
    plt = _get_plt()
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    x = df_grid["E_distance_stress_mean"].to_numpy(float)
    y = df_grid["E_direction_vr_mean"].to_numpy(float)
    z = df_grid["RMSE_anchor_LOO_mean_km"].to_numpy(float)
    fig = plt.figure(figsize=(8.8, 7.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x[~pareto_mask], y[~pareto_mask], z[~pareto_mask], alpha=0.45, s=20, label="All grid points")
    ax.scatter(x[pareto_mask], y[pareto_mask], z[pareto_mask], s=40, label="Pareto front")
    selected_row = _selected_grid_row(df_grid, selected)
    if selected_row is not None:
        ax.scatter(
            [selected_row["E_distance_stress_mean"]], [selected_row["E_direction_vr_mean"]],
            [selected_row["RMSE_anchor_LOO_mean_km"]], marker="*", s=220, c="#d62728",
            edgecolors="black", linewidths=0.8,
            label=f"Selected (alpha={selected_row['alpha']:g}, beta={selected_row['beta']:g})",
        )
    ax.set_xlabel("E_distance (Kruskal stress)")
    ax.set_ylabel("E_direction (violation rate)")
    ax.set_zlabel("RMSE_anchor_LOO (km)")
    ax.set_title("3D Pareto Front")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _plot_pareto_2d_projections(
    df_grid: pd.DataFrame,
    pareto_mask: np.ndarray,
    out_png: Path,
    selected: pd.Series | None = None,
) -> None:
    plt = _get_plt()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    cols = [
        ("E_distance_stress_mean", "E_direction_vr_mean"),
        ("E_distance_stress_mean", "RMSE_anchor_LOO_mean_km"),
        ("E_direction_vr_mean", "RMSE_anchor_LOO_mean_km"),
    ]
    selected_row = _selected_grid_row(df_grid, selected)
    for ax, (cx, cy) in zip(axes, cols):
        ax.scatter(df_grid.loc[~pareto_mask, cx], df_grid.loc[~pareto_mask, cy], alpha=0.45, s=20)
        ax.scatter(df_grid.loc[pareto_mask, cx], df_grid.loc[pareto_mask, cy], s=36)
        if selected_row is not None:
            ax.scatter(
                [selected_row[cx]], [selected_row[cy]], marker="*", s=180, c="#d62728",
                edgecolors="black", linewidths=0.8, zorder=5,
            )
        ax.set_xlabel(cx)
        ax.set_ylabel(cy)
        ax.grid(alpha=0.25)
    if selected_row is not None:
        axes[0].scatter([], [], marker="*", s=150, c="#d62728", edgecolors="black", label="Selected")
        axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("Pareto Front 2D Projections")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _select_one_se_balanced_candidate(
    df_pareto: pd.DataFrame,
    objective_cols: Sequence[str],
) -> tuple[pd.Series, dict]:
    if df_pareto.empty:
        raise ValueError("No Pareto solutions found.")

    min_rmse_idx = df_pareto["RMSE_anchor_LOO_mean_km"].idxmin()
    min_rmse_row = df_pareto.loc[min_rmse_idx]
    n_folds = max(int(min_rmse_row.get("n_folds", 1)), 1)
    min_rmse = float(min_rmse_row["RMSE_anchor_LOO_mean_km"])
    min_se = float(min_rmse_row["RMSE_anchor_LOO_std_km"]) / math.sqrt(n_folds)
    threshold = min_rmse + min_se

    candidates = df_pareto[df_pareto["RMSE_anchor_LOO_mean_km"] <= threshold].copy()
    if candidates.empty:
        candidates = df_pareto.loc[[min_rmse_idx]].copy()

    mins = df_pareto[list(objective_cols)].min()
    ranges = (df_pareto[list(objective_cols)].max() - mins).replace(0, 1.0)
    standardized = (candidates[list(objective_cols)] - mins) / ranges
    candidates["one_se_balanced_score"] = np.sqrt((standardized**2).sum(axis=1))
    selected_idx = candidates["one_se_balanced_score"].idxmin()
    selected = candidates.loc[selected_idx]
    selection_meta = {
        "selection_rule": "pareto_one_se_balanced",
        "one_se_reference_alpha": float(min_rmse_row["alpha"]),
        "one_se_reference_beta": float(min_rmse_row["beta"]),
        "one_se_reference_rmse_anchor_loo_mean_km": min_rmse,
        "one_se_reference_rmse_anchor_loo_std_km": float(min_rmse_row["RMSE_anchor_LOO_std_km"]),
        "one_se_reference_n_folds": n_folds,
        "one_se_min_se_km": min_se,
        "one_se_threshold_km": threshold,
        "one_se_candidate_count": int(len(candidates)),
        "balanced_objectives": list(objective_cols),
        "balanced_score": float(selected["one_se_balanced_score"]),
    }
    return selected, selection_meta


def _run_final_selected_model(
    *,
    selected: pd.Series,
    anchor_labels: Sequence[str],
    anchor_lonlat: Sequence[Tuple[float, float]],
    test_labels: Sequence[str],
    test_lonlat: Sequence[Tuple[float, float]],
    seeds: Sequence[int],
    w_dis: float,
    base_spring_stiffness: float,
    base_directional_force: float,
    base_repulsion_strength: float,
    refer_pos_sim: Sequence[float],
    outdir: Path,
    selection_rule: str = "pareto_one_se_balanced",
    selection_meta: dict | None = None,
) -> None:
    alpha = float(selected["alpha"])
    beta = float(selected["beta"])
    _w_dir, _w_reg, spring, directional_force, repulsion = _weights_from_alpha_beta(
        alpha, beta, w_dis, base_spring_stiffness, base_directional_force, base_repulsion_strength
    )
    anchor_label_for_frame = anchor_labels[0]
    rmse_gt_labels = list(anchor_labels) + list(test_labels)
    rmse_gt_lonlat = list(anchor_lonlat) + list(test_lonlat)

    final_rows: List[dict] = []
    best_seed = None
    best_pos = None
    best_labels = None
    best_rmse = float("inf")

    for seed in seeds:
        metrics, pos_final, vertice, _dni = _run_physics_eval(
            seed=int(seed),
            fixed_labels=anchor_labels,
            fixed_lonlat=anchor_lonlat,
            eval_labels=test_labels,
            rmse_gt_labels=rmse_gt_labels,
            rmse_gt_lonlat=rmse_gt_lonlat,
            anchor_label_for_frame=anchor_label_for_frame,
            spring_stiffness=spring,
            repulsion_strength=repulsion,
            directional_force_magnitude=directional_force,
            refer_pos_sim=refer_pos_sim,
        )
        row = {
            "selection_rule": selection_rule,
            "alpha": alpha,
            "beta": beta,
            "seed": int(seed),
            "E_distance_stress": metrics["E_distance_stress"],
            "E_direction_vr": metrics["E_direction_vr"],
            "RMSE_final_test_km": metrics["RMSE_km"],
        }
        final_rows.append(row)
        if metrics["RMSE_km"] < best_rmse:
            best_rmse = metrics["RMSE_km"]
            best_seed = int(seed)
            best_pos = pos_final
            best_labels = vertice

    df_final = pd.DataFrame(final_rows)
    df_final.to_csv(outdir / "selected_final_runs_by_seed.csv", index=False, encoding="utf-8-sig")
    summary = {
        "selection_rule": selection_rule,
        "alpha": alpha,
        "beta": beta,
        "n_seeds": int(len(seeds)),
        "anchor_labels": list(anchor_labels),
        "test_labels": list(test_labels),
        "RMSE_final_test_mean_km": float(df_final["RMSE_final_test_km"].mean()),
        "RMSE_final_test_std_km": float(df_final["RMSE_final_test_km"].std(ddof=1)),
        "E_distance_stress_mean": float(df_final["E_distance_stress"].mean()),
        "E_direction_vr_mean": float(df_final["E_direction_vr"].mean()),
        "best_seed_by_final_test_rmse": best_seed,
    }
    if selection_meta:
        summary.update(selection_meta)
    (outdir / "selected_final_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if best_pos is not None and best_labels is not None:
        pd.DataFrame(
            {
                "label": list(best_labels),
                "x_y_up_sim": best_pos[:, 0],
                "y_y_up_sim": best_pos[:, 1],
            }
        ).to_csv(outdir / "selected_final_best_positions_y_up_sim.csv", index=False, encoding="utf-8-sig")


def run_anchor_loo_gridsearch_pareto(
    *,
    seeds: Sequence[int],
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
    refer_pos_sim: Sequence[float],
    outdir: str | Path | None,
    export_loo_review: bool = False,
    overwrite: bool = False,
) -> Dict[str, object]:
    _graph, vertice, dni, _edges, _data = load_ini_data_from_csv(FILE_PATHS)
    gt_lonlat_all = uploading_ground_truth(vertice, dni)
    anchor_labels, anchor_lonlat, test_labels, test_lonlat = _load_anchor_and_test_inputs(vertice, dni)
    folds = _build_anchor_loo_folds(anchor_labels, anchor_lonlat)
    alphas, betas = _make_alpha_beta_grid(alpha_min, alpha_max, alpha_step, beta_min, beta_max, beta_step)

    outdir_path = Path(outdir) if outdir else (Path(OUTPUT_DIR) / "ch5_hparam_anchor_loo_gridsearch")
    if outdir_path.exists() and any(outdir_path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"HPO outdir already exists and is not empty: {outdir_path}. "
            "Choose a new --outdir or pass --overwrite intentionally."
        )
    outdir_path.mkdir(parents=True, exist_ok=True)

    run_rows: List[dict] = []
    fold_rows: List[dict] = []
    grid_rows: List[dict] = []
    total_combo = len(alphas) * len(betas)
    combo_idx = 0

    for alpha in alphas:
        for beta in betas:
            combo_idx += 1
            w_dir, w_reg, spring, directional_force, repulsion = _weights_from_alpha_beta(
                float(alpha),
                float(beta),
                w_dis,
                base_spring_stiffness,
                base_directional_force,
                base_repulsion_strength,
            )
            print(
                f"[{combo_idx}/{total_combo}] alpha={alpha:.3f}, beta={beta:.3f} | "
                f"spring={spring:.3g}, dir={directional_force:.3g}, rep={repulsion:.3g}"
            )

            combo_fold_metrics: List[dict] = []
            for fold in folds:
                seed_metrics: List[dict] = []
                for seed in seeds:
                    try:
                        metrics, _pos_final, _vertice, _dni = _run_physics_eval(
                            seed=int(seed),
                            fixed_labels=fold.train_labels,
                            fixed_lonlat=fold.train_lonlat,
                            eval_labels=[fold.heldout_label],
                            rmse_gt_labels=anchor_labels,
                            rmse_gt_lonlat=anchor_lonlat,
                            anchor_label_for_frame=fold.train_anchor_label,
                            spring_stiffness=spring,
                            repulsion_strength=repulsion,
                            directional_force_magnitude=directional_force,
                            refer_pos_sim=refer_pos_sim,
                        )
                    except Exception as exc:
                        print(
                            f"  [WARN] alpha={alpha}, beta={beta}, fold={fold.fold_id}, seed={seed} failed: {exc}"
                        )
                        metrics = {
                            "E_distance_stress": float("nan"),
                            "E_direction_vr": float("nan"),
                            "RMSE_km": float("nan"),
                            "wrong_dir_count": float("nan"),
                            "last_raw_stress_trace": float("nan"),
                        }
                    seed_metrics.append(metrics)
                    run_rows.append(
                        {
                            "alpha": float(alpha),
                            "beta": float(beta),
                            "w_dis": float(w_dis),
                            "w_dir": float(w_dir),
                            "w_reg": float(w_reg),
                            "fold_id": int(fold.fold_id),
                            "train_labels": "|".join(fold.train_labels),
                            "train_anchor_label": fold.train_anchor_label,
                            "heldout_label": fold.heldout_label,
                            "seed": int(seed),
                            "E_distance_stress": metrics["E_distance_stress"],
                            "E_direction_vr": metrics["E_direction_vr"],
                            "RMSE_anchor_LOO_km": metrics["RMSE_km"],
                            "wrong_dir_count": metrics["wrong_dir_count"],
                            "last_raw_stress_trace": metrics["last_raw_stress_trace"],
                        }
                    )

                df_seed = pd.DataFrame(seed_metrics)
                n_failed_seeds = int(df_seed["RMSE_km"].isna().sum())
                fold_summary = {
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "w_dis": float(w_dis),
                    "w_dir": float(w_dir),
                    "w_reg": float(w_reg),
                    "fold_id": int(fold.fold_id),
                    "train_labels": "|".join(fold.train_labels),
                    "train_anchor_label": fold.train_anchor_label,
                    "heldout_label": fold.heldout_label,
                    "n_seeds": int(len(seeds)),
                    "n_failed_seeds": n_failed_seeds,
                    "E_distance_stress_mean": float(df_seed["E_distance_stress"].mean()),
                    "E_distance_stress_std": float(df_seed["E_distance_stress"].std(ddof=1)),
                    "E_direction_vr_mean": float(df_seed["E_direction_vr"].mean()),
                    "E_direction_vr_std": float(df_seed["E_direction_vr"].std(ddof=1)),
                    "RMSE_anchor_LOO_mean_km": float(df_seed["RMSE_km"].mean()),
                    "RMSE_anchor_LOO_std_km": float(df_seed["RMSE_km"].std(ddof=1)),
                }
                fold_rows.append(fold_summary)
                combo_fold_metrics.append(fold_summary)

            df_folds_for_combo = pd.DataFrame(combo_fold_metrics)
            grid_rows.append(
                {
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "w_dis": float(w_dis),
                    "w_dir": float(w_dir),
                    "w_reg": float(w_reg),
                    "spring_stiffness": spring,
                    "directional_force": directional_force,
                    "repulsion_strength": repulsion,
                    "n_folds": int(len(combo_fold_metrics)),
                    "n_seeds_per_fold": int(len(seeds)),
                    "n_failed_runs": int(df_folds_for_combo["n_failed_seeds"].sum()),
                    "E_distance_stress_mean": float(df_folds_for_combo["E_distance_stress_mean"].mean()),
                    "E_distance_stress_std": float(df_folds_for_combo["E_distance_stress_mean"].std(ddof=1)),
                    "E_direction_vr_mean": float(df_folds_for_combo["E_direction_vr_mean"].mean()),
                    "E_direction_vr_std": float(df_folds_for_combo["E_direction_vr_mean"].std(ddof=1)),
                    "RMSE_anchor_LOO_mean_km": float(df_folds_for_combo["RMSE_anchor_LOO_mean_km"].mean()),
                    "RMSE_anchor_LOO_std_km": float(df_folds_for_combo["RMSE_anchor_LOO_mean_km"].std(ddof=1)),
                }
            )

    df_runs = pd.DataFrame(run_rows)
    df_folds = pd.DataFrame(fold_rows)
    df_grid = pd.DataFrame(grid_rows).sort_values(["alpha", "beta"]).reset_index(drop=True)
    objective_cols = ["E_distance_stress_mean", "E_direction_vr_mean", "RMSE_anchor_LOO_mean_km"]
    pareto_mask = _non_dominated_mask(df_grid[objective_cols].to_numpy(float))
    df_grid["is_pareto"] = pareto_mask
    df_pareto = df_grid[df_grid["is_pareto"]].copy()

    df_runs.to_csv(outdir_path / "grid_runs_by_seed.csv", index=False, encoding="utf-8-sig")
    df_folds.to_csv(outdir_path / "grid_folds_mean_std.csv", index=False, encoding="utf-8-sig")
    df_grid.to_csv(outdir_path / "grid_summary_cv.csv", index=False, encoding="utf-8-sig")
    df_pareto.to_csv(outdir_path / "pareto_front_3d.csv", index=False, encoding="utf-8-sig")

    cfg = {
        "validation": "three_anchor_leave_one_anchor_out",
        "objectives": objective_cols,
        "default_selection_rule": "pareto_one_se_balanced",
        "final_rmse_labels": "use_role == test",
        "anchor_labels": list(anchor_labels),
        "test_labels": list(test_labels),
        "lcc_bounds": dict(
            zip(["lon_min", "lon_max", "lat_min", "lat_max"], map(float, get_lcc_bounds()))
        ),
        "lcc_parameters": dict(zip(["lat_1", "lat_2", "lon_0"], map(float, get_lcc_parameters()))),
        "lcc_standard_parallel_rule": "lat_1=lat_min+(lat_max-lat_min)/6; lat_2=lat_max-(lat_max-lat_min)/6",
        "lcc_bounds_source": FILE_PATHS["ground_truth_path"],
        "seeds": list(map(int, seeds)),
        "alpha_range": [alpha_min, alpha_max, alpha_step],
        "beta_range": [beta_min, beta_max, beta_step],
        "alpha_beta_scale": "base-10: w_dir=w_dis*10^alpha, w_reg=w_dis*10^beta",
        "w_dis": float(w_dis),
        "base_spring_stiffness": float(base_spring_stiffness),
        "base_directional_force": float(base_directional_force),
        "base_repulsion_strength": float(base_repulsion_strength),
        "refer_pos_sim": list(map(float, refer_pos_sim)),
    }
    (outdir_path / "gridsearch_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _plot_heatmap(
        df_grid,
        "RMSE_anchor_LOO_mean_km",
        outdir_path / "heatmap_rmse_anchor_loo.png",
        "Grid Heatmap: RMSE_anchor_LOO (km)",
    )
    _plot_heatmap(
        df_grid,
        "E_distance_stress_mean",
        outdir_path / "heatmap_kruskal_stress.png",
        "Grid Heatmap: E_distance (Kruskal stress)",
    )
    _plot_heatmap(
        df_grid,
        "E_direction_vr_mean",
        outdir_path / "heatmap_direction_violation.png",
        "Grid Heatmap: E_direction (Violation rate)",
    )
    _plot_pareto_3d(df_grid, pareto_mask, outdir_path / "pareto_front_3d.png")
    _plot_pareto_2d_projections(df_grid, pareto_mask, outdir_path / "pareto_front_2d_projections.png")

    selected, selection_meta = _select_one_se_balanced_candidate(df_pareto, objective_cols)
    _run_final_selected_model(
        selected=selected,
        anchor_labels=anchor_labels,
        anchor_lonlat=anchor_lonlat,
        test_labels=test_labels,
        test_lonlat=test_lonlat,
        seeds=seeds,
        w_dis=w_dis,
        base_spring_stiffness=base_spring_stiffness,
        base_directional_force=base_directional_force,
        base_repulsion_strength=base_repulsion_strength,
        refer_pos_sim=refer_pos_sim,
        outdir=outdir_path,
        selection_rule=selection_meta["selection_rule"],
        selection_meta=selection_meta,
    )

    print("\n=== Default Pareto candidate by one-SE balanced rule ===")
    print(
        f"alpha={selected['alpha']}, beta={selected['beta']}, "
        f"RMSE_anchor_LOO={selected['RMSE_anchor_LOO_mean_km']:.4f}±{selected['RMSE_anchor_LOO_std_km']:.4f} km, "
        f"stress={selected['E_distance_stress_mean']:.4f}, vr={selected['E_direction_vr_mean']:.4f}, "
        f"one_se_threshold={selection_meta['one_se_threshold_km']:.4f} km"
    )
    print(f"Pareto solutions: {int(df_grid['is_pareto'].sum())}/{len(df_grid)}")
    print(f"Saved to: {outdir_path}")

    if export_loo_review:
        from scripts.export_hpo_loo_review import export_hpo_loo_review

        export_hpo_loo_review(
            hpo_outdir=outdir_path,
            alpha=float(selected["alpha"]),
            beta=float(selected["beta"]),
            seed=int(seeds[0]),
            w_dis=w_dis,
            base_spring_stiffness=base_spring_stiffness,
            base_directional_force=base_directional_force,
            base_repulsion_strength=base_repulsion_strength,
            refer_pos_sim=refer_pos_sim,
        )

    return {
        "df_runs": df_runs,
        "df_folds": df_folds,
        "df_grid": df_grid,
        "df_pareto": df_pareto,
        "outdir": outdir_path,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid-search alpha/beta with three-anchor leave-one-anchor-out validation and Pareto analysis"
    )
    parser.add_argument("--seeds", type=str, default="0,1,2", help="Comma-separated seeds, e.g. 0,1,2")
    parser.add_argument("--alpha-min", type=float, default=-1.0)
    parser.add_argument("--alpha-max", type=float, default=2.0)
    parser.add_argument("--alpha-step", type=float, default=0.5)
    parser.add_argument("--beta-min", type=float, default=-1.0)
    parser.add_argument("--beta-max", type=float, default=2.0)
    parser.add_argument("--beta-step", type=float, default=0.5)
    parser.add_argument("--w-dis", type=float, default=1.0)
    parser.add_argument("--base-spring", type=float, default=SPRING_STIFFNESS_BASE)
    parser.add_argument("--base-dir", type=float, default=DIRECTIONAL_FORCE_MAGNITUDE_BASE)
    parser.add_argument("--base-rep", type=float, default=REPULSION_STRENGTH_BASE)
    parser.add_argument("--outdir", type=str, default="")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty HPO outdir.",
    )
    parser.add_argument(
        "--export-loo-review",
        action="store_true",
        help="Export three LOO fold review CSV/PNG files after selecting HPO parameters.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    seeds = _parse_seed_list(args.seeds)
    run_anchor_loo_gridsearch_pareto(
        seeds=seeds,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        alpha_step=args.alpha_step,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        beta_step=args.beta_step,
        w_dis=args.w_dis,
        base_spring_stiffness=args.base_spring,
        base_directional_force=args.base_dir,
        base_repulsion_strength=args.base_rep,
        refer_pos_sim=DEFAULT_REFER_POS_SIM,
        outdir=args.outdir,
        export_loo_review=args.export_loo_review,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
