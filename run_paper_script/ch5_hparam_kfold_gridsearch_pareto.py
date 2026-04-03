from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from library.config import (
    OUTPUT_DIR,
    FILE_PATHS,
    SPRING_STIFFNESS_BASE,
    DIRECTIONAL_FORCE_MAGNITUDE_BASE,
    REPULSION_STRENGTH_BASE,
    refer_pos_sim as DEFAULT_REFER_POS_SIM,
    km2pix,
)
from library.data_io import load_ini_data_from_csv, uploading_directional_data
from library.initialization import generate_CHEN_initial_positions
from library.physics import main_physics_simulation
from library.units import data_Li2sim, pos_matrix_sim2km
from library.metrics import calculate_kruskals_stress, direction_violation_rate
from library.anchor_frame import px_list_to_km_list
from library.geometry import lcc_transformation_with_anchor


# =========================================================
# User editable defaults (you can leave these empty and use --anchors-json)
# =========================================================
DEFAULT_FIXED_POINT_LABELS: List[str] = []
DEFAULT_FIXED_POINTS_LONLAT: List[Tuple[float, float]] = []
DEFAULT_ANCHOR_GROUPS: List[int] = []  # same order as labels, e.g. [0,0,1,1,2,...]


@dataclass
class FoldSpec:
    fold_id: int
    heldout_group: int
    train_labels: List[str]
    train_lonlat: List[Tuple[float, float]]
    test_labels: List[str]
    train_anchor_label: str


def _ensure_python_list_xy(xy_list: Sequence[Sequence[float]]) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for item in xy_list:
        if len(item) != 2:
            raise ValueError(f"Each lonlat must have length 2, got: {item}")
        out.append((float(item[0]), float(item[1])))
    return out


def _parse_seed_list(s: str) -> List[int]:
    vals = [x.strip() for x in s.split(",") if x.strip()]
    if not vals:
        raise ValueError("--seeds cannot be empty")
    return [int(v) for v in vals]


def _build_anchor_folds(
    fixed_point_labels: Sequence[str],
    fixed_points_lonlat: Sequence[Tuple[float, float]],
    anchor_groups: Sequence[int] | Dict[str, int],
) -> List[FoldSpec]:
    labels = list(fixed_point_labels)
    lonlats = _ensure_python_list_xy(fixed_points_lonlat)
    if len(labels) != len(lonlats):
        raise ValueError("fixed_point_labels and fixed_points_lonlat length mismatch")

    if isinstance(anchor_groups, dict):
        group_ids = [int(anchor_groups[label]) for label in labels]
    else:
        if len(anchor_groups) != len(labels):
            raise ValueError("anchor_groups length mismatch")
        group_ids = [int(g) for g in anchor_groups]

    unique_groups = sorted(set(group_ids))
    folds: List[FoldSpec] =  []
    for fold_id, g in enumerate(unique_groups):
        test_idx = [i for i, gid in enumerate(group_ids) if gid == g]
        train_idx = [i for i, gid in enumerate(group_ids) if gid != g]
        if len(test_idx) == 0:
            continue
        if len(train_idx) < 2:
            # 至少保留 2 個 training anchors，才能固定平移/旋轉自由度
            continue

        train_labels = [labels[i] for i in train_idx]
        train_lonlat = [lonlats[i] for i in train_idx]
        test_labels = [labels[i] for i in test_idx]

        # 動態選擇 training anchor（不再硬編碼鄯善）
        # 用原始輸入順序的第一個 training anchor，確保可重現。
        train_anchor_label = train_labels[0]

        folds.append(
            FoldSpec(
                fold_id=fold_id,
                heldout_group=int(g),
                train_labels=train_labels,
                train_lonlat=train_lonlat,
                test_labels=test_labels,
                train_anchor_label=train_anchor_label,
            )
        )

    if not folds:
        raise ValueError(
            "No valid folds found. Please check anchor_groups and ensure each fold leaves at least 2 training anchors."
        )
    return folds


def _build_gt_lonlat_full(
    dni: Dict[str, int],
    fixed_point_labels: Sequence[str],
    fixed_points_lonlat: Sequence[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    gt_lonlat_full: List[Tuple[float, float]] = [(0.0, 0.0) for _ in range(len(dni))]
    for label, lonlat in zip(fixed_point_labels, fixed_points_lonlat):
        if label not in dni:
            raise KeyError(f"Anchor '{label}' not found in dni")
        gt_lonlat_full[dni[label]] = (float(lonlat[0]), float(lonlat[1]))
    return gt_lonlat_full


def _rmse_holdout_anchors_km(
    pos_y_up_sim: Sequence[Sequence[float]],
    dni: Dict[str, int],
    refer_pos_sim: Sequence[float],
    all_fixed_point_labels: Sequence[str],
    all_fixed_points_lonlat: Sequence[Tuple[float, float]],
    heldout_labels: Sequence[str],
    anchor_label_for_frame: str,
) -> float:
    """RMSE(km) on held-out anchors only.

    Physical simulation 已固定  training anchors，因此不需 Procrustes；
    只要把模擬座標轉成以 training anchor 為原點的 km，和同一 frame 的 LCCGT 比較即可。
    """
    pred_km = px_list_to_km_list(pos_y_up_sim, tuple(refer_pos_sim), km2pix)
    gt_lonlat_full = _build_gt_lonlat_full(dni, all_fixed_point_labels, all_fixed_points_lonlat)
    gt_km = lcc_transformation_with_anchor(dni, gt_lonlat_full, anchor_label=anchor_label_for_frame)

    se: List[float] = []
    for label in heldout_labels:
        if label not in dni:
            continue
        idx = dni[label]
        gx, gy = gt_km[idx]
        if gx is None or gy is None:
            continue
        px, py = pred_km[idx]
        dx = float(px) - float(gx)
        dy = float(py) - float(gy)
        se.append(dx * dx + dy * dy)

    if not se:
        return float("nan")
    return float(math.sqrt(sum(se) / len(se)))


def _non_dominated_mask(points: np.ndarray) -> np.ndarray:
    """Minimization Pareto front mask. points shape=(N,3)."""
    n = points.shape[0]
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not np.isfinite(points[i]).all():
            mask[i] = False
            continue
        for j in range(n):
            if i == j:
                continue
            if not np.isfinite(points[j]).all():
                continue
            # j dominates i if all <= and at least one <
            if np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                mask[i] = False
                break
    return mask


def _make_alpha_beta_grid(a_min: float, a_max: float, a_step: float, b_min: float, b_max: float, b_step: float):
    # 加一點 epsilon 避免浮點漏掉終點
    alphas = np.arange(a_min, a_max + 1e-12, a_step, dtype=float)
    betas = np.arange(b_min, b_max + 1e-12, b_step, dtype=float)
    return alphas, betas


def _plot_heatmap(df_grid: pd.DataFrame, value_col: str, out_png: Path, title: str) -> None:
    pivot = df_grid.pivot(index="beta", columns="alpha", values=value_col).sort_index(ascending=True)

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    im = ax.imshow(pivot.values, origin="lower", aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(value_col)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{x:g}" for x in pivot.columns], rotation=45)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{y:g}" for y in pivot.index])
    ax.set_xlabel("alpha = log(w_dir / w_dis)")
    ax.set_ylabel("beta = log(w_reg / w_dis)")
    ax.set_title(title)

    # 標註最佳點（最小值）
    vals = pivot.values
    if np.isfinite(vals).any():
        iy, ix = np.unravel_index(np.nanargmin(vals), vals.shape)
        ax.scatter([ix], [iy], marker="x", s=120)
        ax.text(ix + 0.1, iy + 0.1, "best", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _plot_pareto_3d(df_grid: pd.DataFrame, pareto_mask: np.ndarray, out_png: Path) -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    x = df_grid["E_distance_stress_mean"].to_numpy(float)
    y = df_grid["E_direction_vr_mean"].to_numpy(float)
    z = df_grid["RMSE_anchor_mean_km"].to_numpy(float)

    fig = plt.figure(figsize=(8.8, 7.0))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(x[~pareto_mask], y[~pareto_mask], z[~pareto_mask], alpha=0.45, s=20, label="All grid points")
    ax.scatter(x[pareto_mask], y[pareto_mask], z[pareto_mask], s=40, label="Pareto front")

    ax.set_xlabel("E_distance (Kruskal stress)")
    ax.set_ylabel("E_direction (violation rate)")
    ax.set_zlabel("RMSE_anchor (km)")
    ax.set_title("3D Pareto Front (Non-dominated Solutions)")
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _plot_pareto_2d_projections(df_grid: pd.DataFrame, pareto_mask: np.ndarray, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    cols = [
        ("E_distance_stress_mean", "E_direction_vr_mean"),
        ("E_distance_stress_mean", "RMSE_anchor_mean_km"),
        ("E_direction_vr_mean", "RMSE_anchor_mean_km"),
    ]
    for ax, (cx, cy) in zip(axes, cols):
        ax.scatter(df_grid.loc[~pareto_mask, cx], df_grid.loc[~pareto_mask, cy], alpha=0.45, s=20)
        ax.scatter(df_grid.loc[pareto_mask, cx], df_grid.loc[pareto_mask, cy], s=36)
        ax.set_xlabel(cx)
        ax.set_ylabel(cy)
        ax.grid(alpha=0.25)
    fig.suptitle("Pareto Front 2D Projections")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _run_one_physics_eval(
    *,
    seed: int,
    fold: FoldSpec,
    spring_stiffness: float,
    repulsion_strength: float,
    directional_force_magnitude: float,
    all_fixed_point_labels: Sequence[str],
    all_fixed_points_lonlat: Sequence[Tuple[float, float]],
    refer_pos_sim: Sequence[float],
) -> Dict[str, float]:
    np.random.seed(seed)

    directional_data = uploading_directional_data()
    _graph, _vertice0, _dni0, _edges0, data_li = load_ini_data_from_csv(FILE_PATHS)
    data_sim = data_Li2sim(data_li)

    vertice, dni, data_li_again, pos_init, fixed_positions_list = generate_CHEN_initial_positions(
        list(refer_pos_sim),
        fold.train_labels,
        fold.train_lonlat,
        anchor_label=fold.train_anchor_label,
    )

    wrong_direction_lists, stress_history, pos_history, pos_final_y_up = main_physics_simulation(
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

    # (1) 距離殘差：Kruskal stress（以 km frame 計）
    pos_final_km = pos_matrix_sim2km(pos_final_y_up.tolist())
    e_distance = float(calculate_kruskals_stress(dni, pos_final_km, data_sim))

    # (2) 方向殘差：Violation Rate（y-up）
    e_direction = float(direction_violation_rate(pos_final_y_up, directional_data, dni))

    # (3) Held-out anchors RMSE (km)
    rmse_anchor = _rmse_holdout_anchors_km(
        pos_y_up_sim=pos_final_y_up,
        dni=dni,
        refer_pos_sim=refer_pos_sim,
        all_fixed_point_labels=all_fixed_point_labels,
        all_fixed_points_lonlat=all_fixed_points_lonlat,
        heldout_labels=fold.test_labels,
        anchor_label_for_frame=fold.train_anchor_label,
    )

    return {
        "E_distance_stress": e_distance,
        "E_direction_vr": e_direction,
        "RMSE_anchor_km": rmse_anchor,
        "wrong_dir_count": float(len(wrong_direction_lists)),
        "last_raw_stress_trace": float(stress_history[-1]) if len(stress_history) > 0 else float("nan"),
    }


def run_kfold_gridsearch_pareto(
    *,
    fixed_point_labels: Sequence[str],
    fixed_points_lonlat: Sequence[Tuple[float, float]],
    anchor_groups: Sequence[int] | Dict[str, int],
    seeds: Sequence[int],
    alpha_min: float = -1.0,
    alpha_max: float = 2.0,
    alpha_step: float = 0.5,
    beta_min: float = -1.0,
    beta_max: float = 2.0,
    beta_step: float = 0.5,
    w_dis: float = 1.0,
    base_spring_stiffness: float = SPRING_STIFFNESS_BASE,
    base_directional_force: float = DIRECTIONAL_FORCE_MAGNITUDE_BASE,
    base_repulsion_strength: float = REPULSION_STRENGTH_BASE,
    refer_pos_sim: Sequence[float] = DEFAULT_REFER_POS_SIM,
    outdir: str | Path | None = None,
) -> Dict[str, object]:
    """
    Grid search over alpha, beta where
      alpha = log(w_dir / w_dis)
      beta  = log(w_reg / w_dis)
    and evaluate by group-based anchor k-fold CV + multi-seed mean±std.
    """
    if len(fixed_point_labels) == 0:
        raise ValueError("fixed_point_labels is empty")
    if len(fixed_point_labels) != len(fixed_points_lonlat):
        raise ValueError("fixed_point_labels / fixed_points_lonlat length mismatch")
    if len(seeds) == 0:
        raise ValueError("seeds is empty")
    if base_repulsion_strength == 0:
        raise ValueError(
            "base_repulsion_strength is 0, so beta search would have no effect. "
            "Please set a non-zero base_repulsion_strength."
        )

    folds = _build_anchor_folds(fixed_point_labels, fixed_points_lonlat, anchor_groups)
    alphas, betas = _make_alpha_beta_grid(alpha_min, alpha_max, alpha_step, beta_min, beta_max, beta_step)

    outdir_path = Path(outdir) if outdir else (Path(OUTPUT_DIR) / "ch5_hparam_kfold_gridsearch")
    outdir_path.mkdir(parents=True, exist_ok=True)

    run_rows: List[dict] = []
    fold_rows: List[dict] = []
    grid_rows: List[dict] = []

    total_combo = len(alphas) * len(betas)
    combo_idx = 0

    for alpha in alphas:
        for beta in betas:
            combo_idx += 1

            w_dir = float(w_dis * math.pow(10, alpha))
            w_reg = float(w_dis * math.pow(10, beta))

            # 把 (w_dis, w_dir, w_reg) 映射到目前程式的三個物理超參數
            spring_stiffness = float(base_spring_stiffness * w_dis)
            directional_force = float(base_spring_stiffness * w_dir)
            repulsion_strength = float(base_spring_stiffness * w_reg)

            print(
                f"[{combo_idx}/{total_combo}] alpha={alpha:.3f}, beta={beta:.3f} | "
                f"spring={spring_stiffness:.3g}, dir={directional_force:.3g}, rep={repulsion_strength:.3g}"
            )

            combo_fold_metrics = []
            for fold in folds:
                one_fold_seed_metrics = []
                for seed in seeds:
                    try:
                        m = _run_one_physics_eval(
                            seed=int(seed),
                            fold=fold,
                            spring_stiffness=spring_stiffness,
                            repulsion_strength=repulsion_strength,
                            directional_force_magnitude=directional_force,
                            all_fixed_point_labels=fixed_point_labels,
                            all_fixed_points_lonlat=fixed_points_lonlat,
                            refer_pos_sim=refer_pos_sim,
                        )
                    except Exception as e:
                        m = {
                            "E_distance_stress": float("nan"),
                            "E_direction_vr": float("nan"),
                            "RMSE_anchor_km": float("nan"),
                            "wrong_dir_count": float("nan"),
                            "last_raw_stress_trace": float("nan"),
                        }
                        print(
                            f"  [WARN] alpha={alpha}, beta={beta}, fold={fold.fold_id}, seed={seed} failed: {e}"
                        )

                    run_rows.append(
                        {
                            "alpha": float(alpha),
                            "beta": float(beta),
                            "w_dis": float(w_dis),
                            "w_dir": float(w_dir),
                            "w_reg": float(w_reg),
                            "fold_id": int(fold.fold_id),
                            "heldout_group": int(fold.heldout_group),
                            "train_anchor_label": fold.train_anchor_label,
                            "test_labels": "|".join(fold.test_labels),
                            "seed": int(seed),
                            **m,
                        }
                    )
                    one_fold_seed_metrics.append(m)

                # fold-level mean ± std across seeds
                df_fold_seed = pd.DataFrame(one_fold_seed_metrics)
                fold_summary = {
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "w_dis": float(w_dis),
                    "w_dir": float(w_dir),
                    "w_reg": float(w_reg),
                    "fold_id": int(fold.fold_id),
                    "heldout_group": int(fold.heldout_group),
                    "train_anchor_label": fold.train_anchor_label,
                    "test_labels": "|".join(fold.test_labels),
                    "n_seeds": int(len(seeds)),
                    "E_distance_stress_mean": float(df_fold_seed["E_distance_stress"].mean()),
                    "E_distance_stress_std": float(df_fold_seed["E_distance_stress"].std(ddof=0)),
                    "E_direction_vr_mean": float(df_fold_seed["E_direction_vr"].mean()),
                    "E_direction_vr_std": float(df_fold_seed["E_direction_vr"].std(ddof=0)),
                    "RMSE_anchor_mean_km": float(df_fold_seed["RMSE_anchor_km"].mean()),
                    "RMSE_anchor_std_km": float(df_fold_seed["RMSE_anchor_km"].std(ddof=0)),
                }
                fold_rows.append(fold_summary)
                combo_fold_metrics.append(fold_summary)

            # combo-level CV mean ± std across folds（每 fold 先對 seeds 取平均，再跨 folds 聚合）
            df_combo_folds = pd.DataFrame(combo_fold_metrics)
            grid_rows.append(
                {
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "w_dis": float(w_dis),
                    "w_dir": float(w_dir),
                    "w_reg": float(w_reg),
                    "spring_stiffness": spring_stiffness,
                    "directional_force": directional_force,
                    "repulsion_strength": repulsion_strength,
                    "n_folds": int(len(combo_fold_metrics)),
                    "n_seeds_per_fold": int(len(seeds)),
                    "E_distance_stress_mean": float(df_combo_folds["E_distance_stress_mean"].mean()),
                    "E_distance_stress_std": float(df_combo_folds["E_distance_stress_mean"].std(ddof=0)),
                    "E_direction_vr_mean": float(df_combo_folds["E_direction_vr_mean"].mean()),
                    "E_direction_vr_std": float(df_combo_folds["E_direction_vr_mean"].std(ddof=0)),
                    "RMSE_anchor_mean_km": float(df_combo_folds["RMSE_anchor_mean_km"].mean()),
                    "RMSE_anchor_std_km": float(df_combo_folds["RMSE_anchor_mean_km"].std(ddof=0)),
                }
            )

    df_runs = pd.DataFrame(run_rows)
    df_folds = pd.DataFrame(fold_rows)
    df_grid = pd.DataFrame(grid_rows).sort_values(["alpha", "beta"]).reset_index(drop=True)

    # Pareto front (3 objectives)
    points = df_grid[["E_distance_stress_mean", "E_direction_vr_mean", "RMSE_anchor_mean_km"]].to_numpy(float)
    pareto_mask = _non_dominated_mask(points)
    df_grid["is_pareto"] = pareto_mask
    df_pareto = df_grid[df_grid["is_pareto"]].copy()

    # Save tabular outputs
    df_runs.to_csv(outdir_path / "grid_runs_by_seed.csv", index=False, encoding="utf-8-sig")
    df_folds.to_csv(outdir_path / "grid_folds_mean_std.csv", index=False, encoding="utf-8-sig")
    df_grid.to_csv(outdir_path / "grid_summary_cv.csv", index=False, encoding="utf-8-sig")
    df_pareto.to_csv(outdir_path / "pareto_front_3d.csv", index=False, encoding="utf-8-sig")

    # Save config for reproducibility
    cfg = {
        "fixed_point_labels": list(fixed_point_labels),
        "fixed_points_lonlat": [list(map(float, x)) for x in fixed_points_lonlat],
        "anchor_groups": anchor_groups if isinstance(anchor_groups, dict) else list(map(int, anchor_groups)),
        "seeds": list(map(int, seeds)),
        "alpha_range": [alpha_min, alpha_max, alpha_step],
        "beta_range": [beta_min, beta_max, beta_step],
        "w_dis": float(w_dis),
        "base_spring_stiffness": float(base_spring_stiffness),
        "base_directional_force": float(base_directional_force),
        "base_repulsion_strength": float(base_repulsion_strength),
        "refer_pos_sim": list(map(float, refer_pos_sim)),
    }
    (outdir_path / "gridsearch_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Plots
    _plot_heatmap(df_grid, "RMSE_anchor_mean_km", outdir_path / "heatmap_rmse_anchor.png", "Grid Heatmap: RMSE_anchor (km)")
    _plot_heatmap(df_grid, "E_distance_stress_mean", outdir_path / "heatmap_kruskal_stress.png", "Grid Heatmap: E_distance (Kruskal stress)")
    _plot_heatmap(df_grid, "E_direction_vr_mean", outdir_path / "heatmap_direction_violation.png", "Grid Heatmap: E_direction (Violation rate)")
    _plot_pareto_3d(df_grid, pareto_mask, outdir_path / "pareto_front_3d.png")
    _plot_pareto_2d_projections(df_grid, pareto_mask, outdir_path / "pareto_front_2d_projections.png")

    # Console summary (best by RMSE and pareto count)
    if len(df_grid) > 0:
        best_idx = df_grid["RMSE_anchor_mean_km"].idxmin()
        best = df_grid.loc[best_idx]
        print("\n=== Best by RMSE_anchor_mean_km ===")
        print(
            f"alpha={best['alpha']}, beta={best['beta']}, "
            f"RMSE={best['RMSE_anchor_mean_km']:.4f}±{best['RMSE_anchor_std_km']:.4f} km, "
            f"stress={best['E_distance_stress_mean']:.4f}, vr={best['E_direction_vr_mean']:.4f}"
        )
        print(f"Pareto solutions: {int(df_grid['is_pareto'].sum())}/{len(df_grid)}")
        print(f"Saved to: {outdir_path}")

    return {
        "df_runs": df_runs,
        "df_folds": df_folds,
        "df_grid": df_grid,
        "df_pareto": df_pareto,
        "outdir": outdir_path,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Grid-search loss weights (alpha,beta) with anchor-group k-fold CV and Pareto analysis"
    )
    p.add_argument("--anchors-json", type=str, default="", help="JSON file with labels/lonlat/groups")
    p.add_argument("--seeds", type=str, default="0,1,2", help="Comma-separated seeds, e.g. 0,1,2,3")
    p.add_argument("--alpha-min", type=float, default=-1.0)
    p.add_argument("--alpha-max", type=float, default=2.0)
    p.add_argument("--alpha-step", type=float, default=0.5)
    p.add_argument("--beta-min", type=float, default=-1.0)
    p.add_argument("--beta-max", type=float, default=2.0)
    p.add_argument("--beta-step", type=float, default=0.5)
    p.add_argument("--w-dis", type=float, default=1.0)
    p.add_argument("--base-spring", type=float, default=SPRING_STIFFNESS_BASE)
    p.add_argument("--base-dir", type=float, default=DIRECTIONAL_FORCE_MAGNITUDE_BASE)
    p.add_argument(
        "--base-rep",
        type=float,
        default=REPULSION_STRENGTH_BASE if REPULSION_STRENGTH_BASE != 0 else 500.0,
        help="Base repulsion strength for beta scaling. If config base is 0, set a non-zero value here.",
    )
    p.add_argument("--outdir", type=str, default="")
    return p.parse_args()


def _load_anchor_inputs_from_json(json_path: str):
    obj = json.loads(Path(json_path).read_text(encoding="utf-8"))
    labels = obj["fixed_point_labels"]
    lonlat = obj["fixed_points_lonlat"]
    groups = obj["anchor_groups"]
    return labels, lonlat, groups


def main() -> None:
    args = _parse_args()
    seeds = _parse_seed_list(args.seeds)

    if args.anchors_json:
        fixed_point_labels, fixed_points_lonlat, anchor_groups = _load_anchor_inputs_from_json(args.anchors_json)
    else:
        fixed_point_labels = DEFAULT_FIXED_POINT_LABELS
        fixed_points_lonlat = DEFAULT_FIXED_POINTS_LONLAT
        anchor_groups = DEFAULT_ANCHOR_GROUPS

    if len(fixed_point_labels) == 0:
        raise SystemExit(
            "Please provide anchors via --anchors-json or fill DEFAULT_FIXED_POINT_LABELS / DEFAULT_FIXED_POINTS_LONLAT / DEFAULT_ANCHOR_GROUPS in the script."
        )

    run_kfold_gridsearch_pareto(
        fixed_point_labels=fixed_point_labels,
        fixed_points_lonlat=fixed_points_lonlat,
        anchor_groups=anchor_groups,
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
    )


if __name__ == "__main__":
    main()
