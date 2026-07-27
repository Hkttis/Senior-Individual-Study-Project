"""DC-SMACOF hyperparameter search.

This script searches alpha = log10(v_weight / w_weight) for DC-SMACOF.
It aligns every run by anchor_align, evaluates RMSE_anc on the remaining
anchor sites, and builds a Pareto front over stress, VR, and RMSE_anc.

Usage
-----
python -m run_paper_script.paper_run ch5-dc-hparam --seeds 0,1,2,3,4,5,6,7,8,9 --alpha-min -4 --alpha-max 0 --alpha-step 0.5 --outdir outputs/ch5_dc_smacof_hparam_wang_current_alpha_-4_0_seed0_9_20260721
"""

from __future__ import annotations

import argparse
import json
import math
import os
from copy import deepcopy
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from library.config import FILE_PATHS, OUTPUT_DIR, refer_pos_sim as DEFAULT_REFER_POS_SIM
from library.data_io import (
    get_anchor_align_label,
    get_anchor_labels,
    load_ini_data_from_csv,
    uploading_directional_data,
    uploading_ground_truth,
)
from library.geometry import get_lcc_bounds, get_lcc_parameters
from library.metrics import alignment_and_scaling, calculate_kruskals_stress, direction_violation_rate
from library.model_cmp import get_dc_smacof_direction_method_metadata, run_directed_MDS
from library.units import data_Li2sim, pos_matrix_sim2km
from run_paper_script.ch5_hparam_kfold_gridsearch_pareto import _rmse_labels_km


_MPLCONFIGDIR = Path(OUTPUT_DIR) / ".matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OBJECTIVE_COLS = ["E_distance_stress_mean", "E_direction_vr_mean", "RMSE_anc_mean_km"]


def _parse_seed_list(raw: str) -> list[int]:
    seeds = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not seeds:
        raise ValueError("--seeds cannot be empty")
    return seeds


def _make_alpha_grid(alpha_min: float, alpha_max: float, alpha_step: float) -> np.ndarray:
    if alpha_step <= 0:
        raise ValueError("--alpha-step must be positive")
    return np.arange(alpha_min, alpha_max + 1e-12, alpha_step, dtype=float)


def _dc_weights_from_alpha(alpha: float, w_weight: float = 1.0) -> tuple[float, float]:
    distance_weight = float(w_weight)
    direction_weight = float(distance_weight * math.pow(10.0, float(alpha)))
    return distance_weight, direction_weight


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


def _series_stats(values: Sequence[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"), "se": float("nan"), "n": 0}
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    return {
        "mean": float(arr.mean()),
        "std": std,
        "se": float(std / math.sqrt(n)) if n > 0 else float("nan"),
        "n": n,
    }


def _anchor_split() -> tuple[list[str], str, list[str]]:
    anchor_labels = get_anchor_labels()
    if len(anchor_labels) < 2:
        raise ValueError(f"DC-SMACOF HPO needs at least two anchor sites, got {anchor_labels}")
    anchor_align_label = get_anchor_align_label()
    if anchor_align_label not in anchor_labels:
        raise ValueError(
            f"anchor_align {anchor_align_label!r} must be one of use_role=anchor labels for DC-SMACOF HPO. "
            f"anchors={anchor_labels}"
        )
    rmse_anchor_labels = [label for label in anchor_labels if label != anchor_align_label]
    if not rmse_anchor_labels:
        raise ValueError("No anchor labels remain for RMSE_anc after excluding anchor_align.")
    return anchor_labels, anchor_align_label, rmse_anchor_labels


def _evaluate_dc_smacof_run(
    *,
    seed: int,
    alpha: float,
    vertice: Sequence[str],
    dni: dict[str, int],
    data_sim,
    directional_data,
    gt_lonlat,
    anchor_labels: Sequence[str],
    anchor_align_label: str,
    rmse_anchor_labels: Sequence[str],
    refer_pos_sim: Sequence[float],
    w_weight: float,
) -> dict:
    distance_weight, direction_weight = _dc_weights_from_alpha(alpha, w_weight=w_weight)
    np.random.seed(int(seed))
    pos_history_li = run_directed_MDS(
        vis=False,
        w_weight_value=distance_weight,
        v_weight_value=direction_weight,
    )
    pos_li = pos_history_li[-1]
    pos_y_up_sim = alignment_and_scaling(
        pos_li,
        vertice,
        dni,
        refer_pos_sim,
        y_down=False,
        anchor_label=anchor_align_label,
    )
    pos = np.asarray(pos_y_up_sim, dtype=float)
    anchor_lonlat = [tuple(gt_lonlat[dni[label]]) for label in anchor_labels]
    rmse_anc = _rmse_labels_km(
        pos_y_up_sim=pos,
        dni=dni,
        refer_pos_sim=refer_pos_sim,
        gt_labels=list(anchor_labels),
        gt_lonlat=anchor_lonlat,
        eval_labels=list(rmse_anchor_labels),
        anchor_label_for_frame=anchor_align_label,
    )
    return {
        "alpha": float(alpha),
        "seed": int(seed),
        "w_weight": float(distance_weight),
        "v_weight": float(direction_weight),
        "status": "ok",
        "error": "",
        "n_iterations": int(len(pos_history_li)),
        "E_distance_stress": float(calculate_kruskals_stress(dni, pos_matrix_sim2km(pos.tolist()), data_sim)),
        "E_direction_vr": float(direction_violation_rate(pos, directional_data, dni)),
        "RMSE_anc_km": float(rmse_anc),
    }


def _build_summary(df_runs: pd.DataFrame, alphas: Sequence[float], seeds: Sequence[int], w_weight: float) -> pd.DataFrame:
    rows: list[dict] = []
    for alpha in alphas:
        group = df_runs[(np.isclose(df_runs["alpha"].astype(float), float(alpha))) & (df_runs["status"] == "ok")]
        distance_weight, direction_weight = _dc_weights_from_alpha(float(alpha), w_weight=w_weight)
        row = {
            "alpha": float(alpha),
            "w_weight": float(distance_weight),
            "v_weight": float(direction_weight),
            "n_seeds": int(len(seeds)),
            "n_success": int(len(group)),
            "failure_count": int(len(seeds) - len(group)),
        }
        for source, prefix in [
            ("E_distance_stress", "E_distance_stress"),
            ("E_direction_vr", "E_direction_vr"),
            ("RMSE_anc_km", "RMSE_anc"),
        ]:
            stats = _series_stats(group[source].tolist() if not group.empty else [])
            row[f"{prefix}_mean" + ("_km" if prefix == "RMSE_anc" else "")] = stats["mean"]
            row[f"{prefix}_std" + ("_km" if prefix == "RMSE_anc" else "")] = stats["std"]
            row[f"{prefix}_se" + ("_km" if prefix == "RMSE_anc" else "")] = stats["se"]
        rows.append(row)
    return pd.DataFrame(rows)


def _select_one_se_balanced_candidate(df_pareto: pd.DataFrame, manual_threshold: int) -> tuple[pd.Series | None, dict]:
    if df_pareto.empty:
        raise ValueError("No Pareto candidates found.")
    if len(df_pareto) <= int(manual_threshold):
        return None, {
            "selection_rule": "manual_review_recommended",
            "pareto_count": int(len(df_pareto)),
            "manual_threshold": int(manual_threshold),
            "reason": "Pareto front is small; inspect candidates manually.",
        }

    min_rmse_idx = df_pareto["RMSE_anc_mean_km"].idxmin()
    min_rmse_row = df_pareto.loc[min_rmse_idx]
    min_rmse = float(min_rmse_row["RMSE_anc_mean_km"])
    min_se = float(min_rmse_row["RMSE_anc_se_km"])
    threshold = min_rmse + min_se
    candidates = df_pareto[df_pareto["RMSE_anc_mean_km"] <= threshold].copy()
    if candidates.empty:
        candidates = df_pareto.loc[[min_rmse_idx]].copy()

    mins = df_pareto[OBJECTIVE_COLS].min()
    ranges = (df_pareto[OBJECTIVE_COLS].max() - mins).replace(0, 1.0)
    standardized = (candidates[OBJECTIVE_COLS] - mins) / ranges
    candidates["one_se_balanced_score"] = np.sqrt((standardized**2).sum(axis=1))
    selected_idx = candidates["one_se_balanced_score"].idxmin()
    selected = candidates.loc[selected_idx]
    return selected, {
        "selection_rule": "pareto_one_se_balanced",
        "pareto_count": int(len(df_pareto)),
        "manual_threshold": int(manual_threshold),
        "one_se_reference_alpha": float(min_rmse_row["alpha"]),
        "one_se_reference_rmse_anc_mean_km": min_rmse,
        "one_se_min_se_km": min_se,
        "one_se_threshold_km": float(threshold),
        "one_se_candidate_count": int(len(candidates)),
        "balanced_objectives": list(OBJECTIVE_COLS),
        "balanced_score": float(selected["one_se_balanced_score"]),
    }


def _plot_metric_lines(df_summary: pd.DataFrame, out_png: Path) -> None:
    metrics = [
        ("E_distance_stress_mean", "Distance stress"),
        ("E_direction_vr_mean", "Direction violation rate"),
        ("RMSE_anc_mean_km", "RMSE_anc (km)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (col, title) in zip(axes, metrics):
        ax.plot(df_summary["alpha"], df_summary[col], marker="o")
        se_col = col.replace("_mean", "_se")
        if se_col in df_summary:
            y = df_summary[col].to_numpy(float)
            se = df_summary[se_col].to_numpy(float)
            ax.fill_between(df_summary["alpha"], y - se, y + se, alpha=0.2)
        ax.set_xlabel("alpha = log10(v_weight / w_weight)")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _plot_pareto_3d(df_summary: pd.DataFrame, pareto_mask: np.ndarray, out_png: Path) -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(8.8, 7.0))
    ax = fig.add_subplot(111, projection="3d")
    x = df_summary["E_distance_stress_mean"].to_numpy(float)
    y = df_summary["E_direction_vr_mean"].to_numpy(float)
    z = df_summary["RMSE_anc_mean_km"].to_numpy(float)
    ax.scatter(x[~pareto_mask], y[~pareto_mask], z[~pareto_mask], alpha=0.45, label="All alpha")
    ax.scatter(x[pareto_mask], y[pareto_mask], z[pareto_mask], s=45, label="Pareto front")
    for _, row in df_summary.iterrows():
        ax.text(row["E_distance_stress_mean"], row["E_direction_vr_mean"], row["RMSE_anc_mean_km"], f"a={row['alpha']:g}", fontsize=8)
    ax.set_xlabel("E_distance_stress")
    ax.set_ylabel("E_direction_vr")
    ax.set_zlabel("RMSE_anc_km")
    ax.set_title("DC-SMACOF Pareto Front")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _plot_pareto_2d(df_summary: pd.DataFrame, pareto_mask: np.ndarray, out_png: Path) -> None:
    pairs = [
        ("E_distance_stress_mean", "E_direction_vr_mean"),
        ("E_distance_stress_mean", "RMSE_anc_mean_km"),
        ("E_direction_vr_mean", "RMSE_anc_mean_km"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (xcol, ycol) in zip(axes, pairs):
        ax.scatter(df_summary.loc[~pareto_mask, xcol], df_summary.loc[~pareto_mask, ycol], alpha=0.45)
        ax.scatter(df_summary.loc[pareto_mask, xcol], df_summary.loc[pareto_mask, ycol], s=42)
        for _, row in df_summary.iterrows():
            ax.annotate(f"{row['alpha']:g}", (row[xcol], row[ycol]), fontsize=8)
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        ax.grid(alpha=0.25)
    fig.suptitle("DC-SMACOF Pareto 2D Projections")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def run_dc_smacof_hparam(
    *,
    seeds: Sequence[int],
    alpha_min: float,
    alpha_max: float,
    alpha_step: float,
    outdir: str | Path,
    w_weight: float = 1.0,
    refer_pos_sim: Sequence[float] = DEFAULT_REFER_POS_SIM,
    manual_threshold: int = 5,
    overwrite: bool = False,
) -> dict:
    outdir_path = Path(outdir)
    if outdir_path.exists() and any(outdir_path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"DC-SMACOF HPO outdir already exists and is not empty: {outdir_path}. "
            "Choose a new --outdir or pass --overwrite intentionally."
        )
    outdir_path.mkdir(parents=True, exist_ok=True)

    graph, vertice, dni, edges, data_li = load_ini_data_from_csv(FILE_PATHS)
    data_sim = data_Li2sim(data_li)
    directional_data = uploading_directional_data()
    direction_method_metadata, direction_preprocessing = get_dc_smacof_direction_method_metadata(
        directional_data,
        dni,
    )
    gt_lonlat = uploading_ground_truth(vertice, dni)
    anchor_labels, anchor_align_label, rmse_anchor_labels = _anchor_split()
    alphas = _make_alpha_grid(alpha_min, alpha_max, alpha_step)

    run_rows: list[dict] = []
    total = len(alphas) * len(seeds)
    idx = 0
    for alpha in alphas:
        for seed in seeds:
            idx += 1
            print(f"[{idx}/{total}] DC-SMACOF alpha={alpha:g}, seed={seed}")
            try:
                run_rows.append(
                    _evaluate_dc_smacof_run(
                        seed=int(seed),
                        alpha=float(alpha),
                        vertice=vertice,
                        dni=dni,
                        data_sim=data_sim,
                        directional_data=directional_data,
                        gt_lonlat=gt_lonlat,
                        anchor_labels=anchor_labels,
                        anchor_align_label=anchor_align_label,
                        rmse_anchor_labels=rmse_anchor_labels,
                        refer_pos_sim=refer_pos_sim,
                        w_weight=w_weight,
                    )
                )
            except Exception as exc:
                distance_weight, direction_weight = _dc_weights_from_alpha(float(alpha), w_weight=w_weight)
                run_rows.append(
                    {
                        "alpha": float(alpha),
                        "seed": int(seed),
                        "w_weight": float(distance_weight),
                        "v_weight": float(direction_weight),
                        "status": "failed",
                        "error": repr(exc),
                        "n_iterations": 0,
                        "E_distance_stress": float("nan"),
                        "E_direction_vr": float("nan"),
                        "RMSE_anc_km": float("nan"),
                    }
                )

    df_runs = pd.DataFrame(run_rows)
    df_summary = _build_summary(df_runs, alphas, seeds, w_weight)
    objectives = df_summary[OBJECTIVE_COLS].to_numpy(float)
    pareto_mask = _non_dominated_mask(objectives)
    df_pareto = df_summary.loc[pareto_mask].copy().sort_values(["RMSE_anc_mean_km", "E_direction_vr_mean", "E_distance_stress_mean"])
    selected, selection_meta = _select_one_se_balanced_candidate(df_pareto, manual_threshold=manual_threshold)

    df_runs.to_csv(outdir_path / "dc_smacof_hparam_runs_by_seed.csv", index=False, encoding="utf-8-sig")
    df_summary.to_csv(outdir_path / "dc_smacof_hparam_summary.csv", index=False, encoding="utf-8-sig")
    df_pareto.to_csv(outdir_path / "dc_smacof_pareto_front.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                **record,
                "original_observations": json.dumps(record["original_observations"], ensure_ascii=False),
            }
            for record in direction_preprocessing
        ]
    ).to_csv(
        outdir_path / "dc_smacof_direction_preprocessing.csv",
        index=False,
        encoding="utf-8-sig",
    )
    if selected is not None:
        pd.DataFrame([selected.to_dict()]).to_csv(outdir_path / "dc_smacof_selected_candidate.csv", index=False, encoding="utf-8-sig")
    (outdir_path / "dc_smacof_selection_meta.json").write_text(
        json.dumps(selection_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    config = {
        "model": "DC-SMACOF",
        **direction_method_metadata,
        "direction_preprocessing_file": "dc_smacof_direction_preprocessing.csv",
        "alpha_range": [float(alpha_min), float(alpha_max), float(alpha_step)],
        "alpha_scale": "base-10: v_weight=w_weight*10^alpha",
        "seeds": [int(s) for s in seeds],
        "w_weight": float(w_weight),
        "anchor_labels": list(anchor_labels),
        "anchor_align_label": anchor_align_label,
        "rmse_anchor_labels": list(rmse_anchor_labels),
        "refer_pos_sim": [float(x) for x in refer_pos_sim],
        "objectives": list(OBJECTIVE_COLS),
        "manual_threshold": int(manual_threshold),
        "selection_meta": selection_meta,
        "lcc_bounds": get_lcc_bounds(),
        "lcc_parameters": get_lcc_parameters(),
        "file_paths": {
            "ini_data": FILE_PATHS["ini_data"],
            "directional_data": FILE_PATHS["directional_data"],
            "ground_truth_path": FILE_PATHS["ground_truth_path"],
        },
    }
    (outdir_path / "dc_smacof_hparam_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _plot_metric_lines(df_summary, outdir_path / "dc_smacof_hparam_metric_lines.png")
    _plot_pareto_3d(df_summary, pareto_mask, outdir_path / "dc_smacof_pareto_front_3d.png")
    _plot_pareto_2d(df_summary, pareto_mask, outdir_path / "dc_smacof_pareto_front_2d.png")

    print(f"[Saved] {outdir_path / 'dc_smacof_hparam_summary.csv'}")
    print(f"[Saved] {outdir_path / 'dc_smacof_pareto_front.csv'}")
    print(f"[Selection] {selection_meta['selection_rule']}")
    return {
        "outdir": str(outdir_path),
        "summary": df_summary,
        "pareto": df_pareto,
        "selected": selected,
        "selection_meta": selection_meta,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DC-SMACOF alpha HPO.")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--alpha-min", type=float, default=-2.0)
    parser.add_argument("--alpha-max", type=float, default=0.0)
    parser.add_argument("--alpha-step", type=float, default=0.5)
    parser.add_argument("--w-weight", type=float, default=1.0)
    parser.add_argument("--outdir", type=str, default=str(Path(OUTPUT_DIR) / "ch5_dc_smacof_hparam"))
    parser.add_argument("--manual-threshold", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_dc_smacof_hparam(
        seeds=_parse_seed_list(args.seeds),
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        alpha_step=args.alpha_step,
        outdir=args.outdir,
        w_weight=args.w_weight,
        manual_threshold=args.manual_threshold,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
