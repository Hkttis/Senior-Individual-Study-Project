"""Visual review plots for DC-SMACOF HPO candidates.

This script reads an existing DC-SMACOF HPO output folder, reruns one
representative seed for each alpha candidate, and exports position plots.

Usage
-----
python -m run_paper_script.paper_run ch5-dc-review --hpo-outdir outputs/ch5_dc_smacof_hparam_alpha_-2_0_seed0_9 --outdir outputs/ch5_dc_smacof_hparam_alpha_-2_0_seed0_9_review
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from library.config import (
    FILE_PATHS,
    OUTPUT_DIR,
    km2pix,
    refer_pos_screen,
    refer_pos_sim as DEFAULT_REFER_POS_SIM,
)
from library.data_io import (
    get_anchor_align_label,
    get_anchor_labels,
    get_test_site_labels,
    load_ini_data_from_csv,
    uploading_directional_data,
    uploading_ground_truth,
)
from library.metrics import alignment_and_scaling, calculate_kruskals_stress, direction_violation_rate
from library.geometry import lcc_transformation_with_anchor
from library.model_cmp import run_directed_MDS
from library.units import data_Li2sim, pos_matrix_sim2km
from library.visualization import visualize_error_map_official
from MDS_model.plot_node_link_diagram import wrong_directions_nonflip
from run_paper_script.ch5_ablation_study import _dc_smacof_weights_from_alpha
from run_paper_script.ch5_hparam_kfold_gridsearch_pareto import _rmse_labels_km


_MPLCONFIGDIR = Path(OUTPUT_DIR) / ".matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ensure_outdir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {path}. Use --overwrite intentionally.")
    path.mkdir(parents=True, exist_ok=True)


def _choose_representative_seed(runs: pd.DataFrame, alpha: float, preferred_seed: int | None) -> int:
    group = runs[(np.isclose(runs["alpha"].astype(float), float(alpha))) & (runs["status"] == "ok")].copy()
    if group.empty:
        raise ValueError(f"No successful DC-SMACOF HPO run found for alpha={alpha}")
    if preferred_seed is not None:
        hit = group[group["seed"].astype(int) == int(preferred_seed)]
        if hit.empty:
            raise ValueError(f"Requested seed={preferred_seed} is not successful for alpha={alpha}")
        return int(preferred_seed)

    metric_cols = ["E_distance_stress", "E_direction_vr", "RMSE_anc_km"]
    means = group[metric_cols].mean()
    stds = group[metric_cols].std(ddof=0).replace(0.0, 1.0).fillna(1.0)
    z = (group[metric_cols] - means) / stds
    group["_representative_distance"] = np.sqrt((z * z).sum(axis=1))
    row = group.sort_values(["_representative_distance", "seed"]).iloc[0]
    return int(row["seed"])


def _evaluate_anchor_rmse(
    *,
    pos_y_up_sim: np.ndarray,
    dni: dict[str, int],
    gt_lonlat: Sequence[Sequence[float]],
    anchor_labels: Sequence[str],
    anchor_align_label: str,
    refer_pos_sim: Sequence[float],
) -> float:
    rmse_labels = [label for label in anchor_labels if label != anchor_align_label]
    anchor_lonlat = [tuple(gt_lonlat[dni[label]]) for label in anchor_labels]
    return float(
        _rmse_labels_km(
            pos_y_up_sim=pos_y_up_sim,
            dni=dni,
            refer_pos_sim=refer_pos_sim,
            gt_labels=list(anchor_labels),
            gt_lonlat=anchor_lonlat,
            eval_labels=rmse_labels,
            anchor_label_for_frame=anchor_align_label,
        )
    )


def _plot_candidate(
    *,
    outpath: Path,
    alpha: float,
    seed: int,
    pos_y_up_sim: np.ndarray,
    gt_y_up_sim: np.ndarray,
    vertice: Sequence[str],
    dni: dict[str, int],
    anchor_labels: Sequence[str],
    anchor_align_label: str,
    test_labels: Sequence[str],
    metrics: dict,
) -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft JhengHei",
        "Microsoft YaHei",
        "Noto Sans CJK TC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.axhline(0, color="#bbbbbb", linewidth=0.8)
    ax.axvline(0, color="#bbbbbb", linewidth=0.8)
    ax.grid(True, color="#dddddd", linewidth=0.7, alpha=0.7)

    def draw_labels(labels: Sequence[str], gt_color: str, pred_color: str, label_prefix: str) -> None:
        first_gt = True
        first_pred = True
        for name in labels:
            idx = dni[name]
            gt = gt_y_up_sim[idx]
            pred = pos_y_up_sim[idx]
            if not np.isfinite(gt).all() or np.allclose(gt, [0.0, 0.0]):
                continue
            ax.plot(
                [gt[0], pred[0]],
                [gt[1], pred[1]],
                color=pred_color,
                alpha=0.25,
                linewidth=1.3,
            )
            ax.scatter(
                gt[0],
                gt[1],
                s=80,
                marker="o",
                color=gt_color,
                label=f"GT {label_prefix}" if first_gt else None,
                zorder=3,
            )
            ax.scatter(
                pred[0],
                pred[1],
                s=80,
                marker="x",
                color=pred_color,
                label=f"Pred {label_prefix}" if first_pred else None,
                zorder=4,
            )
            ax.text(pred[0] + 8, pred[1] + 8, name, fontsize=9, color="#222222")
            first_gt = False
            first_pred = False

    fixed_labels = [anchor_align_label]
    eval_anchor_labels = [label for label in anchor_labels if label != anchor_align_label]
    draw_labels(fixed_labels, "#d62728", "#d62728", "anchor_align")
    draw_labels(eval_anchor_labels, "#ff7f0e", "#ff7f0e", "RMSE_anchor")
    draw_labels(test_labels, "#1f77b4", "#1f77b4", "test")

    all_focus_xy: list[tuple[float, float]] = []
    for name in list(anchor_labels) + list(test_labels):
        idx = dni[name]
        gt = gt_y_up_sim[idx]
        pred = pos_y_up_sim[idx]
        if np.isfinite(gt).all():
            all_focus_xy.append((float(gt[0]), float(gt[1])))
        if np.isfinite(pred).all():
            all_focus_xy.append((float(pred[0]), float(pred[1])))

    for name in vertice:
        if name in set(anchor_labels) or name in set(test_labels):
            continue
        idx = dni[name]
        ax.scatter(pos_y_up_sim[idx, 0], pos_y_up_sim[idx, 1], s=18, color="#777777", alpha=0.35)

    ax.set_aspect("equal", adjustable="box")
    if all_focus_xy:
        xy = np.asarray(all_focus_xy, dtype=float)
        x_min, y_min = xy.min(axis=0)
        x_max, y_max = xy.max(axis=0)
        span = max(float(x_max - x_min), float(y_max - y_min), 100.0)
        pad = span * 0.08
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_xlabel("x sim, aligned by anchor_align")
    ax.set_ylabel("y sim")
    ax.set_title(
        "DC-SMACOF alpha={alpha:g}, seed={seed} | Stress={stress:.4f}, VR={vr:.4f}, RMSE_anc={rmse:.2f} km".format(
            alpha=alpha,
            seed=seed,
            stress=metrics["E_distance_stress"],
            vr=metrics["E_direction_vr"],
            rmse=metrics["RMSE_anc_km"],
        )
    )
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def _to_pygame_display_coordinates(
    pos_y_up_sim: np.ndarray,
    *,
    anchor_idx: int,
    target_anchor_xy: Sequence[float],
) -> list[list[float]]:
    """Flip y-up coordinates for Pygame while keeping the frame anchor fixed."""
    tx, ty = float(target_anchor_xy[0]), float(target_anchor_xy[1])
    flipped = [[float(x), 2.0 * ty - float(y)] for x, y in pos_y_up_sim]
    dx = tx - flipped[anchor_idx][0]
    dy = ty - flipped[anchor_idx][1]
    return [[x + dx, y + dy] for x, y in flipped]


def _export_error_map(
    *,
    outdir: Path,
    alpha: float,
    seed: int,
    pos_y_up_sim: np.ndarray,
    vertice: Sequence[str],
    dni: dict[str, int],
    data_li,
    anchor_align_label: str,
) -> str:
    # Error-map rendering uses Pygame's y-down screen coordinate system.
    pos_pygame = _to_pygame_display_coordinates(
        pos_y_up_sim,
        anchor_idx=dni[anchor_align_label],
        target_anchor_xy=refer_pos_screen,
    )
    alpha_tag = f"{alpha:g}".replace("-", "neg").replace(".", "p")
    file_prefix = f"DC-SMACOF_alpha_{alpha_tag}_seed{seed}_"
    wrong_dir = wrong_directions_nonflip(pos_y_up_sim.tolist(), vertice, dni)
    visualize_error_map_official(
        pos_pygame,
        vertice,
        dni,
        data_li,
        wrong_dir,
        zoom_area=None,
        file_name=file_prefix,
        wait=False,
    )
    source = Path(OUTPUT_DIR) / f"{file_prefix}error_map_full.png"
    target = outdir / source.name
    if source.exists():
        shutil.copy2(source, target)
    else:
        raise FileNotFoundError(f"Expected error map was not created: {source}")
    return target.name


def export_dc_smacof_hparam_review(
    *,
    hpo_outdir: str | Path,
    outdir: str | Path,
    seed: int | None = None,
    overwrite: bool = False,
    refer_pos_sim: Sequence[float] = DEFAULT_REFER_POS_SIM,
) -> None:
    hpo_path = Path(hpo_outdir)
    outdir_path = Path(outdir)
    _ensure_outdir(outdir_path, overwrite)

    summary_path = hpo_path / "dc_smacof_hparam_summary.csv"
    runs_path = hpo_path / "dc_smacof_hparam_runs_by_seed.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary CSV: {summary_path}")
    if not runs_path.exists():
        raise FileNotFoundError(f"Missing runs CSV: {runs_path}")

    summary = pd.read_csv(summary_path)
    runs = pd.read_csv(runs_path)
    if summary.empty:
        raise ValueError(f"Summary CSV is empty: {summary_path}")

    _graph, vertice, dni, _edges, data_li = load_ini_data_from_csv(FILE_PATHS)
    data_sim = data_Li2sim(data_li)
    directional_data = uploading_directional_data()
    gt_lonlat = uploading_ground_truth(vertice, dni)
    anchor_labels = get_anchor_labels()
    anchor_align_label = get_anchor_align_label()
    test_labels = get_test_site_labels()
    gt_km = lcc_transformation_with_anchor(dni, gt_lonlat, anchor_label=anchor_align_label)
    gt_y_up_sim = np.full((len(vertice), 2), np.nan, dtype=float)
    for idx, (x_km, y_km) in enumerate(gt_km):
        if x_km is None or y_km is None:
            continue
        gt_y_up_sim[idx, 0] = float(refer_pos_sim[0]) + float(x_km) * km2pix
        gt_y_up_sim[idx, 1] = float(refer_pos_sim[1]) + float(y_km) * km2pix

    rows: list[dict] = []
    position_rows: list[dict] = []
    for _, row in summary.sort_values("alpha").iterrows():
        alpha = float(row["alpha"])
        selected_seed = _choose_representative_seed(runs, alpha, seed)
        w_weight, v_weight = _dc_smacof_weights_from_alpha(alpha)

        np.random.seed(selected_seed)
        pos_history_li = run_directed_MDS(vis=False, w_weight_value=w_weight, v_weight_value=v_weight)
        pos_li = pos_history_li[-1]
        pos_y_up_sim = np.asarray(
            alignment_and_scaling(
                pos_li,
                vertice,
                dni,
                refer_pos_sim,
                y_down=False,
                anchor_label=anchor_align_label,
            ),
            dtype=float,
        )
        metrics = {
            "E_distance_stress": float(calculate_kruskals_stress(dni, pos_matrix_sim2km(pos_y_up_sim.tolist()), data_sim)),
            "E_direction_vr": float(direction_violation_rate(pos_y_up_sim, directional_data, dni)),
            "RMSE_anc_km": _evaluate_anchor_rmse(
                pos_y_up_sim=pos_y_up_sim,
                dni=dni,
                gt_lonlat=gt_lonlat,
                anchor_labels=anchor_labels,
                anchor_align_label=anchor_align_label,
                refer_pos_sim=refer_pos_sim,
            ),
        }
        alpha_tag = f"{alpha:g}".replace("-", "neg").replace(".", "p")
        png_name = f"DC-SMACOF_alpha_{alpha_tag}_seed{selected_seed}_position_review.png"
        png_path = outdir_path / png_name
        _plot_candidate(
            outpath=png_path,
            alpha=alpha,
            seed=selected_seed,
            pos_y_up_sim=pos_y_up_sim,
            gt_y_up_sim=gt_y_up_sim,
            vertice=vertice,
            dni=dni,
            anchor_labels=anchor_labels,
            anchor_align_label=anchor_align_label,
            test_labels=test_labels,
            metrics=metrics,
        )
        error_map_file = _export_error_map(
            outdir=outdir_path,
            alpha=alpha,
            seed=selected_seed,
            pos_y_up_sim=pos_y_up_sim,
            vertice=vertice,
            dni=dni,
            data_li=data_li,
            anchor_align_label=anchor_align_label,
        )

        rows.append(
            {
                "alpha": alpha,
                "seed": selected_seed,
                "w_weight": w_weight,
                "v_weight": v_weight,
                "plot_file": png_name,
                "error_map_file": error_map_file,
                **metrics,
            }
        )
        for label in vertice:
            idx = dni[label]
            position_rows.append(
                {
                    "alpha": alpha,
                    "seed": selected_seed,
                    "label": label,
                    "x_y_up_sim": float(pos_y_up_sim[idx, 0]),
                    "y_y_up_sim": float(pos_y_up_sim[idx, 1]),
                }
            )
        print(f"[Saved] {png_path}")

    pd.DataFrame(rows).to_csv(outdir_path / "dc_smacof_hparam_review_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(position_rows).to_csv(outdir_path / "dc_smacof_hparam_review_positions.csv", index=False, encoding="utf-8-sig")
    (outdir_path / "dc_smacof_hparam_review_config.json").write_text(
        json.dumps(
            {
                "hpo_outdir": str(hpo_path),
                "seed_mode": "fixed" if seed is not None else "representative_per_alpha",
                "requested_seed": seed,
                "anchor_labels": anchor_labels,
                "anchor_align_label": anchor_align_label,
                "test_labels": test_labels,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[Saved] {outdir_path / 'dc_smacof_hparam_review_metrics.csv'}")
    print(f"[Saved] {outdir_path / 'dc_smacof_hparam_review_positions.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export DC-SMACOF HPO candidate position review plots.")
    parser.add_argument("--hpo-outdir", required=True, help="Existing DC-SMACOF HPO output directory.")
    parser.add_argument("--outdir", required=True, help="Output directory for review plots.")
    parser.add_argument("--seed", type=int, default=None, help="Use one fixed seed for every alpha. Default chooses representative seed per alpha.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty outdir.")
    args = parser.parse_args()
    export_dc_smacof_hparam_review(
        hpo_outdir=args.hpo_outdir,
        outdir=args.outdir,
        seed=args.seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
