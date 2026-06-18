"""Export LOO fold review CSV/PNG files for a selected HPO result.

Usage
-----
python -m scripts.export_hpo_loo_review --hpo-outdir outputs/ch5_hparam_anchor_loo_grid_3x3_seed0_final
python -m scripts.export_hpo_loo_review --hpo-outdir outputs/ch5_hparam_anchor_loo_grid_3x3_seed0_final --alpha 1 --beta 0 --seed 0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from library.config import (
    DIRECTIONAL_FORCE_MAGNITUDE_BASE,
    FILE_PATHS,
    OUTPUT_DIR,
    REPULSION_STRENGTH_BASE,
    SPRING_STIFFNESS_BASE,
    km2pix,
    refer_pos_sim as DEFAULT_REFER_POS_SIM,
)

_MPLCONFIGDIR = Path(OUTPUT_DIR) / ".matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pandas as pd

from library.anchor_frame import px_list_to_km_list
from library.data_io import load_ini_data_from_csv, load_site_points
from library.geometry import inverse_lcc_transformation, lcc_transformation_with_anchor
from run_paper_script.ch5_hparam_kfold_gridsearch_pareto import (
    _build_anchor_loo_folds,
    _load_anchor_and_test_inputs,
    _run_physics_eval,
    _weights_from_alpha_beta,
)


def _safe_name(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", name)


def _selected_alpha_beta(hpo_outdir: Path) -> tuple[float, float]:
    candidate_csv = hpo_outdir / "selected_candidate_summary.csv"
    if candidate_csv.exists():
        df = pd.read_csv(candidate_csv)
        if df.empty:
            raise ValueError(f"selected_candidate_summary.csv is empty: {candidate_csv}")
        return float(df.iloc[0]["alpha"]), float(df.iloc[0]["beta"])

    summary_path = hpo_outdir / "selected_final_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Neither selected_candidate_summary.csv nor selected_final_summary.json found in {hpo_outdir}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return float(summary["alpha"]), float(summary["beta"])


def export_hpo_loo_review(
    *,
    hpo_outdir: str | Path,
    alpha: float | None = None,
    beta: float | None = None,
    seed: int = 0,
    w_dis: float = 1.0,
    base_spring_stiffness: float = SPRING_STIFFNESS_BASE,
    base_directional_force: float = DIRECTIONAL_FORCE_MAGNITUDE_BASE,
    base_repulsion_strength: float = REPULSION_STRENGTH_BASE,
    refer_pos_sim: Sequence[float] = DEFAULT_REFER_POS_SIM,
) -> Path:
    hpo_outdir = Path(hpo_outdir)
    if alpha is None or beta is None:
        selected_alpha, selected_beta = _selected_alpha_beta(hpo_outdir)
        alpha = selected_alpha if alpha is None else alpha
        beta = selected_beta if beta is None else beta

    outdir = hpo_outdir / "loo_fold_review"
    outdir.mkdir(parents=True, exist_ok=True)

    font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    colors = {"anchor_fixed": "#d62728", "anchor_heldout": "#ff7f0e", "test": "#1f77b4"}

    _graph, vertice, dni, _edges, _data = load_ini_data_from_csv(FILE_PATHS)
    site_rows = load_site_points()
    site = {row["name"]: row for row in site_rows}
    anchor_labels, anchor_lonlat, test_labels, _test_lonlat = _load_anchor_and_test_inputs(vertice, dni)
    folds = _build_anchor_loo_folds(anchor_labels, anchor_lonlat)
    _w_dir, _w_reg, spring, directional_force, repulsion = _weights_from_alpha_beta(
        float(alpha),
        float(beta),
        w_dis,
        base_spring_stiffness,
        base_directional_force,
        base_repulsion_strength,
    )

    all_rows: list[dict] = []
    for fold in folds:
        metrics, pos_final, fold_vertice, fold_dni = _run_physics_eval(
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

        pred_km = px_list_to_km_list(pos_final.tolist(), tuple(refer_pos_sim), km2pix)
        align_lonlat = (
            float(site[fold.train_anchor_label]["lon"]),
            float(site[fold.train_anchor_label]["lat"]),
        )
        pred_lonlat = inverse_lcc_transformation(pred_km, align_lonlat)

        gt_lonlat = [(0.0, 0.0) for _ in fold_vertice]
        eval_labels = list(anchor_labels) + list(test_labels)
        for label in eval_labels:
            gt_lonlat[fold_dni[label]] = (float(site[label]["lon"]), float(site[label]["lat"]))
        gt_km = lcc_transformation_with_anchor(fold_dni, gt_lonlat, anchor_label=fold.train_anchor_label)

        rows: list[dict] = []
        for label in eval_labels:
            idx = fold_dni[label]
            px, py = pred_km[idx]
            gx, gy = gt_km[idx]
            if label in fold.train_labels:
                role = "anchor_fixed"
            elif label == fold.heldout_label:
                role = "anchor_heldout"
            else:
                role = "test"
            row = {
                "fold_id": fold.fold_id,
                "alpha": float(alpha),
                "beta": float(beta),
                "seed": int(seed),
                "train_labels": "|".join(fold.train_labels),
                "heldout_label": fold.heldout_label,
                "frame_anchor_label": fold.train_anchor_label,
                "label": label,
                "role": role,
                "gt_lon": gt_lonlat[idx][0],
                "gt_lat": gt_lonlat[idx][1],
                "pred_lon": pred_lonlat[idx][0],
                "pred_lat": pred_lonlat[idx][1],
                "gt_x_km": gx,
                "gt_y_km": gy,
                "pred_x_km": px,
                "pred_y_km": py,
                "error_km": math.hypot(px - gx, py - gy),
                "fold_RMSE_anchor_LOO_km": metrics["RMSE_km"],
                "E_distance_stress": metrics["E_distance_stress"],
                "E_direction_vr": metrics["E_direction_vr"],
            }
            rows.append(row)
            all_rows.append(row)

        df = pd.DataFrame(rows)
        suffix = _safe_name(fold.heldout_label)
        csv_path = outdir / f"fold_{fold.fold_id}_heldout_{suffix}_review.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        fig, ax = plt.subplots(figsize=(9, 7))
        used_labels: set[str] = set()
        for _, row in df.iterrows():
            color = colors[row["role"]]
            gt_label = "GT " + row["role"]
            pred_label = "Pred " + row["role"]
            ax.scatter(
                row["gt_x_km"],
                row["gt_y_km"],
                marker="o",
                s=70,
                color=color,
                label=gt_label if gt_label not in used_labels else None,
            )
            used_labels.add(gt_label)
            ax.scatter(
                row["pred_x_km"],
                row["pred_y_km"],
                marker="x",
                s=72,
                color=color,
                label=pred_label if pred_label not in used_labels else None,
            )
            used_labels.add(pred_label)
            ax.plot(
                [row["gt_x_km"], row["pred_x_km"]],
                [row["gt_y_km"], row["pred_y_km"]],
                color=color,
                alpha=0.28,
                linewidth=1,
            )
            ax.text(row["pred_x_km"] + 4, row["pred_y_km"] + 4, row["label"], fontsize=8, fontproperties=font)

        heldout_row = df[df["role"] == "anchor_heldout"].iloc[0]
        ax.scatter(
            heldout_row["pred_x_km"],
            heldout_row["pred_y_km"],
            marker="s",
            s=160,
            facecolors="none",
            edgecolors="black",
            linewidths=1.4,
        )
        ax.axhline(0, color="#999999", linewidth=0.6)
        ax.axvline(0, color="#999999", linewidth=0.6)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)
        ax.set_xlabel("x km, anchored at " + fold.train_anchor_label, fontproperties=font)
        ax.set_ylabel("y km", fontproperties=font)
        ax.set_title(
            (
                f"LOO fold {fold.fold_id}: fixed {' + '.join(fold.train_labels)}, "
                f"held out {fold.heldout_label}, RMSE={metrics['RMSE_km']:.2f} km"
            ),
            fontproperties=font,
        )
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        png_path = outdir / f"fold_{fold.fold_id}_heldout_{suffix}_map.png"
        fig.savefig(png_path, dpi=180)
        plt.close(fig)
        print(f"[Saved] {png_path}")
        print(f"[Saved] {csv_path}")

    all_df = pd.DataFrame(all_rows)
    all_df.to_csv(outdir / "all_loo_fold_review.csv", index=False, encoding="utf-8-sig")
    print(f"[Saved] {outdir / 'all_loo_fold_review.csv'}")
    return outdir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export selected HPO leave-one-anchor-out review plots.")
    parser.add_argument("--hpo-outdir", required=True, help="Existing HPO output directory.")
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--w-dis", type=float, default=1.0)
    parser.add_argument("--base-spring", type=float, default=SPRING_STIFFNESS_BASE)
    parser.add_argument("--base-dir", type=float, default=DIRECTIONAL_FORCE_MAGNITUDE_BASE)
    parser.add_argument("--base-rep", type=float, default=REPULSION_STRENGTH_BASE)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    export_hpo_loo_review(
        hpo_outdir=args.hpo_outdir,
        alpha=args.alpha,
        beta=args.beta,
        seed=args.seed,
        w_dis=args.w_dis,
        base_spring_stiffness=args.base_spring,
        base_directional_force=args.base_dir,
        base_repulsion_strength=args.base_rep,
        refer_pos_sim=DEFAULT_REFER_POS_SIM,
    )


if __name__ == "__main__":
    main()
