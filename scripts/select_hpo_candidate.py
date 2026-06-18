"""Select a Pareto HPO candidate without modifying the original HPO outputs.

Usage
-----
python -m scripts.select_hpo_candidate --source-hpo-outdir outputs/ch5_hparam_anchor_loo_grid_main_36x10 --alpha 1.0 --beta -1.5 --seeds 0,1,2,3,4,5,6,7,8,9 --outdir outputs/ch5_hparam_anchor_loo_grid_main_36x10_manual_alpha_1_beta_-1.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from library.config import (
    DIRECTIONAL_FORCE_MAGNITUDE_BASE,
    FILE_PATHS,
    REPULSION_STRENGTH_BASE,
    SPRING_STIFFNESS_BASE,
    refer_pos_sim as DEFAULT_REFER_POS_SIM,
)
from library.data_io import get_anchor_labels, get_test_site_labels, load_ini_data_from_csv, uploading_ground_truth
from run_paper_script.ch5_hparam_kfold_gridsearch_pareto import _run_final_selected_model


def _parse_seed_list(raw: str) -> list[int]:
    seeds = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not seeds:
        raise ValueError("--seeds cannot be empty")
    return seeds


def _find_candidate_row(df: pd.DataFrame, alpha: float, beta: float) -> pd.Series:
    mask = np.isclose(df["alpha"].astype(float), float(alpha)) & np.isclose(df["beta"].astype(float), float(beta))
    match = df[mask]
    if match.empty:
        raise ValueError(f"No candidate found for alpha={alpha}, beta={beta}")
    if len(match) > 1:
        raise ValueError(f"Multiple candidates found for alpha={alpha}, beta={beta}")
    return match.iloc[0]


def select_hpo_candidate(
    *,
    source_hpo_outdir: str | Path,
    alpha: float,
    beta: float,
    seeds: Sequence[int],
    outdir: str | Path,
    w_dis: float = 1.0,
    base_spring_stiffness: float = SPRING_STIFFNESS_BASE,
    base_directional_force: float = DIRECTIONAL_FORCE_MAGNITUDE_BASE,
    base_repulsion_strength: float = REPULSION_STRENGTH_BASE,
    refer_pos_sim: Sequence[float] = DEFAULT_REFER_POS_SIM,
    allow_non_pareto: bool = False,
    overwrite: bool = False,
) -> Path:
    source_hpo_outdir = Path(source_hpo_outdir)
    outdir = Path(outdir)
    grid_path = source_hpo_outdir / "grid_summary_cv.csv"
    pareto_path = source_hpo_outdir / "pareto_front_3d.csv"
    if not grid_path.exists():
        raise FileNotFoundError(f"grid_summary_cv.csv not found: {grid_path}")
    if not pareto_path.exists():
        raise FileNotFoundError(f"pareto_front_3d.csv not found: {pareto_path}")

    df_grid = pd.read_csv(grid_path)
    df_pareto = pd.read_csv(pareto_path)
    selected = _find_candidate_row(df_grid, alpha, beta)
    is_in_pareto = not df_pareto[
        np.isclose(df_pareto["alpha"].astype(float), float(alpha))
        & np.isclose(df_pareto["beta"].astype(float), float(beta))
    ].empty
    if not is_in_pareto and not allow_non_pareto:
        raise ValueError(
            f"alpha={alpha}, beta={beta} is not in pareto_front_3d.csv. "
            "Use --allow-non-pareto to select it anyway."
        )

    if outdir.exists() and any(outdir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Selection outdir already exists and is not empty: {outdir}. "
            "Choose a new --outdir or pass --overwrite intentionally."
        )
    outdir.mkdir(parents=True, exist_ok=True)
    selected_df = pd.DataFrame([selected.to_dict()])
    selected_df.insert(0, "manual_selection_rule", "manual_pareto_candidate")
    selected_df.insert(1, "source_hpo_outdir", str(source_hpo_outdir))
    selected_df.insert(2, "is_in_source_pareto_front", bool(is_in_pareto))
    selected_df.to_csv(outdir / "selected_candidate_summary.csv", index=False, encoding="utf-8-sig")
    df_pareto.to_csv(outdir / "pareto_candidates.csv", index=False, encoding="utf-8-sig")

    _graph, vertice, dni, _edges, _data = load_ini_data_from_csv(FILE_PATHS)
    gt_lonlat = uploading_ground_truth(vertice, dni)
    anchor_labels = get_anchor_labels()
    test_labels = get_test_site_labels()
    anchor_lonlat = [tuple(gt_lonlat[dni[label]]) for label in anchor_labels]
    test_lonlat = [tuple(gt_lonlat[dni[label]]) for label in test_labels]

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
        outdir=outdir,
        selection_rule="manual_pareto_candidate",
    )

    selection_record = {
        "selection_rule": "manual_pareto_candidate",
        "source_hpo_outdir": str(source_hpo_outdir),
        "outdir": str(outdir),
        "alpha": float(alpha),
        "beta": float(beta),
        "is_in_source_pareto_front": bool(is_in_pareto),
        "seeds": list(map(int, seeds)),
        "anchor_labels": list(anchor_labels),
        "test_labels": list(test_labels),
        "selected_candidate_csv": "selected_candidate_summary.csv",
        "pareto_candidates_csv": "pareto_candidates.csv",
        "final_summary_json": "selected_final_summary.json",
    }
    (outdir / "candidate_selection.json").write_text(
        json.dumps(selection_record, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[Saved] {outdir / 'selected_candidate_summary.csv'}")
    print(f"[Saved] {outdir / 'pareto_candidates.csv'}")
    print(f"[Saved] {outdir / 'candidate_selection.json'}")
    print(f"[Saved] {outdir / 'selected_final_summary.json'}")
    return outdir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select a manual HPO candidate and rerun final model.")
    parser.add_argument("--source-hpo-outdir", required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--allow-non-pareto", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--w-dis", type=float, default=1.0)
    parser.add_argument("--base-spring", type=float, default=SPRING_STIFFNESS_BASE)
    parser.add_argument("--base-dir", type=float, default=DIRECTIONAL_FORCE_MAGNITUDE_BASE)
    parser.add_argument("--base-rep", type=float, default=REPULSION_STRENGTH_BASE)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    select_hpo_candidate(
        source_hpo_outdir=args.source_hpo_outdir,
        alpha=args.alpha,
        beta=args.beta,
        seeds=_parse_seed_list(args.seeds),
        outdir=args.outdir,
        w_dis=args.w_dis,
        base_spring_stiffness=args.base_spring,
        base_directional_force=args.base_dir,
        base_repulsion_strength=args.base_rep,
        refer_pos_sim=DEFAULT_REFER_POS_SIM,
        allow_non_pareto=args.allow_non_pareto,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
