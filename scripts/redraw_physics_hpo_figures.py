"""Redraw PhysicsSim HPO figures and mark the manually selected candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_paper_script.ch5_hparam_kfold_gridsearch_pareto import (
    _plot_heatmap,
    _plot_pareto_2d_projections,
    _plot_pareto_3d,
)


def redraw_physics_hpo_figures(*, hpo_outdir: str | Path, selected_outdir: str | Path, outdir: str | Path | None = None) -> dict[str, Path]:
    hpo_dir = Path(hpo_outdir)
    selected_dir = Path(selected_outdir)
    destination = Path(outdir) if outdir else hpo_dir
    grid_path = hpo_dir / "grid_summary_cv.csv"
    selected_path = selected_dir / "selected_candidate_summary.csv"
    if not grid_path.exists() or not selected_path.exists():
        raise FileNotFoundError("Expected grid_summary_cv.csv and selected_candidate_summary.csv")
    grid = pd.read_csv(grid_path)
    selected = pd.read_csv(selected_path)
    if len(selected) != 1:
        raise ValueError("Expected exactly one selected PhysicsSim HPO candidate.")
    selected_row = selected.iloc[0]
    pareto_mask = grid["is_pareto"].to_numpy(bool)
    selected_is_pareto = bool(
        ((np.isclose(grid["alpha"], float(selected_row["alpha"])))
         & (np.isclose(grid["beta"], float(selected_row["beta"])))
         & grid["is_pareto"])
        .any()
    )
    if not selected_is_pareto:
        raise ValueError("The selected candidate is not a Pareto-front point in this HPO output.")
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "stress_heatmap": destination / "heatmap_kruskal_stress.png",
        "direction_heatmap": destination / "heatmap_direction_violation.png",
        "rmse_heatmap": destination / "heatmap_rmse_anchor_loo.png",
        "pareto_3d": destination / "pareto_front_3d.png",
        "pareto_2d": destination / "pareto_front_2d_projections.png",
    }
    _plot_heatmap(grid, "E_distance_stress_mean", paths["stress_heatmap"], "Distance stress", selected_row)
    _plot_heatmap(grid, "E_direction_vr_mean", paths["direction_heatmap"], "Direction violation rate", selected_row)
    _plot_heatmap(grid, "RMSE_anchor_LOO_mean_km", paths["rmse_heatmap"], "Anchor LOO RMSE (km)", selected_row)
    _plot_pareto_3d(grid, pareto_mask, paths["pareto_3d"], selected_row)
    _plot_pareto_2d_projections(grid, pareto_mask, paths["pareto_2d"], selected_row)
    (destination / "hpo_figure_selection_metadata.json").write_text(
        json.dumps(
            {
                "selected_alpha": float(selected_row["alpha"]),
                "selected_beta": float(selected_row["beta"]),
                "selected_is_pareto": selected_is_pareto,
                "marker": "red star with black outline",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Redraw PhysicsSim HPO figures with the selected candidate marked.")
    parser.add_argument("--hpo-outdir", required=True)
    parser.add_argument("--selected-outdir", required=True)
    parser.add_argument("--outdir", default="")
    args = parser.parse_args()
    paths = redraw_physics_hpo_figures(
        hpo_outdir=args.hpo_outdir,
        selected_outdir=args.selected_outdir,
        outdir=args.outdir or None,
    )
    for label, path in paths.items():
        print(f"[Saved] {label}: {path}")


if __name__ == "__main__":
    main()
