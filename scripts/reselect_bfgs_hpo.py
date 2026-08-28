"""Recompute BFGS HPO Pareto selection from existing runs without rerunning BFGS."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from run_paper_script.ch5_hparam_kfold_gridsearch_pareto import (
    _non_dominated_mask,
    _plot_heatmap,
    _plot_pareto_2d_projections,
    _plot_pareto_3d,
    _select_one_se_balanced_candidate,
)
from run_paper_script.ch5_scipy_bfgs_hpo import OBJECTIVE_COLUMNS, _eligible_grid_points


def reselect(source_outdir: str | Path, outdir: str | Path) -> dict:
    source = Path(source_outdir)
    outdir = Path(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)

    required = (
        "bfgs_hpo_runs.csv",
        "bfgs_hpo_fold_summary.csv",
        "bfgs_hpo_grid_summary.csv",
        "bfgs_hpo_config.json",
    )
    for name in required:
        if not (source / name).exists():
            raise FileNotFoundError(f"Missing source HPO artifact: {source / name}")

    runs = pd.read_csv(source / "bfgs_hpo_runs.csv")
    folds = pd.read_csv(source / "bfgs_hpo_fold_summary.csv")
    grid = pd.read_csv(source / "bfgs_hpo_grid_summary.csv")
    config = json.loads((source / "bfgs_hpo_config.json").read_text(encoding="utf-8"))

    fold_min = (
        folds.groupby(["alpha", "beta"], as_index=False)["n_successful_runs"]
        .min()
        .rename(columns={"n_successful_runs": "min_successful_runs_per_fold"})
    )
    grid = grid.drop(
        columns=["success_rate", "min_successful_runs_per_fold", "all_folds_have_success"],
        errors="ignore",
    ).merge(fold_min, on=["alpha", "beta"], how="left", validate="one_to_one")
    grid["success_rate"] = grid["n_successful_runs"] / grid["n_expected_runs"]
    grid["all_folds_have_success"] = grid["min_successful_runs_per_fold"] > 0

    eligible = _eligible_grid_points(grid)
    if eligible.empty:
        raise RuntimeError("No grid point has a successful run in every LOO fold.")
    pareto_mask = _non_dominated_mask(eligible[list(OBJECTIVE_COLUMNS)].to_numpy(float))
    eligible["is_pareto"] = pareto_mask
    pareto = eligible[eligible["is_pareto"]].copy()
    selected, selection_meta = _select_one_se_balanced_candidate(pareto, OBJECTIVE_COLUMNS)

    alpha_min, alpha_max, _alpha_step = map(float, config["alpha_range"])
    beta_min, beta_max, _beta_step = map(float, config["beta_range"])
    boundary_meta = {
        "selected_on_alpha_boundary": bool(
            np.isclose(float(selected["alpha"]), alpha_min)
            or np.isclose(float(selected["alpha"]), alpha_max)
        ),
        "selected_on_beta_boundary": bool(
            np.isclose(float(selected["beta"]), beta_min)
            or np.isclose(float(selected["beta"]), beta_max)
        ),
    }
    boundary_meta["selected_on_grid_boundary"] = bool(
        boundary_meta["selected_on_alpha_boundary"]
        or boundary_meta["selected_on_beta_boundary"]
    )
    boundary_meta["boundary_action"] = (
        "expand_grid_before_formal_run"
        if boundary_meta["selected_on_grid_boundary"]
        else "none"
    )
    selected_frame = pd.DataFrame(
        [{**selected.to_dict(), **selection_meta, **boundary_meta}]
    )

    shutil.copy2(source / "bfgs_hpo_runs.csv", outdir / "bfgs_hpo_runs.csv")
    shutil.copy2(
        source / "bfgs_hpo_fold_summary.csv", outdir / "bfgs_hpo_fold_summary.csv"
    )
    grid.to_csv(outdir / "bfgs_hpo_grid_summary.csv", index=False, encoding="utf-8-sig")
    pareto.to_csv(outdir / "bfgs_hpo_pareto_front.csv", index=False, encoding="utf-8-sig")
    selected_frame.to_csv(
        outdir / "bfgs_hpo_selected_candidate.csv", index=False, encoding="utf-8-sig"
    )

    _plot_heatmap(
        eligible,
        "RMSE_anchor_LOO_mean_km",
        outdir / "sensitivity_rmse_anchor_loo.png",
        "BFGS HPO: Anchor LOO RMSE",
        selected,
    )
    _plot_heatmap(
        eligible,
        "E_distance_stress_mean",
        outdir / "sensitivity_stress.png",
        "BFGS HPO: Stress",
        selected,
    )
    _plot_heatmap(
        eligible,
        "E_direction_vr_mean",
        outdir / "sensitivity_violation_rate.png",
        "BFGS HPO: Violation Rate",
        selected,
    )
    _plot_pareto_3d(eligible, pareto_mask, outdir / "pareto_front_3d.png", selected)
    _plot_pareto_2d_projections(
        eligible, pareto_mask, outdir / "pareto_front_2d.png", selected
    )

    config.update(
        {
            "derived_from_hpo_outdir": str(source),
            "selection_population_policy": (
                "include grid points with finite objectives and at least one successful "
                "run in every anchor LOO fold; retain failure counts"
            ),
            "eligible_grid_count": int(len(eligible)),
            "complete_grid_count": int(grid["is_complete"].sum()),
            "raw_runs_reused_without_rerun": True,
        }
    )
    (outdir / "bfgs_hpo_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (outdir / "bfgs_hpo_selection.json").write_text(
        json.dumps(
            {
                "alpha": float(selected["alpha"]),
                "beta": float(selected["beta"]),
                "w_dis": float(selected["w_dis"]),
                **selection_meta,
                **boundary_meta,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "source": str(source),
        "outdir": str(outdir),
        "eligible_grid_count": int(len(eligible)),
        "complete_grid_count": int(grid["is_complete"].sum()),
        "pareto_count": int(len(pareto)),
        "selected_alpha": float(selected["alpha"]),
        "selected_beta": float(selected["beta"]),
        "selected_success_rate": float(selected["success_rate"]),
        **boundary_meta,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-outdir", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    print(json.dumps(reselect(args.source_outdir, args.outdir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
