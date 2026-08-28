"""Select and reproduce representative minima from a formal SciPy-BFGS run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from library.scipy_diagnostics import assign_objective_strata
from run_paper_script.ch5_scipy_bfgs import run_experiment


REPRESENTATIVE_METRICS = (
    "objective_final",
    "RMSE_test_km",
    "E_distance_stress",
    "E_direction_vr",
    "E_direction_mae",
)
VERIFY_METRICS = REPRESENTATIVE_METRICS + ("gradient_norm_inf",)


def select_representative_minima(runs: pd.DataFrame, n_strata: int = 4) -> tuple[pd.DataFrame, list[float]]:
    """Choose the MAD-standardized multimetric medoid within each objective stratum."""

    required = {"seed", "status", *REPRESENTATIVE_METRICS}
    missing = required.difference(runs.columns)
    if missing:
        raise ValueError(f"BFGS runs are missing required columns: {sorted(missing)}")
    ok = runs.loc[runs["status"] == "ok"].copy()
    ok = ok[np.isfinite(ok[list(REPRESENTATIVE_METRICS)].to_numpy(float)).all(axis=1)]
    if len(ok) < n_strata:
        raise ValueError(f"Need at least {n_strata} successful finite BFGS runs.")
    assignments, thresholds = assign_objective_strata(
        ok["objective_final"].to_numpy(float),
        ok["objective_final"].to_numpy(float),
        n_strata,
    )
    ok["objective_stratum"] = assignments
    selected_rows: list[dict] = []
    for stratum, group in ok.groupby("objective_stratum", sort=True):
        values = group.loc[:, REPRESENTATIVE_METRICS].to_numpy(float)
        median = np.median(values, axis=0)
        mad = np.median(np.abs(values - median), axis=0)
        scale = np.where(mad > 0.0, mad, 1.0)
        distance = np.sqrt(np.sum(((values - median) / scale) ** 2, axis=1))
        candidates = group.assign(representative_distance=distance).sort_values(
            ["representative_distance", "seed"], kind="stable"
        )
        selected = candidates.iloc[0]
        row = {
            "objective_stratum": int(stratum),
            "stratum_n": int(len(group)),
            "seed": int(selected["seed"]),
            "representative_distance": float(selected["representative_distance"]),
        }
        row.update({metric: float(selected[metric]) for metric in VERIFY_METRICS})
        selected_rows.append(row)
    return pd.DataFrame(selected_rows), thresholds


def verify_reproduced_minima(
    source_runs: pd.DataFrame,
    source_positions: pd.DataFrame,
    reproduced_runs: pd.DataFrame,
    reproduced_positions: pd.DataFrame,
    selected_seeds: Sequence[int],
    *,
    atol: float = 1e-7,
) -> pd.DataFrame:
    """Numerically compare rerun metrics and coordinates with the formal source."""

    rows = []
    for seed in selected_seeds:
        source_row = source_runs.loc[source_runs["seed"] == seed]
        rerun_row = reproduced_runs.loc[reproduced_runs["seed"] == seed]
        if len(source_row) != 1 or len(rerun_row) != 1:
            raise ValueError(f"Expected exactly one source and rerun row for seed {seed}.")
        metric_diff = max(
            abs(float(source_row.iloc[0][metric]) - float(rerun_row.iloc[0][metric]))
            for metric in VERIFY_METRICS
        )
        source_pos = source_positions.loc[source_positions["seed"] == seed].set_index("label")
        rerun_pos = reproduced_positions.loc[reproduced_positions["seed"] == seed].set_index("label")
        if set(source_pos.index) != set(rerun_pos.index):
            raise ValueError(f"Position labels differ for seed {seed}.")
        rerun_pos = rerun_pos.loc[source_pos.index]
        position_diff = float(
            np.max(
                np.abs(
                    source_pos[["x_y_up_sim", "y_y_up_sim"]].to_numpy(float)
                    - rerun_pos[["x_y_up_sim", "y_y_up_sim"]].to_numpy(float)
                )
            )
        )
        rows.append(
            {
                "seed": int(seed),
                "max_abs_metric_difference": float(metric_diff),
                "max_abs_position_difference_sim": position_diff,
                "verified": bool(metric_diff <= atol and position_diff <= atol),
            }
        )
    report = pd.DataFrame(rows)
    if not report["verified"].all():
        raise ValueError("At least one representative BFGS rerun differs from its formal source.")
    return report


def _plot_objective_rmse(
    runs: pd.DataFrame,
    representatives: pd.DataFrame,
    thresholds: Sequence[float],
    outdir: Path,
) -> None:
    ok = runs.loc[runs["status"] == "ok"].copy()
    strata, _ = assign_objective_strata(
        ok["objective_final"].to_numpy(float),
        ok["objective_final"].to_numpy(float),
        len(thresholds) + 1,
    )
    ok["objective_stratum"] = strata
    fig, axis = plt.subplots(figsize=(8.8, 6.4), constrained_layout=True)
    scatter = axis.scatter(
        ok["objective_final"],
        ok["RMSE_test_km"],
        c=ok["objective_stratum"],
        cmap="viridis",
        s=38,
        alpha=0.72,
        edgecolors="none",
    )
    axis.scatter(
        representatives["objective_final"],
        representatives["RMSE_test_km"],
        marker="*",
        s=190,
        c="#d62728",
        edgecolors="black",
        linewidths=0.7,
        label="Representative minimum",
        zorder=5,
    )
    for row in representatives.itertuples(index=False):
        axis.annotate(
            f"seed {row.seed}",
            (row.objective_final, row.RMSE_test_km),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )
    axis.set_xscale("symlog", linthresh=1e5)
    axis.set_xlabel("Final objective value")
    axis.set_ylabel("Held-out test RMSE (km; post-hoc only)")
    axis.set_title("HPO-selected SciPy-BFGS: Objective versus External RMSE")
    axis.grid(alpha=0.22)
    axis.legend(loc="best")
    colorbar = fig.colorbar(scatter, ax=axis, pad=0.02)
    colorbar.set_label("Objective stratum")
    fig.savefig(outdir / "bfgs_objective_vs_test_rmse.svg")
    fig.savefig(outdir / "bfgs_objective_vs_test_rmse.png", dpi=300)
    plt.close(fig)


def export_representative_minima(
    *,
    source_outdir: str | Path,
    outdir: str | Path,
    n_strata: int = 4,
    make_gif: bool = False,
) -> dict:
    source_outdir = Path(source_outdir)
    outdir = Path(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    config = json.loads(
        (source_outdir / "bfgs_experiment_config.json").read_text(encoding="utf-8")
    )
    runs = pd.read_csv(source_outdir / "bfgs_runs_by_seed.csv")
    positions = pd.read_csv(source_outdir / "bfgs_final_positions_y_up_sim.csv")
    representatives, thresholds = select_representative_minima(runs, n_strata=n_strata)
    seeds = representatives["seed"].astype(int).tolist()
    rerun_dir = outdir / "representative_reruns"
    reproduced = run_experiment(
        seeds=seeds,
        outdir=rerun_dir,
        gtol=float(config["gtol"]),
        maxiter=int(config["maxiter"]),
        visualize_seeds=seeds,
        make_gif=make_gif,
        alpha=float(config["alpha"]),
        beta=float(config["beta"]),
        w_dis=float(config["w_dis"]),
        hpo_source=config.get("hpo_source"),
    )
    verification = verify_reproduced_minima(
        runs,
        positions,
        reproduced["runs"],
        reproduced["positions"],
        seeds,
    )
    representatives.to_csv(
        outdir / "bfgs_representative_minima.csv", index=False, encoding="utf-8-sig"
    )
    verification.to_csv(
        outdir / "bfgs_representative_rerun_verification.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _plot_objective_rmse(runs, representatives, thresholds, outdir)
    review_config = {
        "source_outdir": str(source_outdir),
        "selection_rule": (
            "largest-gap objective strata; within each stratum choose the run with the "
            "smallest Euclidean distance to the metric-wise median after MAD standardization"
        ),
        "selection_metrics": list(REPRESENTATIVE_METRICS),
        "n_strata": int(n_strata),
        "objective_stratum_thresholds": list(map(float, thresholds)),
        "representative_seeds": seeds,
        "rerun_policy": "same seed, weights, gtol and maxiter as the formal source",
        "rerun_verified": bool(verification["verified"].all()),
    }
    (outdir / "bfgs_representative_review_config.json").write_text(
        json.dumps(review_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "representatives": representatives,
        "verification": verification,
        "config": review_config,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-outdir", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--n-strata", type=int, default=4)
    parser.add_argument("--make-gif", action="store_true")
    args = parser.parse_args()
    result = export_representative_minima(
        source_outdir=args.source_outdir,
        outdir=args.outdir,
        n_strata=args.n_strata,
        make_gif=args.make_gif,
    )
    print(result["representatives"].to_string(index=False))
    print(f"[Saved] {Path(args.outdir)}")


if __name__ == "__main__":
    main()
