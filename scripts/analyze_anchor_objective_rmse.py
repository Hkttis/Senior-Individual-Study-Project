"""Compare anchor-robustness RMSE against exact manuscript objective values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from library.config import FILE_PATHS, refer_pos_sim
from library.data_io import load_ini_data_from_csv, uploading_ground_truth
from library.scipy_objective import FixedAnchorObjective, ObjectiveWeights, build_current_objective
from run_paper_script.ch5_ablation_progressive import _target_positions_sim
from scripts.visualize_anchor_robustness_overlays import _overlay_rmse


DEFAULT_SOURCE = "outputs/ch5_anchor_split_robustness_formal_45splits_hpo3_final10_20260824"


def _split_problem(base: FixedAnchorObjective, anchors: dict[int, np.ndarray], weights: ObjectiveWeights) -> FixedAnchorObjective:
    return FixedAnchorObjective(
        vertices=base.vertices,
        distance_pairs=base.distance_pairs,
        distance_targets=base.distance_targets,
        direction_pairs=base.direction_pairs,
        direction_vectors=base.direction_vectors,
        direction_half_widths=base.direction_half_widths,
        anchor_positions=anchors,
        weights=weights,
        epsilon=base.epsilon,
        singularity_tolerance=base.singularity_tolerance,
    )


def _components(problem: FixedAnchorObjective, positions: np.ndarray) -> tuple[object, float]:
    centered = np.asarray(positions, dtype=float) - np.asarray(refer_pos_sim, dtype=float)
    drift = centered[problem.anchor_indices] - problem.anchor_coordinates
    max_drift = float(np.max(np.abs(drift))) if drift.size else 0.0
    centered = centered.copy()
    centered[problem.anchor_indices] = problem.anchor_coordinates
    return problem.components(problem.pack(centered)), max_drift


def _correlation(x: pd.Series, y: pd.Series) -> dict:
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return {"n": int(len(x)), "spearman_rho": None, "spearman_pvalue": None}
    result = spearmanr(x.to_numpy(float), y.to_numpy(float))
    return {
        "n": int(len(x)),
        "spearman_rho": float(result.statistic),
        "spearman_pvalue": float(result.pvalue),
    }


def _slug(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _scatter(frame: pd.DataFrame, x_column: str, title: str, outpath: Path, *, color_column: str) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 7.2))
    values = sorted(frame[color_column].unique())
    palette = plt.get_cmap("viridis")
    for i, value in enumerate(values):
        subset = frame[frame[color_column] == value]
        color = palette(i / max(len(values) - 1, 1))
        ax.scatter(
            subset[x_column],
            subset["RMSE_final_test_km"],
            s=31,
            alpha=0.72,
            color=color,
            label=f"{color_column}={value:g} (n={len(subset)})",
            edgecolors="white",
            linewidths=0.3,
        )
    stats = _correlation(frame[x_column], frame["RMSE_final_test_km"])
    subtitle = f"Spearman rho={stats['spearman_rho']:.3f}, p={stats['spearman_pvalue']:.3g}" if stats["spearman_rho"] is not None else ""
    ax.set_title(f"{title}\n{subtitle}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Objective function F(X)", fontsize=11)
    ax.set_ylabel("Held-out test RMSE (km)", fontsize=11)
    ax.grid(alpha=0.22)
    if len(values) <= 8:
        ax.legend(title="Selected direction exponent", loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath.with_suffix(".png"), dpi=240)
    fig.savefig(outpath.with_suffix(".svg"))
    plt.close(fig)


def analyze_anchor_objective_rmse(source: Path, outdir: Path, reference_alpha: float, reference_beta: float) -> dict:
    split_summary = pd.read_csv(source / "anchor_split_summary.csv").set_index("split_id")
    _, vertices, dni, _, _ = load_ini_data_from_csv(FILE_PATHS)
    gt_lonlat = uploading_ground_truth(vertices, dni)
    base = build_current_objective()
    if list(base.vertices) != list(vertices):
        raise ValueError("Objective vertex order differs from the formal anchor experiment.")
    reference_weights = ObjectiveWeights.from_physics_hpo(alpha=reference_alpha, beta=reference_beta)
    rows = []

    for split_id, split_row in split_summary.iterrows():
        split_dir = source / "splits" / split_id
        config = json.loads((split_dir / "gridsearch_config.json").read_text(encoding="utf-8"))
        targets = _target_positions_sim(dni, gt_lonlat, str(config["final_frame_anchor_label"]), refer_pos_sim)
        anchors = {
            dni[label]: np.asarray(targets[label], dtype=float) - np.asarray(refer_pos_sim, dtype=float)
            for label in config["anchor_labels"]
        }
        alpha, beta = float(split_row["selected_alpha"]), float(split_row["selected_beta"])
        native = _split_problem(base, anchors, ObjectiveWeights.from_physics_hpo(alpha=alpha, beta=beta))
        reference = _split_problem(base, anchors, reference_weights)
        runs = pd.read_csv(split_dir / "selected_final_runs_by_seed.csv")
        all_positions = pd.read_csv(split_dir / "selected_final_positions_y_up_sim.csv")
        for run in runs.itertuples(index=False):
            selected = all_positions[all_positions["seed"] == int(run.seed)].set_index("label")
            if len(selected) != len(vertices) or set(selected.index) != set(vertices):
                raise ValueError(f"Incomplete saved positions for {split_id} seed={run.seed}.")
            points = selected.loc[vertices, ["x_y_up_sim", "y_y_up_sim"]].to_numpy(float)
            recomputed_rmse = _overlay_rmse(points, targets, list(config["test_labels"]), dni)
            if not np.isclose(recomputed_rmse, float(run.RMSE_final_test_km), rtol=1e-9, atol=1e-8):
                raise ValueError(f"Formal RMSE differs from saved positions for {split_id} seed={run.seed}.")
            own, drift = _components(native, points)
            common, _ = _components(reference, points)
            rows.append(
                {
                    "split_id": split_id,
                    "seed": int(run.seed),
                    "alpha": alpha,
                    "beta": beta,
                    "selected_on_grid_boundary": bool(split_row["selected_on_grid_boundary"]),
                    "RMSE_final_test_km": recomputed_rmse,
                    "stress": float(run.E_distance_stress),
                    "violation_rate": float(run.E_direction_vr),
                    "native_objective_total": own.total,
                    "native_objective_distance": own.weighted_distance,
                    "native_objective_direction": own.weighted_direction,
                    "native_objective_repulsion": own.weighted_repulsion,
                    "reference_objective_total": common.total,
                    "reference_objective_distance": common.weighted_distance,
                    "reference_objective_direction": common.weighted_direction,
                    "reference_objective_repulsion": common.weighted_repulsion,
                    "anchor_drift_sim": drift,
                }
            )

    frame = pd.DataFrame(rows)
    if len(frame) != len(split_summary) * 10:
        raise ValueError(f"Unexpected objective row count: {len(frame)}")
    outdir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(outdir / "anchor_objective_rmse_by_run.csv", index=False, encoding="utf-8-sig")
    _scatter(
        frame,
        "reference_objective_total",
        f"Common objective weights: alpha={reference_alpha:g}, beta={reference_beta:g}",
        outdir / "objective_vs_rmse_common_reference_weights",
        color_column="alpha",
    )

    group_rows = []
    for (alpha, beta), group in frame.groupby(["alpha", "beta"]):
        stats = _correlation(group["native_objective_total"], group["RMSE_final_test_km"])
        group_rows.append({"alpha": float(alpha), "beta": float(beta), **stats})
        if len(group) >= 20:
            _scatter(
                group,
                "native_objective_total",
                f"Selected weights: alpha={alpha:g}, beta={beta:g}",
                outdir / f"objective_vs_rmse_alpha_{_slug(alpha)}_beta_{_slug(beta)}",
                color_column="alpha",
            )
    pd.DataFrame(group_rows).to_csv(outdir / "objective_rmse_correlation_by_hyperparameter.csv", index=False, encoding="utf-8-sig")

    split_means = frame.groupby("split_id", as_index=False).agg(
        reference_objective_mean=("reference_objective_total", "mean"),
        native_objective_mean=("native_objective_total", "mean"),
        rmse_mean=("RMSE_final_test_km", "mean"),
    )
    payload = {
        "source": str(source.resolve()),
        "reference_alpha": reference_alpha,
        "reference_beta": reference_beta,
        "n_splits": int(len(split_summary)),
        "n_runs": int(len(frame)),
        "common_reference_run_correlation": _correlation(frame["reference_objective_total"], frame["RMSE_final_test_km"]),
        "common_reference_split_mean_correlation": _correlation(split_means["reference_objective_mean"], split_means["rmse_mean"]),
        "max_anchor_drift_sim": float(frame["anchor_drift_sim"].max()),
        "policy": "Cross-hyperparameter comparisons use common reference weights; native objectives are compared only within identical alpha/beta groups.",
    }
    (outdir / "objective_rmse_analysis_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--outdir", default="outputs/ch6_anchor_objective_rmse_diagnostics_20260825")
    parser.add_argument("--reference-alpha", type=float, default=1.0)
    parser.add_argument("--reference-beta", type=float, default=-0.5)
    args = parser.parse_args()
    analyze_anchor_objective_rmse(Path(args.source), Path(args.outdir), args.reference_alpha, args.reference_beta)


if __name__ == "__main__":
    main()
