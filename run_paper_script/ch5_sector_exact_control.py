"""Sector-based versus exact-direction PhysicsSim control experiment.

The two formulations share the same data, anchors, initialization seeds,
integration settings, HPO grid, validation protocol, and evaluation metrics.
Only the directional objective used by PhysicsSim is changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from library.config import (
    DIRECTIONAL_FORCE_MAGNITUDE_BASE,
    FILE_PATHS,
    PROJECT_ROOT,
    REPULSION_STRENGTH_BASE,
    SPRING_STIFFNESS_BASE,
    refer_pos_sim as DEFAULT_REFER_POS_SIM,
)
from library.data_io import load_ini_data_from_csv
from library.directional_objectives import DIRECTIONAL_OBJECTIVE_EXACT
from run_paper_script.ch5_ablation_progressive import _layout_metrics
from run_paper_script.ch5_ablation_study import _bootstrap_ci_mean
from run_paper_script.ch5_hparam_kfold_gridsearch_pareto import (
    _make_alpha_beta_grid,
    run_anchor_loo_gridsearch_pareto,
)


FORMAL_HPO_SEEDS = tuple(range(10))
FORMAL_FINAL_SEEDS = tuple(range(100))
FORMAL_GRID = {
    "alpha_min": -1.0,
    "alpha_max": 1.5,
    "alpha_step": 0.5,
    "beta_min": -2.0,
    "beta_max": 0.5,
    "beta_step": 0.5,
}
DEFAULT_SECTOR_RUNS = (
    PROJECT_ROOT
    / "outputs"
    / "ch5_progressive_as_physics_alpha_1_beta_-0.5_dc_alpha_-2_wang_current_100seeds_random1000_20260721"
    / "progressive_runs_by_seed.csv"
)

SECTOR_VARIANT = "PhysicsSim-Full"
EXACT_VARIANT = "PhysicsSim-ExactDir"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260904

OFFICIAL_METRICS = (
    "RMSE_test_km",
    "E_distance_stress",
    "E_direction_vr",
    "E_direction_mae",
    "crowding_violation_rate_tau_0p1",
    "collapse_node_rate_tau_0p1",
    "nnd_q05_km",
    "distance_edge_crossing_rate",
)

METRIC_LABELS = {
    "RMSE_test_km": "RMSE (km)",
    "E_distance_stress": "Stress",
    "E_direction_vr": "Violation Rate",
    "E_direction_mae": "Mean Angular Error (rad)",
    "crowding_violation_rate_tau_0p1": "Crowding Violation Rate (tau=0.10)",
    "collapse_node_rate_tau_0p1": "Collapse Node Rate (tau=0.10)",
    "nnd_q05_km": "Nearest-Neighbor Distance, 5th Quantile (km)",
    "distance_edge_crossing_rate": "Crossing-edge Rate",
}


def _parse_seed_list(raw: str) -> list[int]:
    seeds = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not seeds:
        raise ValueError("Seed list cannot be empty.")
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"Seed list contains duplicates: {seeds}")
    return seeds


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_columns(df: pd.DataFrame, columns: Iterable[str], *, source: Path) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {source}: {missing}")


def _validate_formal_protocol(
    *,
    hpo_seeds: Sequence[int],
    final_seeds: Sequence[int],
    alpha_min: float,
    alpha_max: float,
    alpha_step: float,
    beta_min: float,
    beta_max: float,
    beta_step: float,
    allow_smoke: bool,
) -> tuple[np.ndarray, np.ndarray]:
    alphas, betas = _make_alpha_beta_grid(
        alpha_min, alpha_max, alpha_step, beta_min, beta_max, beta_step
    )
    if allow_smoke:
        return alphas, betas

    if tuple(hpo_seeds) != FORMAL_HPO_SEEDS:
        raise ValueError(f"Formal HPO requires seeds 0-9; received {list(hpo_seeds)}.")
    if tuple(final_seeds) != FORMAL_FINAL_SEEDS:
        raise ValueError(f"Formal evaluation requires seeds 0-99; received {list(final_seeds)}.")
    actual_grid = (alpha_min, alpha_max, alpha_step, beta_min, beta_max, beta_step)
    expected_grid = (
        FORMAL_GRID["alpha_min"], FORMAL_GRID["alpha_max"], FORMAL_GRID["alpha_step"],
        FORMAL_GRID["beta_min"], FORMAL_GRID["beta_max"], FORMAL_GRID["beta_step"],
    )
    if not np.allclose(actual_grid, expected_grid, rtol=0.0, atol=1e-12):
        raise ValueError(f"Formal HPO grid must be {expected_grid}; received {actual_grid}.")
    if len(alphas) * len(betas) != 36:
        raise AssertionError("Formal exact-direction grid must contain exactly 36 candidates.")
    return alphas, betas


def _load_sector_runs(path: str | Path, expected_seeds: Sequence[int], *, allow_smoke: bool) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Sector-based source runs not found: {source}")
    runs = pd.read_csv(source)
    _require_columns(runs, ("variant", "seed", "status", *OFFICIAL_METRICS), source=source)
    sector = runs[(runs["variant"] == SECTOR_VARIANT) & (runs["status"] == "ok")].copy()
    sector["seed"] = sector["seed"].astype(int)
    if sector["seed"].duplicated().any():
        duplicates = sector.loc[sector["seed"].duplicated(), "seed"].tolist()
        raise ValueError(f"Duplicate sector seeds in {source}: {duplicates}")
    expected = set(map(int, expected_seeds))
    available = set(sector["seed"].tolist())
    missing = sorted(expected - available)
    if missing:
        raise ValueError(f"Sector source is missing comparison seeds: {missing}")
    sector = sector[sector["seed"].isin(expected)].sort_values("seed").reset_index(drop=True)
    if not allow_smoke and sector["seed"].tolist() != list(FORMAL_FINAL_SEEDS):
        raise ValueError("Formal sector source must provide exactly successful seeds 0-99.")
    return sector


def preflight_report(
    *,
    hpo_seeds: Sequence[int],
    final_seeds: Sequence[int],
    alpha_min: float,
    alpha_max: float,
    alpha_step: float,
    beta_min: float,
    beta_max: float,
    beta_step: float,
    sector_runs: str | Path,
    allow_smoke: bool,
) -> dict:
    alphas, betas = _validate_formal_protocol(
        hpo_seeds=hpo_seeds,
        final_seeds=final_seeds,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        alpha_step=alpha_step,
        beta_min=beta_min,
        beta_max=beta_max,
        beta_step=beta_step,
        allow_smoke=allow_smoke,
    )
    for key in ("chen_data", "directional_data", "ground_truth_path"):
        if not Path(FILE_PATHS[key]).exists():
            raise FileNotFoundError(f"Required input is missing: {FILE_PATHS[key]}")
    sector = _load_sector_runs(sector_runs, final_seeds, allow_smoke=allow_smoke)
    return {
        "status": "passed",
        "protocol": "sector_based_vs_exact_direction_isolated_control",
        "directional_objective": DIRECTIONAL_OBJECTIVE_EXACT,
        "allow_smoke": bool(allow_smoke),
        "hpo_seeds": list(map(int, hpo_seeds)),
        "final_seeds": list(map(int, final_seeds)),
        "alpha_values": [float(value) for value in alphas],
        "beta_values": [float(value) for value in betas],
        "grid_candidates": int(len(alphas) * len(betas)),
        "hpo_model_runs": int(len(alphas) * len(betas) * 3 * len(hpo_seeds)),
        "final_model_runs": int(len(final_seeds)),
        "expected_total_model_runs": int(
            len(alphas) * len(betas) * 3 * len(hpo_seeds) + len(final_seeds)
        ),
        "sector_source": str(Path(sector_runs).resolve()),
        "sector_seed_count": int(len(sector)),
        "input_sha256": {
            "distance_edges": _sha256(FILE_PATHS["chen_data"]),
            "direction_edges": _sha256(FILE_PATHS["directional_data"]),
            "site_points": _sha256(FILE_PATHS["ground_truth_path"]),
            "sector_runs": _sha256(sector_runs),
        },
    }


def _attach_layout_metrics(hpo_outdir: Path) -> pd.DataFrame:
    runs_path = hpo_outdir / "selected_final_runs_by_seed.csv"
    positions_path = hpo_outdir / "selected_final_positions_y_up_sim.csv"
    runs = pd.read_csv(runs_path)
    positions = pd.read_csv(positions_path)
    _require_columns(
        runs,
        ("seed", "RMSE_final_test_km", "E_distance_stress", "E_direction_vr", "E_direction_mae"),
        source=runs_path,
    )
    _require_columns(
        positions,
        ("seed", "node_idx", "label", "x_y_up_sim", "y_y_up_sim"),
        source=positions_path,
    )
    _graph, vertice, _dni, _edges, distance_data = load_ini_data_from_csv(FILE_PATHS)
    layout_rows: list[dict] = []
    for seed, group in positions.groupby("seed", sort=True):
        ordered = group.sort_values("node_idx")
        if ordered["label"].tolist() != list(vertice):
            raise ValueError(f"Saved position labels/order do not match model nodes for seed {seed}.")
        points = ordered[["x_y_up_sim", "y_y_up_sim"]].to_numpy(float)
        layout_rows.append({"seed": int(seed), **_layout_metrics(points, vertice, _dni, distance_data)})
    layout = pd.DataFrame(layout_rows)
    exact = runs.merge(layout, on="seed", how="left", validate="one_to_one")
    exact = exact.rename(columns={"RMSE_final_test_km": "RMSE_test_km"})
    exact["seed"] = exact["seed"].astype(int)
    exact["variant"] = EXACT_VARIANT
    exact["formulation"] = "exact-direction angular penalty"
    exact["status"] = "ok"
    exact["error"] = ""
    _require_columns(exact, OFFICIAL_METRICS, source=runs_path)
    return exact.sort_values("seed").reset_index(drop=True)


def summarize_runs(combined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for variant, group in combined.groupby("variant", sort=False):
        for metric in OFFICIAL_METRICS:
            values = group[metric].to_numpy(float)
            rows.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "n": int(values.size),
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def paired_comparison(combined: pd.DataFrame) -> pd.DataFrame:
    sector = combined[combined["variant"] == SECTOR_VARIANT].set_index("seed")
    exact = combined[combined["variant"] == EXACT_VARIANT].set_index("seed")
    seeds = sorted(set(sector.index).intersection(exact.index))
    if not seeds:
        raise ValueError("No matched seeds are available for the paired comparison.")
    rows: list[dict] = []
    for metric_index, metric in enumerate(OFFICIAL_METRICS):
        differences = exact.loc[seeds, metric].to_numpy(float) - sector.loc[seeds, metric].to_numpy(float)
        lo, hi = _bootstrap_ci_mean(
            differences,
            n_boot=BOOTSTRAP_REPLICATES,
            seed=BOOTSTRAP_SEED + metric_index,
        )
        rows.append(
            {
                "comparison": "PhysicsSim-ExactDir minus PhysicsSim-Full",
                "left_variant": EXACT_VARIANT,
                "right_variant": SECTOR_VARIANT,
                "diff_definition": "exact_minus_sector",
                "metric": metric,
                "metric_label": METRIC_LABELS[metric],
                "n_pairs": int(len(seeds)),
                "paired_diff_mean": float(differences.mean()),
                "paired_diff_sd": float(differences.std(ddof=1)) if differences.size > 1 else 0.0,
                "paired_diff_ci95_lo": lo,
                "paired_diff_ci95_hi": hi,
                "ci_method": "paired percentile bootstrap of the mean",
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_seed": BOOTSTRAP_SEED + metric_index,
                "ci_excludes_zero": bool(lo > 0.0 or hi < 0.0),
            }
        )
    return pd.DataFrame(rows)


def _plot_metric_panels(summary: pd.DataFrame, outdir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {SECTOR_VARIANT: "#0072B2", EXACT_VARIANT: "#E69F00"}
    groups = (
        (OFFICIAL_METRICS[:4], "sector_exact_primary_metrics.png"),
        (OFFICIAL_METRICS[4:], "sector_exact_layout_metrics.png"),
    )
    for metrics, filename in groups:
        fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.2))
        for ax, metric in zip(axes, metrics):
            subset = summary[summary["metric"] == metric].set_index("variant")
            variants = [SECTOR_VARIANT, EXACT_VARIANT]
            means = [float(subset.loc[name, "mean"]) for name in variants]
            sds = [float(subset.loc[name, "sd"]) for name in variants]
            ax.bar(
                range(2), means, yerr=sds, capsize=4,
                color=[colors[name] for name in variants], edgecolor="black", linewidth=0.6,
            )
            ax.set_xticks(range(2), ["Sector", "Exact"])
            ax.set_title(METRIC_LABELS[metric], fontsize=10)
            ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(outdir / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)


def run_sector_exact_control(
    *,
    hpo_seeds: Sequence[int],
    final_seeds: Sequence[int],
    alpha_min: float,
    alpha_max: float,
    alpha_step: float,
    beta_min: float,
    beta_max: float,
    beta_step: float,
    outdir: str | Path,
    sector_runs: str | Path = DEFAULT_SECTOR_RUNS,
    w_dis: float = 1.0,
    base_spring: float = SPRING_STIFFNESS_BASE,
    base_direction: float = DIRECTIONAL_FORCE_MAGNITUDE_BASE,
    base_repulsion: float = REPULSION_STRENGTH_BASE,
    allow_smoke: bool = False,
    overwrite: bool = False,
    generate_plots: bool = True,
) -> dict:
    report = preflight_report(
        hpo_seeds=hpo_seeds,
        final_seeds=final_seeds,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        alpha_step=alpha_step,
        beta_min=beta_min,
        beta_max=beta_max,
        beta_step=beta_step,
        sector_runs=sector_runs,
        allow_smoke=allow_smoke,
    )
    outdir_path = Path(outdir)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    hpo = run_anchor_loo_gridsearch_pareto(
        seeds=hpo_seeds,
        final_seeds=final_seeds,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        alpha_step=alpha_step,
        beta_min=beta_min,
        beta_max=beta_max,
        beta_step=beta_step,
        w_dis=w_dis,
        base_spring_stiffness=base_spring,
        base_directional_force=base_direction,
        base_repulsion_strength=base_repulsion,
        refer_pos_sim=DEFAULT_REFER_POS_SIM,
        outdir=outdir_path,
        overwrite=overwrite,
        generate_plots=generate_plots,
        save_final_positions=True,
        directional_objective=DIRECTIONAL_OBJECTIVE_EXACT,
        fail_on_selected_boundary=not allow_smoke,
    )
    report["selected_alpha"] = float(hpo["selected"]["alpha"])
    report["selected_beta"] = float(hpo["selected"]["beta"])
    report["selection_meta"] = hpo["selection_meta"]
    (outdir_path / "sector_exact_preflight_and_protocol.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    exact = _attach_layout_metrics(outdir_path)
    sector = _load_sector_runs(sector_runs, final_seeds, allow_smoke=allow_smoke).copy()
    sector["formulation"] = "sector-based squared-hinge penalty"
    keep_columns = [
        "variant", "formulation", "seed", "status", "error", *OFFICIAL_METRICS
    ]
    exact.to_csv(outdir_path / "exact_direction_final_runs_by_seed.csv", index=False, encoding="utf-8-sig")
    combined = pd.concat(
        [sector[keep_columns], exact[keep_columns]], ignore_index=True
    ).sort_values(["seed", "variant"]).reset_index(drop=True)
    summary = summarize_runs(combined)
    paired = paired_comparison(combined)
    combined.to_csv(outdir_path / "sector_exact_runs_by_seed.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(outdir_path / "sector_exact_summary.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(outdir_path / "sector_exact_paired_comparison.csv", index=False, encoding="utf-8-sig")
    if generate_plots:
        _plot_metric_panels(summary, outdir_path)
    print(f"[Saved] Sector/exact control outputs: {outdir_path}")
    return {
        "outdir": outdir_path,
        "preflight": report,
        "exact_runs": exact,
        "combined_runs": combined,
        "summary": summary,
        "paired": paired,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the isolated PhysicsSim sector-based versus exact-direction control."
    )
    parser.add_argument("--seeds", default=",".join(map(str, FORMAL_HPO_SEEDS)))
    parser.add_argument("--final-seeds", default=",".join(map(str, FORMAL_FINAL_SEEDS)))
    parser.add_argument("--alpha-min", type=float, default=FORMAL_GRID["alpha_min"])
    parser.add_argument("--alpha-max", type=float, default=FORMAL_GRID["alpha_max"])
    parser.add_argument("--alpha-step", type=float, default=FORMAL_GRID["alpha_step"])
    parser.add_argument("--beta-min", type=float, default=FORMAL_GRID["beta_min"])
    parser.add_argument("--beta-max", type=float, default=FORMAL_GRID["beta_max"])
    parser.add_argument("--beta-step", type=float, default=FORMAL_GRID["beta_step"])
    parser.add_argument("--w-dis", type=float, default=1.0)
    parser.add_argument("--base-spring", type=float, default=SPRING_STIFFNESS_BASE)
    parser.add_argument("--base-dir", type=float, default=DIRECTIONAL_FORCE_MAGNITUDE_BASE)
    parser.add_argument("--base-rep", type=float, default=REPULSION_STRENGTH_BASE)
    parser.add_argument("--sector-runs", default=str(DEFAULT_SECTOR_RUNS))
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--allow-smoke",
        action="store_true",
        help="Allow reduced seeds/grid and a boundary selection for development only.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_sector_exact_control(
        hpo_seeds=_parse_seed_list(args.seeds),
        final_seeds=_parse_seed_list(args.final_seeds),
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        alpha_step=args.alpha_step,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        beta_step=args.beta_step,
        outdir=args.outdir,
        sector_runs=args.sector_runs,
        w_dis=args.w_dis,
        base_spring=args.base_spring,
        base_direction=args.base_dir,
        base_repulsion=args.base_rep,
        allow_smoke=args.allow_smoke,
        overwrite=args.overwrite,
        generate_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
