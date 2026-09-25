"""Sector versus softened-exact PhysicsSim-Full with independent HPO.

This is a new experiment family.  It reuses the established HPO, evaluation,
and paired-bootstrap protocols while leaving both earlier Exact code and all
existing experiment outputs untouched.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pandas as pd

from library.config import (
    DIRECTIONAL_FORCE_MAGNITUDE_BASE,
    FILE_PATHS,
    MIN_DISTANCE_BASE,
    REPULSION_STRENGTH_BASE,
    SPRING_STIFFNESS_BASE,
    refer_pos_sim as DEFAULT_REFER_POS_SIM,
)
from library.directional_objectives import DIRECTIONAL_OBJECTIVE_SOFT_EXACT
from run_paper_script.ch5_hparam_kfold_gridsearch_pareto import (
    _make_alpha_beta_grid,
    run_anchor_loo_gridsearch_pareto,
)
from run_paper_script.ch5_sector_exact_control import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    DEFAULT_SECTOR_RUNS,
    FORMAL_FINAL_SEEDS,
    FORMAL_HPO_SEEDS,
    METRIC_LABELS,
    OFFICIAL_METRICS,
    SECTOR_VARIANT,
    _attach_layout_metrics,
    _bootstrap_ci_mean,
    _load_sector_runs,
    _parse_seed_list,
    _sha256,
    summarize_runs,
)


SOFT_EXACT_VARIANT = "PhysicsSim-SoftExactDir"
SOFT_EXACT_FORMAL_GRID = {
    "alpha_min": -1.0,
    "alpha_max": 1.5,
    "alpha_step": 0.5,
    "beta_min": -3.0,
    "beta_max": 0.5,
    "beta_step": 0.5,
}


def _validate_soft_exact_formal_protocol(
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
        raise ValueError(
            f"Formal evaluation requires seeds 0-99; received {list(final_seeds)}."
        )
    actual_grid = (alpha_min, alpha_max, alpha_step, beta_min, beta_max, beta_step)
    expected_grid = tuple(
        SOFT_EXACT_FORMAL_GRID[key]
        for key in (
            "alpha_min",
            "alpha_max",
            "alpha_step",
            "beta_min",
            "beta_max",
            "beta_step",
        )
    )
    if not np.allclose(actual_grid, expected_grid, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"Formal softened-exact HPO grid must be {expected_grid}; received {actual_grid}."
        )
    if len(alphas) * len(betas) != 48:
        raise AssertionError("Formal softened-exact grid must contain exactly 48 candidates.")
    return alphas, betas


def _soft_exact_preflight_report(
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
    alphas, betas = _validate_soft_exact_formal_protocol(
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


def _plot_soft_metric_panels(summary: pd.DataFrame, outdir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {SECTOR_VARIANT: "#0072B2", SOFT_EXACT_VARIANT: "#E69F00"}
    groups = (
        (OFFICIAL_METRICS[:4], "sector_soft_exact_primary_metrics.png"),
        (OFFICIAL_METRICS[4:], "sector_soft_exact_layout_metrics.png"),
    )
    for metrics, filename in groups:
        fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.2))
        for ax, metric in zip(axes, metrics):
            subset = summary[summary["metric"] == metric].set_index("variant")
            variants = [SECTOR_VARIANT, SOFT_EXACT_VARIANT]
            means = [float(subset.loc[name, "mean"]) for name in variants]
            sds = [float(subset.loc[name, "sd"]) for name in variants]
            ax.bar(
                range(2),
                means,
                yerr=sds,
                capsize=4,
                color=[colors[name] for name in variants],
                edgecolor="black",
                linewidth=0.6,
            )
            ax.set_xticks(range(2), ["Sector", "Softened exact"])
            ax.set_title(METRIC_LABELS[metric], fontsize=10)
            ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(outdir / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)


def paired_comparison(combined: pd.DataFrame) -> pd.DataFrame:
    sector = combined[combined["variant"] == SECTOR_VARIANT].set_index("seed")
    soft = combined[combined["variant"] == SOFT_EXACT_VARIANT].set_index("seed")
    seeds = sorted(set(sector.index).intersection(soft.index))
    if not seeds:
        raise ValueError("No matched seeds are available for the paired comparison.")
    rows = []
    for metric_index, metric in enumerate(OFFICIAL_METRICS):
        differences = (
            soft.loc[seeds, metric].to_numpy(float)
            - sector.loc[seeds, metric].to_numpy(float)
        )
        lo, hi = _bootstrap_ci_mean(
            differences,
            n_boot=BOOTSTRAP_REPLICATES,
            seed=BOOTSTRAP_SEED + metric_index,
        )
        rows.append(
            {
                "comparison": f"{SOFT_EXACT_VARIANT} minus {SECTOR_VARIANT}",
                "left_variant": SOFT_EXACT_VARIANT,
                "right_variant": SECTOR_VARIANT,
                "diff_definition": "soft_exact_minus_sector",
                "metric": metric,
                "metric_label": METRIC_LABELS[metric],
                "n_pairs": len(seeds),
                "paired_diff_mean": float(differences.mean()),
                "paired_diff_sd": float(differences.std(ddof=1)) if len(seeds) > 1 else 0.0,
                "paired_diff_ci95_lo": lo,
                "paired_diff_ci95_hi": hi,
                "ci_method": "paired percentile bootstrap of the mean",
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_seed": BOOTSTRAP_SEED + metric_index,
                "ci_excludes_zero": bool(lo > 0.0 or hi < 0.0),
            }
        )
    return pd.DataFrame(rows)


def run_control(
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
    direction_softening_delta: float = MIN_DISTANCE_BASE,
    w_dis: float = 1.0,
    base_spring: float = SPRING_STIFFNESS_BASE,
    base_direction: float = DIRECTIONAL_FORCE_MAGNITUDE_BASE,
    base_repulsion: float = REPULSION_STRENGTH_BASE,
    allow_smoke: bool = False,
    overwrite: bool = False,
    generate_plots: bool = True,
) -> dict:
    _validate_soft_exact_formal_protocol(
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
    delta = float(direction_softening_delta)
    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError("direction_softening_delta must be finite and positive.")
    report = _soft_exact_preflight_report(
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
    report.update(
        {
            "protocol": "sector_vs_softened_exact_direction_full_control",
            "directional_objective": DIRECTIONAL_OBJECTIVE_SOFT_EXACT,
            "direction_softening_delta": delta,
            "directional_objective_formula": (
                "0.5*w_dir*sum((s^2*phi^2+delta^2*pi^2)/(s^2+delta^2))"
            ),
            "collision_extension": "energy=0.5*w_dir*pi^2; gradient=0",
            "legacy_exact_results_modified": False,
        }
    )
    out = Path(outdir)
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
        outdir=out,
        overwrite=overwrite,
        generate_plots=generate_plots,
        save_final_positions=True,
        directional_objective=DIRECTIONAL_OBJECTIVE_SOFT_EXACT,
        direction_softening_delta=delta,
        fail_on_selected_boundary=not allow_smoke,
    )
    report["selected_alpha"] = float(hpo["selected"]["alpha"])
    report["selected_beta"] = float(hpo["selected"]["beta"])
    report["selection_meta"] = hpo["selection_meta"]
    (out / "soft_exact_preflight_and_protocol.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    soft = _attach_layout_metrics(out)
    soft["variant"] = SOFT_EXACT_VARIANT
    soft["formulation"] = "softened exact-direction angular penalty"
    soft["direction_softening_delta"] = delta
    sector = _load_sector_runs(sector_runs, final_seeds, allow_smoke=allow_smoke).copy()
    sector["formulation"] = "sector-based squared-hinge penalty"
    sector["direction_softening_delta"] = np.nan
    keep = [
        "variant",
        "formulation",
        "direction_softening_delta",
        "seed",
        "status",
        "error",
        *OFFICIAL_METRICS,
    ]
    soft.to_csv(out / "soft_exact_direction_final_runs_by_seed.csv", index=False, encoding="utf-8-sig")
    combined = pd.concat([sector[keep], soft[keep]], ignore_index=True).sort_values(
        ["seed", "variant"]
    )
    summary = summarize_runs(combined)
    paired = paired_comparison(combined)
    combined.to_csv(out / "sector_soft_exact_runs_by_seed.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "sector_soft_exact_summary.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(out / "sector_soft_exact_paired_comparison.csv", index=False, encoding="utf-8-sig")
    if generate_plots:
        _plot_soft_metric_panels(summary, out)
    print(f"[Saved] Sector/softened-exact Full outputs: {out}")
    return {"outdir": out, "summary": summary, "paired": paired, "preflight": report}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PhysicsSim-Full sector versus softened-exact control."
    )
    parser.add_argument("--seeds", default=",".join(map(str, FORMAL_HPO_SEEDS)))
    parser.add_argument("--final-seeds", default=",".join(map(str, FORMAL_FINAL_SEEDS)))
    parser.add_argument("--alpha-min", type=float, default=SOFT_EXACT_FORMAL_GRID["alpha_min"])
    parser.add_argument("--alpha-max", type=float, default=SOFT_EXACT_FORMAL_GRID["alpha_max"])
    parser.add_argument("--alpha-step", type=float, default=SOFT_EXACT_FORMAL_GRID["alpha_step"])
    parser.add_argument("--beta-min", type=float, default=SOFT_EXACT_FORMAL_GRID["beta_min"])
    parser.add_argument("--beta-max", type=float, default=SOFT_EXACT_FORMAL_GRID["beta_max"])
    parser.add_argument("--beta-step", type=float, default=SOFT_EXACT_FORMAL_GRID["beta_step"])
    parser.add_argument("--direction-softening-delta", type=float, default=MIN_DISTANCE_BASE)
    parser.add_argument("--sector-runs", default=str(DEFAULT_SECTOR_RUNS))
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_control(
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
        direction_softening_delta=args.direction_softening_delta,
        allow_smoke=args.allow_smoke,
        overwrite=args.overwrite,
        generate_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
