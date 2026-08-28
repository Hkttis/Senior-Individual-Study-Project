"""Validate inputs and configuration before formal detour sensitivity runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from library.config import (
    DIRECTIONAL_FORCE_MAGNITUDE_BASE,
    REPULSION_STRENGTH_BASE,
    SPRING_STIFFNESS_BASE,
)
from run_paper_script.ch5_anchor_split_robustness import _parse_seed_list
from run_paper_script.ch5_detour_factor_sensitivity import preflight_detour_sensitivity
from scripts.check_direction_data import main as check_direction_data
from scripts.check_site_points import main as check_site_points


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default=",".join(map(str, range(10))))
    parser.add_argument("--final-seeds", default=",".join(map(str, range(100))))
    parser.add_argument("--kappa-min", type=float, default=0.70)
    parser.add_argument("--kappa-max", type=float, default=1.00)
    parser.add_argument("--kappa-step", type=float, default=0.025)
    parser.add_argument("--alpha-min", type=float, default=-1.0)
    parser.add_argument("--alpha-max", type=float, default=1.5)
    parser.add_argument("--alpha-step", type=float, default=0.5)
    parser.add_argument("--beta-min", type=float, default=-2.0)
    parser.add_argument("--beta-max", type=float, default=0.5)
    parser.add_argument("--beta-step", type=float, default=0.5)
    parser.add_argument("--fixed-alpha", type=float)
    parser.add_argument("--fixed-beta", type=float)
    parser.add_argument("--reference-alpha", type=float)
    parser.add_argument("--reference-beta", type=float)
    parser.add_argument("--w-dis", type=float, default=1.0)
    parser.add_argument("--base-spring", type=float, default=SPRING_STIFFNESS_BASE)
    parser.add_argument("--base-dir", type=float, default=DIRECTIONAL_FORCE_MAGNITUDE_BASE)
    parser.add_argument("--base-rep", type=float, default=REPULSION_STRENGTH_BASE)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if check_site_points() != 0 or check_direction_data() != 0:
        print("[FAIL] Core input-data checks failed.")
        return 1
    try:
        report = preflight_detour_sensitivity(
            seeds=_parse_seed_list(args.seeds),
            final_seeds=_parse_seed_list(args.final_seeds),
            kappa_min=args.kappa_min,
            kappa_max=args.kappa_max,
            kappa_step=args.kappa_step,
            alpha_min=args.alpha_min,
            alpha_max=args.alpha_max,
            alpha_step=args.alpha_step,
            beta_min=args.beta_min,
            beta_max=args.beta_max,
            beta_step=args.beta_step,
            fixed_alpha=args.fixed_alpha,
            fixed_beta=args.fixed_beta,
            reference_alpha=args.reference_alpha,
            reference_beta=args.reference_beta,
            outdir=Path(args.outdir),
            w_dis=args.w_dis,
            base_spring_stiffness=args.base_spring,
            base_directional_force=args.base_dir,
            base_repulsion_strength=args.base_rep,
            resume=args.resume,
        )
    except (FileNotFoundError, FileExistsError, OSError, ValueError) as exc:
        print(f"[FAIL] Detour sensitivity preflight failed: {exc}")
        return 1

    print("[OK] Detour sensitivity preflight passed")
    print(f"  Hyperparameter policy: {report['hyperparameter_policy']}")
    if report["hyperparameter_policy"] == "fixed":
        print(f"  Fixed alpha/beta: {report['fixed_alpha']}, {report['fixed_beta']}")
    elif report["hyperparameter_policy"] == "scenario_specific_hpo_with_fixed_reference":
        print(f"  Formal reference alpha/beta: {report['reference_alpha']}, {report['reference_beta']}")
    print(f"  Scenario scales: {report['scenario_scales']}")
    print(f"  Distance edges: {report['n_distance_edges']} (source CSV and ini_data match exactly)")
    print(f"  HPO seeds: {report['hpo_seeds']}")
    print(f"  Final-evaluation seeds: {report['final_evaluation_seeds']}")
    if report["hyperparameter_policy"] == "fixed":
        print("  HPO: skipped")
    else:
        print(f"  Grid: {len(report['alpha_values'])} x {len(report['beta_values'])}")
        print(f"  HPO runs per scenario: {report['hpo_runs_per_scenario']}")
    print(f"  Final runs per scenario: {report['final_runs_per_scenario']}")
    print(f"  Expected total model runs: {report['expected_total_model_runs']}")
    print(f"  Existing completed scenarios: {report['existing_completed_scenarios']}")
    print(f"  Existing incomplete scenarios: {report['existing_incomplete_scenarios']}")
    print(f"  Calibration anchors: {report['anchor_labels']}")
    print(f"  Held-out test sites: {report['test_labels']}")
    print(f"  Free disk: {report['free_disk_bytes'] / 1_000_000_000:.2f} GB")
    print("  Input SHA-256 values:")
    print(json.dumps(report["input_sha256"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
