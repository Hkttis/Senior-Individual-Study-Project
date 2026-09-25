"""Read-only preflight for the sector/exact PhysicsSim control experiment."""

from __future__ import annotations

import argparse
import json

from run_paper_script.ch5_sector_exact_control import (
    DEFAULT_SECTOR_RUNS,
    FORMAL_FINAL_SEEDS,
    FORMAL_GRID,
    FORMAL_HPO_SEEDS,
    _parse_seed_list,
    preflight_report,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check inputs and formal protocol without running models.")
    parser.add_argument("--seeds", default=",".join(map(str, FORMAL_HPO_SEEDS)))
    parser.add_argument("--final-seeds", default=",".join(map(str, FORMAL_FINAL_SEEDS)))
    parser.add_argument("--alpha-min", type=float, default=FORMAL_GRID["alpha_min"])
    parser.add_argument("--alpha-max", type=float, default=FORMAL_GRID["alpha_max"])
    parser.add_argument("--alpha-step", type=float, default=FORMAL_GRID["alpha_step"])
    parser.add_argument("--beta-min", type=float, default=FORMAL_GRID["beta_min"])
    parser.add_argument("--beta-max", type=float, default=FORMAL_GRID["beta_max"])
    parser.add_argument("--beta-step", type=float, default=FORMAL_GRID["beta_step"])
    parser.add_argument("--sector-runs", default=str(DEFAULT_SECTOR_RUNS))
    parser.add_argument("--allow-smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = preflight_report(
        hpo_seeds=_parse_seed_list(args.seeds),
        final_seeds=_parse_seed_list(args.final_seeds),
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        alpha_step=args.alpha_step,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        beta_step=args.beta_step,
        sector_runs=args.sector_runs,
        allow_smoke=args.allow_smoke,
    )
    print("[OK] Sector/exact control preflight passed")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
