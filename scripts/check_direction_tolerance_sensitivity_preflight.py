"""Run the direction-tolerance experiment checks without starting simulations."""

from __future__ import annotations

import argparse
import json

from run_paper_script.ch5_direction_tolerance_sensitivity import (
    FORMAL_FINAL_SEEDS,
    FORMAL_GRID,
    FORMAL_HPO_SEEDS,
    FORMAL_NUMERATORS,
    REFERENCE_ALPHA,
    REFERENCE_BETA,
    preflight_direction_tolerance_sensitivity,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = preflight_direction_tolerance_sensitivity(
        numerators=FORMAL_NUMERATORS,
        hpo_seeds=FORMAL_HPO_SEEDS,
        final_seeds=FORMAL_FINAL_SEEDS,
        reference_alpha=REFERENCE_ALPHA,
        reference_beta=REFERENCE_BETA,
        outdir=args.outdir,
        resume=args.resume,
        allow_smoke=False,
        **FORMAL_GRID,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
