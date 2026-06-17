"""run_paper_script.ch5_run_baseline

Chapter 5 — Baseline model runs.

Runs ONE baseline model and (optionally) generates its built-in figures.

Baselines supported in this repo:
  - StressMajorization (stress majorization MDS)
  - DirectedMDS        (vectorized/directed MDS variant)

Note
----
The baseline implementations in `library.model_cmp` contain their own
visualization/export logic. This script is a thin wrapper so the paper's
experimental section can be reproduced with a single command.

Usage
-----
Run from the physics_simulation project root.
Default fixed anchors come from data/site_rmse_points.csv (use_role=anchor).

python -m run_paper_script.paper_run ch5-baseline --model StressMajorization --vis
python -m run_paper_script.paper_run ch5-baseline --model DirectedMDS --no-vis
"""

from __future__ import annotations

import argparse


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["StressMajorization", "DirectedMDS"],
        help="Which baseline to run",
    )
    p.add_argument("--vis", action="store_true", help="Generate baseline's built-in figures")
    p.add_argument("--no-vis", action="store_true", help="Disable baseline's built-in figures")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    vis = bool(args.vis) and not bool(args.no_vis)

    from library.model_cmp import run_stress_majorization, run_directed_MDS

    if args.model == "StressMajorization":
        run_stress_majorization(vis=vis)
    elif args.model == "DirectedMDS":
        run_directed_MDS(vis=vis)
    else:
        raise SystemExit(f"Unknown model: {args.model}")


if __name__ == "__main__":
    main()
