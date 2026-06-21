"""run_paper_script.ch5_run_baseline

Chapter 5 — Baseline model runs.

Runs ONE baseline model and (optionally) generates its built-in figures.

Baselines supported in this repo:
  - SMACOF
  - DC-SMACOF

Note
----
The baseline implementations in `library.model_cmp` contain their own
visualization/export logic. This script is a thin wrapper so the paper's
experimental section can be reproduced with a single command.
Legacy aliases `StressMajorization` and `DirectedMDS` are still accepted.

Usage
-----
Run from the physics_simulation project root.
Default fixed anchors come from data/site_rmse_points.csv (use_role=anchor).

python -m run_paper_script.paper_run ch5-baseline --model SMACOF --vis
python -m run_paper_script.paper_run ch5-baseline --model DC-SMACOF --no-vis
"""

from __future__ import annotations

import argparse


def _normalize_model_name(model: str) -> str:
    aliases = {
        "SMACOF": "SMACOF",
        "StressMajorization": "SMACOF",
        "stressmajorization": "SMACOF",
        "stress-majorization": "SMACOF",
        "smacof": "SMACOF",
        "DirectedMDS": "DC-SMACOF",
        "Directed_MDS": "DC-SMACOF",
        "directedmds": "DC-SMACOF",
        "directed-mds": "DC-SMACOF",
        "dc-smacof": "DC-SMACOF",
        "DC-SMACOF": "DC-SMACOF",
    }
    return aliases.get(model, aliases.get(model.strip(), ""))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        type=str,
        required=True,
        help="Which baseline to run: SMACOF or DC-SMACOF. Legacy aliases are accepted.",
    )
    p.add_argument("--vis", action="store_true", help="Generate baseline's built-in figures")
    p.add_argument("--no-vis", action="store_true", help="Disable baseline's built-in figures")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    vis = bool(args.vis) and not bool(args.no_vis)

    from library.model_cmp import run_stress_majorization, run_directed_MDS

    model = _normalize_model_name(args.model)
    if model == "SMACOF":
        run_stress_majorization(vis=vis)
    elif model == "DC-SMACOF":
        run_directed_MDS(vis=vis)
    else:
        raise SystemExit(f"Unknown model: {args.model}. Use SMACOF or DC-SMACOF.")


if __name__ == "__main__":
    main()
