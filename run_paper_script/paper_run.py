"""run_paper_script.paper_run

Paper reproduction entrypoint (thin dispatcher).

Run commands from the physics_simulation project root:
  C:\\Users\\hktti\\Desktop\\Codex projects\\physics_simulation

Current site-point convention:
  - data/site_rmse_points.csv contains 3 rows with use_role=anchor:
      鄯善, 車師前, 都護治/烏壘
  - It contains 8 rows with use_role=test for final site-position RMSE.
  - Most scripts read those anchors automatically; do not pass --fixed unless
    you intentionally want to override the default anchor set.

The project has chapter-aligned executable scripts:

Chapter 4 (Core method)
  - run_paper_script/ch4_physics_reconstruct.py

Chapter 5 (Experiments)
  - run_paper_script/ch5_run_baseline.py
  - run_paper_script/ch5_compare_models_convergence.py
  - run_paper_script/ch5_benchmark_models.py
  - run_paper_script/ch5_bootstrap_stability.py
  - run_paper_script/ch5_hparam_kfold_gridsearch_pareto.py

Chapter 6 (Visualizations)
  - run_paper_script/ch6_visualize_single_model.py
  - run_paper_script/ch6_interaction_map.py
  - run_paper_script/ch6_visualize_representative.py

This file keeps a single convenient entrypoint with subcommands that forward to
those scripts.

Examples
--------
python -m scripts.check_site_points
python -m scripts.check_direction_data
python -m scripts.rebuild_ini_data

python -m run_paper_script.paper_run ch4 --seed 0 --no-save
python -m run_paper_script.paper_run ch4 --seed 0 --plot

python -m run_paper_script.paper_run ch5-baseline --model StressMajorization --vis
python -m run_paper_script.paper_run ch5-baseline --model DirectedMDS --vis
python -m run_paper_script.paper_run ch5-compare --seed 37
python -m run_paper_script.paper_run ch5-benchmark --n-runs 100 --save-histories
python -m run_paper_script.paper_run ch5-bootstrap --n-bootstrap 300 --spring-jitter 0.05 --repulse-jitter 0.20

Small HPO smoke test:
python -m run_paper_script.paper_run ch5-hparam-kfold --seeds 0 --alpha-min 0 --alpha-max 0 --alpha-step 1 --beta-min 0 --beta-max 0 --beta-step 1 --outdir outputs/ch5_hparam_anchor_loo_smoke

Example HPO grid:
python -m run_paper_script.paper_run ch5-hparam-kfold --seeds 0 --alpha-min -1 --alpha-max 1 --alpha-step 1 --beta-min -1 --beta-max 1 --beta-step 1 --outdir outputs/ch5_hparam_anchor_loo_grid_3x3_seed0

python -m run_paper_script.paper_run ch6-visualize --model PhysicsSim --seed 0 --no-wait
python -m run_paper_script.paper_run ch6-visualize --model SMACOF --no-wait
python -m run_paper_script.paper_run ch6-visualize --model DC-SMACOF --no-wait
python -m run_paper_script.paper_run ch6-representative
python -m run_paper_script.paper_run ch6-map
"""

from __future__ import annotations

import runpy
import sys


def _as_mod(module: str) -> None:
    """Run another script module as if executed with `python -m`.

    We use `runpy` so we don't depend on OS-specific shelling.
    """
    runpy.run_module(module, run_name="__main__")


def main() -> None:
    # We only look at argv[1] as the command and forward the remaining args
    # to the selected module unchanged.
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(__doc__)
        raise SystemExit(0)

    cmd = sys.argv[1]
    # Shift argv so the target module sees its own flags.
    sys.argv = [sys.argv[0], *sys.argv[2:]]

    if cmd == "ch4":
        _as_mod("run_paper_script.ch4_physics_reconstruct")
    elif cmd == "ch5-baseline":
        _as_mod("run_paper_script.ch5_run_baseline")
    elif cmd == "ch5-compare":
        _as_mod("run_paper_script.ch5_compare_models_convergence")
    elif cmd == "ch5-benchmark":
        _as_mod("run_paper_script.ch5_benchmark_models")
    elif cmd == "ch5-bootstrap":
        _as_mod("run_paper_script.ch5_bootstrap_stability")
    elif cmd == "ch5-hparam-kfold":
        _as_mod("run_paper_script.ch5_hparam_kfold_gridsearch_pareto")
    elif cmd == "ch6-visualize":
        _as_mod("run_paper_script.ch6_visualize_single_model")
    elif cmd == "ch6-map":
        _as_mod("run_paper_script.ch6_interaction_map")
    elif cmd == "ch6-representative":
        _as_mod("run_paper_script.ch6_visualize_representative")
    else:
        raise SystemExit(f"Unknown command: {cmd}. Run with -h for usage.")


if __name__ == "__main__":
    main()
