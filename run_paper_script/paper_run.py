r"""run_paper_script.paper_run

Paper reproduction entrypoint (thin dispatcher).

Run commands from the physics_simulation project root:
  C:\Users\hktti\Desktop\Codex projects\physics_simulation

Current data convention:
  - data/site_rmse_points.csv contains 3 rows with use_role=anchor.
  - It contains 8 rows with use_role=test for final site-position RMSE.
  - LCC bounds and standard parallels are read from site_rmse_points.csv.
  - New HPO/ablation outputs record LCC metadata in their config JSON files.
  - HPO, manual candidate selection, and ablation refuse to write into non-empty
    output directories unless --overwrite is passed intentionally.

Recommended preflight:
python -m scripts.check_site_points
python -m scripts.check_direction_data
python -m scripts.rebuild_ini_data

Chapter 4 examples:
python -m run_paper_script.paper_run ch4 --seed 0 --no-save
python -m run_paper_script.paper_run ch4 --seed 0 --plot

Chapter 5 examples:
python -m run_paper_script.paper_run ch5-baseline --model StressMajorization --vis
python -m run_paper_script.paper_run ch5-baseline --model DirectedMDS --vis
python -m run_paper_script.paper_run ch5-compare --seed 37
python -m run_paper_script.paper_run ch5-benchmark --n-runs 100 --save-histories
python -m run_paper_script.paper_run ch5-bootstrap --n-bootstrap 300 --spring-jitter 0.05 --repulse-jitter 0.20

Preflight HPO smoke test:
python -m run_paper_script.paper_run ch5-hparam-kfold --seeds 0 --alpha-min 1 --alpha-max 1 --alpha-step 1 --beta-min 0.5 --beta-max 0.5 --beta-step 1 --outdir outputs/preflight_hpo_lcc_sitebounds_smoke

Formal HPO grid:
python -m run_paper_script.paper_run ch5-hparam-kfold --seeds 0,1,2,3,4,5,6,7,8,9 --alpha-min -1 --alpha-max 1.5 --alpha-step 0.5 --beta-min -2 --beta-max 0.5 --beta-step 0.5 --outdir outputs/ch5_hparam_anchor_loo_grid_lcc_sitebounds_36x10

Manual candidate selection after HPO, example alpha=1 beta=0.5:
python -m scripts.select_hpo_candidate --source-hpo-outdir outputs/ch5_hparam_anchor_loo_grid_lcc_sitebounds_36x10 --alpha 1 --beta 0.5 --outdir outputs/ch5_hparam_anchor_loo_grid_lcc_sitebounds_36x10_manual_alpha_1_beta_0.5

Ablation smoke test from selected HPO:
python -m run_paper_script.paper_run ch5-ablation --hpo-outdir outputs/ch5_hparam_anchor_loo_grid_lcc_sitebounds_36x10_manual_alpha_1_beta_0.5 --seeds 0 --outdir outputs/ch5_ablation_lcc_sitebounds_smoke

Formal ablation, 100 seeds:
python -m run_paper_script.paper_run ch5-ablation --hpo-outdir outputs/ch5_hparam_anchor_loo_grid_lcc_sitebounds_36x10_manual_alpha_1_beta_0.5 --seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99 --outdir outputs/ch5_ablation_lcc_sitebounds_alpha_1_beta_0.5_100seeds

Chapter 6 examples:
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
    elif cmd == "ch5-ablation":
        _as_mod("run_paper_script.ch5_ablation_study")
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

