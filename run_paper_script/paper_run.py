"""scripts.paper_run

Paper reproduction entrypoint (thin dispatcher).

The project has been reorganized into chapter-aligned, executable scripts:

Chapter 4 (Core method)
  - scripts/ch4_physics_reconstruct.py

Chapter 5 (Experiments)
  - scripts/ch5_run_baseline.py
  - scripts/ch5_compare_models_convergence.py
  - scripts/ch5_benchmark_models.py
  - scripts/ch5_bootstrap_stability.py

Chapter 6 (Visualizations)
  - scripts/ch6_visualize_single_model.py
  - scripts/ch6_interaction_map.py

This file keeps a single convenient entrypoint with subcommands that forward to
those scripts.

Examples
--------
python -m run_paper_script.paper_run ch4 --seed 0 --plot
python -m run_paper_script.paper_run ch5-baseline --model StressMajorization --vis
python -m run_paper_script.paper_run ch5-baseline --model DirectedMDS --vis
python -m run_paper_script.paper_run ch5-compare --seed 37
python -m run_paper_script.paper_run ch5-benchmark --n-runs 100 --save-histories
python -m run_paper_script.paper_run ch5-bootstrap --n-bootstrap 300 --spring-jitter 0.05 --repulse-jitter 0.20

python -m run_paper_script.paper_run ch6-visualize --model PhysicsSim --seed 0
python -m run_paper_script.paper_run ch6-visualize --model StressMajorization
python -m run_paper_script.paper_run ch6-visualize --model DirectedMDS
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
    elif cmd == "ch6-visualize":
        _as_mod("run_paper_script.ch6_visualize_single_model")
    elif cmd == "ch6-map":
        _as_mod("run_paper_script.ch6_interaction_map")
    else:
        raise SystemExit(f"Unknown command: {cmd}. Run with -h for usage.")


if __name__ == "__main__":
    main()
