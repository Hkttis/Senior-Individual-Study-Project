# Code inventory and active workflow

This inventory describes the September 2026 paper workflow. It marks entry
points; it does not delete old implementations or change recorded results.

## Active paper code

- `run_paper_script/paper_run.py`: experiment command dispatcher. Current
  formal analyses use the progressive ablation, BFGS, anchor-split, detour,
  and direction-tolerance commands listed in its help text.
- `library/physics.py`, `library/scipy_objective.py`,
  `library/scipy_minimizer.py`, `library/metrics.py`, and related `library/`
  modules: model computation and evaluation.
- `MDS_model/stress_majorization_mds_model.py` and
  `MDS_model/directed_mds_model.py`: comparison baselines.
- `scripts/check_*`, `scripts/verify_*`, `scripts/export_*`, and
  `scripts/update_paper_results.py`: preflight, verification, and export.
  Some exports rely on helpers in older-looking scripts. Do not remove an
  individual module based only on its filename.
- `tests/`: regression checks. Synthetic fixtures are intentional test data.

## Supplemental and historical code

- `experiments/` contains isolated audits, including DC-SMACOF historical
  snapshots and all-sites BFGS diagnostics. These are not the formal model
  comparison entry points.
- `run_paper_script/ch6_interaction_map.py`, `scripts/spring_main.py`,
  `scripts/spring_confidence.py`, `scripts/multi_spring_confidence_ellipse.py`,
  and `scripts/paper_run_tmp.py` are retained for historical or exploratory
  use. They are excluded from the documented formal workflow. The interactive
  map remains callable with `ch6-map` only when explicitly requested.
- `scripts/create_section_6_5_visual_prototype.py` has a historical name but
  is still imported by manuscript spatial comparison and verification code;
  it is active and must not be moved without updating those imports.

## Results and publication checks

Generated `outputs/`, `results_data/`, `paper_results/`, local backups,
virtual environments, caches, and root-level exploratory images are not
tracked. Their omission from Git does not mean the underlying experiment was
removed. Keep local output folders and backups intact when cleaning the code.
Use the saved experiment configurations and verification scripts to identify
the numerical source of each manuscript figure or table.
