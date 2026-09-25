# Western Han country distribution reconstruction

This directory contains the code and verified input data for the PhysicsSim
paper experiments. The experiment runner is
`python -m run_paper_script.paper_run` from this directory. The source code,
formal outputs, and manuscript exports have different roles; see
[`docs/code_inventory.md`](docs/code_inventory.md) before rerunning or
changing an experiment.

## Current paper workflow

1. Check the site, direction, and distance input data with the relevant
   `scripts/check_*` commands. The historical data live in `data/`.
2. Run HPO and formal model comparisons through `run_paper_script/paper_run.py`.
   The main analyses include progressive ablation, BFGS comparison and
   polishing, anchor-split robustness, detour sensitivity, and direction
   tolerance sensitivity.
3. Generate tables and figures from saved experiment outputs with the
   `scripts/export_*`, `scripts/update_paper_results.py`, and manuscript
   builder scripts. Run their corresponding `scripts/verify_*` checks before
   using an export in the paper.

The runner's `--help` lists exact commands and examples. Each experiment
should use a distinct `--outdir`. Saved `outputs/` and `paper_results/` are
local artifacts, not authoritative source code, and are ignored by Git.

## Layout

| Path | Role |
| --- | --- |
| `data/` | Verified constraints, site references, and anchor candidates |
| `library/` | PhysicsSim, objective functions, metrics, coordinates, and plotting |
| `MDS_model/` | SMACOF and DC-SMACOF baselines |
| `run_paper_script/` | Experiment entry points and CLI dispatcher |
| `scripts/` | Input checks, audits, exports, and output verification |
| `tests/` | Unit, integration, and small synthetic-data tests |
| `experiments/` | Isolated diagnostic experiments and audit snapshots |
| `docs/` | Method notes and code status |

The repository retains historical code for traceability. It is not part of
the current paper workflow unless explicitly cited by an active experiment.
No old script is automatically run by importing this package.
