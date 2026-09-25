# Same-weight sector / exact controls

These additional experiments retain the existing sector objective and runners.
`library/scipy_exact_objective.py` supplies a compatible subclass with a separate
exact-direction calculation kernel. Only the angular loss changes to
`0.5 * sum(phi**2)`; the common BFGS wrapper and strict domain checks are retained.

## Coordinate and evaluation contract

- x east, y north; one simulation unit is 10 Li or 4.15 km.
- Existing LCC projection parameters and site split are retained.
- Three calibration sites: 鄯善, 車師前, 都護治/烏壘. Eight other sites remain held out.
- BFGS works in the Shanshan-centred frame and eliminates the six fixed anchor
  coordinates. Adding [600,250] restores the shared plotting/evaluation frame.
- Unanchored PhysicsSim initializes with an empty fixed-site list and creates
  no pivots. Only after simulation is the whole configuration translated so
  Shanshan lies at [600,250]. This evaluation-frame reference is not a model
  anchor. No rotation, reflection, scale fitting or test-site fitting is used.
- Both formulations are evaluated against the same 44 raw direction records
  using sector violation and excess-angle metrics.

## Experiments

Both use seeds 0–99 and identical initial coordinates within each seed pair.
No HPO is performed. The prescribed Sector-Full parameters are alpha=1,
beta=-0.5: distance coefficient 1500, direction coefficient 10000000, softened
repulsion coefficient 158.11388300841898, epsilon=0.1.

1. **BFGS Full:** distance, direction, repulsion and three fixed anchors. Run
   both direction formulations from matched initial states. Full-memory BFGS
   uses analytic gradients, gtol=1e-3, maxiter=200*dimension, and the existing
   Wolfe line-search settings. Record failures; do not weaken tolerances or
   silently substitute failed endpoints for converged results.
2. **PhysicsSim DistDir:** same distance and direction coefficients, repulsion
   exactly zero, no model anchors. Run both formulations for 1001 updates of
   dt=0.01 with the existing mass and damping parameters.

Summaries use mean and sample SD of successful runs. Paired differences are
Exact minus Sector on common successful seeds, with 10000 percentile-bootstrap
replicates. Failed BFGS endpoints and their metrics are saved as diagnostics.
If no Exact run converges, there is no converged BFGS paired comparison.

## Reproduction

From the physics_simulation directory:

```powershell
.\.venv_codex\Scripts\python.exe -m run_paper_script.ch5_matched_direction_controls --n-seeds 100 --workers 4 --outdir outputs/ch5_matched_direction_controls_new_run
.\.venv_codex\Scripts\python.exe -m scripts.verify_matched_direction_controls --outdir outputs/ch5_matched_direction_controls_new_run
```

Each seed is checkpointed. The runner refuses to resume into an output
directory whose protocol or recorded source hashes have changed. Choose a new
directory for changed code. Existing experiment outputs are never overwritten.

The 20260905 run began before the exact kernel was moved entirely into its
separate module. Its executed source snapshot is preserved, and verification
checks exact equality of the final additive kernel with the executed kernel
on 100 random configurations, as well as saved endpoint objective values.
