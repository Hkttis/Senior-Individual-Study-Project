# DC-SMACOF Wang et al. edge-direction audit

## Scope

This audit tests whether the project can use the Wang et al. edge-direction
D-step without the fixed/mean-distance proxy currently present in the
production DC-SMACOF model. The production model and all existing HPO/AS
outputs were left unchanged.

Paper definition tested:

```text
d'_ij = ||x_i - x_j|| u_ij
```

With the project's +source/-target incidence convention, the stored target is:

```text
D'_ij = -||x_target - x_source|| u_source_to_target
```

Source: Wang et al., "Revisiting Stress Majorization as a Unified Framework
for Interactive Constrained Graph Visualization," IEEE TVCG,
DOI 10.1109/TVCG.2017.2745919.

## Reproducibility snapshot

- Production source snapshot:
  `snapshots/directed_mds_model_production_before_audit.py`
- Snapshot SHA-256:
  `096B61FFAA847992738E5F6FBA35DEF8B3511DA39DCBAF52893DEA545A31C56F`
- The snapshot hash matched `MDS_model/directed_mds_model.py` before the audit.
- The audit never imports the snapshot as the formal project model and never
  modifies the production file.

## Implementations tested

1. `production_proxy`: direction target length is the observed text distance,
   or the mean text distance for a direction-only pair.
2. `wang_current`: every direction target uses the current iteration's pair
   distance.
3. `wang_model_minimal_copy.py`: an exact copy of the production model with
   only the direction target changed to Wang current-distance behavior.

No damping, clipping, ridge term, fallback target length, or NaN recovery was
used in the Wang runs.

## Data and numerical setup

- Nodes: 35
- Distance edges: 44
- Direction edges: 44
- Direction edges without a matching distance edge: 27
- Distance weight: 1
- Direction weights tested: 0.01, 0.0316228, 0.1, 0.316228, 1
- Seeds: 0-9
- Updates per clean audit run: 1000
- Updates per minimal production-copy run: 1001, matching its current loop
- Solver: reduced Laplacian with node 0 fixed, Cholesky triangular solve
- Reduced Laplacian condition number at alpha=-0.5: 21118.7

## Synthetic verification

Five tests passed:

- the Wang target uses the current pair length;
- the incidence sign places the target in the requested DIR8 direction;
- one direction-only update preserves the current pair length;
- an aligned edge has zero direction objective;
- a four-node synthetic run remains finite for 100 updates.

Command:

```powershell
python -m pytest tests/test_dc_smacof_wang2017_audit.py -q -p no:cacheprovider
```

Result: `5 passed`.

## Real-data stability results

The Wang implementation completed all 50 combinations in the original DC
HPO alpha range. No run produced NaN, Inf, a solver exception, or a coordinate
larger than the audit threshold of 1e12 Li.

| alpha | v weight | runs | failures | all frames finite | objective increases | maximum coordinate ever (Li) |
| ---: | ---: | ---: | ---: | --- | ---: | ---: |
| -2.0 | 0.010000 | 10 | 0 | yes | 0 | 2864.39 |
| -1.5 | 0.031623 | 10 | 0 | yes | 155 | 2834.69 |
| -1.0 | 0.100000 | 10 | 0 | yes | 0 | 2734.91 |
| -0.5 | 0.316228 | 10 | 0 | yes | 0 | 2510.41 |
| 0.0 | 1.000000 | 10 | 0 | yes | 468 | 2251.99 |

The non-monotone sections at alpha=-1.5 and alpha=0 were bounded, not
explosive. The largest observed one-step increase in the diagnostic actual
objective was approximately 0.00345. No numerical stabilization was therefore
activated or added.

The minimally edited production clone independently completed seeds 0-9:

- 10/10 successful runs;
- 1002 stored frames per seed;
- every stored coordinate was finite;
- zero increases in the diagnostic actual objective;
- maximum coordinate over every frame: 2510.27 Li.

## Alpha=-0.5 proxy comparison

These values are diagnostic only. Test-site RMSE must not be used to select the
DC-SMACOF hyperparameter.

| mode | Stress, mean +/- SD | Violation Rate, mean +/- SD | Mean Angular Error (rad), mean +/- SD | Test RMSE (km), mean +/- SD |
| --- | ---: | ---: | ---: | ---: |
| production proxy | 0.581753 +/- 0.022567 | 0.031818 +/- 0.011736 | 0.524785 +/- 0.252801 | 267.093 +/- 11.291 |
| Wang current-distance | 0.239958 +/- 0.000864 | 0.006818 +/- 0.010978 | 0.001764 +/- 0.003473 | 318.807 +/- 2.459 |

The Wang definition materially changes the fitted layouts. Existing
DC-SMACOF HPO, AS rows, paper tables, and figures cannot be relabeled as Wang
results without rerunning them.

## Additional matrix/data issue found

The direction data contains two constraints for the same undirected pair:

- 莎車 -> 疏勒: west
- 疏勒 -> 莎車: south

The production model stores one pair weight in `veight/LV`, but stores both
constraints as separate columns in `JV`. Consequently, the left side counts
the pair once while the right side counts it twice. The clean audit incidence
implementation counts both constraints on both sides. This explains the small
difference between the clean Wang runner and the minimally edited production
clone and must be resolved explicitly before the formal rerun.

## Decision

The evidence supports the first branch of the proposed decision tree:

1. Wang current-distance DC-SMACOF can run on the current real dataset.
2. The previously reported NaN/Inf behavior is not reproduced with the current
   sign convention and reduced-Cholesky solver.
3. A proxy target length is not required for numerical stability in the tested
   alpha range and seeds.
4. The formal implementation should be changed to the Wang D-step, after an
   explicit decision on duplicate direction constraints.
5. DC-SMACOF HPO must then be rerun using anchor-only selection metrics.
6. Only DC-SMACOF downstream AS comparisons, summaries, and visualizations need
   regeneration. PhysicsSim HPO and PhysicsSim progressive paired comparisons
   are not affected by this audit.

## Output inventory

- `runs/raw_seed0/`: initial proxy/Wang side-by-side run and iteration traces.
- `runs/wang_raw_seeds0_9/`: alpha=-0.5 Wang runs.
- `runs/production_proxy_seeds0_9/`: alpha=-0.5 proxy runs.
- `runs/wang_raw_alpha_neg2_seeds0_9/`: alpha=-2 Wang runs.
- `runs/wang_raw_alpha_neg1p5_seeds0_9/`: alpha=-1.5 Wang runs.
- `runs/wang_raw_alpha_neg1_seeds0_9/`: alpha=-1 Wang runs.
- `runs/wang_raw_alpha_0_seeds0_9/`: alpha=0 Wang runs.
- `runs/wang_minimal_copy_seeds0_9/`: minimally edited production clone.
- `wang_hpo_range_stability_summary.csv`: consolidated stability and metric summary.
