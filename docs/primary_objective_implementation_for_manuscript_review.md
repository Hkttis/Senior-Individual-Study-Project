# Primary objective implementation: evidence for manuscript and AI review

Verified against the code and saved experiment configurations on 2026-10-18.
The code baseline for this note is Git commit `29a70e3`.

## 1. Scope and source hierarchy

This note describes the primary unsoftened angular-sector PhysicsSim-Full
objective, the independently tuned full-memory BFGS baseline, and the
source-weight-locked PhysicsSim-to-BFGS polishing experiment. The softened
exact-direction controls and direction-tolerance sensitivity experiments are
separate protocols.

For manuscript review, use the actual experiment configuration and run-level
CSV together with the functions below. Do not substitute current CLI defaults,
a different BFGS output directory, or a rounded narrative table for the recorded
experiment settings. The local outputs listed below are not all versioned.

## 2. Distance units and coordinates

Source: [`library/config.py`](../library/config.py),
[`library/units.py`](../library/units.py), and
[`build_current_objective`](../library/scipy_objective.py).

| Quantity | Conversion used in the primary experiment |
| --- | --- |
| Text distance | Input CSV distance is in li |
| Objective target | `d_sim = float(raw_li) * Li2sim`, with `Li2sim = 0.1` |
| Physical conversion | `Li2km = 0.415` |
| Simulation-coordinate conversion | `km2sim = km2pix = 1 / (10 * 0.415)` |
| Consequence | 1 simulation unit = 10 li = 4.15 km |
| Repulsion softening | `epsilon = MIN_DISTANCE_BASE = 0.1` simulation units = 0.415 km |

Both the model-generated distance and target distance inside the reconstruction
objective are in simulation units. The conversion is applied before spring
construction or BFGS objective evaluation, not after evaluating the objective.

For archaeological targets, [`lcc_transformation_with_anchor`](../library/geometry.py)
projects longitude/latitude into LCC meters, subtracts the frame-anchor position,
and divides by 1000 to obtain anchor-relative kilometers.
[`add_fixed_positions`](../library/initialization.py) then calls
`gt_km2sim` and adds the display/frame translation `refer_pos_sim`.
The primary frame anchor is 鄯善.

The SciPy objective uses anchor-relative, y-up simulation coordinates. Saved
`*_y_up_sim.csv` coordinates include the translation `refer_pos_sim`.
`_positions_y_up` adds that translation; `reinsert_exact_anchors`
subtracts it. Pairwise distances are translation invariant. A screen-coordinate
y flip is not part of the objective evaluation.

External held-out RMSE is in kilometers:
[`_evaluate`](../run_paper_script/ch5_ablation_progressive.py)
computes simulation-coordinate error norms and divides by `km2pix`.
Stress is dimensionless: the metric receives positions in km and converts its
simulation-unit targets to km internally.

Do not describe the reconstruction objective itself as being evaluated in km.
The numerical value of the logarithmic repulsion uses the simulation-unit
distance convention. Replacing distances by km inside the formulas would change
the reported objective and require a consistent treatment of epsilon and
coefficients.

## 3. HPO multipliers versus effective objective coefficients

Sources: [`ObjectiveWeights.from_physics_hpo`](../library/scipy_objective.py),
[`_weights_from_alpha_beta`](../run_paper_script/ch5_hparam_kfold_gridsearch_pareto.py),
and [`_physics_forces`](../run_paper_script/ch5_ablation_progressive.py).

The implementation calls the common multiplier `w_dis`. To avoid a
collision with the manuscript's effective distance coefficient, call this
multiplier \(s\) in the manuscript:

\[
w_{\mathrm{dis}}=1500s,\qquad
w_{\mathrm{dir}}=10^6s\,10^\alpha,\qquad
w_{\mathrm{rep}}=500s\,10^\beta.
\]

The code performs:

```python
weights = ObjectiveWeights(
    distance=SPRING_STIFFNESS_BASE * w_dis,
    direction=DIRECTIONAL_FORCE_MAGNITUDE_BASE * w_dis * 10**alpha,
    repulsion=REPULSION_STRENGTH_BASE * w_dis * 10**beta,
)
```

The base constants are 1500, 1,000,000, and 500, respectively. The direction
multiplier increases with alpha: it is \(10^\alpha\), not \(10^{-\alpha}\).
For \(s=1\), the effective ratios are
\(w_{\mathrm{dir}}/w_{\mathrm{dis}}=(10^6/1500)10^\alpha\) and
\(w_{\mathrm{rep}}/w_{\mathrm{dis}}=(500/1500)10^\beta\).

| Experiment | s | alpha | beta | Effective distance | Effective direction | Effective repulsion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Formal PhysicsSim-Full | 1 | 1 | -0.5 | 1500 | 10000000 | 158.11388300841898 |
| Independent-HPO BFGS baseline | 1 | 0.5 | -0.5 | 1500 | 3162277.6601683795 | 158.11388300841898 |
| Source-weight-locked polishing | 1 | 1 | -0.5 | 1500 | 10000000 | 158.11388300841898 |

Thus the independent baseline and PhysicsSim have the same family of objective
terms but different selected coefficients. The polishing before/after comparison
uses identical coefficients.

The three different base constants are not one common normalization factor.
Also, although multiplying a fixed mathematical objective by a positive constant
preserves its minimizers, it need not preserve finite-step PhysicsSim dynamics
or stopping under an absolute BFGS gradient threshold.

## 4. Exact objective evaluated by the primary SciPy code

Source: [`FixedAnchorObjective._components_and_gradient_unchecked`](../library/scipy_objective.py).

Let \(r_{uv}=x_v-x_u\), with all positions and target distances in simulation
units. The raw terms are:

\[
f_{\mathrm{dis}}=\frac12\sum_{(u,v)\in E_d}
(\|r_{uv}\|-d_{uv})^2,
\]

\[
f_{\mathrm{dir}}=\frac12\sum_{(u,v)\in E_r}
\left[\max(0,|\phi_{uv}|-\theta_{h,uv})\right]^2,
\]

\[
f_{\mathrm{rep}}=\sum_{u<v}
\left[-\ln(\|r_{uv}\|+\varepsilon)
-\frac{\varepsilon}{\|r_{uv}\|+\varepsilon}\right].
\]

The total is exactly
\(F=w_{\mathrm{dis}}f_{\mathrm{dis}}+
w_{\mathrm{dir}}f_{\mathrm{dir}}+
w_{\mathrm{rep}}f_{\mathrm{rep}}\).
`ObjectiveComponents` stores raw values, weighted values, and total separately.

Implementation details that affect reproducibility:

- Distance rows are read from `data/distance_edges_verified.csv`; raw li
  targets are multiplied by 0.1 exactly once in `build_current_objective`.
- Direction observations are read from `data/direction_edges_verified.csv`.
  Each row contributes to the angular loss. The primary objective does not apply
  the DC-SMACOF composite-direction preprocessing here.
- The signed angle is `atan2(cross(r_hat, desired_unit), dot(r_hat, desired_unit))`.
  The loss uses its absolute value. Angles are in radians.
- The current primary half-width constants are `theta_thr_4dir = pi/2`
  and `theta_thr_8dir = pi/4` (90 and 45 degrees). Do not silently replace
  these by narrower conventional octant sectors.
- `np.triu_indices(n, k=1)` defines all unordered repulsion pairs.
  For 35 nodes this is 595 pairs, including anchor-anchor pairs. Anchor-anchor
  energies are constant during fixed-anchor optimization but remain in the
  reported total.
- There is no division by edge count or pair count in these three raw sums.
  The distance and direction terms each contain the factor 1/2.
- With `softened = lengths + epsilon`, the repulsion gradient for
  \(r=x_v-x_u\) is \(-r/(\|r\|+\varepsilon)^2\).
  Consequently, the repulsive force on v is
  \(+w_{\mathrm{rep}}r/(\|r\|+\varepsilon)^2\).
  This matches the primary force in [`library/physics.py`](../library/physics.py).
- Spring damping and velocity resistance affect PhysicsSim dynamics but are not
  extra positional terms added to the reported \(F(X)\).

Use the executed angular-hinge branch of `apply_forces` when reviewing
direction forces. Its introductory docstring still includes older cosine-hinge
notation and is not the formula executed by the primary branch.

## 5. Anchors and optimizer handling

Sources: [`FIXED_ANCHORS_SIM`](../library/scipy_objective.py),
[`create_nodes_and_springs`](../library/physics.py),
[`reinsert_exact_anchors`](../library/scipy_diagnostics.py), and
[`run_bfgs`](../library/scipy_minimizer.py).

The primary fixed anchors, in anchor-relative simulation coordinates, are:

| Anchor | x | y |
| --- | ---: | ---: |
| 鄯善 | 0 | 0 |
| 車師前 | 11.965621378532099 | 106.68680310031324 |
| 都護治/烏壘 | -77.72999630158114 | 67.61626239867074 |

These coordinates are constants in the SciPy problem. `build_current_objective`
does not read held-out-site coordinates. The verifier can independently compare
the constants and geographic evaluation with the current input data.

PhysicsSim uses Pymunk `PivotJoint` constraints with `max_force=1e9`
and `max_bias=1e7`; tiny numerical anchor drift can occur.
BFGS optimizes only the free-node coordinates: `pack` excludes anchors and
`unpack` inserts their immutable values. There are 64 free scalar variables
for 35 nodes with three fixed anchors.

Before polishing, the code restores exact anchor coordinates and records their
drift. Free coordinates are retained. Its "before" objective is reevaluated on
this corrected configuration, and both before and after use the same problem.

The optimizer is full-memory `minimize(method="BFGS", jac=True)` with an
analytic float64 gradient, `gtol=1e-3`, infinity norm, `xrtol=0`,
`c1=1e-4`, and `c2=0.9`. The default iteration ceiling is
200 times the free-variable dimension (12800 for the final three-anchor problem).
Success requires SciPy success and gradient infinity norm at most gtol.
This is not an unlimited run or a global-optimality guarantee.

The primary SciPy objective rejects distance/direction collisions at separation
at most 1e-12 and the antipodal angular branch singularity. An invalid evaluation,
including a line-search trial, fails the run rather than supplying a dummy
gradient. PhysicsSim skips direction force at separation below 1e-9. Therefore
"same anchor constraints" and "same objective terms" do not imply identical
numerical handling at singular states.

## 6. HPO provenance: do not merge the two selection histories

The primary PhysicsSim and independent BFGS HPO output configurations both record
alpha from -1 to 1.5 and beta from -2 to 0.5, each with step 0.5: 36 grid points.
They use seeds 0 through 9 and three calibration-anchor LOO folds. Each fold fixes
two calibration anchors and validates the third; held-out test sites are not the
HPO validation set. In BFGS folds, the dimension is 66, rather than final-run 64.

The formal PhysicsSim selection file records
`selection_rule: manual_pareto_candidate`, alpha=1 and beta=-0.5,
with the candidate confirmed to be on the source Pareto front.
Do not describe this saved primary selection as the same automatic selection
used by the independent BFGS baseline.

The BFGS selection file records
`selection_rule: pareto_one_se_balanced`, alpha=0.5 and beta=-0.5.
Its HPO configuration retains grid points with finite objectives and at least
one successful run in every LOO fold, while retaining failure counts.
It records 36 eligible grid points but only 5 complete grid points. Do not
describe all 36 BFGS grid points as having every planned optimization succeed.

Sources:

- PhysicsSim grid: `outputs/ch5_hparam_anchor_loo_grid_lcc_sitebounds_36x10_run2/gridsearch_config.json`
- PhysicsSim final selection: `outputs/ch5_hparam_anchor_loo_grid_lcc_sitebounds_36x10_run2_manual_alpha_1_beta_-0.5/candidate_selection.json`
- BFGS HPO: `outputs/ch5_scipy_bfgs_hpo_grid_36x10_20260823_all_folds_partial_ok/bfgs_hpo_config.json`
- BFGS selection: same directory, `bfgs_hpo_selection.json`
- BFGS implementation: [`ch5_scipy_bfgs_hpo.py`](../run_paper_script/ch5_scipy_bfgs_hpo.py)

A claim of 36 combinations applies to these saved primary grids, not every HPO
experiment or expanded sensitivity grid in the repository.

## 7. Polishing provenance and objective-value reconciliation

The formal output directory is:

```text
outputs/ch5_physics_to_bfgs_polishing_source_weights_100seeds_20260827
```

Its `polishing_config.json` records:

```json
{
  "method_version": "source_weight_locked_fixed_anchor_bfgs_polishing_v2",
  "bfgs_hpo_source": null,
  "alpha": 1.0,
  "beta": -0.5,
  "w_dis": 1.0,
  "effective_weights": {
    "distance": 1500.0,
    "direction": 10000000.0,
    "repulsion": 158.11388300841898
  }
}
```

In [`_source_objective_weights`](../run_paper_script/ch5_physics_bfgs_polishing.py),
the coefficients are recovered from `spring_stiffness`,
`directional_force`, and `repulsion_strength` recorded in the source
PhysicsSim-Full runs. They must be finite, positive, constant across runs, and
consistent with recorded alpha/beta. External BFGS HPO overrides are rejected.

Source PhysicsSim endpoints come from:

```text
outputs/ch5_progressive_as_physics_alpha_1_beta_-0.5_dc_alpha_-2_wang_current_100seeds_random1000_20260721
```

This differs from the independently tuned random-start BFGS output:

```text
outputs/ch5_scipy_bfgs_hpo_selected_alpha_0p5_beta_-0p5_100seeds_20260823
```

Do not substitute the older polishing directory whose name contains
`hpo_selected_100seeds_20260823`.

Mean components from `polishing_summary.csv` are:

| Weighted component | Before | After |
| --- | ---: | ---: |
| Distance | 440274.5366071602 | 157255.836463918 |
| Direction | 114024.5311584675 | 64174.54704962073 |
| Repulsion | -447795.09592308855 | -467017.61347602203 |
| Total | 106503.97184253909 | -245587.2299624833 |

The raw repulsion means are approximately -2832.1048563410795 and
-2953.67873200075. Multiplication by 158.11388300841898 explains the weighted
repulsion magnitudes. An assumed coefficient of only \(10^{-0.5}\) omits the
base coefficient 500 and cannot reproduce these totals.
Increasing a nonnegative distance term alone cannot explain a large negative
objective.

All 595 pair terms and 200 saved endpoint configurations were independently
recomputed when checking this explanation. Before/after component sums agree
with the stored total; rounding components separately can change the displayed
sum by 0.01.

## 8. Paired 95% CI implementation and interpretation

Source: [`export_polishing_paired_ci.py`](../scripts/export_polishing_paired_ci.py),
[`_bootstrap_ci_mean`](../run_paper_script/ch5_ablation_study.py), and
[`tests`](../tests/test_export_polishing_paired_ci.py).

The export reads the existing `polishing_runs.csv`; it does not optimize
coordinates again. It checks the method version, required columns, all 100 unique
configured seeds, finite values, and agreement of stored delta with after minus
before. Rows are sorted by seed.

For each metric:

```python
delta = after - before
lo, hi = _bootstrap_ci_mean(delta, n_boot=10_000, seed=0)
```

The shared helper samples indices with replacement, computes 10000 sample means
(each from 100 paired differences), and returns their 2.5th and 97.5th
percentiles. This is the CI of the mean paired change, not an interval containing
95% of individual country positions or individual run changes.
The explicit 10000/seed=0 setting matches the independent BFGS comparison export.
Progressive-chain comparison currently uses the same helper with its 2000-resample
default; do not claim all existing experiments have identical resample counts.

`polishing_paired_95ci.csv` stores full-precision values:

| Metric | Mean after-minus-before | 95% CI lower | 95% CI upper |
| --- | ---: | ---: | ---: |
| Objective | -352091.20180502243 | -396513.04938469414 | -311616.43731143756 |
| Held-out RMSE (km) | 137.13278046491072 | 127.00914034166195 | 147.12454839829002 |

If the manuscript reports a positive objective decrease magnitude, negate the
mean and reverse the negated CI endpoints:
352091.20 [311616.44, 396513.05].
`polishing_paired_seed_differences.csv` retains all 100 before/after pairs.

## 9. What the comparisons establish

The independent-HPO baseline comparison evaluates resulting geographic accuracy,
constraint-satisfaction metrics, and layout diagnostics under independently
selected coefficients. A lower Stress alone does not establish better
minimization of a common total objective: Stress has its own normalization and
the total objective also contains direction and repulsion terms.

The polishing comparison directly measures further minimization of a fixed
weighted objective from the same initial configuration. The observed mean
objective decreases while mean held-out RMSE increases. It supports an
objective-versus-external-accuracy mismatch in this experiment. It does not prove
a global minimum, identical basins, a unique configuration, or a causal
over-expansion mechanism.

Held-out RMSE is computed after states are generated and is not used by the
polishing objective, line search, stopping rule, or hyperparameter selection.

## 10. Reproduce the checks without rerunning optimization

Run from the project root in its configured Python environment:

```text
python -m scripts.check_physics_bfgs_polishing --outdir outputs/ch5_physics_to_bfgs_polishing_source_weights_100seeds_20260827
python -m scripts.export_polishing_paired_ci --outdir outputs/ch5_physics_to_bfgs_polishing_source_weights_100seeds_20260827
python -m pytest tests/test_export_polishing_paired_ci.py tests/test_physics_bfgs_polishing.py -q
```

The first checker recalculates endpoint objective components and evaluation
metrics, checks anchors and source free coordinates, and audits accepted BFGS
trajectory values. The second command writes only the paired summary and paired
seed CSVs. Raw before/after experiment records remain the numerical source.

The manuscript snapshot builder
[`build_manuscript_ready_results.py`](../scripts/build_manuscript_ready_results.py)
uses the formal source-weight-locked polishing directory and Figure 2's raw
endpoint columns. Its source snapshot copies CSVs when the builder is run;
an older snapshot will not acquire newly generated CI files automatically.

## 11. Manuscript wording safeguards for another AI

- Define \(s\) as the code's common HPO multiplier and \(w_{\mathrm{dis}}\),
  \(w_{\mathrm{dir}}\), \(w_{\mathrm{rep}}\) as the effective coefficients.
  Avoid defining the same symbol as both 1 and 1500.
- State that objective distances and anchor coordinates use simulation units;
  geographic RMSE uses km.
- Give epsilon as 0.1 simulation units, not 0.1 km.
- Distinguish independently tuned BFGS from source-weight-locked polishing.
- Preserve the manual primary PhysicsSim selection provenance.
- Say "same anchor constraints" rather than claiming identical numerical
  constraint handling in Pymunk and reduced-variable BFGS.
- Use Stress/Violation Rate/Mean Angular Error for internal constraint fitting;
  use a same-weight \(F(X)\) comparison for objective minimization.
- Do not infer a numerical bug from negative \(F\): examine the raw and weighted
  repulsion and the simulation-unit convention first.
- Do not assert that the current manuscript has been fully verified from this
  implementation note. Manuscript-to-code consistency still requires reading
  the exact manuscript version being edited.
