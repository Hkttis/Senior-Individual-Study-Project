# SMACOF and DC-SMACOF: Paper Alignment and Project Adaptations

Last audited: 2026-07-21

This document records how the baseline implementations in this project align
with their source papers, where the project intentionally adapts those methods,
and which details must be stated explicitly in the manuscript. It is a method
provenance note, not an experiment result file.

## Source papers

- SMACOF / stress majorization: Gansner, Koren, and North (2004), *Graph
  Drawing by Stress Majorization*.
- DC-SMACOF / directed stress majorization: Wang et al. (2017), *Revisiting
  Stress Majorization as a Unified Framework for Interactive Constrained Graph
  Visualization*.

## Current data contract

- Nodes: 35.
- Verified distance observations: 44 unique undirected pairs.
- Raw verified direction observations used for evaluation: 44.
- Effective DC-SMACOF direction constraints used for optimization: 43.
- The difference between 44 and 43 is caused by preprocessing repeated
  direction observations on the same unordered pair into one vector-consensus
  DIR8 constraint. The raw observations remain unchanged and are still used for
  Violation Rate and Mean Angular Error evaluation.
- Mean verified textual distance: 440.272727 Li.

## SMACOF

### Parts aligned with Gansner et al. (2004)

The implementation minimizes weighted stress over the active distance pairs:

\[
\sum_{\{i,j\}\in E_D}
\frac{1}{d_{ij}^{2}}
\left(\lVert X_i-X_j\rVert-d_{ij}\right)^2.
\]

The following implementation details agree with the paper:

- Distance weights use \(w_{ij}=d_{ij}^{-2}\).
- The weighted Laplacian \(L_W\) uses off-diagonal entries \(-w_{ij}\) and
  diagonal row sums.
- The majorization matrix is recomputed from the current layout at every
  iteration, including the paper's \(\operatorname{inv}(0)=0\) convention.
- Each update solves \(L_W X^{(t+1)}=L_{X^{(t)}}X^{(t)}\).
- The first node is fixed at the origin by removing the first row and column,
  eliminating the translation null space.
- The reduced positive-definite system is solved by Cholesky factorization,
  which is one of the solvers recommended in the paper.
- The current 35-node, 44-edge distance graph is connected and has no duplicate
  undirected distance pairs, so the reduced Laplacian is solvable.

### Project adaptations and differences

1. **Sparse observed-edge objective**

   The canonical graph-drawing presentation commonly defines \(d_{ij}\) for all
   node pairs using graph-theoretic shortest-path distances. This project gives
   nonzero weight only to the 44 directly observed historical distance pairs.
   Unobserved pairs do not enter the SMACOF objective.

   This is an intentional information-matched adaptation: SMACOF and
   PhysicsSim-DistOnly receive the same observed textual distance constraints.
   Gansner et al. also discuss sparse stress models, but the manuscript must not
   describe this implementation as an all-pairs shortest-path SMACOF.

2. **Stopping rule**

   The paper recommends stopping when relative stress improvement is below
   approximately \(10^{-4}\). The current implementation instead performs up to
   1000 updates and stops early only if stress increases.

   A numerical audit of seeds 0-2 found no stress increases. Seeds 0 and 1 first
   crossed the \(10^{-4}\) relative-improvement threshold at updates 998 and 992;
   seed 2 ended at approximately \(1.03\times10^{-4}\). The current results are
   near the paper's convergence threshold, but the stopping protocol is not an
   exact implementation of Equation (10).

3. **Random initialization**

   The project initializes each run uniformly at random and controls it with the
   experiment seed. The paper recommends multiscale or subspace-based initial
   layouts for speed and quality, but random initialization is not prohibited.
   Multiple seeded runs are therefore part of this project's evaluation design.

4. **Repeated Cholesky factorization**

   \(L_W\) is constant, but the current code factorizes it inside every update.
   The paper recommends factorizing it once and reusing the factor. This affects
   runtime, not the mathematical solution.

5. **Evaluation Stress is separate from optimization stress**

   The internal convergence history is the weighted raw stress optimized by
   SMACOF. The paper-results pipeline reports a separate normalized Stress value
   for comparison across models.

   The current evaluation implementation divides by the sum of squared model
   distances, whereas the manuscript formula divides by the sum of squared
   target distances. This discrepancy does not change model positions, but it
   must be corrected or reconciled before final numerical reporting.

6. **Post-hoc alignment is outside the optimizer**

   Formal experiments convert Li to the common evaluation scale, place the
   designated `anchor_align` node in the shared frame, and apply rotation and
   reflection using the specified calibration anchors. This alignment does not
   alter the SMACOF objective and is part of the evaluation protocol rather than
   the Gansner et al. optimization algorithm.

### Recommended manuscript description

> SMACOF was implemented using sparse stress majorization over the observed
> historical distance-edge set, with weights \(w_{ij}=d_{ij}^{-2}\). The
> weighted Laplacian system was solved after fixing one node to remove the
> translation degree of freedom. This observed-edge formulation was used to
> ensure an information-matched comparison with PhysicsSim-DistOnly.

## DC-SMACOF

### Parts aligned with Wang et al. (2017)

- Distance constraints use the same weighted stress-majorization construction
  as the distance-only baseline.
- The direction target vector is updated with the current pairwise distance,
  rather than a fixed textual distance:

  \[
  D'_{ij}=\lVert X_j-X_i\rVert u_{ij}.
  \]

- In project code, the stored vector carries a negative sign because `JV` uses
  `+source/-target`, while verified direction rows describe `source -> target`.
  This is a coordinate/sign convention adaptation and produces the same
  source-to-target direction required by the paper.
- Distance and direction Laplacians are combined in one linear system.
- The translation null space is removed by fixing the first node, and the
  reduced system is solved by Cholesky factorization.
- Independent audits found finite trajectories under the Wang current-distance
  target rule. A 50-run audit over five alpha values and seeds 0-9 found no
  NaN/Inf failure, and focused production-pipeline tests also remained finite.

### Project adaptations and differences

1. **Sparse observed distance constraints**

   DC-SMACOF uses the same 44 observed historical distance edges as the other
   information-matched models rather than an all-pairs shortest-path stress
   objective. This is an intentional experimental-control decision.

2. **Project-specific direction-weight normalization**

   Wang et al. define \(v_{ij}\) as a user-specified direction weight but do not
   require the normalization used here. The project currently sets:

   \[
   v_{ij}=
   \begin{cases}
   v_{\mathrm{weight}}/d_{ij}^{2}, & \text{if a direct textual distance exists},\\
   v_{\mathrm{weight}}/\bar d^{2}, & \text{otherwise},
   \end{cases}
   \]

   where \(\bar d=440.272727\) Li. This normalization preserves a comparable
   inverse-square scale where direct distance information is available and uses
   a fixed, bounded fallback otherwise. It is retained as the project's chosen
   adaptation and must not be attributed to Wang et al.

3. **Repeated direction-observation preprocessing**

   Wang et al. do not specify the project's historical-data issue in which the
   same unordered pair can have multiple independently recorded directions.
   Before DC-SMACOF optimization, repeated observations are converted to a
   common orientation, summed as DIR8 unit vectors, and mapped to one exact DIR8
   consensus. Invalid, cancelling, or non-DIR8 consensus results raise an error.

   The source CSV is not modified. Optimization uses 43 effective constraints;
   evaluation uses all 44 raw verified observations.

4. **Optimization target and reported direction metrics differ**

   DC-SMACOF optimizes deviation from the central DIR8 unit vector. The reported
   Violation Rate tests whether an edge falls outside the allowed direction
   sector, and Mean Angular Error measures excess angular error under the
   evaluation definition. These are related but are not the same objective.

5. **Iteration and convergence protocol**

   The implementation currently uses a fixed iteration budget rather than a
   paper-derived relative convergence tolerance. A zero-length directional pair
   contributes a zero target vector; its undefined unit-direction contribution
   is skipped only in the internal stress history. This guard does not change
   the position update.

6. **No post-hoc Procrustes for formal DC-SMACOF evaluation**

   In the current formal progressive ablation pipeline, DC-SMACOF is placed in
   the `anchor_align` frame but does not receive the rotation/reflection
   Procrustes correction used by SMACOF. This is an experiment-protocol choice,
   not part of Wang et al.'s core optimizer.

### Recommended manuscript description

> DC-SMACOF was implemented following the vectorized stress-majorization
> framework of Wang et al. (2017), with direction targets updated using the
> current pairwise distance. The distance objective was restricted to the
> observed historical distance-edge set for information-matched comparison.
> Direction weights were normalized by the paired textual distance when
> available and otherwise by the global mean textual distance. Repeated
> direction observations on an unordered pair were combined by DIR8 vector
> consensus before optimization, while direction metrics were evaluated against
> all raw verified observations.

## Reporting checklist

Before writing or updating the final manuscript, verify the following:

- State that both baselines use the observed-edge sparse distance objective.
- Do not claim that either baseline uses all-pairs graph shortest-path distances.
- Attribute the stress-majorization equations to the source papers, but identify
  inverse-square direction normalization and vector-consensus preprocessing as
  project adaptations.
- Keep optimization stress separate from the normalized Stress evaluation
  metric reported in tables.
- Reconcile the normalized Stress denominator with the manuscript formula.
- Record the exact stopping rule and maximum iteration count used by the final
  rerun.
- Record whether each model uses anchor-frame placement, Procrustes alignment,
  scaling, rotation, or reflection.
- Do not combine legacy DC-SMACOF results produced before the current-distance
  target and vector-consensus preprocessing with newly generated results.
- Preserve HPO and ablation configuration files with the method metadata for
  every formal result folder.
