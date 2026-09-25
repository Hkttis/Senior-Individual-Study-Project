# Reconstructing candidate country-center configurations in the Western Han Western Regions

This repository accompanies a constraint-based historical-geographic reconstruction study. The
research question is how an overall configuration of country centers can be inferred when most
countries cannot be securely assigned precise modern coordinates, but historical sources preserve
inter-country distances and coarse relative directions. The output is a set of **candidate
configurations conditional on the evidence and modeling assumptions**, not a definitive map of
ancient coordinates.

The working case is the Western Regions described in the *Xiyu zhuan* of the *Hanshu*. The paper
formulates the task as an inverse problem: a candidate configuration is compared with the
relations expected under an explicit working forward model. That model approximates interpreted
route-derived distances by Euclidean separation, interprets textual directions as admissible
angular sectors, and uses selected archaeological site identifications as geographic references.

## Evidence and reconstruction model

The paper dataset contains 35 country nodes, 44 distance edges, 44 directional observations, and
11 archaeological reference sites. Long route records were interpreted and decomposed following
the stated historical-geographic source procedure; textual directions were checked against the
historical text. The verified computational inputs are in `data/`.

For positions `X`, the reconstruction objective has three components:

```text
F(X) = w_dis f_dis(X) + w_dir f_dir(X) + w_rep f_rep(X)
```

`f_dis` measures squared discrepancies between modeled and target distances. `f_dir` penalizes
only angular deviations beyond an allowed direction sector: the paper uses 90 degrees as the
four-direction half-width and 45 degrees as the eight-direction half-width. Archaeological anchors
are positional constraints for methods that use anchors. The softened repulsion term is a
regularizer that discourages crowding; it is a preference over configurations, not another
historical observation.

The distance records cannot all be fitted exactly under the Euclidean approximation, and the
sparse observations do not uniquely determine every country's position. The objective therefore
seeks an approximate, regularized reconstruction. `library/physics.py` generates candidate
configurations by a prescribed number of steps of a damped Pymunk simulation. The appendix
derives an **ideal continuous-time** mechanical realization of the objective and characterizes
its regular stationary states. Its energy and stationarity results do not assert convergence or
global optimality for the finite-step software simulation.

## Evaluation design

In the primary split, three archaeological sites serve as calibration sites; eight further sites
are held out for external evaluation. The calibration sites support alignment and hyperparameter
validation and, where applicable, become model anchors. Held-out coordinates do not enter model
generation or hyperparameter selection. Coordinates are projected into a common Lambert
Conformal Conic evaluation space. Alignment uses only the geometric freedoms left by each model.

The evaluation combines held-out-site RMSE with distance Stress, direction Violation Rate, Mean
Angular Error, and layout diagnostics for crowding, collapse, nearest-neighbor separation, and
crossing edges. These measures answer different questions: fitting textual relations is not the
same as recovering held-out archaeological positions.

The paper compares PhysicsSim with Random+Align, distance-only SMACOF, direction-constrained
DC-SMACOF, and full-memory BFGS. PhysicsSim ablations successively add directional information,
anchor constraints, and repulsion. BFGS polishing starts from PhysicsSim endpoints and tests
whether further minimization of the same weighted objective improves external accuracy. Separate
analyses vary the anchor/test split, route-distance scale, and direction-sector width. The
direction-sector sensitivity and other exploratory controls should be interpreted according to
their own recorded protocols, not as changes to the primary experiment.

## Reproducing the computations

Run commands from this directory. The central dispatcher is:

```text
python -m run_paper_script.paper_run --help
```

Use the corresponding `scripts/check_*` preflight before a formal run. HPO, ablation, BFGS,
robustness, and sensitivity commands live under `run_paper_script/`. Each run should have a
distinct `--outdir`; output configuration files record the chosen inputs and hyperparameters.
Tables and figures are built from saved experiment outputs by the `scripts/export_*`,
`scripts/build_manuscript_ready_results.py`, and `scripts/update_paper_results.py` tools.
Their `scripts/verify_*` counterparts check the exports against source results.

Run the maintained test suite with:

```text
python -m pytest -q tests
```

The directory roles and the status of historical scripts are documented in
[`docs/code_inventory.md`](docs/code_inventory.md). `outputs/`, `paper_results/`, virtual
environments, caches, and local backups are not versioned. The repository is the source-code
record; a Git checkout alone does not contain every numerical run or manuscript figure.

## Scope

The reconstruction remains conditional on route interpretation, site identification, angular
tolerance, anchor selection, and hyperparameter choices. Repeated candidate runs do not describe
the full feasible solution space or a probability distribution over historical locations. The
paper and appendix discuss these limitations and distinguish the ideal objective-based analysis
from the implemented finite-step procedure.
