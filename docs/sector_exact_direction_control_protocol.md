# Sector-based versus Exact-direction Control

## Purpose

This experiment isolates the effect of directional representation inside the
same PhysicsSim framework. It does not compare PhysicsSim with DC-SMACOF and
does not alter the distance, anchor, repulsion, initialization, integration,
HPO, or evaluation protocols.

## Controlled design

- Reference formulation: `PhysicsSim-Full`, using the sector-based squared-hinge directional penalty.
- Control formulation: `PhysicsSim-ExactDir`, using the zero-tolerance exact-direction angular penalty
  \(f_{dir}=\frac{1}{2}w_{dir}\sum \phi_{ij}^{2}\).
- The exact-direction analytic pair force is implemented in
  `library/directional_objectives.py` and selected through the
  `directional_objective` argument. The sector formulation remains the default.
- Both formulations use the same verified distance and direction observations,
  three calibration anchors, 1,001 fixed time-step updates, and random seeds.
- The exact-direction formulation receives an independent HPO using the same
  36-point alpha/beta grid, three-anchor leave-one-anchor-out validation,
  Pareto filtering, and one-SE balanced selection rule as the reference model.
- Held-out test sites are used only after HPO for final evaluation.
- Formal final evaluation uses matched seeds 0-99.

## Evaluation

The official comparison contains:

1. RMSE (km)
2. Stress
3. Violation Rate
4. Mean Angular Error (rad)
5. Crowding Violation Rate (tau = 0.10)
6. Collapse Node Rate (tau = 0.10)
7. Nearest-Neighbor Distance, 5th Quantile (km)
8. Crossing-edge Rate

Violation Rate and Mean Angular Error retain the paper's sector-based reporting
definitions for both formulations. Mean absolute deviation from the nominal
direction is saved only as an audit metric.

Paired differences are defined as `PhysicsSim-ExactDir - PhysicsSim-Full`.
Their 95% confidence intervals use a paired percentile bootstrap of the mean
with 10,000 resamples across matched seeds.

## Safety checks

- Formal mode requires HPO seeds 0-9, final seeds 0-99, and all 36 grid points.
- If the selected HPO candidate lies on an alpha or beta boundary, formal final
  evaluation stops and the affected search range must be expanded.
- Input SHA-256 hashes are saved with the output.
- Saved final coordinates are independently re-evaluated by
  `scripts/verify_sector_exact_control.py`.

## Commands

Read-only preflight:

```powershell
python -m scripts.check_sector_exact_control_preflight
```

Formal experiment:

```powershell
python -m run_paper_script.paper_run ch5-sector-exact-control --outdir outputs/ch5_sector_exact_control_hpo10_final100
```

Independent formal verification:

```powershell
python -m scripts.verify_sector_exact_control --outdir outputs/ch5_sector_exact_control_hpo10_final100 --formal
```

The `--allow-smoke` option is for development only and must not be used for the
reported formal experiment.
