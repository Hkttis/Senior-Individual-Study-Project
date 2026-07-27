# Small model dataset

This fixture provides one shared synthetic dataset for production-level tests
of SMACOF, DC-SMACOF, and the four PhysicsSim variants. It is deliberately
more irregular and less dense than the earlier rectangular fixture.

## Known layout

Coordinates use an arbitrary planar y-up coordinate system. The ten nodes do
not form a regular grid, and the observed direction bearings are not restricted
to multiples of 45 degrees.

- Nodes: 10
- Distance edges: 24 (a complete graph would contain 45)
- Direction observations: 12
- Calibration nodes: `A` (`anchor_align`), `B`, and `C`
- Test nodes: `D` through `J`
- Several direction observations do not have a matching distance edge.

All stored distances are exact integers. Direction labels remain the project's
coarse DIR8 observations, while `expected_bearing_deg` records the continuous
bearing implied by the known layout. The latter is test metadata and is not
passed to the production models.

## Rigidity and uniqueness

The distance graph is a 2D trilateration graph:

1. `A`, `B`, and `C` form a non-collinear base triangle.
2. Every later node is connected to three already localized, non-collinear
   parent nodes, as recorded in `trilateration_order.csv`.
3. Three distances to non-collinear known centers determine each new point
   uniquely. Induction therefore gives one realization up to Euclidean
   congruence (translation, rotation, and reflection).

The automated test additionally verifies that the 2D rigidity matrix has rank
`2n - 3`, which certifies infinitesimal/local rigidity at this realization.

## Files

- `distance_edges.csv`: model-ready distance constraints.
- `direction_edges.csv`: model-ready DIR8 observations plus audit bearings.
- `expected_positions.csv`: known positions and anchor/test roles.
- `trilateration_order.csv`: machine-checkable uniqueness certificate.

## Test command

```text
python -m pytest tests/test_small_model_dataset.py -q
```

The test checks the data contract, sparse-rigidity certificate, and metric
implementation, then runs the existing production functions for SMACOF,
DC-SMACOF, PhysicsSim-DistOnly, PhysicsSim-DistDir,
PhysicsSim-DistDirAnch, and PhysicsSim-Full with shortened iteration counts.
It verifies reconstruction RMSE, Stress, Violation Rate, Mean Angular Error,
and anchor error against deterministic thresholds. It is not a replacement for
the formal Western Regions experiment data.
