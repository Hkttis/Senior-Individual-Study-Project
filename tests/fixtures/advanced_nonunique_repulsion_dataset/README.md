# Advanced sparse non-rigid repulsion dataset

This fixture is a sparse stress test for comparing PhysicsSim-Full with
PhysicsSim-NoRep. It is separate from the formal Western Regions data and from
the globally unique small-model fixture.

## Design

- Nodes: 35
- Distance edges: 40 of 595 possible edges (6.72 percent)
- Direction observations: 26
- Distance-degree-1 nodes: 14
- Direction observations incident to distance leaves: 0
- Calibration nodes: `P00` (`anchor_align`), `P07`, and `P08`
- Strict distance-edge crossings in the expected layout: 0

The distance graph contains a 34-edge spanning tree plus six local braces. All
distance edges join nearby grid nodes and have length 300, 400, or 500
synthetic units. Fourteen nodes are leaves in the distance graph.

The graph is intentionally not locally rigid. Its 40-row rigidity matrix cannot
reach the 2D local-rigidity rank `2n - 3 = 67`. The three calibration anchors
form a fixed non-collinear 3-4-5 triangle and none is a leaf.

Every distance leaf has no direction observation. At experiment initialization,
all non-anchor nodes are sampled independently and uniformly over the complete
expected-position bounding box. The three anchors are then reset to their known
positions. Paired Full/NoRep runs use the same initial layout for each seed.

The ambiguous test node `P34` has one distance edge to `P27`. It can move
continuously on a radius-400 circle while preserving every observed constraint.
In `alternative_positions.csv`, `P34` coincides with `P20`; the expected
and alternative layouts satisfy exactly the same observed distance and
direction data but are not congruent.

Direction observations are restricted to the non-leaf core and remain fewer
than distance observations. This design tests repulsion as a selection bias in
a genuinely underdetermined leaf configuration rather than after directions
have already fixed the leaves.

## Commands

```text
python -m pytest tests/test_advanced_nonunique_repulsion_dataset.py -q
python -m scripts.run_advanced_repulsion_synthetic --seeds 5,6,7,8,9 --outdir outputs/advanced_sparse_random_init_repulsion_seed5_9
python -m scripts.run_advanced_sparse_baselines --seeds 0,1,2,3,4,5,6,7,8,9 --outdir outputs/advanced_sparse_free_leaves_baselines_seed0_9
```
