"""Isolated fold-specific objectives for SciPy BFGS hyperparameter search.

The formal objective builder in :mod:`library.scipy_objective` is intentionally
left unchanged.  This module copies its immutable graph terms into a new
``FixedAnchorObjective`` whose fixed set contains exactly two calibration
anchors for one leave-one-anchor-out fold.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from library.scipy_objective import (
    FIXED_ANCHORS_SIM,
    FixedAnchorObjective,
    ObjectiveWeights,
    build_current_objective,
)


def build_bfgs_hpo_fold_objective(
    *,
    fixed_anchor_positions_sim: Mapping[str, Sequence[float]],
    weights: ObjectiveWeights,
) -> FixedAnchorObjective:
    """Build a 2-anchor, 66-variable HPO objective in the supplied fold frame."""

    labels = tuple(fixed_anchor_positions_sim)
    allowed = set(FIXED_ANCHORS_SIM)
    if len(labels) != 2 or len(set(labels)) != 2:
        raise ValueError("A BFGS HPO fold must fix exactly two distinct anchors.")
    unexpected = set(labels) - allowed
    if unexpected:
        raise ValueError(
            "BFGS HPO fixed positions may contain calibration anchors only; "
            f"unexpected labels: {sorted(unexpected)}"
        )

    formal = build_current_objective(weights=weights)
    index_by_label = {label: index for index, label in enumerate(formal.vertices)}
    fixed_by_index: dict[int, np.ndarray] = {}
    for label, raw_position in fixed_anchor_positions_sim.items():
        if label not in index_by_label:
            raise ValueError(f"Calibration anchor is absent from the graph: {label}")
        position = np.asarray(raw_position, dtype=np.float64)
        if position.shape != (2,) or not np.all(np.isfinite(position)):
            raise ValueError(f"Anchor {label} must have one finite 2-D position.")
        fixed_by_index[index_by_label[label]] = position.copy()

    fold_problem = FixedAnchorObjective(
        vertices=formal.vertices,
        distance_pairs=formal.distance_pairs,
        distance_targets=formal.distance_targets,
        direction_pairs=formal.direction_pairs,
        direction_vectors=formal.direction_vectors,
        direction_half_widths=formal.direction_half_widths,
        anchor_positions=fixed_by_index,
        weights=weights,
        epsilon=formal.epsilon,
        singularity_tolerance=formal.singularity_tolerance,
    )
    if fold_problem.n_vertices != 35 or fold_problem.dimension != 66:
        raise ValueError(
            "Unexpected BFGS HPO dimension: "
            f"n={fold_problem.n_vertices}, d={fold_problem.dimension}."
        )
    return fold_problem


__all__ = ["build_bfgs_hpo_fold_objective"]
