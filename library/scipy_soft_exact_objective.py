"""Softened exact-direction objective for the independent BFGS control.

Only the directional term differs from ``FixedAnchorObjective``.  The legacy
sector objective and the previous unsoftened exact objective remain unchanged.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from library.config import MIN_DISTANCE_BASE
from library.scipy_objective import (
    FixedAnchorObjective,
    ObjectiveComponents,
    ObjectiveDomainError,
    build_current_objective,
)


class SoftenedExactDirectionObjective(FixedAnchorObjective):
    """Fixed-anchor objective with the collision-regularized exact angle loss."""

    def __init__(self, *args, direction_softening_delta: float = MIN_DISTANCE_BASE, **kwargs):
        super().__init__(*args, **kwargs)
        self.direction_softening_delta = float(direction_softening_delta)
        if (
            not math.isfinite(self.direction_softening_delta)
            or self.direction_softening_delta <= 0.0
        ):
            raise ValueError(
                "direction_softening_delta must be finite and strictly positive."
            )

    def _components_and_gradient_unchecked(
        self, free_vector: Sequence[float], *, compute_gradient: bool
    ) -> tuple[ObjectiveComponents, np.ndarray | None]:
        positions = self.unpack(free_vector)
        gradient_full = np.zeros_like(positions) if compute_gradient else None

        distance_value = 0.0
        if len(self.distance_pairs):
            u = self.distance_pairs[:, 0]
            v = self.distance_pairs[:, 1]
            relative = positions[v] - positions[u]
            lengths = np.linalg.norm(relative, axis=1)
            self._require_nonzero(lengths, "distance edge")
            residuals = lengths - self.distance_targets
            distance_value = 0.5 * float(np.dot(residuals, residuals))
            if gradient_full is not None and self.weights.distance != 0.0:
                pair_gradient = residuals[:, None] * relative / lengths[:, None]
                weighted = self.weights.distance * pair_gradient
                np.add.at(gradient_full, u, -weighted)
                np.add.at(gradient_full, v, weighted)

        direction_value = 0.0
        if len(self.direction_pairs):
            u = self.direction_pairs[:, 0]
            v = self.direction_pairs[:, 1]
            relative = positions[v] - positions[u]
            squared_lengths = np.einsum("ij,ij->i", relative, relative)
            lengths = np.sqrt(squared_lengths)
            collisions = lengths <= self.singularity_tolerance
            valid = ~collisions
            angles = np.zeros(len(relative), dtype=np.float64)

            if np.any(valid):
                unit_relative = relative[valid] / lengths[valid, None]
                direction_vectors = self.direction_vectors[valid]
                cross = (
                    unit_relative[:, 0] * direction_vectors[:, 1]
                    - unit_relative[:, 1] * direction_vectors[:, 0]
                )
                dot = np.einsum("ij,ij->i", unit_relative, direction_vectors)
                antipodal = (dot < 0.0) & (
                    np.abs(cross) <= self.singularity_tolerance
                )
                if np.any(antipodal):
                    valid_indices = np.flatnonzero(valid)
                    edge = int(valid_indices[np.flatnonzero(antipodal)[0]])
                    raise ObjectiveDomainError(
                        "Direction edge "
                        f"{edge} lies on the antipodal atan2 branch singularity."
                    )
                angles[valid] = np.arctan2(cross, dot)

            delta_squared = self.direction_softening_delta**2
            denominator = squared_lengths + delta_squared
            numerator = (
                squared_lengths * angles * angles + delta_squared * math.pi**2
            )
            direction_value = 0.5 * float(np.sum(numerator / denominator))

            if gradient_full is not None and self.weights.direction != 0.0:
                pair_gradient = np.zeros_like(relative)
                if np.any(valid):
                    r = relative[valid]
                    phi = angles[valid]
                    denom = denominator[valid]
                    rotated = np.column_stack((-r[:, 1], r[:, 0]))
                    radial = (
                        delta_squared
                        * (phi * phi - math.pi**2)[:, None]
                        * r
                        / (denom * denom)[:, None]
                    )
                    tangential = -phi[:, None] * rotated / denom[:, None]
                    pair_gradient[valid] = radial + tangential
                weighted = self.weights.direction * pair_gradient
                np.add.at(gradient_full, u, -weighted)
                np.add.at(gradient_full, v, weighted)

        repulsion_value = 0.0
        if len(self.repulsion_pairs):
            u = self.repulsion_pairs[:, 0]
            v = self.repulsion_pairs[:, 1]
            relative = positions[v] - positions[u]
            lengths = np.linalg.norm(relative, axis=1)
            softened = lengths + self.epsilon
            repulsion_value = float(
                np.sum(-np.log(softened) - self.epsilon / softened)
            )
            if gradient_full is not None and self.weights.repulsion != 0.0:
                pair_gradient = -relative / (softened[:, None] ** 2)
                weighted = self.weights.repulsion * pair_gradient
                np.add.at(gradient_full, u, -weighted)
                np.add.at(gradient_full, v, weighted)

        weighted_distance = self.weights.distance * distance_value
        weighted_direction = self.weights.direction * direction_value
        weighted_repulsion = self.weights.repulsion * repulsion_value
        components = ObjectiveComponents(
            distance=distance_value,
            direction=direction_value,
            repulsion=repulsion_value,
            weighted_distance=weighted_distance,
            weighted_direction=weighted_direction,
            weighted_repulsion=weighted_repulsion,
            total=weighted_distance + weighted_direction + weighted_repulsion,
        )
        if gradient_full is None:
            return components, None
        return components, gradient_full[self.free_indices].reshape(-1).copy()


def build_current_softened_exact_objective(
    *,
    direction_softening_delta: float = MIN_DISTANCE_BASE,
    **kwargs,
) -> SoftenedExactDirectionObjective:
    """Build the current-data softened exact objective without changing legacy builders."""

    base = build_current_objective(**kwargs)
    return SoftenedExactDirectionObjective(
        vertices=base.vertices,
        distance_pairs=base.distance_pairs,
        distance_targets=base.distance_targets,
        direction_pairs=base.direction_pairs,
        direction_vectors=base.direction_vectors,
        direction_half_widths=base.direction_half_widths,
        anchor_positions=dict(zip(base.anchor_indices, base.anchor_coordinates)),
        weights=base.weights,
        epsilon=base.epsilon,
        singularity_tolerance=base.singularity_tolerance,
        direction_softening_delta=direction_softening_delta,
    )


__all__ = [
    "SoftenedExactDirectionObjective",
    "build_current_softened_exact_objective",
]
