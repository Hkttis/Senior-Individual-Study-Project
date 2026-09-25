"""Exact-direction angular objective with the existing BFGS coordinate contract.

Only f_dir changes: 0.5 * sum(phi**2), with no sector dead zone. Distance,
softened repulsion, fixed numerical anchors, free-coordinate reduction, and
strict collision/antipodal-domain handling are inherited unchanged. Reporting
still uses the original sector-based direction metrics.
"""
from typing import Sequence
import numpy as np
from library.scipy_objective import FixedAnchorObjective, ObjectiveComponents, ObjectiveDomainError, build_current_objective


class ExactDirectionObjective(FixedAnchorObjective):
    # Kept as a separate kernel so the existing sector implementation stays untouched.
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
            lengths = np.linalg.norm(relative, axis=1)
            self._require_nonzero(lengths, "direction edge")
            unit_relative = relative / lengths[:, None]

            cross = (
                unit_relative[:, 0] * self.direction_vectors[:, 1]
                - unit_relative[:, 1] * self.direction_vectors[:, 0]
            )
            dot = np.einsum("ij,ij->i", unit_relative, self.direction_vectors)
            antipodal = (dot < 0.0) & (np.abs(cross) <= self.singularity_tolerance)
            if np.any(antipodal):
                edge = int(np.flatnonzero(antipodal)[0])
                raise ObjectiveDomainError(
                    "Direction edge "
                    f"{edge} lies on the antipodal atan2 branch singularity."
                )

            angles = np.arctan2(cross, dot)
            hinge = np.abs(angles)
            direction_value = 0.5 * float(np.dot(hinge, hinge))

            if gradient_full is not None and self.weights.direction != 0.0:
                active = hinge > 0.0
                if np.any(active):
                    relative_active = relative[active]
                    rotated = np.column_stack(
                        (-relative_active[:, 1], relative_active[:, 0])
                    )
                    pair_gradient = (
                        -hinge[active, None]
                        * np.sign(angles[active])[:, None]
                        * rotated
                        / (lengths[active, None] ** 2)
                    )
                    weighted = self.weights.direction * pair_gradient
                    np.add.at(gradient_full, u[active], -weighted)
                    np.add.at(gradient_full, v[active], weighted)

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
        total = weighted_distance + weighted_direction + weighted_repulsion
        components = ObjectiveComponents(
            distance=distance_value,
            direction=direction_value,
            repulsion=repulsion_value,
            weighted_distance=weighted_distance,
            weighted_direction=weighted_direction,
            weighted_repulsion=weighted_repulsion,
            total=total,
        )

        if gradient_full is None:
            return components, None
        reduced_gradient = gradient_full[self.free_indices].reshape(-1).copy()
        return components, reduced_gradient


def build_current_exact_objective(**kwargs) -> ExactDirectionObjective:
    base = build_current_objective(**kwargs)
    return ExactDirectionObjective(
        vertices=base.vertices, distance_pairs=base.distance_pairs,
        distance_targets=base.distance_targets, direction_pairs=base.direction_pairs,
        direction_vectors=base.direction_vectors,
        direction_half_widths=base.direction_half_widths,
        anchor_positions=dict(zip(base.anchor_indices, base.anchor_coordinates)),
        weights=base.weights, epsilon=base.epsilon,
        singularity_tolerance=base.singularity_tolerance,
    )
