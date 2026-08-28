"""SciPy-ready realization of the manuscript objective with fixed anchors.

This module implements Sections 4.2--4.4 of the latest manuscript in
``Paper_manuscript_06_23_localrun1/main.pdf``.  The three objective terms are
kept exactly as written there:

    f(X) = w_dis f_dis(X) + w_dir f_dir(X) + w_rep f_rep(X).

The manuscript writes anchor locations as hard equality constraints.  For
unconstrained SciPy minimizers we use the equivalent reduced parameterization:
the optimization vector contains only non-anchor coordinates, while the three
archaeological anchor coordinates are inserted as fixed numerical constants
when the full configuration X is reconstructed.  Consequently no anchor
penalty and no SciPy ``constraints=`` argument are needed.

All coordinates and distance targets in this module use the simulation frame:
the x-axis points east, the y-axis points north, and one simulation unit equals
10 li = 4.15 km.

By default, the objective reproduces the effective coefficients used by the
selected PhysicsSim experiment.  The HPO multipliers are applied on top of the
three legacy base coefficients:

    W_dis = spring_base * w_dis
    W_dir = direction_base * w_dis * 10**alpha
    W_rep = repulsion_base * w_dis * 10**beta

with w_dis=1, alpha=1, and beta=-0.5.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from library.config import (
    DIRECTIONAL_FORCE_MAGNITUDE_BASE,
    FILE_PATHS,
    Li2sim,
    MIN_DISTANCE_BASE,
    REPULSION_STRENGTH_BASE,
    SPRING_STIFFNESS_BASE,
    theta_thr_4dir,
    theta_thr_8dir,
)
from library.directions import DIR4_SIM, DIR8_UNIT_SIM


PHYSICS_HPO_SELECTED_ALPHA = 1.0
PHYSICS_HPO_SELECTED_BETA = -0.5
PHYSICS_HPO_SELECTED_W_DIS = 1.0


# LCC coordinates from data/site_rmse_points.csv, centered at 鄯善 and then
# converted to simulation units.  These numbers are deliberately constants:
# the SciPy optimization problem does not reload or optimize anchor positions.
FIXED_ANCHORS_SIM: Mapping[str, tuple[float, float]] = MappingProxyType(
    {
        "鄯善": (0.0, 0.0),
        "車師前": (11.965621378532099, 106.68680310031324),
        "都護治/烏壘": (-77.72999630158114, 67.61626239867074),
    }
)


class InvalidObjectiveEvaluation(FloatingPointError):
    """Raised when an objective/gradient evaluation cannot return finite values."""


class ObjectiveDomainError(InvalidObjectiveEvaluation, ValueError):
    """Raised where the manuscript's analytic gradient is undefined.

    This includes coincident endpoints of an active distance or direction
    observation and the antipodal branch point of the wrapped direction angle.
    The softened repulsion term remains finite and differentiable at a node
    collision, so repulsion-only collisions are not rejected.
    """


@dataclass(frozen=True)
class ObjectiveWeights:
    """Effective nonnegative coefficients multiplying the objective terms.

    The defaults reproduce the selected existing PhysicsSim experiment after
    applying its HPO multipliers to the three legacy base coefficients.
    """

    distance: float = SPRING_STIFFNESS_BASE * PHYSICS_HPO_SELECTED_W_DIS
    direction: float = (
        DIRECTIONAL_FORCE_MAGNITUDE_BASE
        * PHYSICS_HPO_SELECTED_W_DIS
        * 10.0**PHYSICS_HPO_SELECTED_ALPHA
    )
    repulsion: float = (
        REPULSION_STRENGTH_BASE
        * PHYSICS_HPO_SELECTED_W_DIS
        * 10.0**PHYSICS_HPO_SELECTED_BETA
    )

    def __post_init__(self) -> None:
        values = np.asarray(
            [self.distance, self.direction, self.repulsion], dtype=np.float64
        )
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("Objective weights must be finite and nonnegative.")

    @classmethod
    def from_physics_hpo(
        cls,
        *,
        alpha: float = PHYSICS_HPO_SELECTED_ALPHA,
        beta: float = PHYSICS_HPO_SELECTED_BETA,
        w_dis: float = PHYSICS_HPO_SELECTED_W_DIS,
    ) -> "ObjectiveWeights":
        """Convert legacy PhysicsSim HPO multipliers to effective coefficients."""

        values = np.asarray([alpha, beta, w_dis], dtype=np.float64)
        if not np.all(np.isfinite(values)) or w_dis < 0.0:
            raise ValueError("alpha, beta, and w_dis must be finite; w_dis must be nonnegative.")
        return cls(
            distance=SPRING_STIFFNESS_BASE * w_dis,
            direction=(
                DIRECTIONAL_FORCE_MAGNITUDE_BASE
                * w_dis
                * math.pow(10.0, alpha)
            ),
            repulsion=(
                REPULSION_STRENGTH_BASE
                * w_dis
                * math.pow(10.0, beta)
            ),
        )


@dataclass(frozen=True)
class ObjectiveComponents:
    """Raw and weighted values of the three manuscript objective terms."""

    distance: float
    direction: float
    repulsion: float
    weighted_distance: float
    weighted_direction: float
    weighted_repulsion: float
    total: float


class FixedAnchorObjective:
    """Reduced-variable objective suitable for ``scipy.optimize.minimize``.

    Parameters use zero-based vertex indices.  ``anchor_positions`` maps fixed
    vertex indices to immutable two-dimensional coordinates.  The vector seen
    by SciPy contains only the remaining vertices in ascending index order.
    """

    def __init__(
        self,
        *,
        vertices: Sequence[str],
        distance_pairs: Sequence[Sequence[int]],
        distance_targets: Sequence[float],
        direction_pairs: Sequence[Sequence[int]],
        direction_vectors: Sequence[Sequence[float]],
        direction_half_widths: Sequence[float],
        anchor_positions: Mapping[int, Sequence[float]],
        weights: ObjectiveWeights | None = None,
        epsilon: float = MIN_DISTANCE_BASE,
        singularity_tolerance: float = 1e-12,
    ) -> None:
        self.vertices = tuple(str(name) for name in vertices)
        if not self.vertices or len(set(self.vertices)) != len(self.vertices):
            raise ValueError("vertices must be a non-empty sequence of unique names.")

        self.distance_pairs = self._pairs_array(distance_pairs, "distance_pairs")
        self.distance_targets = np.asarray(distance_targets, dtype=np.float64)
        self.direction_pairs = self._pairs_array(direction_pairs, "direction_pairs")
        self.direction_vectors = np.asarray(direction_vectors, dtype=np.float64)
        if self.direction_vectors.size == 0:
            self.direction_vectors = np.empty((0, 2), dtype=float)
        self.direction_half_widths = np.asarray(
            direction_half_widths, dtype=np.float64
        )
        self.weights = weights if weights is not None else ObjectiveWeights()
        self.epsilon = float(epsilon)
        self.singularity_tolerance = float(singularity_tolerance)

        n = len(self.vertices)
        self._validate_pair_indices(self.distance_pairs, n, "distance_pairs")
        self._validate_pair_indices(self.direction_pairs, n, "direction_pairs")

        if self.distance_targets.shape != (len(self.distance_pairs),):
            raise ValueError("distance_targets must have one value per distance pair.")
        if not np.all(np.isfinite(self.distance_targets)) or np.any(
            self.distance_targets <= 0.0
        ):
            raise ValueError("distance_targets must be finite and strictly positive.")

        expected_direction_shape = (len(self.direction_pairs), 2)
        if self.direction_vectors.shape != expected_direction_shape:
            raise ValueError(
                "direction_vectors must have shape "
                f"{expected_direction_shape}, got {self.direction_vectors.shape}."
            )
        if not np.all(np.isfinite(self.direction_vectors)):
            raise ValueError("direction_vectors must be finite.")
        if len(self.direction_vectors):
            vector_norms = np.linalg.norm(self.direction_vectors, axis=1)
            if not np.allclose(vector_norms, 1.0, rtol=0.0, atol=1e-12):
                raise ValueError("Every direction vector must have unit length.")

        if self.direction_half_widths.shape != (len(self.direction_pairs),):
            raise ValueError(
                "direction_half_widths must have one value per direction pair."
            )
        if not np.all(np.isfinite(self.direction_half_widths)) or np.any(
            (self.direction_half_widths <= 0.0)
            | (self.direction_half_widths >= math.pi)
        ):
            raise ValueError("Direction half-widths must lie strictly in (0, pi).")

        if not math.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError("epsilon must be finite and strictly positive.")
        if (
            not math.isfinite(self.singularity_tolerance)
            or self.singularity_tolerance <= 0.0
        ):
            raise ValueError(
                "singularity_tolerance must be finite and strictly positive."
            )

        if not anchor_positions:
            raise ValueError("At least one fixed anchor is required.")
        normalized_anchors: dict[int, np.ndarray] = {}
        for raw_index, raw_position in anchor_positions.items():
            index = int(raw_index)
            if index < 0 or index >= n:
                raise ValueError(f"Anchor index out of range: {index}.")
            position = np.asarray(raw_position, dtype=np.float64)
            if position.shape != (2,) or not np.all(np.isfinite(position)):
                raise ValueError(
                    f"Anchor {index} must have one finite two-dimensional position."
                )
            normalized_anchors[index] = position.copy()

        self.anchor_indices = np.asarray(sorted(normalized_anchors), dtype=np.int64)
        self.anchor_coordinates = np.vstack(
            [normalized_anchors[index] for index in self.anchor_indices]
        )
        anchor_set = set(int(index) for index in self.anchor_indices)
        self.free_indices = np.asarray(
            [index for index in range(n) if index not in anchor_set], dtype=np.int64
        )
        self.repulsion_pairs = np.column_stack(np.triu_indices(n, k=1)).astype(
            np.int64, copy=False
        )

        for array in (
            self.distance_pairs,
            self.distance_targets,
            self.direction_pairs,
            self.direction_vectors,
            self.direction_half_widths,
            self.anchor_indices,
            self.anchor_coordinates,
            self.free_indices,
            self.repulsion_pairs,
        ):
            array.setflags(write=False)

    @staticmethod
    def _pairs_array(values: Sequence[Sequence[int]], name: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.int64)
        if array.size == 0:
            return np.empty((0, 2), dtype=np.int64)
        if array.ndim != 2 or array.shape[1] != 2:
            raise ValueError(f"{name} must have shape (m, 2).")
        return array.copy()

    @staticmethod
    def _validate_pair_indices(pairs: np.ndarray, n: int, name: str) -> None:
        if len(pairs) == 0:
            return
        if np.any(pairs < 0) or np.any(pairs >= n):
            raise ValueError(f"{name} contains an out-of-range vertex index.")
        if np.any(pairs[:, 0] == pairs[:, 1]):
            raise ValueError(f"{name} cannot contain self-pairs.")

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_free_vertices(self) -> int:
        return len(self.free_indices)

    @property
    def dimension(self) -> int:
        """Number of scalar variables exposed to SciPy."""

        return 2 * self.n_free_vertices

    def unpack(self, free_vector: Sequence[float]) -> np.ndarray:
        """Insert fixed numerical anchors and return the full ``(n, 2)`` X."""

        free = np.asarray(free_vector, dtype=np.float64)
        if free.ndim != 1 or free.size != self.dimension:
            raise ValueError(
                f"Expected a flat free vector of length {self.dimension}, "
                f"got shape {free.shape}."
            )
        if not np.all(np.isfinite(free)):
            raise ValueError("The free optimization vector must be finite.")

        full = np.empty((self.n_vertices, 2), dtype=float)
        full[self.anchor_indices] = self.anchor_coordinates
        full[self.free_indices] = free.reshape(self.n_free_vertices, 2)
        return full

    def pack(self, full_positions: Sequence[Sequence[float]]) -> np.ndarray:
        """Extract only non-anchor coordinates from a full configuration."""

        full = np.asarray(full_positions, dtype=np.float64)
        if full.shape != (self.n_vertices, 2):
            raise ValueError(
                f"Expected full positions with shape {(self.n_vertices, 2)}, "
                f"got {full.shape}."
            )
        if not np.all(np.isfinite(full)):
            raise ValueError("Full positions must be finite.")
        return full[self.free_indices].reshape(-1).copy()

    def components(self, free_vector: Sequence[float]) -> ObjectiveComponents:
        """Evaluate the three exact manuscript terms and their weighted sum."""

        components, _gradient = self._components_and_gradient(
            free_vector, compute_gradient=False
        )
        return components

    def fun(self, free_vector: Sequence[float]) -> float:
        """Scalar callable for ``scipy.optimize.minimize``."""

        return self.components(free_vector).total

    def jac(self, free_vector: Sequence[float]) -> np.ndarray:
        """Analytic gradient with respect to the reduced free vector."""

        _components, gradient = self._components_and_gradient(
            free_vector, compute_gradient=True
        )
        assert gradient is not None
        return gradient

    def fun_and_jac(self, free_vector: Sequence[float]) -> tuple[float, np.ndarray]:
        """Combined SciPy callable for ``minimize(..., jac=True)``."""

        components, gradient = self._components_and_gradient(
            free_vector, compute_gradient=True
        )
        assert gradient is not None
        return components.total, gradient

    def _components_and_gradient(
        self, free_vector: Sequence[float], *, compute_gradient: bool
    ) -> tuple[ObjectiveComponents, np.ndarray | None]:
        try:
            with np.errstate(divide="raise", invalid="raise", over="raise"):
                components, gradient = self._components_and_gradient_unchecked(
                    free_vector, compute_gradient=compute_gradient
                )
        except ObjectiveDomainError:
            raise
        except FloatingPointError as exc:
            raise InvalidObjectiveEvaluation(
                f"Floating-point failure during objective evaluation: {exc}"
            ) from exc

        component_values = np.asarray(
            [
                components.distance,
                components.direction,
                components.repulsion,
                components.weighted_distance,
                components.weighted_direction,
                components.weighted_repulsion,
                components.total,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(component_values)):
            raise InvalidObjectiveEvaluation("Non-finite objective component or total.")
        if gradient is not None:
            gradient = np.asarray(gradient, dtype=np.float64).reshape(-1)
            if gradient.shape != (self.dimension,):
                raise InvalidObjectiveEvaluation(
                    "Analytic gradient has an unexpected shape: "
                    f"{gradient.shape}, expected {(self.dimension,)}."
                )
            if not np.all(np.isfinite(gradient)):
                raise InvalidObjectiveEvaluation("Non-finite analytic gradient.")
        return components, gradient

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
            hinge = np.maximum(0.0, np.abs(angles) - self.direction_half_widths)
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

    def _require_nonzero(self, lengths: np.ndarray, edge_type: str) -> None:
        # Both ||r||-based distance gradients (for a positive target distance)
        # and angular direction gradients are undefined at r=0.  Keep this as
        # a domain failure instead of supplying a dummy line-search gradient.
        invalid = lengths <= self.singularity_tolerance
        if np.any(invalid):
            edge = int(np.flatnonzero(invalid)[0])
            raise ObjectiveDomainError(
                f"{edge_type.capitalize()} {edge} has coincident endpoints; "
                "the manuscript gradient is undefined there."
            )


def build_current_objective(
    *,
    weights: ObjectiveWeights | None = None,
    epsilon: float = MIN_DISTANCE_BASE,
) -> FixedAnchorObjective:
    """Build the 35-node manuscript objective from the verified edge CSVs.

    Anchor coordinates come exclusively from ``FIXED_ANCHORS_SIM``; the site
    coordinate CSV is intentionally not read here.
    """

    vertices: list[str] = []
    vertex_index: dict[str, int] = {}
    distance_pairs: list[tuple[int, int]] = []
    distance_targets: list[float] = []

    with open(FILE_PATHS["chen_data"], newline="", encoding="utf-8-sig") as stream:
        rows = csv.reader(stream)
        next(rows, None)
        for line_number, row in enumerate(rows, start=2):
            if len(row) < 3:
                raise ValueError(f"Malformed distance row {line_number}: {row!r}.")
            source, target, raw_li = row[:3]
            for name in (source, target):
                if name not in vertex_index:
                    vertex_index[name] = len(vertices)
                    vertices.append(name)
            distance_pairs.append((vertex_index[source], vertex_index[target]))
            distance_targets.append(float(raw_li) * Li2sim)

    direction_pairs: list[tuple[int, int]] = []
    direction_vectors: list[np.ndarray] = []
    direction_half_widths: list[float] = []
    with open(
        FILE_PATHS["directional_data"], newline="", encoding="utf-8-sig"
    ) as stream:
        rows = csv.reader(stream)
        next(rows, None)
        for line_number, row in enumerate(rows, start=2):
            if len(row) < 3:
                raise ValueError(f"Malformed direction row {line_number}: {row!r}.")
            source, target, direction_name = row[:3]
            direction_name = direction_name.strip()
            missing = [name for name in (source, target) if name not in vertex_index]
            if missing:
                raise ValueError(
                    f"Direction row {line_number} uses unknown vertices: {missing}."
                )
            if direction_name not in DIR8_UNIT_SIM:
                raise ValueError(
                    f"Direction row {line_number} has unknown direction "
                    f"{direction_name!r}."
                )
            direction_pairs.append((vertex_index[source], vertex_index[target]))
            direction_vectors.append(np.asarray(DIR8_UNIT_SIM[direction_name], dtype=float))
            direction_half_widths.append(
                theta_thr_4dir if direction_name in DIR4_SIM else theta_thr_8dir
            )

    anchor_positions: dict[int, tuple[float, float]] = {}
    for name, position in FIXED_ANCHORS_SIM.items():
        if name not in vertex_index:
            raise ValueError(f"Fixed anchor {name!r} is missing from the distance graph.")
        anchor_positions[vertex_index[name]] = position

    return FixedAnchorObjective(
        vertices=vertices,
        distance_pairs=distance_pairs,
        distance_targets=distance_targets,
        direction_pairs=direction_pairs,
        direction_vectors=direction_vectors,
        direction_half_widths=direction_half_widths,
        anchor_positions=anchor_positions,
        weights=weights,
        epsilon=epsilon,
    )


__all__ = [
    "FIXED_ANCHORS_SIM",
    "FixedAnchorObjective",
    "InvalidObjectiveEvaluation",
    "ObjectiveComponents",
    "ObjectiveDomainError",
    "ObjectiveWeights",
    "PHYSICS_HPO_SELECTED_ALPHA",
    "PHYSICS_HPO_SELECTED_BETA",
    "PHYSICS_HPO_SELECTED_W_DIS",
    "build_current_objective",
]
