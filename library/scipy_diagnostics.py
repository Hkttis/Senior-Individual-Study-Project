"""Post-hoc diagnostics for PhysicsSim and SciPy BFGS configurations."""

from __future__ import annotations

from collections import deque
from typing import Mapping, Sequence

import numpy as np

from library.config import km2pix, refer_pos_sim
from library.scipy_objective import FixedAnchorObjective, ObjectiveWeights


def reinsert_exact_anchors(
    positions_y_up_sim: Sequence[Sequence[float]],
    problem: FixedAnchorObjective,
) -> tuple[np.ndarray, float]:
    """Return centered coordinates with immutable anchors reinserted exactly."""

    centered = np.asarray(positions_y_up_sim, dtype=np.float64) - np.asarray(
        refer_pos_sim, dtype=np.float64
    )
    if centered.shape != (problem.n_vertices, 2):
        raise ValueError("Configuration shape differs from the objective vertex order.")
    drift = centered[problem.anchor_indices] - problem.anchor_coordinates
    max_drift = float(np.max(np.abs(drift))) if drift.size else 0.0
    centered = centered.copy()
    centered[problem.anchor_indices] = problem.anchor_coordinates
    return centered, max_drift


def clone_with_weights(
    problem: FixedAnchorObjective, weights: ObjectiveWeights
) -> FixedAnchorObjective:
    anchors = {
        int(index): problem.anchor_coordinates[position].copy()
        for position, index in enumerate(problem.anchor_indices)
    }
    return FixedAnchorObjective(
        vertices=problem.vertices,
        distance_pairs=problem.distance_pairs,
        distance_targets=problem.distance_targets,
        direction_pairs=problem.direction_pairs,
        direction_vectors=problem.direction_vectors,
        direction_half_widths=problem.direction_half_widths,
        anchor_positions=anchors,
        weights=weights,
        epsilon=problem.epsilon,
        singularity_tolerance=problem.singularity_tolerance,
    )


def weighted_gradient_components(
    problem: FixedAnchorObjective, free_vector: Sequence[float]
) -> dict[str, np.ndarray]:
    """Return full-node weighted gradients for each objective component."""

    term_weights = {
        "distance": ObjectiveWeights(
            distance=problem.weights.distance, direction=0.0, repulsion=0.0
        ),
        "direction": ObjectiveWeights(
            distance=0.0, direction=problem.weights.direction, repulsion=0.0
        ),
        "repulsion": ObjectiveWeights(
            distance=0.0, direction=0.0, repulsion=problem.weights.repulsion
        ),
    }
    gradients: dict[str, np.ndarray] = {}
    for name, weights in term_weights.items():
        term_problem = clone_with_weights(problem, weights)
        _value, reduced = term_problem.fun_and_jac(free_vector)
        full = np.zeros((problem.n_vertices, 2), dtype=np.float64)
        full[problem.free_indices] = reduced.reshape(problem.n_free_vertices, 2)
        gradients[name] = full
    gradients["total"] = gradients["distance"] + gradients["direction"] + gradients["repulsion"]
    return gradients


def anchor_centroid_radius_rms_km(
    centered_positions: np.ndarray, problem: FixedAnchorObjective
) -> float:
    centroid = problem.anchor_coordinates.mean(axis=0)
    radii_km = np.linalg.norm(centered_positions - centroid, axis=1) / km2pix
    return float(np.sqrt(np.mean(radii_km**2)))


def test_site_radial_errors(
    *,
    centered_positions: np.ndarray,
    target_centered: Mapping[str, np.ndarray],
    test_labels: Sequence[str],
    dni: Mapping[str, int],
    problem: FixedAnchorObjective,
) -> dict[str, float]:
    """Signed errors along the ground-truth radial direction from anchor centroid."""

    centroid = problem.anchor_coordinates.mean(axis=0)
    result: dict[str, float] = {}
    for label in test_labels:
        target = np.asarray(target_centered[label], dtype=np.float64)
        radial = target - centroid
        norm = float(np.linalg.norm(radial))
        if norm <= 1e-12:
            raise ValueError(f"Ground-truth radial direction is undefined for {label}.")
        radial_unit = radial / norm
        error = centered_positions[dni[label]] - target
        result[label] = float(np.dot(error, radial_unit) / km2pix)
    return result


def graph_node_diagnostics(
    *,
    vertices: Sequence[str],
    dni: Mapping[str, int],
    distance_rows: Sequence[Sequence[str]],
    direction_rows: Sequence[Sequence[str]],
    anchor_labels: Sequence[str],
) -> dict[str, dict[str, int]]:
    """Degrees and distance-edge hop count to the nearest calibration anchor."""

    n = len(vertices)
    distance_degree = np.zeros(n, dtype=int)
    direction_degree = np.zeros(n, dtype=int)
    adjacency: list[set[int]] = [set() for _ in range(n)]
    for row in distance_rows:
        u, v = dni[row[0]], dni[row[1]]
        distance_degree[u] += 1
        distance_degree[v] += 1
        adjacency[u].add(v)
        adjacency[v].add(u)
    for row in direction_rows:
        u, v = dni[row[0]], dni[row[1]]
        direction_degree[u] += 1
        direction_degree[v] += 1

    hops = np.full(n, -1, dtype=int)
    queue: deque[int] = deque()
    for label in anchor_labels:
        index = dni[label]
        hops[index] = 0
        queue.append(index)
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if hops[neighbor] < 0:
                hops[neighbor] = hops[current] + 1
                queue.append(neighbor)

    return {
        label: {
            "distance_edge_degree": int(distance_degree[index]),
            "direction_edge_degree": int(direction_degree[index]),
            "distance_graph_hops_to_nearest_anchor": int(hops[index]),
        }
        for index, label in enumerate(vertices)
    }


def assign_objective_strata(
    values: Sequence[float], reference_values: Sequence[float], n_strata: int = 4
) -> tuple[list[int], list[float]]:
    """Split reference objectives at their largest gaps and classify values."""

    reference = np.sort(np.asarray(reference_values, dtype=np.float64))
    if len(reference) < n_strata:
        raise ValueError("Not enough successful reference runs to derive objective strata.")
    gap_indices = np.sort(np.argsort(np.diff(reference))[-(n_strata - 1) :])
    thresholds = [
        float((reference[index] + reference[index + 1]) / 2.0) for index in gap_indices
    ]
    assignments = [int(np.searchsorted(thresholds, value, side="right") + 1) for value in values]
    return assignments, thresholds


__all__ = [
    "anchor_centroid_radius_rms_km",
    "assign_objective_strata",
    "clone_with_weights",
    "graph_node_diagnostics",
    "reinsert_exact_anchors",
    "test_site_radial_errors",
    "weighted_gradient_components",
]
