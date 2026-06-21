"""Coordinate helpers for the progressive ablation study.

The anchor frame is placed before any Procrustes transform.  Similarity
alignment therefore rotates/scales about the explicit frame anchor rather than
using an unconstrained centroid translation.
"""

from __future__ import annotations

from itertools import combinations
from typing import Mapping, Sequence

import numpy as np


def place_in_anchor_frame(
    positions: Sequence[Sequence[float]],
    dni: Mapping[str, int],
    anchor_label: str,
    refer_pos: Sequence[float],
) -> np.ndarray:
    """Translate positions so the explicit frame anchor lands at ``refer_pos``."""
    if anchor_label not in dni:
        raise KeyError(f"Frame anchor {anchor_label!r} is not a graph node.")
    points = np.asarray(positions, dtype=float)
    anchor = points[dni[anchor_label]]
    return points - anchor + np.asarray(refer_pos, dtype=float)


def anchored_similarity_procrustes(
    positions: Sequence[Sequence[float]],
    dni: Mapping[str, int],
    calibration_labels: Sequence[str],
    target_positions: Mapping[str, Sequence[float]],
    anchor_label: str,
    refer_pos: Sequence[float],
    *,
    allow_scaling: bool,
) -> np.ndarray:
    """Fit rotation/reflection, optionally scale, about a fixed frame anchor."""
    labels = list(calibration_labels)
    if anchor_label not in labels:
        raise ValueError("anchor_label must be explicitly included in calibration_labels.")
    if len(labels) < 2:
        raise ValueError("At least two calibration labels are required for Procrustes alignment.")
    missing = [label for label in labels if label not in dni or label not in target_positions]
    if missing:
        raise KeyError(f"Calibration labels missing from graph/targets: {missing}")

    points = np.asarray(positions, dtype=float)
    source = np.asarray([points[dni[label]] for label in labels], dtype=float)
    target = np.asarray([target_positions[label] for label in labels], dtype=float)
    anchor_idx = labels.index(anchor_label)
    source_centered = source - source[anchor_idx]
    target_centered = target - target[anchor_idx]

    covariance = source_centered.T @ target_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation_or_reflection = u @ vt
    scale = 1.0
    if allow_scaling:
        rotated_source = source_centered @ rotation_or_reflection
        denominator = float(np.sum(source_centered * source_centered))
        if np.isclose(denominator, 0.0):
            raise ValueError("Degenerate calibration geometry cannot determine a similarity scale.")
        scale = float(np.sum(rotated_source * target_centered) / denominator)
        if scale <= 0.0:
            raise ValueError("Similarity alignment produced a non-positive scale.")

    source_all_centered = points - points[dni[anchor_label]]
    return source_all_centered @ rotation_or_reflection * scale + np.asarray(refer_pos, dtype=float)


def anchor_geometry_is_non_degenerate(
    positions: Sequence[Sequence[float]],
    dni: Mapping[str, int],
    calibration_labels: Sequence[str],
    *,
    min_distance: float,
    min_triangle_area: float,
) -> bool:
    """Check the three random calibration anchors before similarity fitting."""
    labels = list(calibration_labels)
    if len(labels) != 3:
        raise ValueError("Random+Align requires exactly three calibration anchors.")
    points = np.asarray(positions, dtype=float)
    anchors = np.asarray([points[dni[label]] for label in labels], dtype=float)
    if any(np.linalg.norm(anchors[i] - anchors[j]) < min_distance for i, j in combinations(range(3), 2)):
        return False
    ab = anchors[1] - anchors[0]
    ac = anchors[2] - anchors[0]
    area = abs(float(ab[0] * ac[1] - ab[1] * ac[0])) / 2.0
    return bool(area >= min_triangle_area)


def sample_non_degenerate_unit_square_layout(
    n_nodes: int,
    dni: Mapping[str, int],
    calibration_labels: Sequence[str],
    rng: np.random.Generator,
    *,
    min_distance: float = 0.05,
    min_triangle_area: float = 0.005,
    max_attempts: int = 10_000,
) -> tuple[np.ndarray, int]:
    """Sample a unit-square layout, rejecting degenerate calibration triangles."""
    for attempt in range(1, max_attempts + 1):
        positions = rng.uniform(0.0, 1.0, size=(n_nodes, 2))
        if anchor_geometry_is_non_degenerate(
            positions,
            dni,
            calibration_labels,
            min_distance=min_distance,
            min_triangle_area=min_triangle_area,
        ):
            return positions, attempt
    raise RuntimeError(f"Could not sample a non-degenerate random anchor geometry in {max_attempts} attempts.")
