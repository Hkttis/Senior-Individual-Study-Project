"""Directional objective kernels shared by PhysicsSim experiments.

The formal PhysicsSim model uses a sector-based squared-hinge directional
penalty.  The exact-direction control keeps the same oriented angular error
but sets the target to the nominal direction with zero tolerance:

    psi_exact(r) = 0.5 * w_dir * phi(r, q)^2.

This module is intentionally independent of Pymunk so the analytic force can
be checked directly with finite differences.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from library.config import MIN_DISTANCE_BASE
from library.directions import DIR8_UNIT_SIM


DIRECTIONAL_OBJECTIVE_SECTOR = "sector"
DIRECTIONAL_OBJECTIVE_EXACT = "exact"
DIRECTIONAL_OBJECTIVE_SOFT_EXACT = "soft_exact"
DIRECTIONAL_OBJECTIVES = frozenset(
    {
        DIRECTIONAL_OBJECTIVE_SECTOR,
        DIRECTIONAL_OBJECTIVE_EXACT,
        DIRECTIONAL_OBJECTIVE_SOFT_EXACT,
    }
)


def normalize_directional_objective(value: str) -> str:
    mode = str(value).strip().lower().replace("-", "_")
    aliases = {
        "sector": DIRECTIONAL_OBJECTIVE_SECTOR,
        "sector_based": DIRECTIONAL_OBJECTIVE_SECTOR,
        "exact": DIRECTIONAL_OBJECTIVE_EXACT,
        "exact_direction": DIRECTIONAL_OBJECTIVE_EXACT,
        "zero_tolerance": DIRECTIONAL_OBJECTIVE_EXACT,
        "soft_exact": DIRECTIONAL_OBJECTIVE_SOFT_EXACT,
        "softened_exact": DIRECTIONAL_OBJECTIVE_SOFT_EXACT,
    }
    try:
        return aliases[mode]
    except KeyError as exc:
        raise ValueError(
            f"Unknown directional objective {value!r}; expected one of "
            f"{sorted(DIRECTIONAL_OBJECTIVES)}."
        ) from exc


def oriented_angular_error(r: Sequence[float], q: Sequence[float]) -> float:
    """Return the wrapped nominal-minus-current angle in (-pi, pi]."""
    r_arr = np.asarray(r, dtype=float).reshape(2)
    q_arr = np.asarray(q, dtype=float).reshape(2)
    distance = float(np.linalg.norm(r_arr))
    q_norm = float(np.linalg.norm(q_arr))
    if not np.isfinite(distance) or distance <= 0.0:
        raise ValueError("Directional angular error is undefined at a collision.")
    if not np.isfinite(q_norm) or q_norm <= 0.0:
        raise ValueError("Nominal direction q must be finite and nonzero.")
    r_hat = r_arr / distance
    q_hat = q_arr / q_norm
    dot = float(np.dot(r_hat, q_hat))
    cross = float(r_hat[0] * q_hat[1] - r_hat[1] * q_hat[0])
    return float(math.atan2(cross, dot))


def exact_direction_pair_energy_force(
    r: Sequence[float],
    q: Sequence[float],
    directional_weight: float,
    *,
    collision_epsilon: float = 1e-9,
) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Return exact-direction energy and equal-and-opposite pair forces.

    ``r`` follows the project convention ``x_v - x_u``.  The returned forces
    are ordered as ``(force_u, force_v)`` and satisfy ``force_v == -force_u``.
    The exact-direction objective is undefined at a direction-edge collision;
    this function raises instead of silently inventing a force.
    """
    r_arr = np.asarray(r, dtype=float).reshape(2)
    distance = float(np.linalg.norm(r_arr))
    weight = float(directional_weight)
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError("directional_weight must be finite and nonnegative.")
    if not np.isfinite(distance) or distance <= float(collision_epsilon):
        raise ValueError("Exact-direction force is undefined at a collision.")

    phi = oriented_angular_error(r_arr, q)
    j_r = np.asarray([-r_arr[1], r_arr[0]], dtype=float)
    force_u = -weight * phi * j_r / (distance * distance)
    force_v = -force_u
    energy = 0.5 * weight * phi * phi
    return float(energy), force_u, force_v, float(phi)


def softened_exact_direction_pair_energy_force(
    r: Sequence[float],
    q: Sequence[float],
    directional_weight: float,
    *,
    delta: float = MIN_DISTANCE_BASE,
    collision_epsilon: float = 1e-12,
) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Return softened exact-direction energy and equal-and-opposite forces.

    At a direction-edge collision, the C1 extension has energy
    ``0.5 * directional_weight * pi**2`` and zero force.  The returned angle
    is NaN at that point because a zero-length edge has no direction.
    """

    r_arr = np.asarray(r, dtype=float).reshape(2)
    q_arr = np.asarray(q, dtype=float).reshape(2)
    weight = float(directional_weight)
    softening = float(delta)
    tolerance = float(collision_epsilon)
    if not np.all(np.isfinite(r_arr)):
        raise ValueError("Direction-edge vector must be finite.")
    q_norm = float(np.linalg.norm(q_arr))
    if not np.isfinite(q_norm) or q_norm <= 0.0:
        raise ValueError("Nominal direction q must be finite and nonzero.")
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError("directional_weight must be finite and nonnegative.")
    if not np.isfinite(softening) or softening <= 0.0:
        raise ValueError("delta must be finite and strictly positive.")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("collision_epsilon must be finite and nonnegative.")

    squared_distance = float(np.dot(r_arr, r_arr))
    distance = math.sqrt(squared_distance)
    if distance <= tolerance:
        zero = np.zeros(2, dtype=float)
        return 0.5 * weight * math.pi**2, zero, zero.copy(), float("nan")

    phi = oriented_angular_error(r_arr, q_arr / q_norm)
    delta_squared = softening * softening
    denominator = squared_distance + delta_squared
    numerator = squared_distance * phi * phi + delta_squared * math.pi**2
    energy = 0.5 * weight * numerator / denominator

    j_r = np.asarray([-r_arr[1], r_arr[0]], dtype=float)
    radial = (
        delta_squared
        * (phi * phi - math.pi**2)
        * r_arr
        / (denominator * denominator)
    )
    tangential = -phi * j_r / denominator
    force_u = weight * (radial + tangential)
    force_v = -force_u
    return float(energy), force_u, force_v, float(phi)


def mean_absolute_nominal_angular_deviation(
    positions: Sequence[Sequence[float]],
    directional_data: Sequence[Sequence[str]],
    dni: Mapping[str, int],
    *,
    collision_epsilon: float = 1e-9,
) -> float:
    """Mean ``abs(phi)`` over valid direction observations, in radians."""
    points = np.asarray(positions, dtype=float)
    deviations: list[float] = []
    for row in directional_data:
        if len(row) < 3:
            continue
        source, target, direction_name = row[0], row[1], str(row[2]).strip()
        if source not in dni or target not in dni or direction_name not in DIR8_UNIT_SIM:
            continue
        r = points[dni[target]] - points[dni[source]]
        if float(np.linalg.norm(r)) <= float(collision_epsilon):
            raise ValueError(
                f"Direction collision encountered for observation {source}->{target}."
            )
        deviations.append(abs(oriented_angular_error(r, DIR8_UNIT_SIM[direction_name])))
    if not deviations:
        return float("nan")
    return float(np.mean(np.asarray(deviations, dtype=float)))
