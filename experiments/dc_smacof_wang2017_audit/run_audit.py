"""Isolated audit of the Wang et al. edge-direction D-step.

This module does not modify or monkeypatch the production DC-SMACOF model.
It compares:

* ``wang_current``: d'_ij = ||x_i - x_j|| u_ij on every direction edge.
* ``production_proxy``: fixed text distance, or the mean text distance when
  the direction edge has no corresponding distance observation.

The incidence convention is +1 at the source and -1 at the target. Therefore
the target vector stored in D is source minus target; a source-to-target DIR8
unit vector must consequently be negated.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.linalg import solve_triangular

from library.config import FILE_PATHS, refer_pos_sim
from library.data_io import (
    get_anchor_align_label,
    get_anchor_labels,
    get_test_site_labels,
    load_ini_data_from_csv,
    uploading_directional_data,
    uploading_ground_truth,
)
from library.directions import DIR8_UNIT_SIM
from library.metrics import (
    alignment_and_scaling,
    calculate_kruskals_stress,
    direction_violation_rate,
    mean_angular_error_violations,
)
from library.units import data_Li2sim, pos_matrix_sim2km
from run_paper_script.ch5_hparam_kfold_gridsearch_pareto import _rmse_labels_km


AUDIT_ROOT = Path(__file__).resolve().parent
PRODUCTION_SNAPSHOT = AUDIT_ROOT / "snapshots" / "directed_mds_model_production_before_audit.py"
MODES = ("production_proxy", "wang_current")


@dataclass(frozen=True)
class AuditProblem:
    labels: tuple[str, ...]
    dni: dict[str, int]
    distance_edges: tuple[tuple[int, int, float], ...]
    direction_edges: tuple[tuple[int, int, np.ndarray, str, str, str], ...]
    mean_text_distance: float
    distance_weight: float
    direction_weight: float
    distance_incidence: np.ndarray
    direction_incidence: np.ndarray
    distance_weights: np.ndarray
    direction_weights: np.ndarray
    left: np.ndarray
    left_reduced: np.ndarray
    cholesky: np.ndarray
    condition_number: float


def _incidence(n: int, pairs: Sequence[tuple[int, int]]) -> np.ndarray:
    matrix = np.zeros((n, len(pairs)), dtype=float)
    for col, (source, target) in enumerate(pairs):
        matrix[source, col] = 1.0
        matrix[target, col] = -1.0
    return matrix


def build_problem(
    labels: Sequence[str],
    dni: dict[str, int],
    distance_rows: Iterable[Sequence[object]],
    direction_rows: Iterable[Sequence[object]],
    *,
    distance_weight: float = 1.0,
    direction_weight: float = 10.0 ** -0.5,
) -> AuditProblem:
    distance_edges: list[tuple[int, int, float]] = []
    distance_by_pair: dict[frozenset[int], float] = {}
    for row in distance_rows:
        source, target, distance = str(row[0]), str(row[1]), float(row[2])
        i, j = dni[source], dni[target]
        if not math.isfinite(distance) or distance <= 0:
            raise ValueError(f"Invalid distance edge: {row!r}")
        distance_edges.append((i, j, distance))
        distance_by_pair[frozenset((i, j))] = distance
    if not distance_edges:
        raise ValueError("At least one distance edge is required.")

    mean_text_distance = float(np.mean([edge[2] for edge in distance_edges]))
    direction_edges: list[tuple[int, int, np.ndarray, str, str, str]] = []
    for row in direction_rows:
        source, target, direction = str(row[0]), str(row[1]), str(row[2])
        if source not in dni or target not in dni:
            raise ValueError(f"Unknown direction endpoint: {row!r}")
        if direction not in DIR8_UNIT_SIM:
            raise ValueError(f"Unknown DIR8 direction: {row!r}")
        direction_edges.append(
            (dni[source], dni[target], np.asarray(DIR8_UNIT_SIM[direction], dtype=float), source, target, direction)
        )

    n = len(labels)
    b_distance = _incidence(n, [(i, j) for i, j, _ in distance_edges])
    b_direction = _incidence(n, [(i, j) for i, j, *_ in direction_edges])
    w_distance = np.asarray([distance_weight / (d * d) for _, _, d in distance_edges], dtype=float)
    w_direction = np.asarray(
        [
            direction_weight
            / (distance_by_pair.get(frozenset((i, j)), mean_text_distance) ** 2)
            for i, j, *_ in direction_edges
        ],
        dtype=float,
    )
    left = (b_distance * w_distance) @ b_distance.T
    if len(direction_edges):
        left = left + (b_direction * w_direction) @ b_direction.T
    left_reduced = left[1:, 1:]
    cholesky = np.linalg.cholesky(left_reduced)
    return AuditProblem(
        labels=tuple(labels),
        dni=dict(dni),
        distance_edges=tuple(distance_edges),
        direction_edges=tuple(direction_edges),
        mean_text_distance=mean_text_distance,
        distance_weight=float(distance_weight),
        direction_weight=float(direction_weight),
        distance_incidence=b_distance,
        direction_incidence=b_direction,
        distance_weights=w_distance,
        direction_weights=w_direction,
        left=left,
        left_reduced=left_reduced,
        cholesky=cholesky,
        condition_number=float(np.linalg.cond(left_reduced)),
    )


def _safe_unit(vector: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, float]:
    length = float(np.linalg.norm(vector))
    if not math.isfinite(length) or length <= eps:
        return np.zeros(2, dtype=float), length
    return vector / length, length


def compute_targets(problem: AuditProblem, positions: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}; expected one of {MODES}")
    distance_targets = np.zeros((len(problem.distance_edges), 2), dtype=float)
    distance_by_pair = {frozenset((i, j)): d for i, j, d in problem.distance_edges}
    for idx, (source, target, ideal_distance) in enumerate(problem.distance_edges):
        unit, _ = _safe_unit(positions[source] - positions[target])
        distance_targets[idx] = ideal_distance * unit

    direction_targets = np.zeros((len(problem.direction_edges), 2), dtype=float)
    for idx, (source, target, unit, *_labels) in enumerate(problem.direction_edges):
        if mode == "wang_current":
            _, target_length = _safe_unit(positions[target] - positions[source])
            if not math.isfinite(target_length) or target_length <= 1e-12:
                target_length = 0.0
        else:
            target_length = distance_by_pair.get(frozenset((source, target)), problem.mean_text_distance)
        direction_targets[idx] = -(target_length * unit)
    return distance_targets, direction_targets


def objective_components(problem: AuditProblem, positions: np.ndarray) -> tuple[float, float]:
    distance_value = 0.0
    for weight, (source, target, ideal_distance) in zip(problem.distance_weights, problem.distance_edges):
        actual = float(np.linalg.norm(positions[source] - positions[target]))
        distance_value += float(weight) * ((actual - ideal_distance) ** 2)

    direction_value = 0.0
    for weight, (source, target, unit, *_labels) in zip(problem.direction_weights, problem.direction_edges):
        vector = positions[target] - positions[source]
        current_unit, current_length = _safe_unit(vector)
        if not math.isfinite(current_length):
            return float("inf"), float("inf")
        direction_value += float(weight) * ((current_length * np.linalg.norm(current_unit - unit)) ** 2)
    return float(distance_value), float(direction_value)


def _layout_scale(positions: np.ndarray) -> tuple[float, float, float]:
    if not np.isfinite(positions).all():
        return float("inf"), float("inf"), float("inf")
    centered = positions - positions.mean(axis=0, keepdims=True)
    max_abs = float(np.max(np.abs(positions)))
    rms_radius = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    differences = positions[:, None, :] - positions[None, :, :]
    max_pair = float(np.max(np.linalg.norm(differences, axis=2)))
    return max_abs, rms_radius, max_pair


def _solve(problem: AuditProblem, right: np.ndarray) -> np.ndarray:
    right_reduced = right[1:, :]
    intermediate = solve_triangular(problem.cholesky, right_reduced, lower=True)
    reduced = solve_triangular(problem.cholesky.T, intermediate, lower=False)
    return np.vstack((np.zeros((1, 2), dtype=float), reduced))


def run_iterations(
    problem: AuditProblem,
    *,
    mode: str,
    seed: int,
    n_iterations: int,
    damping: float = 1.0,
    explosion_limit: float = 1e12,
) -> tuple[np.ndarray, list[dict[str, object]], str, str]:
    if not (0.0 < damping <= 1.0):
        raise ValueError("damping must be in (0, 1].")
    positions = np.random.RandomState(int(seed)).rand(len(problem.labels), 2)
    trace: list[dict[str, object]] = []
    status, failure_reason = "ok", ""

    for iteration in range(n_iterations + 1):
        distance_obj, direction_obj = objective_components(problem, positions)
        max_abs, rms_radius, max_pair = _layout_scale(positions)
        row: dict[str, object] = {
            "mode": mode,
            "seed": int(seed),
            "damping": float(damping),
            "iteration": int(iteration),
            "finite": bool(np.isfinite(positions).all()),
            "max_abs_coordinate_li": max_abs,
            "rms_radius_li": rms_radius,
            "max_pair_distance_li": max_pair,
            "distance_objective": distance_obj,
            "direction_objective": direction_obj,
            "total_objective": distance_obj + direction_obj,
            "step_norm_li": 0.0,
            "candidate_max_abs_coordinate_li": max_abs,
            "condition_number_reduced_laplacian": problem.condition_number,
        }
        trace.append(row)
        if not row["finite"]:
            status, failure_reason = "failed", "non_finite_positions"
            break
        if max_abs > explosion_limit:
            status, failure_reason = "failed", f"max_abs_coordinate_exceeded_{explosion_limit:g}"
            break
        if iteration == n_iterations:
            break

        try:
            distance_targets, direction_targets = compute_targets(problem, positions, mode)
            right = (problem.distance_incidence * problem.distance_weights) @ distance_targets
            if len(problem.direction_edges):
                right = right + (problem.direction_incidence * problem.direction_weights) @ direction_targets
            candidate = _solve(problem, right)
        except Exception as exc:  # diagnostic runner must record the first numerical failure
            status, failure_reason = "failed", f"{type(exc).__name__}: {exc}"
            break

        trace[-1]["candidate_max_abs_coordinate_li"] = _layout_scale(candidate)[0]
        next_positions = positions + float(damping) * (candidate - positions)
        trace[-1]["step_norm_li"] = float(np.linalg.norm(next_positions - positions))
        positions = next_positions

    return positions, trace, status, failure_reason


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parse_ints(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one seed is required.")
    return values


def _parse_floats(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one damping value is required.")
    return values


def _formal_metrics(positions_li: np.ndarray, vertice, dni, data_li, directional_data, gt_lonlat) -> dict[str, float]:
    anchor_label = get_anchor_align_label()
    positions_sim = np.asarray(
        alignment_and_scaling(
            positions_li.tolist(), vertice, dni, refer_pos_sim, y_down=False, anchor_label=anchor_label
        ),
        dtype=float,
    )
    data_sim = data_Li2sim(data_li)
    known_labels = get_anchor_labels() + get_test_site_labels()
    known_lonlat = [tuple(gt_lonlat[dni[label]]) for label in known_labels]
    return {
        "E_distance_stress": float(
            calculate_kruskals_stress(dni, pos_matrix_sim2km(positions_sim.tolist()), data_sim)
        ),
        "E_direction_vr": float(direction_violation_rate(positions_sim, directional_data, dni)),
        "E_direction_mae_rad": float(mean_angular_error_violations(positions_sim, directional_data, dni)),
        "RMSE_test_km": float(
            _rmse_labels_km(
                pos_y_up_sim=positions_sim,
                dni=dni,
                refer_pos_sim=refer_pos_sim,
                gt_labels=known_labels,
                gt_lonlat=known_lonlat,
                eval_labels=get_test_site_labels(),
                anchor_label_for_frame=anchor_label,
            )
        ),
    }


def run_audit(
    *,
    seeds: Sequence[int],
    modes: Sequence[str],
    damping_values: Sequence[float],
    n_iterations: int,
    outdir: Path,
    distance_weight: float,
    direction_weight: float,
    explosion_limit: float,
) -> dict[str, object]:
    outdir.mkdir(parents=True, exist_ok=False)
    graph, vertice, dni, edges, data_li = load_ini_data_from_csv(FILE_PATHS)
    directional_data = uploading_directional_data()
    gt_lonlat = uploading_ground_truth(vertice, dni)
    problem = build_problem(
        vertice,
        dni,
        data_li,
        directional_data,
        distance_weight=distance_weight,
        direction_weight=direction_weight,
    )

    all_trace: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    final_positions: list[dict[str, object]] = []
    for mode in modes:
        for damping in damping_values:
            for seed in seeds:
                print(f"[audit] mode={mode}, damping={damping:g}, seed={seed}")
                positions, trace, status, failure_reason = run_iterations(
                    problem,
                    mode=mode,
                    seed=seed,
                    n_iterations=n_iterations,
                    damping=damping,
                    explosion_limit=explosion_limit,
                )
                all_trace.extend(trace)
                summary: dict[str, object] = {
                    "mode": mode,
                    "damping": float(damping),
                    "seed": int(seed),
                    "status": status,
                    "failure_reason": failure_reason,
                    "iterations_completed": int(trace[-1]["iteration"]),
                    "final_max_abs_coordinate_li": trace[-1]["max_abs_coordinate_li"],
                    "final_max_pair_distance_li": trace[-1]["max_pair_distance_li"],
                    "final_distance_objective": trace[-1]["distance_objective"],
                    "final_direction_objective": trace[-1]["direction_objective"],
                    "condition_number_reduced_laplacian": problem.condition_number,
                }
                if status == "ok":
                    summary.update(_formal_metrics(positions, vertice, dni, data_li, directional_data, gt_lonlat))
                    for label, point in zip(vertice, positions):
                        final_positions.append(
                            {
                                "mode": mode,
                                "damping": float(damping),
                                "seed": int(seed),
                                "label": label,
                                "x_li": float(point[0]),
                                "y_li": float(point[1]),
                            }
                        )
                summaries.append(summary)

    _write_csv(outdir / "iteration_trace.csv", all_trace)
    _write_csv(outdir / "run_summary.csv", summaries)
    _write_csv(outdir / "final_positions_li.csv", final_positions)
    config = {
        "created_at": datetime.now().astimezone().isoformat(),
        "paper_definition": "edge direction D-step: d'_ij = ||x_i-x_j|| u_ij",
        "incidence_convention": "+source, -target; stored direction target is -||target-source||*u",
        "production_source_untouched": str(Path("MDS_model/directed_mds_model.py")),
        "production_snapshot": str(PRODUCTION_SNAPSHOT.relative_to(AUDIT_ROOT.parent.parent)),
        "seeds": [int(seed) for seed in seeds],
        "modes": list(modes),
        "damping_values": [float(value) for value in damping_values],
        "n_iterations": int(n_iterations),
        "distance_weight": float(distance_weight),
        "direction_weight": float(direction_weight),
        "mean_text_distance_li": problem.mean_text_distance,
        "n_distance_edges": len(problem.distance_edges),
        "n_direction_edges": len(problem.direction_edges),
        "n_direction_edges_without_distance": int(
            sum(
                frozenset((source, target))
                not in {frozenset((i, j)) for i, j, _ in problem.distance_edges}
                for source, target, *_ in problem.direction_edges
            )
        ),
        "condition_number_reduced_laplacian": problem.condition_number,
        "explosion_limit": float(explosion_limit),
        "file_paths": {key: str(value) for key, value in FILE_PATHS.items()},
    }
    (outdir / "audit_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    failures = [row for row in summaries if row["status"] != "ok"]
    lines = [
        "# DC-SMACOF Wang et al. (2017) audit log",
        "",
        f"- Created: {config['created_at']}",
        f"- Distance edges: {config['n_distance_edges']}",
        f"- Direction edges: {config['n_direction_edges']}",
        f"- Direction-only edges: {config['n_direction_edges_without_distance']}",
        f"- Reduced Laplacian condition number: {problem.condition_number:.6g}",
        f"- Runs: {len(summaries)}; failures: {len(failures)}",
        "",
        "## Run results",
        "",
        "| mode | damping | seed | status | iterations | max abs (Li) | Stress | VR | RMSE test (km) | failure |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summaries:
        lines.append(
            "| {mode} | {damping:g} | {seed} | {status} | {iterations_completed} | "
            "{final_max_abs_coordinate_li:.6g} | {stress} | {vr} | {rmse} | {failure_reason} |".format(
                stress=(f"{row['E_distance_stress']:.6g}" if "E_distance_stress" in row else "NA"),
                vr=(f"{row['E_direction_vr']:.6g}" if "E_direction_vr" in row else "NA"),
                rmse=(f"{row['RMSE_test_km']:.6g}" if "RMSE_test_km" in row else "NA"),
                **row,
            )
        )
    (outdir / "experiment_log.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"config": config, "summaries": summaries, "failures": failures, "outdir": str(outdir)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the isolated Wang et al. DC-SMACOF numerical audit.")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--modes", default="production_proxy,wang_current")
    parser.add_argument("--damping", default="1.0")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--distance-weight", type=float, default=1.0)
    parser.add_argument("--direction-weight", type=float, default=10.0 ** -0.5)
    parser.add_argument("--explosion-limit", type=float, default=1e12)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    unknown = set(modes) - set(MODES)
    if unknown:
        raise ValueError(f"Unknown modes: {sorted(unknown)}")
    result = run_audit(
        seeds=_parse_ints(args.seeds),
        modes=modes,
        damping_values=_parse_floats(args.damping),
        n_iterations=args.iterations,
        outdir=args.outdir,
        distance_weight=args.distance_weight,
        direction_weight=args.direction_weight,
        explosion_limit=args.explosion_limit,
    )
    print(f"[audit] saved to {result['outdir']}; failures={len(result['failures'])}")


if __name__ == "__main__":
    main()
