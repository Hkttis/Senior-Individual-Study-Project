"""Audit observed distance triangles for triangle-inequality violations."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "distance_edges_verified.csv"
DEFAULT_OUTDIR = PROJECT_ROOT / "outputs" / "distance_triangle_inequality_audit"
REQUIRED_COLUMNS = ("地點一", "地點二", "里程")


def _undirected_pair(first: str, second: str) -> tuple[str, str]:
    return tuple(sorted((first, second)))


def load_distances(path: Path) -> tuple[dict[tuple[str, str], float], int]:
    """Load one positive finite distance for every undirected observed pair."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or any(name not in reader.fieldnames for name in REQUIRED_COLUMNS):
            raise ValueError(f"Expected columns {REQUIRED_COLUMNS}; found {reader.fieldnames}.")
        rows = list(reader)

    distances: dict[tuple[str, str], float] = {}
    for row_number, row in enumerate(rows, start=2):
        first = str(row["地點一"]).strip()
        second = str(row["地點二"]).strip()
        if not first or not second or first == second:
            raise ValueError(f"Invalid endpoint(s) at CSV row {row_number}: {first!r}, {second!r}.")
        try:
            distance = float(row["里程"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid distance at CSV row {row_number}: {row['里程']!r}.") from exc
        if not math.isfinite(distance) or distance <= 0.0:
            raise ValueError(f"Distance must be finite and positive at CSV row {row_number}.")

        pair = _undirected_pair(first, second)
        if pair in distances and not math.isclose(distances[pair], distance, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"Conflicting duplicate distances for {pair}: {distances[pair]} and {distance}."
            )
        distances[pair] = distance
    return distances, len(rows)


def audit_triangles(
    distances: dict[tuple[str, str], float], *, tolerance: float = 1e-12
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return all fully observed triangles and the strict violations among them."""
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and nonnegative.")

    nodes = sorted({node for pair in distances for node in pair})
    triangle_rows: list[dict[str, object]] = []
    violation_rows: list[dict[str, object]] = []

    for node_a, node_b, node_c in itertools.combinations(nodes, 3):
        pairs = (
            (node_a, node_b, node_c),
            (node_a, node_c, node_b),
            (node_b, node_c, node_a),
        )
        if any(_undirected_pair(first, second) not in distances for first, second, _ in pairs):
            continue

        edges = [
            (distances[_undirected_pair(first, second)], first, second, via)
            for first, second, via in pairs
        ]
        longest, endpoint_1, endpoint_2, via = max(edges, key=lambda item: item[0])
        # Pair identities, rather than distance values, identify the two-step alternative path.
        leg_1 = distances[_undirected_pair(endpoint_1, via)]
        leg_2 = distances[_undirected_pair(via, endpoint_2)]
        alternative = leg_1 + leg_2
        violation = longest - alternative
        is_violation = violation > tolerance
        row = {
            "node_a": node_a,
            "node_b": node_b,
            "node_c": node_c,
            "long_edge_endpoint_1": endpoint_1,
            "long_edge_endpoint_2": endpoint_2,
            "long_edge_distance_li": longest,
            "via_node": via,
            "leg_1_distance_li": leg_1,
            "leg_2_distance_li": leg_2,
            "alternative_two_edge_sum_li": alternative,
            "violation_amount_li": max(0.0, violation),
            "violation_fraction_of_long_edge": max(0.0, violation) / longest,
            "violation_fraction_of_two_edge_sum": max(0.0, violation) / alternative,
            "satisfies_triangle_inequality": not is_violation,
        }
        triangle_rows.append(row)
        if is_violation:
            violation_rows.append(row)

    return pd.DataFrame(triangle_rows), pd.DataFrame(violation_rows)


def _distribution(values: np.ndarray) -> dict[str, float | int | None]:
    if values.size == 0:
        return {"count": 0, "min": None, "q25": None, "median": None, "mean": None, "q75": None, "max": None}
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "q75": float(np.quantile(values, 0.75)),
        "max": float(np.max(values)),
    }


def run_audit(input_path: Path, outdir: Path, *, tolerance: float = 1e-12) -> dict[str, object]:
    distances, source_rows = load_distances(input_path)
    triangles, violations = audit_triangles(distances, tolerance=tolerance)
    outdir.mkdir(parents=True, exist_ok=True)
    triangles.to_csv(outdir / "fully_observed_triangles.csv", index=False, encoding="utf-8-sig")
    violations.to_csv(outdir / "triangle_inequality_violations.csv", index=False, encoding="utf-8-sig")

    amounts = (
        violations["violation_amount_li"].to_numpy(float)
        if not violations.empty
        else np.asarray([], dtype=float)
    )
    fractions = (
        violations["violation_fraction_of_long_edge"].to_numpy(float)
        if not violations.empty
        else np.asarray([], dtype=float)
    )
    path_fractions = (
        violations["violation_fraction_of_two_edge_sum"].to_numpy(float)
        if not violations.empty
        else np.asarray([], dtype=float)
    )
    summary = {
        "input_file": str(input_path.resolve()),
        "distance_unit": "li",
        "source_row_count": source_rows,
        "unique_undirected_edge_count": len(distances),
        "unique_node_count": len({node for pair in distances for node in pair}),
        "all_possible_node_triples": math.comb(len({node for pair in distances for node in pair}), 3),
        "fully_observed_triangle_count": int(len(triangles)),
        "violating_triangle_count": int(len(violations)),
        "violating_fraction_of_fully_observed_triangles": (
            float(len(violations) / len(triangles)) if len(triangles) else None
        ),
        "violation_amount_li_distribution": _distribution(amounts),
        "violation_fraction_of_long_edge_distribution": _distribution(fractions),
        "violation_fraction_of_two_edge_sum_distribution": _distribution(path_fractions),
        "comparison_rule": "longest observed edge > sum of the other two observed edges + tolerance",
        "missing_edge_policy": "triples missing any of their three direct observed edges are not tested",
        "numeric_tolerance_li": tolerance,
    }
    (outdir / "triangle_inequality_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    args = parser.parse_args()
    summary = run_audit(args.input, args.outdir, tolerance=args.tolerance)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
