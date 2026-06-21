"""Evaluate PhysicsSim repulsion as an anti-crowding regularizer.

Reads an ablation output directory and compares PhysicsSim-Full with
PhysicsSim-NoRep using paired random seeds.  The confirmatory metrics are
crowding violation rate, collapse-node rate, and nearest-neighbor distances.

Example
-------
python -m scripts.evaluate_repulsion_layout \
  --ablation-outdir outputs/repulsion_seed5_9 \
  --outdir outputs/repulsion_seed5_9_layout_eval
"""

from __future__ import annotations

import argparse
import os
from itertools import combinations
from pathlib import Path
from typing import Iterable

_MPLCONFIGDIR = Path(__file__).resolve().parents[1] / "outputs" / ".matplotlib_repulsion"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from library.config import FILE_PATHS, Li2km
from library.data_io import load_ini_data_from_csv
from library.units import pos_matrix_sim2km


FULL = "PhysicsSim-Full"
NO_REP = "PhysicsSim-NoRep"
TAU_MULTIPLIERS = (0.05, 0.10, 0.15)


def _pairwise_distances_km(points_km: np.ndarray) -> np.ndarray:
    return np.asarray(
        [np.linalg.norm(points_km[i] - points_km[j]) for i, j in combinations(range(len(points_km)), 2)],
        dtype=float,
    )


def _nearest_neighbor_distances_km(points_km: np.ndarray) -> np.ndarray:
    delta = points_km[:, None, :] - points_km[None, :, :]
    distances = np.linalg.norm(delta, axis=2)
    np.fill_diagonal(distances, np.inf)
    return distances.min(axis=1)


def _gini(values: np.ndarray) -> float:
    values = np.sort(np.asarray(values, dtype=float))
    if len(values) == 0 or np.isclose(values.sum(), 0.0):
        return 0.0
    ranks = np.arange(1, len(values) + 1, dtype=float)
    return float((2.0 * np.dot(ranks, values) / (len(values) * values.sum())) - (len(values) + 1.0) / len(values))


def _convex_hull_area(points: np.ndarray) -> float:
    """Return 2D convex-hull area with a dependency-free monotonic chain."""
    unique = sorted({(float(x), float(y)) for x, y in points})
    if len(unique) < 3:
        return 0.0

    def cross(origin, a, b) -> float:
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    xs = np.asarray([point[0] for point in hull])
    ys = np.asarray([point[1] for point in hull])
    return float(0.5 * abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))))


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = b - a
    ac = c - a
    return float(ab[0] * ac[1] - ab[1] * ac[0])


def _segments_cross(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    ab_c, ab_d = _orientation(a, b, c), _orientation(a, b, d)
    cd_a, cd_b = _orientation(c, d, a), _orientation(c, d, b)
    return (ab_c * ab_d < 0.0) and (cd_a * cd_b < 0.0)


def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    denom = float(np.dot(segment, segment))
    if np.isclose(denom, 0.0):
        return float(np.linalg.norm(point - start))
    t = float(np.clip(np.dot(point - start, segment) / denom, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + t * segment)))


def _topology_metrics(points_km: np.ndarray, labels: list[str], distance_edges: list[tuple[str, str]], tau_km: float) -> dict:
    index = {label: i for i, label in enumerate(labels)}
    edges = [(source, target) for source, target in distance_edges if source in index and target in index]
    eligible_pairs = 0
    crossing_count = 0
    for (source_a, target_a), (source_b, target_b) in combinations(edges, 2):
        if {source_a, target_a}.intersection({source_b, target_b}):
            continue
        eligible_pairs += 1
        if _segments_cross(
            points_km[index[source_a]], points_km[index[target_a]], points_km[index[source_b]], points_km[index[target_b]]
        ):
            crossing_count += 1

    node_edge_distances = []
    for source, target in edges:
        start, end = points_km[index[source]], points_km[index[target]]
        for label, node_idx in index.items():
            if label not in {source, target}:
                node_edge_distances.append(_point_segment_distance(points_km[node_idx], start, end))
    node_edge_arr = np.asarray(node_edge_distances, dtype=float)
    return {
        "distance_edge_crossing_rate": float(crossing_count / eligible_pairs) if eligible_pairs else 0.0,
        "node_edge_distance_q05_km": float(np.quantile(node_edge_arr, 0.05)) if len(node_edge_arr) else float("nan"),
        "node_edge_distance_q10_km": float(np.quantile(node_edge_arr, 0.10)) if len(node_edge_arr) else float("nan"),
        "edge_node_overlap_rate": float(np.mean(node_edge_arr < tau_km)) if len(node_edge_arr) else 0.0,
    }


def _bootstrap_mean_ci(values: np.ndarray, *, seed: int = 0, n_boot: int = 5000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[sample_idx].mean(axis=1)
    return tuple(map(float, np.quantile(means, [0.025, 0.975])))


def _summary_by_variant(metrics: pd.DataFrame, metric_cols: Iterable[str]) -> pd.DataFrame:
    rows = []
    for variant, group in metrics.groupby("variant", sort=True):
        for metric in metric_cols:
            values = group[metric].to_numpy(float)
            lo, hi = _bootstrap_mean_ci(values)
            rows.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "n": len(values),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                    "median": float(np.median(values)),
                    "q25": float(np.quantile(values, 0.25)),
                    "q75": float(np.quantile(values, 0.75)),
                    "mean_ci95_lo": lo,
                    "mean_ci95_hi": hi,
                }
            )
    return pd.DataFrame(rows)


def _paired_comparisons(metrics: pd.DataFrame, metric_cols: Iterable[str]) -> pd.DataFrame:
    full = metrics[metrics["variant"] == FULL].set_index("seed")
    no_rep = metrics[metrics["variant"] == NO_REP].set_index("seed")
    common_seeds = sorted(set(full.index).intersection(no_rep.index))
    if not common_seeds:
        raise ValueError("No paired Full/NoRep seeds found.")
    rows = []
    for metric in metric_cols:
        diff = full.loc[common_seeds, metric].to_numpy(float) - no_rep.loc[common_seeds, metric].to_numpy(float)
        lo, hi = _bootstrap_mean_ci(diff)
        rows.append(
            {
                "comparison": "repulsion_given_direction",
                "left_variant": FULL,
                "right_variant": NO_REP,
                "metric": metric,
                "diff_definition": "Full_minus_NoRep",
                "n_pairs": len(diff),
                "paired_diff_mean": float(diff.mean()),
                "paired_diff_median": float(np.median(diff)),
                "paired_diff_q25": float(np.quantile(diff, 0.25)),
                "paired_diff_q75": float(np.quantile(diff, 0.75)),
                "paired_diff_ci95_lo": lo,
                "paired_diff_ci95_hi": hi,
                "ci_excludes_zero": bool((lo > 0.0) or (hi < 0.0)),
            }
        )
    return pd.DataFrame(rows)


def _plot_box(metrics: pd.DataFrame, metric: str, ylabel: str, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    groups = [metrics.loc[metrics["variant"] == variant, metric].to_numpy(float) for variant in (FULL, NO_REP)]
    ax.boxplot(groups, tick_labels=["Full", "NoRep"], showmeans=True)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def _plot_scatter(metrics: pd.DataFrame, x_metric: str, y_metric: str, xlabel: str, ylabel: str, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for variant, color in ((FULL, "#1f77b4"), (NO_REP, "#ff7f0e")):
        group = metrics[metrics["variant"] == variant]
        ax.scatter(group[x_metric], group[y_metric], label=variant.replace("PhysicsSim-", ""), color=color, s=48)
        for _, row in group.iterrows():
            ax.annotate(str(int(row["seed"])), (row[x_metric], row[y_metric]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def evaluate_repulsion_layout(*, ablation_outdir: str | Path, outdir: str | Path) -> dict[str, Path]:
    source = Path(ablation_outdir)
    destination = Path(outdir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    runs = pd.read_csv(source / "ablation_runs_by_seed.csv")
    positions = pd.read_csv(source / "ablation_final_positions_y_up_sim.csv")
    runs = runs[(runs["status"] == "ok") & runs["variant"].isin([FULL, NO_REP])].copy()
    if runs.empty:
        raise ValueError("Ablation output contains no successful PhysicsSim-Full / PhysicsSim-NoRep runs.")

    _graph, vertices, _dni, _edges, distance_data = load_ini_data_from_csv(FILE_PATHS)
    distance_edges = [(str(row[0]), str(row[1])) for row in distance_data]
    target_median_km = float(np.median([float(row[2]) * Li2km for row in distance_data]))

    rows = []
    for _, run in runs.iterrows():
        variant, seed = str(run["variant"]), int(run["seed"])
        group = positions[(positions["variant"] == variant) & (positions["seed"] == seed)].set_index("label")
        missing = set(vertices).difference(group.index)
        if missing:
            raise ValueError(f"Missing positions for variant={variant}, seed={seed}: {sorted(missing)}")
        points_sim = group.loc[vertices, ["x_y_up_sim", "y_y_up_sim"]].to_numpy(float)
        points_km = np.asarray(pos_matrix_sim2km(points_sim.tolist()), dtype=float)
        pairwise = _pairwise_distances_km(points_km)
        nnd = _nearest_neighbor_distances_km(points_km)
        centroid = points_km.mean(axis=0)
        result = {
            "variant": variant,
            "seed": seed,
            "target_distance_median_km": target_median_km,
            "n_nodes": len(points_km),
            "nnd_min_km": float(nnd.min()),
            "nnd_q05_km": float(np.quantile(nnd, 0.05)),
            "nnd_q10_km": float(np.quantile(nnd, 0.10)),
            "nnd_median_km": float(np.median(nnd)),
            "nnd_mean_km": float(nnd.mean()),
            "nnd_cv": float(nnd.std(ddof=0) / nnd.mean()),
            "nnd_gini": _gini(nnd),
            "radius_gyration_km": float(np.sqrt(np.mean(np.sum((points_km - centroid) ** 2, axis=1)))),
            "convex_hull_area_km2": _convex_hull_area(points_km),
            **_topology_metrics(points_km, vertices, distance_edges, target_median_km * TAU_MULTIPLIERS[0]),
        }
        for multiplier in TAU_MULTIPLIERS:
            tau = target_median_km * multiplier
            suffix = str(multiplier).replace(".", "p")
            result[f"crowding_violation_rate_tau_{suffix}"] = float(np.mean(pairwise < tau))
            result[f"collapse_node_rate_tau_{suffix}"] = float(np.mean(nnd < tau))
            result[f"tau_{suffix}_km"] = tau
        for column in ("RMSE_test_km", "E_distance_stress", "E_direction_vr", "E_direction_mae"):
            result[column] = float(run[column])
        rows.append(result)

    metrics = pd.DataFrame(rows).sort_values(["variant", "seed"])
    metric_cols = [column for column in metrics.columns if column not in {"variant", "seed", "n_nodes"} and not column.startswith("tau_")]
    summary = _summary_by_variant(metrics, metric_cols)
    paired = _paired_comparisons(metrics, metric_cols)

    metrics_path = destination / "repulsion_metrics_by_seed.csv"
    summary_path = destination / "repulsion_metrics_summary.csv"
    paired_path = destination / "repulsion_paired_comparisons.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    paired.to_csv(paired_path, index=False, encoding="utf-8-sig")

    _plot_box(metrics, "crowding_violation_rate_tau_0p1", "Crowding violation rate (tau=0.10 median target distance)", destination / "boxplot_cvr_tau_0p1.png")
    _plot_box(metrics, "nnd_q05_km", "NND q05 (km)", destination / "boxplot_nnd_q05.png")
    _plot_scatter(metrics, "RMSE_test_km", "crowding_violation_rate_tau_0p1", "Test RMSE (km)", "CVR (tau=0.10)", destination / "scatter_rmse_vs_cvr.png")
    _plot_scatter(metrics, "E_distance_stress", "crowding_violation_rate_tau_0p1", "Distance stress", "CVR (tau=0.10)", destination / "scatter_stress_vs_cvr.png")
    _plot_box(metrics, "distance_edge_crossing_rate", "Distance-edge crossing rate", destination / "boxplot_distance_edge_crossing_rate.png")
    return {"metrics": metrics_path, "summary": summary_path, "paired": paired_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate repulsion layout diagnostics from an AS output directory.")
    parser.add_argument("--ablation-outdir", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    paths = evaluate_repulsion_layout(ablation_outdir=args.ablation_outdir, outdir=args.outdir)
    for label, path in paths.items():
        print(f"[Saved] {label}: {path}")


if __name__ == "__main__":
    main()
