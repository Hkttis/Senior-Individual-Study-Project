"""Run a paired Full/NoRep experiment on a sparse non-unique fixture."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

import library.physics as physics
from library.config import (
    DIRECTIONAL_FORCE_MAGNITUDE_BASE,
    REPULSION_STRENGTH_BASE,
    SPRING_STIFFNESS_BASE,
)
from library.metrics import (
    calculate_kruskals_stress,
    direction_violation_rate,
    mean_angular_error_violations,
)
from library.units import pos_matrix_sim2km
from scripts.evaluate_repulsion_layout import _topology_metrics


FIXTURE_DIR = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "fixtures"
    / "advanced_nonunique_repulsion_dataset"
)
VARIANTS = ("PhysicsSim-NoRep", "PhysicsSim-Full")
METRICS = (
    "RMSE_test_units",
    "E_distance_stress",
    "E_direction_vr",
    "E_direction_mae",
    "crowding_violation_rate_tau_0p1",
    "collapse_node_rate_tau_0p1",
    "nnd_q05_units",
    "distance_edge_crossing_rate",
    "distance_P20_P34_units",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_dataset() -> dict:
    position_rows = _read_csv(FIXTURE_DIR / "expected_positions.csv")
    distance_rows = _read_csv(FIXTURE_DIR / "distance_edges.csv")
    direction_rows = _read_csv(FIXTURE_DIR / "direction_edges.csv")
    vertices = [row["model_name"] for row in position_rows]
    dni = {label: index for index, label in enumerate(vertices)}
    expected = np.asarray(
        [[float(row["x"]), float(row["y"])] for row in position_rows], dtype=float
    )
    return {
        "vertices": vertices,
        "dni": dni,
        "expected": expected,
        "roles": {row["model_name"]: row["use_role"] for row in position_rows},
        "distance_data": [
            [row["source"], row["target"], str(int(float(row["distance"])))]
            for row in distance_rows
        ],
        "direction_data": [
            [row["source"], row["target"], row["direction"]]
            for row in direction_rows
        ],
    }


def _initial_layout(dataset: dict, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lower = dataset["expected"].min(axis=0)
    upper = dataset["expected"].max(axis=0)
    initial = rng.uniform(lower, upper, size=dataset["expected"].shape)
    for label, role in dataset["roles"].items():
        if role in {"anchor", "anchor_align"}:
            initial[dataset["dni"][label]] = dataset["expected"][dataset["dni"][label]]
    return initial


def _run_variant(
    dataset: dict,
    initial: np.ndarray,
    *,
    use_repulsion: bool,
    repulsion_strength: float,
) -> np.ndarray:
    anchor_labels = [
        label
        for label, role in dataset["roles"].items()
        if role in {"anchor", "anchor_align"}
    ]
    _wrong, _stress_history, _position_history, final = physics.main_physics_simulation(
        vertice=dataset["vertices"],
        dni=dataset["dni"],
        data=dataset["distance_data"],
        pos_matrix=initial.copy().tolist(),
        directional_data=dataset["direction_data"],
        fixed_positions_list=[[label] for label in anchor_labels],
        spring_stiffness=SPRING_STIFFNESS_BASE,
        repulsion_strength=repulsion_strength if use_repulsion else 0.0,
        directional_force_magnitude=DIRECTIONAL_FORCE_MAGNITUDE_BASE * 10.0,
        plot=False,
    )
    return np.asarray(final, dtype=float)


def _nearest_neighbor_distances(points: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    return distances.min(axis=1)


def evaluate_layout(variant: str, seed: int, points: np.ndarray, dataset: dict) -> dict:
    dni = dataset["dni"]
    test_indices = [
        dni[label] for label, role in dataset["roles"].items() if role == "test"
    ]
    errors = np.linalg.norm(points[test_indices] - dataset["expected"][test_indices], axis=1)
    pairwise = np.asarray(
        [np.linalg.norm(points[i] - points[j]) for i, j in combinations(range(len(points)), 2)],
        dtype=float,
    )
    nnd = _nearest_neighbor_distances(points)
    target_median = float(np.median([float(row[2]) for row in dataset["distance_data"]]))
    tau = 0.10 * target_median
    topology = _topology_metrics(
        points,
        dataset["vertices"],
        [(row[0], row[1]) for row in dataset["distance_data"]],
        tau,
    )
    return {
        "variant": variant,
        "seed": int(seed),
        "RMSE_test_units": float(np.sqrt(np.mean(np.square(errors)))),
        "E_distance_stress": float(
            calculate_kruskals_stress(
                dni, pos_matrix_sim2km(points.tolist()), dataset["distance_data"]
            )
        ),
        "E_direction_vr": float(
            direction_violation_rate(points, dataset["direction_data"], dni)
        ),
        "E_direction_mae": float(
            mean_angular_error_violations(points, dataset["direction_data"], dni)
        ),
        "crowding_violation_rate_tau_0p1": float(np.mean(pairwise < tau)),
        "collapse_node_rate_tau_0p1": float(np.mean(nnd < tau)),
        "nnd_q05_units": float(np.quantile(nnd, 0.05)),
        "distance_edge_crossing_rate": topology["distance_edge_crossing_rate"],
        "distance_P20_P34_units": float(
            np.linalg.norm(points[dni["P20"]] - points[dni["P34"]])
        ),
    }


def _summarize(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in runs.groupby("variant", sort=False):
        for metric in METRICS:
            values = group[metric].to_numpy(float)
            rows.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "n": len(values),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "median": float(np.median(values)),
                }
            )
    return pd.DataFrame(rows)


def _paired(runs: pd.DataFrame) -> pd.DataFrame:
    full = runs[runs["variant"] == "PhysicsSim-Full"].set_index("seed")
    no_rep = runs[runs["variant"] == "PhysicsSim-NoRep"].set_index("seed")
    seeds = sorted(set(full.index).intersection(no_rep.index))
    rows = []
    for metric in METRICS:
        difference = full.loc[seeds, metric].to_numpy(float) - no_rep.loc[seeds, metric].to_numpy(float)
        rows.append(
            {
                "metric": metric,
                "diff_definition": "Full_minus_NoRep",
                "n_pairs": len(difference),
                "paired_diff_mean": float(difference.mean()),
                "paired_diff_std": float(difference.std(ddof=1)) if len(difference) > 1 else 0.0,
                "paired_diff_median": float(np.median(difference)),
            }
        )
    return pd.DataFrame(rows)


def run_experiment(
    seeds: list[int],
    *,
    iterations: int = 1000,
    repulsion_strength: float = REPULSION_STRENGTH_BASE * (10.0 ** -0.5),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset = load_dataset()
    rows = []
    position_rows = []
    initial_metric_rows = []
    initial_position_rows = []
    original_iterations = physics.stop_physim_iteration_time
    physics.stop_physim_iteration_time = int(iterations)
    try:
        for seed in seeds:
            initial = _initial_layout(dataset, seed)
            initial_metric_rows.append(evaluate_layout("Initial", seed, initial, dataset))
            for label, point in zip(dataset["vertices"], initial):
                initial_position_rows.append(
                    {
                        "seed": int(seed),
                        "model_name": label,
                        "x": float(point[0]),
                        "y": float(point[1]),
                    }
                )
            for variant, use_repulsion in (
                ("PhysicsSim-NoRep", False),
                ("PhysicsSim-Full", True),
            ):
                points = _run_variant(
                    dataset,
                    initial,
                    use_repulsion=use_repulsion,
                    repulsion_strength=repulsion_strength,
                )
                if not np.all(np.isfinite(points)):
                    raise ValueError(f"Non-finite layout for {variant}, seed={seed}")
                rows.append(evaluate_layout(variant, seed, points, dataset))
                for label, point in zip(dataset["vertices"], points):
                    position_rows.append(
                        {
                            "variant": variant,
                            "seed": int(seed),
                            "model_name": label,
                            "x": float(point[0]),
                            "y": float(point[1]),
                        }
                    )
    finally:
        physics.stop_physim_iteration_time = original_iterations

    runs = pd.DataFrame(rows)
    return (
        runs,
        _summarize(runs),
        _paired(runs),
        pd.DataFrame(position_rows),
        pd.DataFrame(initial_metric_rows),
        pd.DataFrame(initial_position_rows),
    )


def _parse_seeds(text: str) -> list[int]:
    seeds = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--repulsion-strength", type=float, default=REPULSION_STRENGTH_BASE * (10.0 ** -0.5))
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    if outdir.exists() and any(outdir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)

    seeds = _parse_seeds(args.seeds)
    runs, summary, paired, positions, initial_metrics, initial_positions = run_experiment(
        seeds,
        iterations=args.iterations,
        repulsion_strength=args.repulsion_strength,
    )
    runs.to_csv(outdir / "advanced_repulsion_runs.csv", index=False)
    summary.to_csv(outdir / "advanced_repulsion_summary.csv", index=False)
    paired.to_csv(outdir / "advanced_repulsion_paired.csv", index=False)
    positions.to_csv(outdir / "advanced_repulsion_positions.csv", index=False)
    initial_metrics.to_csv(outdir / "advanced_repulsion_initial_metrics.csv", index=False)
    initial_positions.to_csv(outdir / "advanced_repulsion_initial_positions.csv", index=False)
    dataset = load_dataset()
    (outdir / "advanced_repulsion_config.json").write_text(
        json.dumps(
            {
                "seeds": seeds,
                "iterations": args.iterations,
                "repulsion_strength": args.repulsion_strength,
                "distance_density": 40 / 595,
                "initialization": "uniform_over_expected_position_bounding_box_with_anchors_reset",
                "initialization_bounds": {
                    "x_min": float(dataset["expected"][:, 0].min()),
                    "x_max": float(dataset["expected"][:, 0].max()),
                    "y_min": float(dataset["expected"][:, 1].min()),
                    "y_max": float(dataset["expected"][:, 1].max()),
                },
                "fixture": str(FIXTURE_DIR),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print("\nPaired Full - NoRep:\n" + paired.to_string(index=False))
    print(f"\nSaved: {outdir}")


if __name__ == "__main__":
    main()
