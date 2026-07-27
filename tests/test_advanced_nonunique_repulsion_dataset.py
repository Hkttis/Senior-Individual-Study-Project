import csv
import math
from pathlib import Path

import numpy as np
import pytest

from library.directions import DIR4_SIM, DIR8_UNIT_SIM
from library.metrics import (
    calculate_kruskals_stress,
    direction_violation_rate,
    mean_angular_error_violations,
)
from library.units import pos_matrix_sim2km
from scripts.evaluate_repulsion_layout import _topology_metrics
from scripts.run_advanced_repulsion_synthetic import METRICS, _initial_layout, load_dataset, run_experiment
from scripts.run_advanced_sparse_baselines import run_baselines


FIXTURE_DIR = (
    Path(__file__).parent / "fixtures" / "advanced_nonunique_repulsion_dataset"
)


def _read_csv(name):
    with (FIXTURE_DIR / name).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _positions(name, labels):
    rows = _read_csv(name)
    by_label = {
        row["model_name"]: [float(row["x"]), float(row["y"])] for row in rows
    }
    return np.asarray([by_label[label] for label in labels], dtype=float)


def _dataset():
    position_rows = _read_csv("expected_positions.csv")
    labels = [row["model_name"] for row in position_rows]
    return {
        "labels": labels,
        "dni": {label: index for index, label in enumerate(labels)},
        "expected": _positions("expected_positions.csv", labels),
        "alternative": _positions("alternative_positions.csv", labels),
        "roles": {row["model_name"]: row["use_role"] for row in position_rows},
        "distance_rows": _read_csv("distance_edges.csv"),
        "direction_rows": _read_csv("direction_edges.csv"),
        "construction_rows": _read_csv("construction_order.csv"),
    }


def _distance_data(dataset):
    return [
        [row["source"], row["target"], row["distance"]]
        for row in dataset["distance_rows"]
    ]


def _direction_data(dataset):
    return [
        [row["source"], row["target"], row["direction"]]
        for row in dataset["direction_rows"]
    ]


def _rigidity_matrix(dataset):
    labels = dataset["labels"]
    dni = dataset["dni"]
    expected = dataset["expected"]
    matrix = np.zeros((len(dataset["distance_rows"]), 2 * len(labels)), dtype=float)
    for row_index, row in enumerate(dataset["distance_rows"]):
        source_index = dni[row["source"]]
        target_index = dni[row["target"]]
        difference = expected[source_index] - expected[target_index]
        matrix[row_index, 2 * source_index : 2 * source_index + 2] = difference
        matrix[row_index, 2 * target_index : 2 * target_index + 2] = -difference
    return matrix


def test_advanced_dataset_contract_density_bearings_and_crossings():
    dataset = _dataset()
    labels = dataset["labels"]
    dni = dataset["dni"]
    expected = dataset["expected"]
    n = len(labels)

    assert labels == [f"P{index:02d}" for index in range(35)]
    assert len(dataset["distance_rows"]) == 40
    assert len(dataset["direction_rows"]) == 26
    assert 40 / (n * (n - 1) / 2) < 0.07
    assert len(dataset["direction_rows"]) <= len(dataset["distance_rows"])
    assert dataset["roles"]["P00"] == "anchor_align"
    assert dataset["roles"]["P07"] == dataset["roles"]["P08"] == "anchor"
    assert all(
        "P34" not in (row["source"], row["target"])
        for row in dataset["direction_rows"]
    )

    for row in dataset["distance_rows"]:
        actual = np.linalg.norm(
            expected[dni[row["source"]]] - expected[dni[row["target"]]]
        )
        assert actual == pytest.approx(float(row["distance"]), abs=1e-12)
        assert float(row["distance"]) <= 500.0

    non_octilinear = 0
    for row in dataset["direction_rows"]:
        direction = row["direction"]
        assert direction in DIR8_UNIT_SIM
        vector = expected[dni[row["target"]]] - expected[dni[row["source"]]]
        bearing = math.degrees(math.atan2(float(vector[1]), float(vector[0])))
        assert float(row["expected_bearing_deg"]) == pytest.approx(bearing, abs=1e-6)
        if abs(bearing - round(bearing / 45.0) * 45.0) > 1.0:
            non_octilinear += 1

        unit = vector / np.linalg.norm(vector)
        desired = np.asarray(DIR8_UNIT_SIM[direction], dtype=float)
        angle = math.acos(float(np.clip(np.dot(unit, desired), -1.0, 1.0)))
        threshold = math.pi / 2 if direction in DIR4_SIM else math.pi / 4
        assert angle <= threshold + 1e-12
    assert non_octilinear == 6

    topology = _topology_metrics(
        expected,
        labels,
        [(row["source"], row["target"]) for row in dataset["distance_rows"]],
        tau_km=30.0,
    )
    assert topology["distance_edge_crossing_rate"] == 0.0


def test_advanced_dataset_is_connected_but_not_locally_rigid():
    dataset = _dataset()
    labels = dataset["labels"]
    edge_set = {
        frozenset((row["source"], row["target"]))
        for row in dataset["distance_rows"]
    }

    tree_rows = dataset["construction_rows"]
    assert len(tree_rows) == len(labels)
    assert {row["node"] for row in tree_rows} == set(labels)
    for row in tree_rows[1:]:
        assert frozenset((row["node"], row["tree_parent"])) in edge_set

    adjacency = {label: set() for label in labels}
    for source, target in (tuple(edge) for edge in edge_set):
        adjacency[source].add(target)
        adjacency[target].add(source)
    visited = {labels[0]}
    pending = [labels[0]]
    while pending:
        current = pending.pop()
        for neighbor in adjacency[current] - visited:
            visited.add(neighbor)
            pending.append(neighbor)
    assert visited == set(labels)

    degrees = {label: len(adjacency[label]) for label in labels}
    leaves = {label for label, degree in degrees.items() if degree == 1}
    assert len(leaves) == 14
    assert not {"P00", "P07", "P08"}.intersection(leaves)
    direction_degrees = {label: 0 for label in labels}
    for row in dataset["direction_rows"]:
        direction_degrees[row["source"]] += 1
        direction_degrees[row["target"]] += 1
    assert all(direction_degrees[label] == 0 for label in leaves)

    rank = np.linalg.matrix_rank(_rigidity_matrix(dataset))
    local_rigidity_rank = 2 * len(labels) - 3
    assert rank <= len(dataset["distance_rows"])
    assert rank < local_rigidity_rank
    assert 2 * len(labels) - rank - 3 > 0


def test_advanced_dataset_has_continuous_noncongruent_solutions_with_fixed_anchors():
    dataset = _dataset()
    expected = dataset["expected"]
    alternative = dataset["alternative"]
    dni = dataset["dni"]
    distance_data = _distance_data(dataset)
    direction_data = _direction_data(dataset)

    for label in ("P00", "P01", "P07"):
        assert alternative[dni[label]] == pytest.approx(expected[dni[label]], abs=0.0)
    assert alternative[dni["P34"]] == pytest.approx(alternative[dni["P20"]], abs=0.0)

    third_solution = expected.copy()
    third_solution[dni["P34"]] = [2200.0, 1200.0]
    for points in (expected, alternative, third_solution):
        stress = calculate_kruskals_stress(
            dni, pos_matrix_sim2km(points.tolist()), distance_data
        )
        assert stress == pytest.approx(0.0, abs=1e-12)
        assert direction_violation_rate(points, direction_data, dni) == 0.0
        assert mean_angular_error_violations(points, direction_data, dni) == 0.0

    expected_pairwise = np.linalg.norm(
        expected[:, None, :] - expected[None, :, :], axis=2
    )
    alternative_pairwise = np.linalg.norm(
        alternative[:, None, :] - alternative[None, :, :], axis=2
    )
    assert not np.allclose(expected_pairwise, alternative_pairwise)
    assert expected_pairwise[dni["P20"], dni["P34"]] == pytest.approx(800.0)
    assert alternative_pairwise[dni["P20"], dni["P34"]] == pytest.approx(0.0)


def test_advanced_repulsion_experiment_smoke():
    runs, summary, paired, positions, initial_metrics, initial_positions = run_experiment(
        [0], iterations=100
    )

    assert set(runs["variant"]) == {"PhysicsSim-NoRep", "PhysicsSim-Full"}
    assert set(runs["seed"]) == {0}
    assert np.all(np.isfinite(runs[list(METRICS)].to_numpy(float)))
    assert len(summary) == 2 * len(METRICS)
    assert len(paired) == len(METRICS)
    assert len(positions) == 2 * 35
    assert len(initial_metrics) == 1
    assert len(initial_positions) == 35

    anchor_rows = positions[
        positions["model_name"].isin(["P00", "P07", "P08"])
    ]
    expected_anchors = np.asarray([[0.0, 0.0], [0.0, 400.0], [300.0, 400.0]])
    assert np.allclose(
        anchor_rows[["x", "y"]].to_numpy(float),
        np.tile(expected_anchors, (2, 1)),
        atol=1e-6,
    )


def test_advanced_initialization_is_random_reproducible_and_keeps_anchors_fixed():
    dataset = load_dataset()
    initial_a = _initial_layout(dataset, seed=0)
    initial_a_repeat = _initial_layout(dataset, seed=0)
    initial_b = _initial_layout(dataset, seed=1)
    lower = dataset["expected"].min(axis=0)
    upper = dataset["expected"].max(axis=0)
    anchor_labels = [
        label
        for label, role in dataset["roles"].items()
        if role in {"anchor", "anchor_align"}
    ]
    non_anchor_indices = [
        dataset["dni"][label]
        for label in dataset["vertices"]
        if label not in anchor_labels
    ]

    assert np.array_equal(initial_a, initial_a_repeat)
    assert not np.array_equal(initial_a[non_anchor_indices], initial_b[non_anchor_indices])
    assert np.all(initial_a[non_anchor_indices] >= lower)
    assert np.all(initial_a[non_anchor_indices] <= upper)
    assert not np.allclose(
        initial_a[non_anchor_indices], dataset["expected"][non_anchor_indices]
    )
    for label in anchor_labels:
        index = dataset["dni"][label]
        assert initial_a[index] == pytest.approx(dataset["expected"][index], abs=0.0)
        assert initial_b[index] == pytest.approx(dataset["expected"][index], abs=0.0)


def test_advanced_baseline_experiment_smoke():
    runs, summary, positions = run_baselines(
        [0], smacof_iterations=20, dc_iterations=20
    )

    assert set(runs["variant"]) == {"SMACOF", "DC-SMACOF"}
    assert set(runs["seed"]) == {0}
    assert np.all(np.isfinite(runs[list(METRICS)].to_numpy(float)))
    assert len(summary) == 2 * len(METRICS)
    assert len(positions) == 2 * 35
