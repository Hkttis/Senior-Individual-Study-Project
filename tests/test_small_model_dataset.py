import csv
import math
from pathlib import Path

import numpy as np
import pytest

import MDS_model.directed_mds_model as dc_smacof
import MDS_model.stress_majorization_mds_model as smacof
import library.physics as physics
from library.directions import DIR4_SIM, DIR8_UNIT_SIM
from library.metrics import (
    calculate_kruskals_stress,
    direction_violation_rate,
    mean_angular_error_violations,
)
from library.units import pos_matrix_sim2km


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "small_model_dataset"


def _read_csv(name):
    with (FIXTURE_DIR / name).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_dataset():
    position_rows = _read_csv("expected_positions.csv")
    distance_rows = _read_csv("distance_edges.csv")
    direction_rows = _read_csv("direction_edges.csv")
    trilateration_rows = _read_csv("trilateration_order.csv")

    vertices = [row["model_name"] for row in position_rows]
    dni = {label: index for index, label in enumerate(vertices)}
    expected = np.asarray(
        [[float(row["x"]), float(row["y"])] for row in position_rows],
        dtype=float,
    )

    graph = [[] for _ in vertices]
    edges = []
    distance_data = []
    for row in distance_rows:
        source = row["source"]
        target = row["target"]
        distance = str(int(float(row["distance"])))
        edges.append((source, target))
        distance_data.append([source, target, distance])
        graph[dni[source]].append([source, target, "", distance])
        graph[dni[target]].append([target, source, "", distance])

    direction_data = [
        [row["source"], row["target"], row["direction"]]
        for row in direction_rows
    ]
    roles = {row["model_name"]: row["use_role"] for row in position_rows}
    return {
        "vertices": vertices,
        "dni": dni,
        "expected": expected,
        "roles": roles,
        "graph": graph,
        "edges": edges,
        "distance_data": distance_data,
        "direction_data": direction_data,
        "direction_rows": direction_rows,
        "trilateration_rows": trilateration_rows,
    }


def _assert_finite_layout(points, node_count):
    points = np.asarray(points, dtype=float)
    assert points.shape == (node_count, 2)
    assert np.all(np.isfinite(points))
    assert np.ptp(points[:, 0]) + np.ptp(points[:, 1]) > 0.0


def _rigid_align(points, target, fit_indices=None):
    """Remove translation, rotation, and reflection without changing scale."""
    points = np.asarray(points, dtype=float)
    target = np.asarray(target, dtype=float)
    fit_indices = np.arange(len(points)) if fit_indices is None else np.asarray(fit_indices, dtype=int)
    points_center = points[fit_indices].mean(axis=0)
    target_center = target[fit_indices].mean(axis=0)
    points_fit = points[fit_indices] - points_center
    target_fit = target[fit_indices] - target_center
    u, _singular_values, vt = np.linalg.svd(points_fit.T @ target_fit)
    return (points - points_center) @ (u @ vt) + target_center


def _calibration_indices(dataset):
    return [
        dataset["dni"][label]
        for label, role in dataset["roles"].items()
        if role in {"anchor", "anchor_align"}
    ]


def _translate_to_frame_anchor(points, dataset):
    points = np.asarray(points, dtype=float)
    index = dataset["dni"]["A"]
    return points - points[index] + dataset["expected"][index]


def _quality_metrics(points, dataset):
    points = np.asarray(points, dtype=float)
    errors = np.linalg.norm(points - dataset["expected"], axis=1)
    return {
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "stress": float(
            calculate_kruskals_stress(
                dataset["dni"],
                pos_matrix_sim2km(points),
                dataset["distance_data"],
            )
        ),
        "vr": float(
            direction_violation_rate(
                points,
                dataset["direction_data"],
                dataset["dni"],
            )
        ),
        "mae": float(
            mean_angular_error_violations(
                points,
                dataset["direction_data"],
                dataset["dni"],
            )
        ),
    }


def test_small_dataset_contract_and_known_geometry():
    dataset = _load_dataset()
    vertices = dataset["vertices"]
    dni = dataset["dni"]
    expected = dataset["expected"]

    assert vertices == list("ABCDEFGHIJ")
    assert len(dataset["distance_data"]) == 24
    assert len(dataset["direction_data"]) == 12
    assert list(dataset["roles"].values()).count("anchor_align") == 1
    assert list(dataset["roles"].values()).count("anchor") == 2
    assert list(dataset["roles"].values()).count("test") == 7

    observed_nodes = set()
    for source, target, distance_text in dataset["distance_data"]:
        observed_nodes.update((source, target))
        actual = np.linalg.norm(expected[dni[target]] - expected[dni[source]])
        assert actual == pytest.approx(float(distance_text), abs=0.5)
    assert observed_nodes == set(vertices)

    non_octilinear_bearings = 0
    for row, (source, target, direction) in zip(
        dataset["direction_rows"], dataset["direction_data"]
    ):
        assert source in dni and target in dni
        assert direction in DIR8_UNIT_SIM
        vector = expected[dni[target]] - expected[dni[source]]
        unit = vector / np.linalg.norm(vector)
        desired = np.asarray(DIR8_UNIT_SIM[direction], dtype=float)
        angle = math.acos(float(np.clip(np.dot(unit, desired), -1.0, 1.0)))
        threshold = math.pi / 2 if direction in DIR4_SIM else math.pi / 4
        assert angle <= threshold + 1e-12

        bearing = math.degrees(math.atan2(float(vector[1]), float(vector[0])))
        assert float(row["expected_bearing_deg"]) == pytest.approx(bearing, abs=1e-6)
        nearest_octilinear = round(bearing / 45.0) * 45.0
        if abs(bearing - nearest_octilinear) > 1.0:
            non_octilinear_bearings += 1

    assert non_octilinear_bearings == len(dataset["direction_data"])

    distance_pairs = {
        frozenset((source, target))
        for source, target, _distance in dataset["distance_data"]
    }
    direction_only_pairs = {
        frozenset((source, target))
        for source, target, _direction in dataset["direction_data"]
        if frozenset((source, target)) not in distance_pairs
    }
    assert len(direction_only_pairs) >= 6


def test_small_dataset_is_sparse_locally_rigid_and_unique_up_to_congruence():
    dataset = _load_dataset()
    vertices = dataset["vertices"]
    dni = dataset["dni"]
    expected = dataset["expected"]
    n = len(vertices)

    complete_edge_count = n * (n - 1) // 2
    assert len(dataset["edges"]) == 3 * n - 6
    assert len(dataset["edges"]) / complete_edge_count < 0.6

    edge_distances = {
        frozenset((source, target)): float(distance)
        for source, target, distance in dataset["distance_data"]
    }
    order_rows = dataset["trilateration_rows"]
    assert [row["node"] for row in order_rows] == vertices

    base = expected[:3]
    first_side = base[1] - base[0]
    second_side = base[2] - base[0]
    base_twice_area = first_side[0] * second_side[1] - first_side[1] * second_side[0]
    assert abs(float(base_twice_area)) > 1e-9

    localized = set(vertices[:3])
    for row in order_rows[3:]:
        node = row["node"]
        parents = [row["parent_1"], row["parent_2"], row["parent_3"]]
        assert len(set(parents)) == 3
        assert set(parents) <= localized
        for parent in parents:
            assert frozenset((node, parent)) in edge_distances

        parent_points = expected[[dni[parent] for parent in parents]]
        coefficient_matrix = 2.0 * np.vstack(
            (parent_points[1] - parent_points[0], parent_points[2] - parent_points[0])
        )
        assert np.linalg.matrix_rank(coefficient_matrix) == 2

        radii = np.asarray(
            [edge_distances[frozenset((node, parent))] for parent in parents],
            dtype=float,
        )
        squared_norms = np.sum(np.square(parent_points), axis=1)
        right_hand_side = np.asarray(
            [
                radii[0] ** 2
                - radii[index] ** 2
                - squared_norms[0]
                + squared_norms[index]
                for index in (1, 2)
            ]
        )
        unique_solution = np.linalg.solve(coefficient_matrix, right_hand_side)
        assert unique_solution == pytest.approx(expected[dni[node]], abs=1e-9)
        localized.add(node)

    rigidity = np.zeros((len(dataset["edges"]), 2 * n), dtype=float)
    for row_index, (source, target) in enumerate(dataset["edges"]):
        source_index = dni[source]
        target_index = dni[target]
        difference = expected[source_index] - expected[target_index]
        rigidity[row_index, 2 * source_index : 2 * source_index + 2] = difference
        rigidity[row_index, 2 * target_index : 2 * target_index + 2] = -difference
    assert np.linalg.matrix_rank(rigidity) == 2 * n - 3


def test_small_dataset_metrics_are_zero_for_truth_and_detect_bad_layout():
    dataset = _load_dataset()

    perfect = _quality_metrics(dataset["expected"], dataset)
    assert perfect == pytest.approx({"rmse": 0.0, "stress": 0.0, "vr": 0.0, "mae": 0.0}, abs=1e-12)

    bad_layout = dataset["expected"].copy()
    bad_layout[dataset["dni"]["J"]] += [300.0, -400.0]
    bad = _quality_metrics(bad_layout, dataset)
    assert bad == pytest.approx(
        {
            "rmse": math.sqrt(500.0**2 / 10.0),
            "stress": 0.21427488369818257,
            "vr": 1.0 / 12.0,
            "mae": 0.20018106796588864,
        },
        abs=1e-12,
    )


def test_small_dataset_runs_smacof(monkeypatch):
    dataset = _load_dataset()
    monkeypatch.setattr(smacof, "iteration_times", 1000)
    np.random.seed(2)

    points, stress_history, position_history = smacof.stress_majorization(
        dataset["graph"],
        dataset["dni"],
        dataset["vertices"],
        dataset["edges"],
    )

    _assert_finite_layout(points, len(dataset["vertices"]))
    assert len(stress_history) == len(position_history)
    assert len(stress_history) >= 2
    assert np.all(np.isfinite(np.asarray(stress_history, dtype=float)))
    aligned = _rigid_align(points, dataset["expected"], _calibration_indices(dataset))
    quality = _quality_metrics(aligned, dataset)
    assert quality["rmse"] < 1e-6
    assert quality["stress"] < 1e-9
    assert quality["vr"] == 0.0
    assert quality["mae"] == 0.0


def test_small_dataset_runs_dc_smacof(monkeypatch):
    dataset = _load_dataset()
    monkeypatch.setattr(dc_smacof, "stop_iteration_times", 500)
    np.random.seed(2)
    c_data = [dataset["direction_data"], [], []]

    points, stress_history, position_history = dc_smacof.directed_MDS(
        c_data,
        dataset["distance_data"],
        dataset["graph"],
        dataset["vertices"],
        dataset["dni"],
        dataset["edges"],
        distance_weight=1.0,
        direction_weight=0.012,
    )

    _assert_finite_layout(points, len(dataset["vertices"]))
    assert len(stress_history) == len(position_history)
    assert len(stress_history) == 502
    assert np.all(np.isfinite(np.asarray(stress_history, dtype=float)))
    aligned = _translate_to_frame_anchor(points, dataset)
    quality = _quality_metrics(aligned, dataset)
    # Non-octilinear true bearings cannot exactly equal all DIR8 center vectors.
    # The production model should nevertheless recover the known frame closely.
    assert quality["rmse"] < 1.0
    assert quality["stress"] < 0.002
    assert quality["vr"] == 0.0
    assert quality["mae"] == 0.0


@pytest.mark.parametrize(
    ("variant", "use_direction", "use_anchors", "use_repulsion"),
    [
        ("PhysicsSim-DistOnly", False, False, False),
        ("PhysicsSim-DistDir", True, False, False),
        ("PhysicsSim-DistDirAnch", True, True, False),
        ("PhysicsSim-Full", True, True, True),
    ],
)
def test_small_dataset_runs_physics_variants(
    monkeypatch,
    variant,
    use_direction,
    use_anchors,
    use_repulsion,
):
    dataset = _load_dataset()
    monkeypatch.setattr(physics, "stop_physim_iteration_time", 1000)
    rng = np.random.default_rng(0)
    initial = dataset["expected"] + rng.normal(0.0, 30.0, dataset["expected"].shape)

    anchor_labels = [
        label
        for label, role in dataset["roles"].items()
        if role in {"anchor", "anchor_align"}
    ]
    if use_anchors:
        for label in anchor_labels:
            initial[dataset["dni"][label]] = dataset["expected"][dataset["dni"][label]]
    fixed_positions = [[label] for label in anchor_labels] if use_anchors else []

    _wrong, stress_history, position_history, points = physics.main_physics_simulation(
        vertice=dataset["vertices"],
        dni=dataset["dni"],
        data=dataset["distance_data"],
        pos_matrix=initial.tolist(),
        directional_data=dataset["direction_data"] if use_direction else [],
        fixed_positions_list=fixed_positions,
        spring_stiffness=1500.0,
        repulsion_strength=158.11388300841898 if use_repulsion else 0.0,
        directional_force_magnitude=10000000.0 if use_direction else 0.0,
        plot=False,
    )

    _assert_finite_layout(points, len(dataset["vertices"]))
    assert len(stress_history) == len(position_history) == 1001, variant
    assert np.all(np.isfinite(np.asarray(stress_history, dtype=float)))

    intrinsic = _rigid_align(points, dataset["expected"])
    intrinsic_quality = _quality_metrics(intrinsic, dataset)
    assert intrinsic_quality["rmse"] < 0.01, variant

    if use_anchors:
        evaluated = np.asarray(points, dtype=float)
    elif use_direction:
        evaluated = _translate_to_frame_anchor(points, dataset)
    else:
        evaluated = _rigid_align(points, dataset["expected"], _calibration_indices(dataset))
    quality = _quality_metrics(evaluated, dataset)
    assert quality["rmse"] < (15.0 if variant == "PhysicsSim-DistDir" else 0.01), variant
    assert quality["stress"] < 1e-5, variant
    assert quality["vr"] == 0.0, variant
    assert quality["mae"] == 0.0, variant

    if use_anchors:
        for label in anchor_labels:
            index = dataset["dni"][label]
            assert np.asarray(points[index]) == pytest.approx(dataset["expected"][index], abs=1e-6)
