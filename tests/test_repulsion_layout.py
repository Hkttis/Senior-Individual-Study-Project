import numpy as np
import pytest

from scripts.evaluate_repulsion_layout import _topology_metrics


def test_distance_edge_crossing_rate_counts_strict_nonadjacent_crossing():
    labels = ["A", "B", "C", "D"]
    points = np.asarray([[0.0, 0.0], [2.0, 2.0], [0.0, 2.0], [2.0, 0.0]])

    result = _topology_metrics(points, labels, [("A", "B"), ("C", "D")], tau_km=0.1)

    assert result["distance_edge_crossing_rate"] == pytest.approx(1.0)
    assert "distance_edge_crossing_count" not in result


def test_distance_edge_crossing_rate_excludes_edges_sharing_a_node():
    labels = ["A", "B", "C"]
    points = np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])

    result = _topology_metrics(points, labels, [("A", "B"), ("B", "C")], tau_km=0.1)

    assert result["distance_edge_crossing_rate"] == pytest.approx(0.0)


def test_distance_edge_crossing_rate_excludes_endpoint_touching():
    labels = ["A", "B", "C", "D"]
    points = np.asarray([[0.0, 0.0], [2.0, 0.0], [2.0, 0.0], [2.0, 2.0]])

    result = _topology_metrics(points, labels, [("A", "B"), ("C", "D")], tau_km=0.1)

    assert result["distance_edge_crossing_rate"] == pytest.approx(0.0)
