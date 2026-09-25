from pathlib import Path

import pandas as pd

from scripts.audit_distance_triangle_inequality import _undirected_pair, audit_triangles, load_distances


def test_known_triangle_violation_is_detected():
    distances = {
        _undirected_pair("車師前", "高昌壁"): 15.0,
        _undirected_pair("高昌壁", "蒲類"): 235.0,
        _undirected_pair("蒲類", "車師前"): 210.0,
    }
    triangles, violations = audit_triangles(distances)
    assert len(triangles) == 1
    assert len(violations) == 1
    row = violations.iloc[0]
    assert row["long_edge_distance_li"] == 235.0
    assert row["alternative_two_edge_sum_li"] == 225.0
    assert row["violation_amount_li"] == 10.0
    assert row["violation_fraction_of_long_edge"] == 10.0 / 235.0
    assert row["violation_fraction_of_two_edge_sum"] == 10.0 / 225.0


def test_missing_edge_does_not_create_an_imputed_triangle():
    triangles, violations = audit_triangles(
        {_undirected_pair("A", "B"): 2.0, _undirected_pair("B", "C"): 3.0}
    )
    assert triangles.empty
    assert violations.empty


def test_project_distance_file_has_unique_undirected_edges():
    path = Path(__file__).resolve().parents[1] / "data" / "distance_edges_verified.csv"
    distances, source_rows = load_distances(path)
    assert source_rows == 44
    assert len(distances) == 44
