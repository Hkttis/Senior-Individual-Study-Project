import numpy as np
import pandas as pd
import pytest

from library.config import km2pix
from scripts.visualize_detour_representatives import (
    _overlay_rmse,
    _select_representative_run,
    _verify_saved_metrics,
)


def test_detour_representative_selection_uses_all_four_metrics_and_not_best_rmse():
    runs = pd.DataFrame(
        {
            "seed": [0, 1, 2],
            "E_distance_stress": [0.01, 0.02, 0.50],
            "E_direction_vr": [0.01, 0.02, 0.50],
            "E_direction_mae": [0.01, 0.02, 0.50],
            "RMSE_final_test_km": [100.0, 120.0, 140.0],
        }
    )

    selected, details = _select_representative_run(runs)

    assert int(selected["seed"]) == 1
    assert set(details["median_vector"]) == {
        "E_distance_stress", "E_direction_vr", "E_direction_mae", "RMSE_final_test_km"
    }


def test_detour_overlay_rmse_uses_only_explicit_test_sites():
    points = np.asarray([[999.0, 999.0], [3.0 * km2pix, 4.0 * km2pix]])
    targets = {"anchor": np.zeros(2), "test": np.zeros(2)}

    assert _overlay_rmse(points, targets, ["test"], {"anchor": 0, "test": 1}) == pytest.approx(5.0)


def test_detour_visualization_verifies_all_four_saved_metrics():
    points = np.asarray([[0.0, 0.0], [10.0, 0.0]])
    data = [["A", "B", 10.0]]
    directions = [["A", "B", "東"]]
    targets = {"B": np.asarray([10.0, 0.0])}
    row = pd.Series(
        {
            "seed": 0,
            "E_distance_stress": 0.0,
            "E_direction_vr": 0.0,
            "E_direction_mae": 0.0,
            "RMSE_final_test_km": 0.0,
        }
    )

    metrics = _verify_saved_metrics(row, points, data, directions, targets, ["B"], {"A": 0, "B": 1})

    assert all(value == pytest.approx(0.0) for value in metrics.values())
    bad = row.copy()
    bad["RMSE_final_test_km"] = 1.0
    with pytest.raises(ValueError, match="RMSE_final_test_km"):
        _verify_saved_metrics(bad, points, data, directions, targets, ["B"], {"A": 0, "B": 1})
