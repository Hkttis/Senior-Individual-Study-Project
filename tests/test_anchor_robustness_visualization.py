import numpy as np
import pandas as pd
import pytest

from library.config import km2pix, refer_pos_sim
from scripts.visualize_anchor_robustness_overlays import (
    _constraint_metrics,
    _overlay_rmse,
    _rebase_points,
    _select_representative_run,
)


def test_representative_selection_uses_multimetric_median_not_best_rmse():
    runs = pd.DataFrame(
        {
            "seed": [0, 1, 2],
            "E_distance_stress": [0.01, 0.02, 0.50],
            "E_direction_vr": [0.01, 0.02, 0.50],
            "RMSE_final_test_km": [100.0, 120.0, 140.0],
        }
    )

    assert int(_select_representative_run(runs)["seed"]) == 1


def test_rebase_preserves_displacements_in_common_anchor_frame():
    origin = np.asarray(refer_pos_sim, dtype=float)
    points = np.asarray([origin, origin + np.asarray([3.0, 4.0])])
    target = origin + np.asarray([20.0, -10.0])

    rebased = _rebase_points(points, "A", {"A": target})

    assert rebased[0] == pytest.approx(target)
    assert rebased[1] - rebased[0] == pytest.approx([3.0, 4.0])


def test_overlay_rmse_uses_only_explicit_heldout_sites():
    points = np.asarray([[999.0, 999.0], [3.0 * km2pix, 4.0 * km2pix]])
    targets = {"anchor": np.zeros(2), "test": np.zeros(2)}

    assert _overlay_rmse(points, targets, ["test"], {"anchor": 0, "test": 1}) == pytest.approx(5.0)


def test_constraint_metrics_are_invariant_to_coordinate_frame_translation():
    points = np.asarray([[0.0, 0.0], [10.0, 0.0]])
    data_sim = [["A", "B", 10.0]]
    directional_data = [["A", "B", "東"]]
    dni = {"A": 0, "B": 1}

    before = _constraint_metrics(points, data_sim, directional_data, dni)
    after = _constraint_metrics(points + np.asarray([100.0, -250.0]), data_sim, directional_data, dni)

    assert before == pytest.approx(after)
    assert before[0] == pytest.approx(0.0)
    assert before[1] == pytest.approx(0.0)
