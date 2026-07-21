import pandas as pd
import pytest

from run_paper_script.ch5_hparam_kfold_gridsearch_pareto import _selected_grid_row


def test_selected_grid_row_returns_the_manual_candidate_row():
    grid = pd.DataFrame(
        {
            "alpha": [0.5, 1.0],
            "beta": [-1.0, -0.5],
            "E_distance_stress_mean": [0.1, 0.2],
        }
    )

    row = _selected_grid_row(grid, pd.Series({"alpha": 1.0, "beta": -0.5}))

    assert row["E_distance_stress_mean"] == 0.2


def test_selected_grid_row_rejects_a_candidate_missing_from_the_grid():
    grid = pd.DataFrame({"alpha": [1.0], "beta": [-0.5]})

    with pytest.raises(ValueError, match="exactly one HPO grid point"):
        _selected_grid_row(grid, pd.Series({"alpha": 0.0, "beta": 0.0}))
