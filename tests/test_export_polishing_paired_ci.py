import numpy as np
import pandas as pd
import pytest

from scripts.export_polishing_paired_ci import calculate_paired_ci


def _runs():
    seeds = np.arange(100)
    before_objective = seeds.astype(float) * 10
    before_rmse = seeds.astype(float)
    return pd.DataFrame(
        {
            "seed": seeds,
            "before_objective_total": before_objective,
            "after_objective_total": before_objective - 7,
            "delta_objective_total": np.full(100, -7.0),
            "before_RMSE_test_km_posthoc": before_rmse,
            "after_RMSE_test_km_posthoc": before_rmse + 3,
            "delta_RMSE_test_km_posthoc": np.full(100, 3.0),
        }
    )


def test_paired_ci_uses_after_minus_before_and_preserves_seed_pairing():
    runs = _runs().sample(frac=1, random_state=8)
    paired, summary = calculate_paired_ci(runs, list(range(100)))
    assert paired["seed"].tolist() == list(range(100))
    assert summary["paired_mean_difference"].tolist() == pytest.approx([-7, 3])
    assert summary["paired_mean_ci95_lower"].tolist() == pytest.approx([-7, 3])
    assert summary["paired_mean_ci95_upper"].tolist() == pytest.approx([-7, 3])


def test_paired_ci_rejects_duplicate_seed_and_incorrect_delta():
    runs = _runs()
    runs.loc[1, "seed"] = 0
    with pytest.raises(ValueError, match="exactly once"):
        calculate_paired_ci(runs, list(range(100)))
    runs = _runs()
    runs.loc[1, "delta_objective_total"] = 0
    with pytest.raises(ValueError, match="after minus before"):
        calculate_paired_ci(runs, list(range(100)))


def test_paired_ci_rejects_nonfinite_and_incomplete_formal_runs():
    runs = _runs()
    runs.loc[1, "after_RMSE_test_km_posthoc"] = np.inf
    with pytest.raises(ValueError, match="Non-finite"):
        calculate_paired_ci(runs, list(range(100)))
    with pytest.raises(ValueError, match="do not match"):
        calculate_paired_ci(_runs().iloc[:-1], list(range(100)))
