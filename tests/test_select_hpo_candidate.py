from pathlib import Path

import pandas as pd
import pytest

from scripts.select_hpo_candidate import _find_candidate_row, select_hpo_candidate


def test_find_candidate_row_uses_alpha_beta_with_float_tolerance():
    df = pd.DataFrame(
        [
            {"alpha": 0.5, "beta": -1.5, "RMSE_anchor_LOO_mean_km": 157.0},
            {"alpha": 1.0, "beta": -1.5, "RMSE_anchor_LOO_mean_km": 145.0},
        ]
    )

    row = _find_candidate_row(df, 1.0, -1.5)

    assert row["RMSE_anchor_LOO_mean_km"] == pytest.approx(145.0)


def test_find_candidate_row_raises_when_missing():
    df = pd.DataFrame([{"alpha": 0.5, "beta": -1.5}])

    with pytest.raises(ValueError):
        _find_candidate_row(df, 1.0, -1.5)


def test_select_hpo_candidate_refuses_nonempty_outdir():
    source = Path("outputs") / "test_tmp" / "select_source"
    outdir = Path("outputs") / "test_tmp" / "select_out"
    source.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "existing.txt").write_text("keep", encoding="utf-8")

    pd.DataFrame([{"alpha": 1.0, "beta": 0.5, "RMSE_anchor_LOO_mean_km": 1.0}]).to_csv(
        source / "grid_summary_cv.csv", index=False
    )
    pd.DataFrame([{"alpha": 1.0, "beta": 0.5, "RMSE_anchor_LOO_mean_km": 1.0}]).to_csv(
        source / "pareto_front_3d.csv", index=False
    )

    with pytest.raises(FileExistsError):
        select_hpo_candidate(
            source_hpo_outdir=source,
            alpha=1.0,
            beta=0.5,
            seeds=[0],
            outdir=outdir,
        )
