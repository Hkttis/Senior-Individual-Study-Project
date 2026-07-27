import numpy as np
import pandas as pd

from scripts.verify_manuscript_tables import (
    _bootstrap_ci,
    _mean_ci,
    _mean_sd,
    _paired_row,
    _verify_rendered_formats,
)


def test_mean_sd_uses_sample_standard_deviation():
    values = np.array([1.0, 2.0, 3.0])

    assert _mean_sd(values) == "2 ± 1"


def test_bootstrap_ci_is_deterministic_and_uses_requested_seed():
    values = np.array([-3.0, -1.0, 2.0, 6.0])

    first = _bootstrap_ci(values, n_boot=2000, seed=0)
    second = _bootstrap_ci(values, n_boot=2000, seed=0)

    assert first == second
    assert _mean_ci(values).startswith("1 ")


def test_paired_row_subtracts_same_seed_left_minus_right():
    runs = pd.DataFrame(
        [
            {"variant": "Left", "seed": 0, "status": "ok", "RMSE_test_km": 8.0},
            {"variant": "Left", "seed": 1, "status": "ok", "RMSE_test_km": 14.0},
            {"variant": "Right", "seed": 0, "status": "ok", "RMSE_test_km": 10.0},
            {"variant": "Right", "seed": 1, "status": "ok", "RMSE_test_km": 10.0},
        ]
    )

    row = _paired_row(
        runs,
        left="Left",
        right="Right",
        display_label="Left − Right",
        labels=["RMSE (km)"],
    )

    assert row["Model"] == "Left − Right"
    assert row["RMSE (km)"].startswith("1 ")


def test_rendered_format_verifier_rejects_changed_or_duplicated_tex_numbers(tmp_path):
    table = pd.DataFrame([{"Model": "A", "RMSE (km)": "10 ± 2"}])
    stem = "table"
    table.to_csv(tmp_path / f"{stem}.csv", index=False)
    (tmp_path / f"{stem}.md").write_text("Model RMSE (km) A 10 ± 2", encoding="utf-8")
    tex_path = tmp_path / f"{stem}.tex"
    tex_path.write_text(
        "% DATA_ROWS_BEGIN A & 10 $\\pm$ 2 \\\\ \\addlinespace[4pt] % DATA_ROWS_END",
        encoding="utf-8",
    )
    failures = []
    _verify_rendered_formats(stem, table, tmp_path, failures)
    assert failures == []

    tex_path.write_text("% DATA_ROWS_BEGIN A & 10 $\\pm$ 3 % DATA_ROWS_END", encoding="utf-8")
    failures = []
    _verify_rendered_formats(stem, table, tmp_path, failures)
    assert failures == ["TeX data-row numbers differ from CSV values in table"]
