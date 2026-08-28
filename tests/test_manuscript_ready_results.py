from __future__ import annotations

import numpy as np

from scripts.build_manuscript_ready_results import (
    AS_DIR,
    BFGS_DIR,
    DETOUR_DIR,
    POLISH_DIR,
    build_table_1,
    build_table_2,
    build_table_3,
    load_runs,
)
from scripts.create_manuscript_spatial_comparisons import select_bfgs_representative


def test_main_tables_are_recomputed_from_registered_formal_runs():
    as_runs, random_runs, bfgs_runs = load_runs()
    table1, audit1 = build_table_1(as_runs, random_runs, bfgs_runs)
    panel_a, panel_b, audit2 = build_table_2()
    table3, audit3 = build_table_3(as_runs, bfgs_runs)

    assert table1.loc[table1["Model"] == "Random+Align", "Held-out RMSE, mean ± SD (km)"].item() == "663 ± 156"
    assert table1.loc[table1["Model"] == "BFGS", "Held-out RMSE, mean ± SD (km)"].item() == "312 ± 42"
    assert table1.loc[table1["Model"] == "PhysicsSim-Full", "RMSE reduction vs Random+Align"].item() == "72%"
    assert len(audit1) == 8

    direction = panel_a.loc[panel_a["Added component"] == "Direction"].iloc[0]
    repulsion = panel_b.loc[panel_b["Added component"] == "Repulsion"].iloc[0]
    assert direction["ΔRMSE (km) [95% CI]"] == "-333 [-362, -304]"
    assert repulsion["ΔNearest-Neighbor Distance, 5th Quantile (km) [95% CI]"] == "1.7 [1.0, 2.5]"
    assert len(audit2) == 21

    bfgs = table3.loc[table3["Model"] == "BFGS"].iloc[0]
    full = table3.loc[table3["Model"] == "PhysicsSim-Full"].iloc[0]
    assert bfgs["RMSE (km)"] == "312 ± 42"
    assert full["Stress"] == "0.060 ± 0.017"
    assert len(audit3) == 24


def test_figure_source_files_have_registered_cardinality_and_units():
    import pandas as pd

    polishing = pd.read_csv(POLISH_DIR / "polishing_runs.csv")
    detour = pd.read_csv(DETOUR_DIR / "detour_scenario_summary.csv", encoding="utf-8-sig")
    assert len(polishing) == 100
    assert polishing["seed"].nunique() == 100
    assert polishing["optimizer_success"].astype(bool).all()
    assert np.isfinite(polishing[["before_objective_total", "after_objective_total", "before_RMSE_test_km_posthoc", "after_RMSE_test_km_posthoc"]]).all().all()
    assert len(detour) == 13
    assert detour["kappa"].nunique() == 13
    assert np.isclose(detour["kappa"].min(), 0.7)
    assert np.isclose(detour["kappa"].max(), 1.0)


def test_formal_model_sources_are_distinct_and_registered():
    assert AS_DIR.exists()
    assert BFGS_DIR.exists()
    assert AS_DIR != BFGS_DIR


def test_bfgs_spatial_representative_uses_same_four_metric_median_profile_rule():
    import pandas as pd

    runs = pd.read_csv(BFGS_DIR / "bfgs_runs_by_seed.csv", encoding="utf-8-sig")
    selected = select_bfgs_representative(runs)
    assert selected["seed"] == 75
    assert selected["source_variant"] == "SciPy-BFGS"
    assert set(selected["selection_metrics"]) == {
        "E_distance_stress",
        "E_direction_vr",
        "E_direction_mae",
        "RMSE_test_km",
    }
    assert np.isclose(selected["standardized_distance"], 0.5874598331057892)
