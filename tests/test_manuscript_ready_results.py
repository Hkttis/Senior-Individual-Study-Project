from __future__ import annotations

import numpy as np

from scripts.build_manuscript_ready_results import (
    AS_DIR,
    BFGS_DIR,
    DEFAULT_OUTDIR,
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
    direction_layout = panel_b.loc[panel_b["Added component"] == "Direction"].iloc[0]
    anchors_layout = panel_b.loc[panel_b["Added component"] == "Anchors"].iloc[0]
    repulsion = panel_b.loc[panel_b["Added component"] == "Repulsion"].iloc[0]
    assert direction["ΔRMSE (km) [95% CI]"] == "-333 [-362, -304]"
    assert direction_layout["ΔCrowding Violation Rate (τ = 0.10) [95% CI]"] == "-0.0005 [-0.0008, -0.0002]"
    assert anchors_layout["ΔCrowding Violation Rate (τ = 0.10) [95% CI]"] == "0.0001 [-0.0002, 0.0004]"
    assert repulsion["ΔCrowding Violation Rate (τ = 0.10) [95% CI]"] == "-0.0004 [-0.0006, -0.0002]"
    assert repulsion["ΔNearest-Neighbor Distance 5th Quantile (km) [95% CI]"] == "1.7 [1.0, 2.5]"
    assert len(audit2) == 24
    assert audit2["metric"].eq("crowding_violation_rate_tau_0p1").sum() == 3

    bfgs = table3.loc[table3["Model"] == "BFGS"].iloc[0]
    full = table3.loc[table3["Model"] == "PhysicsSim-Full"].iloc[0]
    assert bfgs["RMSE (km)"] == "312 ± 42"
    assert bfgs["Crowding Violation Rate (τ = 0.10)"] == "0.0020 ± 0.0008"
    assert "Collapse Node Rate (τ = 0.10)" in table3.columns
    assert "Nearest-Neighbor Distance, 5th Quantile (km)" in table3.columns
    assert "Crossing-edge Rate" in table3.columns
    assert full["Stress"] == "0.060 ± 0.017"
    assert full["Crowding Violation Rate (τ = 0.10)"] == "0.0019 ± 0.0005"
    assert len(audit3) == 48
    for metric in (
        "crowding_violation_rate_tau_0p1",
        "collapse_node_rate_tau_0p1",
        "nnd_q05_km",
        "distance_edge_crossing_rate",
    ):
        assert audit3["metric"].eq(metric).sum() == 6


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


def test_manuscript_table_formats_all_include_protocol_required_cvr():
    for stem in (
        "table_2_progressive_component_effects",
        "table_3_information_matched_optimizer_comparison",
        ):
        for suffix in (".csv", ".md", ".tex"):
            path = DEFAULT_OUTDIR / "01_main_tables" / f"{stem}{suffix}"
            normalized_text = path.read_text(encoding="utf-8-sig").replace(r"\\", " ")
            assert "Crowding Violation Rate" in normalized_text


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
