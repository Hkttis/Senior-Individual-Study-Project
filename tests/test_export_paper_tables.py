import pandas as pd

from scripts.export_paper_tables import (
    DISTONLY_VS_DISTDIR_METRICS,
    DISTDIR_VS_DISTDIRANCH_METRICS,
    DISTDIRANCH_VS_FULL_METRICS,
    PROGRESSIVE_CHAIN_VARIANTS,
    RANDOM_VARIANT,
    SUMMARY_COLUMNS,
    export_distonly_vs_distdir_comparison,
    export_distdir_vs_distdiranch_comparison,
    export_distdiranch_vs_full_comparison,
    export_smacof_vs_distonly_comparison,
    export_dc_smacof_vs_distdir_comparison,
    export_overall_model_comparison,
    export_smacof_dc_smacof_baseline_full_statistics,
    export_progressive_chain_summary,
    export_random_layout_summary,
)


def _summary_row(variant, metric, mean, std=0.5):
    return {
        "variant": variant,
        "metric": metric,
        "mean": mean,
        "std": std,
        "ci95_lo": mean - 0.1,
        "ci95_hi": mean + 0.1,
    }


def _assert_no_supplementary_error_columns(table):
    assert "Mean Absolute Error (km)" not in table.columns
    assert "Median Error (km)" not in table.columns


def _write_paired_runs(as_outdir, metrics, left_variant, right_variant, left_values, right_values):
    rows = []
    for seed, (left_value, right_value) in enumerate(zip(left_values, right_values)):
        left_row = {"variant": left_variant, "seed": seed}
        right_row = {"variant": right_variant, "seed": seed}
        for metric in metrics:
            left_row[metric] = left_value
            right_row[metric] = right_value
        rows.extend([left_row, right_row])
    pd.DataFrame(rows).to_csv(as_outdir / "progressive_runs_by_seed.csv", index=False)


def test_export_random_layout_summary_writes_aggregate_tables_only(tmp_path):
    as_outdir = tmp_path / "as"
    as_outdir.mkdir()
    row = {"variant": RANDOM_VARIANT}
    row.update({column: 1.0 for column in SUMMARY_COLUMNS if column not in {"metric", "n"}})
    row["metric"] = "RMSE_test_km"
    row["n"] = 1000
    pd.DataFrame([row]).to_csv(as_outdir / "random_align_summary.csv", index=False)

    paths = export_random_layout_summary(as_outdir=as_outdir, outdir=tmp_path / "tables")

    table = pd.read_csv(paths["csv"])
    mean_sd_table = pd.read_csv(paths["mean_sd_csv"])
    markdown = paths["markdown"].read_text(encoding="utf-8")
    assert list(table.columns) == SUMMARY_COLUMNS
    assert table.loc[0, "n"] == 1000
    assert list(mean_sd_table.columns) == ["metric", "Mean ± SD"]
    assert mean_sd_table.loc[0, "Mean ± SD"] == "1 ± 1"
    assert "Per-run positions are intentionally excluded." in markdown
    assert "x_y_up_sim" not in markdown


def test_export_progressive_chain_summary_keeps_all_four_variants(tmp_path):
    as_outdir = tmp_path / "as"
    as_outdir.mkdir()
    rows = []
    for variant in PROGRESSIVE_CHAIN_VARIANTS:
        row = {"variant": variant, "metric": "RMSE_test_km", "n": 100}
        row.update({column: 1.0 for column in SUMMARY_COLUMNS if column not in {"metric", "n"}})
        rows.append(row)
    pd.DataFrame(rows).to_csv(as_outdir / "progressive_summary.csv", index=False)

    paths = export_progressive_chain_summary(as_outdir=as_outdir, outdir=tmp_path / "tables")

    table = pd.read_csv(paths["csv"])
    mean_sd_table = pd.read_csv(paths["mean_sd_csv"])
    assert list(table.columns) == ["variant", *SUMMARY_COLUMNS]
    assert table["variant"].tolist() == PROGRESSIVE_CHAIN_VARIANTS
    assert mean_sd_table["variant"].tolist() == PROGRESSIVE_CHAIN_VARIANTS
    assert list(mean_sd_table.columns) == ["variant", "RMSE (km)"]
    assert mean_sd_table.loc[0, "RMSE (km)"] == "1 ± 1"


def test_export_distonly_vs_distdir_comparison_uses_requested_paired_metrics(tmp_path):
    as_outdir = tmp_path / "as"
    as_outdir.mkdir()
    summary_rows = []
    paired_rows = []
    for metric in DISTONLY_VS_DISTDIR_METRICS:
        for variant, mean in (("PhysicsSim-DistOnly", 2.0), ("PhysicsSim-DistDir", 1.0)):
            summary_rows.append(_summary_row(variant, metric, mean))
        paired_rows.append(
            {
                "comparison": "direction_given_distance",
                "left_variant": "PhysicsSim-DistDir",
                "right_variant": "PhysicsSim-DistOnly",
                "metric": metric,
                "paired_diff_mean": -1.0,
                "paired_diff_median": -1.0,
                "paired_diff_ci95_lo": -1.2,
                "paired_diff_ci95_hi": -0.8,
                "n_pairs": 100,
                "ci_excludes_zero": True,
            }
        )
    pd.DataFrame(summary_rows).to_csv(as_outdir / "progressive_summary.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(as_outdir / "progressive_paired_comparisons.csv", index=False)
    _write_paired_runs(
        as_outdir,
        DISTONLY_VS_DISTDIR_METRICS,
        "PhysicsSim-DistDir",
        "PhysicsSim-DistOnly",
        left_values=[1.0, 2.0],
        right_values=[2.0, 4.0],
    )

    paths = export_distonly_vs_distdir_comparison(as_outdir=as_outdir, outdir=tmp_path / "tables")

    table = pd.read_csv(paths["csv"])
    assert table["model"].tolist() == ["PhysicsSim-DistOnly", "PhysicsSim-DistDir", "DistDir - DistOnly paired"]
    assert table.loc[2, "RMSE (km)"] == "-1 [-1.2, -0.8]"


def test_export_distdir_vs_distdiranch_comparison_uses_anchor_pairing(tmp_path):
    as_outdir = tmp_path / "as"
    as_outdir.mkdir()
    summary_rows, paired_rows = [], []
    for metric in DISTDIR_VS_DISTDIRANCH_METRICS:
        for variant, mean in (("PhysicsSim-DistDir", 1.0), ("PhysicsSim-DistDirAnch", 2.0)):
            summary_rows.append(_summary_row(variant, metric, mean))
        paired_rows.append(
            {
                "comparison": "optimization_anchors_given_distance_direction",
                "left_variant": "PhysicsSim-DistDirAnch",
                "right_variant": "PhysicsSim-DistDir",
                "metric": metric,
                "paired_diff_mean": 1.0,
                "paired_diff_median": 1.0,
                "paired_diff_ci95_lo": 0.8,
                "paired_diff_ci95_hi": 1.2,
            }
        )
    pd.DataFrame(summary_rows).to_csv(as_outdir / "progressive_summary.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(as_outdir / "progressive_paired_comparisons.csv", index=False)
    _write_paired_runs(
        as_outdir,
        DISTDIR_VS_DISTDIRANCH_METRICS,
        "PhysicsSim-DistDirAnch",
        "PhysicsSim-DistDir",
        left_values=[2.0, 4.0],
        right_values=[1.0, 2.0],
    )

    paths = export_distdir_vs_distdiranch_comparison(as_outdir=as_outdir, outdir=tmp_path / "tables")

    table = pd.read_csv(paths["csv"])
    assert table["model"].tolist() == ["PhysicsSim-DistDir", "PhysicsSim-DistDirAnch", "DistDirAnch - DistDir paired"]
    assert table.loc[2, "RMSE (km)"] == "1 [0.8, 1.2]"
    assert "Mean Absolute Error (km)" not in table.columns


def test_export_distdiranch_vs_full_comparison_uses_repulsion_pairing(tmp_path):
    as_outdir = tmp_path / "as"
    as_outdir.mkdir()
    summary_rows, paired_rows = [], []
    for metric in DISTDIRANCH_VS_FULL_METRICS:
        for variant, mean in (("PhysicsSim-DistDirAnch", 2.0), ("PhysicsSim-Full", 1.0)):
            summary_rows.append(_summary_row(variant, metric, mean))
        paired_rows.append(
            {
                "comparison": "repulsion_given_distance_direction_anchors",
                "left_variant": "PhysicsSim-Full",
                "right_variant": "PhysicsSim-DistDirAnch",
                "metric": metric,
                "paired_diff_mean": -1.0,
                "paired_diff_median": -1.0,
                "paired_diff_ci95_lo": -1.2,
                "paired_diff_ci95_hi": -0.8,
            }
        )
    pd.DataFrame(summary_rows).to_csv(as_outdir / "progressive_summary.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(as_outdir / "progressive_paired_comparisons.csv", index=False)
    _write_paired_runs(
        as_outdir,
        DISTDIRANCH_VS_FULL_METRICS,
        "PhysicsSim-Full",
        "PhysicsSim-DistDirAnch",
        left_values=[1.0, 2.0],
        right_values=[2.0, 4.0],
    )

    paths = export_distdiranch_vs_full_comparison(as_outdir=as_outdir, outdir=tmp_path / "tables")

    table = pd.read_csv(paths["csv"])
    assert table["model"].tolist() == ["PhysicsSim-DistDirAnch", "PhysicsSim-Full", "Full - DistDirAnch paired"]
    _assert_no_supplementary_error_columns(table)
    assert list(table.columns[:9]) == [
        "model",
        "RMSE (km)",
        "Stress",
        "Violation Rate",
        "Mean Angular Error (rad)",
        "Crowding Violation Rate (τ = 0.10)",
        "Collapse Node Rate (τ = 0.10)",
        "Nearest-Neighbor Distance, 5th Quantile (km)",
        "Crossing-edge rate",
    ]
    assert table.loc[2, "Nearest-Neighbor Distance, 5th Quantile (km)"] == "-1 [-1.2, -0.8]"


def test_export_smacof_vs_distonly_comparison_uses_all_metrics_without_pairing(tmp_path):
    as_outdir = tmp_path / "as"
    as_outdir.mkdir()
    metrics = [
        "E_direction_mae", "E_direction_vr", "E_distance_stress", "MAE_test_km", "RMSE_test_km",
        "collapse_node_rate_tau_0p1", "crowding_violation_rate_tau_0p1", "distance_edge_crossing_rate",
        "median_error_km", "nnd_q05_km",
    ]
    rows = []
    for variant, mean in (("SMACOF", 1.0), ("PhysicsSim-DistOnly", 2.0)):
        for metric in metrics:
            rows.append(_summary_row(variant, metric, mean))
    pd.DataFrame(rows).to_csv(as_outdir / "progressive_summary.csv", index=False)

    paths = export_smacof_vs_distonly_comparison(as_outdir=as_outdir, outdir=tmp_path / "tables")

    table = pd.read_csv(paths["csv"])
    assert table["model"].tolist() == ["SMACOF", "PhysicsSim-DistOnly"]
    _assert_no_supplementary_error_columns(table)
    assert list(table.columns[:9]) == [
        "model",
        "RMSE (km)",
        "Stress",
        "Violation Rate",
        "Mean Angular Error (rad)",
        "Crowding Violation Rate (τ = 0.10)",
        "Collapse Node Rate (τ = 0.10)",
        "Nearest-Neighbor Distance, 5th Quantile (km)",
        "Crossing-edge rate",
    ]
    assert table.loc[0, "Nearest-Neighbor Distance, 5th Quantile (km)"] == "1 ± 0.5"


def test_export_dc_smacof_vs_distdir_comparison_uses_all_metrics_without_pairing(tmp_path):
    as_outdir = tmp_path / "as"
    as_outdir.mkdir()
    metrics = [
        "E_direction_mae", "E_direction_vr", "E_distance_stress", "MAE_test_km", "RMSE_test_km",
        "collapse_node_rate_tau_0p1", "crowding_violation_rate_tau_0p1", "distance_edge_crossing_rate",
        "median_error_km", "nnd_q05_km",
    ]
    rows = []
    for variant, mean in (("DC-SMACOF", 1.0), ("PhysicsSim-DistDir", 2.0)):
        for metric in metrics:
            rows.append(_summary_row(variant, metric, mean))
    pd.DataFrame(rows).to_csv(as_outdir / "progressive_summary.csv", index=False)

    paths = export_dc_smacof_vs_distdir_comparison(as_outdir=as_outdir, outdir=tmp_path / "tables")

    table = pd.read_csv(paths["csv"])
    assert table["model"].tolist() == ["DC-SMACOF", "PhysicsSim-DistDir"]
    _assert_no_supplementary_error_columns(table)
    assert list(table.columns[:9]) == [
        "model",
        "RMSE (km)",
        "Stress",
        "Violation Rate",
        "Mean Angular Error (rad)",
        "Crowding Violation Rate (τ = 0.10)",
        "Collapse Node Rate (τ = 0.10)",
        "Nearest-Neighbor Distance, 5th Quantile (km)",
        "Crossing-edge rate",
    ]
    assert table.loc[1, "RMSE (km)"] == "2 ± 0.5"


def test_export_overall_model_comparison_uses_three_designated_models(tmp_path):
    as_outdir = tmp_path / "as"
    as_outdir.mkdir()
    rows = []
    for variant, mean in (("SMACOF", 1.0), ("DC-SMACOF", 2.0), ("PhysicsSim-Full", 3.0)):
        rows.append(_summary_row(variant, "RMSE_test_km", mean))
    pd.DataFrame(rows).to_csv(as_outdir / "progressive_summary.csv", index=False)

    paths = export_overall_model_comparison(as_outdir=as_outdir, outdir=tmp_path / "tables")

    table = pd.read_csv(paths["csv"])
    assert table["model"].tolist() == ["SMACOF", "DC-SMACOF", "PhysicsSim-Full"]
    _assert_no_supplementary_error_columns(table)
    assert table.loc[2, "RMSE (km)"] == "3 ± 0.5"


def test_export_smacof_dc_smacof_baseline_full_statistics_keeps_all_summary_columns(tmp_path):
    as_outdir = tmp_path / "as"
    as_outdir.mkdir()
    rows = []
    for variant in ("SMACOF", "DC-SMACOF"):
        row = {"variant": variant, "metric": "RMSE_test_km", "n": 100}
        row.update({column: 1.0 for column in SUMMARY_COLUMNS if column not in {"metric", "n"}})
        rows.append(row)
    pd.DataFrame(rows).to_csv(as_outdir / "progressive_summary.csv", index=False)

    paths = export_smacof_dc_smacof_baseline_full_statistics(as_outdir=as_outdir, outdir=tmp_path / "tables")

    table = pd.read_csv(paths["csv"])
    assert list(table.columns) == ["variant", *SUMMARY_COLUMNS]
    assert table["variant"].tolist() == ["SMACOF", "DC-SMACOF"]
    assert table.loc[0, "n"] == 100
    export_distonly_vs_distdir_comparison,
