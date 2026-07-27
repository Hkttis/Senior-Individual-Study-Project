import json

import pandas as pd

from scripts import verify_paper_results as verifier


VARIANT_COUNTS = {
    "PhysicsSim-DistOnly": 100,
    "PhysicsSim-DistDir": 100,
    "PhysicsSim-DistDirAnch": 100,
    "PhysicsSim-Full": 100,
    "SMACOF": 100,
    "DC-SMACOF": 100,
    "Random+Align": 1000,
}


def _write_synthetic_as(folder, monkeypatch):
    rows = []
    for variant_index, (variant, count) in enumerate(VARIANT_COUNTS.items()):
        for seed in range(count):
            value = float(variant_index + seed / 1000.0)
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "status": "ok",
                    "error": "",
                    "E_distance_stress": value,
                    "E_direction_vr": value,
                    "E_direction_mae": value,
                    "RMSE_test_km": value,
                }
            )
    runs = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {"variant": variant, "metric": "RMSE_test_km", "mean": float(index)}
            for index, variant in enumerate(VARIANT_COUNTS)
        ]
    )
    paired = pd.DataFrame(
        [{"comparison": "direction_given_distance", "metric": "RMSE_test_km", "paired_diff_mean": 1.0}]
    )
    percentiles = pd.DataFrame(
        [{"variant": "PhysicsSim-Full", "metric": "RMSE_test_km", "mean_model_percentile_vs_random": 0.5}]
    )
    monkeypatch.setattr(verifier, "_summary", lambda _runs: summary.copy())
    monkeypatch.setattr(verifier, "_paired", lambda _runs: paired.copy())
    monkeypatch.setattr(verifier, "_random_percentiles", lambda _runs: percentiles.copy())

    runs.to_csv(folder / "progressive_runs_by_seed.csv", index=False)
    pd.DataFrame([{"x_y_up_sim": 1.0, "y_y_up_sim": 2.0}]).to_csv(
        folder / "progressive_final_positions_y_up_sim.csv", index=False
    )
    summary.to_csv(folder / "progressive_summary.csv", index=False)
    paired.to_csv(folder / "progressive_paired_comparisons.csv", index=False)
    percentiles.to_csv(folder / "random_align_percentiles.csv", index=False)
    runs[runs["variant"] == "Random+Align"].to_csv(folder / "random_align_runs.csv", index=False)
    summary[summary["variant"] == "Random+Align"].to_csv(folder / "random_align_summary.csv", index=False)
    runs.groupby(["variant", "status"], dropna=False).size().reset_index(name="n_runs").to_csv(
        folder / "progressive_run_status.csv", index=False
    )
    (folder / "progressive_config.json").write_text(
        json.dumps(
            {
                "failure_count": 0,
                "dc_smacof_hpo": {"alpha": -2.0, "w_weight": 1.0, "v_weight": 0.01},
            }
        ),
        encoding="utf-8",
    )


def test_numerical_verifier_recomputes_and_detects_changed_summary(tmp_path, monkeypatch):
    _write_synthetic_as(tmp_path, monkeypatch)

    failures = []
    verifier._verify_numerical_experiment(tmp_path, failures)
    assert failures == []

    summary_path = tmp_path / "progressive_summary.csv"
    changed = pd.read_csv(summary_path)
    changed.loc[0, "mean"] += 1.0
    changed.to_csv(summary_path, index=False)

    failures = []
    verifier._verify_numerical_experiment(tmp_path, failures)
    assert any("progressive_summary.csv recomputed from runs" in failure for failure in failures)

