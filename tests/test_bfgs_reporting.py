from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backup_bfgs_experiments import backup_experiments
from scripts.export_bfgs_representative_minima import (
    select_representative_minima,
    verify_reproduced_minima,
)
from scripts.export_physics_bfgs_comparison import (
    METRICS,
    build_physics_bfgs_comparison,
    export_physics_bfgs_comparison,
    verify_exported_comparison,
)


def _representative_runs() -> pd.DataFrame:
    rows = []
    for seed, objective in enumerate([-100.0, -98.0, -95.0, -93.0, 100.0, 102.0, 106.0, 109.0]):
        rows.append(
            {
                "seed": seed,
                "status": "ok",
                "objective_final": objective,
                "RMSE_test_km": 200.0 + seed,
                "E_distance_stress": 0.02 + seed * 0.001,
                "E_direction_vr": 0.1 + seed * 0.01,
                "E_direction_mae": 0.2 + seed * 0.01,
                "gradient_norm_inf": 1e-4,
            }
        )
    return pd.DataFrame(rows)


def test_representative_minima_selects_one_seed_per_objective_stratum():
    selected, thresholds = select_representative_minima(_representative_runs(), n_strata=2)
    assert selected["objective_stratum"].tolist() == [1, 2]
    assert selected["stratum_n"].tolist() == [4, 4]
    assert selected["seed"].is_unique
    assert len(thresholds) == 1
    assert -93.0 < thresholds[0] < 100.0


def test_representative_rerun_verifier_checks_metrics_and_positions():
    runs = _representative_runs().iloc[:2].copy()
    positions = pd.DataFrame(
        [
            {"seed": seed, "label": label, "x_y_up_sim": seed + index, "y_y_up_sim": index}
            for seed in (0, 1)
            for index, label in enumerate(("A", "B"))
        ]
    )
    report = verify_reproduced_minima(runs, positions, runs.copy(), positions.copy(), [0, 1])
    assert report["verified"].all()
    assert np.allclose(report["max_abs_position_difference_sim"], 0.0)


def _comparison_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    as_rows = []
    bfgs_rows = []
    for seed in range(100):
        as_row = {"variant": "PhysicsSim-Full", "seed": seed, "status": "ok"}
        bfgs_row = {"seed": seed, "status": "ok"}
        for index, (metric, _label) in enumerate(METRICS, start=1):
            as_row[metric] = float(index + seed / 1000.0)
            bfgs_row[metric] = float(index + 0.5 + seed / 1000.0)
        as_rows.append(as_row)
        bfgs_rows.append(bfgs_row)
    return pd.DataFrame(as_rows), pd.DataFrame(bfgs_rows)


def test_physics_bfgs_table_uses_sample_sd_and_paired_bootstrap_ci():
    as_runs, bfgs_runs = _comparison_frames()
    table, audit = build_physics_bfgs_comparison(as_runs=as_runs, bfgs_runs=bfgs_runs)
    assert table["Model"].tolist() == [
        "PhysicsSim-Full",
        "SciPy-BFGS",
        "Paired difference: BFGS − PhysicsSim-Full",
    ]
    assert table.iloc[2]["RMSE (km)"] == "0.5 [0.5, 0.5]"
    paired = audit[audit["row"] == "BFGS − PhysicsSim-Full"]
    assert np.allclose(paired["mean"], 0.5)
    assert np.allclose(paired["sample_sd"], 0.0, atol=1e-15)


def test_exported_physics_bfgs_table_is_recomputable(tmp_path: Path):
    as_runs, bfgs_runs = _comparison_frames()
    as_dir = tmp_path / "as"
    bfgs_dir = tmp_path / "bfgs"
    outdir = tmp_path / "tables"
    as_dir.mkdir()
    bfgs_dir.mkdir()
    as_runs.to_csv(as_dir / "progressive_runs_by_seed.csv", index=False)
    bfgs_runs.to_csv(bfgs_dir / "bfgs_runs_by_seed.csv", index=False)
    paths = export_physics_bfgs_comparison(
        as_outdir=as_dir, bfgs_outdir=bfgs_dir, outdir=outdir
    )
    verify_exported_comparison(
        as_outdir=as_dir, bfgs_outdir=bfgs_dir, table_csv=paths["csv"]
    )
    assert paths["latex"].exists()
    assert paths["markdown"].exists()


def test_backup_experiments_copies_sources_and_writes_hash_manifest(tmp_path: Path):
    source = tmp_path / "formal_bfgs"
    source.mkdir()
    (source / "result.csv").write_text("seed,value\n0,1\n", encoding="utf-8")
    outdir = tmp_path / "backup"
    manifest = backup_experiments(sources=[source], outdir=outdir)
    rows = pd.read_csv(manifest)
    assert rows["backup_path"].tolist() == ["formal_bfgs/result.csv"]
    assert (outdir / "formal_bfgs" / "result.csv").read_text(encoding="utf-8") == "seed,value\n0,1\n"
