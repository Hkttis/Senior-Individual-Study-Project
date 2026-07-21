import os

import numpy as np
import pandas as pd
import pytest

import library.visualization as visualization
from run_paper_script.ch6_visualize_progressive_representative import (
    SELECTION_METRICS,
    _select_representative_seed,
    _verify_rerun,
)


def test_ground_truth_overlay_rmse_uses_only_explicit_test_labels(monkeypatch, tmp_path):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setattr(visualization, "px_list_to_km_list", lambda positions, *_args: positions)
    monkeypatch.setattr(visualization, "lcc_transformation", lambda *_args: [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)])
    monkeypatch.setattr(visualization, "flipping_gt", lambda positions: positions)
    monkeypatch.setattr(visualization, "calculate_kruskals_stress", lambda *_args: 0.0)
    monkeypatch.setattr(visualization, "get_anchor_labels", lambda: [])
    monkeypatch.setattr(visualization, "get_anchor_align_label", lambda: "A")

    result = visualization.ground_truth_comparison(
        ["A", "B", "C"],
        {"A": 0, "B": 1, "C": 2},
        [],
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        [0.0, 0.0],
        [[9.0, 0.0], [2.0, 0.0], [8.0, 0.0]],
        "test_",
        wait=False,
        eval_labels=["B"],
        output_dir=tmp_path,
    )

    assert result["n_evaluated"] == 1
    assert result["eval_labels"] == ["B"]
    assert result["rmse_km"] == pytest.approx(1.0)
    assert (tmp_path / "test_Overlap.png").exists()


def test_progressive_representative_selection_and_full_metric_verification():
    rows = []
    for seed, base in [(0, 10.0), (1, 2.0), (2, 30.0)]:
        row = {"variant": "PhysicsSim-Full", "seed": seed, "status": "ok"}
        for metric in SELECTION_METRICS:
            row[metric] = base
        row.update({
            "E_distance_stress": base,
            "E_direction_vr": base,
            "E_direction_mae": base,
            "RMSE_test_km": base,
        })
        rows.append(row)
    selected = _select_representative_seed(pd.DataFrame(rows))
    assert selected["seed"] == 0

    recorded = pd.Series(rows[0])
    matched = {metric: recorded[metric] for metric in SELECTION_METRICS}
    verification = _verify_rerun(recorded, matched, abs_tol=1e-8, rel_tol=1e-8)
    assert len(verification) == len(SELECTION_METRICS)
    assert all(row["ok"] for row in verification)

    mismatched = dict(matched)
    mismatched["RMSE_test_km"] += 1.0
    with pytest.raises(ValueError, match="Rerun metrics do not match"):
        _verify_rerun(recorded, mismatched, abs_tol=1e-8, rel_tol=1e-8)
