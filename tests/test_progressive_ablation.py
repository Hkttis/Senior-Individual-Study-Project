import numpy as np
import pandas as pd
import pytest

import run_paper_script.ch5_ablation_progressive as progressive


def test_progressive_physics_variants_use_expected_anchor_and_procrustes_paths(monkeypatch):
    captured = {"fixed": [], "procrustes": 0}

    def fake_generate(_refer, fixed_labels, _fixed_lonlat, anchor_label=None):
        captured["fixed"].append(list(fixed_labels))
        return ["A", "B", "C"], {"A": 0, "B": 1, "C": 2}, [], np.zeros((3, 2)), []

    def fake_physics(*args, **kwargs):
        return [], [], [], np.asarray([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]])

    def fake_frame(points, *_args):
        return np.asarray(points, dtype=float)

    def fake_procrustes(points, *_args, **_kwargs):
        captured["procrustes"] += 1
        return np.asarray(points, dtype=float)

    monkeypatch.setattr(progressive, "generate_CHEN_initial_positions", fake_generate)
    monkeypatch.setattr(progressive, "main_physics_simulation", fake_physics)
    monkeypatch.setattr(progressive, "place_in_anchor_frame", fake_frame)
    monkeypatch.setattr(progressive, "_rigid_procrustes", fake_procrustes)
    monkeypatch.setattr(progressive, "uploading_directional_data", lambda: [])

    calibration = ["A", "B", "C"]
    lonlat = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    for variant, spec in progressive.PHYSICS_VARIANTS.items():
        progressive._run_physics(spec, 0, calibration, lonlat, "A", [600.0, 250.0], 1.0, -0.5)

    assert captured["fixed"] == [[], [], calibration, calibration]
    assert captured["procrustes"] == 1


def test_progressive_baseline_alignment_contracts(monkeypatch):
    captured = {"procrustes": 0}

    monkeypatch.setattr(progressive, "stress_majorization", lambda *_args: (np.zeros((3, 2)), [], []))
    monkeypatch.setattr(progressive, "alignment_and_scaling", lambda points, *_args, **_kwargs: np.asarray(points, dtype=float))
    monkeypatch.setattr(progressive, "_rigid_procrustes", lambda points, *_args, **_kwargs: captured.__setitem__("procrustes", captured["procrustes"] + 1) or np.asarray(points, dtype=float))
    progressive._run_smacof(0, [], ["A", "B", "C"], {"A": 0, "B": 1, "C": 2}, [], ["A", "B", "C"], [(1, 1)] * 3, "A", [600, 250])
    assert captured["procrustes"] == 1


def test_progressive_dc_smacof_uses_anchor_frame_without_procrustes(monkeypatch):
    monkeypatch.setattr(progressive, "run_directed_MDS", lambda **_kwargs: [np.asarray([[1.0, 2.0], [3.0, 4.0]])])
    monkeypatch.setattr(progressive, "alignment_and_scaling", lambda points, *_args, **_kwargs: np.asarray(points, dtype=float))
    monkeypatch.setattr(
        progressive,
        "_rigid_procrustes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("DC-SMACOF must not use Procrustes")),
    )

    points = progressive._run_dc_smacof(
        0, ["A", "B"], {"A": 0, "B": 1}, "A", [600.0, 250.0], {"w_weight": 1.0, "v_weight": 0.1}
    )

    assert points.shape == (2, 2)


def test_progressive_summary_includes_requested_quantiles():
    rows = []
    for seed in (0, 1):
        row = {"variant": "PhysicsSim-Full", "seed": seed, "status": "ok"}
        row.update({metric: float(seed + 1) for metric in progressive.METRICS})
        rows.append(row)

    summary = progressive._summary(pd.DataFrame(rows))

    assert {"q05", "q25", "q75", "q95"}.issubset(summary.columns)
    assert len(summary) == len(progressive.METRICS)


def test_progressive_preflight_rejects_mismatched_lcc_parameters():
    with pytest.raises(ValueError, match="different LCC parameter"):
        progressive._assert_same_lcc(
            "test HPO",
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        )
