import numpy as np
import pytest

from library.model_cmp import run_directed_MDS
from MDS_model.directed_mds_model import compute_DW_DV, stress


def _compute_single_direction_target(positions, text_distance, direction="北"):
    positions = np.asarray(positions, dtype=float)
    dis = np.asarray(
        [
            [0.0, float(text_distance)],
            [float(text_distance), 0.0],
        ]
    )
    _dw, dv = compute_DW_DV(
        2,
        1,
        1,
        1,
        positions,
        [["A", "B", direction]],
        [],
        ["A", "B"],
        {"A": 0, "B": 1},
        [("A", "B")],
        dis,
        [0, 0],
    )
    return dv[0]


def test_production_direction_target_uses_current_pair_distance():
    target = _compute_single_direction_target(
        positions=[[0.0, 0.0], [30.0, 40.0]],
        text_distance=100.0,
    )

    assert np.allclose(target, [0.0, -50.0])
    assert np.linalg.norm(target) == pytest.approx(50.0)


def test_direction_target_changes_when_current_layout_distance_changes():
    target_50 = _compute_single_direction_target(
        positions=[[0.0, 0.0], [30.0, 40.0]],
        text_distance=100.0,
    )
    target_10 = _compute_single_direction_target(
        positions=[[0.0, 0.0], [6.0, 8.0]],
        text_distance=100.0,
    )

    assert np.linalg.norm(target_50) == pytest.approx(50.0)
    assert np.linalg.norm(target_10) == pytest.approx(10.0)


def test_direction_target_does_not_use_text_distance():
    positions = [[0.0, 0.0], [30.0, 40.0]]

    target_with_distance = _compute_single_direction_target(positions, text_distance=100.0)
    target_without_distance = _compute_single_direction_target(positions, text_distance=0.0)

    assert np.allclose(target_with_distance, target_without_distance)
    assert np.linalg.norm(target_without_distance) == pytest.approx(50.0)


def test_zero_length_direction_pair_produces_zero_target():
    target = _compute_single_direction_target(
        positions=[[12.0, 34.0], [12.0, 34.0]],
        text_distance=100.0,
    )

    assert np.array_equal(target, np.zeros(2))


def test_zero_length_direction_pair_keeps_stress_history_finite():
    positions = np.asarray([[12.0, 34.0], [12.0, 34.0]])
    direction_stress = stress(
        2,
        0,
        0,
        1,
        positions,
        np.zeros((2, 2)),
        np.asarray([[0.0, 1.0], [1.0, 0.0]]),
        True,
        {"A": 0, "B": 1},
        [],
        [["A", "B", "北"]],
        np.zeros((2, 2)),
    )

    assert direction_stress == pytest.approx(0.0)
    assert np.isfinite(direction_stress)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_real_data_wang_direction_runs_stay_finite(seed):
    np.random.seed(seed)
    history = run_directed_MDS(vis=False)

    assert len(history) == 1002
    assert all(np.all(np.isfinite(np.asarray(frame, dtype=float))) for frame in history)
