import numpy as np
import pytest

from library.progressive_alignment import (
    anchor_geometry_is_non_degenerate,
    anchored_similarity_procrustes,
    place_in_anchor_frame,
    sample_non_degenerate_unit_square_layout,
)


LABELS = ["A", "B", "C", "D"]
DNI = {label: index for index, label in enumerate(LABELS)}
REFER = [600.0, 250.0]


def test_anchor_frame_places_only_the_explicit_anchor_at_reference_position():
    points = np.asarray([[3.0, 4.0], [8.0, 4.0], [3.0, 9.0], [5.0, 6.0]])

    aligned = place_in_anchor_frame(points, DNI, "A", REFER)

    assert aligned[DNI["A"]] == pytest.approx(REFER)
    assert aligned[DNI["B"]] - aligned[DNI["A"]] == pytest.approx([5.0, 0.0])


def test_anchored_rigid_procrustes_keeps_anchor_and_preserves_distances():
    source = np.asarray([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0], [3.0, 2.0]])
    target = {"A": REFER, "B": [600.0, 252.0], "C": [599.0, 250.0]}

    aligned = anchored_similarity_procrustes(source, DNI, ["A", "B", "C"], target, "A", REFER, allow_scaling=False)

    assert aligned[DNI["A"]] == pytest.approx(REFER)
    assert np.linalg.norm(aligned[DNI["D"]] - aligned[DNI["A"]]) == pytest.approx(
        np.linalg.norm(source[DNI["D"]] - source[DNI["A"]])
    )


def test_anchored_similarity_procrustes_recovers_uniform_scale():
    source = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [2.0, 1.0]])
    target = {"A": REFER, "B": [600.0, 253.0], "C": [594.0, 250.0]}

    aligned = anchored_similarity_procrustes(source, DNI, ["A", "B", "C"], target, "A", REFER, allow_scaling=True)

    assert aligned[DNI["A"]] == pytest.approx(REFER)
    assert aligned[DNI["B"]] == pytest.approx(target["B"])
    assert aligned[DNI["C"]] == pytest.approx(target["C"])


def test_random_anchor_rejection_and_sampling_contract():
    degenerate = np.asarray([[0.0, 0.0], [0.01, 0.0], [0.0, 0.01], [0.8, 0.8]])
    assert not anchor_geometry_is_non_degenerate(
        degenerate, DNI, ["A", "B", "C"], min_distance=0.05, min_triangle_area=0.005
    )

    positions, attempts = sample_non_degenerate_unit_square_layout(
        len(LABELS), DNI, ["A", "B", "C"], np.random.default_rng(7)
    )
    assert attempts >= 1
    assert anchor_geometry_is_non_degenerate(
        positions, DNI, ["A", "B", "C"], min_distance=0.05, min_triangle_area=0.005
    )
