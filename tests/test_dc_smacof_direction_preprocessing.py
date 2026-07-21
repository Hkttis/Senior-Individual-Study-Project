import numpy as np
import pytest

from library.config import FILE_PATHS
from library.data_io import load_ini_data_from_csv, uploading_directional_data
from library.directions import DIR8_UNIT_SIM
from library.model_cmp import (
    _directional_data_to_dc_smacof_c_data,
    _merge_dc_smacof_direction_observations,
    get_dc_smacof_direction_method_metadata,
    run_directed_MDS,
)


def _project_direction_context():
    _, vertices, dni, _, _ = load_ini_data_from_csv(FILE_PATHS)
    return vertices, dni, uploading_directional_data()


def test_project_direction_data_merges_only_the_repeated_unordered_pair():
    vertices, dni, raw_rows = _project_direction_context()
    merged_rows, provenance = _merge_dc_smacof_direction_observations(raw_rows, dni)

    assert len(raw_rows) == 44
    assert len(merged_rows) == 43
    assert len(provenance) == 43

    repeated = [row for row in provenance if row["n_observations"] > 1]
    assert repeated == [
        {
            "effective_source": "莎車",
            "effective_target": "疏勒",
            "effective_direction": "西北",
            "n_observations": 2,
            "original_observations": [
                ("莎車", "疏勒", "西"),
                ("疏勒", "莎車", "南"),
            ],
        }
    ]

    known_nodes = set(vertices)
    unordered_pairs = set()
    for source, target, direction in merged_rows:
        assert source in known_nodes
        assert target in known_nodes
        assert source != target
        assert direction in DIR8_UNIT_SIM
        assert all(np.isfinite(DIR8_UNIT_SIM[direction]))

        pair = frozenset((source, target))
        assert pair not in unordered_pairs
        unordered_pairs.add(pair)


def test_consensus_is_independent_of_input_row_order():
    dni = {"A": 0, "B": 1}
    rows = [["A", "B", "西"], ["B", "A", "南"]]

    forward, _ = _merge_dc_smacof_direction_observations(rows, dni)
    reversed_rows, _ = _merge_dc_smacof_direction_observations(list(reversed(rows)), dni)

    assert forward == [["A", "B", "西北"]]
    assert reversed_rows == forward


def test_reversed_observation_is_converted_to_the_canonical_orientation():
    merged, _ = _merge_dc_smacof_direction_observations(
        [["B", "A", "南"]],
        {"A": 0, "B": 1},
    )

    assert merged == [["A", "B", "北"]]


def test_cancelled_direction_observations_raise_value_error():
    with pytest.raises(ValueError, match="cancel out"):
        _merge_dc_smacof_direction_observations(
            [["A", "B", "東"], ["B", "A", "東"]],
            {"A": 0, "B": 1},
        )


def test_non_dir8_consensus_raises_value_error_instead_of_rounding():
    with pytest.raises(ValueError, match="does not map exactly to DIR8"):
        _merge_dc_smacof_direction_observations(
            [["A", "B", "東"], ["A", "B", "東北"]],
            {"A": 0, "B": 1},
        )


@pytest.mark.parametrize(
    "rows, error_pattern",
    [
        ([[]], "must have source"),
        ([["", "B", "東"]], "empty source"),
        ([["A", "", "東"]], "empty target"),
        ([["A", "A", "東"]], "self-loop"),
        ([["A", "C", "東"]], "unknown node"),
        ([["A", "B", "上"]], "invalid DIR8"),
    ],
)
def test_invalid_direction_rows_raise_value_error(rows, error_pattern):
    with pytest.raises(ValueError, match=error_pattern):
        _merge_dc_smacof_direction_observations(rows, {"A": 0, "B": 1})


def test_legacy_c_data_contains_only_the_merged_constraints():
    _, dni, raw_rows = _project_direction_context()
    c_data = _directional_data_to_dc_smacof_c_data(raw_rows, dni)

    assert len(c_data) == 4
    assert c_data[0] == []
    assert c_data[1] == []
    assert len(c_data[2]) == 43
    assert c_data[3] == []


def test_direction_method_metadata_records_training_and_evaluation_contracts():
    _, dni, raw_rows = _project_direction_context()
    metadata, provenance = get_dc_smacof_direction_method_metadata(raw_rows, dni)

    assert metadata == {
        "direction_target_rule": "wang2017_current_pair_distance",
        "direction_preprocessing": "vector_consensus_by_undirected_pair",
        "raw_direction_observation_count": 44,
        "effective_direction_constraint_count": 43,
        "direction_evaluation_source": "raw_verified_observations",
    }
    assert len(provenance) == 43


def test_dc_smacof_runs_with_preprocessed_direction_data():
    np.random.seed(0)
    history = run_directed_MDS(vis=False)
    positions = np.asarray(history[-1], dtype=float)

    assert positions.ndim == 2
    assert positions.shape[1] == 2
    assert np.all(np.isfinite(positions))
