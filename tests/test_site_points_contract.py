from library.config import FILE_PATHS
from library.data_io import (
    get_anchor_align_label,
    get_anchor_labels,
    get_default_frame_anchor_label,
    get_test_site_labels,
    read_CHEN_csvfile,
    uploading_ground_truth,
)


def test_site_points_have_expected_anchor_and_test_counts():
    assert get_anchor_labels() == ["鄯善", "車師前", "都護治/烏壘"]
    assert len(get_test_site_labels()) == 8


def test_site_points_exist_in_distance_data_and_upload_ground_truth():
    labels = sorted({name for row in read_CHEN_csvfile() for name in row[:2] if name})
    dni = {name: i for i, name in enumerate(labels)}
    required_sites = get_anchor_labels() + get_test_site_labels()

    assert set(required_sites).issubset(dni)

    positions = uploading_ground_truth(labels, dni)
    assert len(positions) == len(dni)
    for site in required_sites:
        assert positions[dni[site]] != [0, 0]


def test_default_frame_anchor_falls_back_to_first_anchor():
    anchors = get_anchor_labels()

    assert get_default_frame_anchor_label() == anchors[0]
    assert get_anchor_align_label() == get_default_frame_anchor_label()


def test_ground_truth_path_uses_project_data_file():
    assert FILE_PATHS["ground_truth_path"].endswith("data\\site_rmse_points.csv") or FILE_PATHS[
        "ground_truth_path"
    ].endswith("data/site_rmse_points.csv")
