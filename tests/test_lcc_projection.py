import csv

from library.config import FILE_PATHS
from library.geometry import get_lcc_bounds, get_lcc_parameters


def test_lcc_standard_parallels_use_one_sixth_rule():
    lon_min, lon_max, lat_min, lat_max = get_lcc_bounds()
    lat_range = lat_max - lat_min
    lat1, lat2, lon0 = get_lcc_parameters()

    assert lat1 == lat_min + lat_range / 6
    assert lat2 == lat_max - lat_range / 6
    assert lon0 == (lon_min + lon_max) / 2
    assert lat_min < lat1 < lat2 < lat_max


def test_lcc_bounds_come_from_site_points_file():
    with open(FILE_PATHS["ground_truth_path"], newline="", encoding="utf-8-sig") as csvfile:
        rows = list(csv.DictReader(csvfile))

    expected_lons = [float(row["lon"]) for row in rows if row["lon"] and row["lat"]]
    expected_lats = [float(row["lat"]) for row in rows if row["lon"] and row["lat"]]

    assert get_lcc_bounds() == (
        min(expected_lons),
        max(expected_lons),
        min(expected_lats),
        max(expected_lats),
    )
