"""Validate site ground-truth CSV for anchor LOO HPO.

Usage:
  python -m scripts.check_site_points
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from library.config import FILE_PATHS
from library.data_io import (
    get_anchor_labels,
    get_test_site_labels,
    read_CHEN_csvfile,
    uploading_ground_truth,
)


EXPECTED_REQUIRED_COLUMNS = {"lon", "lat", "use_role"}
EXPECTED_ROLES = {"anchor", "test"}


def _distance_labels() -> list[str]:
    labels: set[str] = set()
    for row in read_CHEN_csvfile():
        if len(row) >= 2:
            source, target = row[0].strip(), row[1].strip()
            if source:
                labels.add(source)
            if target:
                labels.add(target)
    return sorted(labels)


def _row_name(row: dict[str, str]) -> str:
    return (row.get("model_name") or row.get("節點名稱") or "").strip()


def main() -> int:
    site_path = Path(FILE_PATHS["ground_truth_path"])
    errors: list[str] = []

    if not site_path.exists():
        print(f"[ERROR] site points file not found: {site_path}")
        return 1

    labels = _distance_labels()
    dni = {name: i for i, name in enumerate(labels)}
    seen: set[str] = set()
    rows: list[dict[str, str]] = []

    with site_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or [])
        if not (("model_name" in fieldnames) or ("節點名稱" in fieldnames)):
            errors.append(f"header mismatch: expected model_name or 節點名稱 got {reader.fieldnames}")
        missing_columns = EXPECTED_REQUIRED_COLUMNS.difference(fieldnames)
        if missing_columns:
            errors.append(f"header missing required columns: {sorted(missing_columns)}")

        for line_no, row in enumerate(reader, start=2):
            name = _row_name(row)
            lon_raw = (row.get("lon") or "").strip()
            lat_raw = (row.get("lat") or "").strip()
            use_role = (row.get("use_role") or "").strip()

            if not name:
                errors.append(f"row {line_no}: model_name is empty")
                continue
            if name in seen:
                errors.append(f"row {line_no}: duplicate model_name {name!r}")
            seen.add(name)
            if name not in dni:
                errors.append(f"row {line_no}: model_name not found in distance data: {name!r}")

            try:
                lon = float(lon_raw)
                lat = float(lat_raw)
            except ValueError:
                errors.append(f"row {line_no}: lon/lat must be numeric, got lon={lon_raw!r}, lat={lat_raw!r}")
                continue

            if not (-180.0 <= lon <= 180.0):
                errors.append(f"row {line_no}: lon out of range: {lon}")
            if not (-90.0 <= lat <= 90.0):
                errors.append(f"row {line_no}: lat out of range: {lat}")
            if use_role not in EXPECTED_ROLES:
                errors.append(f"row {line_no}: unexpected use_role {use_role!r}")

            rows.append({"model_name": name, "lon": lon_raw, "lat": lat_raw, "use_role": use_role})

    anchors = get_anchor_labels()
    test_sites = get_test_site_labels()
    if len(anchors) != 3:
        errors.append(f"expected exactly 3 use_role=anchor rows for LOO HPO, got {len(anchors)}: {anchors}")
    if len(test_sites) != 8:
        errors.append(f"expected exactly 8 use_role=test rows for final RMSE, got {len(test_sites)}: {test_sites}")

    if dni:
        positions = uploading_ground_truth(labels, dni)
        if len(positions) != len(dni):
            errors.append(f"uploading_ground_truth returned {len(positions)} positions for {len(dni)} nodes")
        for site in anchors + test_sites:
            if site in dni and positions[dni[site]] == [0, 0]:
                errors.append(f"uploading_ground_truth did not populate required site: {site!r}")

    unknown_names = sorted(seen.difference(dni))
    matched_names = sorted(seen.intersection(dni))

    if errors:
        print("[FAIL] site points check failed")
        for error in errors:
            print(f"[ERROR] {error}")
        print(f"[INFO] site points file: {site_path}")
        print(f"[INFO] rows read: {len(rows)}")
        print(f"[INFO] names matching distance data: {len(matched_names)}")
        print(f"[INFO] names not in distance data: {len(unknown_names)}")
        return 1

    print("[OK] site points check passed")
    print(f"[INFO] rows read: {len(rows)}")
    print(f"[INFO] anchors: {anchors}")
    print(f"[INFO] test sites: {test_sites}")
    print(f"[INFO] names matching distance data: {len(matched_names)}")
    print(f"[INFO] names not in distance data: {len(unknown_names)}")
    print(f"[INFO] site points file: {site_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
