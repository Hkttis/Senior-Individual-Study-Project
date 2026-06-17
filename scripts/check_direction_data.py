"""Validate direction CSV compatibility with the current project code.

Usage:
  python scripts/check_direction_data.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from library.config import FILE_PATHS
from library.data_io import read_CHEN_csvfile, uploading_directional_data
from library.directions import DIR8_UNIT_SIM
from library.metrics import direction_violation_rate, mean_angular_error_violations


EXPECTED_HEADER = ["地點一", "地點二", "方位", "對應原文"]


def _failures_for_distance_rows(rows: list[list[str]]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    labels: set[str] = set()

    if not rows:
        errors.append("distance data has no rows")
        return errors, []

    for line_no, row in enumerate(rows, start=2):
        if len(row) < 3:
            errors.append(f"distance row {line_no}: expected at least 3 columns, got {len(row)}")
            continue
        a, b, distance = row[0].strip(), row[1].strip(), row[2].strip()
        if not a or not b:
            errors.append(f"distance row {line_no}: endpoint is empty")
        else:
            labels.add(a)
            labels.add(b)
        try:
            value = float(distance)
            if value <= 0:
                errors.append(f"distance row {line_no}: distance must be positive, got {distance!r}")
        except ValueError:
            errors.append(f"distance row {line_no}: distance is not numeric: {distance!r}")

    return errors, sorted(labels)


def _load_direction_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        return next(reader, [])


def _failures_for_direction_rows(rows: list[list[str]], labels: set[str]) -> tuple[list[str], list[list[str]]]:
    errors: list[str] = []
    valid_rows: list[list[str]] = []
    known_directions = set(DIR8_UNIT_SIM)

    for line_no, row in enumerate(rows, start=2):
        if len(row) < 3:
            errors.append(f"direction row {line_no}: expected at least 3 columns, got {len(row)}")
            continue

        a, b, direction = row[0].strip(), row[1].strip(), row[2].strip()
        if not a or not b or not direction:
            errors.append(f"direction row {line_no}: endpoint or direction is empty")
            continue
        if a not in labels:
            errors.append(f"direction row {line_no}: unknown source node {a!r}")
        if b not in labels:
            errors.append(f"direction row {line_no}: unknown target node {b!r}")
        if direction not in known_directions:
            errors.append(f"direction row {line_no}: direction {direction!r} is not recognized by library.directions")
        else:
            valid_rows.append([a, b, direction])

    return errors, valid_rows


def main() -> int:
    direction_path = Path(FILE_PATHS["directional_data"])
    distance_path = Path(FILE_PATHS["chen_data"])
    errors: list[str] = []

    if not direction_path.exists():
        errors.append(f"direction file not found: {direction_path}")
    if not distance_path.exists():
        errors.append(f"distance file not found: {distance_path}")
    if errors:
        for error in errors:
            print(f"[ERROR] {error}")
        return 1

    header = _load_direction_header(direction_path)
    if header[:4] != EXPECTED_HEADER:
        errors.append(f"direction header mismatch: expected {EXPECTED_HEADER}, got {header[:4]}")

    distance_rows = read_CHEN_csvfile()
    distance_errors, labels = _failures_for_distance_rows(distance_rows)
    errors.extend(distance_errors)

    direction_rows = uploading_directional_data()
    if len(direction_rows) <= 1:
        errors.append("direction data has no data rows")

    direction_errors, valid_rows = _failures_for_direction_rows(direction_rows, set(labels))
    errors.extend(direction_errors)

    if valid_rows:
        dni = {name: i for i, name in enumerate(labels)}
        theta = np.linspace(0.0, 2.0 * np.pi, len(labels), endpoint=False)
        pos = np.column_stack((np.cos(theta), np.sin(theta)))
        vr = direction_violation_rate(pos, valid_rows, dni)
        mae = mean_angular_error_violations(pos, valid_rows, dni)
        if not (0.0 <= vr <= 1.0):
            errors.append(f"direction_violation_rate returned out-of-range value: {vr}")
        if mae < 0.0:
            errors.append(f"mean_angular_error_violations returned negative value: {mae}")

    if errors:
        print("[FAIL] direction data check failed")
        for error in errors:
            print(f"[ERROR] {error}")
        print(f"[INFO] direction file: {direction_path}")
        print(f"[INFO] distance file: {distance_path}")
        print(f"[INFO] recognized direction tags: {sorted(DIR8_UNIT_SIM)}")
        return 1

    print("[OK] direction data check passed")
    print(f"[INFO] direction rows: {len(direction_rows)}")
    print(f"[INFO] distance rows: {len(distance_rows)}")
    print(f"[INFO] unique nodes: {len(labels)}")
    print(f"[INFO] direction file: {direction_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
