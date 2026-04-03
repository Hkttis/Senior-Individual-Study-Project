"""Run unit tests for direction error metrics.

Usage (from physics_simulation/ directory):
  python -m scripts.run_direction_metrics_tests

Optional sanity check with a real directional CSV:
  python -m scripts.run_direction_metrics_tests --csv /path/to/方向.csv
"""

from __future__ import annotations

import argparse
import csv
import unittest
from pathlib import Path

import numpy as np


def _load_directional_csv(csv_path: Path):
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        return [row for row in csv.reader(f) if row]


def _sanity_check(csv_path: Path) -> int:
    """Lightweight runtime sanity check (does not assert correctness).

    It verifies:
      - CSV can be parsed
      - Metrics functions run without errors
      - Outputs are within basic expected ranges
    """
    from library.metrics import direction_violation_rate, mean_angular_error_violations

    directional_data = _load_directional_csv(csv_path)

    labels = set()
    for row in directional_data:
        if len(row) >= 2:
            labels.add(row[0].strip())
            labels.add(row[1].strip())
    labels = [x for x in labels if x]
    labels.sort()
    dni = {name: i for i, name in enumerate(labels)}

    n = len(labels)
    if n == 0:
        print(f"[SANITY] No labels found in CSV: {csv_path}")
        return 1

    # Deterministic, non-degenerate positions on a circle (y-up)
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    pos = np.stack([np.cos(t), np.sin(t)], axis=1).astype(float)

    vr = direction_violation_rate(pos, directional_data, dni)
    mae = mean_angular_error_violations(pos, directional_data, dni)

    ok = True
    if not (0.0 <= vr <= 1.0):
        ok = False
    if mae < 0.0:
        ok = False

    print(f"[SANITY] CSV: {csv_path}")
    print(f"[SANITY] #labels={n}, #rows={len(directional_data)}")
    print(f"[SANITY] VR={vr:.6f}, MAE_theta(violations)={mae:.6f}")
    print(f"[SANITY] Result: {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional path to directional CSV for a runtime sanity check.",
    )
    args = parser.parse_args(argv)

    # Run unit tests (unittest discovery)
    project_root = Path(__file__).resolve().parents[1]
    test_dir = project_root / "tests"
    if not test_dir.exists():
        print(f"[ERROR] tests/ not found under: {project_root}")
        return 2

    loader = unittest.TestLoader()
    suite = loader.discover(
        start_dir=str(test_dir),
        pattern="test_*.py",
        top_level_dir=str(project_root),
    )
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    code = 0 if result.wasSuccessful() else 1

    # Optional runtime sanity check
    if args.csv:
        csv_path = Path(args.csv).expanduser().resolve()
        if not csv_path.exists():
            print(f"[ERROR] --csv not found: {csv_path}")
            return 2
        code = max(code, _sanity_check(csv_path))

    return code


if __name__ == "__main__":
    raise SystemExit(main())
