import math
import unittest

import numpy as np

from library.config import theta_thr_4dir, theta_thr_8dir
from library.metrics import direction_violation_rate, mean_angular_error_violations


class TestDirectionMetrics(unittest.TestCase):
    def setUp(self):
        # Minimal dni mapping for synthetic tests
        self.dni = {"A": 0, "B": 1, "C": 2}

        # Two valid constraints + three invalid rows (should be ignored)
        self.directional_data = [
            ["A", "B", "東"],
            ["A", "C", "西北"],
            ["A", "X", "西"],        # v not in dni -> ignored
            ["A", "B"],              # malformed -> ignored
            ["A", "B", "不存在"],    # unknown direction -> ignored
        ]

    def test_all_satisfied(self):
        # A->B is exactly East; A->C is exactly Northwest.
        pos = np.array(
            [
                [0.0, 0.0],   # A
                [1.0, 0.0],   # B (east)
                [-1.0, 1.0],  # C (northwest)
            ],
            dtype=float,
        )
        vr = direction_violation_rate(pos, self.directional_data, self.dni)
        mae = mean_angular_error_violations(pos, self.directional_data, self.dni)
        self.assertAlmostEqual(vr, 0.0, places=12)
        self.assertAlmostEqual(mae, 0.0, places=12)

    def test_all_violated_expected_values(self):
        # A->B is opposite to East => phi = pi
        # A->C uses direction "西北" but vector points East => phi = 3pi/4
        pos = np.array(
            [
                [0.0, 0.0],  # A
                [-1.0, 0.0], # B (west)
                [1.0, 0.0],  # C (east)
            ],
            dtype=float,
        )
        vr = direction_violation_rate(pos, self.directional_data, self.dni)
        mae = mean_angular_error_violations(pos, self.directional_data, self.dni)

        # Only the first two rows are valid => |E_theta| = 2
        self.assertAlmostEqual(vr, 1.0, places=12)

        delta_ab = max(0.0, math.pi - float(theta_thr_4dir))
        delta_ac = max(0.0, (3.0 * math.pi / 4.0) - float(theta_thr_8dir))
        expected_mae = 0.5 * (delta_ab + delta_ac)
        self.assertAlmostEqual(mae, expected_mae, places=12)

    def test_mixed_violations(self):
        # A->B satisfied (east). A->C violated (east but expects northwest).
        pos = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],  # B east (satisfy)
                [1.0, 0.0],  # C east (violate for "西北")
            ],
            dtype=float,
        )
        vr = direction_violation_rate(pos, self.directional_data, self.dni)
        mae = mean_angular_error_violations(pos, self.directional_data, self.dni)

        self.assertAlmostEqual(vr, 0.5, places=12)
        expected_delta = max(0.0, (3.0 * math.pi / 4.0) - float(theta_thr_8dir))
        self.assertAlmostEqual(mae, expected_delta, places=12)

    def test_zero_distance_edges_are_ignored(self):
        # A and B coincide => that edge is ignored; only A->C counts.
        pos = np.array(
            [
                [0.0, 0.0],  # A
                [0.0, 0.0],  # B (same as A)
                [-1.0, 1.0], # C (northwest)
            ],
            dtype=float,
        )
        vr = direction_violation_rate(pos, self.directional_data, self.dni)
        mae = mean_angular_error_violations(pos, self.directional_data, self.dni)
        self.assertAlmostEqual(vr, 0.0, places=12)
        self.assertAlmostEqual(mae, 0.0, places=12)

    def test_no_valid_edges_returns_zero(self):
        pos = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
        dni = {"A": 0, "B": 1}
        directional_data = [["A", "B", "不存在"], ["A", "B"]]
        vr = direction_violation_rate(pos, directional_data, dni)
        mae = mean_angular_error_violations(pos, directional_data, dni)
        self.assertAlmostEqual(vr, 0.0, places=12)
        self.assertAlmostEqual(mae, 0.0, places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
