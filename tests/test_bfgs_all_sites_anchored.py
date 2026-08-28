from experiments.bfgs_all_sites_anchored.run_experiment import (
    build_all_sites_anchored_problem,
)
from library.scipy_objective import build_current_objective


def test_oracle_problem_adds_test_sites_without_mutating_formal_problem():
    formal, oracle, _vertices, dni, calibration_labels, test_labels = (
        build_all_sites_anchored_problem()
    )

    formal_fixed = set(int(index) for index in formal.anchor_indices)
    oracle_fixed = set(int(index) for index in oracle.anchor_indices)

    assert len(calibration_labels) == 3
    assert len(test_labels) == 8
    assert len(formal_fixed) == 3
    assert len(oracle_fixed) == 11
    assert formal_fixed == {dni[label] for label in calibration_labels}
    assert oracle_fixed == formal_fixed | {dni[label] for label in test_labels}
    assert not ({dni[label] for label in test_labels} & formal_fixed)

    rebuilt_formal = build_current_objective()
    assert len(rebuilt_formal.anchor_indices) == 3
    assert set(int(index) for index in rebuilt_formal.anchor_indices) == formal_fixed
