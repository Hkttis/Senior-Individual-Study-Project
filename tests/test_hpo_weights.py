import pytest

from run_paper_script.ch5_hparam_kfold_gridsearch_pareto import _weights_from_alpha_beta


def test_weights_alpha_beta_zero_use_base_values():
    w_dir, w_reg, spring, directional, repulsion = _weights_from_alpha_beta(
        alpha=0.0,
        beta=0.0,
        w_dis=1.0,
        base_spring_stiffness=1500.0,
        base_directional_force=10000.0,
        base_repulsion_strength=500.0,
    )

    assert w_dir == pytest.approx(1.0)
    assert w_reg == pytest.approx(1.0)
    assert spring == pytest.approx(1500.0)
    assert directional == pytest.approx(10000.0)
    assert repulsion == pytest.approx(500.0)


def test_weights_alpha_beta_scale_base10():
    w_dir, w_reg, spring, directional, repulsion = _weights_from_alpha_beta(
        alpha=1.0,
        beta=-1.0,
        w_dis=1.0,
        base_spring_stiffness=1500.0,
        base_directional_force=10000.0,
        base_repulsion_strength=500.0,
    )

    assert w_dir == pytest.approx(10.0)
    assert w_reg == pytest.approx(0.1)
    assert spring == pytest.approx(1500.0)
    assert directional == pytest.approx(100000.0)
    assert repulsion == pytest.approx(50.0)
