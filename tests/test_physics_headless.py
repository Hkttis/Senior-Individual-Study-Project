import pytest

from library.metrics import raw_distance_stress_from_sim_data
from library.physics import main_physics_simulation


def test_raw_distance_stress_from_sim_data_uses_sim_distances_against_km_positions():
    data_sim = [["A", "B", "10.0"]]
    dni = {"A": 0, "B": 1}
    pos_km = [[0.0, 0.0], [41.5, 0.0]]

    assert raw_distance_stress_from_sim_data(data_sim, dni, pos_km) == pytest.approx(0.0)


def test_physics_plot_false_does_not_open_display(monkeypatch):
    import library.physics as physics

    def fail_set_mode(*_args, **_kwargs):
        raise AssertionError("pygame display should not be opened when plot=False")

    monkeypatch.setattr(physics.pygame.display, "set_mode", fail_set_mode)
    monkeypatch.setattr(physics, "stop_physim_iteration_time", 0)

    wrong, stress_history, pos_history, pos_final = main_physics_simulation(
        vertice=["A"],
        dni={"A": 0},
        data=[],
        pos_matrix=[[0.0, 0.0]],
        directional_data=[],
        fixed_positions_list=[],
        spring_stiffness=0.0,
        repulsion_strength=0.0,
        directional_force_magnitude=0.0,
        plot=False,
    )

    assert wrong == []
    assert len(stress_history) == 1
    assert len(pos_history) == 1
    assert len(pos_final) == 1
