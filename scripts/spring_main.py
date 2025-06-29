from copy import deepcopy

from library.metrics import *
from library.config import *
from library.data_io import *
from library.geometry import *
from library.visulization import *
from library.physics import *
from library.initialization import *
  
# temporarily stop the plotting_physics_simulation function in run_phy...

def main_function(): # avoid global parameters
    refer_pos = [600,500]
    vertice,dni,data,pos_matrix,fixed_positions_list = generate_CHEN_initial_positions(deepcopy(refer_pos))
    directional_data = uploading_directional_data()
    wrong_direction_lists,stress_history,pos_matrix = main_physics_simulation(vertice,dni,data,pos_matrix,directional_data,fixed_positions_list,SPRING_STIFFNESS_BASE,REPULSION_STRENGTH_BASE,DIRECTIONAL_FORCE_MAGNITUDE_BASE)
    plot_stress_convergence_log(stress_history)
    errors, edge_labels = visualize_error_map_official(deepcopy(pos_matrix), vertice, dni, data, wrong_direction_lists, zoom_area=None)
    visualize_error_map_official(deepcopy(pos_matrix), vertice, dni, data, wrong_direction_lists, zoom_area=(500, 325, 800, 400))
    ground_truth_positions = uploading_ground_truth(vertice,dni)
    ground_truth_comparison(vertice,dni,data,deepcopy(ground_truth_positions),deepcopy(refer_pos), deepcopy(pos_matrix))
    save_vis_data(vertice, dni, deepcopy(pos_matrix), deepcopy(ground_truth_positions), deepcopy(refer_pos))
    save_err_data(vertice, dni, deepcopy(pos_matrix), deepcopy(ground_truth_positions), deepcopy(refer_pos), errors, edge_labels)

if __name__ == "__main__":
    main_function()
