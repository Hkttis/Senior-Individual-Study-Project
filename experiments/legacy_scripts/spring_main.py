from copy import deepcopy

from library.metrics import *
from library.config import *
from library.data_io import *
from library.geometry import *
from library.visualization import *
from library.physics import *
from library.initialization import *
from MDS_model.plot_node_link_diagram  import *
from MDS_model.data_pre_processing import *
from MDS_model.stress_majorization_mds_model import *


def main_function(): # avoid global parameters
    refer_pos_screen = [600,500]
    refer_pos = [refer_pos_screen[0], height - refer_pos_screen[1]]  # y-up SIM anchor

    fixed_point_labels = get_anchor_labels()
    graph, vertice, dni, edges, data = load_ini_data_from_csv(FILE_PATHS)
    gt = uploading_ground_truth(vertice,dni)
    fixed_points_lonlat = [ tuple(gt[dni[cout]]) for cout in fixed_point_labels]
    
    vertice,dni,data,pos_matrix,fixed_positions_list = generate_CHEN_initial_positions(deepcopy(refer_pos), fixed_point_labels, fixed_points_lonlat)
    
    directional_data = uploading_directional_data()
    # the coordinates here have pixle units and flipped y axis ( from here to bottom )
    wrong_direction_lists,stress_history,pos_history,pos_matrix = main_physics_simulation(vertice,dni,data,pos_matrix,directional_data,fixed_positions_list,SPRING_STIFFNESS_BASE,REPULSION_STRENGTH_BASE,DIRECTIONAL_FORCE_MAGNITUDE_BASE, plot = True)
    
    # Visualuzation and Evaluation
    plot_stress_convergence_log(stress_history, file_name = "PhysicsSim_")
    errors, edge_labels = visualize_error_map_official(deepcopy(pos_matrix), vertice, dni, data, wrong_direction_lists, zoom_area=None, file_name = "PhysicsSim_")
    visualize_error_map_official(deepcopy(pos_matrix), vertice, dni, data, wrong_direction_lists, zoom_area=(500, 325, 800, 400), file_name = "PhysicsSim_")
    ground_truth_positions = uploading_ground_truth(vertice,dni)
    ground_truth_comparison(vertice,dni,data,deepcopy(ground_truth_positions),deepcopy(refer_pos), deepcopy(pos_matrix), file_name = "PhysicsSim_")
    
    save_vis_data(vertice, dni, deepcopy(pos_matrix), deepcopy(ground_truth_positions), deepcopy(refer_pos))
    save_err_data(vertice, dni, deepcopy(pos_matrix), deepcopy(ground_truth_positions), deepcopy(refer_pos), errors, edge_labels)

if __name__ == "__main__":
    main_function()
