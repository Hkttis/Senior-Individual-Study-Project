from copy import deepcopy
from MDS_model.data_pre_processing import *
from MDS_model.directed_mds_model import *
from MDS_model.plot_node_link_diagram import *
from MDS_model.stress_majorization_mds_model import *
from library.visualization import *
from library.data_io import *

def run_directed_MDS():
    datanum = ["C:\\Users\\justi\Desktop\\project\\csv doc utf8\\GPT-4_史記_numerals_utf8.csv",
    "C:\\Users\\justi\Desktop\\project\\csv doc utf8\\GPT-4_漢書_numerals_utf8.csv",
    "C:\\Users\\justi\Desktop\\project\\csv doc utf8\\GPT-4_後漢書_numerals_utf8.csv"]
    pre_data = read_csvfile(datanum)
    c_data,disset = data_process(pre_data)
    graph,vertice,dni,edges,data= Chen_csv_and_graph()
    pos_matrix, stress_history, pos_history = directed_MDS(c_data,data,graph,vertice,dni,edges)
    
    plot_stress_convergence_log(stress_history, file_name = "DirectedMDS_")
    draw_node_link_pygame(pos_matrix, vertice, edges)
    animate_node_link_pygame( pos_history, vertice, edges)
    
    wrong_directions_list = wrong_directions_nonflip(pos_matrix, vertice, dni)
    # Turn Li to pixel units
    # directed_MDS should not apply procrustes analysis 
    pos_matrix = alignment_and_scaling(pos_matrix, vertice, dni, refer_pos=[600,375])
    visualize_error_map_official(deepcopy(pos_matrix), vertice, dni, data, wrong_directions_list, file_name = "DirectedMDS_")
    ground_truth_comparison(vertice,dni,data, uploading_ground_truth(vertice,dni),[600,375], deepcopy(pos_matrix), file_name = "DirectedMDS_")

def run_stress_majorization():
    graph,vertice,dni,edges,data= Chen_csv_and_graph()
    pos_matrix, stress_history, pos_history = stress_majorization(graph,dni,vertice,edges)
    
    plot_stress_convergence_log(stress_history, file_name = "StressMj_")
    draw_node_link_pygame(pos_matrix, vertice, edges)
    animate_node_link_pygame( pos_history, vertice, edges)
    
    wrong_directions_list = wrong_directions_nonflip(pos_matrix, vertice, dni)
    # Turn Li to pixel units
    pos_matrix = scaling_and_procrustes_analysis(deepcopy(pos_matrix), vertice, dni, refer_pos=[600,375])
    visualize_error_map_official(deepcopy(pos_matrix), vertice, dni, data, wrong_directions_list, file_name = "StressMj_")
    ground_truth_comparison(vertice,dni,data, uploading_ground_truth(vertice,dni),[600,375], deepcopy(pos_matrix), file_name = "StressMj_")

if __name__ == "__main__" :
    # run_directed_MDS()
    run_stress_majorization()