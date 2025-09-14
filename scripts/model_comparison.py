from copy import deepcopy
from directed_mds_model.data_pre_processing import *
from directed_mds_model.directed_mds_model import *
from directed_mds_model.plot_node_link_diagram import *
from library.visualization import *

if __name__ == "__main__" :
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
    
    pos_matrix = alignment_and_scaling(deepcopy(pos_matrix), vertice, dni, refer_pos=[600,375])
    visualize_error_map_official(deepcopy(pos_matrix), vertice, dni, data, file_name = "DirectedMDS_")
    ground_truth_comparison(vertice,dni,data, uploading_ground_truth(vertice,dni),[600,375], deepcopy(pos_matrix), file_name = "DirectedMDS_")