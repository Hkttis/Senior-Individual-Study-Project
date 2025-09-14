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
    pos_matrix = directed_MDS(c_data,data,graph,vertice,dni,edges)
    draw_node_link_pygame(pos_matrix, vertice, edges)
    
    visualize_error_map_official(deepcopy(pos_matrix), vertice, dni, data)