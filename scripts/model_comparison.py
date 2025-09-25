from copy import deepcopy
from MDS_model.data_pre_processing import *
from MDS_model.directed_mds_model import *
from MDS_model.plot_node_link_diagram import *
from MDS_model.stress_majorization_mds_model import *
from library.visualization import *
from library.initialization import *
from library.physics import *
from library.data_io import *

def run_directed_MDS( vis = True ):
    datanum = ["C:\\Users\\justi\Desktop\\project\\csv doc utf8\\GPT-4_史記_numerals_utf8.csv",
    "C:\\Users\\justi\Desktop\\project\\csv doc utf8\\GPT-4_漢書_numerals_utf8.csv",
    "C:\\Users\\justi\Desktop\\project\\csv doc utf8\\GPT-4_後漢書_numerals_utf8.csv"]
    pre_data = read_csvfile(datanum)
    c_data,disset = data_process(pre_data)
    graph,vertice,dni,edges,data= Chen_csv_and_graph()
    pos_matrix, stress_history, pos_history = directed_MDS(c_data,data,graph,vertice,dni,edges)
    
    if vis == True :
        plot_stress_convergence_log(stress_history, file_name = "DirectedMDS_")
        
        dir_data = uploading_directional_data()
        directional_data = [(row[0], row[1], row[2]) for row in dir_data]
        draw_node_link_pygame_dirarrow(
            pos=pos_matrix,                 # directed_MDS 或 stress majorization 的座標輸出
            vertice=vertice,
            edges=edges,
            directed=False,             # 若只想亮出羅盤箭頭，不把所有邊都變成有向
            directional_data=directional_data,  # ★ 關鍵：把 sel_data 丟進來
            dir_arrow_color=(200, 0, 0),
            dir_arrow_len=28,
            caption="Node-Link with Compass Arrows"
        )
            
        # draw_node_link_pygame(pos_matrix, vertice, edges)
        animate_node_link_pygame( pos_history, vertice, edges)
        
        wrong_directions_list = wrong_directions_nonflip(pos_matrix, vertice, dni)
        # Turn Li to pixel units
        # directed_MDS should not apply procrustes analysis 
        pos_matrix = alignment_and_scaling(pos_matrix, vertice, dni, refer_pos=[600,500])
        visualize_error_map_official(deepcopy(pos_matrix), vertice, dni, data, wrong_directions_list, zoom_area = None , file_name = "DirectedMDS_")
        #visualize_error_map_official(deepcopy(pos_matrix), vertice, dni, data, wrong_directions_list, zoom_area = (200, 200, 800, 400) , file_name = "DirectedMDS_")
        ground_truth_comparison(vertice,dni,data, uploading_ground_truth(vertice,dni),pos_matrix[dni["鄯善"]], deepcopy(pos_matrix), file_name = "DirectedMDS_")
        
    return pos_history

def run_stress_majorization( vis = True ):
    graph,vertice,dni,edges,data= Chen_csv_and_graph()
    pos_matrix, stress_history, pos_history = stress_majorization(graph,dni,vertice,edges)
    
    if vis == True :
        plot_stress_convergence_log(stress_history, file_name = "StressMj_")
        draw_node_link_pygame(pos_matrix, vertice, edges)
        animate_node_link_pygame( pos_history, vertice, edges)
        
        wrong_directions_list = wrong_directions_nonflip(pos_matrix, vertice, dni)
        # Turn Li to pixel units
        pos_matrix = alignment_and_scaling(pos_matrix, vertice, dni, refer_pos=[600,500])
        
        # temporalily use ground truth to simulate given fixed points' positions
        fixed_point_labels = ["鄯善","都護治/烏壘"]
        gt = uploading_ground_truth(vertice,dni)
        fixed_points_lonlat = [ tuple(gt[dni[cout]]) for cout in fixed_point_labels]
        
        
        #pos_matrix = procrustes_analysis_to_gt(deepcopy(pos_matrix), vertice, dni, refer_pos=[600,500])
        #ground_truth_comparison(vertice,dni,data, uploading_ground_truth(vertice,dni),pos_matrix[dni["鄯善"]], deepcopy(pos_matrix), file_name = "StressMj_")
        pos_matrix = procrustes_align_by_fixed_points(deepcopy(pos_matrix),fixed_point_labels = fixed_point_labels, fixed_points_lonlat = fixed_points_lonlat, dni = dni)
        #visualize_error_map_official(deepcopy(pos_matrix), vertice, dni, data, wrong_directions_list, file_name = "StressMj_")
        ground_truth_comparison(vertice,dni,data, uploading_ground_truth(vertice,dni),pos_matrix[dni["鄯善"]], deepcopy(pos_matrix), file_name = "StressMj_")
    
    return pos_history

def run_physics_simulation_model( vis = True) :
    refer_pos = [600,500]
    vertice,dni,data,pos_matrix,fixed_positions_list = generate_CHEN_initial_positions(deepcopy(refer_pos))
    
    directional_data = uploading_directional_data()
    wrong_direction_lists,stress_history,pos_history,pos_matrix = main_physics_simulation(vertice,dni,data,pos_matrix,directional_data,fixed_positions_list,SPRING_STIFFNESS_BASE,REPULSION_STRENGTH_BASE,DIRECTIONAL_FORCE_MAGNITUDE_BASE, plot = vis)
    
    if vis == True :
        # Visualuzation and Evaluation
        plot_stress_convergence_log(stress_history, file_name = "PhysicsSim_")
        errors, edge_labels = visualize_error_map_official(deepcopy(pos_matrix), vertice, dni, data, wrong_direction_lists, zoom_area=None, file_name = "PhysicsSim_")
        visualize_error_map_official(deepcopy(pos_matrix), vertice, dni, data, wrong_direction_lists, zoom_area=(500, 325, 800, 400), file_name = "PhysicsSim_")
        ground_truth_positions = uploading_ground_truth(vertice,dni)
        ground_truth_comparison(vertice,dni,data,deepcopy(ground_truth_positions),deepcopy(refer_pos), deepcopy(pos_matrix), file_name = "PhysicsSim_")
        
        save_vis_data(vertice, dni, deepcopy(pos_matrix), deepcopy(ground_truth_positions), deepcopy(refer_pos))
        save_err_data(vertice, dni, deepcopy(pos_matrix), deepcopy(ground_truth_positions), deepcopy(refer_pos), errors, edge_labels)
    return pos_history

if __name__ == "__main__" :
    
    
    run_stress_majorization(True)
    
    '''
    
    
    graph,vertice,dni,edges,data= Chen_csv_and_graph()
    pos_history_directed_mds = run_directed_MDS(vis = False)
    pos_history_stress_mj = run_stress_majorization(vis = False)
    pos_history_physics_sim = run_physics_simulation_model(vis = False)
    
    
    plot_three_model_convergence_pygame_pixelaware( pos_history_physics_sim, pos_history_directed_mds, pos_history_stress_mj,
        vertice = vertice, dni = dni, data = data, ground_truth_positions=uploading_ground_truth(vertice, dni), refer_pos=(600, 500), bin_size_iters=10 )
    '''