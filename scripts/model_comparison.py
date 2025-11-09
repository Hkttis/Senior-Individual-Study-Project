
from library.model_cmp import *
from library.data_io import *
from library.visualization import plot_three_model_convergence_pygame_pixelaware, plot_force_heatmap_scalar_sum
from MDS_model.plot_node_link_diagram import wrong_directions_flip



def download_and_upload_allpos_in_runs():
    
    
    graph, vertice, dni, edges, data = load_ini_data_from_csv(FILE_PATHS)
    
    fixed_point_labels = ["鄯善","都護治/烏壘"]
    gt = uploading_ground_truth(vertice,dni)
    fixed_points_lonlat = [ tuple(gt[dni[cout]]) for cout in fixed_point_labels]
    
    res = multi_measurement_benchmark(n_runs = 100, refer_pos=(600, 500), fixed_point_labels = fixed_point_labels, fixed_point_lonlat = fixed_points_lonlat)
    save_all_pos_histories_px_csv(res["all_pos_history_px"]["StressMajorization"],
                              res["all_pos_history_px"]["DirectedMDS"],
                              res["all_pos_history_px"]["PhysicsSim"],
                              vertice = vertice)
    all_pos_hist_sm_px, all_pos_hist_dm_px, all_pos_hist_ph_px = load_all_pos_histories_px_csv()
    
    def as_pylists(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (list, tuple)):
            return [as_pylists(x) for x in obj]
        if isinstance(obj, dict):
            return {k: as_pylists(v) for k, v in obj.items()}
        return obj
    
    # Use this for all three asserts:
    assert as_pylists(res["all_pos_history_px"]["PhysicsSim"]) == as_pylists(all_pos_hist_ph_px)
    assert as_pylists(res["all_pos_history_px"]["DirectedMDS"]) == as_pylists(all_pos_hist_dm_px)
    assert as_pylists(res["all_pos_history_px"]["StressMajorization"]) == as_pylists(all_pos_hist_sm_px)    
    
    return all_pos_hist_sm_px, all_pos_hist_dm_px, all_pos_hist_ph_px

if __name__ == "__main__" :
    
    graph, vertice, dni, edges, data = load_ini_data_from_csv(FILE_PATHS)
    
    # temporalily use ground truth to simulate given fixed points' positions
    fixed_point_labels = ["鄯善","都護治/烏壘"]
    gt = uploading_ground_truth(vertice,dni)
    fixed_points_lonlat = [ tuple(gt[dni[cout]]) for cout in fixed_point_labels]
    
    all_pos_hist_sm_px, all_pos_hist_dm_px, all_pos_hist_ph_px = load_all_pos_histories_px_csv()
    
    #med_pos_hist_sm_px = select_median_pos_history( all_pos_hist_sm_px , dni, gt, refer_pos = (600,500))
    #med_pos_hist_dm_px = select_median_pos_history( all_pos_hist_dm_px , dni, gt, refer_pos = (600,500))
    med_pos_hist_ph_px, worst_case_pos_hist = select_median_pos_history( all_pos_hist_ph_px , dni, gt, refer_pos = (600,500), return_worst=True)
    
    #med_pos_hist_sm_px = run_stress_majorization(vis = False)
    #med_pos_hist_dm_px = run_directed_MDS(vis = False)
    #med_pos_hist_ph_px = run_physics_simulation_model(fixed_point_labels, fixed_points_lonlat, vis = False)
    
    pos_matrix = deepcopy(med_pos_hist_ph_px[-1])
    refer_pos = (600,500)
    wrong_direction_lists = wrong_directions_flip(pos_matrix, vertice, dni)
    
    errors, edge_labels = visualize_error_map_official(deepcopy(pos_matrix), vertice, dni, data, wrong_direction_lists, zoom_area=None, file_name = "PhysicsSim_")
    visualize_error_map_official(deepcopy(pos_matrix), vertice, dni, data, wrong_direction_lists, zoom_area=(500, 325, 800, 400), file_name = "PhysicsSim_")
    ground_truth_positions = uploading_ground_truth(vertice,dni)
    ground_truth_comparison(vertice,dni,data,deepcopy(ground_truth_positions),deepcopy(refer_pos), deepcopy(pos_matrix), file_name = "PhysicsSim_")
    
    plot_force_heatmap_scalar_sum(
        pos_matrix=pos_matrix,
        vertice=vertice,
        dni=dni,
        data=data,
        directional_data=uploading_directional_data(),
        canvas_size=(1200, 750),
        sigma_px=28.0,                 # spread of each heat source
        show_points=True,
        save_path="C:/Users/justi/Desktop/project/results/phy_force_heatmap.png",
        window_caption="Force Heatmap (physics-like)"
    )
    
    save_vis_data(vertice, dni, deepcopy(pos_matrix), deepcopy(ground_truth_positions), deepcopy(refer_pos))
    save_err_data(vertice, dni, deepcopy(pos_matrix), deepcopy(ground_truth_positions), deepcopy(refer_pos), errors, edge_labels)
    
    '''
    plot_three_model_convergence_pygame_pixelaware( med_pos_hist_ph_px , med_pos_hist_dm_px, med_pos_hist_sm_px,
        vertice = vertice, dni = dni, data = data, ground_truth_positions=uploading_ground_truth(vertice, dni), fixed_point_labels = fixed_point_labels,
        fixed_point_lonlat = fixed_points_lonlat, refer_pos=(600, 500), bin_size_iters_dm=10, bin_size_iters_sm=25, pre_process=True )
    '''