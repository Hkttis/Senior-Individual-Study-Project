
from library.model_cmp import *
from library.data_io import *
from library.visualization import plot_three_model_convergence_pygame_pixelaware

if __name__ == "__main__" :
    
    
    # temporalily use ground truth to simulate given fixed points' positions
    graph, vertice, dni, edges, data = load_ini_data_from_csv(FILE_PATHS)
    
    
    
    fixed_point_labels = ["鄯善","都護治/烏壘"]
    gt = uploading_ground_truth(vertice,dni)
    fixed_points_lonlat = [ tuple(gt[dni[cout]]) for cout in fixed_point_labels]
    
    #res = multi_measurement_benchmark(n_runs = 100, refer_pos=(600, 500), fixed_point_labels = fixed_point_labels, fixed_point_lonlat = fixed_points_lonlat)
    #save_all_pos_histories_px_csv(res["all_pos_history_px"]["StressMajorization"],
    #                          res["all_pos_history_px"]["DirectedMDS"],
    #                          res["all_pos_history_px"]["PhysicsSim"])
    #all_pos_hist_sm_px, all_pos_hist_dm_px, all_pos_hist_ph_px = load_all_pos_histories_px_csv()
    
    #med_pos_hist_sm_px = select_median_pos_history( all_pos_hist_sm_px , dni, gt, refer_pos = (600,500))
    #med_pos_hist_dm_px = select_median_pos_history( all_pos_hist_dm_px , dni, gt, refer_pos = (600,500))
    #med_pos_hist_ph_px = select_median_pos_history( all_pos_hist_ph_px , dni, gt, refer_pos = (600,500))
    
    med_pos_hist_sm_px = run_stress_majorization(vis = False)
    med_pos_hist_dm_px = run_directed_MDS(vis = False)
    med_pos_hist_ph_px = run_physics_simulation_model(fixed_point_labels, fixed_points_lonlat, vis = False)
    
    #ground_truth_comparison(vertice,dni,data,deepcopy(gt), (600,500), deepcopy(med_pos_hist_ph_px[-1]), file_name = "PhysicsSim_")
    
    #'''
    plot_three_model_convergence_pygame_pixelaware( med_pos_hist_ph_px, med_pos_hist_dm_px, med_pos_hist_sm_px,
        vertice = vertice, dni = dni, data = data, ground_truth_positions=uploading_ground_truth(vertice, dni), fixed_point_labels = fixed_point_labels,
        fixed_point_lonlat = fixed_points_lonlat, refer_pos=(600, 500), bin_size_iters=10, pre_process=False )
    #'''