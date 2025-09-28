
from library.model_cmp import *
from library.data_io import uploading_ground_truth
from library.visualization import plot_three_model_convergence_pygame_pixelaware

if __name__ == "__main__" :
    
    
    # run_stress_majorization(True)
    
    res = multi_measurement_benchmark(n_runs = 5, refer_pos=(600, 500))
    
    
    '''
    graph,vertice,dni,edges,data= Chen_csv_and_graph()
    # temporalily use ground truth to simulate given fixed points' positions
    fixed_point_labels = ["鄯善","都護治/烏壘"]
    gt = uploading_ground_truth(vertice,dni)
    fixed_points_lonlat = [ tuple(gt[dni[cout]]) for cout in fixed_point_labels]
    
    pos_history_directed_mds = run_directed_MDS(vis = False)
    pos_history_stress_mj = run_stress_majorization(vis = False)
    pos_history_physics_sim = run_physics_simulation_model(vis = False)
    
    
    plot_three_model_convergence_pygame_pixelaware( pos_history_physics_sim, pos_history_directed_mds, pos_history_stress_mj,
        vertice = vertice, dni = dni, data = data, ground_truth_positions=uploading_ground_truth(vertice, dni), fixed_point_labels = fixed_point_labels,
        fixed_point_lonlat = fixed_points_lonlat, refer_pos=(600, 500), bin_size_iters=10 )
    '''