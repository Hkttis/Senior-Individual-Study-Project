from copy import deepcopy

from library.geometry import *
from library.metrics import *
from library.config import km2pix, km2Li, FILE_PATHS
from library.data_io import uploading_ground_truth
from library.visualization import *
from library.physics import *
from library.initialization import *

from MDS_model.data_pre_processing import *
from MDS_model.stress_majorization_mds_model import *
from MDS_model.directed_mds_model import *
from MDS_model.plot_node_link_diagram import *



def run_directed_MDS( vis = True ):
    datanum = ["C:\\Users\\justi\Desktop\\project\\csv doc utf8\\GPT-4_史記_numerals_utf8.csv",
    "C:\\Users\\justi\Desktop\\project\\csv doc utf8\\GPT-4_漢書_numerals_utf8.csv",
    "C:\\Users\\justi\Desktop\\project\\csv doc utf8\\GPT-4_後漢書_numerals_utf8.csv"]
    pre_data = read_csvfile(datanum)
    c_data,disset = data_process(pre_data)
    graph, vertice, dni, edges, data = load_ini_data_from_csv(FILE_PATHS)
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
    graph, vertice, dni, edges, data = load_ini_data_from_csv(FILE_PATHS)
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
        fixed_point_lonlat = [ tuple(gt[dni[cout]]) for cout in fixed_point_labels]
        
        
        pos_matrix = procrustes_align_by_fixed_points(deepcopy(pos_matrix), fixed_point_labels, fixed_point_lonlat, dni)
        visualize_error_map_official(deepcopy(pos_matrix), vertice, dni, data, wrong_directions_list, file_name = "StressMj_")
        ground_truth_comparison(vertice,dni,data, uploading_ground_truth(vertice,dni),pos_matrix[dni["鄯善"]], deepcopy(pos_matrix), file_name = "StressMj_")
    
    return pos_history

def run_physics_simulation_model( fixed_point_labels, fixed_points_lonlat, vis = True) :
    refer_pos = [600,500]
    vertice,dni,data,pos_matrix,fixed_positions_list = generate_CHEN_initial_positions(deepcopy(refer_pos), fixed_point_labels, fixed_points_lonlat)
    
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




# Compute the benchmark of three models

def _align_history_stress_mj(pos_history_li, vertice, dni, refer_pos, fixed_point_labels, fixed_point_lonlat):
    """Li→px via alignment_and_scaling, then Procrustes by fixed points (px→px)."""
    out = []
    for frame_li in pos_history_li:
        px = alignment_and_scaling(deepcopy(frame_li), vertice, dni, refer_pos)  # Li->px (+flip)
        px = procrustes_align_by_fixed_points(deepcopy(px), fixed_point_labels, fixed_point_lonlat, dni, refer_pos=refer_pos)
        out.append(px)
    return out

def _align_history_directed_mds(pos_history_li, vertice, dni, refer_pos):
    """Li→px via alignment_and_scaling (no Procrustes)."""
    return [alignment_and_scaling(deepcopy(frame_li), vertice, dni, refer_pos) for frame_li in pos_history_li]

def _align_history_physics(pos_history_px):
    """Physics already in pixels, return deep copies to be safe."""
    return [deepcopy(frame_px) for frame_px in pos_history_px]

def _running_mean_history(mean_hist, new_hist, i_run):
    """
    Online mean of position histories (pixel units).
    If histories have unequal lengths, we average over the overlapping prefix.
    """
    if mean_hist is None:
        return [np.array(f, dtype=float) for f in new_hist]
    L = min(len(mean_hist), len(new_hist))
    for k in range(L):
        mean_hist[k] = (mean_hist[k] * i_run + np.array(new_hist[k], dtype=float)) / (i_run + 1)
    return mean_hist  # extra tail frames (if any) are ignored; see note below.

def _series_stats(x):
    """
    Return {mean, sd (sample), se, t975, ci_lo, ci_hi, n}.
    Falls back to normal 1.96 if SciPy missing.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n == 0:
        return dict(n=0, mean=np.nan, sd=np.nan, se=np.nan, t975=np.nan, ci_lo=np.nan, ci_hi=np.nan)
    mean = float(np.mean(x))
    # sample SD with Bessel's correction
    sd = float(np.sqrt(np.sum((x - mean) ** 2) / max(n - 1, 1)))
    se = sd / math.sqrt(n) if n > 0 else float("nan")
    try:
        from scipy.stats import t
        t975 = float(t.ppf(0.975, df=n - 1)) if n > 1 else float("nan")
    except Exception:
        t975 = 1.96 if n > 1 else float("nan")
    ci_lo = mean - t975 * se if n > 1 else float("nan")
    ci_hi = mean + t975 * se if n > 1 else float("nan")
    return dict(n=n, mean=mean, sd=sd, se=se, t975=t975, ci_lo=ci_lo, ci_hi=ci_hi)

def multi_measurement_benchmark(n_runs: int = 100, refer_pos=(600, 500), fixed_point_labels = [], fixed_point_lonlat = [], *, verbose: bool = True):
    """
    Repeat-measurement benchmark for three models with vis=False.
    - Prints progress "run i/n" and returns a dict with summary statistics and running-mean histories.
    - Computes Kruskal's stress (unitless) and RMSE (km) from each run's *final* positions.
    """
    # Static data (same across runs)
    graph, vertice, dni, edges, data = Chen_csv_and_graph()
    gt_lonlat = uploading_ground_truth(vertice, dni)

    # Running means of histories (in pixel units)
    mean_hist_sm = None
    mean_hist_dm = None
    mean_hist_ph = None

    # Lists of last-frame metrics
    ks_sm, rmse_sm = [], []
    ks_dm, rmse_dm = [], []
    ks_ph, rmse_ph = [], []
    
    all_pos_hist_sm_px = []
    all_pos_hist_dm_px = []
    all_pos_hist_ph_px = []

    for i in range(n_runs):
        # ---------------- Stress Majorization ----------------
        pos_hist_sm_li = run_stress_majorization(vis=False)                  # history in Li units
        pos_hist_sm_px = _align_history_stress_mj(pos_hist_sm_li, vertice, dni, refer_pos,
                                                  fixed_point_labels, fixed_point_lonlat)
        all_pos_hist_sm_px.append(pos_hist_sm_px)
        mean_hist_sm = _running_mean_history(mean_hist_sm, pos_hist_sm_px, i)
        last_sm = deepcopy(pos_hist_sm_px[-1])
        ks_sm.append(float(calculate_kruskals_stress(dni, deepcopy([list(p) for p in last_sm]), data)))
        rmse_sm.append(rmse_km_from_pixels(last_sm, refer_pos, dni, gt_lonlat))

        # ---------------- Directed MDS -----------------------
        pos_hist_dm_li = run_directed_MDS(vis=False)                          # history in Li units
        pos_hist_dm_px = _align_history_directed_mds(pos_hist_dm_li, vertice, dni, refer_pos)
        all_pos_hist_dm_px.append(pos_hist_dm_px)
        mean_hist_dm = _running_mean_history(mean_hist_dm, pos_hist_dm_px, i)
        last_dm = deepcopy(pos_hist_dm_px[-1])
        ks_dm.append(float(calculate_kruskals_stress(dni, deepcopy([list(p) for p in last_dm]), data)))
        rmse_dm.append(rmse_km_from_pixels(last_dm, refer_pos, dni, gt_lonlat))

        # ---------------- Physics Simulation ----------------
        pos_hist_ph_px = run_physics_simulation_model(fixed_point_labels, fixed_point_lonlat, vis = False)              # history already in pixels
        pos_hist_ph_px = _align_history_physics(pos_hist_ph_px)
        all_pos_hist_ph_px.append(pos_hist_ph_px)
        mean_hist_ph = _running_mean_history(mean_hist_ph, pos_hist_ph_px, i)
        last_ph = deepcopy(pos_hist_ph_px[-1])
        ks_ph.append(float(calculate_kruskals_stress(dni, deepcopy([list(p) for p in last_ph]), data)))
        rmse_ph.append(rmse_km_from_pixels(last_ph, refer_pos, dni, gt_lonlat))

        if verbose:
            print(f"[Progress] Completed {i+1}/{n_runs} runs")

    # --- Aggregate statistics ---
    stats = {
        "StressMajorization": {
            "Kruskal": _series_stats(ks_sm),
            "RMSE_km": _series_stats(rmse_sm),
        },
        "DirectedMDS": {
            "Kruskal": _series_stats(ks_dm),
            "RMSE_km": _series_stats(rmse_dm),
        },
        "PhysicsSim": {
            "Kruskal": _series_stats(ks_ph),
            "RMSE_km": _series_stats(rmse_ph),
        },
    }

    # Pretty print (numbers only)
    def _fmt(s): 
        return f"n={s['n']}, mean={s['mean']:.6g}, sd={s['sd']:.6g}, SE={s['se']:.6g}, 95%CI=[{s['ci_lo']:.6g},{s['ci_hi']:.6g}]"

    print("\n=== Summary (Kruskal’s stress) ===")
    for k, v in stats.items():
        print(f"{k:>18}: {_fmt(v['Kruskal'])}")

    print("\n=== Summary (RMSE, km) ===")
    for k, v in stats.items():
        print(f"{k:>18}: {_fmt(v['RMSE_km'])}")

    # Convert running means back to lists for return
    to_list = lambda H: None if H is None else [h.tolist() for h in H]

    return {
        "stats": stats,
        "mean_history": {
            "StressMajorization_px": to_list(mean_hist_sm),
            "DirectedMDS_px": to_list(mean_hist_dm),
            "PhysicsSim_px": to_list(mean_hist_ph),
        },
        "per_run": {
            "Kruskal": {"SM": ks_sm, "DM": ks_dm, "PH": ks_ph},
            "RMSE_km": {"SM": rmse_sm, "DM": rmse_dm, "PH": rmse_ph},
        },
        "all_pos_history_px": {
            "StressMajorization": all_pos_hist_sm_px,
            "DirectedMDS": all_pos_hist_dm_px,
            "PhysicsSim": all_pos_hist_ph_px,
        },
    }


from typing import Callable, List, Tuple, Optional
import math

Pos = List[float]                 # [x, y]
PosMatrix = List[Pos]             # [[x1,y1], [x2,y2], ...]
PosHistory = List[PosMatrix]      # [pos_matrix_iter1, pos_matrix_iter2, ...]
AllRuns = List[PosHistory]        # [pos_history_run0, pos_history_run1, ...]


def select_median_pos_history(
    all_pos_his_data: AllRuns,
    dni,
    ground_truth_positions,
    refer_pos,
    *,
    median_policy: str = "lower",      # "lower" | "upper" | "nearest"
    return_meta: bool = False
) -> PosHistory | Tuple[PosHistory, int, float]:
    """
    Select the pos_history whose FINAL pos_matrix RMSE is the median among runs.

    Parameters
    ----------
    all_pos_his_data : list of runs; each run is a list of pos_matrix over iterations.
    median_policy    : 
        - "lower"  : lower median order statistic (index floor((n-1)/2))
        - "upper"  : upper median order statistic (index ceil((n-1)/2))
        - "nearest": pick run whose RMSE is closest to the numeric median 
                     (for even n, median = average of two middle RMSEs);
                     ties resolved by smaller RMSE then smaller run index.
    return_meta      : if True, also return (run_index, final_rmse).

    Returns
    -------
    pos_history  (and optionally: run_index, final_rmse)
    
    """
    
    
    if not isinstance(all_pos_his_data, list) or len(all_pos_his_data) == 0:
        raise ValueError("all_pos_his_data must be a non-empty list of pos_history runs.")

    # 1) Compute final RMSE per run
    rmses: List[float] = []
    for r, pos_hist in enumerate(all_pos_his_data):
        if not pos_hist:
            raise ValueError(f"Run #{r} has empty pos_history.")
        final_pos: PosMatrix = pos_hist[-1]
        rmse_val = float(rmse_km_from_pixels(final_pos, refer_pos, dni, ground_truth_positions))
        if not (math.isfinite(rmse_val)):
            raise ValueError(f"Non-finite RMSE at run #{r}: {rmse_val!r}")
        rmses.append(rmse_val)

    n = len(rmses)
    # Prepare (rmse, index) pairs for order-statistic selection
    order = sorted([(rm, idx) for idx, rm in enumerate(rmses)], key=lambda p: (p[0], p[1]))

    # 2) Choose the median index according to policy
    if median_policy.lower() == "lower":
        k = (n - 1) // 2
        chosen_rm, chosen_idx = order[k]
    elif median_policy.lower() == "upper":
        k = n // 2
        chosen_rm, chosen_idx = order[k]
    elif median_policy.lower() == "nearest":
        # numeric median (average of two middles when n is even)
        if n % 2 == 1:
            numeric_median = order[(n - 1) // 2][0]
        else:
            numeric_median = 0.5 * (order[n // 2 - 1][0] + order[n // 2][0])
        # pick closest; tie-break by smaller RMSE then smaller index
        chosen_rm, chosen_idx = min(
            ((rm, idx) for idx, rm in enumerate(rmses)),
            key=lambda p: (abs(p[0] - numeric_median), p[0], p[1])
        )
    else:
        raise ValueError("median_policy must be one of: 'lower', 'upper', 'nearest'.")

    # 3) Return the selected full pos_history (and optional meta)
    selected_history = all_pos_his_data[chosen_idx]
    if return_meta:
        return selected_history, chosen_idx, chosen_rm
    return selected_history



