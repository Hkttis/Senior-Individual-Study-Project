import math
import numpy as np
from copy import deepcopy
import builtins

from library.units import *
from library.data_io import uploading_ground_truth, get_anchor_align_label, load_site_points
from library.anchor_frame import px_list_to_km_list
from library.geometry import lcc_transformation
from library.coordinates import flipping_y
from library.config import theta_thr_4dir, theta_thr_8dir
from library.directions import DIR4_SIM, DIR8_UNIT_SIM


def stress_function(data,dni,pos_matrix) : # data units : km
    stress = 0
    for row in data :
        ind1 = dni[row[0]]
        ind2 = dni[row[1]]
        # turn pixel_unit to Li unit
        distance = math.sqrt((pos_matrix[ind2][0]-pos_matrix[ind1][0])**2 + (pos_matrix[ind2][1]-pos_matrix[ind1][1])**2)
        ideal_dist = float(row[2]) / Li2km
        stress += (distance-ideal_dist)**2 / (ideal_dist**2)
    return stress


def calculate_kruskals_stress(dni,pos_matrix,data) : # pos_matrix units = km, data units = sim
    error_square = 0
    dis_square = 0
    for row in data :
        p1 = pos_matrix[dni[row[0]]]
        p2 = pos_matrix[dni[row[1]]]
        actual_dis = np.sqrt( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        ideal_dis = float(row[2]) / km2sim
        error_square += (actual_dis-ideal_dis)**2
        dis_square += actual_dis**2
    kruskal_stress = np.sqrt(error_square/dis_square)
    return kruskal_stress

# ========= Anchor-only Procrustes (fixed points) =========
def procrustes_align_by_fixed_points(
    sim_km,
    fixed_point_labels,             # e.g., ["鄯善","都護治/烏壘", ...]
    fixed_point_lonlat,            # [(lon,lat), ...] same order/length as labels
    dni,
    refer_pos = [600, 500],   # in pixel units
    anchor_label=None,
):
    """
    Align a single Li-frame to a set of fixed points using 2D orthogonal Procrustes.
    Steps:
      1) Convert the Li frame to *km* and center it at the anchor.
      2) Build a per-node lon/lat list (None for unknowns), fill only the fixed points,
         and call lcc_transformation(dni, list) to get target points in km.
      3) Use *only those fixed points* to estimate R via SVD (rotation/reflection),
         rotating about the anchor. Return the aligned frame in km (anchor-centered).
    Notes:
      - Y is kept *north-up* in this function (no pygame flip).
      - Translation is handled by anchor-centering both sets.
    """
    
    if anchor_label is None:
        anchor_label = get_anchor_align_label()
    fixed_point_labels = list(fixed_point_labels)
    fixed_point_lonlat = list(fixed_point_lonlat)
    if anchor_label not in fixed_point_labels:
        site_points = {row["name"]: (float(row["lon"]), float(row["lat"])) for row in load_site_points()}
        if anchor_label not in site_points:
            raise ValueError(f"Anchor align '{anchor_label}' is missing from site points.")
        fixed_point_labels.append(anchor_label)
        fixed_point_lonlat.append(site_points[anchor_label])

    if len(fixed_point_labels) == 0 or (len(fixed_point_labels) != len(fixed_point_lonlat)):
        return sim_km
    
    if anchor_label not in dni:
        raise KeyError(f"Anchor '{anchor_label}' not found in dni.")
    anc = dni[anchor_label]
    
    gt_lonlat = [(0,0) for _ in range(len(dni))]  # None for unknowns
    for label, lonlat in zip(fixed_point_labels, fixed_point_lonlat):
        if label not in dni:
            raise KeyError(f"Fixed point '{label}' not found in dni.")
        gt_lonlat[dni[label]] = lonlat
    
    gt_xy_km = lcc_transformation(dni, gt_lonlat)  # per-node list in km (None where missing)
    
    for i in range(len(gt_xy_km)) :
        x, y = gt_xy_km[i]
        if x is not None and y is not None :
            gt_xy_km[i] = [x*km2pix, y*km2pix] # turn km to pix

    # 2) Remember the full pos_matrix
    X_full = np.asarray(deepcopy(sim_km), dtype=float)
    X_full -= X_full[anc]  # center at anchor
    
    # 3) There may be some nodes missing ground truth; filter them out
    DeX = [[-1,-1] for _ in range(len(fixed_point_labels))]
    GtX = [[-1,-1] for _ in range(len(fixed_point_labels))]
    new_anc = 0
    
    
    for i, label in enumerate(fixed_point_labels):
        if gt_xy_km[dni[label]][0] is None:
            raise ValueError(f"Fixed point '{label}' is missing ground truth coordinates.")
        
        ind = dni[label]
        
        DeX[i][0], DeX[i][1] = sim_km[ind][0], sim_km[ind][1]
        
        GtX[i][0], GtX[i][1] = gt_xy_km[ind][0], gt_xy_km[ind][1]
        
        if label == anchor_label:
            new_anc = i  
        
    sim_km = deepcopy(DeX)
    gt_xy_km = deepcopy(GtX)
    
    X_px = np.asarray(sim_km, dtype=float)
    G_px = np.asarray(gt_xy_km, dtype=float)
    
    
    # Center both sets at the anchor (rotate about 鄯善)
    X0 = X_px - X_px[new_anc]
    G0 = G_px - G_px[new_anc]

    #    Orthogonal Procrustes (rotation or reflection)
    #    Minimize || X0 R - G0 ||_F, subject to R^T R = I, det(R) = +1
    C = X0.T @ G0                      
    U, _, Vt = np.linalg.svd(C)
    R = U @ Vt
    
    # Apply the R matrix (about the anchor), then translate so 鄯善 = refer_pos
    X_rot = X_full @ R
    aligned_pos = X_rot + np.asarray(refer_pos, dtype=float)

    return aligned_pos.tolist()

# For stress majorization
def procrustes_analysis_to_gt(flip, vertice, dni, refer_pos) :
    """
    Align positions (in Li) to ground truth (in km) using rotation/reflection Procrustes
    about the anchor node '鄯善', and return pixel coordinates aligned so that the
    anchor lands at `refer_pos`.
    """
    
    # Basic validation
    if not isinstance(refer_pos, (list, tuple)) or len(refer_pos) != 2:
        raise ValueError("refer_pos must be a 2-element list/tuple like [x, y].")
    if len(flip) != len(vertice):
        raise ValueError("pos_matrix and vertice must have the same length.")

    anchor_label = get_anchor_align_label()
    # Find the anchor index (first occurrence if duplicated)
    try:
        anchor_idx = dni[anchor_label]
    except KeyError:
        raise KeyError(f"Label '{anchor_label}' not found in vertice.") from None

    
    # 1) Do Orthogonal Procrustes to best align with ground truth positions
    ground_truth_positions = uploading_ground_truth(vertice,dni)
    gt_xy_km = lcc_transformation(dni, ground_truth_positions)
    
    for i in range(len(gt_xy_km)) :
        x, y = gt_xy_km[i]
        if x is not None and y is not None :
            gt_xy_km[i] = [x*km2pix, y*km2pix] # turn km to pix
    
    # 2) Remember the full pos_matrix
    X_full = np.asarray(deepcopy(flip), dtype=float)
    X_full -= X_full[anchor_idx]  # center at anchor
    
    # 3) There may be some nodes missing ground truth; filter them out
    DeX = []
    Deg = []
    new_anc = 0
    for i, (gtx, gty) in enumerate(gt_xy_km):
        if gtx is not None and gty is not None:
            DeX.append(flip[i])
            Deg.append([gtx, gty])
        if vertice[i] == anchor_label:
            new_anc = len(DeX) - 1  # new index of anchor in filtered list
    
    flip = deepcopy(DeX)
    gt_xy_km = deepcopy(Deg)
    
    X_px = np.asarray(flip, dtype=float)
    G_px = np.asarray(gt_xy_km, dtype=float)
    # Center both sets at the anchor (rotate about 鄯善)
    X0 = X_px - X_px[new_anc]
    G0 = G_px - G_px[new_anc]

    #    Orthogonal Procrustes (rotation or reflection)
    #    Minimize || X0 R - G0 ||_F, subject to R^T R = I, det(R) = +1
    C = X0.T @ G0                      
    U, _, Vt = np.linalg.svd(C)
    R = U @ Vt

    # Apply the R matrix (about the anchor), then translate so 鄯善 = refer_pos
    X_rot = X_full @ R
    aligned_pos = X_rot + np.asarray(refer_pos, dtype=float)

    return aligned_pos.tolist()

# For directed MDS
def alignment_and_scaling(pos_matrix, vertice, dni, refer_pos, y_down=True, anchor_label=None):
    """
    Scale all coordinates by scale and translate so that the point labeled
    '鄯善' matches refer_pos.
    Raises ValueError
    If '鄯善' is not found or refer_pos is invalid.
    """
    # Basic validation
    if not isinstance(refer_pos, (list, tuple)) or len(refer_pos) != 2:
        raise ValueError("refer_pos must be a 2-element list/tuple like [x, y].")
    if len(pos_matrix) != len(vertice):
        raise ValueError("pos_matrix and vertice must have the same length.")
    
    if anchor_label is None:
        anchor_label = get_anchor_align_label()
    # Find the anchor index (first occurrence if duplicated)
    try:
        anchor_idx = dni[anchor_label]
    except KeyError:
        raise ValueError(f"Label '{anchor_label}' not found in vertice.") from None

    # 1) Scale by 1/10, turn Li to pixel
    scale = Li2pix
    scaled = [[x * scale, y * scale] for x, y in pos_matrix]
    
    # Be aware of the y-axis direction is flipped in pygame
    if y_down :
        flip = flipping_y(scaled)
    else :
        flip = scaled
    # 2) Compute translation so '鄯善' lands at refer_pos
    anchor_x, anchor_y = flip[anchor_idx]
    dx = refer_pos[0] - anchor_x
    dy = refer_pos[1] - anchor_y

    # 3) Apply translation to all points
    aligned = [[x + dx, y + dy] for x, y in flip]
    
    

    return aligned

def rmse_km_from_pixels(pos_px, refer_pos, dni, gt_lonlat):
    """
    Compute RMSE (km) between pixel positions and LCC-projected ground truth.
    - pos_px: list[(x,y)] in pixels, with 鄯善 expected near refer_pos.
    - refer_pos: (x0,y0) in pixels.
    """
    # 1) sim px -> km anchored at refer_pos
    sx_km = px_list_to_km_list(pos_px, tuple(refer_pos), km2pix)
    # 2) project GT lon/lat -> LCC (km), anchor at 鄯善 with north-up convention
    gt_km = lcc_transformation(dni, gt_lonlat)  # returns list[(x,y) or (None,None)]
    # 3) RMSE over nodes with GT
    se = []
    for i, (sx, sy) in enumerate(sx_km):
        gx, gy = gt_km[i]
        if gx is None or gy is None:
            continue
        dx, dy = sx - gx, sy - gy
        se.append(dx * dx + dy * dy)
    return math.sqrt(sum(se) / len(se)) if se else float("nan")


# =========================================================
# Direction error metrics (Angular hinge / atan2-based)
# =========================================================

def _direction_violation_deltas(pos_matrix_px, directional_data, dni, eps=1e-9):
    """
    Internal helper:
    Iterate directional constraints and compute deltas:
      phi = atan2(cross, dot) in (-pi, pi]
      delta = max(0, abs(phi) - theta_h)
    Returns:
      total_valid_edges, n_violations, deltas(list of delta for violated edges)
    """
    # pos_matrix_px is expected to be (n,2) in y-up; be tolerant to list/Vec2d input
    try:
        pos = np.asarray(pos_matrix_px, dtype=float)
        if pos.ndim != 2 or pos.shape[1] != 2:
            raise ValueError
    except Exception:
        # fallback for e.g. list of Vec2d
        pos = np.array([(p[0], p[1]) for p in pos_matrix_px], dtype=float)

    total = 0
    n_viol = 0
    deltas = []

    for row in directional_data:
        # row format: [u_name, v_name, direction_name, ...]
        if row is None or len(row) < 3:
            continue

        u_name = row[0]
        v_name = row[1]
        d_name = str(row[2]).strip()

        if (u_name not in dni) or (v_name not in dni):
            continue
        if d_name not in DIR8_UNIT_SIM:
            continue

        iu = dni[u_name]
        iv = dni[v_name]
        if iu < 0 or iv < 0 or iu >= pos.shape[0] or iv >= pos.shape[0]:
            continue

        rx = float(pos[iv, 0] - pos[iu, 0])
        ry = float(pos[iv, 1] - pos[iu, 1])
        dist = math.hypot(rx, ry)
        if dist < eps:
            continue

        r_hat_x = rx / dist
        r_hat_y = ry / dist

        v_dir = DIR8_UNIT_SIM[d_name]
        v_x = float(v_dir[0])
        v_y = float(v_dir[1])

        # dot = cos(phi), cross = sin(phi) (2D cross z-component)
        d_val = r_hat_x * v_x + r_hat_y * v_y
        c_val = r_hat_x * v_y - r_hat_y * v_x
        phi = math.atan2(c_val, d_val)  # (-pi, pi]

        theta_h = theta_thr_4dir if (d_name in DIR4_SIM) else theta_thr_8dir
        delta = builtins.max(0.0, abs(phi) - theta_h)

        total += 1
        if delta > 0.0:
            n_viol += 1
            deltas.append(float(delta))

    return total, n_viol, deltas


def direction_violation_rate(pos_matrix_px, directional_data, dni):
    """
    Violation Rate:
      VR = (#edges with delta>0) / (#valid directional edges)
    If no valid edges, return 0.0.
    """
    total, n_viol, _ = _direction_violation_deltas(pos_matrix_px, directional_data, dni)
    if total <= 0:
        return 0.0
    return float(n_viol) / float(total)


def mean_angular_error_violations(pos_matrix_px, directional_data, dni):
    """
    Mean Angular Error on violated edges:
      MAE_theta = mean(delta) over edges with delta>0
    If no violated edges, return 0.0 (avoid downstream stats issues).
    """
    _, n_viol, deltas = _direction_violation_deltas(pos_matrix_px, directional_data, dni)
    if n_viol <= 0:
        return 0.0
    return float(np.mean(np.asarray(deltas, dtype=float)))

