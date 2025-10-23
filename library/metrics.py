import math
import numpy as np
from library.config import km2pix, km2Li
from library.data_io import uploading_ground_truth
from library.geometry import lcc_transformation
from copy import deepcopy

def stress_function(data,dni,pos_matrix) : 
    stress = 0
    for row in data :
        ind1 = dni[row[0]]
        ind2 = dni[row[1]]
        # turn pixel_unit to Li unit
        distance = math.sqrt((pos_matrix[ind2][0]-pos_matrix[ind1][0])**2 + (pos_matrix[ind2][1]-pos_matrix[ind1][1])**2)
        ideal_dist = float(row[2])/10
        stress += (distance-ideal_dist)**2 / (ideal_dist**2)
    return stress

def calculate_kruskals_stress(dni,pos_matrix,data) :
    tmp_pos_matrix = []
    for pos_pair in pos_matrix : # transform the pixel unit to km 
        tmp_pos_matrix.append( (pos_pair[0] / km2pix, pos_pair[1] / km2pix) )
    error_square = 0
    dis_square = 0
    for row in data :
        p1 = tmp_pos_matrix[dni[row[0]]]
        p2 = tmp_pos_matrix[dni[row[1]]]
        actual_dis = np.sqrt( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        ideal_dis = float(row[2]) / km2Li
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
    anchor_label="鄯善",
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
            new_anc = len(DeX) - 1  # index in the filtered list
        
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

    # Find the anchor index (first occurrence if duplicated)
    try:
        anchor_idx = dni["鄯善"]
    except KeyError:
        raise KeyError("Label '鄯善' not found in vertice.") from None

    
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
        if vertice[i] == "鄯善":
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
def alignment_and_scaling(pos_matrix, vertice, dni, refer_pos):
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
    
    # Find the anchor index (first occurrence if duplicated)
    try:
        anchor_idx = dni["鄯善"]
    except ValueError:
        raise ValueError("Label '鄯善' not found in vertice.") from None

    # 1) Scale by 1/10, turn Li to pixel
    scale = 0.1
    scaled = [[x * scale, y * scale] for x, y in pos_matrix]
    
    # Be aware of the y-axis direction is flipped in pygame
    flip = flipping_y(scaled, height=750)

    # 2) Compute translation so '鄯善' lands at refer_pos
    anchor_x, anchor_y = flip[anchor_idx]
    dx = refer_pos[0] - anchor_x
    dy = refer_pos[1] - anchor_y

    # 3) Apply translation to all points
    aligned = [[x + dx, y + dy] for x, y in flip]
    
    

    return aligned

def flipping_y(pos_matrix, height):
    flipped = [[x, height - y] for x, y in pos_matrix]
    return flipped

def rmse_km_from_pixels(pos_px, refer_pos, dni, gt_lonlat):
    """
    Compute RMSE (km) between pixel positions and LCC-projected ground truth.
    - pos_px: list[(x,y)] in pixels, with 鄯善 expected near refer_pos.
    - refer_pos: (x0,y0) in pixels.
    """
    # 1) sim px -> km anchored at refer_pos
    sx_km = [((x - refer_pos[0]) / km2pix, (y - refer_pos[1]) / km2pix) for (x, y) in pos_px]
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
