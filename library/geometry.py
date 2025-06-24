import numpy as np
from math import *
from pyproj import Geod, CRS, Transformer

def shift(pos_matrix,scale,center_pos) : # Shift the average of points to the center_pos
    sumx = 0
    sumy = 0
    for pair in pos_matrix :
        sumx += pair[0]
        sumy += pair[1]
    sumx = sumx/len(pos_matrix)
    sumy = sumy/len(pos_matrix)
    for pair in pos_matrix :
        pair[0] = float(( pair[0] - sumx ) / scale + center_pos[0])
        pair[1] = float(( pair[1] - sumy ) / scale + center_pos[1])
    return pos_matrix

def lcc_transformation(dni, ground_truth_positions):
     # use Lambert Conformal Conic to transform ground_truth to shan_shan (0,0) in xy plane
    lon_min, lon_max = 73.24516481, 92.74103523
    lat_min, lat_max = 37.12265816, 44.2843368
    lat1 = lat_min
    lat2 = lat_max
    lon0 = (lon_min + lon_max) / 2
    # 建立 LCC 投影 CRS（單位：公尺）
    crs_lcc = CRS.from_proj4(
        f"+proj=lcc +lat_1={lat1} +lat_2={lat2} "
        f"+lat_0={lat1} +lon_0={lon0} +x_0=0 +y_0=0 "
        "+ellps=WGS84 +units=m"
    )
    # 建立經緯度 (EPSG:4326) → LCC (自訂 CRS) 的轉換器
    transformer = Transformer.from_crs("EPSG:4326", crs_lcc, always_xy=True)
    # ─── 執行投影並轉換為公里 ─────────────────────────────────
    lcc_xy_m = []
    for lon,lat in ground_truth_positions :
        if lon==0 and lat==0 :
            lcc_xy_m.append((None,None))
        else :
            lcc_xy_m.append(transformer.transform(lon, lat))
    align_pos = lcc_xy_m[dni['鄯善']]
    lcc_xy_km = []
    for x,y in lcc_xy_m :
        if x==None and y==None :
            lcc_xy_km.append((None,None))
        else :
            lcc_xy_km.append(((x-align_pos[0])/1000 , -(y-align_pos[1])/1000)) # we use (-y) since the coordinates in pygame is different
    return lcc_xy_km

def inverse_lcc_transformation(lcc_xy_km, wgs_align_pos):
    # Define projection parameters again (must be identical to the original)
    lon_min, lon_max = 73.24516481, 92.74103523
    lat_min, lat_max = 37.12265816, 44.2843368
    lat1 = lat_min
    lat2 = lat_max
    lon0 = (lon_min + lon_max) / 2

    # Rebuild CRS
    crs_lcc = CRS.from_proj4(
        f"+proj=lcc +lat_1={lat1} +lat_2={lat2} "
        f"+lat_0={lat1} +lon_0={lon0} +x_0=0 +y_0=0 "
        "+ellps=WGS84 +units=m"
    )

    # Create inverse transformer: LCC (projected meters) -> WGS84 (lon/lat)
    transformer = Transformer.from_crs(crs_lcc, "EPSG:4326", always_xy=True)

    # Get alignment reference point in LCC meters
    # We recompute the same way as in the forward function
    transformer_fwd = Transformer.from_crs("EPSG:4326", crs_lcc, always_xy=True)
    align_pos = transformer_fwd.transform(wgs_align_pos[0],wgs_align_pos[1])

    # Now back-project all points
    recovered_latlon = []
    for x_km, y_km in lcc_xy_km:
        if x_km is None and y_km is None:
            recovered_latlon.append((None, None))
        else:
            # Undo km to meter and reverse the alignment
            x_m = x_km * 1000 + align_pos[0]
            y_m = -y_km * 1000 + align_pos[1]  # Note the negative to invert pygame Y
            lon, lat = transformer.transform(x_m, y_m)
            recovered_latlon.append((lon, lat))
    
    return recovered_latlon



'''
# OLD functions
def alignment_and_rotation(vertice,dni,data,pos_matrix,theta_real) : 
    for pos in pos_matrix :
        pos[0]*= 10
        pos[1]*= 10
    for pos in pos_matrix : # 里 = 415 公尺
        pos[0] *= 415/1000
        pos[1] *= 415/1000
    x,y = pos_matrix[dni["鄯善"]]
    for pos in pos_matrix :
        pos[0]-= x
        pos[1]-= y
    i0 = dni["鄯善"]
    i1 = dni["都護治/烏壘"]
    theta_sim = np.arctan2(pos_matrix[i1][1] - pos_matrix[i0][1], pos_matrix[i1][0] - pos_matrix[i0][0]) ##remain not sure
    rotation_angle = theta_real - theta_sim
    cos_angle = np.cos(rotation_angle)
    sin_angle = np.sin(rotation_angle)
    rotation_matrix = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])
    for i, (x, y) in enumerate(pos_matrix):
        x_rotated, y_rotated = np.dot(rotation_matrix, [x, y])
        pos_matrix[i] = [x_rotated, y_rotated]
    print(pos_matrix)
    return pos_matrix
def calculate_new_coordinates(lat_ref, lon_ref, dx, dy,geod):
    distance = sqrt(dx**2 + dy**2)
    bearing = degrees(atan2(dy, dx))
    lon, lat, _ = geod.fwd(lon_ref, lat_ref, bearing, distance * 1000)
    return lon, lat
def projection(pos_matrix) :
    lat_ref = 40.527633
    lon_ref = 89.840644
    geod = Geod(ellps="WGS84")
    for i in range(len(pos_matrix)):
        x, y = pos_matrix[i]
        pos_matrix[i] = list(calculate_new_coordinates(lat_ref, lon_ref, x, y,geod))
    return pos_matrix
'''