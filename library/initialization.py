import numpy as np
from library.geometry import *
from library.data_io import *
from library.config import km2pix, km2Li
from library.units import gt_km2sim


def generate_CHEN_initial_positions (refer_pos, fixed_point_labels, fixed_points_lonlat, anchor_label=None): # Initialize position of points from CHEN_STRESSMAJORIZATION/random positions

    graph, vertice, dni, edges, data = load_ini_data_from_csv(FILE_PATHS)
    
    n = len(vertice)
    # initialize random positions in pos_matrix
    pos_matrix = np.column_stack((
        np.random.uniform(120, 1080, size=n),  # First column : canva 0~1200
        np.random.uniform(75, 675, size=n)   # Second column : canva 0~750
    ))
    ## pos_matrix = stress_majorization(graph,dni,vertice,edges) # inherit position form previous model
    
    
    
    center_pos = refer_pos
    
    pos_matrix = shift(pos_matrix,2,center_pos)
    pos_matrix,fixed_positions_list = add_fixed_positions(
        dni, pos_matrix, refer_pos, fixed_point_labels, fixed_points_lonlat, anchor_label=anchor_label
    )
    
    ## pos_matrix = pre_physics_simulation(pos_matrix,dni) #pre_PS ensures the accuracy of pos of fixed points 

    return vertice,dni,data,pos_matrix,fixed_positions_list

def construct_Chen_graph(data):
    countryset = set()
    for row in data :
        countryset.add(row[0])
        countryset.add(row[1])
    vertice = []
    dni = {}
    edges = []
    for coun in countryset :
        dni[coun] = len(vertice)
        vertice.append(coun)
    graph = [[] for i in range(len(vertice))]
    for row in data :
        edges.append((row[0],row[1]))
        graph[dni[row[0]]].append(row)
        graph[dni[row[1]]].append([row[1], row[0]] + row[2:])
    return graph,vertice,dni,edges
def add_fixed_positions(dni, pos_matrix, refer_pos, fixed_point_labels, fixed_point_lonlat, anchor_label=None):
    """
    使用 lcc_transformation 進行 LCC 投影，並將 fixed_points 對齊到畫布上的固定位置。
    回傳:
      - 更新後的 pos_matrix
      - 固定點列表 fixed_positions_list = [
            ['鄯善', x_pixel, y_pixel],
            ['都護治/烏壘', x_pixel, y_pixel]
        ]
    """
    
    if anchor_label is None:
        anchor_label = get_anchor_align_label()

    gt_lonlat = [(0,0) for _ in range(len(dni))]  # None for unknowns
    for label, lonlat in zip(fixed_point_labels, fixed_point_lonlat):
        if label not in dni:
            raise KeyError(f"Fixed point '{label}' not found in dni.")
        gt_lonlat[dni[label]] = lonlat

    if anchor_label not in fixed_point_labels:
        site_points = {row["name"]: (float(row["lon"]), float(row["lat"])) for row in load_site_points()}
        if anchor_label not in site_points:
            raise ValueError(f"Anchor align '{anchor_label}' is missing from site points.")
        gt_lonlat[dni[anchor_label]] = site_points[anchor_label]
    
    gt_xy_km = lcc_transformation_with_anchor(dni, gt_lonlat, anchor_label=anchor_label)
    
    gt_sim = gt_km2sim( gt_xy_km )

    fixed_positions_list = []
    for label in fixed_point_labels:
        fixed_positions_list.append([label, gt_sim[dni[label]][0], gt_sim[dni[label]][1] ])  # placeholder for positions
        pos_matrix[dni[label]] = [gt_sim[dni[label]][0] + refer_pos[0], gt_sim[dni[label]][1] + refer_pos[1]]
    
    
    return pos_matrix, fixed_positions_list
