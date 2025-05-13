import csv
from config import *
from spring_main import lcc_transformation

def read_CHEN_csvfile() :
    # csv : 地點一 地點二 里程 里程 make it compatible to previous method
    data = [] # pouring all data into "data" without distinct book_class
    with open(FILE_PATHS["chen_data"], newline='', encoding='utf-8') as csvfile:
        data_tmp = []
        rows = csv.reader(csvfile)
        for row in rows :
            data_tmp.append(row)
        data_tmp.pop(0) # remove column name
        data = data + data_tmp
    return data
def uploading_fixed_positions(dni, pos_matrix, refer_pos, ground_truth_positions):
    """
    使用 lcc_transformation 進行 LCC 投影，並將「鄯善」與「都護治/烏壘...」對齊到畫布上的固定位置。
    回傳:
      - 更新後的 pos_matrix
      - 固定點列表 fixed_positions_list = [
            ['鄯善', x_pixel, y_pixel],
            ['都護治/烏壘', x_pixel, y_pixel]
        ]
    """
    FIXPOINTS = ['都護治/烏壘']
    
    gt_xy_km = lcc_transformation(dni, ground_truth_positions)
    km2pix = 1.0 / (10 * 0.415)
    fixed_positions_list = []
    
    # SHAN_SHAN   refer_pos 已是 pixel 座標，所以鄯善對齊保持不變
    fixed_positions_list.append(['鄯善', refer_pos[0], refer_pos[1]])
    pos_matrix[dni['鄯善']] = [refer_pos[0], refer_pos[1]]
    
    # FIX POINTS
    for fix_cities in FIXPOINTS :
        index = dni[fix_cities]
        (lcc_x,lcc_y) = gt_xy_km[index]
        px = lcc_x * km2pix + refer_pos[0]
        py = lcc_y * km2pix + refer_pos[1]
        pos_matrix[index] = [px,py]
        fixed_positions_list.append([fix_cities,px,py])
    
    return pos_matrix, fixed_positions_list
def uploading_directional_data():
    csv_file_path = FILE_PATHS["directional_data"]
    directional_data= []
    with open(csv_file_path, mode="r", newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file)  # Create a CSV reader object
        for row in reader:
            directional_data.append(row)
    return directional_data
def uploading_ground_truth(vertice,dni) :
    # uploading ground_truth files
    with open(FILE_PATHS["ground_truth_path"], newline='', encoding='utf-8') as csvfile:
        ## column 3 4 5 6 7 13 23 name, 24 25 x,y coordinates
        reader = csv.reader(csvfile)  # Create a CSV reader object
        next(reader)
        gt_tmp_data = []
        for row in reader:
            gt_tmp_data.append([[row[2],row[3],row[4],row[5],row[6],row[12],row[22]],[float(row[23]),float(row[24])]])
    n = len(dni)
    ground_truth_positions = [ [0,0] for i in range(n)]
    for row in gt_tmp_data :
        for name in row[0] : 
            if name in dni : # the name matchs the one in CHEN
                ground_truth_positions[dni[name]] = row[1]
                break
    return ground_truth_positions