import csv
import random
from library.config import *
from library.geometry import inverse_lcc_transformation

# data input
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

def save_vis_data(vertice, dni, pos_matrix, ground_truth_positions, refer_pos):
    pos_matrix_km = []
    for pos in pos_matrix :
        pos_matrix_km.append(((pos[0]-refer_pos[0])*4.15,(pos[1]-refer_pos[1])*4.15))
    wgs_pos_matrix = inverse_lcc_transformation(pos_matrix_km,ground_truth_positions[dni["鄯善"]])
    vis_data = []
    for i,label in enumerate(vertice) :
        vis_data.append( (label, wgs_pos_matrix[i][0], wgs_pos_matrix[i][1]) )
    with open(FILE_PATHS["save_vis_data"], mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerows(vis_data)

def save_bootstrap_data(vertice, dni, samples, ground_truth_positions, refer_pos):
    pos_matrix_km = []
    for pos_matrix_sample in samples :
        for pos in pos_matrix_sample :
            pos_matrix_km.append(((pos[0]-refer_pos[0])*4.15,(pos[1]-refer_pos[1])*4.15))
    wgs_pos_matrix = inverse_lcc_transformation(pos_matrix_km,ground_truth_positions[dni["鄯善"]])
    bootstrap_data = []
    countries_N = len(vertice)
    for i,pos in enumerate(wgs_pos_matrix) :
        # 每個樣本的每個節點  ( bootstrap_index, countries_index, x, y )
        bootstrap_data.append( ( int(i/countries_N) , vertice[i%countries_N], pos[0], pos[1]) )
    with open(FILE_PATHS["save_bootstrap_data"], mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerows(bootstrap_data)

def loading_vis_data():
    # open saved vis_data
    # transform it into dictionary
    data = {}
    lst = []
    with open(FILE_PATHS["save_vis_data"], newline='', encoding='utf-8') as csvfile:
        rows = csv.reader(csvfile)
        for i, row in enumerate(rows) :
            dict = {}
            dict["name"] = row[0]
            dict["lon"] = float(row[1])
            dict["lat"] = float(row[2])
            dict["cluster"] = 5 #random.randint(0, 5)  # Random cluster for visualization
            lst.append(dict)
    data["nodes"] = lst
    return data
def loading_bootstrap_data(countries_N):
    # open saved bootstrap_data
    # transform it into dictionary
    data = {}
    lst = []
    with open(FILE_PATHS["save_bootstrap_data"], newline='', encoding='utf-8') as csvfile:
        rows = csv.reader(csvfile)
        for i, row in enumerate(rows) :
            dict = {}
            dict["name"] = row[1]
            dict["lon"] = float(row[2])
            dict["lat"] = float(row[3])
            dict["cluster"] = (i%countries_N) % 6
            lst.append(dict)
    data["nodes"] = lst
    return data

def gt_data_dict(vertice, dni):
    ground_truth_positions = uploading_ground_truth(vertice, dni)
    gt_data = []
    for i, pos in enumerate(ground_truth_positions) :
        dict = {
            "name": vertice[i],
            "lon": pos[0],
            "lat": pos[1],
            "cluster": 2  # Random cluster for visualization
        }
        gt_data.append(dict)
    return gt_data

'''
# data output
def turnto_csv(vertice,pos_matrix) :
    data = [[vertice[i],pos_matrix[i][0],pos_matrix[i][1]] for i in range(len(pos_matrix))]
    with open(FILE_PATHS["output_csv"], mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(data)
def classify_nodes():
    dt = FILE_PATHS["classification_data"]
    groupdni = {}
    with open( dt , newline='', encoding='utf-8' ) as csvfile :
        rows = csv.reader(csvfile)
        for row in rows :
            groupdni[row[0]] = int(row[1])
        groupdni['都護治/烏壘']=1
    return groupdni
'''

        
'''
Given the file path : "C:/Usersjusti/Desktop/project/results/visualization_data.json"

I want to make a function save_visualization_data(
'''