import pymunk
import pymunk.pygame_util
import pygame
import os
from pyproj import Geod, CRS, Transformer

from math import *
import math
import numpy as np
import csv
# temporarily stop the plotting_physics_simulation function in run_phy...

from metrics import *
from config import *
from data_io import *
from geometry import *
from visulization import *
from physics import *


def generate_CHEN_initial_positions (refer_pos): # Initialize position of points from CHEN_STRESSMAJORIZATION/random positions
    data = read_CHEN_csvfile()
    graph,vertice,dni,edges= construct_Chen_graph(data)
    n = len(vertice)
    # initialize random positions in pos_matrix
    pos_matrix = np.column_stack((
        np.random.uniform(120, 1080, size=n),  # First column : canva 0~1200
        np.random.uniform(75, 675, size=n)   # Second column : canva 0~750
    ))
    ## pos_matrix = stress_majorization(graph,dni,vertice,edges) # inherit position form previous model
    center_pos = [600,375]
    pos_matrix = shift(pos_matrix,2,center_pos)
    ground_truth_positions = uploading_ground_truth(vertice,dni)
    pos_matrix,fixed_positions_list = add_fixed_positions(dni,pos_matrix,refer_pos, ground_truth_positions)
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
        graph[dni[row[1]]].append([row[1]]+[row[0]]+[row[2]]+[row[3]])
    return graph,vertice,dni,edges
def add_fixed_positions(dni, pos_matrix, refer_pos, ground_truth_positions):
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
  

def main_function(): # avoid global parameters
    refer_pos = [600,500]
    vertice,dni,data,pos_matrix,fixed_positions_list = generate_CHEN_initial_positions(refer_pos)
    directional_data = uploading_directional_data()
    wrong_direction_lists,stress_history,pos_matrix = main_physics_simulation(vertice,dni,data,pos_matrix,directional_data,fixed_positions_list)
    plot_stress_convergence_log(stress_history)
    visualize_error_map_official(pos_matrix, vertice, dni, data, wrong_direction_lists, zoom_area=None)
    visualize_error_map_official(pos_matrix, vertice, dni, data, wrong_direction_lists, zoom_area=(500, 325, 800, 400))
    ground_truth_positions = uploading_ground_truth(vertice,dni)
    ground_truth_comparison(vertice,dni,data,ground_truth_positions,refer_pos,pos_matrix)

def main_function() :
    data = read_CHEN_csvfile()
    graph,vertice,dni,edges= construct_Chen_graph(data)
    n = len(vertice)
    

if __name__ == "__main__":
    main_function()
    import spring_confidence as sc
    # mu, covs, vertice = sc.bootstrap_and_plot()
