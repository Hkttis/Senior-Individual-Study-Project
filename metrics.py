import math
import numpy as np

def stress_function(data,dni,pos_matrix) : # unit : (km^2)； 1里=0.415km
    stress = 0
    for row in data :
        ind1 = dni[row[0]]
        ind2 = dni[row[1]]
        distance = math.sqrt((pos_matrix[ind2][0]-pos_matrix[ind1][0])**2 + (pos_matrix[ind2][1]-pos_matrix[ind1][1])**2)*10*0.415
        stress += (distance-int(row[2])*0.415)**2
    return stress

def calculate_kruskals_stress(dni,pos_matrix,data) :
    for pos_pair in pos_matrix : # transform the pixel unit to km 
        pos_pair[0] = pos_pair[0] *10 *0.415
        pos_pair[1] = pos_pair[1] *10 *0.415
    error_square = 0
    dis_square = 0
    for row in data :
        p1 = pos_matrix[dni[row[0]]]
        p2 = pos_matrix[dni[row[1]]]
        actual_dis = np.sqrt( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        ideal_dis = int(row[2])*0.415
        error_square += (actual_dis-ideal_dis)**2
        dis_square += actual_dis**2
    kruskal_stress = np.sqrt(error_square/dis_square)
    return kruskal_stress
