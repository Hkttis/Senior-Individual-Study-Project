import math
import numpy
import builtins
from copy import deepcopy
from numpy import *
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import cg

from library.config import km2pix, km2Li


insq2 = 1/math.sqrt(2)
unit_direction_dict = {'東':[1,0], '西':[-1,0], '北':[0,1], '南':[0,-1], 
              '東南':[insq2,-insq2], '西北':[-insq2,insq2], '東北':[insq2,insq2], '西南':[-insq2,-insq2]}
# configuration
fix_weight = 1  # weight of fixed points is 10000, let it be fixed because it wanna fit the dij
w_weight = 1
v_weight = 0.0001
stop_iteration_times = 500


def eudis(v1,v2) :
    return math.sqrt((v1[0]-v2[0])**2 + (v1[1]-v2[1])**2)
def fixed_input(n,dni) :
    fixposdni = {}
    #fixposdni = {'鄯善':[0,0],'都護治/烏壘':[-1150,1520],'車師後':[-202,2315]}
    fixed_points_flag = [0 for i in range(n)]
    inipos = numpy.random.rand(n,2)
    for key in fixposdni :
        fixed_points_flag[dni[key]] = 1
        inipos[dni[key]] = numpy.array(fixposdni[key])
    return fixed_points_flag, inipos
def select_data(n,pre_data,data,dni) :
    # Select_data() strives to select the data in c_data and Chen_method data simultaneously,
    #   return sel_data as directional data
    # Notation in_dis_flag denotes the booling array, showing whether the node is in Chen_method data
    # Notation in_direct_flag denotes ..., showing which nodes in c_data are also in Chen_method data
    # As a result, is_dis_flag should be all '1', and in_direct_flag should be the subset of ~dis~ 
    sel_data = []
    in_dis_flag = [0 for i in range(n)]
    in_direct_flag = [0 for i in range(n)]
    for row in data :
        in_dis_flag[dni[row[0]]] = 1
        in_dis_flag[dni[row[1]]] = 1
    for row in pre_data :
        if (row[0] in dni) and (row[1] in dni) :
            sel_data.append(row)
            in_direct_flag[dni[row[0]]] = 1
            in_direct_flag[dni[row[1]]] = 1
    cnt = 0
    for i in range(n) :
        if in_direct_flag[i] == 1 :
            cnt = cnt + 1
    return sel_data,in_dis_flag,in_direct_flag,cnt
def filter(n,X,vertice,dni) :
    Xf = [[0,0]for i in range(n)]
    for i in range(len(X)):
        if vertice[i] in dni :
            Xf[dni[vertice[i]]] = X[i]
    return Xf
def revise_direction(sel_data) :
    direction_dictionary = ['東南','西北','東北','西南','北','東','南','西']
    for i in range(len(sel_data)) :
        for j in range(8) :
            if direction_dictionary[j] in sel_data[i][5] :
                sel_data[i][2] = direction_dictionary[j]
                break
    return sel_data
def avg_dis(dis):
    sum = 0
    cnt = 0
    for i in range(len(dis)) :
        for num in dis[i] :
            if num != 0 :
                sum = sum + num
                cnt = cnt + 1
    return sum/cnt

def compute_weight_LW_veight_LV_JW_JV(n,s,m,t,sel_data,graph,vertice,dni,edges,dis,in_dis_flag,in_direct_flag,fixed_points_flag) :
    weight = [[0 for i in range(n)] for j in range(n)]
    for ver in graph :
        for row in ver :
            if dis[dni[row[0]]][dni[row[1]]] == 0 :
                print("Warning : distance_error")
            weight[dni[row[0]]][dni[row[1]]] = w_weight / (dis[dni[row[0]]][dni[row[1]]]**2)
            weight[dni[row[1]]][dni[row[0]]] = w_weight / (dis[dni[row[0]]][dni[row[1]]]**2)
            if fixed_points_flag[dni[row[0]]]==1 and fixed_points_flag[dni[row[1]]] ==1 :
                weight[dni[row[0]]][dni[row[1]]] = fix_weight
                weight[dni[row[1]]][dni[row[0]]] = fix_weight
    LW = [[0 for i in range(n)] for j in range(n)]
    for i in range(n) :
        sum = 0
        for j in range(n) :
            if i !=j :
                LW[i][j] = (-1)*weight[i][j]
                sum = sum + weight[i][j]
        LW[i][i] = sum
    for i in range(n) :
        for j in range(n) :
            if LW[i][j] != LW[j][i] :
                print('********************warning LW is not symmetric************************')
    
    veight = [[0 for i in range(n)] for j in range(n)]
    for row in sel_data :
        if dis[dni[row[0]]][dni[row[1]]] != 0 :
            veight[dni[row[0]]][dni[row[1]]] = v_weight / (dis[dni[row[0]]][dni[row[1]]]**2)
            veight[dni[row[1]]][dni[row[0]]] = v_weight / (dis[dni[row[0]]][dni[row[1]]]**2)
        else :
            veight[dni[row[0]]][dni[row[1]]] = v_weight / (avg_dis(dis)**2)
            veight[dni[row[1]]][dni[row[0]]] = v_weight / (avg_dis(dis)**2)
    '''
    for i in range(n) : # level up the weight of the points connected to the fixed points
        if fixed_points_flag[i] ==1 :
            for row in graph[i] :
                veight[i][dni[row[1]]] = 1000
                veight[dni[row[1]]][i] = 1000
    '''
    # like weight , add into veight as same as 
    LV = [[0 for i in range(n)] for j in range(n)]
    for i in range(n) :
        sum = 0
        for j in range(n) :
            if i !=j :
                LV[i][j] = (-1)*veight[i][j]
                sum = sum + veight[i][j]
        LV[i][i] = sum
    for i in range(n) :
        for j in range(n) :
            if LV[i][j] != LV[j][i] :
                print('********************warning LV is not symmetric************************')
    
    JW = [[0 for i in range(int(n*(n-1)/2))] for j in range(n)]
    cnt = 0
    for i in range(n) :
        for j in range(i) : # j < i
            if (vertice[i],vertice[j]) in edges or (vertice[j],vertice[i]) in edges :
                JW[j][cnt] = weight[i][j]
                JW[i][cnt] = (-1)*weight[i][j]
            cnt += 1
    
    JV = [[0 for i in range(t)] for j in range(n)]
    for i in range(len(sel_data)) :
        x = dni[sel_data[i][0]]
        y = dni[sel_data[i][1]]
        JV[x][i] = veight[x][y] # the previous one be source node
        JV[y][i] = (-1)*veight[x][y]
    
    array_weight = numpy.array(weight)
    array_LW = numpy.array(LW)
    array_veight = numpy.array(veight)
    array_LV = numpy.array(LV)
    array_JW = numpy.array(JW)
    array_JV = numpy.array(JV)
    return array_weight,array_LW,array_veight,array_LV,array_JW,array_JV
def compute_DW_DV(n,s,m,t,X,sel_data,graph,vertice,dni,edges,dis,fixed_points_flag) :
    
    DW = numpy.zeros((int(n*(n-1)/2),2))
    cnt = 0
    for i in range(n) :
        for j in range(i) : # j < i
            v = X[i]-X[j]
            if linalg.norm(v)==0 :
                unit = numpy.zeros((1,2))
            else : 
                unit = v/linalg.norm(v)
            DW[cnt] = dis[i][j]*deepcopy(unit)
            cnt += 1
    
    DV = numpy.zeros((t,2))
    for i in range(t) :
        x = dni[sel_data[i][0]]
        y = dni[sel_data[i][1]]
        v = X[y]-X[x]
        unit = numpy.array(unit_direction_dict[sel_data[i][2]])
        if dis[x][y] !=0 :
            DV[i] = deepcopy(dis[x][y]*unit)
        else :
            DV[i] = deepcopy((linalg.norm(v)*unit))
    
    return DW,DV
def stress(n,s,m,t,X,weight,veight,in_direct_flag,dni,edges,sel_data,dis) : # weigh in km**2
    stressw = 0
    stressv = 0
    ''' #TODO : check now_stress use previous DWDV or now DWDV 
    for i in range(s) :
        x = max(dni[edges[i][0]],dni[edges[i][1]])
        y = min(dni[edges[i][0]],dni[edges[i][1]])
        v = X[x]-X[y]
        stressw = stressw + weight[x][y]*(linalg.norm(v-DW[i])**2)
        #print(stress)
    for i in range(t) :
        x = dni[sel_data[i][0]]
        y = dni[sel_data[i][1]]
        v = X[y]-X[x]
        stressv = stressv + veight[x][y]*(linalg.norm(v-DV[i])**2)
        #print(v,' ',DV[i],' ',linalg.norm(v-DV[i]))
        #print(stress)
    '''
    for i in range(s) :
        x = dni[edges[i][0]]
        y = dni[edges[i][1]]
        stressw = stressw + weight[x][y]*((linalg.norm(X[x]-X[y])-dis[x][y])**2) / (km2Li**2)
    
    for i in range(t) :
        x = dni[sel_data[i][0]]
        y = dni[sel_data[i][1]]
        v = X[y]-X[x]
        unitx = v/linalg.norm(v)
        unitdata = numpy.array(unit_direction_dict[sel_data[i][2]])
        stressv = stressv + veight[x][y]*(( linalg.norm(v)*linalg.norm(unitx-unitdata) )**2) / (km2Li**2)
        #stressv = stressv + veight[x][y]*((linalg.norm(v)*(numpy.dot(unitx,unitdata)-1))**2)
    return stressw
def iterate(n,s,m,t,sel_data,graph,vertice,dni,edges,dis,fixed_points_flag,in_direct_flag,inipos,weight,LW,veight,LV,JW,JV) :
    iniX = deepcopy(inipos)
    pre_DW,pre_DV = compute_DW_DV(n,s,m,t,iniX,sel_data,graph,vertice,dni,edges,dis,fixed_points_flag)
    pre_stress = stress(n,s,m,t,iniX,weight,veight,in_direct_flag,dni,edges,sel_data,dis)
    now_stress = 0
    Z = iniX
    epsilon = 1
    stress_history = [pre_stress]
    pos_history = [deepcopy(iniX)]  # record the initial positions
    cnt = 0
    #while epsilon >= 0.0001 :
    while cnt <= stop_iteration_times :
        left = LW+LV
        right = numpy.matmul(JW,pre_DW)+numpy.matmul(JV,pre_DV)
        
        X = numpy.zeros(right.shape)  # Preallocate X to hold solutions
        for i in range(right.shape[1]):  # Iterate over each column of b
            x, exit_code = cg(left, right[:, i])
            if exit_code != 0:
                print(f'**********CG failed to converge with exit code {exit_code}*********')
            X[:, i] = x
        now_DW,now_DV = compute_DW_DV(n,s,m,t,X,sel_data,graph,vertice,dni,edges,dis,fixed_points_flag)
        now_stress = stress(n,s,m,t,X,weight,veight,in_direct_flag,dni,edges,sel_data,dis)
        pre_DW = now_DW
        pre_DV = now_DV
        epsilon = abs((pre_stress-now_stress)/(pre_stress))
        stress_history.append(now_stress)
        pre_stress = now_stress
        cnt = cnt + 1
        Z = X
        pos_history.append(deepcopy(Z)) # record the current positions
    return Z, stress_history, pos_history
def directed_MDS(c_data,data,graph,vertice,dni,edges) : # c_data is from data_process, which [0,2] contain directed data
    n = len(vertice) # the number of points             # data~edges are from Chen~_method
    s = len(edges) # the number of the points' edges
    # FIXME : adding the dis between fixed points into data
    fixed_points_flag,  inipos = fixed_input(n,dni)
    sel_data,in_dis_flag,in_direct_flag, m = select_data(n,c_data[0]+c_data[2],data,dni)
    sel_data = revise_direction(sel_data)
    t = len(sel_data)
    dis =  numpy.zeros((n,n))
    for ver in graph :
        for row in ver :
            dis[dni[row[0]]][dni[row[1]]] = int(row[3])
            dis[dni[row[1]]][dni[row[0]]] = int(row[3])
    # n is number of all nodes, s is number of all distance edges
    # m is number of nodes with directional edges (E'), t is the number of directional edges
    weight,LW,veight,LV,JW,JV = compute_weight_LW_veight_LV_JW_JV(n,s,m,t,sel_data,graph,vertice,dni,edges,dis,in_dis_flag,in_direct_flag,fixed_points_flag)
    anspos, stress_history, pos_history = iterate(n,s,m,t,sel_data,graph,vertice,dni,edges,dis,fixed_points_flag,in_direct_flag,inipos,weight,LW,veight,LV,JW,JV)
    
    return anspos, stress_history, pos_history