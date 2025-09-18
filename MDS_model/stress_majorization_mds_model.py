import numpy as np
from numpy import linalg as LA
from scipy.linalg import solve_triangular
from copy import deepcopy
from library.config import km2pix, km2Li

# setting unknown points' weight 0, calculating L^w L^Z, then solve L^w * X(t+1) = L^X(t) * X(t),then
# by cholesky and trangular solver, until epsilon < 10^-3

def inix(n,d) :
    #   initialize ini_X by random ( or by arrange and reshape )
    ini_X = 10000*np.random.rand(n,d)
    return ini_X
def random_distr(n,d,weight,dismatrix) :
    #   show the average of random stress
    sum = 0
    lst = []
    for i in range(100) :
        #   15000 is a parameter control the size of random data ( value of np.random is in (0,1) )
        ini_X = 15000*np.random.rand(n,d)
        num = stress(n,ini_X,weight,dismatrix)
        sum = sum + num
        lst.append(num)
    lst.sort()
    for i in lst :
        print(i)
    print()
    print(sum/100)
    print('---------------------------')
def non_weight_graph_distance_matrix(graph,dni) :
    n = len(graph)
    #   calculate non_weight graph
    nwgraph = []
    for ver in graph :
        nwver = []
        for row in ver :
            nwver.append(dni[row[1]])
        nwgraph.append(nwver)
    #   calculate distance matrix
    dismatrix = np.zeros((n,n))
    for i in range(n) :
        for row in graph[i] :
            dismatrix[i][dni[row[1]]] = row[3]
    #   remind !! we use the shortest path to replace unknown distance 
    # dismatrix = np.array(floyd_warshall(graph,dni))
    return nwgraph,dismatrix
def calculate_weight_LW_dm (nwgraph,dismatrix):
    n = len(nwgraph)
    d = 2
    weight = np.zeros((n,n))
    for i in range(n) :
        for j in nwgraph[i] :
            weight[i][j] = 1
            weight[j][i] = 1
    #   calculate LW
    LW = np.zeros((n,n))
    for i in range(n) :
        sum = 0
        for j in range(n) :
            if i!=j :
                LW[i][j] = (-1)*weight[i][j]
                sum = sum + weight[i][j]
        LW[i][i] = sum
    return weight, LW , dismatrix
def invZ(Z,i,j) :
    num = LA.norm(Z[i]-Z[j])
    if num == 0 :
        return 0
    else :
        return 1/num
def calculate_LZ(n,weight,dismatrix,Z) :
    LZ = np.zeros((n,n))
    for i in range(n) :
        sum = 0
        for j in range(n) :
            if i!=j :
                LZ[i][j] = (-1)*weight[i][j]*dismatrix[i][j]*invZ(Z,i,j)
                sum = sum + LZ[i][j]
        LZ[i][i] = (-1)*sum
    return LZ
def stress(n,X,weight,dismatrix) :
    stress_value = 0
    for i in range(n) :
        for j in range(n) :
            stress_value  = stress_value + weight[i][j] * ( (LA.norm(X[i]-X[j])-dismatrix[i][j])*(LA.norm(X[i]-X[j])-dismatrix[i][j]) ) / (km2Li**2)
    return stress_value
def equation_solver(LW,LZ,Z) :
    #   by cholesky decomposition
    right = np.matmul(LZ,Z)
    right = np.delete(right,(0),axis=0)
    LW = np.delete(LW,(0),axis=0)
    LW = np.delete(LW,(0),axis=1)
    G = np.linalg.cholesky(LW)
    GT = np.transpose(G)
    Y = solve_triangular(G,right,lower = True)
    X = solve_triangular(GT,Y,lower = False)
    X = np.r_[np.array([[0,0]]),X]
    return X
def iterate(n,ini_X,weight,dismatrix,LW,graph,vertice,edges) :
    pre_stress = stress(n,ini_X,weight,dismatrix) 
    epsilon = pre_stress
    stress_history = [pre_stress]
    pos_history = [deepcopy(ini_X)]
    X = ini_X
    while epsilon > 0.0001 and pre_stress > 10:
        Z = deepcopy(X)
        LZ = calculate_LZ(n,weight,dismatrix,Z)
        X = equation_solver(LW,LZ,Z)
        now_stress = stress(n,X,weight,dismatrix)
        epsilon = (pre_stress-now_stress)/(pre_stress)
        pre_stress = now_stress
        stress_history.append(pre_stress)
        pos_history.append(deepcopy(X))
        if epsilon < 0 :
            return X
    return X, stress_history, pos_history
def stress_majorization(graph,dni,vertice,edges) :
    n = len(graph)
    ini_X = inix(n,2)
    nwgraph, dismatrix = non_weight_graph_distance_matrix(graph,dni)
    weight, LW , dismatrix = calculate_weight_LW_dm(nwgraph,dismatrix)
    pos_matrix, stress_history, pos_history = iterate(n,ini_X,weight,dismatrix,LW,graph,vertice,edges)
    return pos_matrix, stress_history, pos_history