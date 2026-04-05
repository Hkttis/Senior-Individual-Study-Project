'''
read_csvfile is from read_csvfile.py in graph_drawing
from addin to data_process is from data_processing.py in graph_drawing
Chen_csv_and_graph is from Chen_Shih_Liang_method_data.py in graph_drawing
'''

def read_csvfile( datanum ): # datanum is the collection of dataname
    import csv
    data = [] # pouring all data into "data" without distinct book_class
    for i in range( len(datanum) ):
        with open( datanum[i] , newline='', encoding='utf-8' ) as csvfile :
            data_tmp = []
            rows = csv.reader(csvfile)
            for row in rows :
                data_tmp.append(row)
            data_tmp.pop(0) # remove column name
            data = data + data_tmp
    return data

# 地點1 地點2 方位 里程 書籍 原文文句
# para=0 dis_dir, para=1 dis, para=2 dir, para=3 other

def addin(row, index, classified_data, country_set):
    classified_data[index].append(row)
    country_set[index].add(row[0])
    country_set[index].add(row[1])

def data_clean_and_classify(data):
    classified_data = [[], [], [], []]
    country_set = [set(), set(), set(), set()]

    p1 = 0
    p2 = 1
    rp = 2
    dp = 3

    # 不要直接改到外面傳進來的 pre_data
    data = [row[:] for row in data]

    for row in data:
        # 讓同一國名/城名統一
        if row[p1][-1] in ['國', '城', '王']:
            row[p1] = row[p1][:-1]
        if row[p2][-1] in ['國', '城', '王']:
            row[p2] = row[p2][:-1]

    for row in data:
        # classify the data
        if row[dp] == '--':   # data without distance
            addin(row, 2, classified_data, country_set)
        else:
            tmp = row[dp][:-1]   # eliminate '里'
            # *warning* isnumeric() function also available when num = '3/4' or '二' or '\u..'
            if tmp.isnumeric():   # correct data
                row[dp] = tmp
                if row[rp] == '--':   # data without direction
                    addin(row, 1, classified_data, country_set)
                else:
                    addin(row, 0, classified_data, country_set)
            else:   # other kinds of data (e.g. 二千餘里)
                addin(row, 3, classified_data, country_set)

    disset = country_set[0].union(country_set[1])
    return classified_data, disset

def data_process(pre_data):
    c_data, disset = data_clean_and_classify(pre_data)
    return c_data, disset

# process the data from Chen Shih Liang's method
import csv
def Chen_csv_and_graph():
    # csv : 地點一 地點二 里程 里程 make it compatible to previous method
    data = [] # pouring all data into "data" without distinct book_class
    with open( "C:\\Users\\hktti\\Desktop\\project\\csv doc utf8\\漢書_陳世良_utf8.csv" , newline='', encoding='utf-8' ) as csvfile :
        data_tmp = []
        rows = csv.reader(csvfile)
        for row in rows :
            data_tmp.append(row)
        data_tmp.pop(0) # remove column name
        data = data + data_tmp
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
    return graph,vertice,dni,edges,data