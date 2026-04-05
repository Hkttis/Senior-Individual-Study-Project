from library.data_io import read_CHEN_csvfile, save_ini_data_to_csv
from library.initialization import construct_Chen_graph
from library.config import FILE_PATHS

def main():
    data = read_CHEN_csvfile()
    graph, vertice, dni, edges = construct_Chen_graph(data)
    save_ini_data_to_csv(FILE_PATHS, graph, vertice, dni, edges, data)
    print("Rebuilt ini_data.csv from chen_data successfully.")

if __name__ == "__main__":
    main()