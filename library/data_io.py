import csv
from library.config import *
from library.geometry import inverse_lcc_transformation
from library.config import km2pix, km2Li

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
        pos_matrix_km.append(((pos[0]-refer_pos[0]) / km2pix,(pos[1]-refer_pos[1]) / km2pix))
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
            pos_matrix_km.append(((pos[0]-refer_pos[0]) / km2pix,(pos[1]-refer_pos[1]) / km2pix))
    wgs_pos_matrix = inverse_lcc_transformation(pos_matrix_km,ground_truth_positions[dni["鄯善"]])
    bootstrap_data = []
    countries_N = len(vertice)
    for i,pos in enumerate(wgs_pos_matrix) :
        # 每個樣本的每個節點  ( bootstrap_index, countries_index, x, y )
        bootstrap_data.append( ( int(i/countries_N) , vertice[i%countries_N], pos[0], pos[1]) )
    with open(FILE_PATHS["save_bootstrap_data"], mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerows(bootstrap_data)

def save_err_data(vertice, dni, pos_matrix, ground_truth_positions, refer_pos, errors, edge_labels):
    pos_matrix_km = []
    for pos in pos_matrix :
        pos_matrix_km.append(((pos[0]-refer_pos[0]) / km2pix,(pos[1]-refer_pos[1]) / km2pix))
    wgs_pos_matrix = inverse_lcc_transformation(pos_matrix_km,ground_truth_positions[dni["鄯善"]])
    
    err_data = []
    for i, (error_rate, (l1, l2)) in enumerate(zip(errors, edge_labels)) :
        err_data.append( (l1, wgs_pos_matrix[dni[l1]][0], wgs_pos_matrix[dni[l1]][1],
                          l2, wgs_pos_matrix[dni[l2]][0], wgs_pos_matrix[dni[l2]][1], error_rate) )
    with open(FILE_PATHS["save_err_data"], mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerows(err_data)



def loading_vis_data():
    # open saved vis_data
    # transform it into dictionary
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
    return lst

def loading_bootstrap_data(countries_N):
    # open saved bootstrap_data
    # transform it into dictionary
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
    return lst

def gt_data_dict(vertice, dni):
    ground_truth_positions = uploading_ground_truth(vertice, dni)
    gt_data = []
    for i, pos in enumerate(ground_truth_positions) :
        dict = {
            "name": vertice[i],
            "lon": pos[0],
            "lat": pos[1],
            "cluster": 2  
        }
        gt_data.append(dict)
    return gt_data

# e={start_lat, start_lon, end_lat, end_lon, err_val}
def loading_err_data():
    # open saved err_data
    # transform it into dictionary
    lst = []
    with open(FILE_PATHS["save_err_data"], newline='', encoding='utf-8') as csvfile:
        rows = csv.reader(csvfile)
        for i, row in enumerate(rows) :
            dict = {}
            dict["start_name"] = row[0]
            dict["start_lon"] = float(row[1])
            dict["start_lat"] = float(row[2])
            dict["end_name"] = row[3]
            dict["end_lon"] = float(row[4])
            dict["end_lat"] = float(row[5])
            dict["err_val"] = float(row[6])
            lst.append(dict)
    return lst





# ======================= CSV Save/Load for Position Histories =======================
import os, csv
from collections import defaultdict
from typing import List, Tuple, Optional

try:
    from library.config import FILE_PATHS
except Exception:
    FILE_PATHS = None

_REQUIRED_KEYS = [
    "save_all_pos_sm_px_data",
    "save_all_pos_dm_px_data",
    "save_all_pos_ph_px_data",
]

def _ensure_required_paths():
    if FILE_PATHS is None:
        raise RuntimeError("FILE_PATHS is not available. Ensure `from library.config import FILE_PATHS` is configured.")
    missing = [k for k in _REQUIRED_KEYS if k not in FILE_PATHS]
    if missing:
        raise KeyError(f"Missing FILE_PATHS keys: {missing}. Please define these in library.config.")

def _mkdir_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

def _coerce_xy(val) -> Tuple[float, float]:
    # Accept (x, y), [x, y], or small numpy arrays
    x, y = val[0], val[1]
    return float(x), float(y)

def _write_model_csv(path: str,
                     histories: List[List[List[Tuple[float, float]]]],
                     vertice: Optional[List[str]],
                     tag: str) -> Tuple[int, int, int]:
    """
    Write one model's histories to CSV in UTF-8-SIG.
    Returns (n_runs, total_frames, total_rows) for logging.
    """
    _mkdir_parent(path)
    n_runs = len(histories)
    total_frames = 0
    total_rows = 0
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["run", "frame", "node_idx", "label", "x_px", "y_px"])
        for run_idx, run in enumerate(histories):
            for frame_idx, frame in enumerate(run):
                total_frames += 1
                for node_idx, pos in enumerate(frame):
                    x, y = _coerce_xy(pos)
                    label = vertice[node_idx] if vertice and node_idx < len(vertice) else ""
                    w.writerow([run_idx, frame_idx, node_idx, label, x, y])
                    total_rows += 1
    print(f"[Saved CSV] {tag}: runs={n_runs}, frames(total)={total_frames}, rows={total_rows} → {path}")
    return n_runs, total_frames, total_rows

def save_all_pos_histories_px_csv(
    all_pos_hist_sm_px: List[List[List[Tuple[float, float]]]],
    all_pos_hist_dm_px: List[List[List[Tuple[float, float]]]],
    all_pos_hist_ph_px: List[List[List[Tuple[float, float]]]],
    *,
    vertice: Optional[List[str]] = None,
) -> None:
    """
    Save three models' per-run position histories in **pixel units** to CSV files.
    CSV schema: run,frame,node_idx,label,x_px,y_px  (encoding='utf-8-sig')
    """
    _ensure_required_paths()
    _write_model_csv(FILE_PATHS["save_all_pos_sm_px_data"], all_pos_hist_sm_px, vertice, "StressMajorization")
    _write_model_csv(FILE_PATHS["save_all_pos_dm_px_data"], all_pos_hist_dm_px, vertice, "DirectedMDS")
    _write_model_csv(FILE_PATHS["save_all_pos_ph_px_data"], all_pos_hist_ph_px, vertice, "PhysicsSim")

def _read_model_csv(path: str) -> Tuple[List[List[List[Tuple[float, float]]]], Optional[List[str]]]:
    """
    Read one model's histories from CSV and reconstruct to [run][frame][(x,y)].
    Returns (histories, labels) where labels is a list[str] (or None if not present).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    # temp store: run -> frame -> node_idx -> (x,y); also collect labels by node_idx
    temp = defaultdict(lambda: defaultdict(dict))
    node_labels: dict[int, str] = {}

    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        required = {"run", "frame", "node_idx", "x_px", "y_px"}
        if not required.issubset(r.fieldnames or set()):
            raise ValueError(f"CSV {path} missing required columns: {required - set(r.fieldnames or [])}")
        has_label = "label" in (r.fieldnames or [])

        for row in r:
            run = int(row["run"])
            frame = int(row["frame"])
            node = int(row["node_idx"])
            x = float(row["x_px"])
            y = float(row["y_px"])
            temp[run][frame][node] = (x, y)
            if has_label:
                lbl = row.get("label", "")
                if lbl and node not in node_labels:
                    node_labels[node] = lbl

    # Rebuild nested list in sorted order
    histories: List[List[List[Tuple[float, float]]]] = []
    for run in sorted(temp.keys()):
        frames_dict = temp[run]
        run_list: List[List[Tuple[float, float]]] = []
        for frame in sorted(frames_dict.keys()):
            node_dict = frames_dict[frame]
            if not node_dict:
                run_list.append([])
                continue
            max_node = max(node_dict.keys())
            frame_list = [None] * (max_node + 1)
            for node, xy in node_dict.items():
                frame_list[node] = xy
            # sanity: ensure no None
            if any(v is None for v in frame_list):
                raise ValueError(f"Missing node positions in run={run}, frame={frame} reading {path}.")
            run_list.append(frame_list)
        histories.append(run_list)

    labels = None
    if node_labels:
        # Convert to dense list by node index order
        max_node = max(node_labels.keys())
        labels = [node_labels.get(i, "") for i in range(max_node + 1)]

    print(f"[Loaded CSV] runs={len(histories)}, frames(total)={sum(len(r) for r in histories)} ← {path}")
    return histories, labels

def load_all_pos_histories_px_csv(return_labels: bool = False):
    """
    Load the three models' histories from CSV (pixel units).
    Returns
    -------
    (sm_hist, dm_hist, ph_hist)                    if return_labels=False
    (sm_hist, dm_hist, ph_hist, labels)            if return_labels=True and labels available
    Notes
    -----
    - If labels differ across files, the first non-empty label list is returned.
    """
    _ensure_required_paths()
    p_sm = FILE_PATHS["save_all_pos_sm_px_data"]
    p_dm = FILE_PATHS["save_all_pos_dm_px_data"]
    p_ph = FILE_PATHS["save_all_pos_ph_px_data"]

    sm_hist, sm_labels = _read_model_csv(p_sm)
    dm_hist, dm_labels = _read_model_csv(p_dm)
    ph_hist, ph_labels = _read_model_csv(p_ph)

    labels = sm_labels or dm_labels or ph_labels
    return (sm_hist, dm_hist, ph_hist, labels) if return_labels else (sm_hist, dm_hist, ph_hist)


import csv
import os
from typing import List, Dict, Tuple

EdgeRow     = List[str]                 # ["src","dst","w1","w2"]
GraphGroup  = List[EdgeRow]             # group of edges
GraphType   = List[GraphGroup]          # list of groups
VertType    = List[str]                 # list of vertex names
DniType     = Dict[str, int]            # name -> index
EdgesSimple = List[Tuple[str, str]]     # list of (src, dst)
DataType    = List[EdgeRow]             # flat list [["src","dst","w1","w2"], ...]


def save_ini_data_to_csv(
    FILE_PATHS: Dict[str, str],
    graph: GraphType,
    vertice: VertType,
    dni: DniType,
    edges: EdgesSimple,
    data: DataType,
) -> None:
    """
    Save graph, vertice, dni, edges, data into ONE CSV at FILE_PATHS["ini_data"].
    Encoding: utf-8-sig. Sections are written strictly in this order:
        GRAPH -> VERTICE -> DNI -> EDGES -> DATA
    """
    out_path = FILE_PATHS["ini_data"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Fixed wide schema; leave unused cells empty to keep CSV simple.
    fieldnames = ["section", "group_id", "name", "index", "src", "dst", "w1", "w2"]

    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        # 1) GRAPH (keep group order; keep edge order inside each group)
        #    We record group_id to reconstruct the nested structure.
        for gid, group in enumerate(graph):
            for row in group:
                src, dst, w1, w2 = row
                w.writerow({
                    "section": "GRAPH",
                    "group_id": gid,
                    "name": "", "index": "",
                    "src": src, "dst": dst, "w1": w1, "w2": w2
                })

        # 2) VERTICE (preserve exact order as provided)
        for name in vertice:
            w.writerow({
                "section": "VERTICE",
                "group_id": "",
                "name": name, "index": "",
                "src": "", "dst": "", "w1": "", "w2": ""
            })

        # 3) DNI (authoritative mapping, write exactly as provided; order preserved)
        for name, idx in dni.items():
            w.writerow({
                "section": "DNI",
                "group_id": "",
                "name": name, "index": idx,
                "src": "", "dst": "", "w1": "", "w2": ""
            })

        # 4) EDGES (verbatim)
        for (src, dst) in edges:
            w.writerow({
                "section": "EDGES",
                "group_id": "",
                "name": "", "index": "",
                "src": src, "dst": dst, "w1": "", "w2": ""
            })

        # 5) DATA (verbatim; this is a single flat list, NOT grouped)
        for src, dst, w1, w2 in data:
            w.writerow({
                "section": "DATA",
                "group_id": "",
                "name": "", "index": "",
                "src": src, "dst": dst, "w1": w1, "w2": w2
            })


def load_ini_data_from_csv(
    FILE_PATHS: Dict[str, str],
) -> Tuple[GraphType, VertType, DniType, EdgesSimple, DataType]:
    """
    Load CSV at FILE_PATHS["ini_data"] and reconstruct:
        graph, vertice, dni, edges, data
    Exactness guarantees:
      - DATA rows are returned exactly as stored (order & strings).
      - EDGES rows are returned exactly as stored.
      - VERTICE list preserves file order (which equals the original order we wrote).
      - DNI mapping is taken exactly from the file rows.
      - GRAPH groups are rebuilt by ascending group_id, preserving edge order per group.
    """
    in_path = FILE_PATHS["ini_data"]

    graph_groups: Dict[int, GraphGroup] = {}
    vertice: VertType = []
    dni: DniType = {}
    edges: EdgesSimple = []
    data: DataType = []

    with open(in_path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            section = row.get("section", "")
            # DO NOT strip(), to preserve any user-supplied spaces inside names.

            if section == "GRAPH":
                gid_raw = row.get("group_id", "")
                try:
                    gid = int(gid_raw)
                except Exception:
                    # If a GRAPH row lacks a valid gid, skip it rather than corrupting structure
                    continue
                src = row.get("src", ""); dst = row.get("dst", "")
                w1  = row.get("w1",  ""); w2  = row.get("w2",  "")
                graph_groups.setdefault(gid, []).append([src, dst, w1, w2])

            elif section == "VERTICE":
                name = row.get("name", "")
                vertice.append(name)

            elif section == "DNI":
                name = row.get("name", "")
                idx  = row.get("index", "")
                try:
                    dni[name] = int(idx)
                except Exception:
                    # keep as-is if malformed? safer to skip bad rows
                    continue

            elif section == "EDGES":
                src = row.get("src", ""); dst = row.get("dst", "")
                edges.append((src, dst))

            elif section == "DATA":
                src = row.get("src", ""); dst = row.get("dst", "")
                w1  = row.get("w1",  ""); w2  = row.get("w2",  "")
                data.append([src, dst, w1, w2])

            else:
                # Unknown section—ignore silently
                pass

    # Rebuild graph as a list ordered by group_id
    graph: GraphType = [graph_groups[k] for k in sorted(graph_groups.keys())]

    return graph, vertice, dni, edges, data




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