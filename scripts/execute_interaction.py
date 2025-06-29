import json, os
import folium
from folium import FeatureGroup, LayerControl
from folium.features import GeoJsonTooltip
from folium.raster_layers import ImageOverlay

# ---- 你現有的計算結果 -------------------
#from library.data_io import load_visualization_data

# ---- 子模組 -----------------------------
from library.interaction import *
from library.data_io import *

# === 使用者可自行修改 =====================
CENTER_LAT = 41.0    # roughly in Tarim Basin
CENTER_LON = 87.0
ZOOM_START = 6
RESULT_DIR = r"C:/Users/justi/Desktop/project/results"

HIST_MAP_PATH   = r"C:/Users/justi/Desktop/project/project_refer/westHan_WR_gt.jpg"
HIST_MAP_BOUNDS = ((35.0, 75.0), (46.0, 99.0))   # south/west, north/east
# =========================================

def _node_color(cluster_id):
    """Return color code by cluster id (you will supply cluster mapping)."""
    palette = ["red", "green", "blue", "orange", "purple", "brown"]
    return palette[cluster_id % len(palette)]

def build_interactive_map():
    
    data = {}
    '''
    data = {
        "nodes": [dict],
            dict = {name, lon, lat, cluster},
        "bootstrap_nodes": [dict],
            dict = {name, lon, lat, cluster},
        "gt_data": [dict],
            dict = {name, lon, lat, cluster},
        "edges": [dict],
            dict = {start_name, start_lon, start_lat, end_name, end_lon, end_lat, err_val},
    }
    '''
    
    # 0) 讀入前處理後的可視化數據 ------------------
    data["nodes"] = loading_vis_data() 
    visdata_vertice = []
    visdata_dni = {}
    for i,n in enumerate(data["nodes"]):
        visdata_vertice.append(n["name"])
        visdata_dni[n["name"]] = i
    
    # 1) 基底地圖 -------------------------------
    m = folium.Map(location=[CENTER_LAT, CENTER_LON],
                   zoom_start=ZOOM_START,
                   control_scale=True,
                   tiles=None)  # we add tiles in tiles.py
    
    add_base_tiles(m,
                   cntr_lat=CENTER_LAT, cntr_lon=CENTER_LON,
                   zoom_start=ZOOM_START,
                   hist_img_path=HIST_MAP_PATH,
                   hist_img_bounds=HIST_MAP_BOUNDS)
    
    
    
    # 2) 節點圖層 --------------------------------
    fg_nodes = FeatureGroup(name="Country Nodes", show=True)

    for n in data["nodes"]:
        folium.CircleMarker(
            location=[n["lat"], n["lon"]],
            radius=5,
            color=_node_color(n["cluster"]),
            fill=True,
            fill_opacity=0.9,
            tooltip=folium.Tooltip(f"<div style='font-size:16px; font-weight:bold'>{n['name']}</div>")  # hover 顯示國名
        ).add_to(fg_nodes)
    fg_nodes.add_to(m)
    
    # 2.1) bootstrap 節點圖層 --------------------------------
    #TODO : add mean positions, labels for bootstrap_data
    data["bootstrap_nodes"] = loading_bootstrap_data(len(data["nodes"]))  # Load bootstrap data for interactive visualization
    fg_nodes = FeatureGroup(name="bootstrap Nodes", show=True)

    for n in data["bootstrap_nodes"]:
        folium.CircleMarker(
            location=[n["lat"], n["lon"]],
            radius=2,
            color=_node_color(n["cluster"]),
            fill=True,
            fill_opacity=0.5,
        ).add_to(fg_nodes)
    fg_nodes.add_to(m)
    
    # 2.5) ground truth 點圖層 --------------------------------
    data["gt_data"] = gt_data_dict(visdata_vertice, visdata_dni)
    fg_nodes = FeatureGroup(name="Ground Truth Positions", show=True)

    for n in data["gt_data"]:
        folium.CircleMarker(
            location=[n["lat"], n["lon"]],
            radius=5,
            color=_node_color(n["cluster"]),
            fill=True,
            fill_opacity=0.9,
            tooltip=folium.Tooltip(f"<div style='font-size:16px; font-weight:bold'>{n['name']}</div>")  # hover 顯示國名
        ).add_to(fg_nodes)
    fg_nodes.add_to(m)
    
    
    data["edges"] = loading_err_data()  # Load error data for edges
    # 3) 邊與誤差標註 (Edge layer) -----------------
    fg_edges = FeatureGroup(name="Distance Errors", show=False)
    for e in data["edges"]:
        coords = [(e["start_lat"], e["start_lon"]),
                  (e["end_lat"],   e["end_lon"])]
        error_clipped = min(max(e["err_val"], 0), 0.03)
        color_val = int(255 * (1 - error_clipped / 0.03))
        folium.PolyLine(
            locations=coords,
            color=f"#{255:02x}{color_val:02x}{color_val :02x}",
            weight=4.5,
            tooltip=folium.Tooltip(f"<div style='font-size:14px'>{e['err_val']*100:.1f}%</div>")
        ).add_to(fg_edges)
    fg_edges.add_to(m)
    
    '''
    # 4) 把你的 PNG 匯出圖自動掛成「可切換」圖層 -----
    img_layers = png_layers_from_directory(
        RESULT_DIR,
        bounds=HIST_MAP_BOUNDS,
        transparent=True)
    for layer in img_layers:
        layer.add_to(m)
    '''
    '''
    # 設定 compass 位置（右上角）
    compass_img = r"C:/Users/justi/Desktop/project/project_refer/compass_rose.png"
    compass_bounds = ((44.5, 97.5), (46, 99))  # ((south, west), (north, east))

    compass_overlay = ImageOverlay(
        name='Compass Rose',
        image=compass_img,
        bounds=compass_bounds,
        opacity=1,
        interactive=False,
        zindex=1000
    )
    compass_overlay.add_to(m)
    '''
    
    
    # 5) Layer control ---------------------------
    LayerControl(collapsed=False).add_to(m)
    

    # 6) Save to html ----------------------------
    out_html = os.path.join(RESULT_DIR, "interactive_map.html")
    m.save(out_html)
    print(f"==> interactive map saved to {out_html}")


if __name__ == "__main__":
    build_interactive_map()