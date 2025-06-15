# interactive_map/tiles.py
import folium
import glob, os
import json, os
from folium import FeatureGroup, LayerControl
from folium.features import GeoJsonTooltip
from folium.raster_layers import ImageOverlay

# ---- 你現有的計算結果 -------------------
#from library.data_io import load_visualization_data
# ↑==> 請在 data_io.py 自行加一個函式，把 pos_matrix、edges、errors
#       dump 成 json / pickle，再用這裡的 loader 讀回。

def add_base_tiles(m, *, cntr_lat, cntr_lon, zoom_start,
                   hist_img_path, hist_img_bounds):
    """
    Parameters
    ----------
    m : folium.Map
        The target map object.
    cntr_lat, cntr_lon : float
        Map centre in WGS84.
    zoom_start : int
        Initial zoom level.
    hist_img_path : str
        Path to the scanned historical map (png / jpg, already georegistered).
    hist_img_bounds : tuple
        ((south, west), (north, east))  in lat/lon.
    """
    # (A) OpenStreetMap – default
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap",
        control=True,
        show=True,        # shown at start-up
        attr="© OSM contributors").add_to(m)

    # (B) 自行掃描的「中國歷史地圖集」影像疊加
    ImageOverlay(
        name="西漢西域歷史底圖",
        image=hist_img_path,
        bounds=hist_img_bounds,
        opacity=0.55,     # 半透明
        interactive=False,  # 只當底圖用
        cross_origin=False).add_to(m)

    # LayerControl 會在 map_app.py 統一新增
    return m

def png_layers_from_directory(directory, *, bounds, transparent=False):
    """
    Scan a directory, create a Folium ImageOverlay for every PNG.

    Parameters
    ----------
    directory : str
        Folder containing exported figures (e.g. stress_convergence_log.png ...).
    bounds : tuple
        Same as tiles.py ((south, west), (north, east)).
    transparent : bool
        If your PNG already has alpha channel set, keep it True.
    """
    layer_list = []
    for f in sorted(glob.glob(os.path.join(directory, "*.png"))):
        name = os.path.splitext(os.path.basename(f))[0]
        layer = ImageOverlay(
            name=name,
            image=f,
            bounds=bounds,
            opacity=0.9 if transparent else 1,
            interactive=False)
        layer_list.append(layer)
    return layer_list


