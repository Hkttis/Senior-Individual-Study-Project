"""library.config

Project-level configuration.

Reproducibility rules (pragmatic):
- Prefer keeping only constants and simple helpers here.
- Avoid importing *other* project modules here to prevent circular imports.

Notes
-----
This project historically relied on many modules doing `from library.config import *`.
To keep backward compatibility while removing circular imports, we keep key constants
here and (optionally) re-export unit conversion ratios from `library.units`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict
import math

# --- Project paths --------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# --- File paths (user-local defaults + relative fallback) -----------------
FILE_PATHS: Dict[str, str] = {
    "chen_data": "C:/Users/hktti/Desktop/project/csv doc utf8/漢書_陳世良_utf8.csv",
    "directional_data": "C:/Users/hktti/Desktop/project/csv doc utf8/方向.csv",
    "classification_data": "C:/Users/hktti/Desktop/project/csv doc utf8/國家分類.csv",
    "output_csv": "cities_pos_try3.csv",
    "font_path": "C:/Windows/Fonts/msyh.ttc",
    "ground_truth_path": "C:/Users/hktti/Desktop/project/csv doc utf8/西漢古城地理位置資訊.csv",
    "visualization_data": "C:/Users/hktti/Desktop/project/results/visualization_data.json",
    "save_vis_data": "C:/Users/hktti/Desktop/project/results_data/vis_data.csv",
    "save_bootstrap_data": "C:/Users/hktti/Desktop/project/results_data/bootstrap_data.csv",
    "save_err_data": "C:/Users/hktti/Desktop/project/results_data/err_data.csv",
    "save_all_pos_sm_px_data": "C:/Users/hktti/Desktop/project/results_data/all_pos_sm_px_data.csv",
    "save_all_pos_dm_px_data": "C:/Users/hktti/Desktop/project/results_data/all_pos_dm_px_data.csv",
    "save_all_pos_ph_px_data": "C:/Users/hktti/Desktop/project/results_data/all_pos_sm_ph_data.csv",
    "ini_data": "C:/Users/hktti/Desktop/project/results_data/ini_data.csv",
}


def resolve_path(key: str) -> str:
    """Return an existing path for FILE_PATHS[key].

    1) If the configured path exists, use it.
    2) Otherwise, try DATA_DIR/<basename>.
    """
    p = Path(FILE_PATHS[key])
    if p.exists():
        return str(p)
    fallback = DATA_DIR / p.name
    return str(fallback)


# --- Window / screen ------------------------------------------------------
width: int = 1200
height: int = 750

# A *screen* anchor used in many legacy utilities. Keep it here for compatibility.
refer_pos = [600,500]
refer_pos_screen = refer_pos
refer_pos_sim = [refer_pos[0], height - refer_pos[1]]



# --- Simulation hyperparameters ------------------------------------------
# Naming: W_dis (spring), W_rep (repulsion), W_dir (directional)
SPRING_STIFFNESS_BASE: float = 1500 #0.0
DIRECTIONAL_FORCE_MAGNITUDE_BASE: float = 10000.0*100
REPULSION_STRENGTH_BASE: float = 5000.0*0.1


# Body / integration parameters
MASS_BASE: float = 10.0
FIXMASS_BASE: float = 1e7
RADIUS_BASE: float = 5.0
VRANGE_BASE: float = 1000.0
SPRING_DAMPING_BASE: float = 50.0
MIN_DISTANCE_BASE: float = 0.1
RESISTANCE_BASE: float = 10.0

# Stop criteria (physics simulation main loop)
stop_physim_iteration_time: int = 1000

theta_thr_4dir = math.pi/2
theta_thr_8dir = math.pi/4


Li2sim = 1/10
Li2pix = 1/10
Li2km = 0.415

sim2pix = 1

km2Li = 1/0.415 # 1 Li = 0.415 km
km2pix = 1/(10 * 0.415) # 1 Li = 0.415 km, 1 pixel = 10 Li
km2sim = 1/(10 * 0.415)
