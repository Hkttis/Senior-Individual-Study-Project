# for CONFIGs

# using dictionary to modularize file paths
FILE_PATHS = {
    "chen_data": "C:/Users/justi/Desktop/project/csv doc utf8/漢書_陳世良_utf8.csv",
    "directional_data": "C:/Users/justi/Desktop/project/csv doc utf8/方向.csv",
    "classification_data": "C:/Users/justi/Desktop/project/csv doc utf8/國家分類.csv",
    "output_csv": "cities_pos_try3.csv",
    "font_path": "C:/Windows/Fonts/msyh.ttc",
    "ground_truth_path" : "C:/Users/justi/Desktop/project/csv doc utf8/西漢古城地理位置資訊.csv"
}

# default force parameters (for bootstrap & normal runs)
SPRING_STIFFNESS_BASE   = 15000
REPULSION_STRENGTH_BASE = 5000
DIRECTIONAL_FORCE_MAGNITUDE_BASE = 10000

MASS_BASE = 10
FIXMASS_BASE = 1e7
RADIUS_BASE = 5
VRANGE_BASE = 100
SPRING_DAMPING_BASE = 50
MIN_DISTANCE_BASE = 0.1
RESISTANCE_BASE = 10

