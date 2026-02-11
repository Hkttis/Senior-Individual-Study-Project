# library/directions.py
import numpy as np

# =========================================================
# Single source of truth for "direction vectors in SIM space"
#
# SIM_Y_IS_UP = False  -> y increases downward (pygame-like)
# SIM_Y_IS_UP = True   -> y increases upward (north-up)
# =========================================================
SIM_Y_IS_UP = True

_INV = -1 if SIM_Y_IS_UP else 1   # helper: when y-down, "north" is -y; when y-up, "north" is +y

# Cardinal (unit)
DIR4_SIM = {
    "東": np.array([ 1.0, 0.0]),
    "西": np.array([-1.0, 0.0]),
    "北": np.array([ 0.0, -1.0]) if not SIM_Y_IS_UP else np.array([0.0, 1.0]),
    "南": np.array([ 0.0,  1.0]) if not SIM_Y_IS_UP else np.array([0.0,-1.0]),
}

# Diagonal (raw, length=sqrt(2)) — keep this to match your existing physics.py logic
DIR4DIAG_RAW_SIM = {
    # y-down: 東南=(+,+), 東北=(+,-), 西北=(-,-), 西南=(-,+)
    # y-up  : 東北=(+,+), 東南=(+,-), 西北=(-,+), 西南=(-,-)
    "東南": np.array([ 1.0,  1.0]) if not SIM_Y_IS_UP else np.array([ 1.0, -1.0]),
    "西北": np.array([-1.0, -1.0]) if not SIM_Y_IS_UP else np.array([-1.0,  1.0]),
    "東北": np.array([ 1.0, -1.0]) if not SIM_Y_IS_UP else np.array([ 1.0,  1.0]),
    "西南": np.array([-1.0,  1.0]) if not SIM_Y_IS_UP else np.array([-1.0, -1.0]),
}

# Full 8-direction (unit) — for directed_mds_model and any cosine/dot computations
DIR8_UNIT_SIM = {}
DIR8_UNIT_SIM.update(DIR4_SIM)
for k, v in DIR4DIAG_RAW_SIM.items():
    DIR8_UNIT_SIM[k] = v / np.sqrt(2.0)
