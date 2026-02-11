# library/anchor_frame.py
from typing import Iterable, List, Tuple
from library.directions import SIM_Y_IS_UP

# If your sim/screen y is DOWN (SIM_Y_IS_UP=False), then to convert to north-up km:
# y_km = -(y_px - ref_y) / km2pix
_Y_SIGN_TO_NORTH_UP = 1.0 if SIM_Y_IS_UP else -1.0

def px_to_anchor_px(p: Tuple[float, float], refer_pos: Tuple[float, float]) -> Tuple[float, float]:
    return (p[0] - refer_pos[0], p[1] - refer_pos[1])

def anchor_px_to_px(p_anchor: Tuple[float, float], refer_pos: Tuple[float, float]) -> Tuple[float, float]:
    return (p_anchor[0] + refer_pos[0], p_anchor[1] + refer_pos[1])

def px_to_km(p_px: Tuple[float, float], refer_pos: Tuple[float, float], km2pix: float) -> Tuple[float, float]:
    ax, ay = px_to_anchor_px(p_px, refer_pos)
    return (ax / km2pix, _Y_SIGN_TO_NORTH_UP * ay / km2pix)

def km_to_px(p_km: Tuple[float, float], refer_pos: Tuple[float, float], km2pix: float) -> Tuple[float, float]:
    # inverse of px_to_km
    x_px = p_km[0] * km2pix + refer_pos[0]
    y_px = (p_km[1] / _Y_SIGN_TO_NORTH_UP) * km2pix + refer_pos[1]
    return (x_px, y_px)

def px_list_to_km_list(pos_px: Iterable[Tuple[float, float]], refer_pos: Tuple[float, float], km2pix: float) -> List[Tuple[float, float]]:
    return [px_to_km((float(x), float(y)), refer_pos, km2pix) for (x, y) in pos_px]

def km_list_to_px_list(pos_km: Iterable[Tuple[float, float]], refer_pos: Tuple[float, float], km2pix: float) -> List[Tuple[float, float]]:
    return [km_to_px((float(x), float(y)), refer_pos, km2pix) for (x, y) in pos_km]
