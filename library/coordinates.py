"""library.coordinates

Coordinate helpers.

Only *screen-y-flip* lives here.
To avoid circular imports, this module does NOT import library.config at import time.
"""

from __future__ import annotations
from typing import Iterable, List, Tuple, Optional

from library.config import height as _H
height = _H

Vec2 = Tuple[float, float]

def flipping_y(pos_matrix: Iterable[Vec2], height = height) -> List[List[float]]:
    """Flip y using screen height: (x, y) -> (x, height - y).
    """
    return [[float(x), float(height) - float(y)] for (x, y) in pos_matrix]

def flipping_gt(gt: Iterable[Vec2]) -> List[List[float]]:
    for i, tup in enumerate(gt) :
        if tup[0] is None or tup[1] is None :
            continue
        else :
            gt[i] = (float(tup[0]), -float(tup[1]))
    return gt

def var_flipping_y( var : float) :
    return float(height)-float(var)