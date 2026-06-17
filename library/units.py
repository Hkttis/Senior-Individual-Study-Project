from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from typing import Sequence, Any

from library.config import *

'''
inidata : Li
simulation : sim = pixel
visualization : pixel
metric : km
'''


def data_Li2sim( data ) :
    sim_data = deepcopy(data)
    for row in sim_data :
        for i in range(2, len(row)):
            row[i] = int(row[i])*Li2sim
    return sim_data

def gt_km2sim( gt_xy_km ):
    sim_gt = deepcopy(gt_xy_km)
    for i in range(len(sim_gt)) :
        x, y = sim_gt[i]
        if x is not None and y is not None :
            sim_gt[i] = [x*km2sim, y*km2sim] # turn km to sim
    return sim_gt

def pos_matrix_sim2km( pos_matrix ):
    kmpos = deepcopy(pos_matrix)
    for pos in  kmpos:
        pos[0] = pos[0] / km2sim
        pos[1] = pos[1] / km2sim
    return kmpos

def pos_matrix_pix2km( pos_matrix ):
    kmpos = deepcopy(pos_matrix)
    for pos in  kmpos:
        pos[0] = pos[0] / km2pix
        pos[1] = pos[1] / km2pix
    return kmpos
