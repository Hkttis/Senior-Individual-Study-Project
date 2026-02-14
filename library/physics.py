import pygame
import pymunk
import pymunk.pygame_util
import numpy as np
import math
from math import sqrt
from copy import deepcopy

from library.config import *
from library.visualization import plotting_physics_simulation
from library.metrics import stress_function
from library.units import pos_matrix_sim2km
from library.directions import DIR4_SIM, DIR4DIAG_RAW_SIM, DIR8_UNIT_SIM


def main_physics_simulation(vertice,dni,data,pos_matrix,directional_data,fixed_positions_list,
                            spring_stiffness,repulsion_strength,directional_force_magnitude, plot = False):
    n = len(vertice)
    
    mass = MASS_BASE
    fixmass = FIXMASS_BASE
    radius = RADIUS_BASE
    vrange = VRANGE_BASE
    spring_damping = SPRING_DAMPING_BASE
    min_distance = MIN_DISTANCE_BASE
    resistance = RESISTANCE_BASE

    pygame.init()
    screen = pygame.display.set_mode((1200, 750))
    space = pymunk.Space()
    pymunk.pygame_util.positive_y_is_up = True
    draw_options = pymunk.pygame_util.DrawOptions(screen)
    font = pygame.font.SysFont("Microsoft YaHei", 12)

    nodes,space = create_nodes_and_springs(
        n,mass,radius,vrange,spring_stiffness,spring_damping,fixmass,
        space,data, dni,pos_matrix,fixed_positions_list
    )

    wrong_direction_lists,stress_history,pos_history, pos_matrix = run_physics_simulation(
        min_distance,repulsion_strength,resistance,directional_force_magnitude,
        screen,space,draw_options,font,nodes,directional_data,data,vertice,dni,pos_matrix, plot
    )

    pygame.display.quit()
    pygame.quit()

    return wrong_direction_lists,stress_history,pos_history,pos_matrix


def create_nodes_and_springs(n,mass,radius,vrange,spring_stiffness,spring_damping,fixmass,
                             space,data, dni,pos_matrix,fixed_positions_list):
    """Create pymunk bodies/springs and anchor constraints.

    Contract:
      - pos_matrix is in SIM coordinates (currently pixel-like).
      - Fixed points are enforced via constraints (PivotJoint to space.static_body).
      - No per-step overwriting of anchor positions.
    """
    # 1) bodies + shapes
    nodes = [pymunk.Body(mass, pymunk.moment_for_circle(mass, 0, radius)) for _ in range(n)]

    for i in range(n):
        body = nodes[i]
        body.position = (float(pos_matrix[i][0]), float(pos_matrix[i][1]))
        body.velocity = (0, 0)
        shape = pymunk.Circle(body, radius)
        shape.filter = pymunk.ShapeFilter(group=1)
        space.add(body, shape)

    # 2) anchor constraints (J^T lambda from solver)
    for row in fixed_positions_list:
        label = row[0]
        if label not in dni:
            continue
        idx = dni[label]
        body = nodes[idx]
        pivot = pymunk.PivotJoint(space.static_body, body, body.position)
        pivot.max_force = 1e9
        pivot.max_bias = 1e7
        space.add(pivot)

    for row in data:
        i = dni[row[0]]
        j = dni[row[1]]
        rest_length = float(row[2])
        spring = pymunk.DampedSpring(nodes[i], nodes[j], (0, 0), (0, 0),
                                     rest_length, spring_stiffness, spring_damping)
        space.add(spring)

    return nodes, space


def run_physics_simulation(min_distance,repulsion_strength,resistance,directional_force_magnitude,
                           screen,space,draw_options,font,nodes,directional_data,data,vertice,dni,pos_matrix,plot):
    clock = pygame.time.Clock()
    iteration = 0
    stress_history = []
    pos_history = []
    while True:
        iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return [], stress_history, pos_history, pos_matrix

        space.step(0.01)
        clock.tick(60)

        nodes,cnt,wrong_direction_lists = apply_forces(
            min_distance,repulsion_strength,resistance,directional_force_magnitude,
            nodes,directional_data, dni
        )

        for i,node in enumerate(nodes):
            pos_matrix[i] = nodes[i].position
        pos_history.append(deepcopy(pos_matrix))

        current_stress = stress_function(data,dni,pos_matrix_sim2km(pos_matrix))

        if plot:
            screen,space = plotting_physics_simulation(
                screen,space,draw_options,font,nodes,data,vertice,dni, pos_matrix,
                cnt,wrong_direction_lists,current_stress
            )

        stress_history.append(current_stress)
        if iteration > stop_physim_iteration_time:
            break


    return wrong_direction_lists,stress_history,pos_history, pos_matrix


def apply_forces(min_distance,repulsion_strength,resistance,directional_force_magnitude,nodes,directional_data, dni):
    """Apply forces:
      1) Repulsion
      2) Directional force = squared hinge loss gradient (paper-aligned)
      3) Linear resistance

    Directional hinge (for each constraint u->v with desired direction v_dir):
      z = cos(theta_h) - <r_hat, v_dir>
      if z>0:
        F_u = w_dir * z * ((I - r_hat r_hat^T) v_dir) / (||r|| + eps)
        F_v = -F_u
    """
    # --- 1) repulsive force ---
    for i, node_a in enumerate(nodes):
        for j, node_b in enumerate(nodes):
            if i >= j:
                continue
            dx = node_b.position.x - node_a.position.x
            dy = node_b.position.y - node_a.position.y
            distance = (dx * dx + dy * dy) ** 0.5 + min_distance
            force_magnitude = repulsion_strength / (distance ** 1)
            fx = force_magnitude * dx / distance
            fy = force_magnitude * dy / distance
            node_a.apply_force_at_world_point((-fx, -fy), node_a.position)
            node_b.apply_force_at_world_point((fx, fy), node_b.position)

    # --- 2) directional hinge force (Updated: Angular Hinge) ---
    # 定義容忍角度 (以 radians 為單位)
    theta_h_4 = theta_thr_4dir
    theta_h_8 = theta_thr_8dir
    eps = 1e-9

    cnt = 0
    wrong_direction_lists = []

    for row in directional_data:
        if len(row) < 3:
            continue
        u_name, v_name, d_name = row[0], row[1], row[2].strip()
        if u_name not in dni or v_name not in dni:
            continue
        if d_name not in DIR8_UNIT_SIM:
            continue

        node_u = nodes[dni[u_name]]
        node_v = nodes[dni[v_name]]

        # 計算邊向量 r = v - u
        r = np.array([node_v.position.x - node_u.position.x,
                      node_v.position.y - node_u.position.y], dtype=float)
        dist = float(np.linalg.norm(r))
        
        # 避免除以零
        if dist < eps:
            continue

        r_hat = r / dist
        v_dir = np.array(DIR8_UNIT_SIM[d_name], dtype=float)

        # --- 核心修改開始 ---
        
        # 1. 計算精確角度 (使用 atan2)
        # dot = cos(phi), cross = sin(phi)
        d_val = float(np.dot(r_hat, v_dir))
        c_val = float(r_hat[0]*v_dir[1] - r_hat[1]*v_dir[0]) # 2D cross product
        phi = math.atan2(c_val, d_val)  # 範圍 (-pi, pi]

        # 2. 判斷是否違規
        current_theta_h = theta_h_4 if d_name in DIR4_SIM else theta_h_8
        violation = abs(phi) - current_theta_h # 違規量 z
        

        if violation <= 0:
            continue # 在容忍範圍內，無力

        # 3. 記錄違規
        cnt += 1
        wrong_direction_lists.append(row)

        # --- 4. 計算梯度力 (Corrected based on Math Derivation) ---
        # 根據推導，施加在 u (尾端) 的力方向，應沿著 r 的「順時針切線」方向與 sgn(phi) 的乘積。
        # t_cw (單位順時針切線) = (r_y, -r_x) / dist
        
        sgn = 1.0 if phi >= 0 else -1.0
        # 定義 r 的順時針單位切線向量
        # r = (r[0], r[1]) -> cw_tangent = (r[1], -r[0])
        tangent_cw_x = r[1]/dist
        tangent_cw_y = -r[0]/dist
        
        coeff = (directional_force_magnitude * violation * sgn) /dist
        
        Fx_u = coeff * tangent_cw_x
        Fy_u = coeff * tangent_cw_y
        
        # --- 5. 施加力 (Newton's 3rd Law) ---
        # F_u: 施加在起點 (Tail) 的力 -> 根據推導，這就是我們算出的正向力
        # F_v: 施加在終點 (Head) 的力 -> 反作用力
        
        node_u.apply_force_at_world_point((Fx_u, Fy_u), node_u.position)
        node_v.apply_force_at_world_point((-Fx_u, -Fy_u), node_v.position)

    # --- 3) resistance ---
    for node in nodes:
        vx, vy = node.velocity
        node.apply_force_at_world_point((-resistance * vx, -resistance * vy), node.position)

    return nodes, cnt, wrong_direction_lists

