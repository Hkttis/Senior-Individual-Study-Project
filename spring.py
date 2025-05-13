import pymunk
import pymunk.pygame_util
import pygame
import os
from pyproj import Geod, CRS, Transformer

from math import *
import math
import numpy as np
import csv
# temporarily stop the plotting_physics_simulation function in run_phy... and stress_history
# and plot_stress_convergence_log in post_procession


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

def generate_CHEN_initial_positions (refer_pos): # Initialize position of points from CHEN_STRESSMAJORIZATION/random positions
    data = read_CHEN_csvfile()
    graph,vertice,dni,edges= construct_Chen_graph(data)
    n = len(vertice)
    # initialize random positions in pos_matrix
    pos_matrix = np.column_stack((
        np.random.uniform(120, 1080, size=n),  # First column : canva 0~1200
        np.random.uniform(75, 675, size=n)   # Second column : canva 0~750
    ))
    ## pos_matrix = stress_majorization(graph,dni,vertice,edges) # inherit position form previous model
    center_pos = [600,375]
    pos_matrix = shift(pos_matrix,2,center_pos)
    ground_truth_positions = uploading_ground_truth(vertice,dni)
    pos_matrix,fixed_positions_list = uploading_fixed_positions(dni,pos_matrix,refer_pos, ground_truth_positions)
    ## pos_matrix = pre_physics_simulation(pos_matrix,dni) #pre_PS ensures the accuracy of pos of fixed points 
    return vertice,dni,data,pos_matrix,fixed_positions_list
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
def construct_Chen_graph(data):
    countryset = set()
    for row in data :
        countryset.add(row[0])
        countryset.add(row[1])
    vertice = []
    dni = {}
    edges = []
    for coun in countryset :
        dni[coun] = len(vertice)
        vertice.append(coun)
    graph = [[] for i in range(len(vertice))]
    for row in data :
        edges.append((row[0],row[1]))
        graph[dni[row[0]]].append(row)
        graph[dni[row[1]]].append([row[1]]+[row[0]]+[row[2]]+[row[3]])
    return graph,vertice,dni,edges
def shift(pos_matrix,scale,center_pos) : # Shift the average of points to the center_pos
    sumx = 0
    sumy = 0
    for pair in pos_matrix :
        sumx += pair[0]
        sumy += pair[1]
    sumx = sumx/len(pos_matrix)
    sumy = sumy/len(pos_matrix)
    for pair in pos_matrix :
        pair[0] = float(( pair[0] - sumx ) / scale + center_pos[0])
        pair[1] = float(( pair[1] - sumy ) / scale + center_pos[1])
    return pos_matrix
def pre_physics_simulation(pos_matrix,dni) : # Initialize the precise positions of three fixed points through prePS
    
    '''set up simulation'''
    pygame.init()
    prescreen = pygame.display.set_mode((1200, 750))
    prespace = pymunk.Space()
    pre_draw_options = pymunk.pygame_util.DrawOptions(prescreen)
    font = pygame.font.SysFont("Microsoft YaHei", 20)
    
    '''set up body and springs in PS'''
    preset = ["鄯善","都護治/烏壘",'車師後']
    prebody = []
    for i in range(3) :
        body = pymunk.Body(5,pymunk.moment_for_circle(5, 0, 5))
        body.position = (pos_matrix[dni[preset[i]]][0], pos_matrix[dni[preset[i]]][1])
        prebody.append(body)
        prespace.add(prebody[i],pymunk.Circle(prebody[i], 5))

    spring1 = pymunk.DampedSpring(
            prebody[0], prebody[1], (0, 0), (0, 0), 1906/10, 1000, 500
        )
    spring2 = pymunk.DampedSpring(
            prebody[0], prebody[2], (0, 0), (0, 0), 2324/10, 1000, 500
        )
    spring3 = pymunk.DampedSpring(
            prebody[1], prebody[2], (0, 0), (0, 0), 1237/10, 1000, 500
        )
    prespace.add(spring1)
    prespace.add(spring2)
    prespace.add(spring3)
    
    '''Running (displaying) the simulation and updating the pos of three fixed points'''
    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        prescreen.fill((255, 255, 255))
        prespace.step(0.02)
        prespace.debug_draw(pre_draw_options)
        for i in range(3):
            label = preset[i]
            text_surface = font.render(label, True, (0, 0, 0))
            prescreen.blit(text_surface, (prebody[i].position[0] - 10, prebody[i].position[1] - 10))
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()
    for i in range(3) :
        pos = prebody[i].position
        index = dni[preset[i]]
        pos_matrix[index] = pos
    return pos_matrix
def uploading_fixed_positions(dni, pos_matrix, refer_pos, ground_truth_positions):
    """
    使用 lcc_transformation 進行 LCC 投影，並將「鄯善」與「都護治/烏壘...」對齊到畫布上的固定位置。
    回傳:
      - 更新後的 pos_matrix
      - 固定點列表 fixed_positions_list = [
            ['鄯善', x_pixel, y_pixel],
            ['都護治/烏壘', x_pixel, y_pixel]
        ]
    """
    FIXPOINTS = ['都護治/烏壘']
    
    gt_xy_km = lcc_transformation(dni, ground_truth_positions)
    km2pix = 1.0 / (10 * 0.415)
    fixed_positions_list = []
    
    # SHAN_SHAN   refer_pos 已是 pixel 座標，所以鄯善對齊保持不變
    fixed_positions_list.append(['鄯善', refer_pos[0], refer_pos[1]])
    pos_matrix[dni['鄯善']] = [refer_pos[0], refer_pos[1]]
    
    # FIX POINTS
    for fix_cities in FIXPOINTS :
        index = dni[fix_cities]
        (lcc_x,lcc_y) = gt_xy_km[index]
        px = lcc_x * km2pix + refer_pos[0]
        py = lcc_y * km2pix + refer_pos[1]
        pos_matrix[index] = [px,py]
        fixed_positions_list.append([fix_cities,px,py])
    
    return pos_matrix, fixed_positions_list
def uploading_directional_data():
    csv_file_path = FILE_PATHS["directional_data"]
    directional_data= []
    with open(csv_file_path, mode="r", newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file)  # Create a CSV reader object
        for row in reader:
            directional_data.append(row)
    return directional_data

def main_physics_simulation(vertice,dni,data,pos_matrix,directional_data,fixed_positions_list) : # Main PS function
    '''initialize constants for main PS'''
    n = len(vertice)
    mass = 10
    fixmass = 1e7
    radius = 5
    vrange = 100
    
    spring_stiffness   = SPRING_STIFFNESS_BASE
    repulsion_strength = REPULSION_STRENGTH_BASE
    directional_force_magnitude = DIRECTIONAL_FORCE_MAGNITUDE_BASE
    
    spring_damping = 50
    min_distance = 0.1
    resistance = 10
    
    '''set up PS'''
    pygame.init()
    screen = pygame.display.set_mode((1200, 750))
    space = pymunk.Space()
    draw_options = pymunk.pygame_util.DrawOptions(screen)
    font = pygame.font.SysFont("Microsoft YaHei", 12)
    nodes,space = create_nodes_and_springs(n,mass,radius,vrange,spring_stiffness,spring_damping,fixmass,space,data, dni,pos_matrix,fixed_positions_list)
    # groupdni = classify_nodes()
    wrong_direction_lists,stress_history,pos_matrix = run_physics_simulation(min_distance,repulsion_strength,resistance,directional_force_magnitude,screen,space,draw_options,font,nodes,directional_data,data,vertice,dni,pos_matrix)
    return wrong_direction_lists,stress_history,pos_matrix
def create_nodes_and_springs(n,mass,radius,vrange,spring_stiffness,spring_damping,fixmass,space,data, dni,pos_matrix,fixed_positions_list) : #add nodes and springs into pymunk
    nodes = [pymunk.Body(mass,pymunk.moment_for_circle(mass, 0, radius)) for _ in range(n)]
    # add fixed nodes
    for row in fixed_positions_list :
        nodes[dni[row[0]]] = pymunk.Body(fixmass,pymunk.moment_for_circle(mass, 0, radius))
        # print(pos_matrix[dni[row[0]]])
    for i in range(n) :
        body = nodes[i]
        body.position = (pos_matrix[i][0], pos_matrix[i][1])
        #body.velocity = (random.uniform((-1)*vrange, vrange), random.uniform((-1)*vrange, vrange))
        body.velocity = (0,0)
        shape = pymunk.Circle(body, radius)
        # control the collision of points
        #shape.filter = pymunk.ShapeFilter(categories=groupdni[vertice[i]], mask=groupdni[vertice[i]]) # collide in same group
        #shape.filter = pymunk.ShapeFilter(group = groupdni[vertice[i]]) # collide in different group
        shape.filter = pymunk.ShapeFilter(group = 1) # not collide with each other
        space.add(body,shape)
    springs = []
    for row in data:
        i = dni[row[0]]
        j = dni[row[1]]
        spring = pymunk.DampedSpring(
            nodes[i], nodes[j], (0, 0), (0, 0), int(row[3])/10, spring_stiffness, spring_damping
        )
        springs.append(spring)
        space.add(spring)
    return nodes,space
def classify_nodes():
    dt = FILE_PATHS["classification_data"]
    groupdni = {}
    with open( dt , newline='', encoding='utf-8' ) as csvfile :
        rows = csv.reader(csvfile)
        for row in rows :
            groupdni[row[0]] = int(row[1])
        groupdni['都護治/烏壘']=1
    return groupdni
def run_physics_simulation(min_distance,repulsion_strength,resistance,directional_force_magnitude,screen,space,draw_options,font,nodes,directional_data,data,vertice,dni,pos_matrix): # run physics_simulation
    clock = pygame.time.Clock() # control pygame time frame
    running = True # running flag
    iteration = 0
    stress_history = []
    while running:
        iteration += 1
        # resistance += 0.001
        for event in pygame.event.get(): # handle events like closing the window
            if event.type == pygame.QUIT:
                running = False
        space.step(0.01) # Advances the Pymunk physics engine by 0.02 seconds per frame, updating positions and velocities.
        clock.tick(60) # Limits the frame rate to 60 FPS to ensure smooth simulation.
        nodes,cnt,wrong_direction_lists = apply_forces(min_distance,repulsion_strength,resistance,directional_force_magnitude,nodes,directional_data, dni)
        for i,node in enumerate(nodes) :
            pos_matrix[i] = nodes[i].position
            # pos_matrix = shift(pos_matrix,1,[600,375]) # shift the points to the center to avoid digressing
        # screen,space,current_stress = plotting_physics_simulation(screen,space,draw_options,font,nodes,data,vertice,dni, pos_matrix,cnt,wrong_direction_lists)
        stress_history = []
        # stress_history.append(current_stress)
        if iteration > 2000:
            # print(f"physics_simulation stops at iteration {iteration}")
            break
    return wrong_direction_lists,stress_history,pos_matrix
def plotting_physics_simulation(screen,space,draw_options,font,nodes,data,vertice,dni, pos_matrix,cnt,wrong_direction_lists):
    # refresh the screen
    screen.fill((255, 255, 255)) # fill the screen with white to clear previous frame
    # Calculate and display stress ； show cnt
    stress_font = pygame.font.SysFont("Microsoft YaHei", 24)
    current_stress = stress_function(data, dni, pos_matrix)
    stress_text = stress_font.render(f"Stress: {current_stress:.2f}", True, (0, 0, 0)) # displays stress with two decimal places.
    screen.blit(stress_text, (10, 10))  # Display at top-left corner
    cnt_text = stress_font.render(f"Wrong edge directions: {cnt}", True, (0, 0, 0))
    screen.blit(cnt_text, (10, 40))
    n= len(wrong_direction_lists)
    for i in range(n) :
        row = wrong_direction_lists[i]
        row_text = font.render(f"{row}",True,(0,0,0))
        screen.blit(row_text,(10,100+20*i))
    # Displays nodes, springs and labels
    space.debug_draw(draw_options) # Uses Pymunk's debug drawing to render objects (nodes, springs) on the screen.
    for i, node in enumerate(nodes):
        label = vertice[i]
        text_surface = font.render(label, True, (0, 0, 0))
        screen.blit(text_surface, (node.position[0] - 10, node.position[1] - 10))
    pygame.display.flip() # Updates the entire screen with new frame data.
    return screen,space,current_stress
def stress_function(data,dni,pos_matrix) : # unit : (km^2)； 1里=0.415km
    stress = 0
    for row in data :
        ind1 = dni[row[0]]
        ind2 = dni[row[1]]
        distance = sqrt((pos_matrix[ind2][0]-pos_matrix[ind1][0])**2 + (pos_matrix[ind2][1]-pos_matrix[ind1][1])**2)*10*0.415
        stress += (distance-int(row[2])*0.415)**2
    return stress
def apply_forces(min_distance,repulsion_strength,resistance,directional_force_magnitude,nodes,directional_data, dni):
    '''replusive force'''
    for i, node_a in enumerate(nodes):
        for j, node_b in enumerate(nodes):
            if i >= j:
                continue
            dx = node_b.position.x - node_a.position.x
            dy = node_b.position.y - node_a.position.y
            distance = max((dx**2 + dy**2) ** 0.5, min_distance)
            force_magnitude = repulsion_strength / (distance ** 1) # should be 2 ???
            fx = force_magnitude * dx / distance
            fy = force_magnitude * dy / distance
            node_a.apply_force_at_world_point((-fx, -fy), node_a.position)
            node_b.apply_force_at_world_point((fx, fy), node_b.position)
    '''direction_revising force'''
    # If we apply forces directory in direction of ideal directions, it will lead to equalibriam. 
    # y-axis is toward negative side
    direction_dict = {'東':np.array([1,0]), '西':np.array([-1,0]), '北':np.array([0,-1]), '南':np.array([0,1])}
    direction_dict2 = {'東南':np.array([1,1]), '西北':np.array([-1,-1]), '東北':np.array([1,-1]), '西南':np.array([-1,1])}
    cnt = 0 # recording the number of the wrong directions
    wrong_direction_lists = []
    for row in directional_data :
        # calculate the distance vector between nodes
        index1 = dni[row[0]] # left one in csv file
        index2 = dni[row[1]] # right one
        node1 = nodes[index1]
        node2 = nodes[index2]
        n1 = node1.position
        n2 = node2.position
        pos_vector = np.array([n2.x-n1.x,n2.y-n1.y])
        pos_vector = pos_vector / np.linalg.norm(pos_vector) # unit vector
        ver_vector1 = np.array([-pos_vector[1],pos_vector[0]])
        ver_vector2 = np.array([pos_vector[1],-pos_vector[0]])
        if row[2] in direction_dict : # for those with rough direction only (東南西北), theta must smaller than pi/4
            cos_similarity = np.dot(pos_vector,direction_dict[row[2]])
            if np.dot(pos_vector,direction_dict[row[2]]) < 0 : # 1/sqrt(2) : # apply force if being on the wrong direction
                cnt+=1
                wrong_direction_lists.append(row)
                '''
                node1.apply_force_at_world_point(tuple(-directional_force_magnitude*direction_dict[row[2]]),node1.position)
                node2.apply_force_at_world_point(tuple(directional_force_magnitude*direction_dict[row[2]]),node2.position)
                '''
                # choose which way to rotates
                if np.dot(ver_vector1,direction_dict[row[2]])>=np.dot(ver_vector2,direction_dict[row[2]]) :
                    node1.apply_force_at_world_point(tuple(directional_force_magnitude*ver_vector2),node1.position)
                    node2.apply_force_at_world_point(tuple(directional_force_magnitude*ver_vector1),node2.position)
                else :
                    node1.apply_force_at_world_point(tuple(directional_force_magnitude*ver_vector1),node1.position)
                    node2.apply_force_at_world_point(tuple(directional_force_magnitude*ver_vector2),node2.position)
                
        else : # for those with more specific direction (東南、西北), the cos(theta) between directional vector and pos_vector must be over cos(pi/8)
            cos_similarity = np.dot(pos_vector,direction_dict2[row[2]])/sqrt(2)
            if cos_similarity < 1/sqrt(2):  # 0.924 : theta > pi/8
                cnt+=1
                wrong_direction_lists.append(row)
                if np.dot(ver_vector1,direction_dict2[row[2]])>=np.dot(ver_vector2,direction_dict2[row[2]]) :
                    node1.apply_force_at_world_point(tuple(directional_force_magnitude*ver_vector2),node1.position)
                    node2.apply_force_at_world_point(tuple(directional_force_magnitude*ver_vector1),node2.position)
                else :
                    node1.apply_force_at_world_point(tuple(directional_force_magnitude*ver_vector1),node1.position)
                    node2.apply_force_at_world_point(tuple(directional_force_magnitude*ver_vector2),node2.position)
                
    '''resistance'''
    for node in nodes:
        vx, vy = node.velocity
        fx = -resistance * vx
        fy = -resistance * vy
        node.apply_force_at_world_point((fx, fy), node.position)
    return nodes,cnt,wrong_direction_lists

def post_procession(vertice,dni,data,refer_pos,wrong_direction_lists,stress_history,pos_matrix): # for any post_processions will be put inside
    # plot_stress_convergence_log(stress_history)
    visualize_error_map_official(pos_matrix, vertice, dni, data, wrong_direction_lists, zoom_area=None)
    visualize_error_map_official(pos_matrix, vertice, dni, data, wrong_direction_lists, zoom_area=(500, 325, 800, 400))
    ground_truth_positions = uploading_ground_truth(vertice,dni)
    ground_truth_comparison(vertice,dni,data,ground_truth_positions,refer_pos,pos_matrix)
    #pos_matrix = alignment_and_rotation(vertice,dni,data,pos_matrix,np.pi)
    #pos_matrix = projection(pos_matrix)
    #turnto_csv(vertice,pos_matrix)
    # plotting(pos_matrix,vertice,dni,edges)
def plot_stress_convergence_log(stress_history):
    """
    Draw the stress convergence curve with log-scaled Y-axis using pygame.
    """
    width, height = 1200, 750
    margin = 60
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Stress Convergence Curve")
    font = pygame.font.SysFont("Arial", 18)
    big_font = pygame.font.SysFont("Arial", 24)
    screen.fill((255, 255, 255))

    # Preprocess for log scale
    log_stress = [math.log10(s + 1e-8) for s in stress_history]
    max_log = max(log_stress)
    min_log = min(log_stress)
    num_steps = len(stress_history)
    plot_width = width - 2 * margin
    plot_height = height - 2 * margin

    # Draw axes
    pygame.draw.line(screen, (0, 0, 0), (margin, margin), (margin, height - margin), 2)  # Y-axis
    pygame.draw.line(screen, (0, 0, 0), (margin, height - margin), (width - margin, height - margin), 2)  # X-axis

    # Draw curve
    prev_point = None
    for i, log_s in enumerate(log_stress):
        x = margin + int(i / (num_steps - 1) * plot_width)
        y = height - margin - int((log_s - min_log) / (max_log - min_log) * plot_height)
        if prev_point:
            pygame.draw.line(screen, (0, 102, 204), prev_point, (x, y), 2)
        prev_point = (x, y)

    # Y-axis ticks (log scale)
    y_ticks = [2, 3, 4, 5, 6, 7, 8]
    for y_val_log in y_ticks:
        y_pos = height - margin - int((y_val_log - y_ticks[0]) * plot_height / 6)
        label = font.render(f"{y_val_log:.2f}", True, (0, 0, 0))
        screen.blit(label, (10, y_pos - 8))
        pygame.draw.line(screen, (200, 200, 200), (margin - 5, y_pos), (width - margin, y_pos), 1)

    # X-axis ticks
    for j in range(6):
        x_val = int(j * (num_steps - 1) / 5)
        x_pos = margin + int(j * plot_width / 5)
        label = font.render(f"{x_val}", True, (0, 0, 0))
        screen.blit(label, (x_pos - 10, height - margin + 8))
        pygame.draw.line(screen, (200, 200, 200), (x_pos, margin), (x_pos, height - margin + 5), 1)

    # Labels and title
    x_label = font.render("Iteration Step", True, (0, 0, 0))
    y_label = font.render("Stress (log scale) (unit : km^2)", True, (0, 0, 0))
    title_surface = big_font.render("Stress Convergence Curve", True, (0, 0, 0))
    screen.blit(x_label, (width // 2 - 50, height - 35))
    screen.blit(y_label, (20, 35))
    screen.blit(title_surface, (width // 2 - 140, 15))
    
    now_stress = stress_history[-1]  # 取最後一個 stress
    now_stress_text = font.render(f"now_stress = {now_stress:.6f}", True, (0, 0, 0))
    screen.blit(now_stress_text, (20, 10))  # 放在 (20,25) 位置

    # Show and save
    pygame.display.update()
    save_path = "C:/Users/justi/Desktop/project/results/stress_convergence_log.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pygame.image.save(screen, save_path)

    # Wait for window close
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    pygame.quit()
def visualize_error_map_official(pos_matrix, vertice, dni, data, wrong_direction_lists, zoom_area):
    """
    Official version for visualizing node error maps with scaled error color,
    top-5 error labels, and a color legend. Suitable for publication or reports.
    """
    width, height = 1200, 750
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    font = pygame.font.SysFont("Microsoft YaHei", 20)
    screen.fill((255, 255, 255))

    # === Zoom / Position handling ===
    scale_factor = 1
    if zoom_area:
        x_min, y_min, x_max, y_max = zoom_area
        zoomed_nodes = {i: pos for i, pos in enumerate(pos_matrix) if x_min <= pos[0] <= x_max and y_min <= pos[1] <= y_max}
        if not zoomed_nodes:
            print("No nodes found in the zoomed area.")
            return
        delta_x = x_max - x_min
        delta_y = y_max - y_min
        scale_factor = sqrt((width * height) / (delta_x * delta_y))
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        canvas_center_x, canvas_center_y = width / 2, height / 2
        adjusted_positions = {
            i: ((pos[0] - center_x) * scale_factor + canvas_center_x,
                (pos[1] - center_y) * scale_factor + canvas_center_y)
            for i, pos in zoomed_nodes.items()
        }
    else:
        adjusted_positions = {i: (pos[0], pos[1]) for i, pos in enumerate(pos_matrix)}

    # === Edge + Error computation ===
    errors = []
    edges = []
    edge_labels = []
    idl_edge_km = []
    for row in data:
        ind1 = dni[row[0]]
        ind2 = dni[row[1]]
        if ind1 in adjusted_positions and ind2 in adjusted_positions:
            x1, y1 = adjusted_positions[ind1]
            x2, y2 = adjusted_positions[ind2]
            actual_dist = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            ideal_dist = float(row[2]) / 10 *scale_factor
            error_rate = abs(actual_dist - ideal_dist) / ideal_dist
            errors.append(error_rate)
            edges.append(((x1, y1), (x2, y2)))
            edge_labels.append((row[0], row[1]))
            idl_edge_km.append(float(row[2])*0.415)
    # === lines with error_rate > 0.03 ===
    sorted_pairs = sorted(zip(errors, edge_labels, edges,idl_edge_km), key=lambda x: x[0], reverse=True)
    top_n = 0
    for i,pair in enumerate(sorted_pairs) :
        if pair[0] > 0.03 :
            top_n = i+1
    top_edges = sorted_pairs[:top_n]

    # === Draw all edges ===
    for error, (label1, label2), ((x1, y1), (x2, y2)) in zip(errors, edge_labels, edges):
        # 固定色階上限為 3%，下限為 0%
        error_clipped = min(max(error, 0), 0.03)
        color_val = int(255 * (1 - error_clipped / 0.03))
        color = (255, color_val, color_val)  # 紅(高誤差) → 淡紅(低誤差)
        pygame.draw.line(screen, color, (x1, y1), (x2, y2), 2)
    
    # 紀錄已放置文字的位置區塊
    used_boxes = []
    padding = 15
    # 嘗試避開擁擠：最多嘗試 9*4 個方向
    offset_candidates = [ (10, -10), (12, 0), (10, 10), (0, 15), (-10, 10), (-12, 0),  
        (-10, -10), (0, -15), (15, -3), (-15, 5) ]
    for i in range(3) :
        k=1.5+i*0.5
        for j in range(9) :
            offset_candidates.append((k*offset_candidates[j][0],k*offset_candidates[j][1]))
    # === Draw top N error labels ===
    for error, (label1, label2), ((x1, y1), (x2, y2)), idlkm in top_edges:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        error_percent = f"{error * 100:.2f}% (+{error*idlkm:.2f}km)"
        text_surface = font.render(error_percent, True, (0, 0, 0))
        
        # 嘗試偏移避免重疊
        placed = False
        for dx, dy in offset_candidates:
            tx, ty = mid_x + dx, mid_y + dy
            text_rect = text_surface.get_rect(topleft=(tx, ty))
            padded_rect = pygame.Rect(tx, ty, text_rect.width + 2*padding, text_rect.height + 2*padding)

            overlap = any(padded_rect.colliderect(pygame.Rect(bx, by, bw, bh)) for bx, by, bw, bh in used_boxes)
            if not overlap:
                screen.blit(text_surface, (tx, ty))
                used_boxes.append((padded_rect.left, padded_rect.top, padded_rect.width, padded_rect.height))
                placed = True
                break
        
        pygame.draw.line(screen, (255, 0, 0), (x1, y1), (x2, y2), 3)  # 強調顏色與粗細
    
    # === Draw nodes ===
    for i, (x, y) in adjusted_positions.items():
        label = vertice[i]
        node_color = (0, 180, 0)
        for row in wrong_direction_lists:
            if row[0] == label or row[1] == label:
                node_color = (255, 0, 0)
                break
        pygame.draw.circle(screen, node_color, (int(x), int(y)), 5)

        for j,(dx, dy) in enumerate(offset_candidates):
            tx = x + dx
            ty = y + dy
            text_surface = font.render(label, True, (0, 0, 0))
            text_rect = text_surface.get_rect(topleft=(tx, ty))
            # 檢查是否與已用區域重疊
            overlap = any(text_rect.colliderect(pygame.Rect(bx, by, bw, bh)) for bx, by, bw, bh in used_boxes)
            if (not overlap) :
                screen.blit(text_surface, (tx, ty))
                used_boxes.append((tx, ty, text_rect.width+2*padding, text_rect.height+2*padding))
                break


    # === Draw legend ===
    for i in range(100):
        val = i / 100
        color_val = int(255 * (1 - val))
        color = (255, color_val, color_val)
        pygame.draw.line(screen, color, (width - 40, height - 150 + i), (width - 20, height - 150 + i), 2)
    screen.blit(font.render("0%", True, (0, 0, 0)), (width - 75, height - 150))
    screen.blit(font.render("3%", True, (0, 0, 0)), (width - 75, height - 50))
    screen.blit(font.render("Error", True, (0, 0, 0)), (width - 70, height - 170))

    # === Save image to specific folder with name based on zoom_area ===
    save_dir = "C:/Users/justi/Desktop/project/results"
    os.makedirs(save_dir, exist_ok=True)

    if zoom_area:
        zoom_name = f"zoomed_{zoom_area[0]}_{zoom_area[1]}_{zoom_area[2]}_{zoom_area[3]}"
        filename = f"error_map_{zoom_name}.png"
    else:
        filename = f"error_map_full.png"

    save_path = os.path.join(save_dir, filename)
    pygame.image.save(screen, save_path)


    # === Wait to close window ===
    pygame.display.flip()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    pygame.quit()
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
def ground_truth_comparison(vertice,dni,data, ground_truth_positions, refer_pos, pos_matrix):
    """
    1. 將 pos_matrix 轉成 km , 並以 refer_pos (鄯善) 為 (0,0)
    2. 用 lcc_transformation 取得 ground truth 的 km 座標
    3. 計算每點誤差與 RMSE
    4. 用 Pygame 繪製 overlay 底圖為 gt, 疊圖為模擬結果
    """
    
    # 1. 模擬結果轉 km 並對齊
    sim_xy_km = []
    for x, y in pos_matrix:
        x_km = (x - refer_pos[0]) * 10 * 0.415
        y_km = (y - refer_pos[1]) * 10 * 0.415
        sim_xy_km.append((x_km, y_km))

    # 2. ground truth 投影
    gt_xy_km = lcc_transformation(dni, ground_truth_positions)
    
    # 3. 計算誤差與 RMSE
    errors = []
    valid_idx = []
    for i, (sx, sy) in enumerate(sim_xy_km):
        gt = gt_xy_km[i]
        if gt[0] is None:  # 無 GT
            continue
        gx, gy = gt
        err = math.hypot(sx - gx, sy - gy)
        errors.append(err)
        valid_idx.append(i)
    max_error = max(errors)
    rmse = math.sqrt(np.mean(np.square(errors)))

    # 4. Pygame 疊圖設定
    pygame.init()
    width, height = 1200, 750
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Overlay: Physics Simulation vs Ground Truth")
    font = pygame.font.SysFont("Microsoft YaHei", 20)
    big_font = pygame.font.SysFont("Microsoft YaHei", 30)
    screen.fill((255, 255, 255))
    title_surf = big_font.render("Overlay: Physics Simulation vs Ground Truth", True, (0, 0, 0))
    screen.blit(title_surf, (width//2 - title_surf.get_width()//2, 20))

    # 平移 + 縮放
    offset_x, offset_y = 700, 500
    scale = 1.2 / (10 * 0.415)

    # 5. 畫底圖：Ground Truth（單一灰色）
    special = { dni['鄯善'], dni['都護治/烏壘'] }
    for i, (gx, gy) in enumerate(gt_xy_km):
        if gx is None: continue
        dx = int(gx * scale + offset_x)
        dy = int(gy * scale + offset_y)
        if i in special:
            color = (100, 100, 100)   # 深灰
            r = 10
        else:
            color = (200, 200, 200)   # 淺灰
            r = 5
        pygame.draw.circle(screen, color, (dx, dy), 5)

    ## 避免標籤碰撞
    # 紀錄已放置文字的位置區塊
    used_boxes = []
    padding = 15
    # 嘗試避開擁擠：最多嘗試 9*4 個方向
    offset_candidates = [ (10, -10), (12, 0), (10, 10), (0, 15), (-10, 10), (-12, 0),  
        (-10, -10), (0, -15), (15, -3), (-15, 5) ]
    for i in range(3) :
        k=1.5+i*0.5
        for j in range(9) :
            offset_candidates.append((k*offset_candidates[j][0],k*offset_candidates[j][1]))
    
    # 6. 畫疊圖：Simulation（色階 + 標籤）
    for idx, err in zip(valid_idx, errors):
        sx, sy = sim_xy_km[idx]
        x = int(sx * scale + offset_x)
        y = int(sy * scale + offset_y)
        # 色階：0→藍, max→紅
        t = min(err / max_error, 1.0)
        color = (int(255 * t), 0, int(255 * (1 - t)))
        pygame.draw.circle(screen, color, (x, y), 6)
        # 國家標籤
        label = vertice[idx]
        for j,(dx, dy) in enumerate(offset_candidates):
            tx = x + dx
            ty = y + dy
            text_surface = font.render(label, True, (0, 0, 0))
            text_rect = text_surface.get_rect(topleft=(tx, ty))
            # 檢查是否與已用區域重疊
            overlap = any(text_rect.colliderect(pygame.Rect(bx, by, bw, bh)) for bx, by, bw, bh in used_boxes)
            if (not overlap) :
                screen.blit(text_surface, (tx, ty))
                used_boxes.append((tx, ty, text_rect.width+2*padding, text_rect.height+2*padding))
                break
    
    


    # 7. 畫色階圖例（右下角）
    bar_h, bar_w = 200, 20
    bx, by = width - 60, height - bar_h - 40
    for i in range(bar_h):
        t = i / bar_h
        c = (int(255 * t), 0, int(255 * (1 - t)))
        pygame.draw.line(screen, c, (bx, by + bar_h - i), (bx + bar_w, by + bar_h - i))
    screen.blit(font.render("0", True, (0,0,0)), (bx - 40, by + bar_h - 10))
    screen.blit(font.render(f"{max_error:.2f}", True, (0,0,0)), (bx - 80, by - 10))
    screen.blit(font.render("Error (km)", True, (0,0,0)), (bx - 100, by - 40))

    # 8. 顯示 RMSE、kruskal's stress（右上角）
    rmse_surf = font.render(f"RMSE = {rmse:.3f} km", True, (0, 0, 0))
    screen.blit(rmse_surf, (width - rmse_surf.get_width() - 20, 30))
    
    kruskal_stress = calculate_kruskals_stress(dni,pos_matrix,data)
    kru_surf = font.render(f"kruskal's stress = {kruskal_stress:.4f}", True, (0, 0, 0))
    screen.blit(kru_surf, (width - kru_surf.get_width() - 20, 80))

    # 9. 更新畫面並存檔
    pygame.display.flip()
    save_path = "C:/Users/justi/Desktop/project/results/Overlap.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pygame.image.save(screen, save_path)

    # 等待關閉
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
    pygame.quit()
def lcc_transformation(dni, ground_truth_positions):
     # use Lambert Conformal Conic to transform ground_truth to shan_shan (0,0) in xy plane
    lon_min, lon_max = 73.24516481, 92.74103523
    lat_min, lat_max = 37.12265816, 44.2843368
    lat1 = lat_min
    lat2 = lat_max
    lon0 = (lon_min + lon_max) / 2
    # 建立 LCC 投影 CRS（單位：公尺）
    crs_lcc = CRS.from_proj4(
        f"+proj=lcc +lat_1={lat1} +lat_2={lat2} "
        f"+lat_0={lat1} +lon_0={lon0} +x_0=0 +y_0=0 "
        "+ellps=WGS84 +units=m"
    )
    # 建立經緯度 (EPSG:4326) → LCC (自訂 CRS) 的轉換器
    transformer = Transformer.from_crs("EPSG:4326", crs_lcc, always_xy=True)
    # ─── 執行投影並轉換為公里 ─────────────────────────────────
    lcc_xy_m = []
    for lon,lat in ground_truth_positions :
        if lon==0 and lat==0 :
            lcc_xy_m.append((None,None))
        else :
            lcc_xy_m.append(transformer.transform(lon, lat))
    align_pos = lcc_xy_m[dni['鄯善']]
    lcc_xy_km = []
    for x,y in lcc_xy_m :
        if x==None and y==None :
            lcc_xy_km.append((None,None))
        else :
            lcc_xy_km.append(((x-align_pos[0])/1000 , -(y-align_pos[1])/1000)) # we use (-y) since the coordinates in pygame is different
    return lcc_xy_km
def calculate_kruskals_stress(dni,pos_matrix,data) :
    for pos_pair in pos_matrix : # transform the pixel unit to km 
        pos_pair[0] = pos_pair[0] *10 *0.415
        pos_pair[1] = pos_pair[1] *10 *0.415
    error_square = 0
    dis_square = 0
    for row in data :
        p1 = pos_matrix[dni[row[0]]]
        p2 = pos_matrix[dni[row[1]]]
        actual_dis = np.sqrt( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        ideal_dis = int(row[2])*0.415
        error_square += (actual_dis-ideal_dis)**2
        dis_square += actual_dis**2
    kruskal_stress = np.sqrt(error_square/dis_square)
    return kruskal_stress
    
def alignment_and_rotation(vertice,dni,data,pos_matrix,theta_real) : # 
    for pos in pos_matrix :
        pos[0]*= 10
        pos[1]*= 10
    for pos in pos_matrix : # 里 = 415 公尺
        pos[0] *= 415/1000
        pos[1] *= 415/1000
    x,y = pos_matrix[dni["鄯善"]]
    for pos in pos_matrix :
        pos[0]-= x
        pos[1]-= y
    i0 = dni["鄯善"]
    i1 = dni["都護治/烏壘"]
    theta_sim = np.arctan2(pos_matrix[i1][1] - pos_matrix[i0][1], pos_matrix[i1][0] - pos_matrix[i0][0]) ##remain not sure
    rotation_angle = theta_real - theta_sim
    cos_angle = np.cos(rotation_angle)
    sin_angle = np.sin(rotation_angle)
    rotation_matrix = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])
    for i, (x, y) in enumerate(pos_matrix):
        x_rotated, y_rotated = np.dot(rotation_matrix, [x, y])
        pos_matrix[i] = [x_rotated, y_rotated]
    print(pos_matrix)
    return pos_matrix
def calculate_new_coordinates(lat_ref, lon_ref, dx, dy,geod):
    distance = sqrt(dx**2 + dy**2)
    bearing = degrees(atan2(dy, dx))
    lon, lat, _ = geod.fwd(lon_ref, lat_ref, bearing, distance * 1000)
    return lon, lat
def projection(pos_matrix) :
    lat_ref = 40.527633
    lon_ref = 89.840644
    geod = Geod(ellps="WGS84")
    for i in range(len(pos_matrix)):
        x, y = pos_matrix[i]
        pos_matrix[i] = list(calculate_new_coordinates(lat_ref, lon_ref, x, y,geod))
    return pos_matrix
def turnto_csv(vertice,pos_matrix) :
    data = [[vertice[i],pos_matrix[i][0],pos_matrix[i][1]] for i in range(len(pos_matrix))]
    with open(FILE_PATHS["output_csv"], mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(data)

def model_cmp(vertice,dni,pos_matrix) :
    refer_pos = [600,500]
    align_pos = pos_matrix[dni['鄯善']]
    for pos in pos_matrix :
        pos[0] = (pos[0]-align_pos[0])*0.412 + refer_pos[0]
        pos[1] = (pos[1]-align_pos[1])*0.412 + refer_pos[1]
    data = read_CHEN_csvfile()
    wrong_direction_lists = []
    visualize_error_map_official(pos_matrix, vertice, dni, data, wrong_direction_lists, zoom_area=None)
    visualize_error_map_official(pos_matrix, vertice, dni, data, wrong_direction_lists, zoom_area=(500, 325, 800, 400))
    ground_truth_positions = uploading_ground_truth(vertice,dni)
    ground_truth_comparison(vertice,dni,data,ground_truth_positions,refer_pos,pos_matrix)

def main_function(): # avoid global parameters
    refer_pos = [600,500]
    vertice,dni,data,pos_matrix,fixed_positions_list = generate_CHEN_initial_positions(refer_pos)
    directional_data = uploading_directional_data()
    wrong_direction_lists,stress_history,pos_matrix = main_physics_simulation(vertice,dni,data,pos_matrix,directional_data,fixed_positions_list)
    post_procession(vertice,dni,data,refer_pos,wrong_direction_lists,stress_history,pos_matrix)
if __name__ == "__main__":
    main_function()
    # import spring_confidence as sc
    # mu, covs, vertice = sc.bootstrap_and_plot()
