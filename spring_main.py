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

from metrics import *
from config import *
from data_io import *
from geometry import *
from visulization import *


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


def main_physics_simulation(vertice,dni,data,pos_matrix,directional_data,fixed_positions_list) : # Main PS function
    '''initialize constants for main PS'''
    n = len(vertice)
    
    mass = MASS_BASE
    fixmass = FIXMASS_BASE
    radius = RADIUS_BASE
    vrange = VRANGE_BASE
    spring_damping = SPRING_DAMPING_BASE
    min_distance = MIN_DISTANCE_BASE
    resistance = RESISTANCE_BASE
    
    # IMPORTANT CONSTANTS
    spring_stiffness   = SPRING_STIFFNESS_BASE
    repulsion_strength = REPULSION_STRENGTH_BASE
    directional_force_magnitude = DIRECTIONAL_FORCE_MAGNITUDE_BASE
    
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
    import spring_confidence as sc
    # mu, covs, vertice = sc.bootstrap_and_plot()
