import pygame
import math
import os
import csv
from math import *
import numpy as np

from library.config import *
from library.metrics import calculate_kruskals_stress,stress_function
from library.geometry import lcc_transformation
from library.data_io import read_CHEN_csvfile,uploading_ground_truth
from library.geometry import inverse_lcc_transformation

def save_vis_data(vertice, dni, pos_matrix, ground_truth_positions, refer_pos):
    pos_matrix_km = []
    for pos in pos_matrix :
        pos_matrix_km.append(((pos[0]-refer_pos[0])*4.15,(pos[1]-refer_pos[1])*4.15))
    wgs_pos_matrix = inverse_lcc_transformation(pos_matrix_km,ground_truth_positions[dni["鄯善"]])
    vis_data = []
    for i,label in enumerate(vertice) :
        vis_data.append( (label, wgs_pos_matrix[i][0], wgs_pos_matrix[i][1]) )
    with open(FILE_PATHS["save_vis_data"], mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerows(vis_data)

def plotting_physics_simulation(screen,space,draw_options,font,nodes,data,vertice,dni, pos_matrix,cnt,wrong_direction_lists,current_stress):
    # refresh the screen
    screen.fill((255, 255, 255)) # fill the screen with white to clear previous frame
    # Calculate and display stress ； show cnt
    stress_font = pygame.font.SysFont("Microsoft YaHei", 24)
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
    return screen,space

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
