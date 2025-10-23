import pygame
import math
import os
from math import *
import numpy as np
from copy import deepcopy

from library.config import *
from library.metrics import stress_function, calculate_kruskals_stress, procrustes_align_by_fixed_points, rmse_km_from_pixels
from library.geometry import lcc_transformation
from library.model_cmp import rmse_km_from_pixels


def plotting_physics_simulation_animation_tmp(space, screen, draw_options,font, data, vertice, dni, pos_history):
    clock = pygame.time.Clock() # control pygame time frame
    stress_history = []
    pos_history = []
    for pos_matrix in pos_history :
        for event in pygame.event.get(): # handle events like closing the window
            if event.type == pygame.QUIT:
                running = False
        space.step(0.01) # Advances the Pymunk physics engine by 0.02 seconds per frame, updating positions and velocities.
        clock.tick(60) # Limits the frame rate to 60 FPS to ensure smooth simulation.
        current_stress = stress_function(data,dni,pos_matrix)
        # refresh the screen
        screen.fill((255, 255, 255)) # fill the screen with white to clear previous frame
        # Calculate and display stress ； show cnt
        stress_font = pygame.font.SysFont("Microsoft YaHei", 24)
        stress_text = stress_font.render(f"Stress: {current_stress:.2f}", True, (0, 0, 0)) # displays stress with two decimal places.
        screen.blit(stress_text, (10, 10))  # Display at top-left corner
        # Displays nodes, springs and labels
        space.debug_draw(draw_options) # Uses Pymunk's debug drawing to render objects (nodes, springs) on the screen.
        for i, pos in enumerate(pos_matrix):
            label = vertice[i]
            text_surface = font.render(label, True, (0, 0, 0))
            screen.blit(text_surface, (pos[0] - 10, pos[1] - 10))
        pygame.display.flip() # Updates the entire screen with new frame data.
        stress_history.append(current_stress)

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

def plot_stress_convergence_log(stress_history, file_name):
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

    # Y-axis ticks (log scale)
    y_ticks = [2, 3, 4, 5, 6, 7, 8]
    y_ticks = [ (y-5) for y in y_ticks]

    # Draw curve
    prev_point = None
    for i, log_s in enumerate(log_stress):
        x = margin + int(i / (num_steps - 1) * plot_width)
        y = height - margin - int((log_s - y_ticks[0]) * plot_height / 6)
        if log_s > y_ticks[-1] :
            continue 
        if prev_point:
            pygame.draw.line(screen, (0, 102, 204), prev_point, (x, y), 2)
        prev_point = (x, y)

    # Draw Y-axis ticks and grid lines
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
    y_label = font.render("Stress (log scale) (no unit)", True, (0, 0, 0))
    title_surface = big_font.render("Stress Convergence Curve", True, (0, 0, 0))
    screen.blit(x_label, (width // 2 - 50, height - 35))
    screen.blit(y_label, (20, 35))
    screen.blit(title_surface, (width // 2 - 140, 15))
    
    now_stress = stress_history[-1]  # 取最後一個 stress
    now_stress_text = font.render(f"now_stress = {now_stress:.6f}", True, (0, 0, 0))
    screen.blit(now_stress_text, (20, 10))  # 放在 (20,25) 位置

    # Show and save
    save_path = f"C:/Users/justi/Desktop/project/results/{file_name}stress_convergence_log.png"
    pygame.display.update()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pygame.image.save(screen, save_path)

    # Wait for window close
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    pygame.quit()

def visualize_error_map_official(pos_matrix, vertice, dni, data, wrong_direction_lists, zoom_area =None , file_name = None):
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
            idl_edge_km.append(float(row[2]) / km2Li)

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
        filename = f"{file_name}error_map_{zoom_name}.png"
    else:
        filename = f"{file_name}error_map_full.png"

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
    
    return errors, edge_labels

def ground_truth_comparison(vertice,dni,data, ground_truth_positions, refer_pos, pos_matrix, file_name):
    """
    1. 將 pos_matrix 轉成 km , 並以 refer_pos (鄯善) 為 (0,0)
    2. 用 lcc_transformation 取得 ground truth 的 km 座標
    3. 計算每點誤差與 RMSE
    4. 用 Pygame 繪製 overlay 底圖為 gt, 疊圖為模擬結果
    """
    
    # 1. 模擬結果轉 km 並對齊
    sim_xy_km = []
    for x, y in pos_matrix:
        x_km = (x - refer_pos[0]) / km2pix
        y_km = (y - refer_pos[1]) / km2pix
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
    scale = 1.2 * km2pix

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
    
    kruskal_stress = calculate_kruskals_stress(dni,deepcopy(pos_matrix),data)
    kru_surf = font.render(f"kruskal's stress = {kruskal_stress:.4f}", True, (0, 0, 0))
    screen.blit(kru_surf, (width - kru_surf.get_width() - 20, 80))

    # 9. 更新畫面並存檔
    pygame.display.flip()
    save_path = f"C:/Users/justi/Desktop/project/results/{file_name}Overlap.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pygame.image.save(screen, save_path)

    # 等待關閉
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
    pygame.quit()


def plot_three_model_convergence_pygame_pixelaware(
    pos_history_physics,
    pos_history_directed_mds,
    pos_history_stress_mj,
    *,
    vertice,
    dni,
    data,
    ground_truth_positions,
    fixed_point_labels = [],
    fixed_point_lonlat = [],
    refer_pos=(600, 500),
    window_size=(1200, 750),
    anchor_label="鄯善",
    orientation="pygame",           # "pygame" (y-down) 或 "north-up"
    stress_y_scale="log",           # "log" 或 "linear"
    bin_size_iters=10,              # ★ 以 iterations 分箱：可設 5、10、20
    band_alpha=38,                  # ★ 包絡帶透明度（~15%）
    pre_process=False,              #  pos_history 是否事先經過轉 px 和 flipping
    save_path="C:/Users/justi/Desktop/project/results/ThreeModels_RMSE_Kruskal_pixelaware.png",
):
    """
    以「固定 iterations 數」分箱的像素友善收斂圖：
      • 上：Kruskal’s stress（預設 log）
      • 下：RMSE（linear）
    每個 bin 繪製：
      - min~max 半透明包絡帶（約 15%）
      - median 實線
    並保留/強化座標軸與 y-tick 邊界檢查。
    """

    # --- 兼容 import（專案/單檔兩種結構） ---
    try:
        from library.config import km2pix, km2Li
        from library.metrics import calculate_kruskals_stress
        from library.geometry import lcc_transformation
    except Exception:
        from metrics import calculate_kruskals_stress
        from geometry import lcc_transformation   # 若你不是 library 結構，請確保有對應模組
        # 若 config 不在 library，可自行設置或 import
        try:
            from config import km2pix, km2Li
        except Exception:
            km2pix, km2Li =1/(10 * 0.415), 1/0.415  # 後備（避免崩潰；數值會不準）

    # flipping_y：盡量沿用你的實作（在 MDS_model.plot_node_link_diagram）
    try:
        from MDS_model.plot_node_link_diagram import flipping_y
    except Exception:
        def flipping_y(P, height):
            return [(x, height - y) for (x, y) in P]

    if anchor_label not in dni:
        raise KeyError(f"Anchor '{anchor_label}' not found in dni.")

    W, H = window_size
    anc_idx = dni[anchor_label]

    # --- Ground Truth 轉 LCC（km） ---
    gt_xy_km = lcc_transformation(dni, ground_truth_positions)
    if orientation == "north-up":
        # pygame y 向下，若使用 north-up，需翻轉
        gt_xy_km = [(gx, -gy) if gx is not None else (None, None) for gx, gy in gt_xy_km]

    # --- 兩種歷程到 km 的轉換 ---
    def phys_px_to_km(frame_px):
        pts = frame_px if orientation == "pygame" else flipping_y(frame_px, height=H)
        return [((x - refer_pos[0]) / km2pix, (y - refer_pos[1]) / km2pix) for (x, y) in pts]

    def mds_li_to_km(frame_li):
        ax, ay = frame_li[anc_idx]
        out = []
        for (x, y) in frame_li:
            kx = (x - ax) / km2Li
            ky = (y - ay) / km2Li
            if orientation == "pygame":
                ky = -ky
            out.append((kx, ky))
        return out

    def mds_li_to_pixels_for_kruskal(frame_li):
        km_pts = mds_li_to_km(frame_li)
        return [(x * km2pix, y * km2pix) for (x, y) in km_pts]

    # --- Series builders（每一步 RMSE / Kruskal） ---
    def compute_rmse_series(pos_history, kind):
        series = []
        for P in pos_history:
            if pre_process :
                series.append( rmse_km_from_pixels(deepcopy(P), refer_pos, dni, ground_truth_positions) )
            else :    
                sim_km = phys_px_to_km(P) if kind == "physics" else mds_li_to_km(P)
                if kind == "strmds" :
                    P = procrustes_align_by_fixed_points(deepcopy(P), fixed_point_labels, fixed_point_lonlat, dni)
                se = []
                for i, (sx, sy) in enumerate(sim_km):
                    gx, gy = gt_xy_km[i]
                    if gx is None:  # 缺 GT
                        continue
                    dx, dy = sx - gx, sy - gy
                    se.append(dx * dx + dy * dy)
                series.append(math.sqrt(sum(se) / len(se)) if se else float("nan"))
        return series

    def compute_kruskal_series(pos_history, kind):
        series = []
        for P in pos_history:
            if pre_process :
                P_pix = deepcopy(P)
            else :    
                if kind == "physics":
                    P_pix = deepcopy(P) if orientation == "pygame" else flipping_y(deepcopy(P), height=H)
                else:
                    P_pix = mds_li_to_pixels_for_kruskal(P)
            ks = float(calculate_kruskals_stress(dni, deepcopy([list(q) for q in P_pix]), data))
            series.append(ks)
        return series

    rmse_ph = compute_rmse_series(pos_history_physics, "physics")
    rmse_dm = compute_rmse_series(pos_history_directed_mds, "dirmds")
    rmse_sm = compute_rmse_series(pos_history_stress_mj, "strmds")

    ks_ph = compute_kruskal_series(pos_history_physics, "physics")
    ks_dm = compute_kruskal_series(pos_history_directed_mds, "mds")
    ks_sm = compute_kruskal_series(pos_history_stress_mj, "mds")

    xs_ph = list(range(len(rmse_ph)))
    xs_dm = list(range(len(rmse_dm)))
    xs_sm = list(range(len(rmse_sm)))
    x_max_iter = max(len(xs_ph), len(xs_dm), len(xs_sm)) - 1 if max(len(xs_ph), len(xs_dm), len(xs_sm)) > 0 else 1

    # --- pygame 視窗 ---
    pygame.init()
    flags = pygame.DOUBLEBUF
    try:
        screen = pygame.display.set_mode(window_size, flags, vsync=1)
    except TypeError:
        screen = pygame.display.set_mode(window_size, flags)
    pygame.display.set_caption("Convergence: Kruskal's Stress & RMSE (Binned Envelope + Median)")
    font = pygame.font.SysFont("Arial", 18)
    title_font = pygame.font.SysFont("Arial", 26)

    M = 80
    top = pygame.Rect(M, M + 20, W - 2 * M, (H - 3 * M) // 2)
    bot = pygame.Rect(M, top.bottom + M, W - 2 * M, (H - 3 * M) // 2)

    def finite(vals):
        out = []
        for v in vals:
            if v is None: 
                continue
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                continue
            out.append(v)
        return out

    # --- y 範圍與 y 映射 ---
    # RMSE linear
    rm_all = finite(rmse_ph + rmse_dm + rmse_sm) or [0.0, 1.0]
    yr_min, yr_max = min(rm_all), max(rm_all)

    # Stress
    ks_all = finite(ks_ph + ks_dm + ks_sm) or [1.0]
    if stress_y_scale == "log":
        pos_vals = [v for v in ks_all if v > 0]
        if not pos_vals:
            pos_vals = [1e-12]
        eps = 0.5 * min(pos_vals)
        def ylog(v): return math.log10(max(v, eps))
        yk_vals = [ylog(v) for v in ks_all]
        yk_min_log, yk_max_log = min(yk_vals), max(yk_vals)
        # decade ticks（邊界檢查：只畫在矩形範圍內）
        ks_max_tick = math.ceil(yk_max_log)
        ks_min_tick = math.floor(yk_min_log)
        ticks_log = list(range(ks_min_tick, ks_max_tick + 1))
        span = max(ks_max_tick - ks_min_tick, 1e-12)
        def map_ks_y(v):
            return int(top.bottom - ((math.log10(max(v, eps)) - ks_min_tick) / span) * top.height)
    else:
        yk_min_lin, yk_max_lin = min(ks_all), max(ks_all)
        ks_min_tick = math.floor(yk_min_lin)
        ks_max_tick = math.ceil(yk_max_lin)
        span = max(ks_max_tick - ks_min_tick, 1e-12)
        def map_ks_y(v):
            return int(top.bottom - ((v - ks_min_tick) / span) * top.height)

    def map_rmse_y(v):
        rm_max_tick = math.floor(yr_max)
        rm_min_tick = math.ceil(yr_min)
        span = max(rm_max_tick - rm_min_tick, 1e-12)
        return int(bot.bottom - ((v - rm_min_tick) / span) * bot.height)

    # --- x 映射（iteration → pixel）---
    def map_x_to_pixel(x_idx, rect):
        if x_max_iter <= 0:
            return rect.left
        return rect.left + int(round((x_idx / x_max_iter) * (rect.width - 1)))

    # --- 以 iteration 分箱，輸出 bin 中心 x 與 y 的 min/max/median ---
    def bin_min_max_median_by_iters(xs, ys, bin_size):
        N = len(xs)
        if N == 0 or bin_size <= 0:
            return [], [], [], []
        bx, by_min, by_max, by_med = [], [], [], []
        i = 0
        while i < N:
            j = min(i + bin_size, N)
            seg = [ys[k] for k in range(i, j) if (ys[k] is not None and not (isinstance(ys[k], float) and (math.isnan(ys[k]) or math.isinf(ys[k]))))]
            if seg:
                vmin = min(seg); vmax = max(seg)
                med = float(np.median(seg))
                x_mid = 0.5 * (xs[i] + xs[j - 1])  # bin 中心
                bx.append(x_mid); by_min.append(vmin); by_max.append(vmax); by_med.append(med)
            i = j
        return bx, by_min, by_max, by_med

    # --- 座標軸繪製（含邊界檢查） ---
    def draw_axes_linear(rect, y_min, y_max, x_max_iter, y_label, x_label):
        screen.fill((255, 255, 255), rect)
        pygame.draw.rect(screen, (245, 245, 245), rect)
        # 軸線
        pygame.draw.line(screen, (0, 0, 0), (rect.left, rect.bottom), (rect.right, rect.bottom), 2)
        pygame.draw.line(screen, (0, 0, 0), (rect.left, rect.top),    (rect.left, rect.bottom), 2)
        # grid
        for k in range(6):
            yy = rect.bottom - int(k * rect.height / 5)
            if rect.top <= yy <= rect.bottom:
                pygame.draw.line(screen, (220, 220, 220), (rect.left, yy), (rect.right, yy), 1)
        for k in range(6):
            xx = rect.left + int(k * rect.width / 5)
            if rect.left <= xx <= rect.right:
                pygame.draw.line(screen, (220, 220, 220), (xx, rect.top), (xx, rect.bottom), 1)
        # 標籤
        screen.blit(font.render(y_label, True, (0, 0, 0)), (rect.left - 10, rect.top - 25))
        screen.blit(font.render(x_label, True, (0, 0, 0)), (rect.centerx - 40, rect.bottom + 10))
        # ticks（超界不畫）
        for k in range(6):
            yv = y_min + (ceil(y_max) - floor(y_min)) * (k / 5.0)
            yy = rect.bottom - int(k * rect.height / 5)
            if rect.top <= yy <= rect.bottom:
                screen.blit(font.render(f"{yv:.3g}", True, (0, 0, 0)), (rect.left - 70, yy - 8))
            xv = int(x_max_iter * (k / 5.0))
            xx = rect.left + int(k * rect.width / 5)
            if rect.left <= xx <= rect.right:
                screen.blit(font.render(f"{xv}", True, (0, 0, 0)), (xx - 10, rect.bottom + 8))

    def draw_axes_log(rect, decade_ticks, y_min_log, y_max_log, x_max_iter, y_label, x_label):
        screen.fill((255, 255, 255), rect)
        pygame.draw.rect(screen, (245, 245, 245), rect)
        pygame.draw.line(screen, (0, 0, 0), (rect.left, rect.bottom), (rect.right, rect.bottom), 2)
        pygame.draw.line(screen, (0, 0, 0), (rect.left, rect.top),    (rect.left, rect.bottom), 2)
        span = max(ceil(y_max_log) - floor(y_min_log), 1e-12)
        # decade grid + labels（僅在範圍內才畫）
        for t in decade_ticks:
            yy = rect.bottom - int(((t - floor(y_min_log)) / span) * rect.height)
            if rect.top <= yy <= rect.bottom:
                pygame.draw.line(screen, (210, 210, 210), (rect.left, yy), (rect.right, yy), 1)
                screen.blit(font.render(f"1e{t}", True, (0, 0, 0)), (rect.left - 60, yy - 8))
        # x-grid
        for k in range(6):
            xx = rect.left + int(k * rect.width / 5)
            if rect.left <= xx <= rect.right:
                pygame.draw.line(screen, (220, 220, 220), (xx, rect.top), (xx, rect.bottom), 1)
        # 標籤
        screen.blit(font.render(y_label, True, (0, 0, 0)), (rect.left - 10, rect.top - 25))
        screen.blit(font.render(x_label, True, (0, 0, 0)), (rect.centerx - 40, rect.bottom + 10))

    # --- 包絡帶 + 中位數繪製（以 iteration 分箱） ---
    def draw_binned_band_and_median(rect, xs, ys, map_y, color, bin_size, alpha=38, median_width=2):
        bx, ymin_v, ymax_v, ymed_v = bin_min_max_median_by_iters(xs, ys, bin_size)
        if not bx:
            return
        # 轉 pixel
        pxs  = [map_x_to_pixel(xm, rect) for xm in bx]
        ymins = [map_y(v) for v in ymin_v]
        ymaxs = [map_y(v) for v in ymax_v]
        ymeds = [map_y(v) for v in ymed_v]

        # --- band polygon（上界左→右，接下界右→左） ---
        poly_points = list(zip(pxs, ymaxs)) + list(zip(reversed(pxs), reversed(ymins)))
        band = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        poly_local = [(x - rect.left, y - rect.top) for (x, y) in poly_points]
        band_color = (color[0], color[1], color[2], max(0, min(255, alpha)))
        if len(poly_local) >= 3:
            pygame.draw.polygon(band, band_color, poly_local)
        screen.blit(band, rect.topleft)

        # --- median 折線（AA）---
        median_points = list(zip(pxs, ymeds))
        if len(median_points) >= 2:
            pygame.draw.aalines(screen, color, False, median_points)
            # 讓線稍微厚一些
            for i in range(1, median_width):
                shifted = [(x, y - i) for (x, y) in median_points]
                pygame.draw.aalines(screen, color, False, shifted)

    # --- 顏色（沿用固定色） ---
    C_PH = (0, 102, 204)    # Physics: Blue
    C_DM = (255, 140, 0)    # Directed-MDS: Orange
    C_SM = (34, 139, 34)    # Stress-Majorization: Green

    # --- clear & title ---
    screen.fill((255, 255, 255))
    title = title_font.render(
        f"Convergence (Binned={bin_size_iters}): Kruskal's Stress & RMSE",
        True, (0, 0, 0)
    )
    screen.blit(title, (W // 2 - title.get_width() // 2, 20))

    # --- TOP: Kruskal's stress（軸） ---
    if stress_y_scale == "log":
        draw_axes_log(top, ticks_log, yk_min_log, yk_max_log, x_max_iter, "Stress", "Iteration")
        mapY_top = map_ks_y
    else:
        draw_axes_linear(top, yk_min_lin, yk_max_lin, x_max_iter, "Stress", "Iteration")
        mapY_top = map_ks_y

    # --- BOTTOM: RMSE（軸） ---
    draw_axes_linear(bot, yr_min, yr_max, x_max_iter, "RMSE (km)", "Iteration")

    bin_size_iters_zero = 1
    
    # --- 繪製包絡帶 + 中位線 ---
    draw_binned_band_and_median(top, xs_ph, ks_ph, mapY_top, C_PH, bin_size_iters_zero, alpha=band_alpha, median_width=2)
    draw_binned_band_and_median(top, xs_dm, ks_dm, mapY_top, C_DM, bin_size_iters_zero, alpha=band_alpha, median_width=2)
    draw_binned_band_and_median(top, xs_sm, ks_sm, mapY_top, C_SM, bin_size_iters_zero, alpha=band_alpha, median_width=2)

    draw_binned_band_and_median(bot, xs_ph, rmse_ph, map_rmse_y, C_PH, bin_size_iters_zero, alpha=band_alpha, median_width=2)
    draw_binned_band_and_median(bot, xs_dm, rmse_dm, map_rmse_y, C_DM, bin_size_iters, alpha=band_alpha, median_width=2)
    draw_binned_band_and_median(bot, xs_sm, rmse_sm, map_rmse_y, C_SM, bin_size_iters_zero, alpha=band_alpha, median_width=2)

    # --- 簡易圖例 ---
    def legend(x, y):
        items = [("Force-directed (our method)", C_PH), ("Vector MDS", C_DM), ("Stress-Majorization", C_SM)]
        dx = 0
        for label, col in items:
            if col == C_DM :
                dx += 100
            pygame.draw.line(screen, col, (x + dx, y), (x + dx + 20, y), 4)
            screen.blit(font.render(label, True, (0, 0, 0)), (x + dx + 30, y - 10))
            dx += 130

    legend(top.right - 550, top.top + 20  )
    legend(bot.right - 550, bot.top + 20 )

    pygame.display.flip()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        pygame.image.save(screen, save_path)

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
    pygame.quit()

