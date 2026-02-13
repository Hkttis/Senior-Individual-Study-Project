
import math
from collections import defaultdict
from typing import List, Tuple, Any, Optional
import pygame
import numpy as np
from math import sqrt

from library.config import km2pix, km2Li, theta_thr_4dir, theta_thr_8dir
from library.data_io import uploading_directional_data
from library.directions import DIR4_SIM, DIR4DIAG_RAW_SIM
from library.directions import DIR8_UNIT_SIM as unit_direction_dict



def wrong_directions_nonflip(pos_matrix, vertice, dni):
    
    directional_data = uploading_directional_data()
    
    '''direction_revising force'''
    # If we apply forces directory in direction of ideal directions, it will lead to equalibriam. 
   
    direction_dict = DIR4_SIM
    direction_dict2 = DIR4DIAG_RAW_SIM

    wrong_direction_lists = []
    for row in directional_data :
        # calculate the distance vector between nodes
        index1 = dni[row[0]] # left one in csv file
        index2 = dni[row[1]] # right one
        n1 = pos_matrix[index1]
        n2 = pos_matrix[index2]
        pos_vector = np.array([n2[0]-n1[0],n2[1]-n1[1]])
        pos_vector = pos_vector / np.linalg.norm(pos_vector) # unit vector
        if row[2] in direction_dict : # for those with rough direction only (東南西北), theta must smaller than pi/4
            cos_similarity = np.dot(pos_vector,direction_dict[row[2]])
            if np.dot(pos_vector,direction_dict[row[2]]) < math.cos(theta_thr_4dir) : # 1/sqrt(2) : # apply force if being on the wrong direction
                wrong_direction_lists.append(row)
                
        else : # for those with more specific direction (東南、西北), the cos(theta) between directional vector and pos_vector must be over cos(pi/8)
            cos_similarity = np.dot(pos_vector,direction_dict2[row[2]])/sqrt(2)
            if cos_similarity < math.cos(theta_thr_8dir):  # 0.924 : theta > pi/8
                wrong_direction_lists.append(row)
        
    return wrong_direction_lists

def wrong_directions_flip(pos_matrix, vertice, dni):
    
    directional_data = uploading_directional_data()
    
    '''direction_revising force'''
    # If we apply forces directory in direction of ideal directions, it will lead to equalibriam. 
    # y-axis is toward negative side
    # y-axis is toward negative side
    direction_dict = DIR4_SIM
    direction_dict2 = DIR4DIAG_RAW_SIM
    wrong_direction_lists = []
    for row in directional_data :
        # calculate the distance vector between nodes
        index1 = dni[row[0]] # left one in csv file
        index2 = dni[row[1]] # right one
        n1 = pos_matrix[index1]
        n2 = pos_matrix[index2]
        pos_vector = np.array([n2[0]-n1[0],n2[1]-n1[1]])
        pos_vector = pos_vector / np.linalg.norm(pos_vector) # unit vector
        if row[2] in direction_dict : # for those with rough direction only (東南西北), theta must smaller than pi/4
            cos_similarity = np.dot(pos_vector,direction_dict[row[2]])
            if np.dot(pos_vector,direction_dict[row[2]]) < 0 : # 1/sqrt(2) : # apply force if being on the wrong direction
                wrong_direction_lists.append(row)
                
        else : # for those with more specific direction (東南、西北), the cos(theta) between directional vector and pos_vector must be over cos(pi/8)
            cos_similarity = np.dot(pos_vector,direction_dict2[row[2]])/sqrt(2)
            if cos_similarity < 1/sqrt(2):  # 0.924 : theta > pi/8
                wrong_direction_lists.append(row)
        
    return wrong_direction_lists

def directional_stress_nonflip(pos_matrix, vertice, dni):
    # Directional stress is not affected by scaling ( changing of units )
    dir_stress = 0
    directional_data = uploading_directional_data()
    
    direction_dict = DIR4_SIM
    direction_dict2 = DIR4DIAG_RAW_SIM
    
    wrong_direction_lists = []
    for row in directional_data :
        # calculate the distance vector between nodes
        index1 = dni[row[0]] # left one in csv file
        index2 = dni[row[1]] # right one
        n1 = pos_matrix[index1]
        n2 = pos_matrix[index2]
        pos_vector = np.array([n2[0]-n1[0],n2[1]-n1[1]])
        pos_vector = pos_vector / np.linalg.norm(pos_vector) # unit vector
        if row[2] in direction_dict : # for those with rough direction only (東南西北), theta must smaller than pi/4
            cos_similarity = np.dot(pos_vector,direction_dict[row[2]])
            if np.dot(pos_vector,direction_dict[row[2]]) < 0 : # 1/sqrt(2) : # apply force if being on the wrong direction
                dir_stress += -cos_similarity
                
        else : # for those with more specific direction (東南、西北), the cos(theta) between directional vector and pos_vector must be over cos(pi/8)
            cos_similarity = np.dot(pos_vector,direction_dict2[row[2]])/sqrt(2)
            if cos_similarity < 1/sqrt(2):  # 0.924 : theta > pi/8
                dir_stress += -cos_similarity + 1/sqrt(2)
    return dir_stress
    

def _match_cjk_font(label_font_size: int, font_path: Optional[str] = None) -> pygame.font.Font:
    pygame.font.init()
    if font_path:
        return pygame.font.Font(font_path, label_font_size)

    candidates = [
        "Noto Sans CJK TC", "Noto Sans CJK SC", "Source Han Sans TW", "Source Han Sans TC",
        "思源黑體", "微軟正黑體", "Microsoft JhengHei", "PingFang TC", "PingFang HK",
        "Heiti TC", "WenQuanYi Zen Hei"
    ]
    # Try to resolve a concrete font file path
    path = pygame.font.match_font(",".join(candidates))
    if path:
        return pygame.font.Font(path, label_font_size)

    # Fallback: try SysFont and verify it can render CJK
    for name in candidates:
        try:
            f = pygame.font.SysFont(name, label_font_size)
            if f.size("鄯善")[0] > 0:
                return f
        except Exception:
            pass

    # Last resort (may not show CJK)
    return pygame.font.Font(pygame.font.get_default_font(), label_font_size)

def draw_node_link_pygame(
    pos: List[Tuple[float, float]],
    vertice: List[Any],
    edges: List[Tuple[Any, Any]],
    *,
    directed: bool = False,
    window_size: Tuple[int, int] = (1200, 750),
    bg_color: Tuple[int, int, int] = (250, 250, 250),
    node_color: Tuple[int, int, int] = (30, 144, 255),
    edge_color: Tuple[int, int, int] = (60, 60, 60),
    label_color: Tuple[int, int, int] = (20, 20, 20),
    node_radius_base: int = 6,
    label_font_size: int = 16,
    font_path: Optional[str] = None,
    caption: str = "節點連結圖（UTF-8）",
    interactive: bool = True,   # 預設關閉所有互動；不使用任何鍵盤快捷鍵
    save_path: Optional[str] = None,  # 若給路徑，會把畫面另存 PNG
) -> None:
    """
    以 Pygame 繪製節點連結圖（支援中文標籤、UTF-8）。
    - 不使用鍵盤快捷鍵；若 interactive=True，僅支援滑鼠拖曳與滾輪縮放。
    - 需要你提供的 (graph, pos, vertice, edges)。pos[i] 對應 vertice[i] 的座標。

    Parameters
    ----------
    directed : 畫有向邊（箭頭）
    interactive : True 啟用滑鼠拖曳/縮放；False 則純靜態（只負責顯示與存檔）
    font_path : 指定一個支援中文的 .ttf/.otf/.ttc 檔（建議）
    save_path : 另存成 PNG 檔路徑（例如 "graph.png"）
    """
    # 名稱→索引
    name_to_idx = {name: i for i, name in enumerate(vertice)}

    # 邊集合（無向圖去重）
    if directed:
        draw_edges = [(u, v) for (u, v) in edges if u in name_to_idx and v in name_to_idx]
    else:
        def norm(u, v):
            return (u, v) if str(u) <= str(v) else (v, u)
        draw_edges = list({norm(u, v) for (u, v) in edges if u in name_to_idx and v in name_to_idx})

    # 度數（給節點大小輕微加權）
    degree = defaultdict(int)
    for u, v in draw_edges:
        degree[u] += 1
        degree[v] += 1

    # --- 計算視窗映射（自動置中與縮放） ---
    xs = [pos[i][0] for i in range(len(vertice))]
    ys = [pos[i][1] for i in range(len(vertice))]
    min_x, max_x = (min(xs), max(xs)) if xs else (0, 1)
    min_y, max_y = (min(ys), max(ys)) if ys else (0, 1)
    data_w = max(max_x - min_x, 1e-6)
    data_h = max(max_y - min_y, 1e-6)

    W, H = window_size
    margin = 60
    scale = 0.9 * min((W - 2 * margin) / data_w, (H - 2 * margin) / data_h)
    offset = [
        (W - scale * (min_x + max_x)) / 2.0,
        (H - scale * (min_y + max_y)) / 2.0,
    ]

    def world_to_screen(pxy):
        return (pxy[0] * scale + offset[0], pxy[1] * scale + offset[1])

    def node_radius(name):
        return int(node_radius_base + 1.5 * math.sqrt(degree.get(name, 1)))

    def draw_arrow(surface, color, a, b, width=2, head_len=12, head_angle=28):
        pygame.draw.aaline(surface, color, a, b)
        dx, dy = b[0] - a[0], b[1] - a[1]
        ang = math.atan2(dy, dx)
        left = (b[0] - head_len * math.cos(ang - math.radians(head_angle)),
                b[1] - head_len * math.sin(ang - math.radians(head_angle)))
        right = (b[0] - head_len * math.cos(ang + math.radians(head_angle)),
                 b[1] - head_len * math.sin(ang + math.radians(head_angle)))
        pygame.draw.polygon(surface, color, [b, left, right])

    # --- Pygame 視窗 ---
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption(caption)
    clock = pygame.time.Clock()
    font = _match_cjk_font(label_font_size, font_path)

    # 預先算好螢幕座標
    scr_pos = [world_to_screen(pos[i]) for i in range(len(vertice))]

    # 互動狀態（僅在 interactive=True 使用）
    dragging = False
    drag_start = (0, 0)
    offset_start = (0, 0)

    # ----------- 繪圖函式 -----------
    def render_frame():
        screen.fill(bg_color)

        # edges
        if directed:
            for (u, v) in draw_edges:
                a = scr_pos[name_to_idx[u]]
                b = scr_pos[name_to_idx[v]]
                draw_arrow(screen, edge_color, a, b, width=2)
        else:
            for (u, v) in draw_edges:
                a = scr_pos[name_to_idx[u]]
                b = scr_pos[name_to_idx[v]]
                pygame.draw.aaline(screen, edge_color, a, b)

        # nodes
        for name in vertice:
            i = name_to_idx[name]
            x, y = scr_pos[i]
            pygame.draw.circle(screen, node_color, (int(x), int(y)), node_radius(name))

        # labels（UTF-8）
        for name in vertice:
            i = name_to_idx[name]
            x, y = scr_pos[i]
            label_surf = font.render(str(name), True, label_color)
            screen.blit(label_surf, (x + 8, y + 4))

        pygame.display.flip()

    # 先畫一張
    render_frame()

    # 若指定存檔，就把目前畫面存成 PNG
    if save_path:
        pygame.image.save(screen, save_path)

    # event loop：不使用任何鍵盤快捷鍵；interactive=False 時只處理關閉視窗
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if not interactive:
                continue  # 完全不處理其他輸入

            # 只啟用滑鼠拖曳與滾輪縮放（沒有鍵盤）
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                dragging = True
                drag_start = pygame.mouse.get_pos()
                offset_start = (offset[0], offset[1])
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging = False
            elif event.type == pygame.MOUSEMOTION and dragging:
                mx, my = pygame.mouse.get_pos()
                dx, dy = mx - drag_start[0], my - drag_start[1]
                offset[0] = offset_start[0] + dx
                offset[1] = offset_start[1] + dy
                # 位置更新後需重算螢幕座標
                for i in range(len(scr_pos)):
                    scr_pos[i] = world_to_screen(pos[i])
                render_frame()
            elif event.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                # 滑鼠座標對應的世界座標（縮放前）
                wx = (mx - offset[0]) / scale
                wy = (my - offset[1]) / scale
                zoom = 1.15 if event.y > 0 else (1 / 1.15)
                scale *= zoom
                # 調整 offset 以維持滑鼠下世界點不飄移
                offset[0] = mx - wx * scale
                offset[1] = my - wy * scale
                for i in range(len(scr_pos)):
                    scr_pos[i] = world_to_screen(pos[i])
                render_frame()

        clock.tick(60)

    pygame.quit()

def draw_node_link_pygame_dirarrow(
    pos: List[Tuple[float, float]],
    vertice: List[Any],
    edges: List[Tuple[Any, Any]],
    *,
    directed: bool = False,                 # 原參數：是否把 edges 全畫成箭頭（沿邊方向）
    window_size: Tuple[int, int] = (1200, 750),
    bg_color: Tuple[int, int, int] = (250, 250, 250),
    node_color: Tuple[int, int, int] = (30, 144, 255),
    edge_color: Tuple[int, int, int] = (60, 60, 60),
    label_color: Tuple[int, int, int] = (20, 20, 20),
    node_radius_base: int = 6,
    label_font_size: int = 16,
    font_path: Optional[str] = None,
    caption: str = "節點連結圖（UTF-8）",
    interactive: bool = True,
    save_path: Optional[str] = None,

    # === 新增：用來畫羅盤方向箭頭（通常來自 sel_data） ===
    # 期待格式：[(src_name, dst_name, dir_str), ...] 例：("鄯善","且末","東南")
    directional_data: Optional[List[Tuple[Any, Any, str]]] = None,
    dir_arrow_color: Tuple[int, int, int] = (200, 0, 0),
    dir_arrow_len: float = 28.0,      # 單支羅盤箭頭的標準長度（像素）
    dir_arrow_width: int = 2,         # 羅盤箭頭的線寬
    dir_head_len: float = 10.0,       # 箭頭頭長度
    dir_head_angle: float = 28.0,     # 箭頭頭張角（度）
    dir_mid_offset: float = 0.0,      # 從邊中點沿方向再位移一點點（像素）
) -> None:
    """
    以 Pygame 繪製節點連結圖，並可「額外」疊加羅盤方向箭頭（由 directional_data + unit_direction_dict 決定）。
    注意：羅盤箭頭方向取自 compass（東南西北…），畫在 u→v 的「線段中點」附近，與邊線無必然同向。
    """

    # --- 名稱→索引 ---
    name_to_idx = {name: i for i, name in enumerate(vertice)}

    # --- 邊集合（若 undirected 就去重） ---
    if directed:
        draw_edges = [(u, v) for (u, v) in edges if u in name_to_idx and v in name_to_idx]
    else:
        def _norm(u, v): return (u, v) if str(u) <= str(v) else (v, u)
        draw_edges = list({_norm(u, v) for (u, v) in edges if u in name_to_idx and v in name_to_idx})

    # --- 度數（給節點大小輕微加權） ---
    degree = defaultdict(int)
    for u, v in draw_edges:
        degree[u] += 1
        degree[v] += 1

    # --- 視窗映射（置中＋縮放） ---
    xs = [pos[i][0] for i in range(len(vertice))]
    ys = [pos[i][1] for i in range(len(vertice))]
    min_x, max_x = (min(xs), max(xs)) if xs else (0.0, 1.0)
    min_y, max_y = (min(ys), max(ys)) if ys else (0.0, 1.0)
    data_w = max(max_x - min_x, 1e-6)
    data_h = max(max_y - min_y, 1e-6)

    W, H = window_size
    margin = 60
    scale = 0.9 * min((W - 2 * margin) / data_w, (H - 2 * margin) / data_h)
    offset = [
        (W - scale * (min_x + max_x)) / 2.0,
        (H - scale * (min_y + max_y)) / 2.0,
    ]

    def world_to_screen(pxy):
        return (pxy[0] * scale + offset[0], pxy[1] * scale + offset[1])

    def node_radius(name):
        return int(node_radius_base + 1.5 * math.sqrt(degree.get(name, 1)))

    # 畫沿兩點方向的箭頭（一般有向邊用）
    def draw_edge_arrow(surface, color, a, b, width=2, head_len=12, head_angle=28):
        pygame.draw.aaline(surface, color, a, b)
        dx, dy = b[0] - a[0], b[1] - a[1]
        ang = math.atan2(dy, dx)
        left = (b[0] - head_len * math.cos(ang - math.radians(head_angle)),
                b[1] - head_len * math.sin(ang - math.radians(head_angle)))
        right = (b[0] - head_len * math.cos(ang + math.radians(head_angle)),
                 b[1] - head_len * math.sin(ang + math.radians(head_angle)))
        pygame.draw.polygon(surface, color, [b, left, right])

    # 新增：畫「羅盤方向箭頭」（不看邊向量；只看 unit_direction_dict）
    def draw_compass_arrow(surface, color, center_xy, unit_vec_2d,
                           length=28.0, width=2, head_len=10.0, head_angle=28.0, advance=0.0):
        """
        在 center_xy 畫一支朝向 unit_vec_2d 的箭頭；整支箭長 'length'。
        若 advance>0，會先沿 unit_vec 方向把箭整體平移一點，避免壓到節點或文字。
        """
        ux, uy = float(unit_vec_2d[0]), float(unit_vec_2d[1])
        # 正規化（保險）
        L = math.hypot(ux, uy) or 1.0
        ux, uy = ux / L, uy / L

        cx, cy = center_xy
        # 先位移
        cx += ux * advance
        cy += uy * advance

        # 以 center 為中點，畫一段 [p -> q]
        half = length * 0.5
        px, py = cx - ux * half, cy - uy * half
        qx, qy = cx + ux * half, cy + uy * half

        pygame.draw.aaline(surface, color, (px, py), (qx, qy), True)

        ang = math.atan2(qy - py, qx - px)
        left = (qx - head_len * math.cos(ang - math.radians(head_angle)),
                qy - head_len * math.sin(ang - math.radians(head_angle)))
        right = (qx - head_len * math.cos(ang + math.radians(head_angle)),
                 qy - head_len * math.sin(ang + math.radians(head_angle)))
        pygame.draw.polygon(surface, color, [(qx, qy), left, right])

    # --- Pygame 視窗 ---
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption(caption)
    clock = pygame.time.Clock()

    # 字型（沿用你原本的挑字型函式）
    font = _match_cjk_font(label_font_size, font_path)

    # 預先算好螢幕座標
    scr_pos = [world_to_screen(pos[i]) for i in range(len(vertice))]

    # 互動狀態
    dragging = False
    drag_start = (0, 0)
    offset_start = (0, 0)

    def render_frame():
        screen.fill(bg_color)

        # --- 先畫邊 ---
        if directed:
            for (u, v) in draw_edges:
                a = scr_pos[name_to_idx[u]]
                b = scr_pos[name_to_idx[v]]
                draw_edge_arrow(screen, edge_color, a, b, width=2)
        else:
            for (u, v) in draw_edges:
                a = scr_pos[name_to_idx[u]]
                b = scr_pos[name_to_idx[v]]
                pygame.draw.aaline(screen, edge_color, a, b)

        # --- 疊加「羅盤方向箭頭」(由 directional_data + unit_direction_dict 決定) ---
        if directional_data:
            for (src, dst, dstr) in directional_data:
                if src not in name_to_idx or dst not in name_to_idx:
                    continue
                if dstr not in unit_direction_dict:
                    continue  # 不在你的字典，就跳過

                a = scr_pos[name_to_idx[src]]
                b = scr_pos[name_to_idx[dst]]
                # 邊中點
                mx, my = ( (a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5 )

                # 羅盤方向單位向量（以你檔案中的定義為準）
                uvec = unit_direction_dict[dstr]

                # 在中點位置，沿「羅盤」方向畫出短箭頭（純顯示約束方向）
                draw_compass_arrow(
                    screen, dir_arrow_color, (mx, my), uvec,
                    length=dir_arrow_len, width=dir_arrow_width,
                    head_len=dir_head_len, head_angle=dir_head_angle,
                    advance=dir_mid_offset
                )

        # --- 畫節點 ---
        for name in vertice:
            i = name_to_idx[name]
            x, y = scr_pos[i]
            pygame.draw.circle(screen, node_color, (int(x), int(y)), node_radius(name))

        # --- 標籤 ---
        for name in vertice:
            i = name_to_idx[name]
            x, y = scr_pos[i]
            label_surf = font.render(str(name), True, label_color)
            screen.blit(label_surf, (x + 8, y + 4))

        pygame.display.flip()

    # 初次繪製
    render_frame()

    # 存檔（如指定）
    if save_path:
        pygame.image.save(screen, save_path)

    # 互動 loop（僅滑鼠拖曳與滾輪縮放）
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if not interactive:
                continue

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                dragging = True
                drag_start = pygame.mouse.get_pos()
                offset_start = (offset[0], offset[1])

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging = False

            elif event.type == pygame.MOUSEMOTION and dragging:
                mx, my = pygame.mouse.get_pos()
                dx, dy = mx - drag_start[0], my - drag_start[1]
                offset[0] = offset_start[0] + dx
                offset[1] = offset_start[1] + dy
                for i in range(len(scr_pos)):
                    scr_pos[i] = world_to_screen(pos[i])
                render_frame()

            elif event.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                wx = (mx - offset[0]) / scale
                wy = (my - offset[1]) / scale
                zoom = 1.15 if event.y > 0 else (1 / 1.15)
                scale *= zoom
                offset[0] = mx - wx * scale
                offset[1] = my - wy * scale
                for i in range(len(scr_pos)):
                    scr_pos[i] = world_to_screen(pos[i])
                render_frame()

        clock.tick(60)

    pygame.quit()


def animate_node_link_pygame(
    pos_history: List,                 # list of (n x 2) numpy arrays or list[list[float]]
    vertice: List[Any],
    edges: List[Tuple[Any, Any]],
    *,
    window_size: Tuple[int, int] = (1200, 750),
    bg_color: Tuple[int, int, int] = (250, 250, 250),
    node_color: Tuple[int, int, int] = (30, 144, 255),
    edge_color: Tuple[int, int, int] = (60, 60, 60),
    label_color: Tuple[int, int, int] = (20, 20, 20),
    node_radius: int = 6,
    label_font_size: int = 16,
    font_path: Optional[str] = None,
    caption: str = "Directed MDS — Iteration Animation",
    fps: int = 30,                      # slightly higher for smoother feel
    directed: bool = False,
    save_frames_pattern: Optional[str] = None,  # e.g. "frames/frame_%04d.bmp"
    antialias_edges: bool = False,
    interp_frames: int = 4              # NEW: # of in-between frames between iterations
) -> None:
    """
    Flicker-free, smooth animation:
      - single back-buffer blit + single flip per frame
      - integer pixel coords to avoid sub-pixel shimmer
      - stable global transform across all frames
      - cached label surfaces
      - optional tweening between solver iterations
    """
    import numpy as _np

    n = len(vertice)
    name_to_idx = {name: i for i, name in enumerate(vertice)}

    # Normalize edges (dedup if undirected)
    if directed:
        draw_edges = [(u, v) for (u, v) in edges if u in name_to_idx and v in name_to_idx]
    else:
        def _norm(u, v): return (u, v) if str(u) <= str(v) else (v, u)
        draw_edges = list({_norm(u, v) for (u, v) in edges if u in name_to_idx and v in name_to_idx})

    # -------- Stable world->screen transform over whole history (no view breathing) --------
    # (same spirit as the fill→draw→flip pattern used elsewhere)  :contentReference[oaicite:5]{index=5}
    all_x, all_y = [], []
    for frame in pos_history:
        all_x.extend([frame[i][0] for i in range(n)])
        all_y.extend([frame[i][1] for i in range(n)])
    min_x, max_x = (min(all_x), max(all_x)) if all_x else (0.0, 1.0)
    min_y, max_y = (min(all_y), max(all_y)) if all_y else (0.0, 1.0)

    W, H = window_size
    margin = 60
    data_w = max(max_x - min_x, 1e-6)
    data_h = max(max_y - min_y, 1e-6)
    base_scale = 0.9 * min((W - 2 * margin) / data_w, (H - 2 * margin) / data_h)
    offset = [
        (W - base_scale * (min_x + max_x)) / 2.0,
        (H - base_scale * (min_y + max_y)) / 2.0,
    ]
    scale = base_scale

    def world_to_screen(px, py):
        # integers prevent sub-pixel shimmer on lines & circles
        return (int(px * scale + offset[0]), int(py * scale + offset[1]))

    # -------- Pygame init (double buffer + flip once per frame) --------
    pygame.init()
    flags = pygame.RESIZABLE | pygame.DOUBLEBUF
    try:
        screen = pygame.display.set_mode((W, H), flags, vsync=1)  # vsync when available
    except TypeError:
        screen = pygame.display.set_mode((W, H), flags)
    pygame.display.set_caption(caption)
    clock = pygame.time.Clock()

    # Back buffer: draw -> blit -> single flip   :contentReference[oaicite:6]{index=6}
    back = pygame.Surface((W, H)).convert()

    # Font & cached labels (UTF-8/Chinese via your helper)  :contentReference[oaicite:7]{index=7}
    font = _match_cjk_font(label_font_size, font_path)
    label_cache = [font.render(str(name), True, label_color).convert_alpha() for name in vertice]

    # Timer-driven frames for steady cadence
    FRAME_TICK = pygame.USEREVENT + 1
    pygame.time.set_timer(FRAME_TICK, max(1, int(1000 / max(1, fps))))

    dragging = False
    drag_start = (0, 0)
    offset_start = (0.0, 0.0)

    # ---------- Tweening helper ----------
    # Builds an interpolated "virtual" frame between i and i+1
    def _blend(i: int, alpha: float):
        """alpha in [0,1). When i == last, alpha is ignored (returns last)."""
        if i >= len(pos_history) - 1 or alpha <= 0.0:
            return pos_history[i]
        a = _np.asarray(pos_history[i], dtype=_np.float32)
        b = _np.asarray(pos_history[i + 1], dtype=_np.float32)
        return (1.0 - alpha) * a + alpha * b

    # Draw one (possibly interpolated) frame
    def render_frame(base_idx: int, alpha: float):
        back.fill(bg_color)

        frame = _blend(base_idx, alpha)
        scr = [world_to_screen(frame[i][0], frame[i][1]) for i in range(n)]

        # Edges (non-AA by default to avoid shimmer)
        if antialias_edges:
            for (u, v) in draw_edges:
                pygame.draw.aaline(back, edge_color, scr[name_to_idx[u]], scr[name_to_idx[v]])
        else:
            for (u, v) in draw_edges:
                pygame.draw.line(back, edge_color, scr[name_to_idx[u]], scr[name_to_idx[v]], 1)

        # Nodes + labels
        for i in range(n):
            pygame.draw.circle(back, node_color, scr[i], node_radius)
        for i in range(n):
            x, y = scr[i]
            back.blit(label_cache[i], (x + 8, y + 4))

        # HUD
        hud = font.render(
            f"Iteration: {min(base_idx+1, len(pos_history))}/{len(pos_history)}  "
            f"tween: {alpha:.2f}", True, (80, 80, 80)
        )
        back.blit(hud, (12, 10))

        # One blit + one flip → no flicker   :contentReference[oaicite:8]{index=8}
        screen.blit(back, (0, 0))
        pygame.display.flip()

        if save_frames_pattern:
            pygame.image.save(screen, save_frames_pattern % (base_idx * (interp_frames + 1) + int(alpha * (interp_frames + 1))))

    # ---- Main loop: advance by sub-frames for smooth motion ----
    base_idx = 0
    sub = 0
    running = True
    need_redraw = True

    # (Optional) limit event queue to keep responsiveness crisp
    pygame.event.set_allowed([pygame.QUIT, FRAME_TICK, pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP,
                              pygame.MOUSEMOTION, pygame.MOUSEWHEEL])

    # First paint
    render_frame(base_idx, 0.0)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == FRAME_TICK:
                # advance tween
                sub += 1
                if sub > interp_frames:
                    sub = 0
                    base_idx = (base_idx + 1) % max(1, len(pos_history))
                need_redraw = True

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                dragging = True
                drag_start = pygame.mouse.get_pos()
                offset_start = (offset[0], offset[1])

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging = False

            elif event.type == pygame.MOUSEMOTION and dragging:
                mx, my = pygame.mouse.get_pos()
                dx, dy = mx - drag_start[0], my - drag_start[1]
                offset[0] = offset_start[0] + dx
                offset[1] = offset_start[1] + dy
                need_redraw = True

            elif event.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                wx = (mx - offset[0]) / scale
                wy = (my - offset[1]) / scale
                zoom = 1.15 if event.y > 0 else (1 / 1.15)
                scale *= zoom
                offset[0] = mx - wx * scale
                offset[1] = my - wy * scale
                need_redraw = True

        if need_redraw:
            alpha = 0.0 if interp_frames <= 0 else (sub / (interp_frames + 1.0))
            render_frame(base_idx, alpha)
            need_redraw = False

        # If vsync is unavailable, this caps the loop tightly to the intended fps
        clock.tick(fps)

    pygame.quit()

