
import math
from collections import defaultdict
from typing import List, Tuple, Any, Optional
import pygame
from library.config import km2pix, km2Li

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

def alignment_and_scaling(pos_matrix, vertice, dni, refer_pos) :
    """
    Scale all coordinates by scale and translate so that the point labeled '鄯善' matches refer_pos.
    Raises ValueError
        If '鄯善' is not found or refer_pos is invalid.
    """
    # Basic validation
    if not isinstance(refer_pos, (list, tuple)) or len(refer_pos) != 2:
        raise ValueError("refer_pos must be a 2-element list/tuple like [x, y].")
    if len(pos_matrix) != len(vertice):
        raise ValueError("pos_matrix and vertice must have the same length.")

    # Find the anchor index (first occurrence if duplicated)
    try:
        anchor_idx = dni["鄯善"]
    except ValueError:
        raise ValueError("Label '鄯善' not found in vertice.") from None

    # 1) Scale by 1/10 *1.2
    scale = 0.1 * 1.5
    scaled = [[x * scale, y * scale] for x, y in pos_matrix]

    # 2) Compute translation so '鄯善' lands at refer_pos
    anchor_x, anchor_y = scaled[anchor_idx]
    dx = refer_pos[0] - anchor_x
    dy = refer_pos[1] - anchor_y

    # 3) Apply translation to all points
    aligned = [[x + dx, y + dy] for x, y in scaled]

    return aligned

def animate_node_link_pygame(
    pos_history: List,                 # list of (n x 2) arrays or list[list[float]]
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
    font_path: Optional[str] = None,   # e.g. r"C:\Windows\Fonts\msjh.ttc" or Noto CJK path
    caption: str = "Directed MDS — Iteration Animation",
    fps: int = 24,
    directed: bool = False,
    save_frames_pattern: Optional[str] = None,  # e.g. "frames/frame_%04d.bmp" (uncompressed)
    antialias_edges: bool = False               # AA can shimmer; keep False for rock-solid lines
) -> None:
    """
    Flicker-free animation:
      - single back-buffer blit + single flip per frame
      - integer pixel coords to avoid sub-pixel shimmer
      - stable global transform across frames
      - CJK labels via pygame.freetype (UTF-8 safe)
    """
    n = len(vertice)
    name_to_idx = {name: i for i, name in enumerate(vertice)}

    # Normalize edges
    if directed:
        draw_edges = [(u, v) for (u, v) in edges if u in name_to_idx and v in name_to_idx]
    else:
        def norm(u, v): return (u, v) if str(u) <= str(v) else (v, u)
        draw_edges = list({norm(u, v) for (u, v) in edges if u in name_to_idx and v in name_to_idx})

    # ---- Stable world->screen transform over the whole history ----
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
        # Use integers to avoid AA shimmer/tearing artifacts
        return (int(px * scale + offset[0]), int(py * scale + offset[1]))

    # ---- Pygame init (double buffer, vsync if available) ----
    pygame.init()
    flags = pygame.RESIZABLE | pygame.DOUBLEBUF
    try:
        screen = pygame.display.set_mode((W, H), flags, vsync=1)  # SDL2 vsync (pygame 2+)
    except TypeError:
        screen = pygame.display.set_mode((W, H), flags)            # older pygame: no vsync kwarg
    pygame.display.set_caption(caption)
    clock = pygame.time.Clock()

    # Back buffer: draw everything here, blit once → flip once
    back = pygame.Surface((W, H)).convert()

    
    # Font & label cache (UTF-8 safe via freetype)
    font = _match_cjk_font(label_font_size, font_path)
    label_cache = [font.render(str(name), True, label_color) for name in vertice]
    

    # Timer-driven frames (stable cadence)
    FRAME_TICK = pygame.USEREVENT + 1
    pygame.time.set_timer(FRAME_TICK, max(1, int(1000 / max(1, fps))))

    dragging = False
    drag_start = (0, 0)
    offset_start = (0.0, 0.0)

    frame_idx = 0
    running = True
    need_redraw = True

    def render_frame(idx: int):
        back.fill(bg_color)
        frame = pos_history[idx]
        scr = [world_to_screen(frame[i][0], frame[i][1]) for i in range(n)]

        # Edges
        if antialias_edges:
            for (u, v) in draw_edges:
                pygame.draw.aaline(back, edge_color, scr[name_to_idx[u]], scr[name_to_idx[v]])
        else:
            for (u, v) in draw_edges:
                pygame.draw.line(back, edge_color, scr[name_to_idx[u]], scr[name_to_idx[v]], 1)

        # Nodes
        for i in range(n):
            pygame.draw.circle(back, node_color, scr[i], node_radius)

        # Labels (CJK OK)
        for i in range(n):
            x, y = scr[i]
            back.blit(label_cache[i], (x + 8, y + 4))

        # HUD
        hud_surf = font.render(f"Iteration: {idx+1}/{len(pos_history)}", True, (80, 80, 80))
        back.blit(hud_surf, (12, 10))

        # One blit + one flip → no flicker
        screen.blit(back, (0, 0))
        pygame.display.flip()

        if save_frames_pattern:
            pygame.image.save(screen, save_frames_pattern % idx)

    render_frame(frame_idx)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == FRAME_TICK:
                frame_idx = (frame_idx + 1) % len(pos_history)
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
            render_frame(frame_idx)
            need_redraw = False

        clock.tick(max(60, fps * 3))  # responsive input, but one flip per frame
    pygame.quit()
