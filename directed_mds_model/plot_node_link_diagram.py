# -*- coding: utf-8 -*-
# node_link_pygame_utf8.py

import math
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Optional

import pygame


def _match_cjk_font(label_font_size: int, font_path: Optional[str] = None) -> pygame.font.Font:
    """
    取得可顯示中文的字體。優先使用 font_path，其次嘗試常見 CJK 字體，最後退回系統預設字體。
    """
    pygame.font.init()
    if font_path:
        try:
            return pygame.font.Font(font_path, label_font_size)
        except Exception:
            pass  # 失敗就往下嘗試系統字體

    # 常見中文/日文/韓文字體名稱嘗試（系統可能取的到相對應檔案）
    candidates = [
        "Noto Sans CJK TC", "Noto Sans CJK SC", "Source Han Sans TW", "Source Han Sans TC",
        "思源黑體", "微軟正黑體", "Microsoft JhengHei", "PingFang TC", "PingFang HK",
        "Heiti TC", "WenQuanYi Zen Hei"
    ]
    try:
        path = pygame.font.match_font(candidates)
        if path:
            return pygame.font.Font(path, label_font_size)
    except Exception:
        pass

    # 最後退回預設字體（可能無法完整顯示 CJK；建議傳入 font_path）
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


# ---------------- 範例（請用你自己的資料取代） ----------------
if __name__ == "__main__":
    # 假資料示意（請改成你從 data_pre_processing / directed_mds_model 產生的內容）
    vertice = ["長安", "酒泉", "敦煌", "樓蘭", "于闐", "龜茲"]
    pos = [(0, 0), (2, 0.2), (3.2, 0.1), (5.4, -0.3), (7.0, 0.6), (8.5, -0.1)]
    edges = [("長安", "酒泉"), ("酒泉", "敦煌"), ("敦煌", "樓蘭"), ("樓蘭", "于闐"), ("于闐", "龜茲")]
    graph = None  # 若不需要可忽略

    # 建議指定能顯示中文的字體路徑，例如：
    # font_path = "/path/to/NotoSansCJKtc-Regular.otf"
    font_path = None

    draw_node_link_pygame(
        graph, pos, vertice, edges,
        directed=True,
        font_path=font_path,
        interactive=False,     # 預設純靜態：沒有任何鍵盤或滑鼠控制
        save_path=None,        # 若要另存圖片就給檔名，例如 "graph.png"
        caption="節點連結圖（UTF-8, 無鍵盤控制）"
    )
