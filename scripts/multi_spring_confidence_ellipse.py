import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from tqdm import trange
from copy import deepcopy
from scipy.stats import chi2, gaussian_kde

# 設定字型與負號顯示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# ==== 引入你現有的 spring.py 主要接口 ============
from library.metrics import *
from library.config import *
from library.data_io import *
from library.geometry import *
from library.visulization import *
from library.physics import *
from library.initialization import *

# ==== 參數可自行調整 ============================
N_BOOTSTRAP          = 20      # 重覆次數（>300 建議跑一夜）
SPRING_JITTER        = 0.05    # 每次彈簧係數隨機 ±5 %
REPULSE_JITTER       = 0.20    # 排斥力常數 ±20 %
ELLIPSES_FILE        = "multi_confidence_ellipses.png"
KDE_COMBINED_FILE    = "combined_kde_density.png"

#-----------------run physics_simulation once (with different parameters)-------------
def _run_once(seed: int,
              k_spring_scale: float = 1.0,
              k_repulse_scale: float = 1.0):
    """
    使用給定 seed 和 jitter 比例執行一次 spring 模擬，返回最終位置與節點名稱。
    """
    np.random.seed(seed)
    vertice, dni, data, pos_matrix, fixed_pos = \
        generate_CHEN_initial_positions([600, 500])
    global SPRING_STIFFNESS_BASE, REPULSION_STRENGTH_BASE
    SPRING_STIFFNESS_BASE *= k_spring_scale
    REPULSION_STRENGTH_BASE *= k_repulse_scale
    dir_data = uploading_directional_data()
    _wrong, sh, final_pos = main_physics_simulation(
        vertice, dni, data, deepcopy(pos_matrix), dir_data, fixed_pos)
    SPRING_STIFFNESS_BASE /= k_spring_scale
    REPULSION_STRENGTH_BASE /= k_repulse_scale
    return np.asarray(final_pos), vertice
#------------------bootstrap_dynamics-----------------------
def _bootstrap_dynamics():
    """
    執行多次模擬，返回所有樣本、平均與協方差矩陣。
    """
    first_pos, vertice = _run_once(0, 1.0, 1.0)
    N_nodes = first_pos.shape[0]
    samples = np.zeros((N_BOOTSTRAP, N_nodes, 2))
    samples[0] = first_pos
    for b in trange(1, N_BOOTSTRAP, desc="Bootstrap"):
        ks = 1.0 + np.random.uniform(-SPRING_JITTER, SPRING_JITTER)
        kr = 1.0 + np.random.uniform(-REPULSE_JITTER, REPULSE_JITTER)
        pos, _ = _run_once(b, ks, kr)
        samples[b] = pos
    return samples, vertice

# ---------- draw ellipses for assigned vertice ----------------------
def _draw_multi_confidence_ellipses(ax, samples, facecolor='steelblue'):
    """
    在 ax 上為一組 2D 樣本繪製 95%/90%/85% 信心橢球。
    """
    mean = samples.mean(axis=0)
    cov = np.atleast_2d(np.cov(samples, rowvar=False))
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    conf_levels = [0.95, 0.90, 0.85]
    alphas      = [0.30, 0.20, 0.10] # opacity

    for lvl, alpha in zip(conf_levels, alphas):
        chi2_val = chi2.ppf(lvl, df=2)
        width, height = 2 * np.sqrt(vals * chi2_val)
        # 若橢球退化（長寬為零），則跳過
        if width <= 0 or height <= 0:
            continue
        angle = np.degrees(np.arctan2(*vecs[:,0][::-1]))
        e = Ellipse(xy=mean, width=width, height=height,
                    angle=angle, facecolor=facecolor,
                    edgecolor='k', lw=1, alpha=alpha, zorder=2)
        ax.add_patch(e)
    # 繪製平均點
    ax.plot(mean[0], mean[1], marker='o', color=facecolor,
            zorder=3, ms=3)

# ---------- 繪製多信心橢球並存檔 ----------------
def _plot_multi_ellipses(samples, vertice):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    cmap = plt.get_cmap('tab20')
    colors = {name: cmap(i % 20) for i, name in enumerate(vertice)}
    for idx, name in enumerate(vertice):
        node_samples = samples[:, idx, :]
        _draw_multi_confidence_ellipses(ax, node_samples, facecolor=colors[name])
        mean = node_samples.mean(axis=0)
        ax.text(mean[0] + 3, mean[1] - 3, name,
                color=colors[name], fontsize=7)
    ax.set_xlim(0, 1200)
    ax.set_ylim(750, 0)
    ax.set_aspect('equal')
    ax.set_title('95/90/85% Confidence Ellipses')
    plt.tight_layout()
    plt.savefig(ELLIPSES_FILE, dpi=300)
    plt.close()
    print(f"==> Ellipses saved as {ELLIPSES_FILE}")

# ---------- 合併 KDE 密度圖並存檔 ----------------
def _plot_kde_combined(samples, vertice):
    """
    在單一圖中合併所有節點（城市）的 KDE 密度等高面與散點，背景維持白色。

    Parameters
    ----------
    samples : ndarray, shape = (B, N_nodes, 2)
        B 為 Bootstrap 重覆次數，每個節點會有 B 個 (x, y) 座標。
    vertice : list of str, length = N_nodes
        每個節點（城市）的名稱串列，與 samples 中的第二維對應。

    修正要點:
    1. 若某節點的所有 bootstrap 點都相同或少於 2 個，跳過 KDE, 僅畫平均點。
    2. 若 bounding box 太小，強制擴展到最小寬度 min_span, 以避免網格生成錯誤。
    3. 確保各個繪圖層次 (zorder) 不會互相覆蓋掉重要資訊。
    4. 列出詳細註解，方便理解 KDE + scatter 的工作流程。
    """
    # 1. 如果 samples 不是三維陣列，或次數 < 1，就直接報錯或跳過
    if samples.ndim != 3:
        raise ValueError("samples 必須為形狀 (B, N_nodes, 2) 的三維陣列")
    B, N_nodes, dim = samples.shape
    if dim != 2:
        raise ValueError("samples 的第三維度必須為 2 (x, y)")

    # 2. 先把 Pygame 的 y 座標翻轉成左下為原點
    #    由於 Pygame 座標 y=0 在上方，y=750 在下方：
    #    新的 y_new = 750 - y_old
    #    下面直接對整個 samples 做翻轉
    samples_flipped = samples.copy()  # 避免改到原陣列
    samples_flipped[:, :, 1] = 750.0 - samples_flipped[:, :, 1]

    # 3. 建立全螢幕網格 (200×200)，範圍從 x=0→1200, y=0→750
    xmin, xmax = 0.0, 1200.0
    ymin, ymax = 0.0, 750.0
    xx, yy = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]  # xx.shape = (200,200), yy.shape = (200,200)
    grid_coords = np.vstack([xx.ravel(), yy.ravel()])  # shape = (2, 200*200)

    # 4. 準備一組可循環的序列調色盤名稱，避免顏色重覆太快
    seq_cmaps = ['Reds', 'Greens', 'Blues', 'Oranges', 'Purples', 'Greys']

    # 5. 建立整張合併圖的 Figure 與一個 Axes，並設定底色為白
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # 6. 依序處理每個節點
    for idx, name in enumerate(vertice):
        # 6.1 取出該節點的所有 B 個 bootstrap 样本 (shape = (B, 2))
        node_samples = samples_flipped[:, idx, :]

        # 6.2 如果該節點的樣本數量 < 2 或所有坐標完全相同 (無法做 KDE)，就
        #     只繪製平均位置的 marker 與 label，並跳過 KDE 下的散點密度
        #     該情況下，node_samples.mean(axis=0) = 唯一/平均坐標
        unique_pts = np.unique(node_samples, axis=0)
        if node_samples.shape[0] < 2 or unique_pts.shape[0] == 1:
            # 只畫平均點
            mean_xy = node_samples.mean(axis=0)
            # 選擇一個顏色 (cycle 避免同色)
            cmap_obj = plt.get_cmap(seq_cmaps[idx % len(seq_cmaps)])
            marker_color = cmap_obj(0.9)
            ax.plot(mean_xy[0], mean_xy[1],
                    marker='x', color=marker_color,
                    ms=8, mew=1.5, zorder=3)
            ax.text(mean_xy[0] + 0.5, mean_xy[1] + 0.5, name,
                    color=marker_color, fontsize=8,
                    fontweight='bold', zorder=4)
            # 跳過當前節點的 KDE + scatter 繪製
            continue

        # 6.3 針對該節點擬合 Gaussian KDE
        try:
            kde = gaussian_kde(node_samples.T)
        except Exception as e:
            # 若因為樣本非常集中或矩陣病態等原因出錯，就只畫散點
            print(f"節點 '{name}' 的 KDE 擬合失敗: {e}. 僅繪製原始點與平均點。")
            # 畫散點
            ax.scatter(node_samples[:, 0], node_samples[:, 1],
                       s=8, color='k', alpha=0.6, zorder=2)
            # 畫平均點
            mean_xy = node_samples.mean(axis=0)
            ax.plot(mean_xy[0], mean_xy[1],
                    marker='x', color='r', ms=8, mew=1.5, zorder=3)
            ax.text(mean_xy[0] + 0.5, mean_xy[1] + 0.5, name,
                    color='r', fontsize=8, fontweight='bold', zorder=4)
            continue
        
        # ====== 改為：先在「node_samples 上」計算每個點的 KDE 密度值，然後用 scatter 著色 ======
        # 6.4.1 先針對該節點的所有 bootstrap 點，算出它們各自的密度 d_i
        # node_samples.shape = (B,2)，我們把它的 (x,y) 堆成 (2,B) 傳給 kde
        densities = kde(node_samples.T)    # 回傳 shape=(B,) 的每個點的 f(x_i,y_i)

        # 6.4.2 選擇該節點的 colormap 與底色、marker 顏色
        cmap_obj    = plt.get_cmap(seq_cmaps[idx % len(seq_cmaps)])
        marker_color = cmap_obj(0.9)   # 之後用來畫平均點、文字

        # 6.4.3 用 scatter，colors= densities，並搭配 colormap 自動映射顏色
        #         這裡 c=densities 讓顏色深淺依照 KDE 高度變化
        ax.scatter(
            node_samples[:, 0], node_samples[:, 1],
            c=densities,               # 以密度值決定顏色
            cmap=seq_cmaps[idx % len(seq_cmaps)], # 或 cmap_obj，直接用此節點的 colormap
            s=3,                       # 半徑調小：原本是 8，改成 4 讓點更細
            alpha=0.8,                 # 可以自行調整透明度(反正顏色本身就會深淺)
            zorder=2
        )
        
        '''
        # 6.4 建立該節點專屬的網格範圍 (限制在 global 範圍內，避免出界)
        # 【改為使用全螢幕固定網格 xx, yy, grid_coords 來計算 KDE】
        zz = kde(grid_coords).reshape(xx.shape)             # 全局 200×200 網格上算密度
        zz_masked = np.ma.masked_where(zz <= 0, zz)         # 掩掉 <= 0 的值，只留正密度

        # 6.5 為該節點選擇顏色：cycle 取不同 colormap
        cmap_obj = plt.get_cmap(seq_cmaps[idx % len(seq_cmaps)])
        fill_color   = cmap_obj(0.6)  # 半透明填充
        marker_color = cmap_obj(0.9)  # 深色，用於散點與平均點

        # 6.6 在 Axes 上繪製等高面 (contourf)
        ax.contourf(
            xx, yy, zz_masked,
            levels=5,                 # 分成 5 個等級
            colors=["none"],      # 全部用同一個顏色
            alpha=0.4,                # 半透明
            zorder=1                  # 底層
        )

        # 6.7 在同一 Axes 上繪製該節點的原始 bootstrap 散點
        ax.scatter(
            node_samples[:, 0], node_samples[:, 1],
            s=8, color=marker_color,
            alpha=0.6, zorder=2
        )
        '''
        mean_xy = node_samples.mean(axis=0)
        '''
        # 6.8 繪製該節點的平均位置 (紅叉號改成 marker_color + 'x')
        ax.plot(
            mean_xy[0], mean_xy[1],
            marker='x', color=marker_color,
            ms=4, mew=1.5, zorder=3
        )
        '''
        
        # 6.9 在平均點旁邊加文字標籤 (節點名稱)
        ax.text(
            mean_xy[0] + 0.5, mean_xy[1] + 0.5,
            name,
            color=marker_color, fontsize=8,
            fontweight='bold', zorder=4
        )

    # 7. 設定整張圖的 x, y 範圍為所有節點的 globalBound
    ax.set_xlim(0, 1200)
    ax.set_ylim(0, 750)
    ax.set_aspect('equal')  # 等比例，讓密度圖與散點不失真

    # 8. 圖片標題與排版
    ax.set_title('Combined KDE Density Map')
    plt.tight_layout()

    # 9. 存檔並關閉
    plt.savefig(KDE_COMBINED_FILE, dpi=300)
    plt.close()

    # 10. 印出存檔成功訊息
    print(f"==> Combined KDE saved as {KDE_COMBINED_FILE}")

# ---------- 對外介面 ----------------------------
def bootstrap_and_plot():
    samples, vertice = _bootstrap_dynamics()
    _plot_multi_ellipses(samples, vertice)
    _plot_kde_combined(samples, vertice)
    return vertice

if __name__ == '__main__':
    bootstrap_and_plot()
