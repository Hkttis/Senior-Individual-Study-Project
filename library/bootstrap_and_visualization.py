import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from tqdm import trange
from copy import deepcopy
from scipy.stats import chi2, gaussian_kde
from matplotlib.colors import LogNorm

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


ELLIPSES_FILE        = "C:/Users/justi/Desktop/project/results/multi_confidence_ellipses.png"
KDE_COMBINED_FILE    = "C:/Users/justi/Desktop/project/results/combined_kde_density.png"

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
    global SPRING_STIFFNESS_BASE, REPULSION_STRENGTH_BASE, DIRECTIONAL_FORCE_MAGNITUDE_BASE
    spring_stiffness = SPRING_STIFFNESS_BASE
    repulsion_strength = REPULSION_STRENGTH_BASE
    directional_force_magnitude = DIRECTIONAL_FORCE_MAGNITUDE_BASE
    spring_stiffness *= k_spring_scale
    repulsion_strength *= k_repulse_scale
    
    dir_data = uploading_directional_data()
    _wrong, sh, final_pos = main_physics_simulation(
        vertice, dni, data, deepcopy(pos_matrix), dir_data, fixed_pos, spring_stiffness, repulsion_strength, directional_force_magnitude)
    return np.asarray(final_pos), vertice

#------------------bootstrap_dynamics-----------------------
def bootstrap_dynamics(N_BOOTSTRAP,SPRING_JITTER,REPULSE_JITTER):
    """
    執行多次模擬，返回所有樣本、平均與協方差矩陣。
    """
    first_pos, vertice = _run_once(0, 1.0, 1.0)
    N_nodes = first_pos.shape[0]
    samples = np.zeros((N_BOOTSTRAP, N_nodes, 2))
    samples[0] = first_pos
    for b in trange(1, N_BOOTSTRAP, desc="Bootstrap"):
        ks = np.random.normal(1.0, SPRING_JITTER/2) # mean = 1, std = spring_jilter/2
        kr = np.random.normal(1.0, REPULSE_JITTER/2 )
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
def plot_multi_ellipses(samples, vertice):
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

def plot_kde_combined(samples, vertice):
    """
    在單一圖中合併所有節點（城市）的 KDE 散點，並：
      1. 統一採用全局相同的顏色比例尺 (colorbar)。
      2. 為每個城市加繪“5 段等高線”，線條細、顏色淡，且與該城市同色調。
    背景維持白色。

    Parameters
    ----------
    samples : ndarray, shape = (B, N_nodes, 2)
        B 為 Bootstrap 重覆次數，每個節點會有 B 個 (x, y) 座標。
    vertice : list of str, length = N_nodes
        每個節點（城市）的名稱串列，與 samples 中的第二維對應。
    """
    # 1. 檢查輸入
    if samples.ndim != 3:
        raise ValueError("samples 必須為形狀 (B, N_nodes, 2) 的三維陣列")
    B, N_nodes, dim = samples.shape
    if dim != 2:
        raise ValueError("samples 的第三維度必須為 2 (x, y)")
    if B < 1:
        raise ValueError("samples 的第一維 B 必須 >= 1")

    # 2. 把 Pygame 的 y 座標翻轉成左下為原點
    samples_flipped = samples.copy()
    samples_flipped[:, :, 1] = 750.0 - samples_flipped[:, :, 1]

    # 3. 建立全螢幕網格 (200×200)，範圍從 x=0→1200, y=0→750
    xmin, xmax = 0.0, 1200.0
    ymin, ymax = 0.0, 750.0
    xx, yy = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
    grid_coords = np.vstack([xx.ravel(), yy.ravel()])  # shape = (2, 200*200)

    # 4. 準備一組可循環的序列調色盤名稱，避免顏色重覆太快
    seq_cmaps = ['Reds', 'Greens', 'Blues', 'Oranges', 'Purples', 'Greys']

    # === 新增：先計算「所有節點所有 bootstrap 點」的 KDE 密度，以便得到全局 vmin, vmax ===
    all_densities = []
    for idx in range(N_nodes):
        pts = samples_flipped[:, idx, :]
        unique_pts = np.unique(pts, axis=0)
        if pts.shape[0] >= 2 and unique_pts.shape[0] > 1:
            kde_tmp = gaussian_kde(pts.T)
            dens_tmp = kde_tmp(pts.T)  # shape = (B,)
            all_densities.append(dens_tmp)
    if len(all_densities) > 0:
        all_densities = np.hstack(all_densities)
        vmin_all = all_densities.min()
        vmax_all = all_densities.max()
    else:
        # 如果所有節點都無法計算 KDE，就預設一個範圍
        vmin_all, vmax_all = 0.0, 1.0

    # 5. 建立整張合併圖的 Figure 與一個 Axes，並設定底色為白
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # 6. 依序處理每個節點
    last_scatter = None  # 用來之後繪製 colorbar
    for idx, name in enumerate(vertice):
        # 6.1 取出該節點的所有 B 個 bootstrap 樣本 (shape = (B, 2))
        node_samples = samples_flipped[:, idx, :]

        # 6.2 如果該節點的樣本數量 < 2 或所有坐標完全相同，就只畫平均點 + 標籤
        unique_pts = np.unique(node_samples, axis=0)
        if node_samples.shape[0] < 2 or unique_pts.shape[0] == 1:
            mean_xy = node_samples.mean(axis=0)
            cmap_obj = plt.get_cmap(seq_cmaps[idx % len(seq_cmaps)])
            marker_color = cmap_obj(0.9)
            ax.plot(mean_xy[0], mean_xy[1],
                    marker='x', color=marker_color,
                    ms=8, mew=1.5, zorder=3)
            ax.text(mean_xy[0] + 0.5, mean_xy[1] + 0.5, name,
                    color=marker_color, fontsize=8,
                    fontweight='bold', zorder=4)
            continue

        # 6.3 擬合 Gaussian KDE
        try:
            kde = gaussian_kde(node_samples.T)
        except Exception as e:
            print(f"節點 '{name}' 的 KDE 擬合失敗: {e}. 僅繪製原始點與平均點。")
            ax.scatter(node_samples[:, 0], node_samples[:, 1],
                       s=4, color='k', alpha=0.6, zorder=2)
            mean_xy = node_samples.mean(axis=0)
            ax.plot(mean_xy[0], mean_xy[1],
                    marker='x', color='r', ms=8, mew=1.5, zorder=3)
            ax.text(mean_xy[0] + 0.5, mean_xy[1] + 0.5, name,
                    color='r', fontsize=8, fontweight='bold', zorder=4)
            continue

        # === 6.4 先計算該節點所有 bootstrap 點的 density，並用 scatter 著色 ===
        densities = kde(node_samples.T)  # shape = (B,)

        cmap_obj = plt.get_cmap(seq_cmaps[idx % len(seq_cmaps)])
        marker_color = cmap_obj(0.9)

        sc = ax.scatter(
            node_samples[:, 0], node_samples[:, 1],
            c=densities,
            cmap=cmap_obj,     # 該城市專屬色系
            norm=LogNorm(vmin=vmin_all, vmax=vmax_all),
            s=2,               # 點半徑調小
            alpha=0.9,         # 半透明
            zorder=2
        )
        last_scatter = sc  # 保留最後一個 scatter，供 colorbar 用

        # 6.5 繪製該節點的“5 段等高線”，線條細、顏色淡，與該城市同色
        # —— 改回「原本自动 levels=5 的写法」 ——  
        zz = kde(grid_coords).reshape(xx.shape)
        zz_masked = np.ma.masked_where(zz <= 0, zz) 
        line_color = cmap_obj(0.6)
        ax.contour(xx, yy, zz_masked, levels=5, colors=[line_color], linewidths=0.5, linestyles='-', alpha=0.6, zorder=1)


        # 6.6 繪製該節點的平均位置 (marker='x') 與節點文字
        mean_xy = node_samples.mean(axis=0)
        #ax.plot(mean_xy[0], mean_xy[1],
        #        marker='x', color=marker_color,
        #        ms=6, mew=1.5, zorder=3)
        ax.text(mean_xy[0] + 0.5, mean_xy[1] + 0.5,
                name, color=marker_color,
                fontsize=4, fontweight='bold', zorder=4)

    # 7. 設定整張圖的 x, y 範圍與比例
    ax.set_xlim(0, 1200)
    ax.set_ylim(0, 750)
    ax.set_aspect('equal')
    ax.margins(x=0.02, y=0.02)  # 在四周留一點空白

    # === 新增：繪製『統一顏色比例尺』（Colorbar） ===
    if last_scatter is not None:
        cb = fig.colorbar(last_scatter, ax=ax, fraction=0.046, pad=0.04, shrink=0.65)
        cb.set_label('KDE Density (Global)')

    # 8. 圖片標題與排版
    ax.set_title('Combined KDE Density Map', pad=15)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # 9. 存檔並關閉
    plt.savefig(KDE_COMBINED_FILE, dpi=300)
    plt.close()

    # 10. 印出存檔成功訊息
    print(f"==> Combined KDE saved as {ELLIPSES_FILE}")