# spring_confidence.py
# ------------------------------------------------------------
# Author : <your name> (2025/05/11)
# 功能   : 對 spring.py 的力導向模型做 Bootstrap‑Dynamics，
#          計算每節點 95% 信賴區間橢球並繪圖。
# 依賴   : numpy, matplotlib, tqdm (可選), spring.py
# ------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from tqdm import trange
from copy import deepcopy

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
# ==== 引入你現有的 spring.py 主要接口 ============
import spring   # 請確定 spring.py 與本檔在同資料夾

# ==== 參數可自行調整 ============================
N_BOOTSTRAP      = 50          # 重覆次數（>300 建議跑一夜）
SPRING_JITTER    = 0.05         # 每次彈簧係數隨機 ±5 %
REPULSE_JITTER   = 0.20         # 排斥力常數 ±20 %
PLOT_FILE        = "confidence_ellipses.png"

# ---------- 輔助：一次模擬 -----------------------
def _run_once(seed:int,
              k_spring_scale:float=1.0,
              k_repulse_scale:float=1.0):
    """
    跑一次完整的 spring 模擬並回傳最終 Nx2 位置
    """
    np.random.seed(seed)

    # 1. 初始化（重用 generate_CHEN_initial_positions）
    vertice, dni, data, pos_matrix, fixed_pos = \
        spring.generate_CHEN_initial_positions([600, 500])

    # 2. 動態調整彈簧 & 排斥係數（透過 spring.py 中全域變數）
    spring.SPRING_STIFFNESS_BASE = spring.SPRING_STIFFNESS_BASE * k_spring_scale
    spring.REPULSION_STRENGTH_BASE = spring.REPULSION_STRENGTH_BASE * k_repulse_scale

    # 3. 跑核心物理模擬
    dir_data = spring.uploading_directional_data()
    _wrong, sh, final_pos = spring.main_physics_simulation(
        vertice, dni, data,
        deepcopy(pos_matrix),               # 傳入初始座標副本
        dir_data, fixed_pos
    )
    return np.asarray(final_pos), vertice

# ---------- Bootstrap + 信賴區計算 ---------------
def _bootstrap_dynamics():
    """
    回傳
    mu      : (N,2) 平均座標
    covs    : (N,2,2) 協方差矩陣
    vertice : 名稱 list
    """
    print(f"==> Bootstrap {N_BOOTSTRAP} runs, jitter spring±{SPRING_JITTER*100:.1f} % "
          f"repulse±{REPULSE_JITTER*100:.1f} %")

    # 第一次取樣以確定 N
    first_pos, vertice = _run_once(
        seed=0,
        k_spring_scale=1.0,
        k_repulse_scale=1.0
    )
    N_nodes = first_pos.shape[0]
    samples = np.zeros((N_BOOTSTRAP, N_nodes, 2))
    samples[0] = first_pos

    # 其餘 bootstrap
    for b in trange(1, N_BOOTSTRAP, desc="Bootstrap"):
        ks = 1.0 + np.random.uniform(-SPRING_JITTER, SPRING_JITTER)
        kr = 1.0 + np.random.uniform(-REPULSE_JITTER, REPULSE_JITTER)
        pos, _ = _run_once(seed=b, k_spring_scale=ks, k_repulse_scale=kr)
        samples[b] = pos
    mu = samples.mean(axis=0)                   # (N,2)
    covs = np.zeros((N_nodes, 2, 2))
    for i in range(N_nodes):
        covs[i] = np.cov(samples[:, i, 0], samples[:, i, 1], rowvar=False)

    return mu, covs, vertice

# ---------- 繪製橢球 ------------------------------
def _plot_ellipses(mu, covs, vertice):
    """
    將平均位置 + 95% 信賴橢球存成 PNG
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor("white")

    # 繪製平均點
    ax.scatter(mu[:, 0], mu[:, 1], c='k', s=8, zorder=5)

    chi2_95 = 5.991  # df=2
    for (mx, my), Sigma in zip(mu, covs):
        # Eigen decomposition
        vals, vecs = np.linalg.eigh(Sigma)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * np.sqrt(chi2_95 * vals)  # 直徑
        ellipse = Ellipse((mx, my), width, height,
                          angle=angle, edgecolor='red',
                          linewidth=0.7, facecolor='none', alpha=0.5, zorder=4)
        ax.add_patch(ellipse)

    # 標註名稱
    for (mx, my), name in zip(mu, vertice):
        ax.text(mx+3, my-3, name, fontsize=7)

    ax.set_xlim(0, 1200); ax.set_ylim(750, 0)
    ax.set_title("95% confidence ellipses (Bootstrap Dynamics)")
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=300)
    print(f"==> Figure saved as {PLOT_FILE}")
    plt.close()

# ---------- 對外介面 ------------------------------
def bootstrap_and_plot():
    mu, covs, vertice = _bootstrap_dynamics()
    _plot_ellipses(mu, covs, vertice)
    # 回傳平均座標供後續使用
    return mu, covs, vertice
