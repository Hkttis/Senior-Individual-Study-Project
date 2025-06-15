
from library.bootstrap_and_visualization import *

def bootstrap_and_plot():
    # Parameters
    N_BOOTSTRAP          = 140      # 重覆次數（>300 建議跑一夜）
    SPRING_JITTER        = 0.05    # 每次彈簧係數隨機 ±5 %
    REPULSE_JITTER       = 0.20    # 排斥力常數 ±20 %
    
    samples, vertice = bootstrap_dynamics(N_BOOTSTRAP,SPRING_JITTER,REPULSE_JITTER)
    plot_multi_ellipses(samples, vertice)
    plot_kde_combined(samples, vertice)
    return vertice

if __name__ == '__main__':
    bootstrap_and_plot()
