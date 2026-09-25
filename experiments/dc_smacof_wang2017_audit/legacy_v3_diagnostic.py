"""
test_dc_smacof_v3.py — DC-SMACOF 進階驗證

T5: 複雜合成資料 (10節點, 含噪音/矛盾方向)
T6: 真實西域資料參數掃描 (v_weight sweep)
T7: 真實西域資料最佳參數下的詳細方向分析

用法：
  cd physics_simulation
  python -m tests.test_dc_smacof_v3
"""

import numpy as np
import math
from copy import deepcopy
from numpy import linalg
from scipy.sparse.linalg import cg

from library.directions import DIR8_UNIT_SIM as unit_direction_dict

# ============================================================
# 工具函數 (和 v2 相同)
# ============================================================

def compute_direction_accuracy(X, sel_data, dni):
    dir4 = {'東', '西', '南', '北'}
    total, correct, violations = 0, 0, []
    for row in sel_data:
        a_name, b_name, d_name = row[0], row[1], row[2].strip()
        if a_name not in dni or b_name not in dni or d_name not in unit_direction_dict:
            continue
        ia, ib = dni[a_name], dni[b_name]
        r = X[ib] - X[ia]
        dist = np.linalg.norm(r)
        if dist < 1e-9:
            continue
        r_hat = r / dist
        v_dir = np.array(unit_direction_dict[d_name], dtype=float)
        dot_val = float(np.dot(r_hat, v_dir))
        cross_val = float(r_hat[0] * v_dir[1] - r_hat[1] * v_dir[0])
        phi = math.atan2(cross_val, dot_val)
        theta_h = math.pi / 2 if d_name in dir4 else math.pi / 4
        total += 1
        if abs(phi) <= theta_h:
            correct += 1
        else:
            violations.append((a_name, b_name, d_name, math.degrees(phi)))
    return (correct / total if total > 0 else 0.0), violations


def compute_distance_stress(X, edges_data, dni):
    """Kruskal's stress"""
    num, den = 0, 0
    for row in edges_data:
        i, j = dni[row[0]], dni[row[1]]
        d_actual = np.linalg.norm(X[i] - X[j])
        d_target = float(row[2])
        num += (d_actual - d_target) ** 2
        den += d_target ** 2
    return math.sqrt(num / den) if den > 0 else float('nan')


def build_and_run(n, vertice, edges, graph, sel_data, dni, dis, data,
                  w_w, v_w, n_iters=500, seed=42):
    """建構矩陣 + 跑修正版迭代，回傳最終座標"""
    t = len(sel_data)
    s_edges = n * (n - 1) // 2

    # weight, veight
    weight = np.zeros((n, n))
    for ver_list in graph:
        for row in ver_list:
            i, j = dni[row[0]], dni[row[1]]
            if dis[i][j] != 0:
                weight[i][j] = w_w / (dis[i][j] ** 2)
                weight[j][i] = w_w / (dis[i][j] ** 2)

    veight = np.zeros((n, n))
    for row in sel_data:
        i, j = dni[row[0]], dni[row[1]]
        if dis[i][j] != 0:
            veight[i][j] = v_w / (dis[i][j] ** 2)
            veight[j][i] = v_w / (dis[i][j] ** 2)
        else:
            avg = np.mean(dis[dis > 0]) if np.any(dis > 0) else 1.0
            veight[i][j] = v_w / (avg ** 2)
            veight[j][i] = v_w / (avg ** 2)

    # LW, LV
    LW = np.zeros((n, n))
    for i in range(n):
        s = sum(weight[i][j] for j in range(n) if j != i)
        LW[i][i] = s
        for j in range(n):
            if i != j: LW[i][j] = -weight[i][j]

    LV = np.zeros((n, n))
    for i in range(n):
        s = sum(veight[i][j] for j in range(n) if j != i)
        LV[i][i] = s
        for j in range(n):
            if i != j: LV[i][j] = -veight[i][j]

    # JW, JV
    JW = np.zeros((n, s_edges))
    cnt = 0
    for i in range(n):
        for j in range(i):
            if (vertice[i], vertice[j]) in edges or (vertice[j], vertice[i]) in edges:
                JW[j][cnt] = weight[i][j]
                JW[i][cnt] = -weight[i][j]
            cnt += 1

    JV = np.zeros((n, t))
    for k in range(t):
        x = dni[sel_data[k][0]]
        y = dni[sel_data[k][1]]
        JV[x][k] = veight[x][y]
        JV[y][k] = -veight[x][y]

    # 迭代 (修正版)
    np.random.seed(seed)
    X = np.random.rand(n, 2) * 500

    for iteration in range(n_iters):
        DW = np.zeros((s_edges, 2))
        cnt = 0
        for i in range(n):
            for j in range(i):
                v = X[j] - X[i]  # 修正：source(j) - target(i)
                norm = linalg.norm(v)
                unit = v / norm if norm > 1e-12 else np.zeros(2)
                DW[cnt] = dis[i][j] * unit
                cnt += 1

        DV = np.zeros((t, 2))
        for k in range(t):
            x_idx = dni[sel_data[k][0]]
            y_idx = dni[sel_data[k][1]]
            v = X[y_idx] - X[x_idx]
            current_dist = linalg.norm(v)
            dir_unit = np.array(unit_direction_dict[sel_data[k][2]])
            if current_dist > 1e-9:
                DV[k] = -(current_dist * dir_unit)  # 修正：取負號
            else:
                DV[k] = np.zeros(2)

        left = LW + LV
        right = JW @ DW + JV @ DV
        X_new = np.zeros_like(right)
        for col in range(right.shape[1]):
            x_sol, _ = cg(left, right[:, col], x0=X[:, col])
            X_new[:, col] = x_sol
        X = X_new

    return X


# ============================================================
# T5: 複雜合成資料 (10 節點，含矛盾方向)
# ============================================================
def test_T5():
    print("=" * 60)
    print("T5: 複雜合成資料 (10 節點，部分方向有噪音)")
    print("=" * 60)

    # 10 個節點排成不規則形狀
    gt = np.array([
        [0, 0], [100, 0], [200, 50], [300, 0], [400, 0],
        [0, 150], [100, 200], [200, 150], [300, 200], [400, 150],
    ], dtype=float)
    n = 10
    vertice = [f'V{i}' for i in range(n)]
    dni = {f'V{i}': i for i in range(n)}

    # 距離邊：相鄰節點
    edge_pairs = [(0,1),(1,2),(2,3),(3,4),(5,6),(6,7),(7,8),(8,9),
                  (0,5),(1,6),(2,7),(3,8),(4,9),(0,6),(2,8)]
    edges = [(vertice[i], vertice[j]) for i, j in edge_pairs]
    data = []
    graph = [[] for _ in range(n)]
    for i, j in edge_pairs:
        d = int(np.linalg.norm(gt[i] - gt[j]))
        data.append([vertice[i], vertice[j], str(d)])
        graph[i].append([vertice[i], vertice[j], '', str(d)])
        graph[j].append([vertice[j], vertice[i], '', str(d)])

    dis = np.zeros((n, n))
    for i, j in edge_pairs:
        d = int(np.linalg.norm(gt[i] - gt[j]))
        dis[i][j] = d
        dis[j][i] = d

    # 方向資料：根據真實幾何，加入一些故意錯誤
    def get_dir(i, j):
        dx = gt[j][0] - gt[i][0]
        dy = gt[j][1] - gt[i][1]
        angle = math.degrees(math.atan2(dy, dx)) % 360
        dir_map = {0: '東', 45: '東北', 90: '北', 135: '西北',
                   180: '西', 225: '西南', 270: '南', 315: '東南'}
        closest = min(dir_map.keys(), key=lambda a: min(abs(a-angle), 360-abs(a-angle)))
        return dir_map[closest]

    # 正確方向
    dir_pairs_correct = [(0,1), (1,2), (0,5), (1,6), (2,7), (3,4), (5,6), (7,8)]
    # 故意錯誤方向 (矛盾)
    dir_pairs_wrong = [(4,9)]  # 實際是北，但標成南

    sel_data = []
    for i, j in dir_pairs_correct:
        d = get_dir(i, j)
        sel_data.append([vertice[i], vertice[j], d, '', '', d])

    for i, j in dir_pairs_wrong:
        sel_data.append([vertice[i], vertice[j], '南', '', '', '南'])  # 故意錯

    total_dir = len(sel_data)
    correct_dir = len(dir_pairs_correct)
    print(f"  {n} 個節點, {len(edges)} 條距離邊, {total_dir} 條方向邊 (其中 {total_dir - correct_dir} 條故意錯)")

    # 測試不同 v/w
    ratios = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
    print(f"\n  {'v/w':>8} | {'Dir Acc':>8} | {'Stress':>10} | 違規數")
    print(f"  {'-'*8} | {'-'*8} | {'-'*10} | {'-'*6}")

    for ratio in ratios:
        X = build_and_run(n, vertice, edges, graph, sel_data, dni, dis, data,
                          w_w=1.0, v_w=ratio, n_iters=500)
        dir_acc, violations = compute_direction_accuracy(X, sel_data, dni)
        dist_stress = compute_distance_stress(X, data, dni)
        print(f"  {ratio:>8.3f} | {dir_acc*100:>7.1f}% | {dist_stress:>10.6f} | {len(violations)}")


# ============================================================
# T6: 真實西域資料參數掃描
# ============================================================
def test_T6():
    print("\n" + "=" * 60)
    print("T6: 真實西域資料 — v_weight 參數掃描")
    print("=" * 60)

    try:
        from library.config import FILE_PATHS
        from library.data_io import load_ini_data_from_csv, uploading_directional_data

        graph, vertice, dni, edges, data = load_ini_data_from_csv(FILE_PATHS)
        directional_data = uploading_directional_data()
    except Exception as e:
        print(f"  無法載入真實資料: {e}")
        print("  跳過 T6")
        return None, None, None, None, None, None

    n = len(vertice)
    s = len(edges)

    # 建構 dis 矩陣
    dis = np.zeros((n, n))
    for ver_list in graph:
        for row in ver_list:
            dis[dni[row[0]]][dni[row[1]]] = int(row[3])
            dis[dni[row[1]]][dni[row[0]]] = int(row[3])

    # 將 directional_data 轉為 sel_data 格式
    sel_data = []
    for row in directional_data:
        if len(row) >= 3 and row[0] in dni and row[1] in dni:
            d_name = row[2].strip()
            if d_name in unit_direction_dict:
                sel_data.append([row[0], row[1], d_name, '', '', d_name])

    print(f"  {n} 個節點, {s} 條距離邊, {len(sel_data)} 條方向邊")

    # 建構 data list (for stress calculation)
    data_for_stress = []
    for ver_list in graph:
        for row in ver_list:
            pair = (row[0], row[1])
            reverse = (row[1], row[0])
            already = any(d[0] == pair[0] and d[1] == pair[1] for d in data_for_stress)
            already2 = any(d[0] == reverse[0] and d[1] == reverse[1] for d in data_for_stress)
            if not already and not already2:
                data_for_stress.append([row[0], row[1], row[3]])

    # 參數掃描
    v_weights = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
    print(f"\n  {'v_weight':>10} | {'Dir Acc':>8} | {'Stress':>10} | 違規數/{len(sel_data)}")
    print(f"  {'-'*10} | {'-'*8} | {'-'*10} | {'-'*10}")

    best_acc = 0
    best_vw = 0
    best_X = None

    for v_w in v_weights:
        # 跑 3 個 seed 取平均
        accs, stresses = [], []
        for seed in [0, 42, 123]:
            X = build_and_run(n, vertice, edges, graph, sel_data, dni, dis, data_for_stress,
                              w_w=1.0, v_w=v_w, n_iters=300, seed=seed)
            dir_acc, violations = compute_direction_accuracy(X, sel_data, dni)
            dist_stress = compute_distance_stress(X, data_for_stress, dni)
            accs.append(dir_acc)
            stresses.append(dist_stress)

        mean_acc = np.mean(accs)
        mean_stress = np.mean(stresses)
        n_viol = int((1 - mean_acc) * len(sel_data))
        print(f"  {v_w:>10.4f} | {mean_acc*100:>7.1f}% | {mean_stress:>10.6f} | {n_viol}")

        if mean_acc > best_acc:
            best_acc = mean_acc
            best_vw = v_w

    print(f"\n  最佳 v_weight = {best_vw} (方向正確率 {best_acc*100:.1f}%)")
    return vertice, dni, sel_data, dis, graph, data_for_stress


# ============================================================
# T7: 真實資料詳細方向分析
# ============================================================
def test_T7(vertice, dni, sel_data, dis, graph, data_for_stress):
    print("\n" + "=" * 60)
    print("T7: 真實西域資料 — 最佳參數下的詳細方向分析")
    print("=" * 60)

    if vertice is None:
        print("  T6 未成功，跳過 T7")
        return

    n = len(vertice)
    edges_set = set()
    for ver_list in graph:
        for row in ver_list:
            edges_set.add((row[0], row[1]))
            edges_set.add((row[1], row[0]))
    edges = list(set((min(a,b), max(a,b)) for a,b in edges_set))

    # 用幾個候選 v_weight 跑
    for v_w in [0.01, 0.1, 1.0]:
        print(f"\n  --- v_weight = {v_w} ---")
        X = build_and_run(n, vertice, edges, graph, sel_data, dni, dis, data_for_stress,
                          w_w=1.0, v_w=v_w, n_iters=500, seed=42)
        dir_acc, violations = compute_direction_accuracy(X, sel_data, dni)
        dist_stress = compute_distance_stress(X, data_for_stress, dni)

        print(f"  方向正確率: {dir_acc*100:.1f}% ({len(sel_data)-len(violations)}/{len(sel_data)})")
        print(f"  距離 Stress: {dist_stress:.6f}")

        if violations:
            print(f"  違規邊 ({len(violations)} 條):")
            # 按偏差角度排序，顯示最嚴重的
            violations.sort(key=lambda x: abs(x[3]), reverse=True)
            for u, v, d, phi in violations[:10]:
                print(f"    {u}→{v}: 應為{d}, 偏差 {phi:.1f}°")
            if len(violations) > 10:
                print(f"    ... (省略 {len(violations)-10} 條)")


# ============================================================
if __name__ == "__main__":
    print("DC-SMACOF 進階驗證 v3")
    print("=" * 60)

    test_T5()
    result = test_T6()
    if result[0] is not None:
        test_T7(*result)

    print("\n" + "=" * 60)
    print("所有測試完成")