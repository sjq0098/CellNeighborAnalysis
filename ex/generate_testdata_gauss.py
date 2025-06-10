import numpy as np
import pandas as pd
import random
import argparse
import os
from scipy.spatial.distance import cdist

def generate_test_data_clustered(num_cells=10000, A_ratio=0.3, seed=2025, cluster_std=0.1):
    """
    生成测试数据，使得 B 细胞在每个 A 细胞周围呈高斯簇分布。
    参数:
      num_cells: 总细胞数
      A_ratio: A 细胞比例（0~1）
      seed: 随机种子
      cluster_std: B 细胞簇的标准差（高斯分布标准差）
    返回:
      DataFrame，列 ['cellid','x','y','celltype']，cellid 顺序已打乱，介于 0..num_cells-1
    """
    np.random.seed(seed)
    random.seed(seed)

    # 计算 A/B 数量
    n_A = int(num_cells * A_ratio)
    if n_A < 1:
        n_A = 1
    if n_A > num_cells - 1:
        n_A = num_cells - 1
    n_B = num_cells - n_A

    # 1. 生成 A 细胞位置：均匀分布
    A_x = np.random.uniform(0, 100, size=n_A)
    A_y = np.random.uniform(0, 100, size=n_A)

    # 2. 为每个 A 生成周围 B 簇
    # 平均分配 B 数量到各簇
    base = n_B // n_A
    rem = n_B - base * n_A  # 前 rem 个 A 多一个
    B_x_list = []
    B_y_list = []
    for i in range(n_A):
        cnt = base + (1 if i < rem else 0)
        if cnt <= 0:
            continue
        # 以 A_x[i], A_y[i] 为中心，高斯分布
        xs = np.random.normal(loc=A_x[i], scale=cluster_std, size=cnt)
        ys = np.random.normal(loc=A_y[i], scale=cluster_std, size=cnt)
        # 裁剪到 [0,100]
        xs = np.clip(xs, 0, 100)
        ys = np.clip(ys, 0, 100)
        B_x_list.append(xs)
        B_y_list.append(ys)
    if B_x_list:
        B_x = np.concatenate(B_x_list)
        B_y = np.concatenate(B_y_list)
    else:
        B_x = np.array([])
        B_y = np.array([])

    # 检查数量
    if len(B_x) != n_B:
        # 理论上 len(B_x)==n_B，如因浮点或逻辑误差不等，可截断或补全
        if len(B_x) > n_B:
            B_x = B_x[:n_B]
            B_y = B_y[:n_B]
        else:
            # 若少了，可在剩余位置均匀生成
            deficit = n_B - len(B_x)
            ux = np.random.uniform(0, 100, size=deficit)
            uy = np.random.uniform(0, 100, size=deficit)
            B_x = np.concatenate([B_x, ux])
            B_y = np.concatenate([B_y, uy])

    # 3. 合并 A/B 数据
    # 先把 A 放前面，B 放后面，之后打乱顺序
    xs_all = np.concatenate([A_x, B_x])
    ys_all = np.concatenate([A_y, B_y])
    types_all = ['A'] * n_A + ['B'] * n_B

    df = pd.DataFrame({
        'x': xs_all,
        'y': ys_all,
        'celltype': types_all
    })
    # 打乱顺序并重新编号 cellid
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df['cellid'] = df.index  # 0..num_cells-1
    # 最终列顺序：cellid, x, y, celltype
    df = df[['cellid', 'x', 'y', 'celltype']]
    return df

def compute_nearest_B(df):
    """
    计算每个 A 细胞最近的 B 细胞 ID 和距离，返回 DataFrame ['cellid','nearest_B_id','nearest_B_dist']。
    注意：当 B 细胞非常多时，cdist 会消耗大量内存，可能需要其他方法。
    """
    A_cells = df[df['celltype'] == 'A'].copy().reset_index(drop=True)
    B_cells = df[df['celltype'] == 'B'].copy().reset_index(drop=True)
    if A_cells.empty:
        return pd.DataFrame(columns=['cellid','nearest_B_id','nearest_B_dist'])
    if B_cells.empty:
        # 所有 nearest_B_id 标记 -1
        return pd.DataFrame({
            'cellid': A_cells['cellid'].values,
            'nearest_B_id': [-1]*len(A_cells),
            'nearest_B_dist': [np.nan]*len(A_cells)
        })
    A_coords = A_cells[['x','y']].values
    B_coords = B_cells[['x','y']].values
    # cdist
    distances = cdist(A_coords, B_coords, metric='euclidean')
    nearest_idx = np.argmin(distances, axis=1)
    nearest_dists = distances[np.arange(len(A_cells)), nearest_idx]
    nearest_ids = B_cells.iloc[nearest_idx]['cellid'].values
    return pd.DataFrame({
        'cellid': A_cells['cellid'].values,
        'nearest_B_id': nearest_ids,
        'nearest_B_dist': nearest_dists
    })

def compute_B_within_radius(df, radius=10.0):
    """
    计算每个 A 细胞在 radius 半径内的 B 细胞数量，返回 ['cellid','B_count_within_R','radius']。
    """
    A_cells = df[df['celltype'] == 'A'].copy().reset_index(drop=True)
    B_cells = df[df['celltype'] == 'B'].copy().reset_index(drop=True)
    if A_cells.empty:
        return pd.DataFrame(columns=['cellid','B_count_within_R','radius'])
    if B_cells.empty:
        return pd.DataFrame({
            'cellid': A_cells['cellid'].values,
            'B_count_within_R': [0]*len(A_cells),
            'radius': [radius]*len(A_cells)
        })
    A_coords = A_cells[['x','y']].values
    B_coords = B_cells[['x','y']].values
    distances = cdist(A_coords, B_coords, metric='euclidean')
    counts = (distances <= radius).sum(axis=1)
    return pd.DataFrame({
        'cellid': A_cells['cellid'].values,
        'B_count_within_R': counts,
        'radius': [radius]*len(A_cells)
    })

def main():
    parser = argparse.ArgumentParser(description="生成簇状 B 细胞围绕 A 细胞的测试数据并计算最近邻、范围统计")
    parser.add_argument('--num_cells', type=int, default=100000, help='总细胞数量')
    parser.add_argument('--A_ratio', type=float, default=0.3, help='A 细胞比例')
    parser.add_argument('--seed', type=int, default=2025, help='随机种子')
    parser.add_argument('--cluster_std', type=float, default=0.1, help='B 簇分布标准差')
    parser.add_argument('--radius', type=float, default=10.0, help='范围统计半径')
    parser.add_argument('--output_prefix', type=str, default='test_cells_clustered', help='输出文件前缀')
    args = parser.parse_args()

    print(f"生成 {args.num_cells} 个细胞，A_ratio={args.A_ratio}, seed={args.seed}, cluster_std={args.cluster_std}")
    df = generate_test_data_clustered(num_cells=args.num_cells,
                                      A_ratio=args.A_ratio,
                                      seed=args.seed,
                                      cluster_std=args.cluster_std)
    # 保存全部数据
    full_fname = f"{args.output_prefix}.csv"
    df.to_csv(full_fname, index=False)
    print(f"已保存全部细胞数据到 {full_fname}")

    # 计算最近邻和范围统计
    print("计算最近邻 B 细胞 ...")
    nearest_df = compute_nearest_B(df)
    print("计算半径内 B 细胞数量 ...")
    radius_df = compute_B_within_radius(df, radius=args.radius)
    # 合并 A 细胞答案
    if not nearest_df.empty:
        merged = pd.merge(nearest_df, radius_df, on='cellid', how='outer')
        A_base = df[df['celltype']=='A'][['cellid','x','y','celltype']].copy()
        result_df = pd.merge(A_base, merged, on='cellid', how='left')
    else:
        result_df = pd.DataFrame(columns=['cellid','x','y','celltype','nearest_B_id','nearest_B_dist','B_count_within_R','radius'])
    ans_fname = f"{args.output_prefix}_A_answers.csv"
    result_df.to_csv(ans_fname, index=False)
    print(f"已保存 A 细胞答案数据到 {ans_fname}")

    # 打印统计信息
    print("=== 数据生成统计 ===")
    print(f"总细胞数: {len(df)}")
    nA = (df['celltype']=='A').sum()
    print(f"A 细胞数: {nA}, B 细胞数: {len(df)-nA}")
    print("示例 A 细胞答案：")
    print(result_df.head())

if __name__ == "__main__":
    main()
