import numpy as np
import pandas as pd
import random
import argparse
import os
from scipy.spatial.distance import cdist

def generate_test_data_extreme_clusters(num_cells=100000, A_ratio=0.3, seed=2025, 
                                       num_B_clusters=8, cluster_std=15.0, 
                                       space_size=500, A_distribution='uniform'):
    """
    生成极端团块状分布测试数据，专门用于测试空间索引算法性能
    
    参数:
      num_cells: 总细胞数
      A_ratio: A 细胞比例（0~1）  
      seed: 随机种子
      num_B_clusters: B 细胞团块数量
      cluster_std: B 细胞团块标准差（控制团块紧密程度）
      space_size: 空间大小（正方形边长）
      A_distribution: A 细胞分布模式 ('uniform', 'sparse', 'interstitial')
      
    返回:
      DataFrame，列 ['cellid','x','y','celltype']
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

    # 1. 生成 B 细胞团块
    # 团块中心位置：在空间中央区域随机分布，避免边界效应
    margin = 0.15 * space_size
    cluster_centers = np.random.uniform(margin, space_size - margin, size=(num_B_clusters, 2))
    
    # 每个团块的 B 细胞数量（不均匀分配，制造更极端的情况）
    cluster_weights = np.random.exponential(scale=1.0, size=num_B_clusters)
    cluster_weights = cluster_weights / cluster_weights.sum()
    cluster_sizes = (cluster_weights * n_B).astype(int)
    
    # 确保总数正确
    cluster_sizes[-1] += n_B - cluster_sizes.sum()
    
    B_x_list = []
    B_y_list = []
    
    for i, (cx, cy) in enumerate(cluster_centers):
        size = cluster_sizes[i]
        if size <= 0:
            continue
            
        # 使用不同大小的团块制造更极端的分布
        current_std = cluster_std * np.random.uniform(0.5, 2.0)
        
        xs = np.random.normal(loc=cx, scale=current_std, size=size)
        ys = np.random.normal(loc=cy, scale=current_std, size=size)
        
        # 裁剪到空间范围内
        xs = np.clip(xs, 0, space_size)
        ys = np.clip(ys, 0, space_size)
        
        B_x_list.extend(xs)
        B_y_list.extend(ys)
    
    B_x = np.array(B_x_list)
    B_y = np.array(B_y_list)

    # 2. 生成 A 细胞
    if A_distribution == 'uniform':
        # 均匀分布在整个空间
        A_x = np.random.uniform(0, space_size, size=n_A)
        A_y = np.random.uniform(0, space_size, size=n_A)
        
    elif A_distribution == 'sparse':
        # 稀疏分布，主要在团块之间的空隙中
        A_x, A_y = [], []
        attempts = 0
        max_attempts = n_A * 50
        
        while len(A_x) < n_A and attempts < max_attempts:
            x = np.random.uniform(0, space_size)
            y = np.random.uniform(0, space_size)
            
            # 检查是否距离所有团块中心足够远
            distances_to_clusters = np.sqrt((x - cluster_centers[:,0])**2 + 
                                          (y - cluster_centers[:,1])**2)
            min_distance = np.min(distances_to_clusters)
            
            if min_distance > cluster_std * 2.5:  # 距离团块较远
                A_x.append(x)
                A_y.append(y)
            
            attempts += 1
        
        # 如果没有生成足够的 A 细胞，用均匀分布补足
        while len(A_x) < n_A:
            A_x.append(np.random.uniform(0, space_size))
            A_y.append(np.random.uniform(0, space_size))
            
        A_x = np.array(A_x)
        A_y = np.array(A_y)
        
    elif A_distribution == 'interstitial':
        # 主要分布在团块边缘和团块之间
        A_x, A_y = [], []
        attempts = 0
        max_attempts = n_A * 30
        
        while len(A_x) < n_A and attempts < max_attempts:
            x = np.random.uniform(0, space_size)
            y = np.random.uniform(0, space_size)
            
            distances_to_clusters = np.sqrt((x - cluster_centers[:,0])**2 + 
                                          (y - cluster_centers[:,1])**2)
            min_distance = np.min(distances_to_clusters)
            
            # 在团块边缘区域（距离适中）
            if cluster_std * 1.5 < min_distance < cluster_std * 4.0:
                A_x.append(x)
                A_y.append(y)
            
            attempts += 1
        
        # 补足不够的数量
        while len(A_x) < n_A:
            A_x.append(np.random.uniform(0, space_size))
            A_y.append(np.random.uniform(0, space_size))
            
        A_x = np.array(A_x)
        A_y = np.array(A_y)

    # 3. 合并数据
    xs_all = np.concatenate([A_x, B_x])
    ys_all = np.concatenate([A_y, B_y])
    types_all = ['A'] * len(A_x) + ['B'] * len(B_x)

    df = pd.DataFrame({
        'x': xs_all,
        'y': ys_all,
        'celltype': types_all
    })
    
    # 打乱顺序并重新编号
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df['cellid'] = df.index
    df = df[['cellid', 'x', 'y', 'celltype']]
    
    return df, cluster_centers

def compute_nearest_B(df):
    """计算每个 A 细胞最近的 B 细胞 ID 和距离"""
    A_cells = df[df['celltype'] == 'A'].copy().reset_index(drop=True)
    B_cells = df[df['celltype'] == 'B'].copy().reset_index(drop=True)
    
    if A_cells.empty:
        return pd.DataFrame(columns=['cellid','nearest_B_id','nearest_B_dist'])
    if B_cells.empty:
        return pd.DataFrame({
            'cellid': A_cells['cellid'].values,
            'nearest_B_id': [-1]*len(A_cells),
            'nearest_B_dist': [np.nan]*len(A_cells)
        })
    
    A_coords = A_cells[['x','y']].values
    B_coords = B_cells[['x','y']].values
    
    # 对于大数据量，分批计算避免内存溢出
    batch_size = 1000
    results = []
    
    for i in range(0, len(A_cells), batch_size):
        end_idx = min(i + batch_size, len(A_cells))
        batch_A = A_coords[i:end_idx]
        
        distances = cdist(batch_A, B_coords, metric='euclidean')
        nearest_idx = np.argmin(distances, axis=1)
        nearest_dists = distances[np.arange(len(batch_A)), nearest_idx]
        nearest_ids = B_cells.iloc[nearest_idx]['cellid'].values
        
        batch_result = pd.DataFrame({
            'cellid': A_cells.iloc[i:end_idx]['cellid'].values,
            'nearest_B_id': nearest_ids,
            'nearest_B_dist': nearest_dists
        })
        results.append(batch_result)
    
    return pd.concat(results, ignore_index=True)

def compute_B_within_radius(df, radius=10.0):
    """计算每个 A 细胞在半径内的 B 细胞数量"""
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
    
    # 分批计算
    batch_size = 1000
    results = []
    
    for i in range(0, len(A_cells), batch_size):
        end_idx = min(i + batch_size, len(A_cells))
        batch_A = A_coords[i:end_idx]
        
        distances = cdist(batch_A, B_coords, metric='euclidean')
        counts = (distances <= radius).sum(axis=1)
        
        batch_result = pd.DataFrame({
            'cellid': A_cells.iloc[i:end_idx]['cellid'].values,
            'B_count_within_R': counts,
            'radius': [radius]*len(counts)
        })
        results.append(batch_result)
    
    return pd.concat(results, ignore_index=True)

def print_data_stats(df, cluster_centers):
    """打印数据统计信息"""
    print("=== 极端团块数据统计 ===")
    print(f"总细胞数: {len(df)}")
    nA = (df['celltype'] == 'A').sum()
    nB = (df['celltype'] == 'B').sum()
    print(f"A 细胞数: {nA}, B 细胞数: {nB}")
    print(f"B 细胞团块数: {len(cluster_centers)}")
    
    # 计算空间范围
    print(f"X 范围: [{df['x'].min():.1f}, {df['x'].max():.1f}]")
    print(f"Y 范围: [{df['y'].min():.1f}, {df['y'].max():.1f}]")
    
    # 计算 B 细胞密度分布
    B_cells = df[df['celltype'] == 'B']
    if len(B_cells) > 0:
        B_coords = B_cells[['x','y']].values
        
        # 计算每个团块中心附近的 B 细胞数量
        for i, (cx, cy) in enumerate(cluster_centers):
            distances = np.sqrt((B_coords[:,0] - cx)**2 + (B_coords[:,1] - cy)**2)
            cells_in_cluster = np.sum(distances <= 30)  # 30 像素半径内
            print(f"团块 {i+1} 中心 ({cx:.1f}, {cy:.1f}) 附近 B 细胞数: {cells_in_cluster}")

def main():
    parser = argparse.ArgumentParser(description="生成极端团块状分布的测试数据，用于测试空间索引算法性能")
    parser.add_argument('--num_cells', type=int, default=100000, help='总细胞数量')
    parser.add_argument('--A_ratio', type=float, default=0.3, help='A 细胞比例')
    parser.add_argument('--seed', type=int, default=2025, help='随机种子')
    parser.add_argument('--num_B_clusters', type=int, default=8, help='B 细胞团块数量')
    parser.add_argument('--cluster_std', type=float, default=15.0, help='B 细胞团块标准差')
    parser.add_argument('--space_size', type=float, default=500.0, help='空间大小')
    parser.add_argument('--A_distribution', type=str, default='interstitial', 
                       choices=['uniform', 'sparse', 'interstitial'], help='A 细胞分布模式')
    parser.add_argument('--radius', type=float, default=10.0, help='范围统计半径')
    parser.add_argument('--output_prefix', type=str, default='test_cells_extreme', help='输出文件前缀')
    
    args = parser.parse_args()

    print(f"生成极端团块数据: {args.num_cells} 个细胞，{args.num_B_clusters} 个B团块")
    print(f"参数: A_ratio={args.A_ratio}, cluster_std={args.cluster_std}, A_distribution={args.A_distribution}")
    
    df, cluster_centers = generate_test_data_extreme_clusters(
        num_cells=args.num_cells,
        A_ratio=args.A_ratio,
        seed=args.seed,
        num_B_clusters=args.num_B_clusters,
        cluster_std=args.cluster_std,
        space_size=args.space_size,
        A_distribution=args.A_distribution
    )
    
    # 保存全部数据 - 兼容 C++ 测试框架
    full_fname = "test_cells.csv"  # 固定文件名，方便C++程序读取
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
    print_data_stats(df, cluster_centers)
    print("\n示例 A 细胞答案：")
    print(result_df.head())

if __name__ == "__main__":
    main()