import numpy as np
import pandas as pd
import random
import argparse
import math
from scipy.spatial.distance import cdist

def generate_linear_data(num_cells=100000, A_ratio=0.3, seed=2025, 
                        distribution_type='single_line', space_size=500):
    """
    生成线性/螺旋状分布数据，测试KD-Tree优势
    
    参数:
      num_cells: 总细胞数
      A_ratio: A 细胞比例
      seed: 随机种子
      distribution_type: 分布类型 
        - 'single_line': 单条直线
        - 'multi_lines': 多条平行线
        - 'cross_lines': 交叉线
        - 'spiral': 螺旋状
        - 'sine_wave': 正弦波
      space_size: 空间大小
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

    # 生成 B 细胞的特殊分布
    B_x, B_y = [], []
    
    if distribution_type == 'single_line':
        # 单条对角线
        t = np.linspace(0, 1, n_B)
        B_x = t * space_size * 0.8 + space_size * 0.1
        B_y = t * space_size * 0.8 + space_size * 0.1
        # 添加少量噪音
        noise_std = space_size * 0.02
        B_x += np.random.normal(0, noise_std, n_B)
        B_y += np.random.normal(0, noise_std, n_B)
        
    elif distribution_type == 'multi_lines':
        # 三条平行线
        n_per_line = n_B // 3
        for i in range(3):
            n_this_line = n_per_line + (1 if i < n_B % 3 else 0)
            t = np.linspace(0, 1, n_this_line)
            
            # 不同的平行线
            offset = (i - 1) * space_size * 0.2
            line_x = t * space_size * 0.8 + space_size * 0.1
            line_y = t * space_size * 0.8 + space_size * 0.1 + offset
            
            # 添加噪音
            noise_std = space_size * 0.015
            line_x += np.random.normal(0, noise_std, n_this_line)
            line_y += np.random.normal(0, noise_std, n_this_line)
            
            B_x.extend(line_x)
            B_y.extend(line_y)
            
    elif distribution_type == 'cross_lines':
        # 两条交叉线
        n_per_line = n_B // 2
        
        # 第一条线：对角线
        t1 = np.linspace(0, 1, n_per_line)
        line1_x = t1 * space_size * 0.8 + space_size * 0.1
        line1_y = t1 * space_size * 0.8 + space_size * 0.1
        
        # 第二条线：反对角线
        n_line2 = n_B - n_per_line
        t2 = np.linspace(0, 1, n_line2)
        line2_x = t2 * space_size * 0.8 + space_size * 0.1
        line2_y = (1 - t2) * space_size * 0.8 + space_size * 0.1
        
        B_x = np.concatenate([line1_x, line2_x])
        B_y = np.concatenate([line1_y, line2_y])
        
        # 添加噪音
        noise_std = space_size * 0.02
        B_x += np.random.normal(0, noise_std, len(B_x))
        B_y += np.random.normal(0, noise_std, len(B_y))
        
    elif distribution_type == 'spiral':
        # 阿基米德螺旋
        t = np.linspace(0, 4 * np.pi, n_B)  # 4圈螺旋
        r = t * space_size * 0.08  # 半径随角度增长
        
        # 极坐标转直角坐标
        B_x = r * np.cos(t) + space_size * 0.5
        B_y = r * np.sin(t) + space_size * 0.5
        
        # 添加噪音
        noise_std = space_size * 0.015
        B_x += np.random.normal(0, noise_std, n_B)
        B_y += np.random.normal(0, noise_std, n_B)
        
    elif distribution_type == 'sine_wave':
        # 正弦波形
        x = np.linspace(0, space_size * 0.8, n_B) + space_size * 0.1
        y = np.sin(x / space_size * 4 * np.pi) * space_size * 0.15 + space_size * 0.5
        
        B_x = x
        B_y = y
        
        # 添加噪音
        noise_std = space_size * 0.02
        B_x += np.random.normal(0, noise_std, n_B)
        B_y += np.random.normal(0, noise_std, n_B)
    
    # 确保坐标在范围内
    B_x = np.clip(B_x, 0, space_size)
    B_y = np.clip(B_y, 0, space_size)
    
    # 生成 A 细胞：均匀分布在整个空间
    A_x = np.random.uniform(0, space_size, size=n_A)
    A_y = np.random.uniform(0, space_size, size=n_A)

    # 合并数据
    xs_all = np.concatenate([A_x, B_x])
    ys_all = np.concatenate([A_y, B_y])
    types_all = ['A'] * n_A + ['B'] * n_B

    df = pd.DataFrame({
        'x': xs_all,
        'y': ys_all,
        'celltype': types_all
    })
    
    # 打乱顺序并重新编号
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df['cellid'] = df.index
    df = df[['cellid', 'x', 'y', 'celltype']]
    
    return df

def generate_extreme_sparse_data(num_cells=100000, A_ratio=0.3, seed=2025, space_size=500):
    """
    生成极端稀疏分布：B细胞集中在空间一角，A细胞均匀分布
    这种分布对Grid Search极不友好
    """
    np.random.seed(seed)
    random.seed(seed)

    n_A = int(num_cells * A_ratio)
    if n_A < 1:
        n_A = 1
    if n_A > num_cells - 1:
        n_A = num_cells - 1
    n_B = num_cells - n_A

    # A细胞：均匀分布在整个空间
    A_x = np.random.uniform(0, space_size, size=n_A)
    A_y = np.random.uniform(0, space_size, size=n_A)

    # B细胞：极度集中在左下角小区域
    corner_size = space_size * 0.15  # 只占15%的空间
    B_x = np.random.uniform(0, corner_size, size=n_B)
    B_y = np.random.uniform(0, corner_size, size=n_B)

    # 合并数据
    xs_all = np.concatenate([A_x, B_x])
    ys_all = np.concatenate([A_y, B_y])
    types_all = ['A'] * n_A + ['B'] * n_B

    df = pd.DataFrame({
        'x': xs_all,
        'y': ys_all,
        'celltype': types_all
    })
    
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df['cellid'] = df.index
    df = df[['cellid', 'x', 'y', 'celltype']]
    
    return df

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
    
    # 分批计算避免内存溢出
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

def print_data_stats(df, distribution_type):
    """打印数据统计信息"""
    print(f"=== {distribution_type} 数据统计 ===")
    print(f"总细胞数: {len(df)}")
    nA = (df['celltype'] == 'A').sum()
    nB = (df['celltype'] == 'B').sum()
    print(f"A 细胞数: {nA}, B 细胞数: {nB}")
    
    # 计算空间范围
    print(f"X 范围: [{df['x'].min():.1f}, {df['x'].max():.1f}]")
    print(f"Y 范围: [{df['y'].min():.1f}, {df['y'].max():.1f}]")
    
    # 分析B细胞分布特性
    B_cells = df[df['celltype'] == 'B']
    if len(B_cells) > 0:
        B_coords = B_cells[['x','y']].values
        
        # 计算B细胞的密度分布
        print(f"B细胞密度分析:")
        print(f"  X方向标准差: {np.std(B_coords[:,0]):.2f}")
        print(f"  Y方向标准差: {np.std(B_coords[:,1]):.2f}")
        
        # 计算B细胞的空间占用率
        x_span = B_coords[:,0].max() - B_coords[:,0].min()
        y_span = B_coords[:,1].max() - B_coords[:,1].min()
        total_span = df['x'].max() - df['x'].min()
        coverage = (x_span * y_span) / (total_span * total_span) * 100
        print(f"  B细胞空间覆盖率: {coverage:.1f}%")

def visualize_data(df, distribution_type, save_plot=False):
    """可选：生成数据可视化图"""
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 8))
        
        # 绘制A细胞
        A_cells = df[df['celltype'] == 'A']
        plt.scatter(A_cells['x'], A_cells['y'], c='red', s=1, alpha=0.6, label='A cells')
        
        # 绘制B细胞
        B_cells = df[df['celltype'] == 'B']
        plt.scatter(B_cells['x'], B_cells['y'], c='blue', s=2, alpha=0.8, label='B cells')
        
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.title(f'{distribution_type} Distribution')
        plt.legend()
        plt.axis('equal')
        
        if save_plot:
            plt.savefig(f'{distribution_type}_distribution.png', dpi=150, bbox_inches='tight')
            print(f"已保存可视化图到 {distribution_type}_distribution.png")
        else:
            plt.show()
        
        plt.close()
        
    except ImportError:
        print("matplotlib not available, skipping visualization")

def main():
    parser = argparse.ArgumentParser(description="生成线性/螺旋状分布数据，测试KD-Tree算法优势")
    parser.add_argument('--num_cells', type=int, default=100000, help='总细胞数量')
    parser.add_argument('--A_ratio', type=float, default=0.3, help='A 细胞比例')
    parser.add_argument('--seed', type=int, default=2025, help='随机种子')
    parser.add_argument('--distribution', type=str, default='spiral', 
                       choices=['single_line', 'multi_lines', 'cross_lines', 'spiral', 'sine_wave', 'extreme_sparse'],
                       help='B细胞分布类型')
    parser.add_argument('--space_size', type=float, default=500.0, help='空间大小')
    parser.add_argument('--radius', type=float, default=10.0, help='范围统计半径')
    parser.add_argument('--visualize', action='store_true', help='生成可视化图')
    
    args = parser.parse_args()

    print(f"生成 {args.distribution} 分布数据: {args.num_cells} 个细胞")
    print(f"参数: A_ratio={args.A_ratio}, space_size={args.space_size}")
    
    if args.distribution == 'extreme_sparse':
        df = generate_extreme_sparse_data(
            num_cells=args.num_cells,
            A_ratio=args.A_ratio,
            seed=args.seed,
            space_size=args.space_size
        )
    else:
        df = generate_linear_data(
            num_cells=args.num_cells,
            A_ratio=args.A_ratio,
            seed=args.seed,
            distribution_type=args.distribution,
            space_size=args.space_size
        )
    
    # 保存全部数据 - 兼容 C++ 测试框架
    full_fname = "test_cells.csv"
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
    
    ans_fname = f"test_cells_{args.distribution}_answers.csv"
    result_df.to_csv(ans_fname, index=False)
    print(f"已保存 A 细胞答案数据到 {ans_fname}")

    # 打印统计信息
    print_data_stats(df, args.distribution)
    
    # 可视化
    if args.visualize:
        visualize_data(df, args.distribution, save_plot=True)
    
    print("\n示例 A 细胞答案：")
    print(result_df.head())
    
    print(f"\n现在可以运行 C++ 程序测试 {args.distribution} 分布下的算法性能!")

if __name__ == "__main__":
    main()