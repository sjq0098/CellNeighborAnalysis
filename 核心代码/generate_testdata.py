import numpy as np
import pandas as pd
import random

def generate_test_data(num_cells=100000, A_ratio=0.3, seed=2025):
    np.random.seed(seed)
    random.seed(seed)

    # 随机生成坐标
    x = np.random.uniform(0, 100, size=num_cells)
    y = np.random.uniform(0, 100, size=num_cells)
    
    # 随机分配 celltype，设 A 类型比例为 A_ratio
    celltypes = ['A' if random.random() < A_ratio else 'B' for _ in range(num_cells)]

    # 构建 DataFrame
    df = pd.DataFrame({
        'cellid': list(range(num_cells)),
        'x': x,
        'y': y,
        'celltype': celltypes
    })
    return df

# 示例生成
df = generate_test_data()
print("=== 原始数据示例 ===")
print(df.head())


from scipy.spatial.distance import cdist

def compute_nearest_B(df):
    A_cells = df[df['celltype'] == 'A'].copy()
    B_cells = df[df['celltype'] == 'B'].copy()

    A_coords = A_cells[['x', 'y']].values
    B_coords = B_cells[['x', 'y']].values

    distances = cdist(A_coords, B_coords, metric='euclidean')
    nearest_idx = distances.argmin(axis=1)
    nearest_dists = distances.min(axis=1)

    A_cells['nearest_B_id'] = B_cells.iloc[nearest_idx]['cellid'].values
    A_cells['nearest_B_dist'] = nearest_dists

    return A_cells[['cellid', 'nearest_B_id', 'nearest_B_dist']]

# 示例计算
nearest_B = compute_nearest_B(df)
print(nearest_B.head())


def compute_B_within_radius(df, radius=10):
    A_cells = df[df['celltype'] == 'A'].copy()
    B_cells = df[df['celltype'] == 'B'].copy()

    A_coords = A_cells[['x', 'y']].values
    B_coords = B_cells[['x', 'y']].values

    # 计算距离矩阵
    distances = cdist(A_coords, B_coords, metric='euclidean')
    
    B_in_radius_count = (distances <= radius).sum(axis=1)
    A_cells['B_count_within_R'] = B_in_radius_count
    A_cells['radius'] = radius

    return A_cells[['cellid', 'B_count_within_R', 'radius']]

# 示例计算
radius_stats = compute_B_within_radius(df, radius=10)
print(radius_stats.head())


# 只为A细胞生成答案数据，确保与C++输出格式一致
A_cells_with_answers = df[df['celltype'] == 'A'].copy()
A_cells_with_answers = A_cells_with_answers.merge(nearest_B, on='cellid', how='left')
A_cells_with_answers = A_cells_with_answers.merge(radius_stats, on='cellid', how='left')

# 保存完整的原始数据（所有细胞）
df.to_csv("test_cells.csv", index=False)

# 保存A细胞的答案数据（与C++输出格式匹配）
A_cells_with_answers.to_csv("test_cells_with_answers.csv", index=False)

print("=== 数据生成统计 ===")
print(f"总细胞数: {len(df)}")
print(f"A细胞数: {len(A_cells_with_answers)}")
print(f"B细胞数: {len(df) - len(A_cells_with_answers)}")
print("\n=== A细胞答案数据示例 ===")
print(A_cells_with_answers.head())
