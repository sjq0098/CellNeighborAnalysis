#!/usr/bin/env python3
"""
测试极端数据分布对算法性能的影响
"""

import subprocess
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import importlib.util

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def run_command_simple(cmd):
    """简单运行命令"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, 
                              encoding='utf-8', errors='ignore', timeout=120)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def parse_cpp_output(output):
    """解析C++程序输出"""
    data = {}
    lines = output.split('\n')
    
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            # 尝试转换为数字
            try:
                if '.' in value:
                    data[key] = float(value)
                elif value.isdigit():
                    data[key] = int(value)
                else:
                    data[key] = value
            except ValueError:
                data[key] = value
    
    return data

def test_distribution_performance(distribution_type, num_cells=20000):
    """测试单个分布类型的性能"""
    print(f"\n=== 测试分布类型: {distribution_type} ===")
    
    # 1. 生成特殊分布数据
    print("步骤1: 生成特殊分布数据...")
    
    if distribution_type == 'extreme_sparse':
        cmd = f"python -c \"import sys; sys.path.append('.'); from generate_testdata_linear import generate_extreme_sparse_data; import pandas as pd; df = generate_extreme_sparse_data({num_cells}, 0.3, 2025); df.to_csv('test_cells.csv', index=False); print('生成极端稀疏分布完成')\""
    else:
        cmd = f"python -c \"import sys; sys.path.append('.'); from generate_testdata_linear import generate_linear_data; import pandas as pd; df = generate_linear_data({num_cells}, 0.3, 2025, '{distribution_type}'); df.to_csv('test_cells.csv', index=False); print('生成{distribution_type}分布完成')\""
    
    success, output, error = run_command_simple(cmd)
    
    if not success:
        print(f"数据生成失败: {error}")
        return None
    
    print("数据生成完成")
    
    # 2. 运行性能测试
    print("步骤2: 运行算法性能测试...")
    
    cmd = "main.exe" if os.name == 'nt' else "./main.exe"
    success, output, error = run_command_simple(cmd)
    
    if not success:
        print(f"性能测试失败: {error}")
        return None
    
    # 3. 解析结果
    try:
        # 从输出中提取时间信息
        import re
        
        bf_match = re.search(r'Brute Force:\s+(\d+)\s+us', output)
        kd_match = re.search(r'KD-Tree:\s+(\d+)\s+us', output)
        grid_match = re.search(r'Grid Search:\s+(\d+)\s+us', output)
        
        if bf_match and kd_match and grid_match:
            bf_time = int(bf_match.group(1))
            kd_time = int(kd_match.group(1))
            grid_time = int(grid_match.group(1))
            
            # 计算加速比
            kd_speedup = bf_time / kd_time if kd_time > 0 else 0
            grid_speedup = bf_time / grid_time if grid_time > 0 else 0
            
            result = {
                'distribution': distribution_type,
                'num_cells': num_cells,
                'brute_force_time': bf_time,
                'kd_tree_time': kd_time,
                'grid_search_time': grid_time,
                'kd_speedup': kd_speedup,
                'grid_speedup': grid_speedup
            }
            
            print(f"测试完成:")
            print(f"  暴力搜索: {bf_time:,} μs")
            print(f"  KD树:    {kd_time:,} μs ({kd_speedup:.1f}x)")
            print(f"  网格搜索: {grid_time:,} μs ({grid_speedup:.1f}x)")
            
            return result
        else:
            print("无法解析性能数据")
            return None
            
    except Exception as e:
        print(f"解析结果失败: {e}")
        return None

def visualize_distribution_data(distribution_type, save_path=None):
    """可视化数据分布"""
    try:
        df = pd.read_csv("test_cells.csv")
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # 分别绘制A和B细胞
        a_cells = df[df['celltype'] == 'A']
        b_cells = df[df['celltype'] == 'B']
        
        ax.scatter(a_cells['x'], a_cells['y'], c='red', alpha=0.6, s=2, label='A细胞')
        ax.scatter(b_cells['x'], b_cells['y'], c='blue', alpha=0.8, s=3, label='B细胞')
        
        ax.set_title(f'{distribution_type} 分布 (总细胞数: {len(df):,})', fontweight='bold', fontsize=14)
        ax.set_xlabel('X坐标')
        ax.set_ylabel('Y坐标')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"分布图已保存到: {save_path}")
        
        plt.show()
        
        return fig
        
    except Exception as e:
        print(f"可视化失败: {e}")
        return None

def run_extreme_distribution_tests():
    """运行极端分布测试"""
    print("极端数据分布算法性能测试")
    print("="*60)
    
    # 定义要测试的分布类型
    distributions = [
        ('uniform', '均匀分布(基准)'),
        ('single_line', '单线分布'),
        ('multi_lines', '多线分布'), 
        ('cross_lines', '交叉线分布'),
        ('spiral', '螺旋分布'),
        ('sine_wave', '正弦波分布'),
        ('extreme_sparse', '极端稀疏分布')
    ]
    
    test_size = 20000  # 测试规模
    results = []
    
    # 首先生成基准的均匀分布
    print("生成基准数据...")
    success, output, error = run_command_simple(f"python generate_testdata.py --num_cells {test_size} --seed 2025")
    if success:
        baseline_result = test_distribution_performance('uniform', test_size)
        if baseline_result:
            results.append(baseline_result)
    
    # 测试每种极端分布
    for dist_type, dist_name in distributions[1:]:  # 跳过uniform因为已经测试了
        print(f"\n{'='*60}")
        print(f"测试分布: {dist_name}")
        print("="*60)
        
        result = test_distribution_performance(dist_type, test_size)
        if result:
            results.append(result)
            
            # 保存分布可视化
            save_path = f"{dist_type}_distribution.png"
            visualize_distribution_data(dist_type, save_path)
        
        time.sleep(1)  # 短暂等待
    
    return results

def analyze_distribution_effects(results):
    """分析分布对算法的影响"""
    if not results:
        print("没有测试结果进行分析")
        return
    
    print("\n" + "="*80)
    print("极端分布算法性能分析报告")
    print("="*80)
    
    # 创建结果表格
    print(f"{'分布类型':<15} {'暴力搜索':<12} {'KD树':<12} {'网格搜索':<12} {'KD加速比':<10} {'网格加速比':<10}")
    print("-" * 80)
    
    baseline_result = None
    
    for result in results:
        dist_name = result['distribution']
        if dist_name == 'uniform':
            baseline_result = result
            dist_display = '均匀分布(基准)'
        else:
            dist_display = {
                'single_line': '单线分布',
                'multi_lines': '多线分布',
                'cross_lines': '交叉线分布',
                'spiral': '螺旋分布',
                'sine_wave': '正弦波分布',
                'extreme_sparse': '极端稀疏分布'
            }.get(dist_name, dist_name)
        
        print(f"{dist_display:<15} "
              f"{result['brute_force_time']:<12,} "
              f"{result['kd_tree_time']:<12,} "
              f"{result['grid_search_time']:<12,} "
              f"{result['kd_speedup']:<10.1f} "
              f"{result['grid_speedup']:<10.1f}")
    
    # 相对基准的性能分析
    if baseline_result:
        print(f"\n{'='*80}")
        print("相对于均匀分布的性能变化")
        print("="*80)
        print(f"{'分布类型':<15} {'暴力搜索变化':<15} {'KD树变化':<15} {'网格搜索变化':<15}")
        print("-" * 65)
        
        baseline_bf = baseline_result['brute_force_time']
        baseline_kd = baseline_result['kd_tree_time']
        baseline_grid = baseline_result['grid_search_time']
        
        for result in results:
            if result['distribution'] == 'uniform':
                continue
                
            dist_display = {
                'single_line': '单线分布',
                'multi_lines': '多线分布', 
                'cross_lines': '交叉线分布',
                'spiral': '螺旋分布',
                'sine_wave': '正弦波分布',
                'extreme_sparse': '极端稀疏分布'
            }.get(result['distribution'], result['distribution'])
            
            bf_change = result['brute_force_time'] / baseline_bf
            kd_change = result['kd_tree_time'] / baseline_kd  
            grid_change = result['grid_search_time'] / baseline_grid
            
            print(f"{dist_display:<15} "
                  f"{bf_change:<15.2f} "
                  f"{kd_change:<15.2f} "
                  f"{grid_change:<15.2f}")
    
    # 找出最适合的算法
    print(f"\n{'='*80}")
    print("算法适用性分析")
    print("="*80)
    
    for result in results:
        dist_name = result['distribution']
        kd_speedup = result['kd_speedup']
        grid_speedup = result['grid_speedup']
        
        if grid_speedup > kd_speedup and grid_speedup > 5:
            best_algo = "网格搜索占优"
        elif kd_speedup > grid_speedup and kd_speedup > 3:
            best_algo = "KD树占优"
        elif abs(kd_speedup - grid_speedup) < 1:
            best_algo = "性能接近"
        else:
            best_algo = "暴力搜索可能更优"
        
        dist_display = {
            'uniform': '均匀分布',
            'single_line': '单线分布',
            'multi_lines': '多线分布',
            'cross_lines': '交叉线分布', 
            'spiral': '螺旋分布',
            'sine_wave': '正弦波分布',
            'extreme_sparse': '极端稀疏分布'
        }.get(dist_name, dist_name)
        
        print(f"{dist_display:<15}: {best_algo}")

def plot_distribution_comparison(results):
    """绘制分布比较图表"""
    if not results:
        return
    
    print("\n=== 生成分布比较图表 ===")
    
    # 提取数据
    distributions = [r['distribution'] for r in results]
    bf_times = [r['brute_force_time'] for r in results]
    kd_times = [r['kd_tree_time'] for r in results]
    grid_times = [r['grid_search_time'] for r in results]
    kd_speedups = [r['kd_speedup'] for r in results]
    grid_speedups = [r['grid_speedup'] for r in results]
    
    # 分布名称映射
    dist_names = {
        'uniform': '均匀分布',
        'single_line': '单线分布',
        'multi_lines': '多线分布',
        'cross_lines': '交叉线分布',
        'spiral': '螺旋分布',
        'sine_wave': '正弦波分布',
        'extreme_sparse': '极端稀疏分布'
    }
    
    display_names = [dist_names.get(d, d) for d in distributions]
    
    # 创建图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('极端数据分布对算法性能的影响', fontsize=16, fontweight='bold')
    
    x = np.arange(len(distributions))
    width = 0.25
    
    # 图1: 绝对时间比较
    ax1.bar(x - width, bf_times, width, label='暴力搜索', color='red', alpha=0.8)
    ax1.bar(x, kd_times, width, label='KD树', color='blue', alpha=0.8)
    ax1.bar(x + width, grid_times, width, label='网格搜索', color='green', alpha=0.8)
    
    ax1.set_title('算法执行时间对比', fontweight='bold')
    ax1.set_xlabel('数据分布类型')
    ax1.set_ylabel('执行时间 (微秒)')
    ax1.set_yscale('log')
    ax1.set_xticks(x)
    ax1.set_xticklabels(display_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 图2: 加速比比较
    ax2.bar(x - width/2, kd_speedups, width, label='KD树加速比', color='blue', alpha=0.8)
    ax2.bar(x + width/2, grid_speedups, width, label='网格搜索加速比', color='green', alpha=0.8)
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='基准线')
    
    ax2.set_title('相对于暴力搜索的加速比', fontweight='bold')
    ax2.set_xlabel('数据分布类型')
    ax2.set_ylabel('加速比')
    ax2.set_xticks(x)
    ax2.set_xticklabels(display_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 图3: KD树性能变化
    if len(results) > 1:
        baseline_kd = kd_times[0] if distributions[0] == 'uniform' else kd_times[distributions.index('uniform')] if 'uniform' in distributions else kd_times[0]
        kd_relative = [t / baseline_kd for t in kd_times]
        
        ax3.plot(range(len(distributions)), kd_relative, 'o-', color='blue', linewidth=2, markersize=8, label='KD树相对性能')
        ax3.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='基准线')
        
        ax3.set_title('KD树算法在不同分布下的性能变化', fontweight='bold')
        ax3.set_xlabel('数据分布类型')
        ax3.set_ylabel('相对性能 (基准=1)')
        ax3.set_xticks(range(len(distributions)))
        ax3.set_xticklabels(display_names, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 图4: 网格搜索性能变化
    if len(results) > 1:
        baseline_grid = grid_times[0] if distributions[0] == 'uniform' else grid_times[distributions.index('uniform')] if 'uniform' in distributions else grid_times[0]
        grid_relative = [t / baseline_grid for t in grid_times]
        
        ax4.plot(range(len(distributions)), grid_relative, 'o-', color='green', linewidth=2, markersize=8, label='网格搜索相对性能')
        ax4.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='基准线')
        
        ax4.set_title('网格搜索算法在不同分布下的性能变化', fontweight='bold')
        ax4.set_xlabel('数据分布类型')
        ax4.set_ylabel('相对性能 (基准=1)')
        ax4.set_xticks(range(len(distributions)))
        ax4.set_xticklabels(display_names, rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = "extreme_distributions_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"比较图表已保存到: {save_path}")
    
    plt.show()
    
    return fig

def save_distribution_results(results, filename="extreme_distribution_results.csv"):
    """保存测试结果"""
    try:
        df = pd.DataFrame(results)
        df.to_csv(filename, index=False)
        print(f"结果已保存到: {filename}")
    except Exception as e:
        print(f"保存失败: {e}")

def main():
    """主函数"""
    print("极端数据分布算法性能测试")
    print("="*50)
    
    # 检查依赖
    if not os.path.exists("main.exe"):
        print("找不到 main.exe，请先编译C++程序")
        return 1
    
    if not os.path.exists("generate_testdata_linear.py"):
        print("找不到 generate_testdata_linear.py")
        return 1
    
    try:
        import numpy
        import pandas
        import matplotlib
        print("Python依赖检查通过")
    except ImportError as e:
        print(f"Python依赖缺失: {e}")
        return 1
    
    # 运行测试
    results = run_extreme_distribution_tests()
    
    if not results:
        print("测试失败，没有获得结果")
        return 1
    
    # 保存结果
    save_distribution_results(results)
    
    # 分析结果
    analyze_distribution_effects(results)
    
    # 绘制比较图表
    try:
        plot_distribution_comparison(results)
        print("\n极端分布测试完成！")
    except Exception as e:
        print(f"绘图失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())