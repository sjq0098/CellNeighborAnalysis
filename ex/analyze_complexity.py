#!/usr/bin/env python3
"""
算法复杂度分析脚本
拟合算法时间与数据规模的关系并绘制图表
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats
import os
import sys

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def linear_func(x, a, b):
    """线性函数: y = ax + b"""
    return a * x + b

def quadratic_func(x, a, b, c):
    """二次函数: y = ax² + bx + c"""
    return a * x**2 + b * x + c

def nlogn_func(x, a, b):
    """n log n 函数: y = a * x * log(x) + b"""
    return a * x * np.log(x) + b

def power_func(x, a, b):
    """幂函数: y = a * x^b"""
    return a * (x ** b)

def fit_complexity(sizes, times, algorithm_name):
    """拟合算法复杂度"""
    print(f"\n=== {algorithm_name} 复杂度分析 ===")
    
    # 转换为numpy数组
    sizes = np.array(sizes)
    times = np.array(times)
    
    # 过滤掉时间为0的数据点
    valid_mask = times > 0
    sizes_valid = sizes[valid_mask]
    times_valid = times[valid_mask]
    
    if len(sizes_valid) < 3:
        print(f"❌ {algorithm_name}: 有效数据点不足")
        return None
    
    results = {}
    
    # 1. 对数-对数回归分析 (log-log)
    log_sizes = np.log(sizes_valid)
    log_times = np.log(times_valid)
    
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_sizes, log_times)
        results['log_log'] = {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2,
            'complexity': f"O(n^{slope:.2f})"
        }
        print(f"对数-对数拟合: 复杂度 = O(n^{slope:.2f}), R² = {r_value**2:.4f}")
    except Exception as e:
        print(f"对数-对数拟合失败: {e}")
        results['log_log'] = None
    
    # 2. 线性拟合 O(n)
    try:
        popt, pcov = curve_fit(linear_func, sizes_valid, times_valid)
        y_pred = linear_func(sizes_valid, *popt)
        ss_res = np.sum((times_valid - y_pred) ** 2)
        ss_tot = np.sum((times_valid - np.mean(times_valid)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        results['linear'] = {
            'params': popt,
            'r_squared': r_squared,
            'formula': f"T = {popt[0]:.2e}*n + {popt[1]:.2e}"
        }
        print(f"线性拟合 O(n): R² = {r_squared:.4f}")
    except Exception as e:
        print(f"线性拟合失败: {e}")
        results['linear'] = None
    
    # 3. n log n 拟合
    try:
        popt, pcov = curve_fit(nlogn_func, sizes_valid, times_valid)
        y_pred = nlogn_func(sizes_valid, *popt)
        ss_res = np.sum((times_valid - y_pred) ** 2)
        ss_tot = np.sum((times_valid - np.mean(times_valid)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        results['nlogn'] = {
            'params': popt,
            'r_squared': r_squared,
            'formula': f"T = {popt[0]:.2e}*n*log(n) + {popt[1]:.2e}"
        }
        print(f"n log n 拟合 O(n log n): R² = {r_squared:.4f}")
    except Exception as e:
        print(f"n log n 拟合失败: {e}")
        results['nlogn'] = None
    
    # 4. 二次拟合 O(n²)
    try:
        popt, pcov = curve_fit(quadratic_func, sizes_valid, times_valid)
        y_pred = quadratic_func(sizes_valid, *popt)
        ss_res = np.sum((times_valid - y_pred) ** 2)
        ss_tot = np.sum((times_valid - np.mean(times_valid)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        results['quadratic'] = {
            'params': popt,
            'r_squared': r_squared,
            'formula': f"T = {popt[0]:.2e}*n² + {popt[1]:.2e}*n + {popt[2]:.2e}"
        }
        print(f"二次拟合 O(n²): R² = {r_squared:.4f}")
    except Exception as e:
        print(f"二次拟合失败: {e}")
        results['quadratic'] = None
    
    # 找到最佳拟合
    best_fit = None
    best_r2 = -1
    for fit_type, result in results.items():
        if result and result['r_squared'] > best_r2:
            best_r2 = result['r_squared']
            best_fit = fit_type
    
    if best_fit:
        print(f"最佳拟合: {best_fit} (R² = {best_r2:.4f})")
        results['best_fit'] = best_fit
    
    return results

def plot_complexity_analysis(data, save_path="complexity_analysis.png"):
    """绘制复杂度分析图表"""
    print("\n=== 绘制复杂度分析图表 ===")
    
    # 创建图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('算法复杂度分析', fontsize=16, fontweight='bold')
    
    sizes = np.array(data['size'])
    algorithms = ['brute_force', 'kd_tree', 'grid_search']
    algorithm_names = ['暴力搜索', 'KD树', '网格搜索']
    colors = ['red', 'blue', 'green']
    
    # 图1: 线性坐标系时间比较
    ax1.set_title('算法性能比较 (线性坐标)', fontweight='bold')
    for i, (alg, name, color) in enumerate(zip(algorithms, algorithm_names, colors)):
        times = np.array(data[alg])
        ax1.plot(sizes, times, 'o-', color=color, label=name, linewidth=2, markersize=6)
    
    ax1.set_xlabel('数据规模 (细胞数量)')
    ax1.set_ylabel('执行时间 (微秒)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 图2: 对数坐标系时间比较
    ax2.set_title('算法性能比较 (对数坐标)', fontweight='bold')
    for i, (alg, name, color) in enumerate(zip(algorithms, algorithm_names, colors)):
        times = np.array(data[alg])
        valid_mask = times > 0
        ax2.loglog(sizes[valid_mask], times[valid_mask], 'o-', color=color, label=name, linewidth=2, markersize=6)
    
    ax2.set_xlabel('数据规模 (细胞数量)')
    ax2.set_ylabel('执行时间 (微秒)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 图3: 加速比分析
    ax3.set_title('相对于暴力搜索的加速比', fontweight='bold')
    bf_times = np.array(data['brute_force'])
    
    for i, (alg, name, color) in enumerate(zip(algorithms[1:], algorithm_names[1:], colors[1:])):
        times = np.array(data[alg])
        speedup = bf_times / times
        valid_mask = (times > 0) & (bf_times > 0)
        ax3.plot(sizes[valid_mask], speedup[valid_mask], 'o-', color=color, label=f'{name}加速比', linewidth=2, markersize=6)
    
    ax3.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='基准线 (暴力搜索)')
    ax3.set_xlabel('数据规模 (细胞数量)')
    ax3.set_ylabel('加速比')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # 图4: 理论复杂度比较
    ax4.set_title('理论复杂度对比', fontweight='bold')
    
    # 生成理论曲线数据
    x_theory = np.linspace(sizes.min(), sizes.max(), 100)
    
    # 归一化到相同起点
    n0 = sizes[0]
    t0 = 1000  # 基准时间 (微秒)
    
    # 理论复杂度曲线
    theory_linear = t0 * (x_theory / n0)
    theory_nlogn = t0 * (x_theory / n0) * np.log(x_theory / n0)
    theory_quadratic = t0 * (x_theory / n0) ** 2
    
    ax4.plot(x_theory, theory_linear, '--', color='green', alpha=0.7, label='O(n) - 理论线性', linewidth=2)
    ax4.plot(x_theory, theory_nlogn, '--', color='blue', alpha=0.7, label='O(n log n) - 理论', linewidth=2)
    ax4.plot(x_theory, theory_quadratic, '--', color='red', alpha=0.7, label='O(n2) - 理论二次', linewidth=2)
    
    # 实际数据点
    for i, (alg, name, color) in enumerate(zip(algorithms, algorithm_names, colors)):
        times = np.array(data[alg])
        ax4.plot(sizes, times, 'o', color=color, label=f'{name} - 实际数据', markersize=6, alpha=0.8)
    
    ax4.set_xlabel('数据规模 (细胞数量)')
    ax4.set_ylabel('执行时间 (微秒)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    ax4.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {save_path}")
    
    return fig

def create_complexity_report(results_data, fit_results):
    """创建复杂度分析报告"""
    print("\n" + "="*60)
    print("算法复杂度分析报告")
    print("="*60)
    
    algorithms = ['brute_force', 'kd_tree', 'grid_search']
    algorithm_names = ['暴力搜索', 'KD树搜索', '网格搜索']
    theoretical = ['O(n²)', 'O(n log n)', 'O(n)']
    
    print(f"{'算法':<12} {'理论复杂度':<15} {'实际拟合':<15} {'拟合度(R²)':<12} {'结论'}")
    print("-" * 70)
    
    for i, (alg, name, theory) in enumerate(zip(algorithms, algorithm_names, theoretical)):
        if alg in fit_results and fit_results[alg]:
            result = fit_results[alg]
            if 'log_log' in result and result['log_log']:
                actual_complexity = result['log_log']['complexity']
                r_squared = result['log_log']['r_squared']
                
                # 判断是否符合理论
                slope = result['log_log']['slope']
                if alg == 'brute_force':
                    expected_slope = 2.0
                elif alg == 'kd_tree':
                    expected_slope = 1.1  # n log n 的斜率约为 1+
                else:  # grid_search
                    expected_slope = 1.0
                
                deviation = abs(slope - expected_slope)
                if deviation < 0.3:
                    conclusion = "符合理论"
                elif deviation < 0.5:
                    conclusion = "基本符合"
                else:
                    conclusion = "偏离理论"
                
                print(f"{name:<12} {theory:<15} {actual_complexity:<15} {r_squared:<12.3f} {conclusion}")
            else:
                print(f"{name:<12} {theory:<15} {'拟合失败':<15} {'N/A':<12} {'无法分析'}")
        else:
            print(f"{name:<12} {theory:<15} {'无数据':<15} {'N/A':<12} {'无数据'}")
    
    # 性能排名分析
    print(f"\n{'='*60}")
    print("性能排名分析 (基于最大规模数据)")
    print("="*60)
    
    sizes = np.array(results_data['size'])
    max_size_idx = np.argmax(sizes)
    max_size = sizes[max_size_idx]
    
    performance_data = []
    for i, (alg, name) in enumerate(zip(algorithms, algorithm_names)):
        time = results_data[alg][max_size_idx]
        performance_data.append((name, time))
    
    # 按时间排序
    performance_data.sort(key=lambda x: x[1])
    
    print(f"在 {max_size:,} 个细胞规模下:")
    for rank, (name, time) in enumerate(performance_data, 1):
        speedup = performance_data[-1][1] / time  # 相对于最慢算法的加速比
        print(f"{rank}. {name:<12} {time:>10,} μs  ({speedup:.1f}x 快于最慢算法)")

def main():
    """主函数"""
    print("算法复杂度分析脚本")
    print("="*50)
    
    # 检查是否存在测试结果文件
    if os.path.exists("test_results.csv"):
        print("读取现有测试结果...")
        try:
            df = pd.read_csv("test_results.csv")
            print(f"成功读取 {len(df)} 条测试记录")
        except Exception as e:
            print(f"读取CSV文件失败: {e}")
            return 1
    else:
        print(" 找不到 test_results.csv 文件")
        print("请先运行 test_simple.py 生成测试数据")
        return 1
    
    # 检查数据完整性
    required_columns = ['size', 'brute_force', 'kd_tree', 'grid_search']
    for col in required_columns:
        if col not in df.columns:
            print(f"缺少必要的列: {col}")
            return 1
    
    # 转换为字典格式
    data = {}
    for col in required_columns:
        data[col] = df[col].tolist()
    
    print(f"数据规模范围: {min(data['size']):,} - {max(data['size']):,} 个细胞")
    
    # 进行复杂度拟合分析
    algorithms = ['brute_force', 'kd_tree', 'grid_search']
    algorithm_names = ['暴力搜索', 'KD树搜索', '网格搜索']
    
    fit_results = {}
    for alg, name in zip(algorithms, algorithm_names):
        try:
            result = fit_complexity(data['size'], data[alg], name)
            fit_results[alg] = result
        except Exception as e:
            print(f"{name} 复杂度分析失败: {e}")
            fit_results[alg] = None
    
    # 绘制图表
    try:
        fig = plot_complexity_analysis(data)
        plt.show()
    except Exception as e:
        print(f"绘图失败: {e}")
        print("可能需要安装matplotlib: pip install matplotlib")
    
    # 生成分析报告
    create_complexity_report(data, fit_results)
    
    # 保存详细拟合结果
    try:
        report_file = "complexity_analysis_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("算法复杂度分析详细报告\n")
            f.write("="*50 + "\n\n")
            
            for alg, name in zip(algorithms, algorithm_names):
                f.write(f"{name} 分析结果:\n")
                f.write("-" * 30 + "\n")
                
                if alg in fit_results and fit_results[alg]:
                    result = fit_results[alg]
                    for fit_type, fit_data in result.items():
                        if fit_type != 'best_fit' and fit_data:
                            f.write(f"{fit_type}: {fit_data}\n")
                    f.write(f"\n最佳拟合: {result.get('best_fit', 'N/A')}\n")
                else:
                    f.write("分析失败\n")
                f.write("\n")
        
        print(f"\n详细报告已保存到: {report_file}")
    except Exception as e:
        print(f"保存报告失败: {e}")
    
    print("\n复杂度分析完成！")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 