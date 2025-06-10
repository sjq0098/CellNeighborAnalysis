#!/usr/bin/env python3
"""
简单的算法性能测试脚本
逐步测试不同规模的数据
"""

import subprocess
import os
import sys
import time

def run_command(cmd, description=""):
    """运行命令并返回结果"""
    print(f"正在执行: {description}")
    print(f"命令: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, 
                              encoding='utf-8', errors='ignore', timeout=60)
        
        if result.returncode == 0:
            print("✅ 成功")
            if result.stdout.strip():
                print("输出:")
                print(result.stdout)
            return True, result.stdout
        else:
            print("❌ 失败")
            if result.stderr.strip():
                print("错误信息:")
                print(result.stderr)
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        print("❌ 超时")
        return False, "命令执行超时"
    except Exception as e:
        print(f"❌ 异常: {e}")
        return False, str(e)

def test_data_generation():
    """测试数据生成"""
    print("=" * 50)
    print("测试数据生成")
    print("=" * 50)
    
    # 测试小规模数据
    test_sizes = [1000, 5000, 10000]
    
    for size in test_sizes:
        print(f"\n--- 测试 {size} 个细胞 ---")
        
        cmd = f"python generate_testdata.py --num_cells {size} --seed 2025"
        success, output = run_command(cmd, f"生成 {size} 个细胞")
        
        if success:
            # 检查文件是否生成
            if os.path.exists("test_cells.csv"):
                file_size = os.path.getsize("test_cells.csv")
                print(f"文件大小: {file_size} 字节")
            else:
                print("❌ test_cells.csv 文件未生成")
                return False
        else:
            print(f"❌ 数据生成失败: {output}")
            return False
        
        time.sleep(1)
    
    return True

def test_cpp_compilation():
    """测试C++编译"""
    print("=" * 50)
    print("测试C++编译")
    print("=" * 50)
    
    if not os.path.exists("main.cpp"):
        print("❌ 找不到 main.cpp 文件")
        return False
    
    cmd = "g++ -std=c++11 -O2 -Wall -o main.exe main.cpp"
    success, output = run_command(cmd, "编译C++程序")
    
    if success and os.path.exists("main.exe"):
        print("✅ 编译成功，生成 main.exe")
        return True
    else:
        print("❌ 编译失败")
        return False

def test_cpp_execution():
    """测试C++程序执行"""
    print("=" * 50)
    print("测试C++程序执行")
    print("=" * 50)
    
    if not os.path.exists("main.exe"):
        print("❌ 找不到 main.exe 文件")
        return False
    
    if not os.path.exists("test_cells.csv"):
        print("❌ 找不到 test_cells.csv 文件")
        return False
    
    # Windows兼容性：直接使用main.exe而不是./main.exe
    cmd = "main.exe" if os.name == 'nt' else "./main.exe"
    success, output = run_command(cmd, "运行C++算法测试")
    
    if success:
        print("✅ C++程序执行成功")
        
        # 检查输出中是否包含时间信息
        if "Brute Force:" in output and "KD-Tree:" in output and "Grid Search:" in output:
            print("✅ 找到性能测试结果")
            
            # 提取时间信息
            lines = output.split('\n')
            for line in lines:
                if "Brute Force:" in line or "KD-Tree:" in line or "Grid Search:" in line:
                    print(f"  {line.strip()}")
            
            return True
        else:
            print("⚠️ 输出中未找到预期的性能数据")
            return False
    else:
        print(f"❌ C++程序执行失败: {output}")
        return False

def run_scale_test():
    """运行规模测试"""
    print("=" * 50)
    print("运行规模测试")
    print("=" * 50)
    
    test_sizes = [1000, 2000, 5000, 8000,10000,15000, 20000, 50000,80000, 100000]
    results = []
    
    for size in test_sizes:
        print(f"\n--- 测试规模: {size} ---")
        
        # 生成数据
        cmd = f"python generate_testdata.py --num_cells {size} --seed {2025 + size}"
        success, _ = run_command(cmd, f"生成 {size} 个细胞")
        
        if not success:
            print(f"❌ 数据生成失败，跳过规模 {size}")
            continue
        
        # 运行测试
        cmd = "main.exe" if os.name == 'nt' else "./main.exe"
        success, output = run_command(cmd, f"测试 {size} 个细胞")
        
        if success:
            # 解析时间数据
            import re
            bf_match = re.search(r'Brute Force:\s+(\d+)\s+us', output)
            kd_match = re.search(r'KD-Tree:\s+(\d+)\s+us', output)
            grid_match = re.search(r'Grid Search:\s+(\d+)\s+us', output)
            
            if bf_match and kd_match and grid_match:
                bf_time = int(bf_match.group(1))
                kd_time = int(kd_match.group(1))
                grid_time = int(grid_match.group(1))
                
                results.append({
                    'size': size,
                    'brute_force': bf_time,
                    'kd_tree': kd_time,
                    'grid_search': grid_time
                })
                
                print(f"✅ 测试完成:")
                print(f"  暴力搜索: {bf_time:,} us")
                print(f"  KD树:    {kd_time:,} us")
                print(f"  网格搜索: {grid_time:,} us")
                
                # 计算加速比
                if bf_time > 0:
                    kd_speedup = bf_time / kd_time if kd_time > 0 else 0
                    grid_speedup = bf_time / grid_time if grid_time > 0 else 0
                    print(f"  KD树加速比: {kd_speedup:.2f}x")
                    print(f"  网格加速比: {grid_speedup:.2f}x")
            else:
                print("❌ 无法解析性能数据")
        else:
            print(f"❌ 测试失败")
        
        time.sleep(1)
    
    # 打印汇总结果
    if results:
        print("\n" + "=" * 50)
        print("测试结果汇总")
        print("=" * 50)
        print(f"{'规模':<8} {'暴力搜索':<12} {'KD树':<12} {'网格搜索':<12}")
        print("-" * 50)
        
        for result in results:
            print(f"{result['size']:<8} {result['brute_force']:<12} {result['kd_tree']:<12} {result['grid_search']:<12}")
        
        # 保存结果到CSV
        try:
            import pandas as pd
            df = pd.DataFrame(results)
            df.to_csv("test_results.csv", index=False)
            print(f"\n✅ 结果已保存到 test_results.csv")
        except ImportError:
            print("\n⚠️ 无法保存CSV文件（需要pandas）")
    
    return len(results) > 0

def main():
    """主函数"""
    print("算法性能测试 - 简化版")
    print("=" * 50)
    
    # 检查依赖
    print("检查依赖...")
    
    if not os.path.exists("generate_testdata.py"):
        print("❌ 找不到 generate_testdata.py")
        return 1
    
    if not os.path.exists("main.cpp"):
        print("❌ 找不到 main.cpp")
        return 1
    
    try:
        import numpy
        import pandas
        print("✅ Python依赖检查通过")
    except ImportError as e:
        print(f"❌ Python依赖缺失: {e}")
        return 1
    
    # 步骤1: 测试数据生成
    if not test_data_generation():
        print("❌ 数据生成测试失败")
        return 1
    
    # 步骤2: 测试C++编译
    if not test_cpp_compilation():
        print("❌ C++编译测试失败")
        return 1
    
    # 步骤3: 测试C++执行
    if not test_cpp_execution():
        print("❌ C++执行测试失败")
        return 1
    
    # 步骤4: 运行规模测试
    if not run_scale_test():
        print("❌ 规模测试失败")
        return 1
    
    print("\n" + "=" * 50)
    print("✅ 所有测试完成！")
    print("=" * 50)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 