#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <limits>
#include <chrono>
#include "datastruct.h"
#include "bruce.h"
#include "kdtree.h"
#include "Grid.h"
using namespace std;

// 读取CSV文件
vector<Cell> readCellsFromCSV(const string& filename) {
    vector<Cell> cells;
    ifstream file(filename);
    string line;
    
    if (!file.is_open()) {
        cerr << "Error: Cannot open file " << filename << endl;
        return cells;
    }
    
    // 跳过标题行
    getline(file, line);
    
    while (getline(file, line)) {
        stringstream ss(line);
        string token;
        Cell cell;
        
        // 读取cellid
        getline(ss, token, ',');
        cell.id = stoi(token);
        
        // 读取x坐标
        getline(ss, token, ',');
        cell.x = stod(token);
        
        // 读取y坐标
        getline(ss, token, ',');
        cell.y = stod(token);
        
        // 读取celltype
        getline(ss, token, ',');
        cell.type = token[0];
        
        cells.push_back(cell);
    }
    
    file.close();
    cout << "Successfully loaded " << cells.size() << " cells" << endl;
    return cells;
}


// 将结果写入CSV文件
void writeResultsToCSV(const vector<CellAnalysisResult>& results, const string& filename) {
    ofstream file(filename);
    
    if (!file.is_open()) {
        cerr << "Error: Cannot create output file " << filename << endl;
        return;
    }
    
    // 写入标题行
    file << "cellid,x,y,celltype,nearest_B_id,nearest_B_dist,B_count_within_R,radius\n";
    
    // 写入数据
    for (const auto& result : results) {
        file << result.cellid << ","
             << result.x << ","
             << result.y << ","
             << result.celltype << ","
             << result.nearest_B_id << ","
             << result.nearest_B_dist << ","
             << result.B_count_within_radius << ","
             << result.radius << "\n";
    }
    
    file.close();
    cout << "Results saved to " << filename << endl;
}

// 打印统计信息
void printStatistics(const vector<CellAnalysisResult>& results) {
    if (results.empty()) {
        cout << "No result data" << endl;
        return;
    }
    
    double total_dist = 0.0;
    double min_dist = numeric_limits<double>::max();
    double max_dist = 0.0;
    int total_B_count = 0;
    
    for (const auto& result : results) {
        total_dist += result.nearest_B_dist;
        min_dist = min(min_dist, result.nearest_B_dist);
        max_dist = max(max_dist, result.nearest_B_dist);
        total_B_count += result.B_count_within_radius;
    }
    
    cout << "\n=== Analysis Statistics ===" << endl;
    cout << "Total A cells: " << results.size() << endl;
    cout << "Nearest B cell distance statistics:" << endl;
    cout << "  Average distance: " << total_dist / results.size() << endl;
    cout << "  Minimum distance: " << min_dist << endl;
    cout << "  Maximum distance: " << max_dist << endl;
    cout << "B cells within radius statistics (radius=" << results[0].radius << "):" << endl;
    cout << "  Total count: " << total_B_count << endl;
    cout << "  Average per A cell: " << (double)total_B_count / results.size() << " B cells" << endl;
}

int main() {
    cout << "=== Cell Neighbor Analysis Program ===" << endl;
    
    // 读取测试数据
    vector<Cell> cells = readCellsFromCSV("test_cells.csv");
    if (cells.empty()) {
        cerr << "Failed to load cell data, exiting..." << endl;
        return 1;
    }
    
    // 设置分析半径
    double radius = 1.0;
    cout << "\nUsing analysis radius: " << radius << endl;
    
    // 数据预处理：分离A细胞和B细胞（不计入算法耗时）
    cout << "\nPreparing data..." << endl;
    vector<Cell> A_cells, B_cells;
    for (const auto& cell : cells) {
        if (cell.type == 'A') {
            A_cells.push_back(cell);
        } else if (cell.type == 'B') {
            B_cells.push_back(cell);
        }
    }
    
    cout << "A cells count: " << A_cells.size() << endl;
    cout << "B cells count: " << B_cells.size() << endl;
    
        // ===========================================
    // 算法选择区域 - 通过注释/解注释来切换算法
    // ===========================================
    
    // 选项1：暴力搜索算法（O(n²)复杂度）
    /*
    cout << "\n=== Testing Brute Force Algorithm ===" << endl;
    auto start_time1 = chrono::high_resolution_clock::now();
    vector<CellAnalysisResult> results = bruteForceSearch(A_cells, B_cells, radius);
    auto end_time1 = chrono::high_resolution_clock::now();
    auto duration_ms1 = chrono::duration_cast<chrono::milliseconds>(end_time1 - start_time1);
    auto duration_us1 = chrono::duration_cast<chrono::microseconds>(end_time1 - start_time1);
    cout << "Brute Force completed, time elapsed: " << duration_ms1.count() << " ms (" << duration_us1.count() << " microseconds)" << endl;
    */
    
    // 选项2：KD树优化算法（O(n log n)复杂度）
    /*
    cout << "\n=== Testing KD-Tree Algorithm ===" << endl;
    auto start_time2 = chrono::high_resolution_clock::now();
    vector<CellAnalysisResult> results = kdTreeSearch(A_cells, B_cells, radius);
    auto end_time2 = chrono::high_resolution_clock::now();
    auto duration_ms2 = chrono::duration_cast<chrono::milliseconds>(end_time2 - start_time2);
    auto duration_us2 = chrono::duration_cast<chrono::microseconds>(end_time2 - start_time2);
    cout << "KD-Tree completed, time elapsed: " << duration_ms2.count() << " ms (" << duration_us2.count() << " microseconds)" << endl;
    */
    
    // 选项3：网格搜索算法（空间哈希优化）
    /*
    cout << "\n=== Testing Grid Search Algorithm ===" << endl;
    auto start_time3 = chrono::high_resolution_clock::now();
    vector<CellAnalysisResult> results = gridSearch(A_cells, B_cells, radius);
    auto end_time3 = chrono::high_resolution_clock::now();
    auto duration_ms3 = chrono::duration_cast<chrono::milliseconds>(end_time3 - start_time3);
    auto duration_us3 = chrono::duration_cast<chrono::microseconds>(end_time3 - start_time3);
    cout << "Grid Search completed, time elapsed: " << duration_ms3.count() << " ms (" << duration_us3.count() << " microseconds)" << endl;
    */

    // 多算法性能比较（解注释以启用）
 
    // 运行所有算法进行比较
    cout << "\n=== Multi-Algorithm Performance Comparison ===" << endl;
    
    // 1. 暴力搜索
    auto start_bf = chrono::high_resolution_clock::now();
    vector<CellAnalysisResult> results_bf = bruteForceSearch(A_cells, B_cells, radius);
    auto end_bf = chrono::high_resolution_clock::now();
    auto duration_bf = chrono::duration_cast<chrono::microseconds>(end_bf - start_bf);
    
    // 2. KD树
    auto start_kd = chrono::high_resolution_clock::now();
    vector<CellAnalysisResult> results_kd = kdTreeSearch(A_cells, B_cells, radius);
    auto end_kd = chrono::high_resolution_clock::now();
    auto duration_kd = chrono::duration_cast<chrono::microseconds>(end_kd - start_kd);
    
    // 3. 网格搜索
    auto start_grid = chrono::high_resolution_clock::now();
    vector<CellAnalysisResult> results_grid = gridSearch(A_cells, B_cells, radius);
    auto end_grid = chrono::high_resolution_clock::now();
    auto duration_grid = chrono::duration_cast<chrono::microseconds>(end_grid - start_grid);
    
    // 性能报告
    cout << "\nAlgorithm Performance Report:" << endl;
    cout << "Brute Force:  " << duration_bf.count() << " us" << endl;
    cout << "KD-Tree:      " << duration_kd.count() << " us" << endl;
    cout << "Grid Search:  " << duration_grid.count() << " us" << endl;
    
    // 计算加速比
    if (duration_bf.count() > 0) {
        cout << "\nSpeedup vs Brute Force:" << endl;
        cout << "KD-Tree:     " << (double)duration_bf.count() / duration_kd.count() << "x" << endl;
        cout << "Grid Search: " << (double)duration_bf.count() / duration_grid.count() << "x" << endl;
    }
    
    // 结果验证
    cout << "\n=== Result Verification ===" << endl;
    bool all_match = true;
    
    // 验证KD树与暴力搜索
    if (results_bf.size() == results_kd.size()) {
        for (size_t i = 0; i < results_bf.size(); i++) {
            if (results_bf[i].nearest_B_id != results_kd[i].nearest_B_id ||
                abs(results_bf[i].nearest_B_dist - results_kd[i].nearest_B_dist) > 1e-6 ||
                results_bf[i].B_count_within_radius != results_kd[i].B_count_within_radius) {
                cout << " KD-Tree mismatch at cell " << results_bf[i].cellid << endl;
                all_match = false;
                break;
            }
        }
    }
    
    // 验证网格搜索与暴力搜索
    if (results_bf.size() == results_grid.size()) {
        for (size_t i = 0; i < results_bf.size(); i++) {
            if (results_bf[i].nearest_B_id != results_grid[i].nearest_B_id ||
                abs(results_bf[i].nearest_B_dist - results_grid[i].nearest_B_dist) > 1e-6 ||
                results_bf[i].B_count_within_radius != results_grid[i].B_count_within_radius) {
                cout << "Grid Search mismatch at cell " << results_bf[i].cellid << endl;
                all_match = false;
                break;
            }
        }
    }
    
    if (all_match) {
        cout << "All algorithms produce identical results!" << endl;
    }
    
    // 使用暴力搜索的结果作为标准答案
    vector<CellAnalysisResult> results = results_bf;

    
    // 输出统计信息
    printStatistics(results);
    
    // 保存结果
    writeResultsToCSV(results, "cpp_results.csv");
    
    cout << "\nProgram execution completed!" << endl;
    return 0;
} 