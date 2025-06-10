#pragma once
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
using namespace std;




// 暴力搜索算法
vector<CellAnalysisResult> bruteForceSearch(const vector<Cell>& A_cells, const vector<Cell>& B_cells, double radius = 10.0) {
    vector<CellAnalysisResult> results;
    
    // 对每个A细胞进行分析
    for (const auto& A_cell : A_cells) {
        CellAnalysisResult result;
        result.cellid = A_cell.id;
        result.x = A_cell.x;
        result.y = A_cell.y;
        result.celltype = A_cell.type;
        result.radius = radius;
        
        // 初始化最近距离为无穷大
        double min_distance = numeric_limits<double>::max();
        int nearest_B_id = -1;
        int B_count_within_radius = 0;
        
        // 暴力搜索所有B细胞
        for (const auto& B_cell : B_cells) {
            double distance = calculateDistance(A_cell, B_cell);
            
            // 更新最近的B细胞
            if (distance < min_distance) {
                min_distance = distance;
                nearest_B_id = B_cell.id;
            }
            
            // 计算半径内的B细胞数量
            if (distance <= radius) {
                B_count_within_radius++;
            }
        }
        
        result.nearest_B_id = nearest_B_id;
        result.nearest_B_dist = min_distance;
        result.B_count_within_radius = B_count_within_radius;
        
        results.push_back(result);
    }
    
    return results;
}