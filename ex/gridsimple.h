#pragma once
#include <vector>
#include <cmath>
#include <limits>
#include <iostream>
#include <algorithm>
#include "datastruct.h"
using namespace std;



class SpatialGridSimple {
private:
    double minX, maxX, minY, maxY;
    double cellSize;
    int gridWidth, gridHeight;
    // 网格存储：每个格子存指向 B 细胞的指针列表
    vector<vector<vector<const Cell*>>> gridArr;

    // 计算坐标 x 对应的格子索引 gx = floor((x - minX)/cellSize)
    inline int coordToGridX(double x) const {
        int ix = static_cast<int>(floor((x - minX) / cellSize));
        return ix;
    }
    inline int coordToGridY(double y) const {
        int iy = static_cast<int>(floor((y - minY) / cellSize));
        return iy;
    }
    // 判断索引是否在合法范围 [0, gridWidth) / [0, gridHeight)
    inline bool inRange(int gx, int gy) const {
        return (gx >= 0 && gx < gridWidth && gy >= 0 && gy < gridHeight);
    }

public:
    // 构造时传入 B 细胞列表，只插入 type=='B' 的细胞
    SpatialGridSimple(const vector<Cell>& cells, double cell_size)
        : minX(0.0), maxX(0.0), minY(0.0), maxY(0.0),
          cellSize(cell_size), gridWidth(0), gridHeight(0)
    {
        if (cells.empty()) {
            // 空数据：1x1 空网格
            minX = minY = 0.0;
            maxX = maxY = 0.0;
            gridWidth = gridHeight = 1;
            gridArr.assign(1, vector<vector<const Cell*>>(1));
            return;
        }
        // 计算 B 细胞边界
        bool firstB = true;
        for (const auto& c : cells) {
            if (c.type != 'B') continue;
            if (firstB) {
                minX = maxX = c.x;
                minY = maxY = c.y;
                firstB = false;
            } else {
                minX = min(minX, c.x);
                maxX = max(maxX, c.x);
                minY = min(minY, c.y);
                maxY = max(maxY, c.y);
            }
        }
        if (firstB) {
            // 没有 B 细胞：同样建立 1x1 空网格
            minX = minY = 0.0;
            maxX = maxY = 0.0;
            gridWidth = gridHeight = 1;
            gridArr.assign(1, vector<vector<const Cell*>>(1));
            return;
        }
        // 计算网格尺寸，不加额外缓存
        double spanX = maxX - minX;
        double spanY = maxY - minY;
        if (spanX <= 0.0) spanX = 1.0;
        if (spanY <= 0.0) spanY = 1.0;
        gridWidth = static_cast<int>(ceil(spanX / cellSize));
        gridHeight = static_cast<int>(ceil(spanY / cellSize));
        if (gridWidth < 1) gridWidth = 1;
        if (gridHeight < 1) gridHeight = 1;
        // 初始化二维数组
        gridArr.clear();
        gridArr.resize(gridWidth);
        for (int i = 0; i < gridWidth; ++i) {
            gridArr[i].resize(gridHeight);
        }
        // 插入 B 细胞指针
        for (const auto& c : cells) {
            if (c.type != 'B') continue;
            int gx = coordToGridX(c.x);
            int gy = coordToGridY(c.y);
            if (inRange(gx, gy)) {
                gridArr[gx][gy].push_back(&c);
            }
        }
    }

    // 最近邻查询：仅检查中心格子及其 8 邻格（不做盒状剪枝）
    // 若所有相关格均无 B，则返回 (-1, -1.0)
    pair<int,double> findNearestB(const Cell& queryCell) const {
        int agx = coordToGridX(queryCell.x);
        int agy = coordToGridY(queryCell.y);
        double bestDist2 = numeric_limits<double>::infinity();
        int bestId = -1;
        // 枚举中心及 8 邻格
        for (int dx = -1; dx <= 1; ++dx) {
            int gx = agx + dx;
            if (gx < 0 || gx >= gridWidth) continue;
            for (int dy = -1; dy <= 1; ++dy) {
                int gy = agy + dy;
                if (gy < 0 || gy >= gridHeight) continue;
                const auto& bucket = gridArr[gx][gy];
                for (const Cell* pb : bucket) {
                    double dx2 = queryCell.x - pb->x;
                    double dy2 = queryCell.y - pb->y;
                    double d2 = dx2*dx2 + dy2*dy2;
                    if (d2 < bestDist2) {
                        bestDist2 = d2;
                        bestId = pb->id;
                    }
                }
            }
        }
        if (bestId < 0) {
            return make_pair(-1, -1.0);
        }
        return make_pair(bestId, sqrt(bestDist2));
    }

    // 范围统计查询：检查所有可能格子，不先做盒状剪枝
    int countBCellsWithinRadius(const Cell& queryCell, double radius) const {
        int agx = coordToGridX(queryCell.x);
        int agy = coordToGridY(queryCell.y);
        int dr = static_cast<int>(ceil(radius / cellSize));
        double R2 = radius * radius;
        int count = 0;
        for (int dx = -dr; dx <= dr; ++dx) {
            int gx = agx + dx;
            if (gx < 0 || gx >= gridWidth) continue;
            for (int dy = -dr; dy <= dr; ++dy) {
                int gy = agy + dy;
                if (gy < 0 || gy >= gridHeight) continue;
                const auto& bucket = gridArr[gx][gy];
                for (const Cell* pb : bucket) {
                    double dx2 = queryCell.x - pb->x;
                    double dy2 = queryCell.y - pb->y;
                    double d2 = dx2*dx2 + dy2*dy2;
                    if (d2 <= R2) {
                        ++count;
                    }
                }
            }
        }
        return count;
    }
};

// 网格搜索主函数（简化版）
vector<CellAnalysisResult> gridSearchSimple(
    const vector<Cell>& A_cells,
    const vector<Cell>& B_cells,
    double radius = 10.0)
{
    vector<CellAnalysisResult> results;
    // 若无 B 细胞，返回所有 A 结果标记无邻居
    if (B_cells.empty()) {
        for (const auto& A_cell : A_cells) {
            CellAnalysisResult result;
            result.cellid = A_cell.id;
            result.x = A_cell.x;
            result.y = A_cell.y;
            result.celltype = A_cell.type;
            result.nearest_B_id = -1;
            result.nearest_B_dist = -1.0;
            result.B_count_within_radius = 0;
            result.radius = radius;
            results.push_back(result);
        }
        return results;
    }
    // 选取格子大小（可与优化版不同）
    double cellSize = 0.8*radius; // 例如直接取 radius
    SpatialGridSimple grid(B_cells, cellSize);
    for (const auto& A_cell : A_cells) {
        CellAnalysisResult result;
        result.cellid = A_cell.id;
        result.x = A_cell.x;
        result.y = A_cell.y;
        result.celltype = A_cell.type;
        result.radius = radius;
        // 最近邻：仅中心+8邻
        auto nn = grid.findNearestB(A_cell);
        result.nearest_B_id = nn.first;
        result.nearest_B_dist = nn.second;
        // 范围统计：无盒状剪枝
        result.B_count_within_radius = grid.countBCellsWithinRadius(A_cell, radius);
        results.push_back(result);
    }
    return results;
}
