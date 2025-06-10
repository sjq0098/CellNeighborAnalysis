#pragma once
#include <vector>
#include <cmath>
#include <limits>
#include <iostream>
#include <algorithm>
#include "datastruct.h"
using namespace std;

class SpatialGridOptimized {
private:
    double minX, maxX, minY, maxY;
    double cellSize;
    int gridWidth, gridHeight;
    // 网格存储：每个格子存指向 B 细胞的指针列表
    vector<vector<vector<const Cell*>>> gridArr;

    // 计算坐标 x 对应的格子索引 gx = floor((x - minX)/cellSize)
    inline int coordToGridX(double x) const {
        double v = (x - minX) / cellSize;
        int ix = (int)floor(v);
        return ix;
    }
    inline int coordToGridY(double y) const {
        double v = (y - minY) / cellSize;
        int iy = (int)floor(v);
        return iy;
    }
    // 判断索引是否在合法范围 [0, gridWidth) / [0, gridHeight)
    inline bool inRange(int gx, int gy) const {
        if (gx < 0) return false;
        if (gx >= gridWidth) return false;
        if (gy < 0) return false;
        if (gy >= gridHeight) return false;
        return true;
    }
    // 计算点 a 到格子 (gx,gy) 对应矩形区域的最小距离平方
    inline double computeBoxMinDist2(const Cell& a, int gx, int gy) const {
        // 格子矩形在 x 方向: [rectMinX, rectMaxX] = [minX + gx*cellSize, minX + (gx+1)*cellSize]
        double rectMinX = minX + gx * cellSize;
        double rectMaxX = rectMinX + cellSize;
        double dx = 0.0;
        if (a.x < rectMinX) {
            dx = rectMinX - a.x;
        } else if (a.x > rectMaxX) {
            dx = a.x - rectMaxX;
        }
        double rectMinY = minY + gy * cellSize;
        double rectMaxY = rectMinY + cellSize;
        double dy = 0.0;
        if (a.y < rectMinY) {
            dy = rectMinY - a.y;
        } else if (a.y > rectMaxY) {
            dy = a.y - rectMaxY;
        }
        return dx*dx + dy*dy;
    }
    // 计算两点平方距离
    inline static double squaredDistance(const Cell& a, const Cell& b) {
        double dx = a.x - b.x;
        double dy = a.y - b.y;
        return dx*dx + dy*dy;
    }

public:
    // 构造时传入 B 细胞列表，只插入 type=='B' 的细胞
    SpatialGridOptimized(const vector<Cell>& cells, double cell_size)
        : minX(0.0), maxX(0.0), minY(0.0), maxY(0.0),
          cellSize(cell_size), gridWidth(0), gridHeight(0)
    {
        if (cells.empty()) {
            // 空数据，建立 1x1 空网格
            minX = minY = 0.0;
            maxX = maxY = 0.0;
            gridWidth = 1;
            gridHeight = 1;
            gridArr.assign(1, vector<vector<const Cell*>>(1));
            return;
        }
        // 先计算传入 B_cells 的边界
        bool first = true;
        for (size_t i = 0; i < cells.size(); ++i) {
            const Cell& c = cells[i];
            if (c.type != 'B') {
                continue;
            }
            if (first) {
                minX = maxX = c.x;
                minY = maxY = c.y;
                first = false;
            } else {
                if (c.x < minX) minX = c.x;
                if (c.x > maxX) maxX = c.x;
                if (c.y < minY) minY = c.y;
                if (c.y > maxY) maxY = c.y;
            }
        }
        if (first) {
            // 没有 B 细胞
            minX = minY = 0.0;
            maxX = maxY = 0.0;
            gridWidth = 1;
            gridHeight = 1;
            gridArr.assign(1, vector<vector<const Cell*>>(1));
            return;
        }
        // 增加少量缓冲，防止边界点落在最后一格边界上出界
        double spanX = maxX - minX;
        double spanY = maxY - minY;
        if (spanX <= 0.0) {
            spanX = 1.0;
        }
        if (spanY <= 0.0) {
            spanY = 1.0;
        }
        double bufferX = spanX * 0.001;
        double bufferY = spanY * 0.001;
        minX -= bufferX;
        maxX += bufferX;
        minY -= bufferY;
        maxY += bufferY;
        // 计算网格尺寸
        // 至少一格
        gridWidth = static_cast<int>(ceil((maxX - minX) / cellSize));
        if (gridWidth < 1) {
            gridWidth = 1;
        }
        gridHeight = static_cast<int>(ceil((maxY - minY) / cellSize));
        if (gridHeight < 1) {
            gridHeight = 1;
        }
        // 准备二维数组
        gridArr.clear();
        gridArr.resize(gridWidth);
        for (int i = 0; i < gridWidth; ++i) {
            gridArr[i].resize(gridHeight);
        }
        // 插入 B 细胞指针
        for (size_t i = 0; i < cells.size(); ++i) {
            const Cell& c = cells[i];
            if (c.type != 'B') {
                continue;
            }
            int gx = coordToGridX(c.x);
            int gy = coordToGridY(c.y);
            if (gx < 0 || gx >= gridWidth || gy < 0 || gy >= gridHeight) {
                // 坐标若超出边界，则跳过
                continue;
            }
            gridArr[gx][gy].push_back(&c);
        }
        // 可选：打印统计信息
        /*
        std::cout << "[SpatialGridOptimized] gridWidth=" << gridWidth
                  << " gridHeight=" << gridHeight
                  << " cellSize=" << cellSize << std::endl;
        */
    }

    // 查找最近 B 细胞；若无 B，则返回 (-1, -1.0)
    pair<int,double> findNearestB(const Cell& queryCell) const {
        // 先判断是否无 B 细胞
        bool hasB = false;
        for (int i = 0; i < gridWidth; ++i) {
            for (int j = 0; j < gridHeight; ++j) {
                if (!gridArr[i][j].empty()) {
                    hasB = true;
                    break;
                }
            }
            if (hasB) break;
        }
        if (!hasB) {
            return make_pair(-1, -1.0);
        }
        int agx = coordToGridX(queryCell.x);
        int agy = coordToGridY(queryCell.y);
        double bestDist2 = numeric_limits<double>::infinity();
        int bestId = -1;
        // 第一轮：检查中心格子
        if (inRange(agx, agy)) {
            const vector<const Cell*>& bucket = gridArr[agx][agy];
            for (size_t k = 0; k < bucket.size(); ++k) {
                const Cell* pb = bucket[k];
                double d2 = squaredDistance(queryCell, *pb);
                if (d2 < bestDist2) {
                    bestDist2 = d2;
                    bestId = pb->id;
                }
            }
        }
        // 按层扩展环形格子
        int maxLayer = max(gridWidth, gridHeight);
        for (int layer = 1; layer < maxLayer; ++layer) {
            bool anyVisited = false;
            // 左右列: gx = agx - layer, agx + layer; gy from agy - layer to agy + layer
            int gx_left = agx - layer;
            int gx_right = agx + layer;
            for (int dy = -layer; dy <= layer; ++dy) {
                int gy = agy + dy;
                // 左侧格子
                if (gx_left >= 0 && gx_left < gridWidth && gy >= 0 && gy < gridHeight) {
                    double boxDist2 = computeBoxMinDist2(queryCell, gx_left, gy);
                    if (boxDist2 < bestDist2) {
                        anyVisited = true;
                        const vector<const Cell*>& bucket = gridArr[gx_left][gy];
                        for (size_t k = 0; k < bucket.size(); ++k) {
                            const Cell* pb = bucket[k];
                            double d2 = squaredDistance(queryCell, *pb);
                            if (d2 < bestDist2) {
                                bestDist2 = d2;
                                bestId = pb->id;
                            }
                        }
                    }
                }
                // 右侧格子
                if (gx_right >= 0 && gx_right < gridWidth && gy >= 0 && gy < gridHeight) {
                    double boxDist2 = computeBoxMinDist2(queryCell, gx_right, gy);
                    if (boxDist2 < bestDist2) {
                        anyVisited = true;
                        const vector<const Cell*>& bucket = gridArr[gx_right][gy];
                        for (size_t k = 0; k < bucket.size(); ++k) {
                            const Cell* pb = bucket[k];
                            double d2 = squaredDistance(queryCell, *pb);
                            if (d2 < bestDist2) {
                                bestDist2 = d2;
                                bestId = pb->id;
                            }
                        }
                    }
                }
            }
            // 上下行: gy = agy - layer, agy + layer; gx from agx - layer + 1 to agx + layer - 1
            int gy_top = agy - layer;
            int gy_bottom = agy + layer;
            for (int dx = -layer + 1; dx <= layer - 1; ++dx) {
                int gx = agx + dx;
                // 上方格子
                if (gx >= 0 && gx < gridWidth && gy_top >= 0 && gy_top < gridHeight) {
                    double boxDist2 = computeBoxMinDist2(queryCell, gx, gy_top);
                    if (boxDist2 < bestDist2) {
                        anyVisited = true;
                        const vector<const Cell*>& bucket = gridArr[gx][gy_top];
                        for (size_t k = 0; k < bucket.size(); ++k) {
                            const Cell* pb = bucket[k];
                            double d2 = squaredDistance(queryCell, *pb);
                            if (d2 < bestDist2) {
                                bestDist2 = d2;
                                bestId = pb->id;
                            }
                        }
                    }
                }
                // 下方格子
                if (gx >= 0 && gx < gridWidth && gy_bottom >= 0 && gy_bottom < gridHeight) {
                    double boxDist2 = computeBoxMinDist2(queryCell, gx, gy_bottom);
                    if (boxDist2 < bestDist2) {
                        anyVisited = true;
                        const vector<const Cell*>& bucket = gridArr[gx][gy_bottom];
                        for (size_t k = 0; k < bucket.size(); ++k) {
                            const Cell* pb = bucket[k];
                            double d2 = squaredDistance(queryCell, *pb);
                            if (d2 < bestDist2) {
                                bestDist2 = d2;
                                bestId = pb->id;
                            }
                        }
                    }
                }
            }
            // 如果本层没有任何格子被访问（即所有格子下界距离 >= bestDist2），则可提前结束
            if (!anyVisited) {
                break;
            }
        }
        if (bestId < 0) {
            return make_pair(-1, -1.0);
        }
        return make_pair(bestId, sqrt(bestDist2));
    }

    // 统计半径内 B 细胞数量
    int countBCellsWithinRadius(const Cell& queryCell, double radius) const {
        double R2 = radius * radius;
        int agx = coordToGridX(queryCell.x);
        int agy = coordToGridY(queryCell.y);
        int dr = static_cast<int>(ceil(radius / cellSize));
        int count = 0;
        for (int dx = -dr; dx <= dr; ++dx) {
            int gx = agx + dx;
            if (gx < 0 || gx >= gridWidth) {
                continue;
            }
            for (int dy = -dr; dy <= dr; ++dy) {
                int gy = agy + dy;
                if (gy < 0 || gy >= gridHeight) {
                    continue;
                }
                double boxDist2 = computeBoxMinDist2(queryCell, gx, gy);
                if (boxDist2 > R2) {
                    continue;
                }
                const vector<const Cell*>& bucket = gridArr[gx][gy];
                for (size_t k = 0; k < bucket.size(); ++k) {
                    const Cell* pb = bucket[k];
                    double d2 = squaredDistance(queryCell, *pb);
                    if (d2 <= R2) {
                        count++;
                    }
                }
            }
        }
        return count;
    }

    // 可选：打印网格统计信息
    void printGridStats() const {
        cout << "=== SpatialGridOptimized Statistics ===\n";
        cout << "Bounds X: [" << minX << ", " << maxX << "], "
                  << "Y: [" << minY << ", " << maxY << "]\n";
        cout << "cellSize: " << cellSize
                  << ", gridWidth: " << gridWidth
                  << ", gridHeight: " << gridHeight << "\n";
        size_t nonEmpty = 0;
        size_t totalCells = 0;
        size_t maxPer = 0;
        for (int i = 0; i < gridWidth; ++i) {
            for (int j = 0; j < gridHeight; ++j) {
                size_t sz = gridArr[i][j].size();
                if (sz > 0) {
                    nonEmpty++;
                    totalCells += sz;
                    if (sz > maxPer) {
                        maxPer = sz;
                    }
                }
            }
        }
        if (nonEmpty > 0) {
            cout << "Non-empty grids: " << nonEmpty
                      << ", avg cells/grid: " << (double)totalCells / nonEmpty
                      << ", max cells in a grid: " << maxPer << "\n";
        } else {
            cout << "No B cells inserted.\n";
        }
    }
};

// 网格搜索算法
vector<CellAnalysisResult> gridSearch(const vector<Cell>& A_cells, const vector<Cell>& B_cells, double radius = 10.0) {
    vector<CellAnalysisResult> results;
    
    if (B_cells.empty()) {
        // 如果没有B细胞，返回空结果
        for (const auto& A_cell : A_cells) {
            CellAnalysisResult result;
            result.cellid = A_cell.id;
            result.x = A_cell.x;
            result.y = A_cell.y;
            result.celltype = A_cell.type;
            result.radius = radius;
            result.nearest_B_id = -1;
            result.nearest_B_dist = -1.0;
            result.B_count_within_radius = 0;
            results.push_back(result);
        }
        return results;
    }
    
    // 确定合适的格子大小，通常设为搜索半径的一半到两倍之间
    double cellSize = radius * 0.6;  // 可以根据需要调整
    
    // 构建空间网格，只插入B细胞
    SpatialGridOptimized grid(B_cells, cellSize);
    
    // 对每个A细胞进行分析
    for (const auto& A_cell : A_cells) {
        CellAnalysisResult result;
        result.cellid = A_cell.id;
        result.x = A_cell.x;
        result.y = A_cell.y;
        result.celltype = A_cell.type;
        result.radius = radius;
        
        // 查找最近的B细胞
        pair<int, double> nearest = grid.findNearestB(A_cell);
        result.nearest_B_id = nearest.first;
        result.nearest_B_dist = nearest.second;
        
        // 计算半径内的B细胞数量
        result.B_count_within_radius = grid.countBCellsWithinRadius(A_cell, radius);
        
        results.push_back(result);
    }
    
    return results;
}
