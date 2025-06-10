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



struct kdnode {
    Cell cell;
    kdnode* left = nullptr;
    kdnode* right = nullptr;
    // 不再保存边界框
};

class kdtree {
public:
    kdtree(const vector<Cell>& cells) {
        vector<Cell> pts = cells;
        root = build(pts, 0);
    }
    ~kdtree() {
        destroy(root);
    }

    // 最近邻查询接口保持不变
    pair<int, double> nearestNeighbor(const Cell& query) {
        best_id = -1;
        best_dist2 = numeric_limits<double>::infinity();
        searchNearest(root, query, 0);
        if (best_id < 0) {
            return make_pair(-1, -1.0);
        }
        return make_pair(best_id, sqrt(best_dist2));
    }

    // 范围计数接口保持不变
    int countWithinRadius(const Cell& query, double radius) {
        double r2 = radius * radius;
        int count = 0;
        searchRange(root, query, r2, count);
        return count;
    }

private:
    kdnode* root;
    int best_id;
    double best_dist2;

    // 递归构建：按 axis 交替分割
    kdnode* build(vector<Cell>& points, int depth) {
        if (points.empty()) {
            return nullptr;
        }
        int axis = depth % 2; // 0: x, 1: y
        size_t mid = points.size() / 2;
        if (axis == 0) {
            nth_element(points.begin(), points.begin() + mid, points.end(),
                        [](const Cell& a, const Cell& b){ return a.x < b.x; });
        } else {
            nth_element(points.begin(), points.begin() + mid, points.end(),
                        [](const Cell& a, const Cell& b){ return a.y < b.y; });
        }
        kdnode* node = new kdnode();
        node->cell = points[mid];
        // 左右子集
        vector<Cell> leftPts;
        vector<Cell> rightPts;
        leftPts.reserve(mid);
        rightPts.reserve(points.size() - mid - 1);
        for (size_t i = 0; i < points.size(); ++i) {
            if (i == mid) continue;
            if (axis == 0) {
                if (points[i].x < points[mid].x) leftPts.push_back(points[i]);
                else rightPts.push_back(points[i]);
            } else {
                if (points[i].y < points[mid].y) leftPts.push_back(points[i]);
                else rightPts.push_back(points[i]);
            }
        }
        node->left = build(leftPts, depth + 1);
        node->right = build(rightPts, depth + 1);
        return node;
    }

    void destroy(kdnode* node) {
        if (!node) return;
        destroy(node->left);
        destroy(node->right);
        delete node;
    }

    // 计算两点平方距离
    inline static double squaredDistance(const Cell& a, const Cell& b) {
        double dx = a.x - b.x;
        double dy = a.y - b.y;
        return dx*dx + dy*dy;
    }

    // 最近邻搜索：仅基于轴平面剪枝，不用子树边界框
    void searchNearest(kdnode* node, const Cell& query, int depth) {
        if (!node) return;
        // 如果节点类型为 'B'，检查距离
        if (node->cell.type == 'B') {
            double d2 = squaredDistance(node->cell, query);
            if (d2 < best_dist2) {
                best_dist2 = d2;
                best_id = node->cell.id;
            }
        }
        int axis = depth % 2;
        double delta = (axis == 0 ? query.x - node->cell.x : query.y - node->cell.y);
        kdnode* nearChild = (delta < 0 ? node->left : node->right);
        kdnode* farChild  = (delta < 0 ? node->right : node->left);
        // 先探索 nearer 子树
        if (nearChild) {
            searchNearest(nearChild, query, depth + 1);
        }
        // 是否需要探索 farther 子树：仅根据平面距离判断
        double delta2 = delta * delta;
        if (farChild && delta2 < best_dist2) {
            searchNearest(farChild, query, depth + 1);
        }
    }

    // 范围计数：递归遍历整棵树，不做剪枝
    void searchRange(kdnode* node, const Cell& query, double r2, int& count) {
        if (!node) return;
        if (node->cell.type == 'B') {
            double d2 = squaredDistance(node->cell, query);
            if (d2 <= r2) {
                ++count;
            }
        }
        // 继续遍历左右子树，无任何剪枝
        if (node->left)  searchRange(node->left, query, r2, count);
        if (node->right) searchRange(node->right, query, r2, count);
    }
};

// KD树朴素版搜索函数，接口与原版一致
vector<CellAnalysisResult> kdTreeSearchSimple(
    const vector<Cell>& A_cells,
    const vector<Cell>& B_cells,
    double radius = 10.0)
{
    vector<CellAnalysisResult> results;
    if (B_cells.empty()) {
        // 若无 B，返回 A 细胞结果但标记无邻居
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
    // 构造朴素 KD-Tree，只插入 B 细胞
    kdtree tree(B_cells);
    results.reserve(A_cells.size());
    for (const auto& A_cell : A_cells) {
        CellAnalysisResult result;
        result.cellid = A_cell.id;
        result.x = A_cell.x;
        result.y = A_cell.y;
        result.celltype = A_cell.type;
        result.radius = radius;
        auto nn = tree.nearestNeighbor(A_cell);
        result.nearest_B_id = nn.first;
        result.nearest_B_dist = nn.second;
        result.B_count_within_radius = tree.countWithinRadius(A_cell, radius);
        results.push_back(result);
    }
    return results;
}
