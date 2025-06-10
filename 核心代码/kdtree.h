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
    // 边界框，用于范围查询剪枝
    double minX, maxX, minY, maxY;
};

class kdtree{
public:
    kdtree(const vector<Cell>& cells) {
        vector<Cell> pts = cells;
        root = build(pts, 0);
    }
    ~kdtree() {
        destroy(root);
    }

    pair<int, double> nearestNeighbor(const Cell& query) {
        best_id = -1;
        best_dist2 = numeric_limits<double>::infinity();
        searchNearest(root, query, 0);
        if (best_id < 0) {
            // 没有 B 细胞
            return make_pair(-1, -1.0);
        }
        double best_dist = sqrt(best_dist2);
        return make_pair(best_id, best_dist);
    }

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

    kdnode* build(vector<Cell>& points, int depth) {
        if (points.empty()) {
            return NULL;
        }
        
        int axis = depth % 2; // 交替分割，保证平衡

        // 找中位索引
        size_t mid = points.size() / 2;
        
        if (axis == 0) {
            nth_element(points.begin(), points.begin() + mid, points.end(),
                        [](const Cell& a, const Cell& b) {
                            return a.x < b.x;
                        });
        } else {
            nth_element(points.begin(), points.begin() + mid, points.end(),
                        [](const Cell& a, const Cell& b) {
                            return a.y < b.y;
                        });
        }
        // 创建节点
        kdnode* node = new kdnode();
        node->cell = points[mid];
        // 分割左右子集
        vector<Cell> leftPts;
        vector<Cell> rightPts;
        for (size_t i = 0; i < points.size(); ++i) {
            if (i == mid) {
                continue;
            }
            if (axis == 0) {
                if (points[i].x < points[mid].x) {
                    leftPts.push_back(points[i]);
                } else {
                    rightPts.push_back(points[i]);
                }
            } else {
                if (points[i].y < points[mid].y) {
                    leftPts.push_back(points[i]);
                } else {
                    rightPts.push_back(points[i]);
                }
            }
        }
        // 递归构建
        node->left = build(leftPts, depth + 1);
        node->right = build(rightPts, depth + 1);
        // 计算并保存子树边界框
        computeBounds(node);
        return node;
    }


    //计算边界便于剪枝
    void computeBounds(kdnode* node) {
        if (node == NULL) {
            return;
        }
        // 初始化为节点自身坐标
        node->minX = node->cell.x;
        node->maxX = node->cell.x;
        node->minY = node->cell.y;
        node->maxY = node->cell.y;
        // 如果有左子树，先递归计算左，再更新边界
        if (node->left != NULL) {
            computeBounds(node->left);
            if (node->left->minX < node->minX) {
                node->minX = node->left->minX;
            }
            if (node->left->maxX > node->maxX) {
                node->maxX = node->left->maxX;
            }
            if (node->left->minY < node->minY) {
                node->minY = node->left->minY;
            }
            if (node->left->maxY > node->maxY) {
                node->maxY = node->left->maxY;
            }
        }
        // 如果有右子树，递归计算右，再更新边界
        if (node->right != NULL) {
            computeBounds(node->right);
            if (node->right->minX < node->minX) {
                node->minX = node->right->minX;
            }
            if (node->right->maxX > node->maxX) {
                node->maxX = node->right->maxX;
            }
            if (node->right->minY < node->minY) {
                node->minY = node->right->minY;
            }
            if (node->right->maxY > node->maxY) {
                node->maxY = node->right->maxY;
            }
        }
    }

    void destroy(kdnode* node) {
        if (node == NULL) {
            return;
        }
        destroy(node->left);
        destroy(node->right);
        delete node;
    }
    
    void searchNearest(kdnode* node, const Cell& query, int depth) {
        if (node == NULL) {
            return;
        }
        // 如果是 B 类型，检查距离
        if (node->cell.type == 'B') {
            double d2 = squaredDistance(node->cell, query);
            if (d2 < best_dist2) {
                best_dist2 = d2;
                best_id = node->cell.id;
            }
        }
        // 选择分割维度
        int axis = depth % 2;
        double delta = 0.0;
        if (axis == 0) {
            delta = query.x - node->cell.x;
        } else {
            delta = query.y - node->cell.y;
        }
        // 明确划分 near 分支和 far 分支
        kdnode* nearChild = NULL;
        kdnode* farChild = NULL;
        if (delta < 0.0) {
            nearChild = node->left;
            farChild = node->right;
        } else {
            nearChild = node->right;
            farChild = node->left;
        }
        // 先搜索 near 分支
        if (nearChild != NULL) {
            searchNearest(nearChild, query, depth + 1);
        }
        if (farChild != NULL) {
            double delta2 = delta * delta;
            if (delta2 < best_dist2) {
                searchNearest(farChild, query, depth + 1);
            }
        }
    }

    void searchRange(kdnode* node, const Cell& query, double r2, int& count) {
        if (node == NULL) {
            return;
        }
        // 先用子树边界框与 query 比较，若最小距离大于 r2，则剪枝
        double dx = 0.0;
        double dy = 0.0;
        if (query.x < node->minX) {
            dx = node->minX - query.x;
        } else if (query.x > node->maxX) {
            dx = query.x - node->maxX;
        }
        if (query.y < node->minY) {
            dy = node->minY - query.y;
        } else if (query.y > node->maxY) {
            dy = query.y - node->maxY;
        }
        double boxDist2 = dx * dx + dy * dy;
        if (boxDist2 > r2) {
            return;
        }
        // 当前节点若为 B 类型，检查其到 query 的距离
        if (node->cell.type == 'B') {
            double d2 = squaredDistance(node->cell, query);
            if (d2 <= r2) {
                count += 1;
            }
        }
        // 递归左右子树
        if (node->left != NULL) {
            searchRange(node->left, query, r2, count);
        }
        if (node->right != NULL) {
            searchRange(node->right, query, r2, count);
        }
    }
};

// KD树优化算法
vector<CellAnalysisResult> kdTreeSearch(const vector<Cell>& A_cells,
                                        const vector<Cell>& B_cells,
                                        double radius = 10.0) {
    vector<CellAnalysisResult> results;
    if (B_cells.empty()) {
        return results;
    }
    // 构造 KD-树，只插入 B 细胞
    kdtree tree(B_cells);
    results.reserve(A_cells.size());
    for (size_t i = 0; i < A_cells.size(); ++i) {
        const Cell& A_cell = A_cells[i];
        CellAnalysisResult result;
        result.cellid = A_cell.id;
        result.x = A_cell.x;
        result.y = A_cell.y;
        result.celltype = A_cell.type;
        result.radius = radius;
        pair<int, double> nn = tree.nearestNeighbor(A_cell);
        result.nearest_B_id = nn.first;
        result.nearest_B_dist = nn.second;
        result.B_count_within_radius = tree.countWithinRadius(A_cell, radius);
        results.push_back(result);
    }
    return results;
}
