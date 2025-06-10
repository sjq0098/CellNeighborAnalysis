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
using namespace std;

struct Cell {
    int id;
    double x;
    double y;
    char type;
};

// 结果结构体，用于存储分析结果
struct CellAnalysisResult {
    int cellid;
    double x;
    double y;
    char celltype;
    int nearest_B_id;
    double nearest_B_dist;
    int B_count_within_radius;
    double radius;
};


double squaredDistance(const Cell& cell1, const Cell& cell2) {
    double dx = cell1.x - cell2.x;
    double dy = cell1.y - cell2.y;
    return dx * dx + dy * dy;
}

double squaredDistance(double x1, double y1, double x2, double y2) {
    double dx = x1 - x2;
    double dy = y1 - y2;
    return dx * dx + dy * dy;
}


double calculateDistance(const Cell& cell1, const Cell& cell2) {
    double dx = cell1.x - cell2.x;
    double dy = cell1.y - cell2.y;
    return sqrt(dx * dx + dy * dy);
}