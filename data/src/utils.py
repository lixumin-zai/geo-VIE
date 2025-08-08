# -*- coding: utf-8 -*-
# @Time    :   2025/06/18 17:28:49
# @Author  :   lixumin1030@gmail.com
# @FileName:   utils.py

import numpy as np
from objects import *

def find_lines_intersection_point(line1: MyLine, line2: MyLine) -> Optional[np.ndarray]:
    """
    计算两条线段的交点。
    
    :param line1: 第一条线段
    :param line2: 第二条线段
    :return: 如果相交，返回交点的np.ndarray；否则返回None。
    """
    # 提取点 (我们只关心 x, y 坐标)
    p1 = line1.start_point[:2]
    p2 = line1.end_point[:2]
    p3 = line2.start_point[:2]
    p4 = line2.end_point[:2]

    # 计算方向向量
    v1 = p2 - p1
    v2 = p4 - p3

    # 计算分母：v1和v2的二维叉积
    # (v1_x * v2_y) - (v1_y * v2_x)
    denominator = np.cross(v1, v2)

    # 检查线段是否平行或共线
    # 使用一个小的容差(epsilon)来处理浮点数精度问题
    if abs(denominator) < 1e-9:
        # 平行或共线，此简化实现中我们认为它们不相交
        # (严格的实现需要检查共线且重叠的情况)
        return None

    # 计算参数 t 和 u 的分子
    dp = p3 - p1
    t_numerator = np.cross(dp, v2)
    u_numerator = np.cross(dp, v1)

    # 计算参数 t 和 u
    t = t_numerator / denominator
    u = u_numerator / denominator

    # 检查交点是否在两条线段内部
    # 0 <= t <= 1 且 0 <= u <= 1
    if 0 <= t <= 1 and 0 <= u <= 1:
        # 计算交点
        intersection_point = p1 + t * v1
        intersection_point = np.concatenate((intersection_point, np.array([0])))
        return intersection_point
    # 交点在延长线上，线段不相交
    return None


def point_to_line_distance(point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> float:
    """计算点到线段的距离"""
    line_vec = line_end - line_start
    point_vec = point - line_start
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        return np.linalg.norm(point_vec)
    
    line_unitvec = line_vec / line_len
    proj_length = np.dot(point_vec, line_unitvec)
    proj = line_start + proj_length * line_unitvec
    return np.linalg.norm(point - proj)

def is_happen(happen_probability, description=""):
    return np.random.choice([True, False], p=[happen_probability, 1-happen_probability])
