# -*- coding: utf-8 -*-
# @Time    :   2024/04/04 15:34:39
# @Author  :   lixumin1030@gmail.com
# @FileName:   scene.py

from hmac import new
from manim import *
from objects import *
import random
import string
import json
import os
import traceback
import itertools
import numpy as np
import math
from typing import List, Dict, Tuple, Optional
from utils import *


class CollisionDetector:
    """碰撞检测器，用于检测几何元素之间的碰撞"""
    
    @staticmethod
    def min_distance_between_bboxes(bbox1: List[np.ndarray], bbox2: List[np.ndarray]) -> float:
        """计算两个边界框之间的最小距离"""
        min_dist = float('inf')
        for point1 in bbox1:
            for point2 in bbox2:
                dist = np.linalg.norm(point1 - point2)
                min_dist = min(min_dist, dist)
        return min_dist
    
    @staticmethod
    def check_collision(element1: GeometricElement, element2: GeometricElement, 
                       min_distance: float = 0.5) -> bool:
        """检查两个几何元素是否发生碰撞"""
        bbox1 = element1.get_bounding_box()
        bbox2 = element2.get_bounding_box()
        return CollisionDetector.min_distance_between_bboxes(bbox1, bbox2) < min_distance
    
    @staticmethod
    def find_safe_position(new_element: GeometricElement, existing_elements: List[GeometricElement],
                          search_radius: float = 3.0, min_distance: float = 0.5) -> Optional[np.ndarray]:
        """为新元素找到一个安全的位置，避免与现有元素碰撞"""
        for _ in range(100):  # 最多尝试100次
            # 在搜索半径内随机生成位置
            angle = random.uniform(0, 2 * np.pi)
            radius = random.uniform(0.5, search_radius)
            offset = np.array([radius * np.cos(angle), radius * np.sin(angle), 0])
            
            # 创建临时元素测试位置
            if hasattr(new_element, 'location'):
                test_location = new_element.location + offset
                new_element.location = test_location
                new_element.mobject = new_element._create_mobject()
            
            # 检查是否与现有元素碰撞
            collision_found = False
            for existing in existing_elements:
                if CollisionDetector.check_collision(new_element, existing, min_distance):
                    collision_found = True
                    break
            
            if not collision_found:
                return test_location if hasattr(new_element, 'location') else None
        
        return None

class Config:
    def __init__(self):
        self.show_angle_text_probability = 0.8
        self.draw_line_probability = 0.2
        self.draw_angle_probability = 0.9
        self.point_generation_radius = 8
        self.min_point_distance = 5
        self.max_points = 5
        self.data = None
        self.is_online_type = 0

class GeometricSceneGenerator(Scene):
    """几何场景生成器，使用新的几何元素架构"""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.geometry_scene = GeometryScene()
        self.collision_detector = CollisionDetector()
        
        # 标签池
        self.point_labels = self._generate_point_labels()
        self.line_labels = self._generate_line_labels()
        self.angle_labels = self._generate_angle_labels()
        
        # 样式配置
        self.style = {
            "COLOR": BLACK,
            "STROKE_WIDTH": random.randint(4, 9),
            "FILL_COLOR": WHITE,
            "FILL_OPACITY": 0,
            "FONT_SIZE": random.randint(50, 60),
            "TEXT_COLOR": BLACK
        }
        
        # 输出数据
        self.output_data = {
            "points": [],
            "lines": [],
            "angles": [],
            "circles": [],
            "relations": [],
            "text_boxes": []
        }
    
    def _generate_point_labels(self) -> List[str]:
        """生成点标签池"""
        point_labels = list(string.ascii_uppercase.replace("I", "").replace("X", ""))
        random.shuffle(point_labels)
        return point_labels
    
    def _generate_line_labels(self) -> List[str]:
        """生成线段标签池"""
        line_labels = ["a", "b", "c", "d", "k", "x", "y", "m", "n", "p"] * 3 + [""] * 20
        random.shuffle(line_labels)
        return line_labels
    
    def _generate_angle_labels(self) -> List[str]:
        """生成角度标签池"""
        angle_labels = ["1", "2", "3", "4", "5", "α", "β", "θ", "γ", "δ"] * 2 + [""] * 20
        random.shuffle(angle_labels)
        return angle_labels
    
    def generate_base_points(self):
        """使用泊松圆盘采样生成基础点"""
        radius = self.config.point_generation_radius
        r = self.config.min_point_distance
        k = 10
        
        # 泊松圆盘采样算法
        cell_size = r / np.sqrt(2)
        grid_width = int(2 * radius / cell_size) + 1
        grid_height = int(2 * radius / cell_size) + 1
        grid = [[None for _ in range(grid_height)] for _ in range(grid_width)]
        
        active_list = []
        points = []
        
        # 生成第一个点
        while True:
            first_point = np.array([random.uniform(-radius, radius), random.uniform(-radius, radius), 0])
            if np.linalg.norm(first_point[:2]) <= radius:
                points.append(first_point)
                active_list.append(first_point)
                break
        
        def get_grid_coords(point):
            return int((point[0] + radius) / cell_size), int((point[1] + radius) / cell_size)
        
        def in_neighborhood(point, points, r):
            for p in points:
                if np.linalg.norm(point - p) < r:
                    return True
            return False
        
        grid[get_grid_coords(first_point)[0]][get_grid_coords(first_point)[1]] = first_point
        
        while active_list:
            index = random.randint(0, len(active_list) - 1)
            point = active_list[index]
            found = False
            
            for _ in range(k):
                angle = random.uniform(0, 2 * np.pi)
                offset = np.array([np.cos(angle), np.sin(angle), 0]) * random.uniform(r, 2 * r)
                new_point = point + offset
                
                if np.linalg.norm(new_point[:2]) <= radius:
                    grid_coords = get_grid_coords(new_point)
                    if (0 <= grid_coords[0] < grid_width and 0 <= grid_coords[1] < grid_height and
                        grid[grid_coords[0]][grid_coords[1]] is None and 
                        not in_neighborhood(new_point, points, r)):
                        points.append(new_point)
                        active_list.append(new_point)
                        grid[grid_coords[0]][grid_coords[1]] = new_point
                        found = True
                        break
            
            if not found:
                active_list.pop(index)
        
        # 限制点的数量
        point_count = min(random.randint(4, self.config.max_points), len(points))
        selected_points = random.sample(points, point_count)
        
        # 创建MyPoint对象
        for i, point in enumerate(selected_points):
            if self.point_labels:
                my_point = MyPoint(point, self.style)
                self.geometry_scene.add_element(my_point)

    def add_intermediate_points(self):
        """在长线段上添加中间点"""
        new_points = []
        points = self.geometry_scene.get_elements_by_type(Element.POINT)
        point_combinations = list(itertools.combinations(points, 2))
        
        for point1, point2 in point_combinations:
            distance = np.linalg.norm(point1.point - point2.point)
            
            # 检查是否有其他点在这两点之间
            has_intermediate = False
            for other_point in points:
                if other_point in [point1, point2]:
                    continue
                # 简单的共线检测
                if self._are_collinear(point1.point, point2.point, other_point.point):
                    has_intermediate = True
                    break
            
            if not has_intermediate and distance > 7 and is_happen(0.3):
                # 添加一到两个中间点
                num_intermediate = 1 if distance < 10 else 2
                for i in range(num_intermediate):
                    if not self.point_labels:
                        break
                    
                    proportion = random.uniform(0.3, 0.7) if num_intermediate == 1 else (0.3 + i * 0.4)
                    intermediate_point = point1.point + proportion * (point2.point - point1.point)
                    
                    my_point = MyPoint(intermediate_point, self.style)
                    new_points.append(my_point)
                    self.geometry_scene.add_element(my_point)
        
        return new_points
    
    def generate_lines(self):
        """生成线段并建立关系"""
        lines = []
        points = self.geometry_scene.get_elements_by_type(Element.POINT)
        point_combinations = list(itertools.combinations(points, 2))
        random.shuffle(point_combinations)
        
        # 限制每个点的连接数
        point_connection_count = {point.id: 0 for point in points}
        max_connections = 4
        
        for point1, point2 in point_combinations:
            # 检查连接数限制
            if (point_connection_count[point1.id] >= max_connections or 
                point_connection_count[point2.id] >= max_connections):
                continue
            
            # 检查是否有其他点过于接近这条线
            line_valid = True
            for other_point in points:
                if other_point in [point1, point2]:
                    continue
                
                # 计算点到线的距离
                dist = point_to_line_distance(other_point.point, point1.point, point2.point)
                if dist <= 0.5:
                    if point1.point[0] < other_point.point[0] < point2.point[0] or point2.point[0] < other_point.point[0] < point1.point[0]:
                        line_valid = False
                        break
                
                if 0.5 < dist < 1:
                    line_valid = False
                    break
            
            if line_valid:
                line_label = self.line_labels.pop() if self.line_labels else ""
                line = MyLine(point1.point, point2.point, self.style)
                
                lines.append(line)
                self.geometry_scene.add_element(line)
                
                # 建立线段与点的构成关系
                relation = LinePointsRelation(line, point1, point2, f"线段{line.id}由点{point1.id}和{point2.id}构成")
                self.geometry_scene.add_relation(relation)
                
                point_connection_count[point1.id] += 1
                point_connection_count[point2.id] += 1
        
        return lines
    
    def find_intersections(self):
        """找到线段交点并创建新点"""
        # 获取所有线段并开始递归处理
        lines = self.geometry_scene.get_elements_by_type(Element.LINE)
        line_combinations = list(itertools.combinations(lines, 2))
        found_new_point = False
        
        point_elements = self.geometry_scene.get_elements_by_type(Element.POINT)
        new_points = []
        for line1, line2 in line_combinations:
            

            intersection = find_lines_intersection_point(line1, line2)
            if isinstance(intersection, np.ndarray):
                # 有交点
                too_close = False

                # 获取线段关系的点对象
                for r in self.geometry_scene.get_relations_by_type(RelationType.LINE_POINTS):
                    if line1 == r._line:
                        r1 = r
                        line1_start_point = r._start_point
                        line1_end_point = r._end_point
                    if line2 == r._line:
                        r2 = r
                        line2_start_point = r._start_point
                        line2_end_point = r._end_point

                for point in point_elements:
                    if np.linalg.norm(intersection - point.point) < 0.2:
                        too_close = True
                        break

                if not too_close:
                    # 创建交点
                    intersection_point = MyPoint(intersection, self.style)
                    new_points.append(intersection_point)
                    self.geometry_scene.add_element(intersection_point)
                    # 分割原有线段,创建新的线段
                    # 分割line1
                    new_line1a = MyLine(line1.start_point, intersection, self.style)
                    relation1a = LinePointsRelation(new_line1a, line1_start_point, intersection_point, f"线段{new_line1a.id}由点{line1_start_point.id}和{intersection_point.id}构成")

                    new_line1b = MyLine(intersection, line1.end_point, self.style)
                    relation1b = LinePointsRelation(new_line1b, intersection_point, line1_end_point, f"线段{new_line1b.id}由点{intersection_point.id}和{line1_end_point.id}构成")

                    # 分割line2
                    new_line2a = MyLine(line2.start_point, intersection, self.style)
                    relation2a = LinePointsRelation(new_line2a, line2_start_point, intersection_point, f"线段{new_line2a.id}由点{intersection_point.id}和{line2_start_point.id}构成")

                    new_line2b = MyLine(intersection, line2.end_point, self.style)
                    relation2b = LinePointsRelation(new_line2b, intersection_point, line2_end_point, f"线段{new_line2b.id}由点{intersection_point.id}和{line2_end_point.id}构成")

                    # 添加新线段到场景
                    new_lines = [new_line1a, new_line1b, new_line2a, new_line2b]
                    new_relations = [relation1a, relation1b, relation2a, relation2b]
                    for line, relation in zip(new_lines, new_relations):
                        self.geometry_scene.add_element(line)
                        # 建立线段与点的构成关系
                        self.geometry_scene.add_relation(relation)

                    # 移除原有线段
                    self.geometry_scene.remove_element(line1)
                    self.geometry_scene.remove_element(line2)
                    # 移除关系
                    self.geometry_scene.remove_relation(r1)
                    self.geometry_scene.remove_relation(r2)
                    found_new_point = True
                    break
            else:
                continue
            
        if found_new_point:
            self.find_intersections()
        
        return True

    def delete_lines(self):
        lines = self.geometry_scene.get_elements_by_type(Element.LINE)
        print(len(lines))
        for line in lines:
            if is_happen(0.2, "删除线段"):
                print(line.id)
                self.geometry_scene.remove_element(line)
                for r in self.geometry_scene.get_relations_by_type(RelationType.LINE_POINTS):
                    if line == r._line:
                        self.geometry_scene.remove_relation(r)
                            
    def generate_angles(self):
        """生成角度并建立关系"""
        angles = []

        points = self.geometry_scene.get_elements_by_type(Element.POINT)
        for point in points:
            # 通过关系系统找到以该点为顶点的所有线段
            connected_lines = self.geometry_scene.get_lines_through_point(point)
            
            if len(connected_lines) < 2:
                continue
            
            # 计算线段的角度并排序
            line_angles = []
            for line in connected_lines:
                # 确定线段的方向（从当前点出发）
                degree = line.angle_deg_on_point(point.point)
                line_angles.append((line, degree))
            
            line_angles.sort(key=lambda x: x[1])
            
            # 生成相邻线段之间的角
            for i in range(len(line_angles)):
                
                line1, angle1 = line_angles[i]
                line2, angle2 = line_angles[(i + 1) % len(line_angles)]
                
                # 如何线段的长度太短，则不生成
                if line1.length < 1 or line2.length < 1:
                    continue

                # 最后一个线段跟第一个线段
                if line2 == line_angles[0][0]:
                    angle2 = angle2 + 360

                # 计算角度差
                angle_diff = (angle2 - angle1) 
                
                # 过滤过大或过小的角
                if angle_diff > 175 or angle_diff < 20:
                    continue
                
                # 确定角的三个点 
                # start_point 为 line1 上半径为0.5的点
                start_point = self.geometry_scene.get_angle_point(line1, point.point)
                vertex_point = point
                end_point = self.geometry_scene.get_angle_point(line2, point.point)
                
                # print(start_point, vertex_point, end_point)

                # 检查是否是直角
                if 85 < angle_diff < 95:
                    angle_obj = MyRightAngle(start_point.point, vertex_point.point, end_point.point, self.style)
                else:
                    angle_obj = MyAngle(start_point.point, vertex_point.point, end_point.point, self.style)
                
                angles.append(angle_obj)
                self.geometry_scene.add_element(angle_obj)
                self.geometry_scene.add_relation(AngleLineRelation(angle_obj, line1, line2))
        
        return angles
    
    def add_text_labels_with_collision_detection(self):
        """添加文本标签并进行碰撞检测"""
        all_elements = self.geometry_scene.elements
        text_elements = []
        
        # 为点添加文本标签
        for element in all_elements:
            if element.type == Element.POINT and element.key:
                # 计算标签位置（避免与其他元素碰撞）
                label_position = self._find_safe_label_position(element, all_elements + text_elements)
                if label_position is not None:
                    text_label = MyText(element.key, label_position, self.style)
                    text_elements.append(text_label)
                    self.geometry_scene.add_element(text_label, f"{element.key}_label")
                    
                    # 建立点文本关系
                    relation = PointTextRelation(element, text_label, f"点{element.key}的标注")
                    self.geometry_scene.add_relation(relation)
        
        # 为线段添加长度标签
        for element in all_elements:
            if element.type == Element.LINE and element.value and is_happen(self.config.draw_line_probability):
                # 计算标签位置
                midpoint = (element.start_point + element.end_point) / 2
                # 计算垂直方向
                direction = element.end_point - element.start_point
                perpendicular = np.array([-direction[1], direction[0], 0])
                perpendicular = perpendicular / np.linalg.norm(perpendicular) * 0.5
                
                label_position = midpoint + perpendicular
                text_label = MyText(str(element.value), label_position, self.style)
                
                # 检查碰撞
                if not any(self.collision_detector.check_collision(text_label, existing) 
                          for existing in all_elements + text_elements):
                    text_elements.append(text_label)
                    self.geometry_scene.add_element(text_label, f"{element.key}_length_label")
                    
                    # 建立线段文本关系
                    relation = LineTextRelation(element, text_label, f"线段{element.key}的长度标注")
                    self.geometry_scene.add_relation(relation)
        
        # 为角添加角度标签
        for element in all_elements:
            if element.type in [Element.ANGLE, Element.RIGHTANGLE] and element.value and is_happen(self.config.show_angle_text_probability):
                # 计算角平分线上的位置
                label_position = self._calculate_angle_label_position(element)
                text_label = MyText(str(element.value), label_position, self.style)
                
                # 检查碰撞
                if not any(self.collision_detector.check_collision(text_label, existing) 
                          for existing in all_elements + text_elements):
                    text_elements.append(text_label)
                    self.geometry_scene.add_element(text_label, f"{element.key}_angle_label")
                    
                    # 建立角文本关系
                    relation = AngleTextRelation(element, text_label, f"角{element.key}的角度标注")
                    self.geometry_scene.add_relation(relation)
    
    def _find_safe_label_position(self, element: GeometricElement, existing_elements: List[GeometricElement]) -> Optional[np.ndarray]:
        """为元素找到安全的标签位置"""
        base_position = element.point if hasattr(element, 'point') else np.array([0, 0, 0])
        
        # 尝试8个方向
        directions = [
            np.array([1, 0, 0]), np.array([-1, 0, 0]), np.array([0, 1, 0]), np.array([0, -1, 0]),
            np.array([1, 1, 0]), np.array([-1, 1, 0]), np.array([1, -1, 0]), np.array([-1, -1, 0])
        ]
        
        for direction in directions:
            direction = direction / np.linalg.norm(direction)
            for distance in [0.4, 0.6, 0.8]:
                test_position = base_position + direction * distance
                test_text = MyText(element.key or "T", test_position, self.style)
                
                # 检查碰撞
                collision_found = False
                for existing in existing_elements:
                    if self.collision_detector.check_collision(test_text, existing, 0.3):
                        collision_found = True
                        break
                
                if not collision_found:
                    return test_position
        
        return None
    
    def _calculate_angle_label_position(self, angle_element) -> np.ndarray:
        """计算角度标签的位置"""
        # 简化实现：返回顶点附近的位置
        if hasattr(angle_element, 'vertex_point'):
            return angle_element.vertex_point + np.array([0.3, 0.3, 0])
        return np.array([0, 0, 0])
    
    def render_scene(self):
        """渲染场景"""
        # 获取需要渲染的Manim对象并添加到场景
        colors_hex = [
    '#e68c2d', '#34e6b1', '#a534e6', '#e6d92d', '#3d2de6', '#62e634', '#e62d85', '#2de6e0', '#e63434', '#6be62d',
    '#2d6fe6', '#e62dae', '#b1e634', '#3434e6', '#e69d2d', '#2de68e', '#d92de6', '#e0e62d', '#2d55e6', '#82e634',
    '#e62d68', '#2de6d3', '#e6452d', '#55e62d', '#2d85e6', '#e62dbb', '#c8e634', '#3445e6', '#e6b12d', '#2de671',
    '#e62de0', '#d3e62d', '#2d45e6', '#99e634', '#e62d51', '#2de6c8', '#e65c2d', '#3fe62d', '#2daee6', '#e62dce',
    '#e0e634', '#345ce6', '#e6c82d', '#2de655', '#e62d99', '#bce62d', '#2d3fe6', '#e6712d', '#34e6c8', '#aee62d',
    '#45e634', '#e62d45', '#68e62d', '#2dbbe6', '#e62d2e', '#8ee634', '#342de6', '#e6b1e6', '#2d99e6', '#2de671',
    '#e68c41', '#34e69d', '#b12de6', '#e6d941', '#3d41e6', '#62e634', '#e64185', '#41e6e0', '#e63434', '#6b41e6',
    '#416fe6', '#e641ae', '#b1e634', '#3434e6', '#e69d41', '#41e68e', '#d941e6', '#e0e641', '#4155e6', '#82e634',
    '#e64168', '#41e6d3', '#e64541', '#55e641', '#4185e6', '#e641bb', '#c8e634', '#3445e6', '#e6b141', '#41e671',
    '#e641e0', '#d3e641', '#4145e6', '#99e634', '#e64151', '#41e6c8', '#e65c41', '#3fe641', '#41aee6', '#e641ce'
]*2

        for idx, obj in enumerate(self.geometry_scene.elements):
            self.add(obj._create_mobject(color=colors_hex[idx]))
            # self.add(Text(f"{obj.id}", color=BLACK, font_size=30).move_to(obj.show_id_point))
            # 在对象出现后，暂停 0.5 秒
            # self.wait(0.05)
        
        for relation in self.geometry_scene.relations:
            print(relation)

        print(len(self.geometry_scene.get_elements_by_type(Element.POINT)))
        print(len(self.geometry_scene.get_elements_by_type(Element.LINE)))
        # 收集输出数据
        self._collect_output_data()
    
    def _collect_output_data(self):
        """收集输出数据"""
        # 收集点数据
        for element in self.geometry_scene.get_elements_by_type(Element.POINT):
            if element.id:
                self.output_data["points"].append({
                    "id": element.id,
                    "position": element.point.tolist()
                })
        
        # 收集线段数据
        for element in self.geometry_scene.get_elements_by_type(Element.LINE):
            if element.id:
                self.output_data["lines"].append({
                    "id": element.id,
                    "length": element.length,
                    "value": element.value
                })
        
        # 收集角度数据
        for element in self.geometry_scene.get_elements_by_type(Element.ANGLE):
            if element.id:
                self.output_data["angles"].append({
                    "id": element.id,
                    "degree": element.angle_deg,
                    "value": element.value
                })
        
        # 收集关系数据
        for relation in self.geometry_scene.relations:
            self.output_data["relations"].append({
                "type": relation.relation_type.value,
                "description": relation.description,
                "elements": [elem.id for elem in relation.elements if elem.id]
            })
    
    # 辅助方法
    def _are_collinear(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, tolerance: float = 0.1) -> bool:
        """检查三点在三维空间中是否共线"""
        # 确保向量是三维的，如果原始点是二维的，可以补零
        if p1.shape[0] < 3:
            # 这是一个示例处理，实际中你可能需要根据情况调整
            p1 = np.append(p1, [0] * (3 - p1.shape[0]))
            p2 = np.append(p2, [0] * (3 - p2.shape[0]))
            p3 = np.append(p3, [0] * (3 - p3.shape[0]))

        v1 = p2 - p1
        v2 = p3 - p1
        
        # 计算三维向量的叉积，结果是一个三维向量
        cross_product_vector = np.cross(v1, v2)
        
        # 如果点共线，叉积向量的模（长度）应该接近于 0
        # np.linalg.norm 计算向量的模
        return np.linalg.norm(cross_product_vector) < tolerance
    
    def _line_intersection(self, line1: MyLine, line2: MyLine) -> Optional[np.ndarray]:
        """计算两条线段的交点"""
        p1, p2 = line1.start_point[:2], line1.end_point[:2]
        p3, p4 = line2.start_point[:2], line2.end_point[:2]
        
        denom = (p1[0] - p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (p3[0] - p4[0])
        if abs(denom) < 1e-10:
            return None
        
        t = ((p1[0] - p3[0]) * (p3[1] - p4[1]) - (p1[1] - p3[1]) * (p3[0] - p4[0])) / denom
        u = -((p1[0] - p2[0]) * (p1[1] - p3[1]) - (p1[1] - p2[1]) * (p1[0] - p3[0])) / denom
        
        if 0 <= t <= 1 and 0 <= u <= 1:
            intersection = p1 + t * (p2 - p1)
            return np.array([intersection[0], intersection[1], 0])
        
        return None
    
    def _point_on_segment(self, point: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray, tolerance: float = 0.1) -> bool:
        """检查点是否在线段上"""
        # 检查点是否在线段的边界框内
        min_x, max_x = min(seg_start[0], seg_end[0]), max(seg_start[0], seg_end[0])
        min_y, max_y = min(seg_start[1], seg_end[1]), max(seg_start[1], seg_end[1])
        
        return (min_x - tolerance <= point[0] <= max_x + tolerance and 
                min_y - tolerance <= point[1] <= max_y + tolerance)
    
    def construct(self):
        """构建几何场景的主要流程"""
        max_attempts = 10
        
        for attempt in range(max_attempts):
            try:
                print(f"尝试生成场景 {attempt + 1}/{max_attempts}")
                
                # 重置场景
                self.geometry_scene = GeometryScene()
                self.output_data = {
                    "points": [],
                    "lines": [],
                    "angles": [],
                    "circles": [],
                    "relations": [],
                    "text_boxes": []
                }
                
                # 1. 生成基础点
                self.generate_base_points()
                if len(self.geometry_scene.get_elements_by_type(Element.POINT)) < 2:
                    continue
                
                # 2. 添加中间点
                self.add_intermediate_points()

                # 3. 生成线段
                self.generate_lines()
                
                # 4. 找到交点
                is_break = self.find_intersections()
                
                # 5. 删除一些线段，是样例更多样
                self.delete_lines()

                # 5. 生成角度
                self.generate_angles()
                
                
                # # 8. 验证场景有效性
                # if self._validate_scene():
                #     # 9. 渲染场景
                #     self.render_scene()
                #     print("场景生成成功！")
                #     break
                # else:
                #     print("场景验证失败，重新生成...")

                self.render_scene()
                break
                    
            except Exception as e:
                print(f"生成场景时出错: {e}")
                traceback.print_exc()
                continue
        else:
            print("无法生成有效场景")
    
    def _validate_scene(self) -> bool:
        """验证场景的有效性"""
        points = self.geometry_scene.get_elements_by_type(Element.POINT)
        lines = self.geometry_scene.get_elements_by_type(Element.LINE)
        
        # 检查点的数量
        if len(points) < 2:
            return False
        
        # 检查线段的数量
        if len(lines) < 1:
            return False
        
        # 检查点之间的最小距离
        for i, point1 in enumerate(points):
            for point2 in points[i+1:]:
                if np.linalg.norm(point1.point - point2.point) < 0.7:
                    return False
        
        return True

# 主程序入口
if __name__ == "__main__":
    config.frame_height = 25.0
    config.frame_width = 25.0
    
    with tempconfig({
        "background_color": WHITE,
        "pixel_height": 1400,
        "pixel_width": 1400
    }):
        os.makedirs("/app/data/images/", exist_ok=True)
        os.makedirs("/app/data/infos/", exist_ok=True)
        with open(f"/app/data/1.json", "w", encoding="utf-8") as f:
            config.output_file = f"/app/data/1"
            try:
                scene_config = Config()
                scene = GeometricSceneGenerator(scene_config)
                scene.render()
                
                # 保存输出数据
                with open("/app/data/scene_data.json", "w", encoding="utf-8") as f:
                    json.dump(scene.output_data, f, ensure_ascii=False, indent=2)
                    
            except Exception as e:
                print(f"渲染失败: {e}")
                traceback.print_exc()
    
    # /root/project/geo-vie/data/data/1.png