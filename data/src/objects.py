# -*- coding: utf-8 -*-
# @Time    :   2025/06/17 10:09:16
# @Author  :   lixumin1030@gmail.com
# @FileName:   geo-object.py

from enum import Enum
from manim import *
import random
from typing import List, Union, Optional, Dict, Any
import numpy as np
import math

# 默认样式配置
DEFAULT_STYLE = {
    "COLOR": BLACK,
    "STROKE_WIDTH": 2,
    "FILL_COLOR": WHITE,
    "FILL_OPACITY": 0,
    "FONT_SIZE": 24,
    "TEXT_COLOR": BLACK
}


class GeometricElement:
    """
    所有几何元素的基类，用于统一接口和管理通用属性。
    """
    _instance_counter = 0
    
    def __init__(self, style: dict = None):
        """
        初始化基类。
        
        :param style: 一个字典，用于覆盖 DEFAULT_STYLE 中的默认样式。
        """
        GeometricElement._instance_counter += 1
        self.id = GeometricElement._instance_counter

        # 合并默认样式和用户自定义样式
        self.style = DEFAULT_STYLE.copy() 
        if style:
            self.style.update(style)
        
        self.key = None           # 元素的唯一标识符，如 "A", "AB"
        self.mobject = None       # 存储核心的 Manim 可视化对象 (e.g., Dot, Line)
        self.label = None         # 存储标签对象 (e.g., Text, MathTex)
        self.type = None          # 元素类型
        self.value = None         # 元素的数值（长度、角度值等）

    def get_bounding_box(self) -> List[np.ndarray]:
        """
        获取元素及其标签组合后的整体边界框，可用于碰撞检测和自动排版。
        这里使用点列表保存实体边界，两个对象中的 最小的点跟点距离 判断是否碰撞
        
        :return: 点列表
        """
        raise NotImplementedError("子类必须实现此方法")

    def _create_mobject(self) -> Mobject:
        """
        创建Manim对象的通用函数。
        
        :return: 创建好的 Manim 对象。
        """
        raise NotImplementedError("子类必须实现此方法")

    def set_key(self, key: str):
        """设置元素的唯一标识符"""
        self.key = key
        return self

    def set_value(self, value: Any):
        """设置元素的数值"""
        self.value = value
        return self


class Element(Enum):
    TEXT = 0
    POINT = 1
    LINE = 2
    ANGLE = 3
    RIGHTANGLE = 4
    ARC = 5
    CIRCLE = 6


class MyPoint(GeometricElement):
    """
    表示一个点
    """
    def __init__(self, point: np.ndarray, style: dict = None):
        super().__init__(style)
        self.type = Element.POINT
        self.point = point
        self.mobject = self._create_mobject()

    def get_bounding_box(self) -> List[np.ndarray]:
        """
        获取点的边界框
        
        :return: 包含单个点的列表
        """
        return [self.point]

    def _create_mobject(self) -> Mobject:
        """创建点的可视化对象"""
        dot = Dot(self.point, color=self.style["COLOR"], radius=0.05)
        return dot


class MyLine(GeometricElement):
    """
    表示一条线段（或虚线），由两个点定义。
    """
    def __init__(self, start_point: np.ndarray, end_point: np.ndarray, style: dict = None):
        super().__init__(style)
        self.type = Element.LINE
        self.start_point = start_point
        self.end_point = end_point
        self.mobject = self._create_mobject()
        # 计算线段长度
        self.value = np.linalg.norm(end_point - start_point)

    @property
    def vector(self) -> np.ndarray:
        return self.mobject.get_vector()
    
    @property
    def unit_vector(self) -> np.ndarray:
        return self.mobject.get_unit_vector()

    @property
    def angle_deg(self) -> float:
        return self.mobject.get_angle() * 180 / PI
    
    @property
    def length(self) -> float:
        """获取线段长度"""
        return self.value
    
    def get_bounding_box(self) -> List[np.ndarray]:
        """
        获取线段的边界框
        
        :return: 线段上的点列表
        """
        num_points = 100
        alpha_values = np.linspace(0, 1, num_points)
        bounding_box = [self.mobject.point_from_proportion(alpha) for alpha in alpha_values]
        return bounding_box
        
    def _create_mobject(self) -> Mobject:
        """创建线段的可视化对象"""
        line = Line(self.start_point, self.end_point, 
                   color=self.style["COLOR"], 
                   stroke_width=self.style["STROKE_WIDTH"])
        return line


class MyAngle(GeometricElement):
    """
    表示一个角，由三点（两边端点和公共顶点）定义。
    """
    def __init__(self, start_point: np.ndarray, vertex_point: np.ndarray, end_point: np.ndarray, style: dict = None):
        """
        极坐标下 start_point 的角度一定要小于 end_point（逆时针）， 绘出的角度才是从 start_point ->  end_point
        """
        super().__init__(style)
        self.type = Element.ANGLE
        self.start_point = start_point
        self.vertex_point = vertex_point
        self.end_point = end_point
        self.mobject = self._create_mobject()
        # 计算角度值
        self.value = self._calculate_angle()

    def _calculate_angle(self) -> float:
        """计算角度值（弧度）"""
        vec1 = self.start_point - self.vertex_point
        vec2 = self.end_point - self.vertex_point
        
        # 计算两向量夹角
        cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        # 防止数值误差导致的域错误
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        return angle_rad
    
    @property
    def angle_deg(self) -> float:
        """获取角度值（度）"""
        return self.value * 180 / np.pi
    
    @property
    def angle_rad(self) -> float:
        """获取角度值（弧度）"""
        return self.value
    
    def get_bounding_box(self) -> List[np.ndarray]:
        """
        获取角的边界框
        
        :return: 角弧线上的点列表
        """
        num_points = 100
        alpha_values = np.linspace(0, 1, num_points)
        bounding_box = [self.mobject.point_from_proportion(alpha) for alpha in alpha_values]
        return bounding_box
        
    def _create_mobject(self) -> Mobject:
        """创建角的可视化对象"""
        angle = Angle(Line(self.vertex_point, self.start_point), 
                     Line(self.vertex_point, self.end_point),
                     radius=0.5, 
                     color=self.style["COLOR"],
                     stroke_width=self.style["STROKE_WIDTH"])
        return angle


class MyRightAngle(GeometricElement):
    """
    表示一个直角符号。
    """
    def __init__(self, start_point: np.ndarray, vertex_point: np.ndarray, end_point: np.ndarray, style: dict = None):
        super().__init__(style)
        self.type = Element.RIGHTANGLE
        self.start_point = start_point
        self.vertex_point = vertex_point
        self.end_point = end_point
        self.mobject = self._create_mobject()
        # 直角固定为90度
        self.value = np.pi / 2
    
    @property
    def angle_deg(self) -> float:
        """获取角度值（度）"""
        return 90.0
    
    @property
    def angle_rad(self) -> float:
        """获取角度值（弧度）"""
        return self.value
    
    def get_bounding_box(self) -> List[np.ndarray]:
        """
        获取直角符号的边界框
        
        :return: 直角符号的点列表
        """
        num_points = 100
        alpha_values = np.linspace(0, 1, num_points)
        bounding_box = [self.mobject.point_from_proportion(alpha) for alpha in alpha_values]
        return bounding_box
        
    def _create_mobject(self) -> Mobject:
        """创建直角符号的可视化对象"""
        angle = RightAngle(Line(self.vertex_point, self.start_point), 
                          Line(self.vertex_point, self.end_point),
                          length=0.2, 
                          color=self.style["COLOR"],
                          stroke_width=self.style["STROKE_WIDTH"])
        return angle


class MyArc(GeometricElement):
    """
    表示一段圆弧。
    """
    def __init__(self, start_point: np.ndarray, vertex_point: np.ndarray, end_point: np.ndarray, style: dict = None):
        super().__init__(style)
        self.type = Element.ARC
        self.start_point = start_point
        self.vertex_point = vertex_point
        self.end_point = end_point
        self.mobject = self._create_mobject()
    
    def get_bounding_box(self) -> List[np.ndarray]:
        """
        获取圆弧的边界框
        
        :return: 圆弧上的点列表
        """
        num_points = 100
        alpha_values = np.linspace(0, 1, num_points)
        bounding_box = [self.mobject.point_from_proportion(alpha) for alpha in alpha_values]
        return bounding_box
        
    def _create_mobject(self) -> Mobject:
        """创建圆弧的可视化对象"""
        angle = Angle(Line(self.vertex_point, self.start_point), 
                     Line(self.vertex_point, self.end_point),
                     radius=1, 
                     color=self.style["COLOR"],
                     stroke_width=self.style["STROKE_WIDTH"])
        return angle


class MyCircle(GeometricElement):
    """
    表示一个完整的圆。
    """
    def __init__(self, center_point: np.ndarray, radius: float = 1.0, style: dict = None):
        super().__init__(style)
        self.type = Element.CIRCLE
        self.center_point = center_point
        self.radius = radius
        self.value = radius  # 圆的值为半径
        self.mobject = self._create_mobject()
    
    def get_bounding_box(self) -> List[np.ndarray]:
        """
        获取圆的边界框
        
        :return: 圆周上的点列表
        """
        num_points = 200
        alpha_values = np.linspace(0, 1, num_points)
        bounding_box = [self.mobject.point_from_proportion(alpha) for alpha in alpha_values]
        return bounding_box
        
    def _create_mobject(self) -> Mobject:
        """创建圆的可视化对象"""
        circle = Circle(
            radius=self.radius, 
            color=self.style["COLOR"],
            stroke_width=self.style["STROKE_WIDTH"],
            fill_color=self.style["FILL_COLOR"],
            fill_opacity=self.style["FILL_OPACITY"]
        ).move_to(self.center_point)
        return circle

    def point_on_circle(self, angle_rad: float) -> np.ndarray:
        """获取圆上指定角度的点"""
        x = self.center_point[0] + self.radius * np.cos(angle_rad)
        y = self.center_point[1] + self.radius * np.sin(angle_rad)
        z = self.center_point[2]
        return np.array([x, y, z])


class MyText(GeometricElement):
    """
    表示文本标签
    """
    def __init__(self, text: str, location: np.ndarray, style: dict = None):
        super().__init__(style)
        self.type = Element.TEXT
        self.text = text
        self.location = location
        self.value = text  # 文本的值就是文本内容
        self.mobject = self._create_mobject()

    def get_bounding_box(self) -> List[np.ndarray]:
        """
        获取文本的边界框
        
        :return: 文本边界框的四个角点
        """
        # 获取文本对象的边界框
        bbox = self.mobject.get_bounding_box()
        return [bbox[0], bbox[1]]  # 返回左下角和右上角
    
    def _create_mobject(self) -> Mobject:
        """创建文本的可视化对象"""
        text_obj = Text(
            self.text,
            font_size=self.style["FONT_SIZE"],
            color=self.style["TEXT_COLOR"]
        ).move_to(self.location)
        return text_obj


# --- 关系与场景管理器 ---
class RelationType(Enum):
    """关系类型枚举"""
    # 基础标注关系
    POINT_TEXT = "point_text"           # 点与文本的标注关系
    LINE_TEXT = "line_text"             # 线段与文本的标注关系（表示长度）
    ANGLE_TEXT = "angle_text"           # 角与文本的标注关系（表示角度值）
    CIRCLE_TEXT = "circle_text"         # 圆与文本的标注关系（表示半径）
    
    # 构成关系
    LINE_POINTS = "line_points"         # 线段由两点构成的关系
    ANGLE_POINTS = "angle_points"       # 角由三点构成的关系
    CIRCLE_CENTER = "circle_center"     # 圆心关系
    
    # 位置关系
    POINT_ON_LINE = "point_on_line"     # 点在线段上
    POINT_ON_CIRCLE = "point_on_circle" # 点在圆上
    POINT_IN_CIRCLE = "point_in_circle" # 点在圆内
    LINE_TANGENT_CIRCLE = "line_tangent_circle"  # 直线与圆相切
    
    # 几何关系
    VERTICAL = "vertical"               # 垂直关系
    PARALLEL = "parallel"               # 平行关系
    EQUAL_LENGTH = "equal_length"       # 等长关系
    EQUAL_ANGLE = "equal_angle"         # 等角关系
    INTERSECTION = "intersection"       # 交点关系
    
    # 复合关系
    NESTED_RELATION = "nested_relation" # 嵌套关系


class ElementRelation:
    """元素关系的基类"""
    _instance_counter = 0
    
    def __init__(self, relation_type: RelationType, elements: List[GeometricElement], description: str = ""):
        ElementRelation._instance_counter += 1
        self.id = ElementRelation._instance_counter
        self.relation_type = relation_type
        self.elements = elements
        self.description = description
        self.properties = {}  # 存储关系的额外属性
        self.nested_relations: List['ElementRelation'] = []  # 嵌套的子关系

    def add_property(self, key: str, value):
        """添加关系属性"""
        self.properties[key] = value
        return self

    def get_property(self, key: str, default=None):
        """获取关系属性"""
        return self.properties.get(key, default)

    def add_nested_relation(self, relation: 'ElementRelation'):
        """添加嵌套关系"""
        self.nested_relations.append(relation)
        return self

    def get_nested_relations(self, relation_type: RelationType = None) -> List['ElementRelation']:
        """获取嵌套关系"""
        if relation_type is None:
            return self.nested_relations
        return [r for r in self.nested_relations if r.relation_type == relation_type]

    def validate(self) -> bool:
        """验证关系是否有效"""
        return len(self.elements) > 0

    def __repr__(self) -> str:
        element_ids = [str(elem.id) for elem in self.elements]
        result = f"{self.relation_type.value}({', '.join(element_ids)}): {self.description}"
        if self.nested_relations:
            result += f" [嵌套关系: {len(self.nested_relations)}个]"
        return result


class PointTextRelation(ElementRelation):
    """点与文本的标注关系"""
    def __init__(self, point: MyPoint, text: MyText, description: str = ""):
        super().__init__(RelationType.POINT_TEXT, [point, text], description)
        self._point = point
        self._text = text

    @property
    def text(self) -> str:
        return self._text.text

    @property
    def point(self) -> np.ndarray:
        return self._point.point
    
    def __repr__(self) -> str:
        coords = self._point.point
        return f"点{self.text}({coords[0]:.2f}, {coords[1]:.2f})"


class LineTextRelation(ElementRelation):
    """线段与文本的标注关系（表示长度）"""
    def __init__(self, line: MyLine, text: MyText, description: str = ""):
        super().__init__(RelationType.LINE_TEXT, [line, text], description)
        self._line = line
        self._text = text

    @property
    def length_text(self) -> str:
        """获取长度文本"""
        return self._text.text

    @property
    def actual_length(self) -> float:
        """获取实际长度"""
        return self._line.length

    def __repr__(self) -> str:
        return f"线段{self._line.key}长度: {self.length_text} (实际: {self.actual_length:.2f})"


class AngleTextRelation(ElementRelation):
    """角与文本的标注关系（表示角度值）"""
    def __init__(self, angle: Union[MyAngle, MyRightAngle], text: MyText, description: str = ""):
        super().__init__(RelationType.ANGLE_TEXT, [angle, text], description)
        self._angle = angle
        self._text = text

    @property
    def angle_text(self) -> str:
        """获取角度文本"""
        return self._text.text

    @property
    def actual_angle_deg(self) -> float:
        """获取实际角度（度）"""
        return self._angle.angle_deg

    def __repr__(self) -> str:
        return f"角{self._angle.key}度数: {self.angle_text} (实际: {self.actual_angle_deg:.1f}°)"


class CircleTextRelation(ElementRelation):
    """圆与文本的标注关系（表示半径）"""
    def __init__(self, circle: MyCircle, text: MyText, description: str = ""):
        super().__init__(RelationType.CIRCLE_TEXT, [circle, text], description)
        self._circle = circle
        self._text = text

    @property
    def radius_text(self) -> str:
        """获取半径文本"""
        return self._text.text

    @property
    def actual_radius(self) -> float:
        """获取实际半径"""
        return self._circle.radius

    def __repr__(self) -> str:
        return f"圆{self._circle.key}半径: {self.radius_text} (实际: {self.actual_radius:.2f})"


class LinePointsRelation(ElementRelation):
    """线段由两点构成的关系"""
    def __init__(self, line: MyLine, start_point: MyPoint, end_point: MyPoint, description: str = ""):
        super().__init__(RelationType.LINE_POINTS, [line, start_point, end_point], description)
        self._line = line
        self._start_point = start_point
        self._end_point = end_point

    @property
    def line(self) -> MyLine:
        return self._line

    @property
    def start_point(self) -> MyPoint:
        return self._start_point

    @property
    def end_point(self) -> MyPoint:
        return self._end_point

    def __repr__(self) -> str:
        return f"线段{self._line.key}由点{self._start_point.key}和点{self._end_point.key}构成"


class CircleCenterRelation(ElementRelation):
    """圆心关系"""
    def __init__(self, circle: MyCircle, center_point: MyPoint, description: str = ""):
        super().__init__(RelationType.CIRCLE_CENTER, [circle, center_point], description)
        self._circle = circle
        self._center_point = center_point

    @property
    def circle(self) -> MyCircle:
        return self._circle

    @property
    def center_point(self) -> MyPoint:
        return self._center_point

    def __repr__(self) -> str:
        return f"圆{self._circle.key}的圆心是点{self._center_point.key}"


class PointOnCircleRelation(ElementRelation):
    """点在圆上的关系"""
    def __init__(self, point: MyPoint, circle: MyCircle, description: str = ""):
        super().__init__(RelationType.POINT_ON_CIRCLE, [point, circle], description)
        self._point = point
        self._circle = circle

    @property
    def point(self) -> MyPoint:
        return self._point

    @property
    def circle(self) -> MyCircle:
        return self._circle

    def is_on_circle(self, tolerance: float = 1e-6) -> bool:
        """检查点是否真的在圆上"""
        distance = np.linalg.norm(self._point.point - self._circle.center_point)
        return abs(distance - self._circle.radius) < tolerance

    def __repr__(self) -> str:
        return f"点{self._point.key}在圆{self._circle.key}上"


class ParallelRelation(ElementRelation):
    """平行关系"""
    def __init__(self, line1: MyLine, line2: MyLine, description: str = ""):
        super().__init__(RelationType.PARALLEL, [line1, line2], description)
        self._line1 = line1
        self._line2 = line2

    def __repr__(self) -> str:
        return f"线段{self._line1.key} ∥ 线段{self._line2.key}"


class VerticalRelation(ElementRelation):
    """垂直关系"""
    def __init__(self, line1: MyLine, line2: MyLine, description: str = ""):
        super().__init__(RelationType.VERTICAL, [line1, line2], description)
        self._line1 = line1
        self._line2 = line2

    def __repr__(self) -> str:
        return f"线段{self._line1.key} ⊥ 线段{self._line2.key}"


class EqualLengthRelation(ElementRelation):
    """等长关系"""
    def __init__(self, lines: List[MyLine], description: str = ""):
        super().__init__(RelationType.EQUAL_LENGTH, lines, description)
        self._lines = lines

    def __repr__(self) -> str:
        line_keys = [line.key for line in self._lines if line.key]
        return f"等长: {' = '.join(line_keys)}"


class NestedRelation(ElementRelation):
    """嵌套关系 - 用于表示复杂的组合关系"""
    def __init__(self, primary_relation: ElementRelation, description: str = ""):
        super().__init__(RelationType.NESTED_RELATION, primary_relation.elements, description)
        self.primary_relation = primary_relation

    def __repr__(self) -> str:
        result = f"嵌套关系: {self.primary_relation}"
        if self.nested_relations:
            result += "\n  包含子关系:"
            for i, rel in enumerate(self.nested_relations, 1):
                result += f"\n    {i}. {rel}"
        return result


class GeometryScene:
    """几何场景管理器"""
    def __init__(self):
        self.elements: List[GeometricElement] = []
        self.relations: List[ElementRelation] = []
        self.element_map = {}  # key -> element 的映射

    def add_element(self, element: GeometricElement, key: str = None) -> GeometricElement:
        """添加几何元素"""
        self.elements.append(element)
        if key:
            element.set_key(key)
            self.element_map[key] = element
        return element

    def add_relation(self, relation: ElementRelation) -> ElementRelation:
        """添加元素关系"""
        if relation.validate():
            self.relations.append(relation)
        return relation

    def get_element(self, key: str) -> Optional[GeometricElement]:
        """根据key获取元素"""
        return self.element_map.get(key)

    def get_relations_by_type(self, relation_type: RelationType) -> List[ElementRelation]:
        """根据类型获取关系"""
        return [r for r in self.relations if r.relation_type == relation_type]

    def get_mobjects(self) -> List[Mobject]:
        """获取所有可视化对象"""
        mobjects = []
        for element in self.elements:
            if element.mobject:
                mobjects.append(element.mobject)
            if element.label:
                mobjects.append(element.label)
        return mobjects

    def create_complex_relation_example(self):
        """创建复杂嵌套关系的示例"""
        # 创建一个圆和圆心
        center = MyPoint(np.array([0, 0, 0]))
        circle = MyCircle(center.point, 2.0)
        self.add_element(center, "O")
        self.add_element(circle, "circle_O")
        
        # 创建圆心关系
        center_relation = CircleCenterRelation(circle, center, "圆心关系")
        self.add_relation(center_relation)
        
        # 在圆上创建几个点
        point_A = MyPoint(circle.point_on_circle(0))  # 0度
        point_B = MyPoint(circle.point_on_circle(np.pi/2))  # 90度
        point_C = MyPoint(circle.point_on_circle(np.pi))  # 180度
        
        self.add_element(point_A, "A")
        self.add_element(point_B, "B")
        self.add_element(point_C, "C")
        
        # 创建点在圆上的关系
        point_on_circle_A = PointOnCircleRelation(point_A, circle)
        point_on_circle_B = PointOnCircleRelation(point_B, circle)
        point_on_circle_C = PointOnCircleRelation(point_C, circle)
        
        self.add_relation(point_on_circle_A)
        self.add_relation(point_on_circle_B)
        self.add_relation(point_on_circle_C)
        
        # 创建半径线段
        radius_OA = MyLine(center.point, point_A.point)
        radius_OB = MyLine(center.point, point_B.point)
        
        self.add_element(radius_OA, "OA")
        self.add_element(radius_OB, "OB")
        
        # 创建线段构成关系
        line_relation_OA = LinePointsRelation(radius_OA, center, point_A)
        line_relation_OB = LinePointsRelation(radius_OB, center, point_B)
        
        self.add_relation(line_relation_OA)
        self.add_relation(line_relation_OB)
        
        # 创建等长关系（半径相等）
        equal_radius = EqualLengthRelation([radius_OA, radius_OB], "半径相等")
        self.add_relation(equal_radius)
        
        # 创建长度标注
        radius_text = MyText("r", np.array([1, 0.2, 0]))
        self.add_element(radius_text, "radius_label")
        
        radius_text_relation = LineTextRelation(radius_OA, radius_text, "半径标注")
        self.add_relation(radius_text_relation)
        
        # 创建嵌套关系示例
        # 圆心关系可以嵌套点在圆上的关系
        center_relation.add_nested_relation(point_on_circle_A)
        center_relation.add_nested_relation(point_on_circle_B)
        center_relation.add_nested_relation(point_on_circle_C)
        
        # 等长关系可以嵌套线段构成关系
        equal_radius.add_nested_relation(line_relation_OA)
        equal_radius.add_nested_relation(line_relation_OB)
        
        return center_relation, equal_radius

    def __repr__(self) -> str:
        result = f"几何场景 - 元素数量: {len(self.elements)}, 关系数量: {len(self.relations)}\n"
        result += "元素:\n"
        for element in self.elements:
            result += f"  {element.key or element.id}: {element.type.name}"
            if hasattr(element, 'value') and element.value is not None:
                result += f" (值: {element.value})"
            result += "\n"
        result += "关系:\n"
        for relation in self.relations:
            result += f"  {relation}\n"
            # 显示嵌套关系
            for nested in relation.nested_relations:
                result += f"    └─ {nested}\n"
        return result


class GeometryExampleScene(Scene):
    """几何示例场景"""
    def construct(self):
        # 创建几何场景管理器
        geo_scene = GeometryScene()
        
        # 创建复杂关系示例
        center_relation, equal_radius = geo_scene.create_complex_relation_example()
        
        # 获取所有可视化对象并添加到场景
        mobjects = geo_scene.get_mobjects()
        
        # 分组动画显示
        # 1. 首先显示圆心
        center = geo_scene.get_element("O")
        self.play(Create(center.mobject))
        
        # 2. 显示圆
        circle = geo_scene.get_element("circle_O")
        self.play(Create(circle.mobject))
        
        # 3. 显示圆上的点
        points = [geo_scene.get_element("A").mobject, 
                 geo_scene.get_element("B").mobject, 
                 geo_scene.get_element("C").mobject]
        self.play(*[Create(point) for point in points])
        
        # 4. 显示半径
        radii = [geo_scene.get_element("OA").mobject, 
                geo_scene.get_element("OB").mobject]
        self.play(*[Create(radius) for radius in radii])
        
        # 5. 显示半径标注
        radius_label = geo_scene.get_element("radius_label")
        self.play(Write(radius_label.mobject))
        
        # 添加标题
        title = Text("复杂几何关系示例", font_size=36, color=BLACK).to_edge(UP)
        self.play(Write(title))
        
        # 打印场景信息到控制台
        print("\n=== 复杂几何场景信息 ===")
        print(geo_scene)
        
        # 等待一段时间
        self.wait(3)


# 示例使用
if __name__ == "__main__":
    config.frame_height = 25.0
    config.frame_width = 25.0
    with tempconfig(
        {
            # "quality": "low_quality",
            "background_color": WHITE,
            "pixel_height": 1400,
            "pixel_width": 1400
        }):
        scene = GeometryExampleScene()
        scene.render()