# -*- coding: utf-8 -*-
# @Time    :   2025/06/17 10:09:16
# @Author  :   lixumin1030@gmail.com
# @FileName:   object.py

from enum import Enum
from manim import *
import random
from typing import List, Union, Optional
import numpy as np

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

    @property
    def vector(self) -> np.ndarray:
        return self.mobject.get_vector()
    
    @property
    def unit_vector(self) -> np.ndarray:
        return self.mobject.get_unit_vector()

    @property
    def angle_deg(self) -> float:
        return self.mobject.get_angle() * 180 / PI
    
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


class MyText(GeometricElement):
    """
    表示文本标签
    """
    def __init__(self, text: str, location: np.ndarray, style: dict = None):
        super().__init__(style)
        self.type = Element.TEXT
        self.text = text
        self.location = location
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
    POINT_TEXT = "point_text"           # 点与文本的标注关系
    LINE_TEXT = "line_text"             # 线段与文本的标注关系
    ANGLE_TEXT = "angle_text"           # 角与文本的标注关系
    LINE_POINTS = "line_points"         # 线段由两点构成的关系
    ANGLE_POINTS = "angle_points"       # 角由三点构成的关系
    VERTICAL = "vertical"               # 垂直关系
    PARALLEL = "parallel"               # 平行关系
    CIRCLE_POINT = "circle_point"       # 圆心关系
    EQUAL_LENGTH = "equal_length"       # 等长关系
    EQUAL_ANGLE = "equal_angle"         # 等角关系
    TANGENT = "tangent"                 # 切线关系
    INTERSECTION = "intersection"       # 交点关系


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

    def add_property(self, key: str, value):
        """添加关系属性"""
        self.properties[key] = value
        return self

    def get_property(self, key: str, default=None):
        """获取关系属性"""
        return self.properties.get(key, default)

    def validate(self) -> bool:
        """验证关系是否有效"""
        return len(self.elements) > 0

    def __repr__(self) -> str:
        element_ids = [str(elem.id) for elem in self.elements]
        return f"{self.relation_type.value}({', '.join(element_ids)}): {self.description}"


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
        return f"线段由点{self._start_point.key}和点{self._end_point.key}构成"


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

    def __repr__(self) -> str:
        result = f"几何场景 - 元素数量: {len(self.elements)}, 关系数量: {len(self.relations)}\n"
        result += "元素:\n"
        for element in self.elements:
            result += f"  {element.key or element.id}: {element.type.name}\n"
        result += "关系:\n"
        for relation in self.relations:
            result += f"  {relation}\n"
        return result


class GeometryExampleScene(Scene):
    """几何示例场景"""
    def construct(self):
        # 创建几何场景管理器
        geo_scene = GeometryScene()
        
        # 创建一个等腰直角三角形的示例
        # 定义三个顶点
        point_A = MyPoint(np.array([-2, -1, 0]))
        point_B = MyPoint(np.array([2, -1, 0]))
        point_C = MyPoint(np.array([0, 2, 0]))
        
        # 添加点到场景
        geo_scene.add_element(point_A, "A")
        geo_scene.add_element(point_B, "B")
        geo_scene.add_element(point_C, "C")
        
        # 创建文本标签
        text_A = MyText("A", np.array([-2.3, -1.3, 0]))
        text_B = MyText("B", np.array([2.3, -1.3, 0]))
        text_C = MyText("C", np.array([0, 2.3, 0]))
        
        geo_scene.add_element(text_A, "label_A")
        geo_scene.add_element(text_B, "label_B")
        geo_scene.add_element(text_C, "label_C")
        
        # 创建三角形的三条边
        line_AB = MyLine(point_A.point, point_B.point)
        line_BC = MyLine(point_B.point, point_C.point)
        line_CA = MyLine(point_C.point, point_A.point)
        
        geo_scene.add_element(line_AB, "AB")
        geo_scene.add_element(line_BC, "BC")
        geo_scene.add_element(line_CA, "CA")
        
        # 创建角度标记
        angle_A = MyAngle(point_B.point, point_A.point, point_C.point, {"COLOR": BLUE})
        angle_B = MyAngle(point_C.point, point_B.point, point_A.point, {"COLOR": GREEN})
        angle_C = MyRightAngle(point_A.point, point_C.point, point_B.point, {"COLOR": RED})
        
        geo_scene.add_element(angle_A, "angle_A")
        geo_scene.add_element(angle_B, "angle_B")
        geo_scene.add_element(angle_C, "angle_C")
        
        # 添加一个外接圆
        # 计算三角形外心（对于直角三角形，外心是斜边中点）
        circumcenter = (point_A.point + point_B.point) / 2
        circumradius = np.linalg.norm(point_C.point - circumcenter)
        circle = MyCircle(circumcenter, circumradius, {"COLOR": PURPLE, "STROKE_WIDTH": 1})
        geo_scene.add_element(circle, "circumcircle")
        
        # 创建关系
        geo_scene.add_relation(PointTextRelation(point_A, text_A, "点A的标注"))
        geo_scene.add_relation(PointTextRelation(point_B, text_B, "点B的标注"))
        geo_scene.add_relation(PointTextRelation(point_C, text_C, "点C的标注"))
        
        geo_scene.add_relation(LinePointsRelation(line_AB, point_A, point_B, "线段AB"))
        geo_scene.add_relation(LinePointsRelation(line_BC, point_B, point_C, "线段BC"))
        geo_scene.add_relation(LinePointsRelation(line_CA, point_C, point_A, "线段CA"))
        
        # 添加等长关系（等腰直角三角形的两腰相等）
        geo_scene.add_relation(EqualLengthRelation([line_BC, line_CA], "BC = CA"))
        
        # 添加垂直关系
        geo_scene.add_relation(VerticalRelation(line_BC, line_CA, "BC ⊥ CA"))
        
        # 获取所有可视化对象并添加到场景
        mobjects = geo_scene.get_mobjects()
        
        # 分组动画显示
        # 1. 首先显示点
        points = [point_A.mobject, point_B.mobject, point_C.mobject]
        self.add(*points)
        
        # 2. 显示点的标签
        labels = [text_A.mobject, text_B.mobject, text_C.mobject]
        self.add(*labels)
        
        # 3. 绘制三角形的边
        lines = [line_AB.mobject, line_BC.mobject, line_CA.mobject]
        self.add(*lines)
        
        # 4. 显示角度标记
        angles = [angle_A.mobject, angle_B.mobject, angle_C.mobject]
        self.add(*angles)
        
        # 5. 最后显示外接圆
        self.add(circle.mobject)
        
        # 添加一些文字说明
        title = Text("等腰直角三角形及其外接圆", font_size=36, color=BLACK).to_edge(UP)
        self.add(title)
        
        # 添加性质说明
        properties = VGroup(
            Text("性质:", font_size=24, color=BLACK),
            Text("• BC = CA (等腰)", font_size=20, color=BLACK),
            Text("• ∠C = 90° (直角)", font_size=20, color=BLACK),
            Text("• ∠A = ∠B = 45°", font_size=20, color=BLACK),
            Text("• 外心在斜边中点", font_size=20, color=BLACK)
        ).arrange(DOWN, aligned_edge=LEFT).to_edge(LEFT).shift(DOWN * 2)
        
        self.add(properties)
        
        # 打印场景信息到控制台
        print("\n=== 几何场景信息 ===")
        print(geo_scene)
        
        # 等待一段时间
        # self.wait(3)
# /root/project/geo-vie/data/src/media/images/GeometryExampleScene_ManimCE_v0.19.0.png

# 示例使用
if __name__ == "__main__":
    config.frame_height = 10.0
    config.frame_width = 10.0
    with tempconfig(
        {
            # "quality": "low_quality",
            "background_color": WHITE,
            "pixel_height": 1400,
            "pixel_width": 1400
        }):
        scene = GeometryExampleScene()
        scene.render()