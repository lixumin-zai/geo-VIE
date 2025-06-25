# -*- coding: utf-8 -*-
# @Time    :   2025/06/17 10:09:16
# @Author  :   lixumin1030@gmail.com
# @FileName:   object.py

from enum import Enum
from manim import *
import random


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
        self.mobject = Mobject()  # 存储核心的 Manim 可视化对象 (e.g., Dot, Line)
        self.label = Mobject()    # 存储标签对象 (e.g., Text, MathTex)

    def get_bounding_box(self) -> list[np.ndarray]:
        """
        获取元素及其标签组合后的整体边界框，可用于碰撞检测和自动排版。
        这里使用点列表保存实体边界，两个对象中的 最小的点跟点距离 判断是否碰撞
        
        :return: 点列表
        """
        return ...

    def _create_label(self) -> Mobject:
        """
        一个通用的标签创建函数。
        
        :return: 创建好的 Manim 文本对象。
        """
        
        return None


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
        """

        """
        super().__init__(style)
        self.type = Element.POINT
        self.point = point
        self.mobject = self._create_label()


    def get_bounding_box(self) -> list[np.ndarray]:
        """
        获取元素及其标签组合后的整体边界框，可用于碰撞检测和自动排版。
        这里使用点列表保存实体边界，两个对象中的 最小的点跟点距离 判断是否碰撞
        
        :return: 点列表
        """
        return np.array([self.point])


    def _create_label(self) -> Mobject:
        dot = Dot(self.point, color=RED)
        return dot


class MyLine(GeometricElement):
    """
    表示一条线段（或虚线），由两个点定义。
    """
    def __init__(self, start_point: np.ndarray, end_point: np.ndarray):
        super().__init__(style)
        self.type = Element.LINE
        self.start_point = start_point
        self.end_point = end_point
        self.mobject = self._create_label()

    
    @property
    def vector(self) -> np.ndarray:
        return self.mobject.get_vector()
    
    @property
    def unit_vector(self) -> np.ndarray:
        return self.mobject.get_unit_vector()

    @property
    def angle_deg(self) -> float:
        return self.mobject.get_angle() * 180 / PI
    
    def get_bounding_box(self) -> list[np.ndarray]:
        """
        获取元素及其标签组合后的整体边界框，可用于碰撞检测和自动排版。
        这里使用点列表保存实体边界，两个对象中的 最小的点跟点距离 判断是否碰撞
        
        :return: 点列表
        """
        num_points = 100 # 定义 100 个点
        alpha_values = np.linspace(0, 1, num_points) # 100个点
        bounding_box = [self.mobject.point_from_proportion(alpha) for alpha in alpha_values]
        return bounding_box
        
    def _create_label(self) -> Mobject:
        line = Line(self.start_point, self.end_point, color=BLACK, stroke_width=self.style["STROKE_WIDTH"])
        return line

class MyAngle(GeometricElement):
    """
    表示一个角，由三点（两边端点和公共顶点）定义。
    """
    def __init__(self, start_point: np.ndarray, vertex_point: np.ndarray, end_point: np.ndarray):
        """
        极坐标下 start_point 的角度一定要小于 end_point（逆时针）， 绘出的角度才是从 start_point ->  end_point
        """
        super().__init__(style)
        self.type = Element.ANGLE
        self.start_point = start_point
        self.vertex_point = vertex_point
        self.end_point = end_point
        self.mobject = self._create_label()
    
    def get_bounding_box(self) -> list[np.ndarray]:
        """
        获取元素及其标签组合后的整体边界框，可用于碰撞检测和自动排版。
        这里使用点列表保存实体边界，两个对象中的 最小的点跟点距离 判断是否碰撞
        对象本身就是一个 VMobject（向量图形对象），它代表了那段弧线。因此，你可以使用 VMobject 的方法来获取弧上的点。
        最常用的方法是 point_from_proportion(alpha)，其中 alpha 是一个从 0 到 1 的值，代表了沿着弧线路径的比例位置。
        :return: 点列表
        """
        num_points = 100 # 定义 100 个点
        alpha_values = np.linspace(0, 1, num_points) # 100个点
        bounding_box = [self.mobject.point_from_proportion(alpha) for alpha in alpha_values]
        return bounding_box
        
    def _create_label(self) -> Mobject:
        angle = Angle(Line(self.vertex_point, self.start_point), Line(self.vertex_point, self.end_point),
            radius=0.5, 
            color=BLACK,
            stroke_width=1
        )
        return angle


class MyRightAngle(GeometricElement):
    """
    表示一个直角符号。
    """
    def __init__(self, start_point: np.ndarray, vertex_point: np.ndarray, end_point: np.ndarray):
        """
        极坐标下 start_point 的角度一定要小于 end_point（逆时针）， 绘出的角度才是从 start_point ->  end_point
        """
        super().__init__(style)
        self.type = Element.ANGLE
        self.start_point = start_point
        self.vertex_point = vertex_point
        self.end_point = end_point
        self.mobject = self._create_label()
    
    def get_bounding_box(self) -> list[np.ndarray]:
        """
        获取元素及其标签组合后的整体边界框，可用于碰撞检测和自动排版。
        这里使用点列表保存实体边界，两个对象中的 最小的点跟点距离 判断是否碰撞
        对象本身就是一个 VMobject（向量图形对象），它代表了那段弧线。因此，你可以使用 VMobject 的方法来获取弧上的点。
        最常用的方法是 point_from_proportion(alpha)，其中 alpha 是一个从 0 到 1 的值，代表了沿着弧线路径的比例位置。
        :return: 点列表
        """
        num_points = 100 # 定义 100 个点
        alpha_values = np.linspace(0, 1, num_points) # 100个点
        bounding_box = [self.mobject.point_from_proportion(alpha) for alpha in alpha_values]
        return bounding_box
        
    def _create_label(self) -> Mobject:
        angle = RightAngle(Line(self.vertex_point, self.start_point), Line(self.vertex_point, self.end_point),
            length=0.2, 
            color=BLACK,
            stroke_width=1
        )
        return angle

class MyArc(GeometricElement):
    """
    表示一段圆弧。
    """
    def __init__(self, start_point: np.ndarray, vertex_point: np.ndarray, end_point: np.ndarray):
        """
        极坐标下 start_point 的角度一定要小于 end_point（逆时针）， 绘出的角度才是从 start_point ->  end_point
        """
        super().__init__(style)
        self.type = Element.ARC
        self.start_point = start_point
        self.vertex_point = vertex_point
        self.end_point = end_point
        self.mobject = self._create_label()
    
    def get_bounding_box(self) -> list[np.ndarray]:
        """
        获取元素及其标签组合后的整体边界框，可用于碰撞检测和自动排版。
        这里使用点列表保存实体边界，两个对象中的 最小的点跟点距离 判断是否碰撞
        对象本身就是一个 VMobject（向量图形对象），它代表了那段弧线。因此，你可以使用 VMobject 的方法来获取弧上的点。
        最常用的方法是 point_from_proportion(alpha)，其中 alpha 是一个从 0 到 1 的值，代表了沿着弧线路径的比例位置。
        :return: 点列表
        """
        num_points = 100 # 定义 100 个点
        alpha_values = np.linspace(0, 1, num_points) # 100个点
        bounding_box = [self.mobject.point_from_proportion(alpha) for alpha in alpha_values]
        return bounding_box
        
    def _create_label(self) -> Mobject:
        angle = Angle(Line(self.vertex_point, self.start_point), Line(self.vertex_point, self.end_point),
            radius=1, 
            color=BLACK,
            stroke_width=1
        )
        return angle

class MyCircle(GeometricElement):
    """
    表示一个完整的圆。
    """
    def __init__(self, origin_point: np.ndarray):
        """
        极坐标下 start_point 的角度一定要小于 end_point（逆时针）， 绘出的角度才是从 start_point ->  end_point
        """
        super().__init__(style)
        self.type = Element.ANGLE
        self.origin_point = origin_point
        self.mobject = self._create_label()
    
    def get_bounding_box(self) -> list[np.ndarray]:
        """
        获取元素及其标签组合后的整体边界框，可用于碰撞检测和自动排版。
        这里使用点列表保存实体边界，两个对象中的 最小的点跟点距离 判断是否碰撞
        Circle.from_three_points
        :return: 点列表
        """
        num_points = 200 # 定义 100 个点
        alpha_values = np.linspace(0, 1, num_points) # 1000个点
        bounding_box = [self.mobject.point_from_proportion(alpha) for alpha in alpha_values]
        return bounding_box
        
    def _create_mobject(self) -> Mobject:
        circle = Circle(
            radius=self.radius, 
            color=self.style["COLOR"],
            stroke_width=self.style["STROKE_WIDTH"]
        ).move_to(self.origin_point)
        return circle


# --- 关系与场景管理器 ---
class Relation(Enum):
    TEXT = 0
    POINT = 1
    LINE = 2
    ANGLE = 3
    RIGHTANGLE = 4
    ARC = 5
    CIRCLE = 6


class ElementRelation:
    _instance_counter = 0
    def __init__(self):
        ElementRelation._instance_counter += 1
        self.id = ElementRelation._instance_counter

        self.description = []
    

class 



#  elements: list[GeometricElement]

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
