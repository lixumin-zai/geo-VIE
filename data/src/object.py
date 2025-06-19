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
    CIRCLE = 4


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
    def __init__(self, ):
        """
        :param key: 角的唯一标识符，例如 "ABC"。
        :param p1: 角的一条边上的点。
        :param vertex: 角的顶点。
        :param p2: 角的另一条边上的点。
        :param label_text: 角的度数标签。如果为 "auto"，则自动计算并显示度数。
        :param radius: 角的弧线半径。如果为 None，则自动调整。
        :param style: 自定义样式。
        """
        super().__init__(style)
        self.type = Element.ANGLE
    
    def get_bounding_box(self) -> list[np.ndarray]:
        """
        获取元素及其标签组合后的整体边界框，可用于碰撞检测和自动排版。
        这里使用点列表保存实体边界，两个对象中的 最小的点跟点距离 判断是否碰撞
        
        :return: 点列表
        """
        return np.array([self.point])
        
    def _create_label(self) -> Mobject:
        line = Arc(self.start_point, self.end_point, color=BLACK, stroke_width=self.style["STROKE_WIDTH"])
        return line


class MyRightAngleSymbol(GeometricElement):
    """
    表示一个直角符号。
    """
    def __init__(self, key: str, p1: MyPoint, vertex: MyPoint, p2: MyPoint, size: float = None, style: dict = None):
        """
        :param key: 符号的唯一标识符。
        :param p1, vertex, p2: 定义直角的三个点。
        :param size: 符号的大小。如果为 None，则自动调整。
        :param style: 自定义样式。
        """
        super().__init__(style)
        self.key = key
        line1 = Line(vertex.position, p1.position)
        line2 = Line(vertex.position, p2.position)
        
        if size is None:
            # --- 更正点 ---
            len1 = np.linalg.norm(line1.get_vector())
            len2 = np.linalg.norm(line2.get_vector())
            size = min(len1, len2) * 0.25 # 自动调整大小
            
        self.mobject = RightAngle(line1, line2, length=size, color=self.style["SYMBOL_COLOR"])
        self.group = VGroup(self.mobject)

class MyArc(GeometricElement):
    """
    表示一段圆弧。
    """
    def __init__(self, key: str, center: MyPoint, radius: float, start_angle_deg: float, angle_deg: float, style: dict = None):
        """
        :param key: 弧的唯一标识符。
        :param center: 圆心 (MyPoint 对象)。
        :param radius: 半径。
        :param start_angle_deg: 起始角度（度）。
        :param angle_deg: 扫过的角度（度），正为逆时针，负为顺时针。
        :param style: 自定义样式。
        """
        super().__init__(style)
        self.key = key
        self.center = center
        
        self.mobject = Arc(
            radius=radius,
            start_angle=start_angle_deg * DEGREES,
            angle=angle_deg * DEGREES,
            arc_center=center.position,
            color=self.style["ARC_COLOR"],
            stroke_width=self.style["STROKE_WIDTH"]
        )
        self.group = VGroup(self.mobject)

class MyCircle(GeometricElement):
    """
    表示一个完整的圆。
    """
    def __init__(self, key: str, center: MyPoint, radius: float, style: dict = None):
        """
        :param key: 圆的唯一标识符。
        :param center: 圆心 (MyPoint 对象)。
        :param radius: 半径。
        :param style: 自定义样式。
        """
        super().__init__(style)
        self.key = key
        self.center = center

        self.mobject = Circle(
            radius=radius,
            arc_center=center.position,
            color=self.style["ARC_COLOR"],
            stroke_width=self.style["STROKE_WIDTH"]
        )
        self.group = VGroup(self.mobject)

class MyParallelMark(GeometricElement):
    """
    表示平行线符号（例如 > 或 >>）。
    """
    def __init__(self, key: str, line: MyLine, position_on_line: float = 0.5, size: float = 0.1, count: int = 1, style: dict = None):
        """
        :param key: 符号的唯一标识符。
        :param line: 要在其上放置符号的线段 (MyLine 对象)。
        :param position_on_line: 符号在线段上的位置 (0.0=起点, 0.5=中点, 1.0=终点)。
        :param size: 符号的大小。
        :param count: 符号箭头的数量 (例如 1 for >, 2 for >>)。
        :param style: 自定义样式。
        """
        super().__init__(style)
        self.key = key
        
        # 创建箭头符号
        symbol = VGroup()
        for i in range(count):
            arrow = Line(LEFT * size, RIGHT * size, stroke_width=self.style["STROKE_WIDTH"])
            arrow_tip = Line(ORIGIN, UP * size * 0.5 + LEFT * size * 0.5)
            arrow.add(VGroup(arrow_tip, arrow_tip.copy().rotate(PI/2, about_point=ORIGIN)).move_to(RIGHT*size))
            arrow.shift(UP * i * size * 0.8) # 如果有多个箭头，则错开排列
            symbol.add(arrow)
        
        # 旋转并移动到正确位置
        symbol.rotate(line.mobject.get_angle())
        symbol.move_to(line.mobject.point_from_proportion(position_on_line))
        symbol.set_color(self.style["SYMBOL_COLOR"])
        
        self.mobject = symbol
        self.group = VGroup(self.mobject)

class MyCoordinateSystem(GeometricElement):
    """
    表示一个二维笛卡尔坐标系。
    """
    def __init__(self, x_range=(-8, 8, 1), y_range=(-5, 5, 1), style: dict = None):
        """
        :param x_range: (xmin, xmax, xstep) for x-axis.
        :param y_range: (ymin, ymax, ystep) for y-axis.
        :param style: 自定义样式。
        """
        super().__init__(style)
        self.axes = Axes(
            x_range=x_range,
            y_range=y_range,
            x_length=x_range[1] - x_range[0],
            y_length=y_range[1] - y_range[0],
            axis_config={"color": self.style["LINE_COLOR"]}
        )
        self.mobject = self.axes
        self.group = VGroup(self.mobject)

    def plot(self, func, color=BLUE, key=None):
        """
        在坐标系上绘制函数图像。
        
        :param func: 要绘制的函数，例如 lambda x: x**2。
        :param color: 图像颜色。
        :param key: 图像的唯一标识符（可选）。
        :return: 一个包含函数图像的 GeometricElement 对象。
        """
        graph_mobject = self.axes.plot(func, color=color)
        graph_element = GeometricElement(self.style)
        graph_element.key = key
        graph_element.mobject = graph_mobject
        graph_element.group = VGroup(graph_mobject)
        return graph_element


# --- 关系与场景管理器 ---

class GeometryManager:
    """
    一个管理器，用于创建、存储和“绑定”所有几何元素。
    这是构建复杂几何图形的核心。
    """
    def __init__(self, scene):
        """
        :param scene: 当前的 Manim Scene 对象，用于添加 Mobjects。
        """
        self.scene = scene
        self.elements = {}  # 存储所有几何元素，键为元素的 key

    def add(self, element: GeometricElement):
        """
        向管理器和 Manim 场景中添加一个几何元素。
        
        :param element: 要添加的 GeometricElement 对象。
        """
        if element.key in self.elements:
            print(f"警告: 元素 key '{element.key}' 已存在，将被覆盖。")
        self.elements[element.key] = element
        self.scene.add(element.group) # 将元素的 VGroup 添加到场景
        return element

    def get(self, key: str) -> GeometricElement:
        """
        通过 key 获取一个已添加的元素。
        
        :param key: 元素的唯一标识符。
        :return: 对应的 GeometricElement 对象。
        """
        if key not in self.elements:
            raise KeyError(f"错误: 找不到 key 为 '{key}' 的元素。")
        return self.elements[key]

    # --- 便捷的创建方法 ---
    
    def add_point(self, key: str, position: np.ndarray, **kwargs):
        point = MyPoint(key, position, **kwargs)
        return self.add(point)

    def add_line(self, key: str, p1_key: str, p2_key: str, **kwargs):
        p1 = self.get(p1_key)
        p2 = self.get(p2_key)
        line = MyLine(key, p1, p2, **kwargs)
        return self.add(line)
        
    def add_angle(self, key: str, p1_key: str, vertex_key: str, p2_key: str, **kwargs):
        p1 = self.get(p1_key)
        vertex = self.get(vertex_key)
        p2 = self.get(p2_key)
        angle = MyAngle(key, p1, vertex, p2, **kwargs)
        return self.add(angle)

    def add_right_angle(self, key: str, p1_key: str, vertex_key: str, p2_key: str, **kwargs):
        p1 = self.get(p1_key)
        vertex = self.get(vertex_key)
        p2 = self.get(p2_key)
        right_angle = MyRightAngleSymbol(key, p1, vertex, p2, **kwargs)
        return self.add(right_angle)

    def add_parallel_mark(self, key: str, line_key: str, **kwargs):
        line = self.get(line_key)
        mark = MyParallelMark(key, line, **kwargs)
        return self.add(mark)
        
    def add_circle(self, key: str, center_key: str, radius: float, **kwargs):
        center = self.get(center_key)
        circle = MyCircle(key, center, radius, **kwargs)
        return self.add(circle)

# --- 使用示例 ---
# 要运行此示例，请将此文件保存为 basic_element.py，
# 然后在另一个文件中导入并运行 GeometryExampleScene。
# 命令行运行: manim -pql your_scene_file.py GeometryExampleScene

class GeometryExampleScene(Scene):
    def construct(self):
        # 1. 初始化几何管理器
        gm = GeometryManager(self)
        
        # 2. 添加点
        # 点的标签会自动放置在随机方向
        gm.add_point("A", LEFT * 2 + UP * 2, label_text="A")
        gm.add_point("B", RIGHT * 3 + UP * 2, label_text="B")
        gm.add_point("C", RIGHT * 3 + DOWN * 2, label_text="C")
        # 您也可以指定标签方向和使用 LaTeX
        gm.add_point("D", LEFT * 2 + DOWN * 2, label_text="D_1", label_direction=DOWN+LEFT)

        self.wait(0.5)

        # 3. 添加线（自动绑定到已创建的点）
        gm.add_line("AB", "A", "B", label_text="5")
        gm.add_line("BC", "B", "C", dashed=True) # 创建虚线
        gm.add_line("CD", "C", "D", label_text="5")
        gm.add_line("DA", "D", "A")
        gm.add_line("AC", "A", "C")
        
        self.wait(0.5)

        # 4. 添加角和符号
        # 自动计算并显示角度
        gm.add_angle("DAC", "D", "A", "C", label_text="auto") 
        # 添加直角符号
        gm.add_right_angle("ADC", "A", "D", "C")
        # 添加平行符号 (两个 >>)
        gm.add_parallel_mark("p1", "AB", count=2, position_on_line=0.4)
        gm.add_parallel_mark("p2", "CD", count=2, position_on_line=0.6)

        self.wait(0.5)
        
        # 5. 添加圆
        gm.add_point("O", ORIGIN, label_text="O")
        gm.add_circle("circ1", "O", 1.0)
        
        self.wait(2)


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
