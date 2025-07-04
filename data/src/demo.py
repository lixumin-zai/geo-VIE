# -*- coding: utf-8 -*-
# @Time    :   2025/01/20 
# @Author  :   Unified Geometric Figure Demo
# @FileName:   geo-object-unified.py

from objects import *
import numpy as np
from manim import *

class UnifiedGeometricFigure(Scene):
    """统一的几何图形演示 - 所有元素构成一个完整图形"""
    
    def construct(self):
        # 创建几何场景管理器
        geo_scene = GeometryScene()
        
        # 创建一个综合的几何图形：包含圆、三角形、角度标注等
        self.create_unified_figure(geo_scene)
        
        # 动画展示整个图形
        self.animate_unified_figure(geo_scene)
        
        # 打印场景信息
        print("\n=== 统一几何图形场景 ===")
        print(geo_scene)
        
        self.wait(3)
    
    def create_unified_figure(self, geo_scene):
        """创建统一的几何图形"""
        
        # === 1. 创建主圆和圆心 ===
        center_O = MyPoint(np.array([0, 0, 0]), {"COLOR": BLACK})
        main_circle = MyCircle(center_O.point, 2.5, {
            "COLOR": BLUE, 
            "STROKE_WIDTH": 3
        })
        
        geo_scene.add_element(center_O, "O")
        geo_scene.add_element(main_circle, "circle_O")
        geo_scene.add_relation(CircleCenterRelation(main_circle, center_O, "主圆圆心"))
        
        # === 2. 在圆上创建关键点 ===
        # 创建正三角形的三个顶点
        angle_A = 0  # 0度
        angle_B = 2 * np.pi / 3  # 120度
        angle_C = 4 * np.pi / 3  # 240度
        
        point_A = MyPoint(main_circle.point_on_circle(angle_A), {"COLOR": RED})
        point_B = MyPoint(main_circle.point_on_circle(angle_B), {"COLOR": RED})
        point_C = MyPoint(main_circle.point_on_circle(angle_C), {"COLOR": RED})
        
        geo_scene.add_element(point_A, "A")
        geo_scene.add_element(point_B, "B")
        geo_scene.add_element(point_C, "C")
        
        # 创建点在圆上的关系
        geo_scene.add_relation(PointOnCircleRelation(point_A, main_circle, "点A在圆上"))
        geo_scene.add_relation(PointOnCircleRelation(point_B, main_circle, "点B在圆上"))
        geo_scene.add_relation(PointOnCircleRelation(point_C, main_circle, "点C在圆上"))
        
        # === 3. 创建三角形的边 ===
        line_AB = MyLine(point_A.point, point_B.point, {"COLOR": GREEN, "STROKE_WIDTH": 3})
        line_BC = MyLine(point_B.point, point_C.point, {"COLOR": GREEN, "STROKE_WIDTH": 3})
        line_CA = MyLine(point_C.point, point_A.point, {"COLOR": GREEN, "STROKE_WIDTH": 3})
        
        geo_scene.add_element(line_AB, "AB")
        geo_scene.add_element(line_BC, "BC")
        geo_scene.add_element(line_CA, "CA")
        
        # 创建线段构成关系
        geo_scene.add_relation(LinePointsRelation(line_AB, point_A, point_B, "线段AB"))
        geo_scene.add_relation(LinePointsRelation(line_BC, point_B, point_C, "线段BC"))
        geo_scene.add_relation(LinePointsRelation(line_CA, point_C, point_A, "线段CA"))
        
        # 创建等长关系（等边三角形）
        geo_scene.add_relation(EqualLengthRelation([line_AB, line_BC, line_CA], "等边三角形"))
        
        # === 4. 创建半径线段 ===
        radius_OA = MyLine(center_O.point, point_A.point, {"COLOR": PURPLE, "STROKE_WIDTH": 2})
        radius_OB = MyLine(center_O.point, point_B.point, {"COLOR": PURPLE, "STROKE_WIDTH": 2})
        radius_OC = MyLine(center_O.point, point_C.point, {"COLOR": PURPLE, "STROKE_WIDTH": 2})
        
        geo_scene.add_element(radius_OA, "OA")
        geo_scene.add_element(radius_OB, "OB")
        geo_scene.add_element(radius_OC, "OC")
        
        # 创建半径等长关系
        geo_scene.add_relation(EqualLengthRelation([radius_OA, radius_OB, radius_OC], "半径相等"))
        
        # === 5. 创建角度 ===
        # 三角形内角（每个60度）
        angle_BAC = MyAngle(point_B.point, point_A.point, point_C.point, {"COLOR": ORANGE})
        angle_ABC = MyAngle(point_A.point, point_B.point, point_C.point, {"COLOR": ORANGE})
        angle_BCA = MyAngle(point_B.point, point_C.point, point_A.point, {"COLOR": ORANGE})
        
        geo_scene.add_element(angle_BAC, "angle_A")
        geo_scene.add_element(angle_ABC, "angle_B")
        geo_scene.add_element(angle_BCA, "angle_C")
        
        # 圆心角（每个120度）
        angle_AOB = MyAngle(point_A.point, center_O.point, point_B.point, {"COLOR": TEAL})
        angle_BOC = MyAngle(point_B.point, center_O.point, point_C.point, {"COLOR": TEAL})
        angle_COA = MyAngle(point_C.point, center_O.point, point_A.point, {"COLOR": TEAL})
        
        geo_scene.add_element(angle_AOB, "angle_AOB")
        geo_scene.add_element(angle_BOC, "angle_BOC")
        geo_scene.add_element(angle_COA, "angle_COA")
        
        # === 6. 创建圆弧 ===
        # 在三角形外侧创建圆弧
        arc_AB = MyArc(point_A.point, center_O.point, point_B.point, {"COLOR": PINK})
        arc_BC = MyArc(point_B.point, center_O.point, point_C.point, {"COLOR": PINK})
        arc_CA = MyArc(point_C.point, center_O.point, point_A.point, {"COLOR": PINK})
        
        geo_scene.add_element(arc_AB, "arc_AB")
        geo_scene.add_element(arc_BC, "arc_BC")
        geo_scene.add_element(arc_CA, "arc_CA")
        
        # === 7. 创建内切圆 ===
        # 计算内切圆半径（对于边长为s的等边三角形，内切圆半径 r = s/(2*sqrt(3))）
        side_length = line_AB.length
        incircle_radius = side_length / (2 * np.sqrt(3))
        
        incircle = MyCircle(center_O.point, incircle_radius, {
            "COLOR": YELLOW,
            "STROKE_WIDTH": 2,
            "FILL_COLOR": YELLOW,
            "FILL_OPACITY": 0.3
        })
        
        geo_scene.add_element(incircle, "incircle")
        
        # === 8. 添加直角标记 ===
        # 在半径与切点处添加直角标记（理论上的切点）
        # 为简化，我们在一个半径上添加直角标记
        mid_point_AB = MyPoint((point_A.point + point_B.point) / 2)
        geo_scene.add_element(mid_point_AB, "M_AB")
        
        # 从圆心到边中点的垂线
        perpendicular = MyLine(center_O.point, mid_point_AB.point, {"COLOR": DARK_BLUE, "STROKE_WIDTH": 2})
        geo_scene.add_element(perpendicular, "OM_AB")
        
        # 添加直角标记
        right_angle = MyRightAngle(point_A.point, mid_point_AB.point, center_O.point, {"COLOR": RED})
        geo_scene.add_element(right_angle, "right_angle")
        
        # === 9. 添加文本标注 ===
        # 点标签
        label_O = MyText("O", center_O.point + np.array([-0.2, -0.2, 0]), {"FONT_SIZE": 20, "TEXT_COLOR": BLACK})
        label_A = MyText("A", point_A.point + np.array([0.2, 0.2, 0]), {"FONT_SIZE": 20, "TEXT_COLOR": RED})
        label_B = MyText("B", point_B.point + np.array([-0.2, 0.2, 0]), {"FONT_SIZE": 20, "TEXT_COLOR": RED})
        label_C = MyText("C", point_C.point + np.array([0, -0.3, 0]), {"FONT_SIZE": 20, "TEXT_COLOR": RED})
        
        geo_scene.add_element(label_O, "label_O")
        geo_scene.add_element(label_A, "label_A")
        geo_scene.add_element(label_B, "label_B")
        geo_scene.add_element(label_C, "label_C")
        
        # 长度标注
        side_label = MyText(f"边长: {side_length:.1f}", np.array([0, -3.5, 0]), {"FONT_SIZE": 16, "TEXT_COLOR": GREEN})
        radius_label = MyText(f"半径: {main_circle.radius}", np.array([0, 3.2, 0]), {"FONT_SIZE": 16, "TEXT_COLOR": BLUE})
        
        geo_scene.add_element(side_label, "side_label")
        geo_scene.add_element(radius_label, "radius_label")
        
        # 角度标注
        angle_60_label = MyText("60°", point_A.point + np.array([-0.5, -0.3, 0]), {"FONT_SIZE": 14, "TEXT_COLOR": ORANGE})
        angle_120_label = MyText("120°", center_O.point + np.array([0.8, 0.8, 0]), {"FONT_SIZE": 14, "TEXT_COLOR": TEAL})
        
        geo_scene.add_element(angle_60_label, "angle_60_label")
        geo_scene.add_element(angle_120_label, "angle_120_label")
        
        # 创建标注关系
        geo_scene.add_relation(PointTextRelation(center_O, label_O, "圆心标注"))
        geo_scene.add_relation(PointTextRelation(point_A, label_A, "点A标注"))
        geo_scene.add_relation(PointTextRelation(point_B, label_B, "点B标注"))
        geo_scene.add_relation(PointTextRelation(point_C, label_C, "点C标注"))
        
        geo_scene.add_relation(LineTextRelation(line_AB, side_label, "边长标注"))
        geo_scene.add_relation(CircleTextRelation(main_circle, radius_label, "半径标注"))
        
        geo_scene.add_relation(AngleTextRelation(angle_BAC, angle_60_label, "60度角标注"))
        geo_scene.add_relation(AngleTextRelation(angle_AOB, angle_120_label, "120度角标注"))
        
        # === 10. 添加标题 ===
        title = MyText("圆内接等边三角形综合图形", np.array([0, 4, 0]), {
            "FONT_SIZE": 24, 
            "TEXT_COLOR": BLACK
        })
        geo_scene.add_element(title, "title")
    
    def animate_unified_figure(self, geo_scene):
        """动画展示统一图形"""
        
        # 1. 显示标题
        title = geo_scene.get_element("title")
        self.play(Write(title.mobject))
        self.wait(1)
        
        # 2. 显示圆心
        center_O = geo_scene.get_element("O")
        label_O = geo_scene.get_element("label_O")
        self.play(Create(center_O.mobject))
        self.play(Write(label_O.mobject))
        self.wait(0.5)
        
        # 3. 显示主圆
        main_circle = geo_scene.get_element("circle_O")
        radius_label = geo_scene.get_element("radius_label")
        self.play(Create(main_circle.mobject))
        self.play(Write(radius_label.mobject))
        self.wait(1)
        
        # 4. 显示圆上的三个点
        points = [geo_scene.get_element(key).mobject for key in ["A", "B", "C"]]
        point_labels = [geo_scene.get_element(key).mobject for key in ["label_A", "label_B", "label_C"]]
        self.play(*[Create(point) for point in points])
        self.play(*[Write(label) for label in point_labels])
        self.wait(1)
        
        # 5. 显示半径
        radii = [geo_scene.get_element(key).mobject for key in ["OA", "OB", "OC"]]
        self.play(*[Create(radius) for radius in radii])
        self.wait(1)
        
        # 6. 显示三角形的边
        triangle_sides = [geo_scene.get_element(key).mobject for key in ["AB", "BC", "CA"]]
        side_label = geo_scene.get_element("side_label")
        self.play(*[Create(side) for side in triangle_sides])
        self.play(Write(side_label.mobject))
        self.wait(1)
        
        # 7. 显示内角
        inner_angles = [geo_scene.get_element(key).mobject for key in ["angle_A", "angle_B", "angle_C"]]
        angle_60_label = geo_scene.get_element("angle_60_label")
        self.play(*[Create(angle) for angle in inner_angles])
        self.play(Write(angle_60_label.mobject))
        self.wait(1)
        
        # 8. 显示圆心角
        center_angles = [geo_scene.get_element(key).mobject for key in ["angle_AOB", "angle_BOC", "angle_COA"]]
        angle_120_label = geo_scene.get_element("angle_120_label")
        self.play(*[Create(angle) for angle in center_angles])
        self.play(Write(angle_120_label.mobject))
        self.wait(1)
        
        # 9. 显示圆弧
        arcs = [geo_scene.get_element(key).mobject for key in ["arc_AB", "arc_BC", "arc_CA"]]
        self.play(*[Create(arc) for arc in arcs])
        self.wait(1)
        
        # 10. 显示内切圆
        incircle = geo_scene.get_element("incircle")
        self.play(Create(incircle.mobject))
        self.wait(1)
        
        # 11. 显示垂线和直角标记
        mid_point = geo_scene.get_element("M_AB")
        perpendicular = geo_scene.get_element("OM_AB")
        right_angle = geo_scene.get_element("right_angle")
        
        self.play(Create(mid_point.mobject))
        self.play(Create(perpendicular.mobject))
        self.play(Create(right_angle.mobject))
        
        self.wait(2)


if __name__ == "__main__":
    # 配置场景
    config.frame_height = 10.0
    config.frame_width = 10.0
    
    with tempconfig({
        "background_color": WHITE,
        "pixel_height": 1200,
        "pixel_width": 1200
    }):
        scene = UnifiedGeometricFigure()
        scene.render()

# 这是我重新定义的元素和元素之间的关系，这是需要修改的代码，结合新定义的元素，修改整体的代码结构，构建scene需要一系列的流程：首先构造点，在构造线段，线段与线段之间会产生新的点，再构造角度（需要满足一定的要求），最后按照一定的数据结构保存他们的关系。
# - 需要对元素之间构建一个碰撞检测，元素体积就是get_bounding_box获取，这样文本或元素之间才不会出现叠在一起的情况