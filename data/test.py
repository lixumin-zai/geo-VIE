from manim import *
import numpy as np

class Test(Scene):
    def construct(self):
        start_point = np.array([3, 0, 0])
        vertex_point = np.array([0, 0, 0])
        end_point = np.array([2.9999, -0.0001, 0])
        self.add(Dot(start_point, color=RED))
        self.add(Dot(vertex_point, color=RED))
        self.add(Dot(end_point, color=RED))
        angle = Angle(Line(vertex_point, start_point), Line(vertex_point, end_point),
                # length=0.2,
                radius=1,
                color=BLACK
        )

        num_points = 100 # 定义 100 个点
        alpha_values = np.linspace(0, 1, num_points) # 100个点
        bounding_box = [angle.point_from_proportion(alpha) for alpha in alpha_values]
        print(len(bounding_box))
        arc_start_point = angle.point_from_proportion(0.6)
        self.add(angle)
        self.add(Dot(arc_start_point, color=RED))

# /root/project/geo-vie/data/media/images/Test_ManimCE_v0.19.0.png

# Use a font that supports Chinese characters
# For Manim Community v0.17.3 and later, you can often just use Text directly
# If not, you might need to specify font or use TexTemplate for CJK with LaTeX
# For simplicity, we'll try direct Text first. If issues, use specific font.
DEFAULT_FONT = "SimHei" # Or "Arial Unicode MS", "Noto Sans CJK SC", etc.

class FoldRectangle(ThreeDScene):
    def construct(self):
        # --- Constants and Coordinates ---
        s2 = np.sqrt(2)
        A_coord = np.array([0, 0, 0])
        D_coord = np.array([0, s2, 0])
        E_coord = np.array([s2, s2, 0])
        C_coord = np.array([s2 + s2/2, s2, 0])
        B_coord = np.array([s2 + s2/2, 0, 0])

        # --- Phase 1: Draw Figure 1 ---
        title_fig1 = Text("图 1", font=DEFAULT_FONT).to_edge(UP)

        # Create Dots
        dot_A = Dot(A_coord, color=BLUE)
        dot_B = Dot(B_coord, color=BLUE)
        dot_C = Dot(C_coord, color=BLUE)
        dot_D = Dot(D_coord, color=RED) # D will move
        dot_E = Dot(E_coord, color=GREEN)

        # Create Labels for points
        label_A = MathTex("A").next_to(dot_A, DL, buff=0.1)
        label_B = MathTex("B").next_to(dot_B, DR, buff=0.1)
        label_C = MathTex("C").next_to(dot_C, UR, buff=0.1)
        label_D_orig = MathTex("D").next_to(dot_D, UL, buff=0.1)
        label_E = MathTex("E").next_to(dot_E, UR, buff=0.1)
        
        # Create Lines for the rectangle and AE
        line_AD = Line(A_coord, D_coord)
        line_DE = Line(D_coord, E_coord)
        line_EC = Line(E_coord, C_coord)
        line_CB = Line(C_coord, B_coord)
        line_BA = Line(B_coord, A_coord)
        line_AE = Line(A_coord, E_coord, stroke_width=2, color=YELLOW) # Fold line

        # Dimension Labels
        dim_AD = MathTex(r"\sqrt{2}").next_to(line_AD, LEFT, buff=0.1).scale(0.7)
        dim_DE = MathTex(r"\sqrt{2}").next_to(line_DE, UP, buff=0.1).scale(0.7)
        dim_EC = MathTex(r"\frac{\sqrt{2}}{2}").next_to(line_EC, UP, buff=0.1).scale(0.7)

        fig1_group = VGroup(
            dot_A, dot_B, dot_C, dot_D, dot_E,
            label_A, label_B, label_C, label_D_orig, label_E,
            line_AD, line_DE, line_EC, line_CB, line_BA, line_AE,
            dim_AD, dim_DE, dim_EC
        )

        # Set camera for 2D-like view initially
        self.set_camera_orientation(phi=0 * DEGREES, theta=-90 * DEGREES, gamma=0 * DEGREES)
        
        self.play(Write(title_fig1))
        self.play(Create(fig1_group), run_time=3)
        self.wait(1)

        # --- Phase 2: Show folding instruction text ---
        instruction_text_parts = [
            Text("如图1,在矩形 ABCD 中,点 E 是边 CD 上的一点,", font=DEFAULT_FONT, font_size=24),
            Text("且 AD=DE=2CE=√2.", font=DEFAULT_FONT, font_size=24),
            Text("现将△ADE 沿 AE 向上翻折,", font=DEFAULT_FONT, font_size=24),
            Text("使得点 D 到点 P 的位置,", font=DEFAULT_FONT, font_size=24),
            Text("得到如图2所示的四棱锥 P-ABCE.", font=DEFAULT_FONT, font_size=24),
            Text("点 G 是线段 PB 上的一点.", font=DEFAULT_FONT, font_size=24)
        ]
        
        # Concatenate relevant part for this animation step
        full_instruction_text_str = " ".join([
            "如图1,在矩形 ABCD 中,点 E 是边 CD 上的一点,且 AD=DE=2CE=√2.",
            "现将△ADE 沿 AE 向上翻折,使得点 D 到点 P 的位置,"
        ])
        
        instruction_text_display = Text(
            full_instruction_text_str, 
            font=DEFAULT_FONT, 
            font_size=20,
            line_spacing=0.8
        ).to_edge(DOWN)

        # Create a background box for the text for better readability
        text_bg = SurroundingRectangle(instruction_text_display, buff=0.2, fill_color=BLACK, fill_opacity=0.7, stroke_width=0)
        
        self.play(FadeIn(text_bg), Write(instruction_text_display), run_time=2)
        self.wait(3)
        self.play(FadeOut(instruction_text_display), FadeOut(text_bg), FadeOut(title_fig1))
        self.wait(0.5)

        # --- Phase 3: Animate the fold ---
        # Adjust camera for 3D view
        self.move_camera(phi=65 * DEGREES, theta=-60 * DEGREES, run_time=2)
        
        # Objects that form the base and remain static during folding
        base_ABCE_dots = VGroup(dot_A, dot_B, dot_C, dot_E) # D is removed, will be P
        base_ABCE_labels = VGroup(label_A, label_B, label_C, label_E) # D label removed
        base_ABCE_lines = VGroup(line_CB, line_BA, line_EC) # AD, DE will be replaced by AP, EP
        # Keep AE (fold line) visible
        
        # Objects to be folded (triangle ADE parts)
        # Note: line_AD and line_DE were already created. dot_D and label_D_orig too.
        # We need to ensure they are treated as 3D objects now.
        
        # Group the parts that will rotate
        # Original line_AD, line_DE, dot_D, label_D_orig
        folding_parts = VGroup(line_AD, line_DE, dot_D, label_D_orig)
        
        # Axis of rotation is vector AE, about point A
        axis_of_rotation = E_coord - A_coord # Vector from A to E
        point_on_axis = A_coord
        
        # P_coord: As calculated, P = (√2/2, √2/2, 1)
        # However, for rotation, Manim's Rotate will handle the new position.
        # We rotate by PI/2 (90 degrees) around AE.
        # The z-axis for "upwards" will be perpendicular to the plane of AE and AD.
        # The vector normal to plane ADE (at A) can be cross(AD_vec, AE_vec).
        # A simpler way is to use the Rodrigues' rotation formula result for P.
        # P_coord = np.array([s2/2, s2/2, 1.0]) (This is if AE is along an axis, simpler to use Rotate)

        self.play(
            Rotate(
                folding_parts,
                angle=PI/2, # Fold 90 degrees
                axis=normalize(axis_of_rotation), # Unit vector for axis
                about_point=point_on_axis
            ),
            run_time=3
        )
        self.wait(0.5)

        # --- Phase 4: Show the resulting pyramid P-ABCE ---
        # After rotation, dot_D is now at P's position.
        P_coord_final = dot_D.get_center() # Get the new coordinates of the rotated D
        
        # Relabel D to P
        label_P = MathTex("P").move_to(label_D_orig.get_center()).set_color(RED) # Place P label where D label was
        self.play(FadeOut(label_D_orig), FadeIn(label_P))

        # line_AD is now line_AP, line_DE is now line_EP. Color them if needed.
        line_AP = line_AD # It was transformed
        line_EP = line_DE # It was transformed
        line_AP.set_color(PURPLE)
        line_EP.set_color(PURPLE)

        # Draw new lines for the pyramid: PB, PC
        line_PB = Line(P_coord_final, B_coord, color=PURPLE)
        line_PC = Line(P_coord_final, C_coord, color=PURPLE)

        pyramid_edges_new = VGroup(line_PB, line_PC)
        self.play(Create(pyramid_edges_new))
        
        result_text_str = "得到如图2所示的四棱锥 P-ABCE."
        result_text_display = Text(
            result_text_str, 
            font=DEFAULT_FONT, 
            font_size=20
        ).to_corner(DR)
        result_text_bg = SurroundingRectangle(result_text_display, buff=0.1, fill_color=BLACK, fill_opacity=0.7, stroke_width=0)

        self.play(FadeIn(result_text_bg), Write(result_text_display))
        self.wait(1)

        # --- Phase 5: Show point G ---
        # G is a point on PB. Let's place it at the midpoint for visualization.
        G_coord = (P_coord_final + B_coord) / 2
        dot_G = Dot(G_coord, color=ORANGE)
        label_G = MathTex("G").next_to(dot_G, OUT + RIGHT*0.5, buff=0.1).scale(0.8)

        g_text_str = "点 G 是线段 PB 上的一点."
        g_text_display = Text(
            g_text_str, 
            font=DEFAULT_FONT, 
            font_size=20
        ).next_to(result_text_display, UP, aligned_edge=RIGHT)
        g_text_bg = SurroundingRectangle(g_text_display, buff=0.1, fill_color=BLACK, fill_opacity=0.7, stroke_width=0)
        
        self.play(FadeIn(g_text_bg), Write(g_text_display))
        self.play(Create(dot_G), Write(label_G))
        
        # Optional: Ambient rotation to show off the 3D structure
        self.begin_ambient_camera_rotation(rate=0.1, about="theta") # Rotate around vertical axis
        self.wait(5)
        self.stop_ambient_camera_rotation()
        
        self.wait(2)

# To render, save this as a .py file (e.g., fold_animation.py)
# and run in terminal: manim -pql fold_animation.py FoldRectangle
# For higher quality: manim -pqh fold_animation.py FoldRectangle
# For preview quality: manim -pql fold_animation.py FoldRectangle --renderer=opengl (for faster rendering with opengl)

if __name__ == "__main__":
    config.frame_height =10
    with tempconfig(
        {
            "quality": "low_quality",
            "background_color": WHITE,
            "pixel_height": 1400,
            "pixel_width": 1400
        }):
        scene = Test()
        scene.render()

        # /root/project/geo-vie/data/media/images/Test_ManimCE_v0.19.0.png