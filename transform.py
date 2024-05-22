"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
# Implements image augmentation

import albumentations as alb
# from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageFont
import random
import string
import os

def alb_wrapper(transform):
    def f(im):
        return transform(image=np.asarray(im))["image"]

    return f


class Erosion(alb.ImageOnlyTransform):
    """
    Apply erosion operation to an image.

    Erosion is a morphological operation that shrinks the white regions in a binary image.

    Args:
        scale (int or tuple/list of int): The scale or range for the size of the erosion kernel.
            If an integer is provided, a square kernel of that size will be used.
            If a tuple or list is provided, it should contain two integers representing the minimum
            and maximum sizes for the erosion kernel.
        always_apply (bool, optional): Whether to always apply this transformation. Default is False.
        p (float, optional): The probability of applying this transformation. Default is 0.5.

    Returns:
        numpy.ndarray: The transformed image.
    """

    def __init__(self, scale, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        if type(scale) is tuple or type(scale) is list:
            assert len(scale) == 2
            self.scale = scale
        else:
            self.scale = (scale, scale)

    def apply(self, img, **params):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, tuple(np.random.randint(self.scale[0], self.scale[1], 2))
        )
        img = cv2.erode(img, kernel, iterations=1)
        return img


class Dilation(alb.ImageOnlyTransform):
    """
    Apply dilation operation to an image.

    Dilation is a morphological operation that expands the white regions in a binary image.

    Args:
        scale (int or tuple/list of int): The scale or range for the size of the dilation kernel.
            If an integer is provided, a square kernel of that size will be used.
            If a tuple or list is provided, it should contain two integers representing the minimum
            and maximum sizes for the dilation kernel.
        always_apply (bool, optional): Whether to always apply this transformation. Default is False.
        p (float, optional): The probability of applying this transformation. Default is 0.5.

    Returns:
        numpy.ndarray: The transformed image.
    """

    def __init__(self, scale, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        if type(scale) is tuple or type(scale) is list:
            assert len(scale) == 2
            self.scale = scale
        else:
            self.scale = (scale, scale)

    def apply(self, img, **params):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, tuple(np.random.randint(self.scale[0], self.scale[1], 2))
        )
        img = cv2.dilate(img, kernel, iterations=1)
        return img


class Bitmap(alb.ImageOnlyTransform):
    """
    Apply a bitmap-style transformation to an image.

    This transformation replaces all pixel values below a certain threshold with a specified value.

    Args:
        value (int, optional): The value to replace pixels below the threshold with. Default is 0.
        lower (int, optional): The threshold value below which pixels will be replaced. Default is 200.
        always_apply (bool, optional): Whether to always apply this transformation. Default is False.
        p (float, optional): The probability of applying this transformation. Default is 0.5.

    Returns:
        numpy.ndarray: The transformed image.
    """

    def __init__(self, value=0, lower=200, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.lower = lower
        self.value = value

    def apply(self, img, **params):
        img = img.copy()
        img[img < self.lower] = self.value
        return img




train_transform = alb_wrapper(
    alb.Compose(
        [
            # alb.OneOf([Dilation((2, 3))], p=0.05),
            # alb.Affine(shear={"x": (0, 3), "y": (-3, 0)}, cval=(255, 255, 255), p=0.1),
            alb.ShiftScaleRotate(
                shift_limit_x=(0, 0.04),
                shift_limit_y=(0, 0.03),
                scale_limit=(-0.15, 0.03),
                rotate_limit=2,
                border_mode=0,
                interpolation=2,
                value=(255, 255, 255),
                p=0.1,
            ),
            alb.GridDistortion(
                distort_limit=0.05,
                border_mode=0,
                interpolation=2,
                value=(255, 255, 255),
                p=0.1,
            ),
            # alb.Compose(
            #     [
            #         alb.Affine(
            #             translate_px=(0, 5), always_apply=True, cval=(255, 255, 255)
            #         ),
            #         alb.ElasticTransform(
            #             p=1,
            #             alpha=50,
            #             sigma=120 * 0.1,
            #             alpha_affine=120 * 0.01,
            #             border_mode=0,
            #             value=(255, 255, 255),
            #         ),
            #     ],
            #     p=1,
            # ),
            # .
            # alb.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),  # Adjust brightness and contrast
            # alb.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),  # Adjust hue, saturation, value            alb.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.2, hue=0.1, p=1), # 模拟毛刺
            # # alb.RandomBrightnessContrast(0.1, 0.5, True, p=1),
            # # alb.ImageCompression(95, p=0.07),
            # alb.GaussNoise(20, p=0.08),
            # # alb.GaussianBlur((3, 3), p=0.1),
            alb.RandomScale(scale_limit=(-0.6, -0.3), p=0.5),
            alb.LongestMaxSize(max_size=800),  # 首先等比例缩放图像的最长边为1024
            alb.PadIfNeeded(min_height=1280, min_width=960, border_mode=1, value=(255, 255, 255)),  # 然后添加必要的填充到1024x1024，使用边界模式0（常数填充）
        ]
    )
)


################################################################################################

def apply_watermark(src_image, text, text_size, rotation_angle):
    # Load the original image
    # Create a single watermark
    watermark = np.zeros((random.randint(60, 200), random.randint(100, 400), 3), dtype=np.uint8)
    r, g, b = random.randint(192, 255), random.randint(192, 255), random.randint(192, 255)
    watermark = put_text_husky(watermark, text, (r, g, b), text_size, "Times New Roman")

    # Define horizontal and vertical repeat counts based on the size of the source image
    h_repeat = src_image.shape[1] // watermark.shape[1] + 1
    v_repeat = src_image.shape[0] // watermark.shape[0] + 1

    # Create tiled watermark
    tiled_watermark = np.tile(watermark, (v_repeat, h_repeat, 1))

    # Crop the tiled watermark to the size of the original image
    tiled_watermark = tiled_watermark[:src_image.shape[0], :src_image.shape[1]]

    # Rotate the watermark
    center = (tiled_watermark.shape[1] // 2, tiled_watermark.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    rotated_watermark = cv2.warpAffine(tiled_watermark, rotation_matrix, (tiled_watermark.shape[1], tiled_watermark.shape[0]))

    # Blend the watermark with the original image
    cv2.addWeighted(src_image, 1.0, rotated_watermark, random.uniform(0.02, 0.05), 0, src_image)

    return src_image

def put_text_husky(img, text, color, font_size, font_name, italic=False, underline=False):
    # Convert OpenCV image to PIL format
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # Set font style
    font_style = ''
    if italic:
        font_style += 'I'
    if underline:
        font_style += 'U'
    
    # Load font or default
    try:
        # font = ImageFont.truetype(f'{font_name}{font_style}.ttf', font_size)
        font_name = os.listdir("/root/cv/lixumin/cv-donut/font")
        font = ImageFont.truetype(f"/root/cv/lixumin/cv-donut/font/{random.choice(font_name)}", font_size)
        # font_name = os.listdir("/Users/lixumin/Desktop/code/manim/xizhi/geo3k_gen/assess/fonts/AaCaiYuanGunGunKeAiTi")
        # font = ImageFont.truetype(f"/Users/lixumin/Desktop/code/manim/xizhi/geo3k_gen/assess/fonts/AaCaiYuanGunGunKeAiTi/{random.choice(font_name)}", font_size)
    
    except IOError:
        print(f"Font {font_name} with style {font_style} not found. Using default font.")
        font = ImageFont.load_default()

    # Calculate text position for center alignment
    text_width, text_height = draw.textsize(text, font=font)
    orgX = (img.shape[1] - text_width) // 2
    orgY = (img.shape[0] - text_height) // 2

    # Draw text
    draw.text((orgX, orgY), text, font=font, fill=(int(color[0]), int(color[1]), int(color[2])))

    # Convert back to OpenCV format
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def generate_balanced_watermark(length=15):
    # 定义可能的字符池
    chinese_chars = "".join([generate_random_chinese_char() for i in range(20)])
    english_chars = string.ascii_uppercase  # A-Z
    digits = string.digits  # 0-9
    
    # 确保每种类型的字符至少出现一次
    if length < 3:
        raise ValueError("Length must be at least 3 to include at least one of each character type.")
    
    # 生成包含至少一个中文、一个英文字母和一个数字的基础水印文本
    watermark_text = [
        random.choice(chinese_chars),
        random.choice(english_chars),
        random.choice(digits)
    ]
    
    # 填充剩余的字符
    all_chars = chinese_chars + english_chars + digits
    watermark_text += [random.choice(all_chars) for _ in range(length - 3)]
    
    # 混洗以增加随机性
    random.shuffle(watermark_text)
    
    # 将列表转换为字符串
    return ''.join(watermark_text)

class watermark(alb.ImageOnlyTransform):
    """
    """

    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        random_watermark = generate_balanced_watermark(random.randint(3, 7))  # 生成15个字符的水印文本
        font_size = random.randint(24, 50)
        rotation_angle = random.randint(-50, 50)
        # Example usage:
        result_img = apply_watermark(img, random_watermark, font_size, rotation_angle)
        return result_img

################################################################################################


################################################################################################

import random

import cv2
import numpy as np
from scipy.stats import norm


def _decayed_value_in_norm(x, max_value, min_value, center, range):
    """
    根据高斯分布从最大值衰减到最小值
    params:
    - x: 当前点的位置
    - max_value: 最大亮度值
    - min_value: 最小亮度值
    - center: 高斯分布的中心位置
    - range: 高斯分布的范围
    return:
    - x_value: 根据高斯分布调整后的亮度值
    """
    radius = range / 3
    center_prob = norm.pdf(center, center, radius)
    x_prob = norm.pdf(x, center, radius)
    x_value = (x_prob / center_prob) * (max_value - min_value) + min_value
    return x_value


def _decayed_value_in_linear(x, max_value, padding_center, decay_rate):
    """
    根据线性衰减从最大值衰减到最小值
    params:
    - x: 当前点的位置
    - max_value: 最大亮度值
    - padding_center: 线性衰减的中心点
    - decay_rate: 线性衰减率
    return:
    - x_value: 根据线性衰减调整后的亮度值
    """
    x_value = max_value - abs(padding_center - x) * decay_rate
    if x_value < 0:
        x_value = 1
    return x_value


def generate_parallel_light_mask(mask_size,
                                 position=None,
                                 direction=None,
                                 max_brightness=255,
                                 min_brightness=0,
                                 mode="gaussian",
                                 linear_decay_rate=None):
    """
    生成模拟平行光效果的mask
    params:
    - mask_size: 掩膜的大小，格式为(w, h)
    - position: 光源的位置，格式为(x, y)，默认为None，随机生成
    - direction: 光线的方向，以度为单位，从0到360，默认为None，随机生成
    - max_brightness: 掩膜中的最大亮度值
    - min_brightness: 掩膜中的最小亮度值
    - mode: 亮度衰减的模式，可选"linear_static"、"linear_dynamic"或"gaussian"
    - linear_decay_rate: 线性衰减模式下的衰减率，仅在线性模式下有效
    return:
    - light_mask: 根据指定参数生成的光效掩膜
    """
    # 如果没有指定位置或方向，随机生成
    if position is None:
        pos_x = random.randint(0, mask_size[0])
        pos_y = random.randint(0, mask_size[1])
    else:
        pos_x = position[0]
        pos_y = position[1]
    if direction is None:
        direction = random.randint(0, 360)
        # print("Rotate degree: ", direction)
    # 根据模式确定衰减率
    if linear_decay_rate is None:
        if mode == "linear_static":
            linear_decay_rate = random.uniform(0.2, 2)
        if mode == "linear_dynamic":
            linear_decay_rate = (max_brightness - min_brightness) / max(mask_size)
    assert mode in ["linear_dynamic", "linear_static", "gaussian"], \
        "mode must be linear_dynamic, linear_static or gaussian"
    # 添加填充以满足旋转后的裁剪
    padding = int(max(mask_size) * np.sqrt(2))
    canvas_x = padding * 2 + mask_size[0]
    canvas_y = padding * 2 + mask_size[1]
    mask = np.zeros(shape=(canvas_y, canvas_x), dtype=np.float32)
    # 初始化mask的左上角和右下角坐标
    init_mask_ul = (int(padding), int(padding))
    init_mask_br = (int(padding+mask_size[0]), int(padding+mask_size[1]))
    init_light_pos = (padding + pos_x, padding + pos_y)
    # 逐行填充掩膜，值从中心衰减
    for i in range(canvas_y):
        if mode == "linear":
            i_value = _decayed_value_in_linear(i, max_brightness, init_light_pos[1], linear_decay_rate)
        elif mode == "gaussian":
            i_value = _decayed_value_in_norm(i, max_brightness, min_brightness, init_light_pos[1], mask_size[1])
        else:
            i_value = 0
        mask[i] = i_value
    # 旋转mask
    rotate_M = cv2.getRotationMatrix2D(init_light_pos, direction, 1)
    mask = cv2.warpAffine(mask, rotate_M, (canvas_x,  canvas_y))
    # 裁剪
    mask = mask[init_mask_ul[1]:init_mask_br[1], init_mask_ul[0]:init_mask_br[0]]
    mask = np.asarray(mask, dtype=np.uint8)
    # 添加中值模糊
    mask = cv2.medianBlur(mask, 9)
    mask = 255 - mask
    return mask


def add_parallel_light(
        frame,
        light_position=None,
        direction=None,
        max_brightness=255,
        min_brightness=0,
        mode="gaussian",
        linear_decay_rate=None,
        transparency=None
):
    """
    向给定图像添加由平行光源生成的掩膜效果
    参数:
    - image: 输入的原始图像
    - light_position: 光源位置，格式为(x, y)，默认为None，随机生成
    - direction: 光线方向，以度为单位，从0到360，默认为None，随机生成
    - max_brightness: 掩膜中的最大亮度
    - min_brightness: 掩膜中的最小亮度
    - mode: 亮度衰减的模式，可选"linear_static"、"linear_dynamic"或"gaussian"
    - linear_decay_rate: 线性衰减模式下的衰减率，仅在线性模式下有效
    - transparency: 应用掩膜后的透明度，用于调整掩膜与原图的融合程度
    返回:
    - frame: 应用了平行光照明效果后的图像
    """
    # 设置透明度，默认值范围在0.5到0.85之间
    if transparency is None:
        transparency = random.uniform(0.7, 0.9)
    height, width, _ = frame.shape
    # 转换图像到HSV色彩空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 生成平行光mask
    mask = generate_parallel_light_mask(mask_size=(width, height),
                                        position=light_position,
                                        direction=direction,
                                        max_brightness=max_brightness,
                                        min_brightness=min_brightness,
                                        mode=mode,
                                        linear_decay_rate=linear_decay_rate)
    # 将mask应用到图像的亮度通道
    hsv[:, :, 2] = hsv[:, :, 2] * transparency + mask * (1 - transparency)
    # 将HSV图像转换回BGR色彩空间
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # 确保图像的亮度值不超过255
    frame[frame > 255] = 255
    frame = np.asarray(frame, dtype=np.uint8)
    return frame


def generate_spot_light_mask(mask_size,
                             position=None,
                             max_brightness=255,
                             min_brightness=0,
                             mode="gaussian",
                             linear_decay_rate=None,
                             speedup=False):
    """
    生成模拟聚光效果的掩膜，支持多个聚光源
    参数:
    - mask_size: 掩膜的大小，格式为(w, h)
    - position: 聚光源位置的列表，每个元素为一个元组(x, y)，默认为None，随机生成一个位置
    - max_brightness: 掩膜中的最大亮度
    - min_brightness: 掩膜中的最小亮度
    - mode: 亮度衰减的模式，"linear"或"gaussian"
    - linear_decay_rate: 线性衰减模式下的衰减率，仅在线性模式下有效
    - speedup: 是否使用加速计算方法，默认为False
    返回:
    - mask: 根据指定参数生成的聚光效果掩膜
    """
    # 如果未指定聚光位置，则随机生成一个位置
    if position is None:
        position = [(random.randint(0, mask_size[0]), random.randint(0, mask_size[1]))]
    # 如果未指定线性衰减率，并且模式为"linear_static"，则随机生成一个衰减率
    if linear_decay_rate is None:
        if mode == "linear_static":
            linear_decay_rate = random.uniform(0.25, 1)
    assert mode in ["linear_static", "gaussian"], \
        "mode must be linear_static or gaussian"
    # 初始化mask
    mask = np.zeros(shape=(mask_size[1], mask_size[0]), dtype=np.float32)
    # 如果模式为"gaussian"，则使用高斯分布计算掩膜
    if mode == "gaussian":
        mu = np.sqrt(mask.shape[0]**2+mask.shape[1]**2)
        dev = mu / 3.5  # 标准差
        mask = _decay_value_radically_norm_in_matrix(mask_size, position, max_brightness, min_brightness, dev)
    # 将mask转换为无符号整型，并应用中值模糊
    mask = np.asarray(mask, dtype=np.uint8)
    mask = cv2.medianBlur(mask, 5)
    mask = 255 - mask
    return mask


def _decay_value_radically_norm_in_matrix(mask_size, centers, max_value, min_value, dev):
    """
    在矩阵中使用高斯分布计算从中心向外衰减的亮度值，支持多个光源中心
    参数:
    - mask_size: 掩膜的大小，格式为(w, h)
    - centers: 光源中心列表，每个元素为一个坐标(x, y)
    - max_value: 最大亮度值
    - min_value: 最小亮度值
    - dev: 高斯分布的标准差
    返回:
    - mask: 根据高斯分布计算得到的亮度衰减掩膜
    """
    center_prob = norm.pdf(0, 0, dev)  # 中心点的概率密度值
    x_value_rate = np.zeros((mask_size[1], mask_size[0]))
    for center in centers:
        coord_x = np.arange(mask_size[0])
        coord_y = np.arange(mask_size[1])
        xv, yv = np.meshgrid(coord_x, coord_y)  # 生成网格坐标矩阵
        dist_x = xv - center[0]   # 计算每个点与光源中心的x轴距离
        dist_y = yv - center[1]   # 计算每个点与光源中心的y轴距离
        dist = np.sqrt(np.power(dist_x, 2) + np.power(dist_y, 2))  # 计算距离
        x_value_rate += norm.pdf(dist, 0, dev) / center_prob  # 计算每个点的亮度衰减比例
    mask = x_value_rate * (max_value - min_value) + min_value  # # 计算最终的亮度值
    mask[mask > 255] = 255  # 限制亮度值不超过255
    return mask


def _decay_value_radically_norm(x, centers, max_value, min_value, dev):
    """
    向给定图像添加聚光效果
    参数:
    - image: 输入的原始图像
    - light_position: 聚光灯位置列表，每个元素为一个坐标(x, y)，默认为None，随机生成一个位置
    - max_brightness: 聚光掩膜中的最大亮度
    - min_brightness: 聚光掩膜中的最小亮度
    - mode: 亮度衰减的模式，"linear"或"gaussian"
    - linear_decay_rate: 线性衰减模式下的衰减率，仅在线性模式下有效
    - transparency: 应用聚光掩膜后的透明度，用于调整掩膜与原图的融合程度
    返回:
    - frame: 应用了聚光效果后的图像
    """
    center_prob = norm.pdf(0, 0, dev)
    x_value_rate = 0
    for center in centers:
        distance = np.sqrt((center[0]-x[0])**2 + (center[1]-x[1])**2)
        x_value_rate += norm.pdf(distance, 0, dev) / center_prob
    x_value = x_value_rate * (max_value - min_value) + min_value
    x_value = 255 if x_value > 255 else x_value
    return x_value


def add_spot_light(
        frame,
        light_position=None,
        max_brightness=255,
        min_brightness=0,
        mode='gaussian',
        linear_decay_rate=None,
        transparency=None
):
    """
    向给定图像添加聚光效果
    参数:
    - image: 输入的原始图像
    - light_position: 聚光灯位置列表，每个元素为一个坐标(x, y)，默认为None，随机生成一个位置
    - max_brightness: 聚光掩膜中的最大亮度
    - min_brightness: 聚光掩膜中的最小亮度
    - mode: 亮度衰减的模式，"linear"或"gaussian"
    - linear_decay_rate: 线性衰减模式下的衰减率，仅在线性模式下有效
    - transparency: 应用聚光掩膜后的透明度，用于调整掩膜与原图的融合程度
    返回:
    - frame: 应用了聚光效果后的图像
    """
    # 设置透明度，默认值范围在0.5到0.85之间
    if transparency is None:
        transparency = random.uniform(0.7, 0.9)
    height, width, _ = frame.shape
    # 转换图像到HSV色彩空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 生成聚光mask
    mask = generate_spot_light_mask(mask_size=(width, height),
                                    position=light_position,
                                    max_brightness=max_brightness,
                                    min_brightness=min_brightness,
                                    mode=mode,
                                    linear_decay_rate=linear_decay_rate)
    # 将HSV图像转换回BGR色彩空间
    hsv[:, :, 2] = hsv[:, :, 2] * transparency + mask * (1 - transparency)
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # 确保图像的亮度值不超过255
    frame[frame > 255] = 255
    frame = np.asarray(frame, dtype=np.uint8)
    return frame


def random_by_spot_light(image):
    image_h, image_w = image.shape[:2]

    # 设置平行光或聚光灯效果的参数
    light_position = None  # 光源位置
    direction = None  # 平行光方向，以角度表示
    max_brightness = 255
    min_brightness = 254

    mode = random.choice(["linear_static", "gaussian"])  # 光线衰减模式
    linear_decay_rate = None  # 对于高斯模式，这个参数不会被使用
    transparency = None  # 透明度设置

    # 应用聚光效果
    enhanced_image = add_spot_light(image, light_position, max_brightness, min_brightness, mode, linear_decay_rate)
    return enhanced_image


def random_by_parallel_light(image):
    image_h, image_w = image.shape[:2]

    # 设置平行光或聚光灯效果的参数
    light_position = None  # 光源位置
    direction = None  # 平行光方向，以角度表示
    max_brightness = 255
    min_brightness = 254

    mode = random.choice(["linear_dynamic", "linear_static", "gaussian"])  # 光线衰减模式
    linear_decay_rate = None  # 对于高斯模式，这个参数不会被使用
    transparency = None  # 透明度设置

    # 应用平行光效果
    enhanced_image = add_parallel_light(image, light_position, direction, max_brightness, min_brightness, mode, linear_decay_rate)
    return enhanced_image


def random_brightness_reduction(image, min_factor=0.5, max_factor=1.0):
    """
    随机减弱图像亮度，模拟真实拍照场景下的减弱效果

    params：
    - image: 输入的BGR图像（NumPy数组）
    - min_factor: 允许的最小亮度减弱因子（0代表全黑，1代表原亮度）
    - max_factor: 允许的最大亮度减弱因子（应该大于min_factor且小于等于1）

    return：
    - 亮度减弱后的图像
    """
    # 生成随机的亮度减弱因子
    factor = random.uniform(min_factor, max_factor)

    # 转换到浮点数以允许减弱操作
    image_float = image.astype(np.float32)
    # 减弱亮度
    image_darkened = cv2.multiply(image_float, factor)
    # 转换回原始数据类型（uint8）
    image_darkened = np.clip(image_darkened, 0, 255).astype(np.uint8)

    return image_darkened


def random_enhance_light(image):

    enhance_funcs = ["random_by_parallel_light", "random_by_spot_light"]
    enhanced_image = eval(random.choice(enhance_funcs))(image)


    # enhanced_image = random_brightness_reduction(enhanced_image)

    return enhanced_image

class enhance(alb.ImageOnlyTransform):
    """
    """

    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        new_img = random_enhance_light(img)
        return new_img

################################################################################################

################################################################################################

def simulate_camera_blur(image, blur_chance=0.5):
    """
    模拟真实相机的模糊效果

    params:
    - image: 输入的图像
    - blur_chance: 模糊效果应用的概率

    return:
    - 模拟模糊后的图像
    """
    # 按一定概率不应用模糊，直接返回原图
    if np.random.rand() > blur_chance:
        return image

    blur_type = np.random.choice(['gaussian', 'mean', 'motion'])
    
    # 高斯模糊
    if blur_type == 'gaussian':
        kernel_size = np.random.choice([3, 5])  # 随机选择核大小
        blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
    # 均值模糊
    elif blur_type == 'mean':
        kernel_size = np.random.choice([3, 5])  # 随机选择核大小
        blurred_image = cv2.blur(image, (kernel_size, kernel_size))
        
    # 运动模糊
    elif blur_type == 'motion':
        kernel_size = np.random.choice([5, 9, 13])  # 运动模糊的核大小
        kernel_motion_blur = np.zeros((kernel_size, kernel_size))
        kernel_motion_blur[int((kernel_size-1)/2), :] = np.ones(kernel_size)
        kernel_motion_blur = kernel_motion_blur / kernel_size
        blurred_image = cv2.filter2D(image, -1, kernel_motion_blur)

    return blurred_image

class blur(alb.ImageOnlyTransform):
    """
    """

    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        new_img = simulate_camera_blur(img)
        return new_img


################################################################################################


################################################################################################

def simulate_color_jitter(image, intensity=0.001):
    # 创建一个与图像尺寸相同的随机噪声矩阵
    noise = np.random.randn(*image.shape) * 255 * intensity
    # 将噪声添加到原始图像上
    jittered_image = cv2.add(image, noise.astype(np.uint8))
    return jittered_image


def add_random_background_color(image, light_alpha=0.1, heavy_alpha=0.5):
    # 随机决定背景颜色的强度
    if random.random() < 0.3:  # 30%的概率选择较重的背景色
        # 较重的背景色，更低的亮度值
        background_color = np.random.randint(low=100, high=150, size=(1, 1, 3), dtype=np.uint8)
        alpha = heavy_alpha
    else:
        # 较浅的背景色，更高的亮度值
        background_color = np.random.randint(low=200, high=256, size=(1, 1, 3), dtype=np.uint8)
        alpha = light_alpha

    # 生成背景层
    background = np.tile(background_color, (image.shape[0], image.shape[1], 1))

    # 将背景与原图像结合
    blended_image = cv2.addWeighted(background, alpha, image, 1 - alpha, 0)
    return blended_image

class color(alb.ImageOnlyTransform):
    """
    """

    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        new_img = add_random_background_color(img)
        return new_img

class ResizeIfMaxSideExceeds(alb.ImageOnlyTransform):
    def __init__(self, max_size, always_apply=False, p=1.0):
        super(ResizeIfMaxSideExceeds, self).__init__(always_apply, p)
        self.max_size = max_size

    def apply(self, img, **params):
        # 获取图片的高度和宽度
        height, width = img.shape[:2]
        # 获取最长边
        max_side = max(height, width)
        
        if max_side > self.max_size:
            # 计算缩放比例
            scale = self.max_size / max_side
            new_height, new_width = int(height * scale), int(width * scale)
            # 等比例缩放图片
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return img
    
class AddBlurredBlackDots(alb.ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(AddBlurredBlackDots, self).__init__(always_apply, p)

    def apply(self, image, **params):
        return self.add_blurred_black_dots(image)

    def add_blurred_black_dots(self, image):
        h, w = image.shape[:2]
        image = np.copy(image)
        num_dots = np.random.randint(5, 20)  # 随机生成5到20个黑点

        for _ in range(num_dots):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            radius = np.random.randint(1, 5)  # 随机半径
            color = (0, 0, 0)  # 黑色
            thickness = -1  # 实心圆

            # 在图像上绘制黑点
            cv2.circle(image, (x, y), radius, color, thickness)

            # 对黑点进行模糊处理
            ksize = np.random.choice([3, 5])  # 随机选择模糊核大小
            image = cv2.GaussianBlur(image, (ksize, ksize), 0)

        return image

################################################################################################

train_transform =  alb_wrapper(
    alb.Compose(
        [
            #alb.OneOf([Dilation((2, 3)), Erosion((2, 3))], p=0.05),
            #alb.Affine(shear={"x": (0, 3), "y": (-3, 0)}, cval=(255, 255, 255), p=0.1),
            #alb.ShiftScaleRotate(
            #    shift_limit_x=(0, 0.04),
            #    shift_limit_y=(0, 0.03),
            #    scale_limit=(-0.15, 0.03),
            #    rotate_limit=2,
            #    border_mode=0,
            #    interpolation=2,
            #    value=(255, 255, 255),
            #    p=0.1,
            #),
            #alb.GridDistortion(
            #    distort_limit=0.05,
            #    border_mode=0,
            #    interpolation=2,
            #    value=(255, 255, 255),
            #    p=0.1,
            #),
            alb.GaussNoise(20, p=1),
            alb.GaussianBlur((3, 3), p=1),
            alb.RandomScale(scale_limit=(-0.6, 0), p=1),
            ResizeIfMaxSideExceeds(max_size=512),  # 首先等比例缩放图像的最长边为1024
            alb.PadIfNeeded(min_height=512, min_width=512, border_mode=1, value=(255, 255, 255), position="random"),  # 然后添加必要的填充到1024x1024，使用边界模式0（常数填充）
            # watermark(p=0.1),
            enhance(p=0.8),
            color(p=0.8),
        ],
    )
)

import random

def generate_random_chinese_char():
    # 生成一个随机的汉字Unicode编码值
    unicode_val = random.randint(0x4E00, 0x9FFF)
    # 将Unicode编码值转换为对应的字符
    return chr(unicode_val)

if __name__ == '__main__':

    # 调用函数生成一个随机中文字符
    # random_chinese_char = generate_random_chinese_char()
    # print("Random Chinese Character:", random_chinese_char)

    image = Image.open('/Users/lixumin/Desktop/code/manim/cv-geometric-dataset/media/images/img/1.png').convert('RGB')
    # image = prepare_input(image)
    image = Image.fromarray(train_transform(image))
    # cropped_img = image.crop((200, 400, 760, 800), )
    # image_with_padding = ImageOps.expand(cropped_img, border=(0, 0), fill=(255, 255, 255))
    # image_with_padding.save("./show.png")
    image.save("./show.png")
