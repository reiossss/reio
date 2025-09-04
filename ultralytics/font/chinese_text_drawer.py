import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class ChineseTextDrawer:
    """
    一个用于在OpenCV图像上绘制中文文本的工具类。

    该类在初始化时加载字体文件，并提供一个方法用于在图像上绘制文本，
    从而避免了每次绘制时重复加载字体的开销。
    """

    def __init__(self, font_path="D:/project/ultralytics/font/msyhbd.ttf", font_size=15):
        """
        初始化绘制器。

        参数:
        font_path (str): ttf字体文件的路径。
        font_size (int): 默认的字体大小。
        """
        try:
            self.font = ImageFont.truetype(font_path, font_size)
        except IOError:
            raise IOError(f"字体文件未找到或无法读取，请检查路径：{font_path}")
        self.font_size = font_size

    def put_text(self, img, text, position, font_color, bg_color=None, font_size=None):
        """
        在图像上绘制中文文本，可选择添加背景。

        参数:
        img (numpy.ndarray): 输入图像 (OpenCV格式, BGR)。
        text (str): 要绘制的中文文本。
        position (tuple): 文本左上角的位置 (x, y)。
        font_color (tuple): 字体颜色 (BGR格式, e.g., (255, 0, 0) for blue)。
        bg_color (tuple, optional): 背景颜色 (BGR格式)。如果为None则不绘制背景。默认为 None。
        font_size (int, optional): 字体大小。如果为None，则使用初始化时的默认大小。默认为 None。

        返回:
        numpy.ndarray: 绘制了文本的图像 (OpenCV格式, BGR)。
        """
        if not isinstance(img, np.ndarray):
            raise TypeError("输入图像必须是OpenCV格式的numpy.ndarray。")

        # 将OpenCV图像转换为PIL图像 (BGR -> RGB)
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # 如果指定了新的字体大小，则使用新大小的字体，否则使用默认字体
        current_font = self.font
        if font_size and font_size != self.font_size:
            try:
                current_font = ImageFont.truetype(self.font.path, font_size)
            except IOError:
                print(f"警告：无法加载大小为 {font_size} 的字体，将使用默认大小 {self.font_size}。")

        # 获取文本尺寸 (使用 getbbox 更准确)
        bbox = current_font.getbbox(text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # 如果有背景颜色，绘制背景矩形
        if bg_color is not None:
            # 将BGR转换为RGB
            bg_color_rgb = (bg_color[2], bg_color[1], bg_color[0])
            # 绘制背景矩形
            draw.rectangle(
                [position[0], position[1] + 4, position[0] + text_width, position[1] + text_height + 4],
                fill=bg_color_rgb
            )

        # 将字体颜色BGR转换为RGB
        font_color_rgb = (font_color[2], font_color[1], font_color[0])

        # 绘制文本
        draw.text(position, text, font=current_font, fill=font_color_rgb)

        # 将PIL图像转换回OpenCV格式 (RGB -> BGR)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)