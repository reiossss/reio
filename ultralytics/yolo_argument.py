# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
实现旋转、平移、缩放、左右翻转、亮度、饱和度、2幅图像混合等一系列随机混合增强，默认使数据集扩充到原有的2倍左右。
A series of random blending enhancements, such as rotation, translation, scaling,
left and right flipping, brightness, saturation, and two-image mixing, are realized,
and the dataset is expanded to about twice the original by default.

usage:
    python yolo_argument.py \
                --input /yolo/dataset1/SODA \
                --output /yolo/dataset1/hat_and_clothes \
                [--factor 2]

folder:
    .
    └── PATH_TO_input_folder
        ├── images
            ├── 1.jpg
            ├── 2.jpg
            └── ...
        ├── labels
            ├── 1.txt
            ├── 2.txt
            └── ...
    .
    └── PATH_TO_output_folder
        ├── images
        ├── labels

"""

import os
import cv2
import numpy as np
import random
import math
import argparse
from pathlib import Path
import shutil
from PIL import Image, ImageEnhance


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='YOLO格式数据集增强工具')
    parser.add_argument('--input', type=str, required=True,
                        help='输入数据集目录，必须包含images和labels子目录')
    parser.add_argument('--output', type=str, required=True,
                        help='输出增强数据集目录')
    parser.add_argument('--factor', type=int, default=2,
                        help='每张原始图像生成的增强图像数量（默认为2，即最终3倍数据）')
    return parser.parse_args()


class Augmentor:
    """数据增强处理类"""

    def __init__(self):
        pass

    @staticmethod
    def random_rotation(image, labels, angle_range=(-45, 45)):
        """随机旋转图像和边界框"""
        angle = random.uniform(angle_range[0], angle_range[1])
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        # 创建旋转矩阵
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # 计算新边界尺寸
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # 调整旋转矩阵
        M[0, 2] += (nW / 2) - center[0]
        M[1, 2] += (nH / 2) - center[1]

        # 应用旋转
        rotated = cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # 转换标签
        new_labels = []
        for label in labels:
            cls, x, y, bw, bh = label

            # 转换为角点坐标
            x_min = (x - bw / 2) * w
            y_min = (y - bh / 2) * h
            x_max = (x + bw / 2) * w
            y_max = (y + bh / 2) * h

            # 应用旋转
            points = np.array([[[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]])
            transformed = cv2.transform(points, M).squeeze()

            # 计算新边界框
            x_min_t, y_min_t = transformed.min(axis=0)
            x_max_t, y_max_t = transformed.max(axis=0)

            # 转换回YOLO格式
            x_center = ((x_min_t + x_max_t) / 2) / nW
            y_center = ((y_min_t + y_max_t) / 2) / nH
            width = (x_max_t - x_min_t) / nW
            height = (y_max_t - y_min_t) / nH

            # 确保边界框在图像内
            if 0 < width < 1 and 0 < height < 1 and x_center > 0 and y_center > 0:
                new_labels.append([cls, x_center, y_center, width, height])

        return rotated, new_labels

    @staticmethod
    def random_translation(image, labels, trans_range=(-0.2, 0.2)):
        """随机平移图像和边界框"""
        h, w = image.shape[:2]
        tx = random.uniform(trans_range[0], trans_range[1]) * w
        ty = random.uniform(trans_range[0], trans_range[1]) * h

        # 创建平移矩阵
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        translated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # 更新标签
        new_labels = []
        for label in labels:
            cls, x, y, bw, bh = label
            new_x = (x * w + tx) / w
            new_y = (y * h + ty) / h

            # 检查边界框是否在图像内
            if 0 < new_x < 1 and 0 < new_y < 1:
                new_labels.append([cls, new_x, new_y, bw, bh])

        return translated, new_labels

    @staticmethod
    def random_scale(image, labels, scale_range=(0.8, 1.2)):
        """随机缩放图像和边界框 - 修复尺寸问题版本"""
        scale = random.uniform(scale_range[0], scale_range[1])
        h, w = image.shape[:2]
        new_w, new_h = int(w * scale), int(h * scale)

        # 缩放图像
        scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 创建新画布（保持原始尺寸）
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        x_offset = max(0, (w - new_w) // 2)  # 确保偏移量非负
        y_offset = max(0, (h - new_h) // 2)

        # 计算目标区域
        y_end = min(y_offset + new_h, h)
        x_end = min(x_offset + new_w, w)

        # 计算源区域
        src_h = min(new_h, h - y_offset)
        src_w = min(new_w, w - x_offset)

        # 确保区域有效
        if src_h > 0 and src_w > 0:
            canvas[y_offset:y_end, x_offset:x_end] = scaled[:src_h, :src_w]

        # 更新标签
        new_labels = []
        for label in labels:
            cls, x, y, bw, bh = label
            new_x = (x * w * scale + x_offset) / w
            new_y = (y * h * scale + y_offset) / h
            new_bw = bw * scale
            new_bh = bh * scale

            # 检查边界框是否在图像内
            if (0 < new_x < 1 and 0 < new_y < 1 and
                    0 < new_bw < 1 and 0 < new_bh < 1):
                new_labels.append([cls, new_x, new_y, new_bw, new_bh])

        return canvas, new_labels

    @staticmethod
    def random_flip(image, labels):
        """随机水平翻转图像和边界框"""
        if random.random() < 0.5:
            flipped = cv2.flip(image, 1)
            h, w = image.shape[:2]

            # 更新标签
            new_labels = []
            for label in labels:
                cls, x, y, bw, bh = label
                new_x = 1.0 - x
                new_labels.append([cls, new_x, y, bw, bh])

            return flipped, new_labels
        return image, labels

    @staticmethod
    def adjust_brightness(image, factor_range=(0.5, 1.5)):
        """随机调整亮度"""
        factor = random.uniform(factor_range[0], factor_range[1])
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    @staticmethod
    def adjust_saturation(image, factor_range=(0.5, 1.5)):
        """随机调整饱和度"""
        factor = random.uniform(factor_range[0], factor_range[1])
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Color(img_pil)
        img_pil = enhancer.enhance(factor)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # @staticmethod
    # def cutmix_two(image1, labels1, image2, labels2):
    #     """
    #     两张图像的裁剪混合
    #     保持原始图像大小，通过裁剪不同区域进行混合
    #     """
    #     h, w, _ = image1.shape
    #     result_image = image1.copy()
    #     result_labels = labels1.copy() if labels1 else []
    #
    #     # 随机确定裁剪区域大小（原始图像的20%-50%）
    #     crop_w = int(w * random.uniform(0.2, 0.5))
    #     crop_h = int(h * random.uniform(0.2, 0.5))
    #
    #     # 随机确定裁剪位置
    #     x = random.randint(0, w - crop_w)
    #     y = random.randint(0, h - crop_h)
    #
    #     # 从第二张图像中裁剪区域
    #     crop_img = image2[y:y + crop_h, x:x + crop_w]
    #
    #     # 将裁剪区域覆盖到基础图像
    #     result_image[y:y + crop_h, x:x + crop_w] = crop_img
    #
    #     # 处理标签：只添加在裁剪区域内的标签
    #     if labels2:
    #         for label in labels2:
    #             cls_id, cx, cy, bw, bh = label
    #             # 转换为绝对坐标
    #             box_x = cx * w
    #             box_y = cy * h
    #             box_w = bw * w
    #             box_h = bh * h
    #
    #             # 计算边界框坐标
    #             x1 = box_x - box_w / 2
    #             y1 = box_y - box_h / 2
    #             x2 = box_x + box_w / 2
    #             y2 = box_y + box_h / 2
    #
    #             # 检查边界框是否在裁剪区域内
    #             if (x1 >= x and y1 >= y and x2 <= x + crop_w and y2 <= y + crop_h):
    #                 # 转换为相对于整个图像的坐标
    #                 new_cx = (x1 + x2) / (2 * w)
    #                 new_cy = (y1 + y2) / (2 * h)
    #                 new_bw = box_w / w
    #                 new_bh = box_h / h
    #                 result_labels.append([cls_id, new_cx, new_cy, new_bw, new_bh])
    #
    #     return result_image, result_labels


def main():
    """主函数"""
    args = parse_arguments()

    input_dir = args.input
    output_dir = args.output
    augment_factor = args.factor

    # 验证输入目录结构
    if not os.path.exists(f'{input_dir}/images'):
        print(f"错误：输入目录 {input_dir} 缺少images子目录")
        exit(1)
    if not os.path.exists(f'{input_dir}/labels'):
        print(f"警告：输入目录 {input_dir} 缺少labels子目录，将仅处理图像")

    # 创建输出目录结构
    os.makedirs(f'{output_dir}/images', exist_ok=True)
    os.makedirs(f'{output_dir}/labels', exist_ok=True)

    # 获取所有图片文件
    image_files = [f for f in os.listdir(f'{input_dir}/images') if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f'发现 {len(image_files)} 张图像，开始数据增强...')

    if not image_files:
        print("错误：输入目录中没有找到图像文件")
        exit(1)

    # 复制原始文件到输出目录
    print("复制原始文件到输出目录...")
    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]

        # 复制图片
        src_img = f'{input_dir}/images/{img_file}'
        dst_img = f'{output_dir}/images/{img_file}'
        shutil.copy(src_img, dst_img)

        # 复制标签
        src_label = f'{input_dir}/labels/{base_name}.txt'
        dst_label = f'{output_dir}/labels/{base_name}.txt'
        if os.path.exists(src_label):
            shutil.copy(src_label, dst_label)

    # 主增强循环
    print(f"正在生成增强数据 (每张原始图像生成 {augment_factor} 个增强版本)...")
    for i, img_file in enumerate(image_files):
        base_name = os.path.splitext(img_file)[0]
        img_path = f'{input_dir}/images/{img_file}'
        label_path = f'{input_dir}/labels/{base_name}.txt'

        # 读取原始图像和标签
        image = cv2.imread(img_path)
        if image is None:
            print(f"警告：无法读取图像 {img_path}，跳过")
            continue

        # 读取标签
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        labels.append([int(parts[0]), *map(float, parts[1:5])])

        # 为每张图像生成多个增强版本
        for j in range(augment_factor):
            # 创建增强副本
            aug_image = image.copy()
            aug_labels = [label.copy() for label in labels] if labels else []

            # 随机应用基础增强
            if random.random() < 0.7 and labels:  # 70%概率应用旋转
                aug_image, aug_labels = Augmentor.random_rotation(aug_image, aug_labels)

            if random.random() < 0.7 and labels:  # 70%概率应用平移
                aug_image, aug_labels = Augmentor.random_translation(aug_image, aug_labels)

            if random.random() < 0.7:  # 70%概率应用缩放
                aug_image, aug_labels = Augmentor.random_scale(aug_image, aug_labels)

            if labels:  # 确保有标签时才进行翻转
                aug_image, aug_labels = Augmentor.random_flip(aug_image, aug_labels)

            # 应用颜色增强
            if random.random() < 0.5:
                aug_image = Augmentor.adjust_brightness(aug_image)

            if random.random() < 0.5:
                aug_image = Augmentor.adjust_saturation(aug_image)

            # 随机选择另一张图像进行裁剪混合
            # if random.random() < 0.5 and len(image_files) > 1:  # 50%概率应用裁剪混合
            #     other_file = random.choice([f for f in image_files if f != img_file])
            #     other_base = os.path.splitext(other_file)[0]
            #     other_img = cv2.imread(f'{input_dir}/images/{other_file}')
            #
            #     if other_img is not None:
            #         # 调整其他图像大小以匹配基础图像
            #         if other_img.shape[:2] != aug_image.shape[:2]:
            #             other_img = cv2.resize(other_img, (aug_image.shape[1], aug_image.shape[0]))
            #
            #         # 读取其他图像的标签
            #         other_labels = []
            #         other_label_path = f'{input_dir}/labels/{other_base}.txt'
            #         if os.path.exists(other_label_path):
            #             with open(other_label_path, 'r') as f:
            #                 for line in f:
            #                     parts = line.strip().split()
            #                     if len(parts) == 5:
            #                         other_labels.append([int(parts[0]), *map(float, parts[1:5])])
            #
            #         # 应用两张图像的裁剪混合
            #         aug_image, aug_labels = Augmentor.cutmix_two(
            #             aug_image, aug_labels,
            #             other_img, other_labels
            #         )

            # 保存增强结果
            output_img_path = f'{output_dir}/images/{base_name}_aug{j}.jpg'
            output_label_path = f'{output_dir}/labels/{base_name}_aug{j}.txt'

            cv2.imwrite(output_img_path, aug_image)

            # 保存标签
            if aug_labels:
                with open(output_label_path, 'w') as f:
                    for label in aug_labels:
                        cls_id, x, y, w, h = label
                        f.write(f"{int(cls_id)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
            elif os.path.exists(output_label_path):  # 如果没有标签但文件存在则删除
                if os.path.exists(output_label_path):
                    os.remove(output_label_path)

        if (i + 1) % 100 == 0 or (i + 1) == len(image_files):
            print(f'已处理 {i + 1}/{len(image_files)} 张图像')

    original_count = len(image_files)
    augmented_count = original_count * augment_factor
    total_count = original_count + augmented_count
    print(f'数据增强完成!')
    print(f'原始图像: {original_count}张')
    print(f'增强图像: {augmented_count}张')
    print(f'总计: {total_count}张 (约{total_count / original_count:.1f}倍)')


if __name__ == '__main__':
    main()