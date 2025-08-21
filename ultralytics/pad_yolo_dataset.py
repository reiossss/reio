# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
将图像和YOLO标签填充到指定尺寸。图像将保持其原始宽高比，并以黑色背景居中填充。
Fill the image and YOLO labels to the specified size.
The image will maintain its original aspect ratio and be centered with a black background.

usage:
    python pad_yolo_dataset.py \
                --input path/to/input \
                --output path/to/output \
                [--size 1280]

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

import cv2
import numpy as np
import os
import argparse
from tqdm import tqdm


def process_and_pad(image_path, label_path, output_image_path, output_label_path, target_size):
    """
    对单个图像进行缩放和填充，并相应地调整其YOLO标签。

    Args:
        image_path (str): 输入图像的路径。
        label_path (str): 输入YOLO标签文件的路径。
        output_image_path (str): 处理后图像的保存路径。
        output_label_path (str): 处理后标签的保存路径。
        target_size (int): 目标正方形尺寸 (宽和高)。
    """
    # 1. 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"警告：无法读取图像文件 {image_path}，已跳过。")
        return

    original_h, original_w, _ = image.shape

    # 2. 计算缩放比例
    # 如果原始图像的宽或高大于目标尺寸，则进行缩放
    scale = 1.0
    if original_w > target_size or original_h > target_size:
        scale = min(target_size / original_w, target_size / original_h)

    new_w = int(original_w * scale)
    new_h = int(original_h * scale)

    # 3. 缩放图像
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 4. 创建黑色背景画布并填充
    # 创建一个 (target_size, target_size) 的黑色画布
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)

    # 计算粘贴位置的左上角坐标 (偏移量)
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2

    # 将缩放后的图像粘贴到画布中央
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image

    # 5. 保存处理后的图像
    cv2.imwrite(output_image_path, canvas)

    # 6. 修改并保存对应的YOLO标签
    if not os.path.exists(label_path):
        print(f"警告：找不到图像 '{os.path.basename(image_path)}' 对应的标签文件，仅处理图像。")
        return

    # 计算坐标变换所需的比例和偏移量
    # new_w / target_size 是宽度方向上的缩放因子
    # new_h / target_size 是高度方向上的缩放因子
    # x_offset / target_size 是x方向上的归一化偏移
    # y_offset / target_size 是y方向上的归一化偏移
    w_ratio = new_w / target_size
    h_ratio = new_h / target_size
    x_offset_norm = x_offset / target_size
    y_offset_norm = y_offset / target_size

    new_labels = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = parts[0]
            # YOLO 格式: class_id, x_center, y_center, width, height (均为归一化值)
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            # 重新计算坐标
            # 新的宽度/高度 = 原宽度/高度 * 尺寸缩放比例
            new_width = width * w_ratio
            new_height = height * h_ratio
            # 新的中心点 = (原中心点 * 尺寸缩放比例) + 归一化偏移量
            new_x_center = x_center * w_ratio + x_offset_norm
            new_y_center = y_center * h_ratio + y_offset_norm

            new_labels.append(f"{class_id} {new_x_center:.6f} {new_y_center:.6f} {new_width:.6f} {new_height:.6f}")

    with open(output_label_path, 'w') as f:
        f.write("\n".join(new_labels))


def main():
    """
    主函数，用于解析命令行参数并启动处理流程。
    """
    parser = argparse.ArgumentParser(
        description="将图像和YOLO标签填充到指定尺寸。图像将保持其原始宽高比，并以黑色背景居中填充。")
    parser.add_argument('--input', type=str, required=True,
                        help="输入数据的主目录，应包含 'images' 和 'labels' 子目录。")
    parser.add_argument('--output', type=str, required=True, help="输出处理后数据的主目录。")
    parser.add_argument('--size', type=int, default=1280, help="目标正方形尺寸 (例如: 1280)。")

    args = parser.parse_args()

    # 构建输入和输出路径
    input_images_dir = os.path.join(args.input, 'images')
    input_labels_dir = os.path.join(args.input, 'labels')
    output_images_dir = os.path.join(args.output, 'images')
    output_labels_dir = os.path.join(args.output, 'labels')

    # 检查输入路径是否存在
    if not os.path.isdir(input_images_dir):
        print(f"错误：输入图像目录不存在 -> {input_images_dir}")
        return
    if not os.path.isdir(input_labels_dir):
        print(f"错误：输入标签目录不存在 -> {input_labels_dir}")
        return

    # 创建输出目录
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    # 获取所有图像文件
    image_files = [f for f in os.listdir(input_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"发现 {len(image_files)} 张图像，开始处理...")

    # 使用tqdm创建进度条
    for image_name in tqdm(image_files, desc="处理进度"):
        image_path = os.path.join(input_images_dir, image_name)

        # 构建对应的标签文件路径
        base_name = os.path.splitext(image_name)[0]
        label_name = f"{base_name}.txt"
        label_path = os.path.join(input_labels_dir, label_name)

        # 构建输出文件路径
        output_image_path = os.path.join(output_images_dir, image_name)
        output_label_path = os.path.join(output_labels_dir, label_name)

        # 调用核心处理函数
        process_and_pad(image_path, label_path, output_image_path, output_label_path, args.size)

    print(f"处理完成！所有文件已保存到 '{args.output}' 目录。")


if __name__ == '__main__':
    main()