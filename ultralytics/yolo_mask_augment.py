# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
yolo分割框数据增强脚本，实现旋转、平移、缩放、左右翻转、亮度、饱和度等一系列随机混合增强，默认使数据集扩充到原有的3倍左右。
The YOLO mask data augmentation script implements a series of random mixed enhancements
such as rotation, translation, scaling, horizontal flipping, brightness, and saturation,
and by default expands the dataset to about three times its original size.

usage:
    python yolo_mask_augment.py \
                --input path/to/input \
                --output path/to/output \
                [--num 2]

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
import argparse
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm
import shutil


def augment_yolo_segmentation(input_dir, output_dir, num_augmentations=2):
    """
    对YOLO分割数据集进行数据增强。

    Args:
        input_dir (str): 输入数据集的路径，包含 'images' 和 'labels' 子文件夹。
        output_dir (str): 存放增强后数据的新目录的路径。
        num_augmentations (int): 每张原始图像要生成的增强版本数量。
    """
    # 1. 定义数据增强的流程
    # Albumentations可以同时处理图像和掩码（masks）
    # 几何变换：翻转、拉伸、旋转
    # 颜色变换：亮度、对比度、饱和度、色调
    transform = A.Compose([
        # 几何变换
        A.HorizontalFlip(p=0.7),
        A.Affine(
            scale=(0.9, 1.1),  # 图像缩放范围（80% - 120%）
            translate_percent=(-0.0625, 0.0625),  # 平移范围
            rotate=(-45, 45),  # 旋转范围 (-45 到 45 度)
            p=0.8,
        ),
        # 颜色变换
        A.ColorJitter(
            brightness=0.3,  # 亮度调整范围
            contrast=0.3,  # 对比度调整范围
            saturation=0.3,  # 饱和度调整范围
            hue=0.2,  # 色调调整范围
            p=0.7  # 应用此变换的概率
        ),
        # 可以添加更多增强操作，例如：
        # A.RandomBrightnessContrast(p=0.5),
        # A.GaussNoise(p=0.3),
        # A.Blur(p=0.3),
    ])

    # 2. 设置输入和输出路径
    image_dir = os.path.join(input_dir, 'images')
    label_dir = os.path.join(input_dir, 'labels')

    output_image_dir = os.path.join(output_dir, 'images')
    output_label_dir = os.path.join(output_dir, 'labels')

    # 创建输出目录
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"找到 {len(image_files)} 张图像，开始数据增强...")

    # 3. 遍历每张图像进行处理
    for image_filename in tqdm(image_files, desc="处理图像中"):
        base_name = os.path.splitext(image_filename)[0]

        # 复制图片
        src_img = f'{input_dir}/images/{image_filename}'
        dst_img = f'{output_dir}/images/{image_filename}'
        shutil.copy(src_img, dst_img)

        # 复制标签
        src_label = f'{input_dir}/labels/{base_name}.txt'
        dst_label = f'{output_dir}/labels/{base_name}.txt'
        if os.path.exists(src_label):
            shutil.copy(src_label, dst_label)
        image_path = os.path.join(image_dir, image_filename)
        label_filename = os.path.splitext(image_filename)[0] + '.txt'
        label_path = os.path.join(label_dir, label_filename)

        if not os.path.exists(label_path):
            print(f"警告：找不到对应的标签文件 {label_path}，跳过此图像。")
            continue

        # 读取图像
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Albumentations 使用 RGB 格式
        h, w = image.shape[:2]

        # 读取YOLO分割标签并创建掩码
        masks = []
        class_ids = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                polygon_norm = np.array([float(p) for p in parts[1:]]).reshape(-1, 2)

                # 将归一化坐标转换为绝对像素坐标
                polygon_abs = polygon_norm * np.array([w, h])

                # 创建一个空白掩码
                mask = np.zeros((h, w), dtype=np.uint8)
                # 在掩码上绘制实心多边形
                cv2.fillPoly(mask, [polygon_abs.astype(np.int32)], 1)

                masks.append(mask)
                class_ids.append(class_id)

        # 4. 应用数据增强
        for i in range(num_augmentations):
            try:
                # 将图像和所有掩码一起传入进行变换
                augmented = transform(image=image, masks=masks)
                aug_image = augmented['image']
                aug_masks = augmented['masks']

                # 5. 保存增强后的图像和标签
                new_h, new_w = aug_image.shape[:2]

                # 构建新的文件名
                base_name, ext = os.path.splitext(image_filename)
                new_image_filename = f"{base_name}_aug_{i}{ext}"
                new_label_filename = f"{base_name}_aug_{i}.txt"

                # 保存增强后的图像 (转回 BGR)
                cv2.imwrite(
                    os.path.join(output_image_dir, new_image_filename),
                    cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                )

                # 处理增强后的掩码，转换回YOLO格式
                yolo_labels = []
                for class_id, aug_mask in zip(class_ids, aug_masks):
                    # 从二值掩码中寻找轮廓
                    contours, _ = cv2.findContours(aug_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if contours:
                        # 通常只有一个外部轮廓，我们取最大的那个
                        contour = max(contours, key=cv2.contourArea)

                        # 确保轮廓有足够的点
                        if contour.shape[0] >= 3:
                            # 归一化轮廓点
                            contour_norm = contour.astype(np.float32).squeeze(1)
                            contour_norm[:, 0] /= new_w
                            contour_norm[:, 1] /= new_h

                            # 展平为 [x1, y1, x2, y2, ...] 格式
                            flat_contour = contour_norm.flatten()

                            # 格式化为YOLO字符串
                            yolo_line = f"{class_id} " + " ".join([f"{coord:.6f}" for coord in flat_contour])
                            yolo_labels.append(yolo_line)

                # 保存新的标签文件
                if yolo_labels:
                    with open(os.path.join(output_label_dir, new_label_filename), 'w') as f:
                        f.write("\n".join(yolo_labels))

            except Exception as e:
                print(f"处理 {image_filename} 时发生错误：{e}")


def main():
    parser = argparse.ArgumentParser(description="YOLO Segmentation Data Augmentation Script")
    parser.add_argument('--input', type=str, required=True,
                        help='Path to the input dataset directory (containing images and labels subfolders).')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to the directory to save augmented data.')
    parser.add_argument('--num', type=int, default=2,
                        help='Number of augmented versions to generate per original image.')

    args = parser.parse_args()

    augment_yolo_segmentation(args.input, args.output, args.num)
    print("\n数据增强完成！")
    print(f"结果已保存在: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()