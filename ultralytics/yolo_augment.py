# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
yolo检测框/分割框数据增强脚本，实现旋转、平移、缩放、左右翻转、亮度、饱和度等一系列随机混合增强，默认使数据集扩充到原有的3倍左右。
The YOLO detection box / mask data augmentation script implements a series of random mixed enhancements
such as rotation, translation, scaling, horizontal flipping, brightness, and saturation,
and by default expands the dataset to about three times its original size.

usage:
    python yolo_augment.py \
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
import cv2
import shutil
import argparse
import numpy as np
import albumentations as A
from tqdm import tqdm


def get_augmentation_pipelines():
    """定义并返回用于检测和分割的两种增强管道"""

    # 共享的变换逻辑
    shared_transforms = [
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2, p=0.75),
        A.RandomBrightnessContrast(p=0.7),
        A.HorizontalFlip(p=0.7),
        A.Affine(
            scale=(0.8, 1.2),
            translate_percent=(-0.1, 0.1),
            rotate=(-30, 30),
            p=0.75
        ),
    ]

    # 用于检测框的管道
    detection_pipeline = A.Compose(
        shared_transforms,
        bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels']
        )
    )

    # 用于分割的管道（不需要bbox_params）
    segmentation_pipeline = A.Compose(shared_transforms)

    return detection_pipeline, segmentation_pipeline


def augment_dataset(input_dir, output_dir, num_aug_per_image=2):
    """
    对数据集进行智能数据增强，自动识别标签类型。
    """
    # 1. 设置路径
    input_images_dir = os.path.join(input_dir, 'images')
    input_labels_dir = os.path.join(input_dir, 'labels')
    output_images_dir = os.path.join(output_dir, 'images')
    output_labels_dir = os.path.join(output_dir, 'labels')

    # 2. 创建输出目录并复制原始文件
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    print("Step 1: 正在复制原始文件...")
    for subdir, output_subdir in [(input_images_dir, output_images_dir), (input_labels_dir, output_labels_dir)]:
        if not os.path.exists(subdir):
            print(f"Warning: Directory not found, skipping: {subdir}")
            continue
        for filename in tqdm(os.listdir(subdir), desc=f"复制 {os.path.basename(subdir)}"):
            shutil.copy2(os.path.join(subdir, filename), os.path.join(output_subdir, filename))
    print("原始文件复制完成！\n")

    # 3. 获取增强管道和图像列表
    detection_transform, segmentation_transform = get_augmentation_pipelines()
    image_files = [f for f in os.listdir(input_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    detected_format = None  # 用于存储检测到的格式

    print(f"Step 2: 开始为 {len(image_files)} 张图像生成增强数据...")
    # 4. 遍历所有图像进行处理
    for image_filename in tqdm(image_files, desc="生成增强数据中"):
        base_name, ext = os.path.splitext(image_filename)
        image_path = os.path.join(input_images_dir, image_filename)
        label_path = os.path.join(input_labels_dir, base_name + '.txt')

        if not os.path.exists(label_path):
            continue

        # --- 自动格式检测 ---
        current_format = None
        with open(label_path, 'r') as f:
            first_line = f.readline().strip()
            if first_line:
                parts = first_line.split()
                num_parts = len(parts)
                if num_parts == 5:
                    current_format = 'detection'
                elif num_parts > 5 and num_parts % 2 != 0:
                    current_format = 'segmentation'

        if not current_format:
            print(f"\n警告：无法识别标签格式或文件为空，已跳过 {label_path}")
            continue

        if detected_format is None:
            detected_format = current_format
            print(f"\n--- 数据集格式自动检测为: {detected_format.upper()} ---")
        elif detected_format != current_format:
            print(f"\n警告：数据集中标签格式不一致，已跳过 {label_path}")
            continue

        # --- 读取数据和增强 ---
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for i in range(num_aug_per_image):
            aug_base_name = f"{base_name}_aug_{i}"

            try:
                if detected_format == 'detection':
                    # --- 检测框处理流程 ---
                    bboxes, class_labels = [], []
                    with open(label_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            class_labels.append(int(parts[0]))
                            bboxes.append([float(p) for p in parts[1:]])

                    augmented = detection_transform(image=image_rgb, bboxes=bboxes, class_labels=class_labels)

                    if not augmented['bboxes']: continue

                    aug_image_path = os.path.join(output_images_dir, aug_base_name + ext)
                    cv2.imwrite(aug_image_path, cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR))

                    aug_label_path = os.path.join(output_labels_dir, aug_base_name + '.txt')
                    with open(aug_label_path, 'w') as f:
                        for bbox, class_id in zip(augmented['bboxes'], augmented['class_labels']):
                            f.write(f"{int(class_id)} {' '.join(f'{c:.6f}' for c in bbox)}\n")

                elif detected_format == 'segmentation':
                    # --- 分割处理流程 ---
                    h, w = image_rgb.shape[:2]
                    masks, class_ids = [], []
                    with open(label_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            class_ids.append(int(parts[0]))
                            polygon_norm = np.array([float(p) for p in parts[1:]]).reshape(-1, 2)
                            polygon_abs = polygon_norm * np.array([w, h])
                            mask = np.zeros((h, w), dtype=np.uint8)
                            cv2.fillPoly(mask, [polygon_abs.astype(np.int32)], 1)
                            masks.append(mask)

                    augmented = segmentation_transform(image=image_rgb, masks=masks)
                    new_h, new_w = augmented['image'].shape[:2]

                    aug_image_path = os.path.join(output_images_dir, aug_base_name + ext)
                    cv2.imwrite(aug_image_path, cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR))

                    aug_label_path = os.path.join(output_labels_dir, aug_base_name + '.txt')
                    yolo_labels = []
                    for class_id, aug_mask in zip(class_ids, augmented['masks']):
                        contours, _ = cv2.findContours(aug_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            contour = max(contours, key=cv2.contourArea)
                            if contour.shape[0] >= 3:
                                contour_norm = contour.astype(np.float32).squeeze(1)
                                contour_norm[:, 0] /= new_w
                                contour_norm[:, 1] /= new_h
                                yolo_line = f"{class_id} " + " ".join([f"{c:.6f}" for c in contour_norm.flatten()])
                                yolo_labels.append(yolo_line)

                    if yolo_labels:
                        with open(aug_label_path, 'w') as f:
                            f.write("\n".join(yolo_labels))
            except Exception as e:
                print(f"\n处理 {image_filename} 时发生错误 (增强版本 {i}): {e}")

    print("\n数据增强流程全部完成！")
    print(f"结果已保存在: {os.path.abspath(output_dir)}")


def main():
    parser = argparse.ArgumentParser(description="Intelligent YOLO Detection/Segmentation Data Augmentation Script")
    parser.add_argument('--input', type=str, required=True,
                        help='Path to the input dataset directory.')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to the directory to save the augmented dataset.')
    parser.add_argument('--num', type=int, default=2,
                        help='Number of augmented versions to generate per original image.')

    args = parser.parse_args()
    augment_dataset(args.input, args.output, args.num)


if __name__ == "__main__":
    main()