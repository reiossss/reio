# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
yolo检测框数据增强脚本，实现旋转、平移、缩放、左右翻转、亮度、饱和度等一系列随机混合增强，默认使数据集扩充到原有的3倍左右。
The YOLO detection box data augmentation script implements a series of random mixed enhancements
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
import albumentations as A
from tqdm import tqdm


def get_augmentation_pipeline():
    """
    定义并返回一个数据增强流程。
    Albumentations 可以同时转换图像和其对应的边界框。
    """
    return A.Compose([
        # 颜色和亮度变换
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2, p=0.75),
        A.RandomBrightnessContrast(p=0.5),

        # 几何变换
        A.HorizontalFlip(p=0.8),
        # Affine 变换集成了平移、缩放（拉伸）和旋转
        A.Affine(
            scale=(0.8, 1.2),  # 图像缩放范围（80% - 120%）
            translate_percent=(-0.1, 0.1),  # 平移范围
            rotate=(-30, 30),  # 旋转范围 (-30 到 30 度)
            p=0.75
        ),
    ], bbox_params=A.BboxParams(
        format='yolo',  # 关键！指定边界框是 YOLO 格式
        label_fields=['class_labels']  # 关联类别标签
    ))


def augment_dataset(input_dir, output_dir, num_aug_per_image=2):
    """
    对整个数据集进行数据增强。

    Args:
        input_dir (str): 输入数据集的路径。
        output_dir (str): 输出增强后数据集的路径。
        num_aug_per_image (int): 每张原始图像要生成的增强版本数量。
    """
    # 1. 设置路径
    input_images_dir = os.path.join(input_dir, 'images')
    input_labels_dir = os.path.join(input_dir, 'labels')
    output_images_dir = os.path.join(output_dir, 'images')
    output_labels_dir = os.path.join(output_dir, 'labels')

    # 2. 创建输出目录
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    print("Step 1: 正在复制原始文件到输出目录...")
    # 3. 复制原始文件
    for subdir, output_subdir in [(input_images_dir, output_images_dir), (input_labels_dir, output_labels_dir)]:
        for filename in tqdm(os.listdir(subdir), desc=f"复制 {os.path.basename(subdir)}"):
            shutil.copy2(os.path.join(subdir, filename), os.path.join(output_subdir, filename))
    print("原始文件复制完成！\n")

    # 4. 获取数据增强管道和图像列表
    transform = get_augmentation_pipeline()
    image_files = [f for f in os.listdir(input_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"Step 2: 开始为 {len(image_files)} 张图像生成增强数据...")
    # 5. 循环处理每张图像
    for image_filename in tqdm(image_files, desc="生成增强数据中"):
        base_name, ext = os.path.splitext(image_filename)
        image_path = os.path.join(input_images_dir, image_filename)
        label_path = os.path.join(input_labels_dir, base_name + '.txt')

        # 读取图像 (Albumentations 需要 RGB 格式)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 读取 YOLO 标签
        bboxes = []
        class_labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        coords = [float(p) for p in parts[1:]]
                        bboxes.append(coords)
                        class_labels.append(class_id)

        # 为每张图片生成 N 个增强版本
        for i in range(num_aug_per_image):
            try:
                # 应用变换
                augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                aug_image = augmented['image']
                aug_bboxes = augmented['bboxes']

                # 如果增强后所有框都消失了，则跳过此次保存
                if not aug_bboxes:
                    continue

                # 生成新的文件名
                aug_base_name = f"{base_name}_aug_{i}"
                aug_image_path = os.path.join(output_images_dir, aug_base_name + ext)
                aug_label_path = os.path.join(output_labels_dir, aug_base_name + '.txt')

                # 保存增强后的图像 (转回 BGR 以便 cv2 保存)
                cv2.imwrite(aug_image_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))

                # 保存增强后的标签
                with open(aug_label_path, 'w') as f:
                    for bbox, class_id in zip(aug_bboxes, augmented['class_labels']):
                        # bbox 格式为 [x_center, y_center, width, height]
                        line = f"{class_id} {' '.join(f'{c:.6f}' for c in bbox)}\n"
                        f.write(line)
            except Exception as e:
                print(f"处理 {image_filename} 时出错 (增强版本 {i}): {e}")

    print("\n数据增强流程全部完成！")
    print(f"结果已保存在: {os.path.abspath(output_dir)}")


def main():
    parser = argparse.ArgumentParser(description="YOLO Detection Dataset Augmentation Script")
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