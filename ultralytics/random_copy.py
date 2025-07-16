# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
随机复制一定比例输入文件夹内的图像和标签到到输出文件夹内
Randomly copy a certain percentage of images and labels from the input folder to the output folder

usage:
    python random_copy.py \
                    --input /yolo/dataset1/hat_and_clothes \
                    --output /yolo/dataset1/hat_and_clothes_v2 \
                    [--percent 20]

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
            ├── 1.jpg
            ├── 2.jpg
            └── ...
        ├── labels
            ├── 1.txt
            ├── 2.txt
            └── ...

"""

import argparse
import os
import shutil
import random
from pathlib import Path


def parse_arguments():
    parser = argparse.ArgumentParser(description="Randomly copy a percentage of images and labels")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing images/ and labels/")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--percent", type=int, default=20,
                        help="Percentage of files to copy (default: 20)")
    return parser.parse_args()


def main():
    args = parse_arguments()

    # 验证百分比范围
    if args.percent <= 0 or args.percent > 100:
        print("Error: Percent must be between 1 and 100")
        return

    # 验证输入目录结构
    input_path = Path(args.input)
    image_dir = input_path / "images"
    label_dir = input_path / "labels"

    if not image_dir.exists() or not image_dir.is_dir():
        print(f"Error: Missing images directory in {input_path}")
        return
    if not label_dir.exists() or not label_dir.is_dir():
        print(f"Error: Missing labels directory in {input_path}")
        return

    # 创建输出目录
    output_path = Path(args.output)
    output_image_dir = output_path / "images"
    output_label_dir = output_path / "labels"

    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    # 收集所有图片文件（支持多种格式）
    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.webp"]:
        image_files.extend(image_dir.glob(ext))

    if not image_files:
        print("Error: No image files found in the input directory")
        return

    # 计算需要复制的文件数量
    total_files = len(image_files)
    copy_count = max(1, int(total_files * args.percent / 100))

    print(f"Found {total_files} images in input directory")
    print(f"Copying {copy_count} random files ({args.percent}%) to output directory")

    # 随机选择文件
    selected_files = random.sample(image_files, copy_count)

    # 复制选中的文件和对应的标签
    copied_count = 0
    for image_file in selected_files:
        # 获取对应的标签文件路径
        label_file = label_dir / f"{image_file.stem}.txt"

        # 确保标签文件存在
        if not label_file.exists():
            print(f"Warning: Missing label file for {image_file.name}")
            continue

        try:
            # 复制图片
            shutil.copy(image_file, output_image_dir / image_file.name)
            # 复制标签
            shutil.copy(label_file, output_label_dir / label_file.name)
            copied_count += 1
        except Exception as e:
            print(f"Error copying {image_file.name}: {e}")

    print(f"Successfully copied {copied_count} image-label pairs")
    print(f"Output directory: {output_path.resolve()}")


if __name__ == "__main__":
    main()