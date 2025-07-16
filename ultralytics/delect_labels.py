# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
在输入文件夹内删除指定label外的所有其他labels，若txt标签文件内没有指定label则将image和txt标签文件一并删除
Delete all other labels except the specified label in the input folder,
and delete the image and txt label files together if no label is specified in the txt label file

usage:
    python delect_labels.py \
                    --input /yolo/dataset1/hat_and_clothes_v2 \
                    --labels [0] \
                    [--correct]

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

"""

import argparse
import os
import shutil
import json
from pathlib import Path


def parse_arguments():
    parser = argparse.ArgumentParser(description="Filter and remove labels except specified ones")
    parser.add_argument("--input", type=str, required=True,
                        help="Input directory containing images/ and labels/")
    parser.add_argument("--labels", type=str, required=True,
                        help="List of labels to keep e.g. [0,2]")
    parser.add_argument("--correct", action="store_true",
                        help="Relabel kept classes sequentially")
    return parser.parse_args()


def main():
    args = parse_arguments()

    # 转换标签字符串为整数列表
    try:
        keep_labels = json.loads(args.labels.replace("'", "\""))
        if not isinstance(keep_labels, list):
            raise ValueError("Labels must be a list")
        keep_labels = [int(x) for x in keep_labels]
    except Exception as e:
        print(f"Error parsing labels: {e}")
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

    # 创建标签映射（如果需要重新编号）
    label_mapping = {old: new for new, old in enumerate(keep_labels)} if args.correct else None

    # 用于统计的变量
    total_files = 0
    modified_files = 0
    removed_files = 0
    kept_labels = 0
    removed_labels = 0

    # 处理每个标签文件
    for label_file in label_dir.glob("*.txt"):
        total_files += 1

        # 查找对应的图片文件
        image_file = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]:
            possible_image = image_dir / f"{label_file.stem}{ext}"
            if possible_image.exists():
                image_file = possible_image
                break

        # 如果没有找到图片文件，跳过此标签文件
        if not image_file:
            print(f"Warning: Missing image for {label_file.stem}")
            continue

        # 处理标签
        new_lines = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue

                try:
                    cls_id = int(parts[0])
                    if cls_id in keep_labels:
                        # 更新标签ID（如果需要重新编号）
                        if label_mapping is not None:
                            parts[0] = str(label_mapping[cls_id])
                        new_lines.append(" ".join(parts))
                        kept_labels += 1
                    else:
                        removed_labels += 1
                except ValueError:
                    # 跳过无效行
                    continue

        # 如果有保留的标签，更新标签文件
        if new_lines:
            modified_files += 1
            with open(label_file, 'w') as f:
                f.write("\n".join(new_lines))
        # 如果没有保留的标签，删除标签文件和对应的图片
        else:
            removed_files += 1
            try:
                # 删除标签文件
                label_file.unlink()
                print(f"Removed label file: {label_file.name}")

                # 删除图片文件
                if image_file.exists():
                    image_file.unlink()
                    print(f"Removed image file: {image_file.name}")
            except Exception as e:
                print(f"Error deleting files for {label_file.stem}: {e}")

    # 打印统计信息
    print("\nProcessing complete!")
    print(f"Total files processed: {total_files}")
    print(f"Files modified: {modified_files}")
    print(f"Files removed: {removed_files}")
    print(f"Labels kept: {kept_labels}")
    print(f"Labels removed: {removed_labels}")
    if args.correct:
        print(f"Label mapping applied: {label_mapping}")


if __name__ == "__main__":
    main()