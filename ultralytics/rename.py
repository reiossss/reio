# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
批量重命名文件夹内的图片及标签文件名称，按数字顺序排序，图片后缀不变
Rename the tag file names of the pictures in the folder in batches,
sort them in numerical order, and keep the image suffix unchanged

usage:
    python rename.py \
                    --input path/to/input_folder \
                    [--prefix '' \]
                    [--start 1 \]
                    [--digits 3 \]

folder:
    .
    └── PATH_TO_input_folder
        ├── images
            ├── 001.jpg
            ├── 002.jpg
            └── ...
        ├── labels
            ├── 001.txt
            ├── 002.txt
            └── ...
"""

import os
import argparse
import re


def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='批量重命名图像和标签文件为数字序列')
    parser.add_argument('--input', type=str, required=True,
                        help='包含images和labels子目录的父文件夹路径')
    parser.add_argument('--prefix', type=str, default='',
                        help='文件名前缀（可选）')
    parser.add_argument('--start', type=int, default=1,
                        help='起始编号（默认：1）')
    parser.add_argument('--digits', type=int, default=3,
                        help='数字位数（默认：3）')

    args = parser.parse_args()

    # 验证参数
    if not os.path.exists(args.input) or not os.path.isdir(args.input):
        print(f"错误：输入目录不存在或不是文件夹 - {args.input}")
        return

    if args.digits < 1:
        print("错误：数字位数必须大于0")
        return

    if args.start < 0:
        print("错误：起始编号必须是非负数")
        return

    # 检查images和labels子目录
    image_dir = os.path.join(args.input, 'images')
    label_dir = os.path.join(args.input, 'labels')

    if not os.path.exists(image_dir) or not os.path.isdir(image_dir):
        print(f"错误：图片目录不存在或不是文件夹 - {image_dir}")
        return

    # labels目录不存在时只处理图片
    process_labels = True
    if not os.path.exists(label_dir) or not os.path.isdir(label_dir):
        print(f"警告：标签目录不存在，将只处理图片 - {label_dir}")
        process_labels = False

    # 获取所有图片文件
    image_files = []
    for f in os.listdir(image_dir):
        full_path = os.path.join(image_dir, f)
        if os.path.isfile(full_path):
            # 检查常见图片扩展名
            if re.search(r'\.(jpg|jpeg|png|bmp|tiff|webp|gif)$', f, re.IGNORECASE):
                image_files.append(f)

    if not image_files:
        print(f"错误：在图片目录中未找到图片文件 - {image_dir}")
        return

    # 按文件名排序
    image_files.sort()

    # 计数器
    count = args.start

    # 重命名文件
    renamed_image_count = 0
    renamed_label_count = 0
    skipped_images = []
    skipped_labels = []

    for old_image_name in image_files:
        # 获取图片文件扩展名
        _, img_ext = os.path.splitext(old_image_name)
        img_ext = img_ext.lower()  # 统一使用小写扩展名

        # 获取对应的标签文件名（不带路径）
        label_name = os.path.splitext(old_image_name)[0] + '.txt'
        has_label = False

        # 检查标签文件是否存在
        if process_labels:
            label_path = os.path.join(label_dir, label_name)
            has_label = os.path.exists(label_path) and os.path.isfile(label_path)

        # 生成新文件名
        new_base = f"{args.prefix}{count:0{args.digits}d}"
        new_image_name = new_base + img_ext
        new_label_name = new_base + '.txt'

        # 完整路径
        old_image_path = os.path.join(image_dir, old_image_name)
        new_image_path = os.path.join(image_dir, new_image_name)

        if process_labels and has_label:
            old_label_path = os.path.join(label_dir, label_name)
            new_label_path = os.path.join(label_dir, new_label_name)
        else:
            old_label_path = None
            new_label_path = None

        # 检查新文件名是否已存在
        skip_current = False

        # 检查图片文件是否已存在
        if os.path.exists(new_image_path):
            print(f"警告：图片文件 '{new_image_name}' 已存在，跳过 '{old_image_name}'")
            skipped_images.append(old_image_name)
            skip_current = True

        # 检查标签文件是否已存在
        if not skip_current and process_labels and has_label and os.path.exists(new_label_path):
            print(f"警告：标签文件 '{new_label_name}' 已存在，跳过 '{old_image_name}'")
            skipped_labels.append(label_name)
            skip_current = True

        if skip_current:
            continue

        try:
            # 重命名图片文件
            os.rename(old_image_path, new_image_path)
            renamed_image_count += 1
            print(f"重命名图片: {old_image_name} -> {new_image_name}")

            # 重命名对应的标签文件（如果存在）
            if process_labels and has_label:
                os.rename(old_label_path, new_label_path)
                renamed_label_count += 1
                print(f"重命名标签: {label_name} -> {new_label_name}")

            count += 1
        except Exception as e:
            print(f"重命名 {old_image_name} 失败: {e}")

    print(f"\n重命名完成!")
    print(f"处理图片总数: {len(image_files)}")
    print(f"成功重命名图片: {renamed_image_count}")
    print(f"成功重命名标签: {renamed_label_count}")
    print(f"起始编号: {args.start}")
    print(f"结束编号: {count - 1}")
    print(f"数字位数: {args.digits}")
    if args.prefix:
        print(f"文件名前缀: '{args.prefix}'")

    # 输出跳过信息
    if skipped_images:
        print(f"\n跳过的图片文件 ({len(skipped_images)}):")
        for img in skipped_images:
            print(f"  - {img}")
    if skipped_labels:
        print(f"\n跳过的标签文件 ({len(skipped_labels)}):")
        for lbl in skipped_labels:
            print(f"  - {lbl}")


if __name__ == "__main__":
    main()