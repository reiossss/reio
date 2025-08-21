# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
训练集和验证集划分脚本
Training set and validation set division scripts

usage:
    python split_train_val_v3.py \
                    --input /yolo/dataset1/HatAndClothes \
                    [--ratio 0.8]

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

import os
import random
import argparse


def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='划分训练集和验证集')
    parser.add_argument('--input', type=str, required=True,
                        help='数据集图片所在的目录')
    parser.add_argument('--ratio', type=float, default=0.8,
                        help='训练集划分比例 (默认: 0.8)')

    args = parser.parse_args()

    # 检查划分比例是否在有效范围内
    if args.ratio <= 0 or args.ratio >= 1:
        print("错误：划分比例必须在0到1之间")
        return

    # 检查输入目录是否存在
    if not os.path.exists(args.input):
        print(f"错误：输入目录不存在 - {args.input}")
        return

    # 确定输出文件路径（在输入目录内）
    train_txt = os.path.join(args.input, "train_list.txt")
    val_txt = os.path.join(args.input, "val_list.txt")

    images_dir = os.path.join(args.input, "images")

    # 获取所有图片的文件名
    image_files = [f for f in os.listdir(images_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'))]

    if not image_files:
        print(f"错误：在目录中未找到图片文件 - {images_dir}")
        return

    num_images = len(image_files)

    # 打乱顺序
    random.shuffle(image_files)

    # 划分训练集和验证集
    num_train = int(args.ratio * num_images)
    train_images = image_files[:num_train]
    val_images = image_files[num_train:]

    # 将训练集和验证集写入各自的文件
    with open(train_txt, 'w') as train_f, open(val_txt, 'w') as val_f:
        for img in train_images:
            # 写入图片的完整路径
            train_f.write(os.path.join(images_dir, img) + '\n')
        for img in val_images:
            val_f.write(os.path.join(images_dir, img) + '\n')

    print(f"数据集划分完成:")
    print(f"  图片总数: {num_images}")
    print(f"  训练集图片: {len(train_images)} ({len(train_images) / num_images:.1%})")
    print(f"  验证集图片: {len(val_images)} ({len(val_images) / num_images:.1%})")


if __name__ == "__main__":
    main()