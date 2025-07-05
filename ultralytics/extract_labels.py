# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
实现检索并提取输入文件夹下带有指定编号标签的数据集图片和txt标签并复制到输出文件夹下的脚本
Retrieve and extract dataset images and txt tags with the specified labels in the input folder
and copy them to the output folder.

usage:
    python extract_labels.py \
                    --input /yolo/dataset1/SODA \
                    --output /yolo/dataset1/hat_and_clothes \
                    --labels [0,1,2] \
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
import json
from pathlib import Path


def parse_arguments():
    parser = argparse.ArgumentParser(description="Extract images and labels with specified class IDs")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing images/ and labels/")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--labels", type=str, required=True, help="List of labels to extract e.g. [1],[0,2],[1,4,5]")
    parser.add_argument("--correct", action="store_true", default=False, help="Relabel classes sequentially e.g. [1,2,7] to [0,1,2]")
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Convert label string to list of integers
    try:
        label_list = json.loads(args.labels.replace("'", "\""))
        if not isinstance(label_list, list):
            raise ValueError("Labels must be a list")
        label_list = [int(x) for x in label_list]
    except Exception as e:
        print(f"Error parsing labels: {e}")
        return

    # Validate input directory structure
    input_path = Path(args.input)
    image_dir = input_path / "images"
    label_dir = input_path / "labels"

    if not image_dir.exists() or not image_dir.is_dir():
        print(f"Error: Missing images directory in {input_path}")
        return
    if not label_dir.exists() or not label_dir.is_dir():
        print(f"Error: Missing labels directory in {input_path}")
        return

    # Create output directories
    output_path = Path(args.output)
    output_image_dir = output_path / "images"
    output_label_dir = output_path / "labels"

    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    # Create mapping for label correction
    label_mapping = {old: new for new, old in enumerate(label_list)} if args.correct else None

    # Process each label file
    processed_count = 0
    for label_file in label_dir.glob("*.txt"):
        image_file = image_dir / f"{label_file.stem}.jpg"
        if not image_file.exists():
            # Try other common image formats
            for ext in [".png", ".jpeg", ".bmp", ".tiff"]:
                alt_image = image_dir / f"{label_file.stem}{ext}"
                if alt_image.exists():
                    image_file = alt_image
                    break
            else:
                print(f"Warning: Missing image for {label_file.name}")
                continue

        # Process labels
        new_labels = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue

                try:
                    cls_id = int(parts[0])
                    if cls_id in label_list:
                        if label_mapping is not None:
                            parts[0] = str(label_mapping[cls_id])
                        new_labels.append(" ".join(parts))
                except ValueError:
                    continue

        # Only copy files if labels were found
        if new_labels:
            # Copy image
            shutil.copy(image_file, output_image_dir / image_file.name)

            # Write processed labels
            with open(output_label_dir / label_file.name, 'w') as f:
                f.write("\n".join(new_labels))

            processed_count += 1

    print(f"Processed {processed_count} files")
    print(f"Extracted labels: {label_list}")
    if args.correct:
        print(f"Applied label mapping: {label_mapping}")


if __name__ == "__main__":
    main()