# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
视频批量抽帧脚本
Video batch frame extraction script

usage:
    python extract_frames.py \
                    --input path/to/input_folder \
                    -- output path/to/output_folder \
                    [--prefix '' \]
                    [--interval 60]
                    [--start 1 \]

folder:
    .
    └── PATH_TO_input_folder
            ├── 001.mp4
            ├── 002.avi
            └── ...
    └── PATH_TO_output_folder
            ├── 1.jpg
            ├── 2.jpg
            └── ...
"""

import cv2
import os
import argparse


def extract_frames(video_path, output_dir, prefix='frame', interval=1, saved_count=0):
    """
    从视频中提取帧并保存为图像文件

    参数:
        video_path: 视频文件路径
        output_dir: 输出目录路径
        prefix: 输出文件名前缀
        interval: 抽帧间隔（每多少帧提取一帧）
        saved_count: 起始保存计数
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {video_path}")
        return saved_count  # 返回当前计数

    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n视频信息:")
    print(f"  路径: {video_path}")
    print(f"  总帧数: {total_frames}")
    print(f"  帧率: {fps:.2f} fps")
    print(f"  分辨率: {width}x{height}")
    print(f"  抽帧间隔: 每 {interval} 帧提取一帧")
    print("-" * 40)

    # 逐帧处理
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 按间隔保存帧
        if frame_count % interval == 0:
            filename = f"{prefix}{saved_count:04d}.jpg"
            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, frame)
            saved_count += 1

        # 更新进度显示
        progress = (frame_count + 1) / total_frames * 100
        print(f"处理进度: [{frame_count + 1}/{total_frames}] {progress:.1f}%", end='\r')
        frame_count += 1

    # 释放资源
    cap.release()
    print(f"\n抽帧完成！共保存 {saved_count} 张图像")
    print(f"输出目录: {output_dir}")
    print("-" * 40)

    return saved_count  # 返回当前计数


def list_video_files(directory):
    """
    递归获取文件夹中所有视频文件路径
    [1,6,8](@ref)

    参数:
        directory: 要搜索的目录路径

    返回:
        视频文件路径列表
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.mpg', '.mpeg', '.rmvb', '.3gp']
    video_files = []

    # 使用os.walk递归遍历目录[2,3,4](@ref)
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(root, file))
    return video_files


if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='视频抽帧工具（支持文件夹批量处理）')
    parser.add_argument('--input', type=str, required=True, help='输入文件夹路径，包含视频文件')
    parser.add_argument('--output', type=str, required=True, help='基础输出目录路径')
    parser.add_argument('--prefix', type=str, default='', help='输出文件名前缀')
    parser.add_argument('--interval', type=int, default=60, help='抽帧间隔（每多少帧提取一帧）')
    parser.add_argument('--start', type=int, default=1, help='保存起始数字')

    args = parser.parse_args()

    # 获取所有视频文件[1,2,6](@ref)
    video_files = list_video_files(args.input)
    total_videos = len(video_files)

    if total_videos == 0:
        print(f"在目录 {args.input} 中未找到视频文件")
        exit()

    print(f"发现 {total_videos} 个视频文件，开始批量处理...")
    print("=" * 60)

    current_count = args.saved_count
    # 处理每个视频文件
    for idx, video_path in enumerate(video_files):
        print(f"\n处理视频 {idx + 1}/{total_videos}: {video_path}")

        # 为每个视频创建单独的输出子目录[1](@ref)
        # video_filename = os.path.basename(video_path)
        # video_name = os.path.splitext(video_filename)[0]
        output_dir = args.output

        # 调用抽帧函数
        try:
            current_count = extract_frames(
                video_path=video_path,
                output_dir=output_dir,
                prefix=args.prefix,
                interval=args.interval,
                saved_count=current_count  # 使用累计计数
            )
        except Exception as e:
            print(f"处理视频 {video_path} 时出错: {str(e)}")

    print("\n" + "=" * 60)
    print(f"所有视频处理完成！共处理 {total_videos} 个视频")
    print(f"总输出目录: {args.output}")