# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
视频批量抽帧脚本 (多进程版)
Video batch frame extraction script (Multiprocessing)

usage:
    python extract_frames_mp.py \
                    --input path/to/input_folder \
                    --output path/to/output_folder \
                    [--prefix '']
                    [--interval 60]
                    [--start 1]
                    [--processes 4]

folder:
    .
    └── PATH_TO_input_folder
            ├── 001.mp4
            ├── 002.avi
            └── ...
    └── PATH_TO_output_folder
            ├── 0001.jpg
            ├── 0002.jpg
            └── ...
"""

import cv2
import os
import argparse
import multiprocessing
from tqdm import tqdm


def extract_frames(video_path, output_dir, prefix, interval, start_count, process_id):
    """
    从视频中提取帧并保存为图像文件 (进程安全版本)

    参数:
        video_path: 视频文件路径
        output_dir: 输出目录路径
        prefix: 输出文件名前缀
        interval: 抽帧间隔（每多少帧提取一帧）
        start_count: 起始保存计数
        process_id: 进程ID (用于标识)
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[进程 {process_id}] 错误：无法打开视频文件 {video_path}")
        return start_count

    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[进程 {process_id}] 视频信息:")
    print(f"[进程 {process_id}]   路径: {video_path}")
    print(f"[进程 {process_id}]   总帧数: {total_frames}")
    print(f"[进程 {process_id}]   帧率: {fps:.2f} fps")
    print(f"[进程 {process_id}]   分辨率: {width}x{height}")
    print(f"[进程 {process_id}]   抽帧间隔: 每 {interval} 帧提取一帧")
    print(f"[进程 {process_id}]   " + "-" * 30)

    # 逐帧处理
    frame_count = 0
    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 按间隔保存帧
        if frame_count % interval == 0:
            filename = f"{prefix}{start_count + saved_count:04d}.jpg"
            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, frame)
            saved_count += 1

        # 更新进度
        frame_count += 1
        if frame_count % 100 == 0:  # 每100帧更新一次进度
            progress = frame_count / total_frames * 100
            print(f"[进程 {process_id}] 处理进度: [{frame_count}/{total_frames}] {progress:.1f}%", end='\r')

    # 释放资源
    cap.release()
    print(f"\n[进程 {process_id}] 抽帧完成！共保存 {saved_count} 张图像")
    print(f"[进程 {process_id}] 输出目录: {output_dir}")
    print(f"[进程 {process_id}] " + "-" * 40)

    return saved_count  # 返回本视频保存的图像数量


def process_video(args):
    """
    处理单个视频的包装函数，用于多进程
    """
    video_path, output_dir, prefix, interval, start_count, process_id = args
    try:
        saved_count = extract_frames(video_path, output_dir, prefix, interval, start_count, process_id)
        return saved_count
    except Exception as e:
        print(f"[进程 {process_id}] 处理视频 {video_path} 时出错: {str(e)}")
        return 0


def list_video_files(directory):
    """
    递归获取文件夹中所有视频文件路径

    参数:
        directory: 要搜索的目录路径

    返回:
        视频文件路径列表
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.mpg', '.mpeg', '.rmvb', '.3gp']
    video_files = []

    # 使用os.walk递归遍历目录
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(root, file))
    return video_files


if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='视频抽帧工具（多进程版）')
    parser.add_argument('--input', type=str, required=True, help='输入文件夹路径，包含视频文件')
    parser.add_argument('--output', type=str, required=True, help='基础输出目录路径')
    parser.add_argument('--prefix', type=str, default='', help='输出文件名前缀')
    parser.add_argument('--interval', type=int, default=60, help='抽帧间隔（每多少帧提取一帧）')
    parser.add_argument('--start', type=int, default=1, help='保存起始数字')
    parser.add_argument('--processes', type=int, default=4, help=f'进程数量 (默认: CPU核心数 {multiprocessing.cpu_count()})')

    args = parser.parse_args()

    # 获取所有视频文件
    video_files = list_video_files(args.input)
    total_videos = len(video_files)

    if total_videos == 0:
        print(f"在目录 {args.input} 中未找到视频文件")
        exit()

    print(f"发现 {total_videos} 个视频文件，使用 {args.processes} 个进程开始批量处理...")
    print("=" * 60)

    # 准备任务参数
    tasks = []
    current_count = args.start
    for i, video_path in enumerate(video_files):
        # 为每个视频分配唯一的起始计数
        tasks.append((video_path, args.output, args.prefix, args.interval, current_count, i))
        # 预估本视频将生成的图像数量 (实际数量可能不同)
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            estimated_images = total_frames // args.interval
            current_count += estimated_images

    # 创建进程池
    pool = multiprocessing.Pool(processes=args.processes)

    # 使用tqdm显示进度条
    total_images = 0
    with tqdm(total=total_videos, desc="处理视频进度") as pbar:
        for result in pool.imap_unordered(process_video, tasks):
            total_images += result
            pbar.update(1)

    # 关闭进程池
    pool.close()
    pool.join()

    print("\n" + "=" * 60)
    print(f"所有视频处理完成！共处理 {total_videos} 个视频")
    print(f"总输出目录: {args.output}")
    print(f"总生成图像数量: {total_images}")