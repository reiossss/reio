# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
tif图像目标检测脚本，读取config.yaml文件获取参数划分网格实现多线程或批处理目标检测推理
TIF image object detection script that reads the config.yaml file
to obtain parameters, divides the grid, and implements multiprocessing object detection inference.

usage:
    python tif_grid.py

config.yaml:
    # ------------------- #
    #  文件与目录路径
    # ------------------- #
    # 输入的大型TIF文件
    input_tif: 'D:/downlord/p3.tif'
    # 输出目录，用于存放最终结果
    output_dir: 'D:/downlord/results'
    # 目标检测模型的.pt文件路径 (例如YOLOv5, v7, v8)
    model_path: 'models/blue_v1.pt'

    # ------------------- #
    #  切片参数
    # ------------------- #
    # 切片尺寸 (正方形)
    tile_size: 640
    # 切片间的重叠像素
    overlap: 16

    # ------------------- #
    #  模型推理参数
    # ------------------- #
    # GPU设备ID (如果使用CPU，则设为 'cpu')
    device: 'cuda:0' # 或者 'cpu'
    # 目标检测的置信度阈值
    conf_threshold: 0.4
    # NMS (非极大值抑制) 的IOU阈值
    iou_threshold: 0.5
    # 推理时的图像尺寸（应与模型训练尺寸匹配）
    inference_img_size: 640

    # 使用的进程数 (0 表示使用所有可用的CPU核心)
    num_workers: 4

    # 使用批处理
    batch_size: 16

"""

import os
import yaml
import time
import torch
import rasterio
import numpy as np
import cv2
import logging
from ultralytics import YOLO
from rasterio.windows import Window
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import torchvision


# --- 配置日志系统 --- #
def setup_logger(output_dir):
    """配置并返回logger对象"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # 创建日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 文件处理器
    log_file = os.path.join(output_dir, "detected.txt")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


# --- 打印配置信息 --- #
def print_config(config, logger):
    """打印配置信息"""
    logger.info("=" * 50)
    logger.info("检测配置参数:")
    for key, value in config.items():
        # 跳过可能的大对象
        if isinstance(value, (list, dict)) and len(str(value)) > 100:
            logger.info(f"{key}: [数据过长不显示]")
        else:
            logger.info(f"{key}: {value}")
    logger.info("=" * 50)


# --- 工作进程函数 --- #
def process_tile(args):
    """工作进程函数，用于在单个切片上执行推理"""
    tile_window, config = args
    model = None
    all_detections = []

    try:
        # 在进程内加载模型
        device = torch.device(config['device'])
        model = YOLO(config['model_path'])
        model.to(device)
        class_names = model.names

        # 读取切片数据
        with rasterio.open(config['input_tif']) as src:
            tile_data = src.read(window=tile_window)

        # 图像预处理
        img = np.moveaxis(tile_data, 0, -1)

        # 处理多通道图像
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif img.shape[2] > 4:
            img = img[:, :, :3]  # 只取前三个通道

        # 归一化到uint8
        if img.dtype != np.uint8:
            p2, p98 = np.percentile(img, (2, 98))
            img = np.clip((img - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)

        # 模型推理
        results = model(img, verbose=False)
        dets = results[0].boxes.data

        if dets is not None and len(dets):
            for *xyxy, conf, cls in reversed(dets):
                if conf < config['conf_threshold']:
                    continue

                # 转换到全局坐标
                x_min_global = int(xyxy[0]) + tile_window.col_off
                y_min_global = int(xyxy[1]) + tile_window.row_off
                x_max_global = int(xyxy[2]) + tile_window.col_off
                y_max_global = int(xyxy[3]) + tile_window.row_off

                label_index = int(cls)
                label_name = class_names[label_index]

                all_detections.append([
                    x_min_global, y_min_global, x_max_global, y_max_global,
                    float(conf), label_name
                ])

    except Exception as e:
        import traceback
        error_msg = f"进程 {os.getpid()} 在处理切片 {tile_window} 时出错: {e}\n{traceback.format_exc()}"
        return all_detections, error_msg

    return all_detections, None


def batch_inference(windows, config):
    """单进程批量处理模式（使用 YOLO 官方批量推理方法）"""
    device = torch.device(config['device'])
    model = YOLO(config['model_path']).to(device)
    class_names = model.names
    all_detections = []

    # 批量处理窗口
    for i in tqdm(range(0, len(windows), config['batch_size']), desc="批处理推理"):
        batch_windows = windows[i:i + config['batch_size']]
        batch_images = []
        batch_metas = []  # 存储每个图像的元数据

        # 读取批次数据
        with rasterio.open(config['input_tif']) as src:
            for window in batch_windows:
                tile_data = src.read(window=window)
                # 图像预处理
                img = np.moveaxis(tile_data, 0, -1)

                # 处理多通道图像
                if img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                elif img.shape[2] > 4:
                    img = img[:, :, :3]  # 只取前三个通道

                # 归一化到uint8
                if img.dtype != np.uint8:
                    p2, p98 = np.percentile(img, (2, 98))
                    img = np.clip((img - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)

                # 记录窗口位置信息
                meta = {
                    'orig_shape': img.shape[:2],
                    'window': window
                }
                batch_images.append(img)
                batch_metas.append(meta)

        # 使用 YOLO 的批量推理接口
        results = model(batch_images, verbose=False)

        # 处理结果
        for j, (result, meta) in enumerate(zip(results, batch_metas)):
            dets = result.boxes.data
            window = meta['window']
            orig_shape = meta['orig_shape']

            if dets is not None and len(dets):
                for *xyxy, conf, cls in reversed(dets):
                    if conf < config['conf_threshold']:
                        continue

                    # 转换到全局坐标
                    x_min_global = int(xyxy[0]) + window.col_off
                    y_min_global = int(xyxy[1]) + window.row_off
                    x_max_global = int(xyxy[2]) + window.col_off
                    y_max_global = int(xyxy[3]) + window.row_off

                    label_index = int(cls)
                    label_name = class_names[label_index]

                    all_detections.append([
                        x_min_global, y_min_global, x_max_global, y_max_global,
                        float(conf), label_name
                    ])

    return all_detections


# --- 主函数 --- #
def main(config):
    """主函数，协调整个推理流程"""
    # 设置日志
    os.makedirs(config['output_dir'], exist_ok=True)
    logger = setup_logger(config['output_dir'])
    print_config(config, logger)

    # 1. 生成切片网格
    logger.info("步骤 1/5: 计算切片网格...")
    step = config['tile_size'] - config['overlap']
    windows = []

    with rasterio.open(config['input_tif']) as src:
        image_width, image_height = src.width, src.height
        meta = src.meta.copy()
        for y_off in range(0, image_height, step):
            for x_off in range(0, image_width, step):
                width = min(config['tile_size'], image_width - x_off)
                height = min(config['tile_size'], image_height - y_off)
                windows.append(Window(x_off, y_off, width, height))

    logger.info(f"完成。共找到 {len(windows)} 个切片。")

    # 2. 多进程推理
    logger.info(f"步骤 2/5: 使用 {config['num_workers']} 个进程进行并行推理...")
    all_detections_flat = []
    errors = []
    tasks = [(window, config) for window in windows]

    with Pool(processes=config['num_workers']) as pool:
        results = list(tqdm(pool.imap(process_tile, tasks), total=len(tasks), desc="推理进度"))

    for detections, error in results:
        if error:
            errors.append(error)
        if detections:
            all_detections_flat.extend(detections)

    # 记录错误信息
    if errors:
        logger.error(f"处理过程中遇到 {len(errors)} 个错误:")
        for error in errors:
            logger.error(error)

    # # 在 main() 中替换多进程部分
    # logger.info("使用单进程批处理模式...")
    # all_detections_flat = batch_inference(windows, config)

    logger.info(f"完成。所有切片共检测到 {len(all_detections_flat)} 个初步目标。")

    # 3. 全局非极大值抑制 (NMS)
    logger.info("步骤 3/5: 执行全局非极大值抑制 (NMS)...")
    if not all_detections_flat:
        logger.info("未检测到任何目标，流程结束。")
        return

    boxes = torch.tensor([det[:4] for det in all_detections_flat], dtype=torch.float32)
    scores = torch.tensor([det[4] for det in all_detections_flat], dtype=torch.float32)
    indices = torchvision.ops.nms(boxes, scores, config['iou_threshold'])
    final_detections = [all_detections_flat[i] for i in indices]
    logger.info(f"完成。NMS后剩余 {len(final_detections)} 个唯一目标。")

    # 4. 在原图上绘制最终结果
    logger.info("步骤 4/5: 在原图上绘制最终边界框并保存检测目标结果图...")
    with rasterio.open(config['input_tif']) as src:
        output_image = np.moveaxis(src.read(), 0, -1).copy()

        # 处理多通道图像
        if output_image.shape[2] == 4:
            output_image = cv2.cvtColor(output_image, cv2.COLOR_RGBA2RGB)
        elif output_image.shape[2] > 4:
            output_image = output_image[:, :, :3]  # 只取前三个通道

        # 归一化到uint8
        if output_image.dtype != np.uint8:
            p2, p98 = np.percentile(output_image, (2, 98))
            output_image = np.clip((output_image - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)

        count = 1
        output_filename = os.path.basename(config['input_tif'])
        image_dir = os.path.join(config['output_dir'], f"{os.path.splitext(output_filename)[0]}")
        os.makedirs(image_dir, exist_ok=True)

        for x1, y1, x2, y2, conf, label in tqdm(final_detections, desc="绘制边界框"):
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

            cropped_img = output_image[(y1 - 50) : (y2 + 50), (x1 - 50) : (x2 + 50)]
            cropped_img= cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)
            # x0, y0 = (x1 + x2) / 2, (y1 + y2) / 2
            cut_image = os.path.join(image_dir, f"{count}.png")
            count += 1
            cv2.imwrite(cut_image, cropped_img)

            cv2.rectangle(output_image, (x1 - 50, y1 - 50), (x2 + 50, y2 + 50), (255, 53, 94), 4)
            label_text = f"{label}: {conf:.2f}"
            cv2.putText(output_image, label_text, (x1 - 50, y1 - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 53, 94), 4)

    logger.info("完成。")

    # 5. 保存结果
    logger.info("步骤 5/5: 保存带有检测结果的TIF图像...")
    output_image_raster = np.moveaxis(output_image, -1, 0)

    # 更新元数据
    meta.update({
        'driver': 'GTiff',
        'dtype': 'uint8',
        'count': 3,
        'compress': 'lzw'
    })

    output_filename = os.path.basename(config['input_tif'])
    output_filepath = os.path.join(config['output_dir'], f"{os.path.splitext(output_filename)[0]}_detected.tif")

    with rasterio.open(output_filepath, 'w', **meta) as dst:
        dst.write(output_image_raster)

    logger.info(f"处理成功！最终结果已保存至: {output_filepath}")


if __name__ == '__main__':
    start_time = time.time()

    # 加载配置文件
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("错误: `config.yaml` 未找到。请根据模板创建并配置该文件。")
        exit()
    except Exception as e:
        print(f"加载配置文件时出错: {e}")
        exit()

    # 动态设置进程数
    if config.get('num_workers', 0) == 0:
        config['num_workers'] = cpu_count()

    # 设置默认值
    config.setdefault('conf_threshold', 0.25)
    config.setdefault('iou_threshold', 0.45)
    config.setdefault('overlap', 16)

    # 检查输入文件
    if not os.path.exists(config['input_tif']):
        print(f"错误: 输入的TIF文件未找到 -> {config['input_tif']}")
        exit()

    main(config)

    end_time = time.time()
    print(f"总耗时: {(end_time - start_time):.2f} s")