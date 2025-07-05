from typing import Optional, Dict, Tuple, Any
import gradio as gr
import cv2
import tempfile
import os
import numpy as np
from ultralytics import YOLO
import torch
from torchvision.ops import box_iou
from torchvision.ops import box_area
from collections import defaultdict
from ultralytics.engine.results import Results
from shapely.geometry import Polygon

class GrYoloPredict():
    """
    路面违停识别算法：
    一个简单基于gradio的YOLO11模型推理可视化页面，实现两个模型的目标检测推理。

    Attributes:
        model1：推理违停区域，为图像分割模型
        model2：推理车辆类型，为目标检测模型
        yolo_inference_for_examples：案例函数，yolo11官方案例
        compute_coverage_and_highlight：计算第二个模型检测框被第一个模型检测框覆盖的面积百分比，并标记高覆盖率的检测框
        merge_yolo_results：合并两个模型的推理结果
        yolo_inference：路面违停识别算法实现，标红重叠占比达到70%的检测框，并且统计违停时间。
                        若违停时间超过一定值则告警。一定时间内未出现违停情况则重置违停时间。
    """

    def __init__(self):
        """
        初始化全局变量
        """
        self.y_offset = 30  # 初始Y坐标偏移量
        self.current_time = 0.0
        self.tracking_data = {}
        self.violation_status = {}
        self.violation_text = ""
        self.track_history = defaultdict(lambda: [])


    def reset(self):
        """
        重置所有变量
        """
        self.current_time = 0.0
        self.tracking_data = {}
        self.violation_status = {}
        self.violation_text = ""
        self.track_history = defaultdict(lambda: [])


    def yolo_inference(self, image, video, model_id, image_size, conf_threshold):
        """
        路面违停识别算法实现，标红重叠占比达到70%的检测框，并且统计违停时间。
        若违停时间超过一定值则告警。一定时间内未出现违停情况则重置违停时间。

        Args:
            image(str):输入图像
            video(str):输入视频
            model_id(str):选择的模型名称
            image_size(int):检测图像或视频大小
            conf_threshold(float):置信度阈值

        Returns:图像或视频检测结果

        """
        model_path = os.path.join('models', model_id)
        model1 = YOLO('models/yellow_grid_v4.pt')
        model2 = YOLO(model_path)
        if image:
            results1 = model1.predict(source=image, imgsz=image_size, conf=conf_threshold, device=0)
            plot1 = results1[0].plot()  # 第一个模型结果

            results2 = model2.predict(source=plot1[:, :, ::-1], imgsz=image_size, conf=conf_threshold, device=0)

            # 计算覆盖率并获取高亮索引
            coverage_ratios, highlight_indices = self.compute_coverage_and_highlight(results1[0], results2[0])
            plot2 = results2[0].plot()  # 第二个模型结果

            # 在第二个模型的可视化结果上标红高IOU框
            if hasattr(results2[0], 'boxes'):
                boxes = results2[0].boxes.xyxy.cpu().numpy()
                for idx in highlight_indices:
                    box = boxes[idx].astype(int)
                    cv2.rectangle(plot2, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 3)  # 红色边框

            return plot2[:, :, ::-1], None
        else:
            temp_dir = "temp"  # 例如: "C:/my_temp" 或 "/tmp/my_custom_temp"
            os.makedirs(temp_dir, exist_ok=True)
            video_path = tempfile.mktemp(suffix=".webm", dir=temp_dir)
            with open(video_path, "wb") as f:
                with open(video, "rb") as g:
                    f.write(g.read())

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            output_video_path = tempfile.mktemp(suffix=".webm", dir=temp_dir)
            out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'vp80'), fps, (frame_width, frame_height))
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                self.current_time = frame_count / fps

                results1 = model1.predict(source=frame, imgsz=image_size, conf=conf_threshold, device=0)
                results2 = model2.predict(source=frame, imgsz=image_size, conf=conf_threshold, device=0)
                trace = model2.track(frame, persist=True)[0]

                # 计算覆盖率并获取高亮索引
                coverage_ratios, highlight_indices = self.compute_coverage_and_highlight(results1[0], results2[0])
                plot = self.plot_masks_only(results1[0])  # 第一个模型推理掩码结果

                # Get the boxes and track IDs
                if trace.boxes and trace.boxes.is_track:
                    boxes = trace.boxes.xywh.cpu()
                    track_ids = trace.boxes.id.int().cpu().tolist()
                    classes = results2[0].boxes.cls.cpu().tolist()

                    # Plot the tracks
                    for box, track_id, cls_id in zip(boxes, track_ids, classes):
                        # 初始化跟踪数据
                        if track_id not in self.tracking_data:
                            self.tracking_data[track_id] = {
                                'start_time': None,
                                'total_time': 0.0,
                                'last_active': 0.0,
                                'class_id': int(cls_id)
                            }

                        # 更新轨迹历史
                        if track_id not in self.track_history:
                            self.track_history[track_id] = []

                # 在第二个模型的可视化结果上标红高IOU框
                if hasattr(results2[0], 'boxes') and len(highlight_indices) > 0:
                    boxes = results2[0].boxes.xyxy.cpu().numpy()

                    for idx in highlight_indices:
                        if idx < len(boxes):
                            box = boxes[idx].astype(int)
                            track_id = trace.boxes.id[idx].item() if idx < len(trace.boxes) else -1

                            # 绘制红色边框
                            cv2.rectangle(plot, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 1)

                            # 更新时间统计
                            if track_id != -1 and track_id in self.tracking_data:
                                data = self.tracking_data[track_id]
                                if data['start_time'] is None:
                                    data['start_time'] = self.current_time
                                data['last_active'] = self.current_time
                                data['total_time'] = self.current_time - data['start_time']

                                # 检查违规条件
                                if data['total_time'] > 1:
                                    self.violation_status[track_id] = True

                # 检查告警消失条件（5秒未激活）
                for track_id, data in list(self.tracking_data.items()):
                    if self.current_time - data['last_active'] > 5:
                        if track_id in self.violation_status:
                            del self.violation_status[track_id]
                        data['start_time'] = None  # 重置计时

                # 添加文本描绘违停车辆、违停时间和告警
                for i, track_id in enumerate(sorted(self.violation_status.keys())):
                    if track_id in self.tracking_data:
                        class_id = self.tracking_data[track_id]['class_id']
                        class_name = results2[0].names.get(class_id, f"Class_{class_id}")
                        self.violation_text += f"{class_name}({track_id}):{self.tracking_data[track_id]['total_time']:.2f}s"
                        if self.tracking_data[track_id]['total_time'] > 10:
                            self.violation_text += " | Violation"

                        self.y_offset += i * 30
                        cv2.putText(plot, self.violation_text.rstrip(', '), (10, self.y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    self.violation_text = ""
                    self.y_offset = 30

                out.write(plot)

            cap.release()
            out.release()
            self.reset()

            return None, output_video_path


    def yolo_inference_for_examples(self, image, model_id, image_size, conf_threshold):
        annotated_image, _ = self.yolo_inference(image, None, model_id, image_size, conf_threshold)
        return annotated_image


    def compute_coverage_and_highlight(self, model1_results, model2_results, coverage_threshold=0.4):
        """
        计算第二个模型检测框被第一个模型检测框覆盖的面积百分比，并标记高覆盖率的检测框

        Args:
        model1_results(ultralytics.engine.results.Results)：第一个模型的推理结果
        model2_results(ultralytics.engine.results.Results)：第二个模型的推理结果
        coverage_threshold(float)：覆盖率阈值，超过此值则标记为高覆盖 (默认0.7)

        returns:
        (coverage_ratios, highlight_indices):
            coverage_ratios(list[float])：第二个模型每个检测框的最大覆盖率
            highlight_indices：第二个模型中需要高亮的框索引列表
        """
        # 提取两个模型的检测框 (xyxy格式)
        boxes1 = model1_results.boxes.xyxy.cpu()
        boxes2 = model2_results.boxes.xyxy.cpu()

        # 初始化结果
        coverage_ratios = []
        highlight_indices = []

        # 计算第二个模型每个检测框的面积
        areas2 = box_area(boxes2) if len(boxes2) > 0 else torch.tensor([])

        # 计算每个第二个模型框被第一个模型框覆盖的比例
        for i, box1 in enumerate(boxes1):
            for j, box2 in enumerate(boxes2):
                max_coverage = 0.0

                # 计算当前框的面积
                area2 = areas2[j].item()

                if len(boxes1) > 0:
                    x1, y1, x2, y2 = box2.tolist()  # 将张量转换为Python列表
                    corners = [
                        (x1, y1),  # 左上角
                        (x2, y1),  # 右上角
                        (x2, y2),  # 右下角
                        (x1, y2)  # 左下角
                    ]

                    mask1 = model1_results.masks.xy

                    # 创建矩形和多边形对象
                    corner = Polygon(corners)
                    mask = Polygon(mask1[i].tolist()).buffer(0)

                    # 计算交集面积
                    intersection = corner.intersection(mask)

                    # 计算覆盖率 = 交集面积 / 当前框面积
                    coverage_ratios_j = intersection.area / area2

                    # 获取最大覆盖率（考虑多个覆盖的情况）
                    max_coverage = coverage_ratios_j

                coverage_ratios.append(max_coverage)

                # 检查是否超过阈值
                if max_coverage >= coverage_threshold:
                    highlight_indices.append(j)

        return coverage_ratios, highlight_indices


    def plot_masks_only(self, result, alpha=0.5):
        """
        仅绘制掩码（不绘制边界框和标签）
        result: 单个图像的预测结果（results[0]）
        alpha: 掩码透明度 (0-1)
        """
        # 获取原始图像
        img = result.orig_img.copy()
        if result.masks is None:
            return img  # 如果没有检测到掩码，返回原图

        # 确保掩码尺寸与原始图像匹配
        orig_h, orig_w = img.shape[:2]
        masks = result.masks.data.cpu().numpy()  # 掩码数据 (n, H, W)

        # 调整掩码尺寸（如果需要）
        if masks.shape[1:] != (orig_h, orig_w):
            # 创建空数组存放调整后的掩码
            resized_masks = np.zeros((masks.shape[0], orig_h, orig_w), dtype=masks.dtype)

            # 调整每个掩码尺寸
            for i, mask in enumerate(masks):
                # 使用最近邻插值保持二值特性
                resized_masks[i] = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            masks = resized_masks

        clss = result.boxes.cls.cpu().numpy()  # 类别索引

        # 为每个掩码创建彩色覆盖层
        for i, (mask, cls) in enumerate(zip(masks, clss)):
            color = [255, 0, 0]  # 获取BGR颜色
            colored_mask = np.zeros_like(img, dtype=np.uint8)
            colored_mask[:] = color

            # 确保掩码是单通道8位无符号整数
            mask_uint8 = mask.astype(np.uint8)

            # 应用掩码并叠加到图像
            masked_colored = cv2.bitwise_and(colored_mask, colored_mask, mask=mask_uint8)
            img = cv2.addWeighted(img, 1, masked_colored, alpha, 0)

        return img


gr_predict = GrYoloPredict()

def app():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Image", visible=True)
                video = gr.Video(label="Video", visible=False)
                input_type = gr.Radio(
                    choices=["Image", "Video"],
                    value="Image",
                    label="Input Type",
                )
                model_id = gr.Dropdown(
                    label="Model",
                    choices=[
                        "yolo11x.pt",
                        "car_type_v5_p2.pt"
                    ],
                    value="yolo11x.pt",
                )
                image_size = gr.Slider(
                    label="Image Size",
                    minimum=640,
                    maximum=1920,
                    step=32,
                    value=640,
                )
                conf_threshold = gr.Slider(
                    label="Confidence Threshold",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.25,
                )
                yolo_infer = gr.Button(value="Detect Objects")

            with gr.Column():
                output_image = gr.Image(type="numpy", label="Annotated Image", visible=True)
                output_video = gr.Video(label="Annotated Video", visible=False)

        def update_visibility(input_type):
            image = gr.update(visible=True) if input_type == "Image" else gr.update(visible=False)
            video = gr.update(visible=False) if input_type == "Image" else gr.update(visible=True)
            output_image = gr.update(visible=True) if input_type == "Image" else gr.update(visible=False)
            output_video = gr.update(visible=False) if input_type == "Image" else gr.update(visible=True)

            return image, video, output_image, output_video

        input_type.change(
            fn=update_visibility,
            inputs=[input_type],
            outputs=[image, video, output_image, output_video],
        )

        def run_inference(image, video, model_id, image_size, conf_threshold, input_type):
            if input_type == "Image":
                return gr_predict.yolo_inference(image, None, model_id, image_size, conf_threshold)
            else:
                return gr_predict.yolo_inference(None, video, model_id, image_size, conf_threshold)


        yolo_infer.click(
            fn=run_inference,
            inputs=[image, video, model_id, image_size, conf_threshold, input_type],
            outputs=[output_image, output_video],
        )

        gr.Examples(
            examples=[
                [
                    "ultralytics/assets/bus.jpg",
                    "yolo11x.pt",
                    640,
                    0.25,
                ],
                [
                    "ultralytics/assets/zidane.jpg",
                    "yolo11x.pt",
                    640,
                    0.25,
                ],
            ],
            fn=gr_predict.yolo_inference_for_examples,
            inputs=[
                image,
                model_id,
                image_size,
                conf_threshold,
            ],
            outputs=[output_image],
            cache_examples='lazy',
        )

gradio_app = gr.Blocks()
with gradio_app:
    gr.HTML(
        """
    <h1 style='text-align: center'>
    YOLOv11: An Overview of the Key Architectural Enhancements
    </h1>
    """)
    gr.HTML(
        """
        <h3 style='text-align: center'>
        <a href='https://arxiv.org/abs/2410.17725' target='_blank'>arXiv</a> | <a href='https://github.com/ultralytics/ultralytics' target='_blank'>github</a>
        </h3>
        """)
    with gr.Row():
        with gr.Column():
            app()
if __name__ == '__main__':
    gradio_app.launch(server_port=8188)
