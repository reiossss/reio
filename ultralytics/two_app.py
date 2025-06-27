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
            temp_dir = "D:/project/ultralytics/temp"  # 例如: "C:/my_temp" 或 "/tmp/my_custom_temp"
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
                plot = results1[0].plot()  # 两个模型推理结果

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
                        # x, y, w, h = box
                        # track = self.track_history[track_id]
                        # track.append((float(x), float(y)))  # x, y center point
                        # if len(track) > 30:  # retain 30 tracks for 30 frames
                        #     track.pop(0)
                        #
                        # # Draw the tracking lines
                        # points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        # cv2.polylines(plot, [points], isClosed=False, color=(0, 0, 255), thickness=2)

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

            return None, output_video_path


    def yolo_inference_for_examples(self, image, model_id, image_size, conf_threshold):
        annotated_image, _ = self.yolo_inference(image, None, model_id, image_size, conf_threshold)
        return annotated_image

    '''
    # 计算两个模型检测框之间的IOU
    def compute_cross_model_iou(model1_results, model2_results, iou_threshold=0.1):
        """
        计算两个YOLO模型结果之间的检测框IOU，并标记高IOU的检测框
    
        参数:
        model1_results -- 第一个模型的推理结果 (ultralytics.engine.results.Results)
        model2_results -- 第二个模型的推理结果 (ultralytics.engine.results.Results)
        iou_threshold -- IOU阈值，超过此值则标记为高匹配 (默认0.7)
    
        返回:
        (iou_matrix, highlight_indices) 元组:
            iou_matrix -- 两个模型检测框的IOU矩阵 (torch.Tensor)
            highlight_indices -- 第二个模型中需要高亮的框索引列表
        """
        # 提取两个模型的检测框 (xyxy格式)
        boxes1 = model1_results.boxes.xyxy.cpu()  # 移动到CPU并转为tensor
        boxes2 = model2_results.boxes.xyxy.cpu()
    
        # 初始化空结果
        iou_matrix = torch.tensor([])
        highlight_indices = []
    
        # 仅当两个模型都有检测框时才计算IOU
        if len(boxes1) > 0 and len(boxes2) > 0:
            # 计算IOU矩阵 (shape: [boxes1_count, boxes2_count])
            iou_matrix = box_iou(boxes1, boxes2)
    
            # 找到需要高亮的框 (第二个模型中与第一个模型任意框IOU>阈值的框)
            for j in range(iou_matrix.shape[1]):
                if torch.any(iou_matrix[:, j] > iou_threshold):
                    highlight_indices.append(j)
    
        return iou_matrix, highlight_indices
    '''


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

"""
    def merge_names(self, names1: Dict[int, str], names2: Dict[int, str]) -> tuple[dict[Any, Any], dict[int, dict[Any, Any]]]:
        """"""
        合并两个names字典，并返回映射关系
        返回:
            merged_names: 合并后的{id: name}字典
            cls_mapping: 原始ID到新ID的映射 {model1_id: new_id}, {model2_id: new_id}
        """"""
        merged_names = {}
        cls_mapping = {1: {}, 2: {}}  # 分别存储model1和model2的ID映射

        # 合并names1（保留原有ID优先）
        max_id = 0
        for id_, name in names1.items():
            if name not in merged_names.values():
                merged_names[id_] = name
                cls_mapping[1][id_] = id_  # model1的ID保持不变
                max_id = max(max_id, id_)

        # 合并names2（处理冲突）
        for id_, name in names2.items():
            if name in merged_names.values():
                # 名称已存在，找到对应ID
                existing_id = [k for k, v in merged_names.items() if v == name][0]
                cls_mapping[2][id_] = existing_id
            else:
                # 新名称，分配新ID
                new_id = max_id + 1
                merged_names[new_id] = name
                cls_mapping[2][id_] = new_id
                max_id = new_id

        return merged_names, cls_mapping

    def merge_yolo_results(self,
            results1: Results,
            results2: Results,
            image: Optional[np.ndarray] = None
    ):
        """"""
        合并两个YOLO模型的推理结果

        参数:
            results1: 第一个模型的Results对象
            results2: 第二个模型的Results对象

        返回:
            合并后的Results对象

        示例:
            merged_results = merge_yolo_results(results1[0], results2[0], image)
            merged_results.show()  # 显示合并结果
        """"""

        # 初始化空容器
        boxes = []
        masks = []
        probs = []
        keypoints = []
        obb = []

        # 合并names并获取ID映射
        merged_names, cls_mapping = self.merge_names(results1.names, results2.names)

        # 提取results1的数据 (如果存在)
        if hasattr(results1, 'boxes') and results1.boxes is not None:
            boxes1 = results1.boxes.data.clone()
            for old_id, new_id in cls_mapping[1].items():
                boxes1[boxes1[:, 5] == old_id, 5] = new_id  # 重映射cls
            boxes.append(boxes1)
            if hasattr(results1, 'masks') and results1.masks is not None:
                masks.append(results1.masks.data)
        if hasattr(results1, 'keypoints') and results1.keypoints is not None:
            keypoints.append(results1.keypoints.data)
        if hasattr(results1, 'probs') and results1.probs is not None:
            probs.append(results1.probs.data)
        if hasattr(results1, 'obb') and results1.obb is not None:
            obb.append(results1.obb.data)

        # 提取results2的数据 (如果存在)
        if hasattr(results2, 'boxes') and results2.boxes is not None:
            boxes2 = results2.boxes.data.clone()
            for old_id, new_id in cls_mapping[2].items():
                boxes2[boxes2[:, 5] == old_id, 5] = new_id  # 重映射cls
            boxes.append(boxes2)
            if hasattr(results2.boxes, 'masks') and results2.masks is not None:
                masks.append(results2.masks.data)
        if hasattr(results2, 'keypoints') and results2.keypoints is not None:
            keypoints.append(results2.keypoints.data)
        if hasattr(results2, 'probs') and results2.probs is not None:
            probs.append(results2.probs.data)
        if hasattr(results2, 'obb') and results2.obb is not None:
            obb.append(results2.obb.data)

        # 合并所有检测结果
        merged_boxes = torch.cat(boxes, dim=0) if boxes else torch.empty((0, 6))
        merged_masks = torch.cat(masks, dim=0) if masks else None
        merged_keypoints = torch.cat(keypoints, dim=0) if keypoints else None
        merged_probs = torch.stack(probs).mean(dim=0) if probs else None
        merged_obb = torch.cat(obb, dim=0) if obb else None

        # 创建新的Results对象
        merged_results = Results(
            orig_img=image if image is not None else results1.orig_img,
            path=results1.path,
            names=merged_names,
            boxes=merged_boxes,
            masks=merged_masks,
            keypoints=merged_keypoints,
            probs=merged_probs,
            obb=merged_obb
        )

        return merged_results
"""

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
