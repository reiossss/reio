import cv2
import tempfile
import os
from ultralytics import YOLO
import torch
from torchvision.ops import box_area
from collections import defaultdict
from shapely.geometry import Polygon

class YoloPredict():
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
        model1 = YOLO('models/yellow_grid_v5.pt')
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        model2 = YOLO(model_path)
        if image:
            results1 = model1.predict(source=image, imgsz=image_size, conf=conf_threshold, device=0)
            results2 = model2.predict(source=image, imgsz=image_size, conf=conf_threshold, device=0)

            # 计算覆盖率并获取高亮索引
            coverage_ratios, highlight_indices = self.compute_coverage_and_highlight(results1[0], results2[0])
            plot = results1[0].plot(boxes=False)  # 第一个模型推理掩码结果
            image_path = tempfile.mktemp(suffix=".png", dir=temp_dir)
            cv2.imwrite(image_path, plot)

            # 在第二个模型的可视化结果上标红高IOU框
            if hasattr(results2[0], 'boxes'):
                boxes = results2[0].boxes.xyxy.cpu().numpy()
                for idx in highlight_indices:
                    box = boxes[idx].astype(int)
                    cv2.rectangle(plot, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 3)  # 红色边框

            return image_path, None
        else:
            video_path = tempfile.mktemp(suffix=".mp4", dir=temp_dir)
            with open(video_path, "wb") as f:
                with open(video, "rb") as g:
                    f.write(g.read())

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            output_video_path = tempfile.mktemp(suffix=".mp4", dir=temp_dir)
            out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                self.current_time = frame_count / fps

                results1 = model1.predict(source=frame, imgsz=image_size, conf=conf_threshold, device=0)
                trace = model2.track(frame, persist=True)[0]

                # 计算覆盖率并获取高亮索引
                coverage_ratios, highlight_indices = self.compute_coverage_and_highlight(results1[0], trace)
                plot = results1[0].plot(boxes=False)  # 第一个模型推理掩码结果

                # Get the boxes and track IDs
                if trace.boxes and trace.boxes.is_track:
                    boxes = trace.boxes.xywh.cpu()
                    track_ids = trace.boxes.id.int().cpu().tolist()
                    classes = trace.boxes.cls.cpu().tolist()

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

                # 在第一个模型的掩码结果上标红高IOU框
                if hasattr(trace, 'boxes') and len(highlight_indices) > 0:
                    boxes = trace.boxes.xyxy.cpu().numpy()

                    for idx in highlight_indices:
                        if idx < len(boxes):
                            box = boxes[idx].astype(int)
                            try:
                                track_id = trace.boxes.id[idx].item()
                            except (AttributeError, TypeError, IndexError):
                                track_id = -1

                            # 更新时间统计
                            if track_id != -1 and track_id in self.tracking_data:
                                data = self.tracking_data[track_id]
                                if data['start_time'] is None:
                                    data['start_time'] = self.current_time
                                data['last_active'] = self.current_time
                                data['total_time'] = self.current_time - data['start_time']

                                # 检查违规条件
                                if data['total_time'] > 2:
                                    self.violation_status[track_id] = True

                                # 绘制红色边框
                                class_id = data['class_id']
                                class_name = trace.names.get(class_id, f"Class_{class_id}")
                                cv2.rectangle(plot, (box[0], box[1]), (box[2], box[3]), (94,53,255), 2)
                                self.violation_text += f"{class_name}[{int(track_id)}] : {self.tracking_data[track_id]['total_time']:.2f}s"
                                if self.tracking_data[track_id]['total_time'] > 10:
                                    self.violation_text += " | Violation"

                                # 计算文本背景
                                (text_width, text_height), _ = cv2.getTextSize(self.violation_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                                cv2.rectangle(plot, (box[0], box[1] - 2 * text_height), (box[0] + text_width, box[1]), (94,53,255), -1)
                                cv2.putText(plot, self.violation_text, (box[0] + 2, box[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                            self.violation_text = ""

                # 检查告警消失条件（5秒未激活）
                for track_id, data in list(self.tracking_data.items()):
                    if self.current_time - data['last_active'] > 5:
                        if track_id in self.violation_status:
                            del self.violation_status[track_id]
                        data['start_time'] = None  # 重置计时

                # # 添加文本描绘违停车辆、违停时间和告警
                # for i, track_id in enumerate(sorted(self.violation_status.keys())):
                #     if track_id in self.tracking_data:
                #         class_id = self.tracking_data[track_id]['class_id']
                #         class_name = trace.names.get(class_id, f"Class_{class_id}")
                #         self.violation_text += f"{class_name}[{int(track_id)}] : {self.tracking_data[track_id]['total_time']:.2f}s"
                #         if self.tracking_data[track_id]['total_time'] > 10:
                #             self.violation_text += " | Violation"
                #
                #         self.y_offset += i * 30
                #         cv2.putText(plot, self.violation_text.rstrip(', '), (10, self.y_offset),
                #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                #
                #     self.violation_text = ""
                #     self.y_offset = 30

                out.write(plot)

            cap.release()
            out.release()
            self.reset()

            return None, output_video_path


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


gr_predict = YoloPredict()


if __name__ == '__main__':
    video = "data/vision/violation.mp4"
    model_id = "car_type_v11.pt"
    image_size = 640
    conf_threshold = 0.25
    gr_predict.yolo_inference(None, video, model_id, image_size, conf_threshold)
