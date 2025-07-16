import cv2
import tempfile
import os
from ultralytics import YOLO
import numpy as np

class VehicleSpeedEstimator():
    def __init__(self):
        self.vehicle_tracks = {}
        self.vehicle_speeds = {}
        self.pixels_per_meter = 32
        self.window_size = 10


    def reset(self):
        self.vehicle_tracks = {}
        self.vehicle_speeds = {}


    def yolo_inference(self, image, video, model_id, image_size, conf_threshold):
        """
        Args:
            image(str):输入图像
            video(str):输入视频
            model_id(str):选择的模型名称
            image_size(int):检测图像或视频大小
            conf_threshold(float):置信度阈值

        Returns:图像或视频检测结果
        """
        model_path = f"models/{model_id}"
        model = YOLO(model_path)
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        if image:
            results = model.predict(source=image, imgsz=image_size, conf=conf_threshold, device=0)
            plot = results[0].plot()
            image_path = tempfile.mktemp(suffix=".png", dir=temp_dir)
            cv2.imwrite(image_path, plot)

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

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 获取当前帧的时间戳（秒）
                current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

                results = model.track(frame, persist=True)

                # plot = results[0].plot() # 绘制原模型跟踪结果

                # 提取检测结果
                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

                    # 处理每个检测到的车辆
                    for i, box in enumerate(boxes):
                        class_id = class_ids[i]
                        track_id = track_ids[i]

                        x1, y1, x2, y2 = box
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2

                        # 初始化轨迹记录
                        if track_id not in self.vehicle_tracks:
                            self.vehicle_tracks[track_id] = []
                            self.vehicle_speeds[track_id] = []

                        # 添加当前点
                        self.vehicle_tracks[track_id].append((current_time, center_x, center_y))

                        # 保持最近的N个点用于回归
                        if len(self.vehicle_tracks[track_id]) > self.window_size:
                            self.vehicle_tracks[track_id].pop(0)

                        # 计算速度（至少需要2个点）
                        if len(self.vehicle_tracks[track_id]) >= 2:
                            # 准备回归数据
                            times = np.array([t[0] for t in self.vehicle_tracks[track_id]])
                            x_pos = np.array([t[1] for t in self.vehicle_tracks[track_id]])
                            y_pos = np.array([t[2] for t in self.vehicle_tracks[track_id]])

                            # 执行线性回归 (x = a + b*t)
                            A = np.vstack([times, np.ones(len(times))]).T
                            bx, _, _, _ = np.linalg.lstsq(A, x_pos, rcond=None)
                            by, _, _, _ = np.linalg.lstsq(A, y_pos, rcond=None)

                            # 计算速度分量（像素/秒）
                            vx_pixels = bx[0]
                            vy_pixels = by[0]

                            # 转换为米/秒
                            vx_mps = vx_pixels / self.pixels_per_meter
                            vy_mps = vy_pixels / self.pixels_per_meter

                            # 计算合成速度
                            speed_mps = np.sqrt(vx_mps ** 2 + vy_mps ** 2)
                            speed_kmph = speed_mps * 3.6  # 转换为km/h

                            # 存储当前速度
                            self.vehicle_speeds[track_id].append(speed_kmph)

                            # 计算平均速度（用于显示）
                            avg_speed = np.mean(self.vehicle_speeds[track_id][-5:]) if len(
                                self.vehicle_speeds[track_id]) > 0 else 0

                            # 在帧上绘制结果
                            class_name = results[0].names.get(class_id, f"Class_{class_id}")
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (125, 125, 125), 2)
                            cv2.putText(frame, f"{class_name}[{track_id}] : {avg_speed:.1f} km/h", (int(x1), int(y1) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (125, 125, 125), 2)

                            # 绘制轨迹
                            for j in range(1, len(self.vehicle_tracks[track_id])):
                                prev = self.vehicle_tracks[track_id][j - 1]
                                curr = self.vehicle_tracks[track_id][j]
                                cv2.line(frame,
                                         (int(prev[1]), int(prev[2])),
                                         (int(curr[1]), int(curr[2])),
                                         (255, 0, 0), 2)

                out.write(frame)

            cap.release()
            out.release()
            self.reset()

            return None, output_video_path


velocity = VehicleSpeedEstimator()


if __name__ == '__main__':
    video = "/yolo/yolo11/data/vision/velocity2.mp4"
    model_id = "car_type_v11.pt"
    image_size = 640
    conf_threshold = 0.4
    velocity.yolo_inference(None, video, model_id, image_size, conf_threshold)
