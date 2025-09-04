import cv2
import tempfile
import os
from ultralytics import YOLO
import numpy as np
from sklearn.cluster import DBSCAN
from font import ChineseTextDrawer
# from PIL import Image, ImageDraw, ImageFont, ImageTk

class VehicleSpeedEstimator():
    def __init__(self):
        self.vehicle_tracks = {}
        self.vehicle_speeds = {}
        self.centers=[]
        self.box_heights=[]
        self.box_widths=[]
        self.pixels_per_meter = 30
        self.window_size = 10
        self.drone_speed = 1.5
        self.color = (94, 53, 255)
        self.speed_threshold = 5
        self.base_eps = 200
        self.base_h = 30
        self.min_samples = 3
        self.is_crowd_count = 0
        self.drawer = ChineseTextDrawer()


    def reset(self):
        self.vehicle_tracks = {}
        self.vehicle_speeds = {}


    def yolo_inference(self, video, model_id):
        """
        Args:
            video(str):输入视频
            model_id(str):选择的模型名称

        Returns:图像或视频检测结果
        """
        model_path = f"models/{model_id}"
        model = YOLO(model_path)
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
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
            if results[0].boxes.id is not None and current_time > 7:
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
                        vy_mps = vy_pixels / self.pixels_per_meter + self.drone_speed

                        # 计算合成速度
                        speed_mps = np.sqrt(vx_mps ** 2 + vy_mps ** 2)
                        speed_kmph = speed_mps * 3.6  # 转换为km/h

                        # 存储当前速度
                        self.vehicle_speeds[track_id].append(speed_kmph)

                        # 计算平均速度（用于显示）
                        avg_speed = np.mean(self.vehicle_speeds[track_id][-5:]) if len(
                            self.vehicle_speeds[track_id]) > 0 else 0

                        # 在帧上绘制结果
                        if avg_speed > self.speed_threshold:
                            box_height = y2 - y1
                            box_width = x2 - x1
                            self.centers.append((center_x, center_y))
                            self.box_heights.append(box_height)
                            self.box_widths.append(box_width)
                            # class_name = results[0].names.get(class_id, f"Class_{class_id}")
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), self.color, 1)
                            frame = self.drawer.put_text(frame,
                                                         f"奔跑",
                                                         (int(x1), int(y1) - 17),
                                                         font_color=(255, 255, 255),
                                                         bg_color=self.color,
                                                         font_size=15)


                            # # 绘制轨迹
                            # for j in range(1, len(self.vehicle_tracks[track_id])):
                            #     prev = self.vehicle_tracks[track_id][j - 1]
                            #     curr = self.vehicle_tracks[track_id][j]
                            #     cv2.line(frame,
                            #              (int(prev[1]), int(prev[2])),
                            #              (int(curr[1]), int(curr[2])),
                            #              (255, 0, 0), 2)

                avg_height_global = np.mean(self.box_heights) if self.box_heights else 50

                # 转换为NumPy数组
                points = np.array(self.centers)
                if len(points) > 0:

                    # 确保points是二维数组 (n, 2)
                    if points.ndim == 1:
                        points = points.reshape(-1, 2)

                    eps = self.base_eps * (avg_height_global / self.base_h)  # 动态eps
                    # min_samples = 5  # 计数阈值：单位：人
                    avg_width = np.mean(self.box_widths) if self.box_widths else 0
                    # 全局DBSCAN聚类
                    dbscan = DBSCAN(eps=eps, min_samples=self.min_samples, metric='euclidean')
                    labels = dbscan.fit_predict(points)

                    unique_labels = set(labels) - {-1}

                    # 可视化聚类结果
                    for label in unique_labels:
                        cluster_pts = points[labels == label].astype(np.float32)

                        # 计算最小包围圆
                        if len(cluster_pts) >= self.min_samples:  # minEnclosingCircle需要至少3个点
                            self.is_crowd_count += 1
                            center, radius = cv2.minEnclosingCircle(cluster_pts)
                            center = (int(center[0]), int(center[1]))
                            radius = int(radius)

                            # 绘制包围圆
                            cv2.circle(frame, center, int(radius + 2 * avg_width), self.color, 2)
                            frame = self.drawer.put_text(img = frame,
                                                         text=f"疑似异常奔跑人群：{len(cluster_pts)}人",
                                                         position=(center[0] - 85, int(center[1] - radius - 2 * avg_width - 25)),
                                                         font_color= self.color,
                                                         font_size=20)

            out.write(frame)
            self.centers = []
            self.box_heights = []
            self.box_widths = []

        cap.release()
        out.release()
        self.reset()

        return output_video_path


velocity = VehicleSpeedEstimator()


if __name__ == '__main__':
    video = "D:/download/run/run2.mp4"
    model_id = "people_v14.pt"
    velocity.yolo_inference(video, model_id)
