from ultralytics import YOLO
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

if __name__ == "__main__":
    model_path = '/yolo/weights/train/yolo11m.pt'
    model = YOLO(model_path)
    model.train(
        data='/data/blue_v5/data.yaml',
        epochs=200,
        batch=12,
        imgsz=640,
        device=[0, 1, 2], # 使用 GPU（3 表示第4个GPU）
        project='yolo',
        name='blue_v5'
    )
