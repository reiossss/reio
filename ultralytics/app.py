import gradio as gr
import cv2
import tempfile
import os
from ultralytics import YOLO


def yolo_inference(image, video, model_id, image_size, conf_threshold):
    """
    Args:
        image(str):输入图像
        video(str):输入视频
        model_id(str):选择的模型名称
        image_size(int):检测图像或视频大小
        conf_threshold(float):置信度阈值

    Returns:图像或视频检测结果
    """
    model_path = os.path.join('models', model_id)
    model = YOLO(model_path)
    if image:
        results = model.predict(source=image, imgsz=image_size, conf=conf_threshold, device=0)
        plot = results[0].plot()

        return plot[:, :, ::-1], None
    else:
        temp_dir = "D:/project/ultralytics/temp"
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

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(source=frame, imgsz=image_size, conf=conf_threshold, device=0)
            plot = results[0].plot()

            out.write(plot)

        cap.release()
        out.release()

        return None, output_video_path


def yolo_inference_for_examples(image, model_id, image_size, conf_threshold):
    annotated_image, _ = yolo_inference(image, None, model_id, image_size, conf_threshold)
    return annotated_image


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
                return yolo_inference(image, None, model_id, image_size, conf_threshold)
            else:
                return yolo_inference(None, video, model_id, image_size, conf_threshold)

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
            fn=yolo_inference_for_examples,
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
