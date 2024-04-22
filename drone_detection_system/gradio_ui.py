import gradio as gr
from ultralytics import YOLO
import cv2
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# Load your trained model
model = YOLO(r"C:\Users\Falcon\Downloads\Projects\Falcon Projects\Drone-Detection-System\drone_detection_system\best.pt")

def predict_image(image):
    return model.track(source=image, conf=0.3, iou=0.5, show=True)

def predict_video(video):
    return model.track(video, conf=0.3, iou=0.5, show=True)

with gr.Blocks() as demo:
    gr.Markdown("### Drone Detection System")
    with gr.Tab("Introduction"):
        gr.Markdown("**This Application helps in detection of DRONES in an IMAGE, VIDEO or from your WEBCAM depending on your App mode.**")
        gr.Markdown("""**You Don't Necessarily need a Drone to run this app you can use an image from google.**


    ### SAMPLE OUTPUT:\n""")
        gr.Video("C:\\Users\\Falcon\\Downloads\\Projects\Falcon Projects\\Drone-Detection-System\\Drone_Detection_Using_YOLOv5\\Drone Detection.mp4", width=800, height=600)
    with gr.Tab("Upload Image"):
        image_input = gr.Image()
        image_output = gr.Image()
        image_input.change(fn=predict_image, inputs=image_input, outputs=image_output)
    with gr.Tab("Upload Video"):
        video_input = gr.Video()
        video_output = gr.Video()
        video_input.change(fn=predict_video, inputs=video_input, outputs=video_output)
    with gr.Tab("Live"):
        gr.Markdown("Live detection will be implemented soon.")

demo.launch()
