import gradio as gr
from ultralytics import YOLO
import cv2
import os
import numpy as np
import sys
import asyncio

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# Load your trained model
model = YOLO(r"C:\Users\Falcon\Downloads\Projects\Falcon Projects\Drone-Detection-System\drone_detection_system\best.pt")

def predict_image(image):
    # Perform detection
    detections = model.track(source=image, conf=0.3, iou=0.5, show=False)
    
    # Render the detected image
    detected_image = np.squeeze(detections[0].plot())
    
    return detected_image  # Return the detected image


def predict_video(video_path):
    # Load the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Error: Could not open video."

    # Prepare to write the output video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps = cap.get(cv2.CAP_PROP_FPS)
    output_path = os.path.join('C:\\Users\\Falcon\\Downloads\\Projects\\Falcon Projects\\Drone-Detection-System\\drone_detection_system', 'output_video.mp4')
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), out_fps, (frame_width, frame_height))

    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection on the frame using the correct parameter
        results = model.track(frame, imgsz=640, conf=0.3, iou=0.5, show=False)

        # Draw boxes on the frame
        annotated_frame = np.squeeze(results[0].plot())

        # Write the frame to the output video
        out.write(annotated_frame)

    # Release everything when done
    cap.release()
    out.release()

    return output_path  # Return the path to the output video




with gr.Blocks() as demo:
    gr.Markdown("### Drone Detection System")
    with gr.Tab("Introduction"):
        gr.Markdown("**This Application helps in detection of DRONES in an IMAGE, VIDEO or from your WEBCAM depending on your App mode.**")
        gr.Markdown("""**You Don't Necessarily need a Drone to run this app you can use an image from google.**


    ### SAMPLE OUTPUT:\n""")
        gr.Video("C:\\Users\\Falcon\\Downloads\\Projects\Falcon Projects\\Drone-Detection-System\\Drone_Detection_Using_YOLOv5\\Drone Detection.mp4", width=800, height=600)
    with gr.Tab("Upload Image"):
        image_input = gr.Image()
        image_output = gr.Image(height='600px', width='auto')
        image_input.change(fn=predict_image, inputs=image_input, outputs=image_output)
    with gr.Tab("Upload Video"):
        video_input = gr.Video()
        video_output = gr.Video(render=True)
        video_input.change(fn=predict_video, inputs=video_input, outputs=video_output)
    with gr.Tab("Live"):
        gr.Markdown("Live detection will be implemented soon.")

demo.launch()
