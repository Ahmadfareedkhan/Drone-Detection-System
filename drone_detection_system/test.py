import gradio as gr
from ultralytics import YOLO
import cv2
import os
import numpy as np
import sys
import asyncio
from gmail_api import get_service, create_message, send_message

# Make sure that browser accesses the filepaths correctly
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Load your trained model
model = YOLO("best.pt")

# Global Gmail API service
service = get_service()



def predict_image(image):
    # Perform detection
    detections = model.track(source=image, conf=0.3, iou=0.5, show=False)
    detected_image = np.squeeze(detections[0].plot())
    num_drones = len(detections[0])  # Update based on your detection results structure

    if num_drones > 0:
        message = create_message("aneeblba@gmail.com", "maneebajmal@gmail.com", "Drone Detection Alert", "A drone was detected in the uploaded image.")
        send_message(service, "me", message)
        alert_message = "Drone detected! Notification sent."
    else:
        alert_message = "No drones detected."

    return detected_image, alert_message



# Function to detect drones in a video
def predict_video(video_path):
    # Load the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Error: Could not open video.", ""

    drone_detected = False
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps = cap.get(cv2.CAP_PROP_FPS)
    output_path = 'output_video.mp4'
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), out_fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection on the frame
        results = model.track(frame, imgsz=640, conf=0.3, iou=0.5)
        
        # Check if any drones were detected in this frame
        if len(results[0]) > 0:
            drone_detected = True

        # Draw boxes on the frame
        annotated_frame = np.squeeze(results[0].plot())
        out.write(annotated_frame)

    # Release everything when done
    cap.release()
    out.release()

    if drone_detected:
        message = create_message("aneeblba@gmail.com", "maneebajmal@gmail.com", "Drone Detection Alert", "A drone was detected in the uploaded video.")
        send_message(service, "me", message)
        alert_message = "Drone detected in video! Notification sent."
    else:
        alert_message = "No drones detected in video."

    return output_path, alert_message



# UI Implementation
with gr.Blocks() as demo:
    gr.Markdown("### Drone Detection System")

    with gr.Tab("Introduction"):
        gr.Markdown("**This Application helps in detection of DRONES in an IMAGE, VIDEO or from your WEBCAM depending on your App mode.**")
        gr.Markdown("You Don't Necessarily need a Drone to run this app; you can use an image from google.\n\n**SAMPLE OUTPUT:**")
        gr.Video("Drone Detection.mp4", width=800, height=600)

    with gr.Tab("Upload Image"):
        image_input = gr.Image(sources="upload")
        image_output = gr.Image()
        alert = gr.Label()
        image_input.change(fn=predict_image, inputs=image_input, outputs=[image_output, alert])

    with gr.Tab("Upload Video"):
        video_input = gr.Video(sources="upload")
        video_output = gr.Video(render=True)
        alert_video = gr.Label()
        video_input.change(fn=predict_video, inputs=video_input, outputs=[video_output, alert_video])
        
    with gr.Tab("Live"):
        gr.Markdown("Live detection will be implemented soon.")

demo.launch()