import gradio as gr
from ultralytics import YOLO
import cv2
import os
import numpy as np
import sys
import asyncio

# Make sure that browser accesses the filepaths correctly
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# Load your trained model
model = YOLO("best.pt")


# Function to detect drones in an image
def predict_image(image):
    # Perform detection
    detections = model.track(source=image, conf=0.3, iou=0.5, show=False)
    
    # Render the detected image
    detected_image = np.squeeze(detections[0].plot())
    
    # Check if any drones were detected
    num_drones = len(detections[0])  # Assuming detection results are stored in xywh format
    message = "Drone detected!" if num_drones > 0 else "No drones detected."

    return detected_image, message  # Return the detected image and the message



# Function to detect drones in a video
def predict_video(video_path):
    # Load the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Error: Could not open video.", ""

    drone_detected = False
    # Prepare to write the output video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps = cap.get(cv2.CAP_PROP_FPS)
    output_path = 'output_video.mp4'
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), out_fps, (frame_width, frame_height))

    # Process each frame
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

        # Write the frame to the output video
        out.write(annotated_frame)

    # Release everything when done
    cap.release()
    out.release()

    message = "Drone detected in video!" if drone_detected else "No drones detected in video."
    print("Video processing complete. Saved to:", output_path)
    return output_path, message  # Return the path to the output video and the message


# UI Implementation
with gr.Blocks() as demo:
    gr.Markdown("### Drone Detection System")

    with gr.Tab("Introduction"):
        gr.Markdown("**This Application helps in detection of DRONES in an IMAGE, VIDEO or from your WEBCAM depending on your App mode.**")
        gr.Markdown("You Don't Necessarily need a Drone to run this app; you can use an image from google.\n\n**SAMPLE OUTPUT:**")
        gr.Video("Drone Detection.mp4", width=800, height=600)

    with gr.Tab("Upload Image"):
        image_input = gr.Image()
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