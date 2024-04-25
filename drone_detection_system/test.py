import gradio as gr
from ultralytics import YOLO
import cv2
import os
import numpy as np
import threading

# Make sure to use the correct model initialization for YOLO
model = YOLO(r"C:\Users\Falcon\Downloads\Projects\Falcon Projects\Drone-Detection-System\drone_detection_system\best.pt")

def predict_video(video_path):
    # Load the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Error: Could not open video."

    # Prepare to write the output video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps = cap.get(cv2.CAP_PROP_FPS)
    temp_output_path = os.path.join("C:\\Users\\Falcon\\Downloads\\Projects\\Falcon Projects\\Drone-Detection-System", "temp_output.mp4")
    output_path = os.path.join('C:\\Users\\Falcon\\Downloads\\Projects\\Falcon Projects\\Drone-Detection-System\\drone_detection_system', 'output_video.mp4')
    out = cv2.VideoWriter(temp_output_path, cv2.VideoWriter_fourcc(*'mp4v'), out_fps, (frame_width, frame_height))

    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection on the frame
        results = model.track(frame, imgsz=640, conf=0.3, iou=0.5, show=False)

        # Draw boxes on the frame
        annotated_frame = np.squeeze(results[0].plot())

        # Write the frame to the output video
        out.write(annotated_frame)

    # Release everything when done
    cap.release()
    out.release()

    # Rename the temporary file to the final output path
    os.rename(temp_output_path, output_path)

    return output_path  # Return the path to the fully processed video

def process_video_and_update_interface(video_path, interface):
    def update_interface(processed_video_path):
        interface.update(value=processed_video_path, element_id="video_output")

    def process_video(video_path):
        processed_video_path = predict_video(video_path)
        interface.send(update_interface, processed_video_path)

    # Process the video in a separate thread to avoid blocking the interface
    thread = threading.Thread(target=process_video, args=(video_path,))
    thread.start()

# Set up the Gradio interface
with gr.Blocks() as demo:
    with gr.Tab("Upload Video"):
        video_input = gr.Video(label="Input Video")
        video_output = gr.Video(label="Processed Video")

        video_input.change(
            fn=lambda video_path: process_video_and_update_interface(video_path, demo),
            inputs=video_input,
            outputs=video_output
        )

# Launch the Gradio app
demo.launch()
