import gradio as gr
from ultralytics import YOLO
import cv2
import os
import numpy as np
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load your trained model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=r"best.pt", source='github')

def predict_image(image):
    # Convert PIL image to cv2 format
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Perform detection
    results = model(image)
    results.render()  # Draw the bounding boxes on the image
    
    # Extract rendered image and convert from BGR to RGB
    detected_image = results.ims[0]
    detected_image = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)
    
    # Check if any objects were detected
    num_drones = len(results.xyxy[0])  # xyxy is the tensor of bounding boxes [xmin, ymin, xmax, ymax, confidence, class]
    message = "Drone detected!" if num_drones > 0 else "No drones detected."
    
    return detected_image, message

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
        results = model(frame)
        results.render()
        
        # Check if any drones were detected in this frame
        if len(results.xyxy[0]) > 0:
            drone_detected = True

        # Write the frame to the output video
        annotated_frame = cv2.cvtColor(results.ims[0], cv2.COLOR_RGB2BGR)
        out.write(annotated_frame)

    # Release everything when done
    cap.release()
    out.release()

    message = "Drone detected in video!" if drone_detected else "No drones detected in video."
    return output_path, message


def start_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error opening video file")

    # Process video
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Perform inference
            results = model(frame)

            # Render results on frame
            results.render()  # updates results.imgs with boxes and labels
            cv2.imshow('YOLOv5 Detection', results.ims[0])

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break
        else:
            break

    # Release everything when job is finished
    cap.release()
    cv2.destroyAllWindows()


# The Gradio app code with webcam support
with gr.Blocks() as demo:
    gr.Markdown("### Drone Detection System")
    with gr.Tab("Introduction"):
        gr.Markdown("**This Application helps in detection of DRONES in an IMAGE, VIDEO or from your WEBCAM depending on your App mode.**")
        gr.Markdown("You Don't Necessarily need a Drone to run this app; you can use an image from Google.\n\n**SAMPLE OUTPUT:**")
        gr.Video(r"Drone Detection.mp4", width=800, height=600)
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
        gr.Markdown("**Click the button below to start the webcam for real-time drone detection.**")
        start_button = gr.Button("Start Webcam")

        start_button.click(fn=start_webcam, inputs=[])

demo.launch()

