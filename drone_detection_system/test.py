import torch
from PIL import Image
import numpy as np
import streamlit as st
import cv2
import requests
import json
import torchvision.transforms as transforms

# Define the image transformation
def prepare_image(image):
    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # Adjust the size as per your model's requirement
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model = checkpoint['model']
        model.eval()
        return model
    elif isinstance(checkpoint, torch.nn.Module):
        checkpoint.eval()
        return checkpoint
    else:
        raise TypeError("Unexpected model format in the file.")

def pushbullet_noti(title, body):
    TOKEN = 'o.rFm7mweBTeWeX1lnovShXyzfTn6nAvrF'
    msg = {"type": "note", "title": title, "body": body}
    resp = requests.post('https://api.pushbullet.com/v2/pushes',
                         data=json.dumps(msg),
                         headers={'Authorization': 'Bearer ' + TOKEN, 'Content-Type': 'application/json'})
    if resp.status_code != 200:
        raise Exception('Error', resp.status_code)
    else:
        print('Message sent')

st.title('Drone Detection App')
st.sidebar.title('Drone Detection Sidebar')
st.sidebar.subheader('Parameters')

detection_type = st.sidebar.selectbox(
    'Choose the App mode', ['About APP', 'Run on Image', 'Run on Video', 'Go live'])

if detection_type == 'About APP':
    st.markdown(
        'This Application helps in detection of DRONES in an IMAGE, VIDEO or from your WEBCAM depending on your App mode.')
    st.video(r"C:\Users\HP\.vscode\drone_detection\Drone-Detection-System\drone_detection_system\test\pexels-joseph-redfield-8459631 (1080p).mp4")

elif detection_type == "Run on Image":
    confidence = st.sidebar.slider('Detection Confidence', 0.0, 1.0, 0.6)
    image_file = st.sidebar.file_uploader("UPLOAD an IMAGE", type=['jpg', 'png', 'jpeg'])

    if image_file:
        image = Image.open(image_file).convert("RGB")
        model = load_model(r"C:\Users\HP\.vscode\drone_detection\Drone-Detection-System\drone_detection_system\best.pt")
        prepared_image = prepare_image(image)
        output = model(prepared_image)

        drone_positive = output.pandas().xyxy[0]['name']
        confidence_sent = output.pandas().xyxy[0]['confidence']
        number_of_drones = output.pandas().xyxy[0].value_counts('name')[0]

        if "drone" in drone_positive:
            pushbullet_noti("Warning", f"{number_of_drones} Drone Detected with a confidence of {confidence_sent}")

        st.image(prepared_image.squeeze(0).permute(1, 2, 0), caption='Processed Image')
        st.write(output.pandas().xyxy[0])

elif detection_type == "Go live":
    confidence = st.sidebar.slider('Detection Confidence', 0.0, 1.0, 0.6)
    confirm = st.checkbox('Start the webcam')
    FRAME_WINDOW = st.image([])

    if confirm:
        st.write("Making connection to your webcam......Please wait.")
        cap = cv2.VideoCapture(0)  # Use 0 for the primary camera
        model = load_model(r"C:\Users\HP\.vscode\drone_detection\Drone-Detection-System\drone_detection_system\best.pt")

        while confirm:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture video")
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            prepared_frame = prepare_image(frame)
            results = model(prepared_frame)
            FRAME_WINDOW.image(np.squeeze(results.render()), use_column_width=True)

    else:
        st.write('Please select the checkbox to start your webcam.')
