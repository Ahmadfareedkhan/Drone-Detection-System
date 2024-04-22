from PIL import Image
import cv2
import streamlit as st
import torch
from matplotlib import pyplot as plt
import numpy as np
import requests
import json
number_of_drones = 0


def pushbullet_noti(title, body):

    TOKEN = 'o.rFm7mweBTeWeX1lnovShXyzfTn6nAvrF'

    msg = {"type": "note", "title": title, "body": body}

    resp = requests.post('https://api.pushbullet.com/v2/pushes',
                         data=json.dumps(msg),
                         headers={'Authorization': 'Bearer ' + TOKEN,
                                  'Content-Type': 'application/json'})
    if resp.status_code != 200:
        raise Exception('Error', resp.status_code)
    else:
        print('Message sent')


st.title('Drone Detection App')
st.sidebar.title('Drone Detection Sidebar')
st.sidebar.subheader('parameters')
# @st.cache()
detection_type = st.sidebar.selectbox(
    'Choose the App mode', ['About APP', 'Run on IMage', 'Run on Video', 'Go live'])
if detection_type == 'About APP':
    st.markdown(
        'This Application helps in detection of DRONES in an IMAGE, VIDEO or from your WEBCAM depending on your App mode. ')
    st.markdown('''
    You Don't Necessarily need a Drone to run this app you can use an image from google.


    SAMPLE OUTPUT:\n
    ''')

    st.video(r"C:\Users\HP\.vscode\drone_detection\Drone_Detection_Using_YOLOv5\Drone Detection.mp4")


elif detection_type == "Run on IMage":
    st.sidebar.markdown('-----')
    confidence = st.sidebar.slider(
        'Detection Confidence', min_value=0.0, max_value=1.0, value=0.6)
    st.sidebar.markdown('-----')
    image = st.sidebar.file_uploader(
        "UPLOAD an IMAGE", type=['jpg', 'png', 'jpeg'])
    if image is None:

        model = torch.hub.load(
            'yolov5', 'custom', path='best.pt', source='local', force_reload=True)
        model.conf = 0.25

        output = model('drone5.jpg')

        drone_positive = output.pandas().xyxy[0]['name']
        confidence_sent = output.pandas().xyxy[0]['confidence']
        number_of_drones = output.pandas().xyxy[0].value_counts('name')[0]
        drone_positive = drone_positive[0]
        if drone_positive == "drone":
            pushbullet_noti(
                "Warning", f"{number_of_drones} Drone Detected with a confidence of {confidence_sent}")

        st.sidebar.text("original image")
        st.sidebar.image('drone5.jpg')
        plt.imshow(np.squeeze(output.render()))

        st.image(output.render())

    else:
        model = torch.hub.load('ultralytics/yolov5',
                               'custom', path='best.pt', force_reload=True)
        model.conf = 0.25
        image = Image.open(image)
        img_array = np.array(image)  # if you want to pass it to OpenCV
        output1 = model(Image.fromarray(img_array))

        drone_positive = output1.pandas().xyxy[0]['name']
        number_of_drones = output1.pandas().xyxy[0].value_counts('name')[0]
        drone_positive = drone_positive[0]
        confidence_sent = output1.pandas().xyxy[0]['confidence'][0]

        if drone_positive == "drone":
            pushbullet_noti(
                "Warning", f"{number_of_drones} Drone Detected with a confidence of {confidence_sent}")
        plt.imshow(np.squeeze(output1.render()))
        st.image(output1.render())

elif detection_type == "Go live":
    st.sidebar.markdown('-----')
    confidence = st.sidebar.slider(
        'Detection Confidence', min_value=0.0, max_value=1.0, value=0.6)
    st.sidebar.markdown('-----')

    st.title("Going LIve")
    confirm = st.checkbox('Start the webcam')
    if confirm:
        st.write("Making connection to your webcam......Wait for a while")

    FRAME_WINDOW = st.image([])

    source = 0
    while confirm:

        try:

            cam = cv2.VideoCapture(source)
            ret, frame = cam.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            model = torch.hub.load('ultralytics/yolov5', 'custom',
                                   path='best.pt', force_reload=True)

            model.conf = confidence
            results = model(frame)
            FRAME_WINDOW.image(np.squeeze(results.render()))
        except Exception as e:
            st.write("Unable to connect to your webcam kindly retry the App")
            source = source+1
    else:

        st.write('Kindly select the checkbox to start your webcam')
