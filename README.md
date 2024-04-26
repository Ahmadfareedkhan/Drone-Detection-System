# Drone Detection System

**Introduction**
This Drone Detection System is designed to identify drones in images and videos using a trained YOLO model. The application allows users to upload images and videos for drone detection and will soon support live detection via webcam. It provides an intuitive Gradio interface for easy interaction.

# Features
**Drone Detection in Images:** Upload images and get immediate feedback on whether any drones are detected.

**Drone Detection in Videos:** Upload videos to process and detect drones frame by frame.

**Alert System:** The application notifies the user with an alert message if drones are detected in the uploaded content.

**Easy-to-Use Interface:** A user-friendly web interface powered by Gradio.


# Installation
To set up the Drone Detection System on your local machine, follow these steps:


# Requirements
Ensure you have Python installed on your machine (Python 3.8 or later is recommended). You will also need the following packages:

Gradio
OpenCV
NumPy
Ultralytics YOLO


# Setup
**Clone the repository**

1. git clone https://github.com/Ahmadfareedkhan/Drone-Detection-System.git

2. cd drone-detection-system


# Install dependencies
pip install -r requirements.txt


# Download trained model
Ensure you have the trained YOLO model (best.pt) placed in the appropriate directory as expected by the script.


# Run the application
python app.py
This will launch the Gradio interface accessible via a web browser.


# Usage
After launching the application, navigate to http://localhost:7860 in your web browser. You will see multiple tabs:

**Introduction:** General information and sample output.

**Upload Image:** Upload images for drone detection.

**Upload Video:** Upload videos for processing and drone detection.

**Live:** (Future feature) Real-time drone detection using a webcam.

For image and video detection, upload your file and the system will automatically process it and display results along with an alert message indicating the presence of drones.

# Contributions
Contributions to the Drone Detection System are welcome. Please ensure to follow the project's coding standards and submit pull requests for any new features or bug fixes.

# License
MIT
