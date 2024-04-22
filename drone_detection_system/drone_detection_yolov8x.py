from ultralytics import YOLO
import cv2  # Ensure OpenCV is installed
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# Load the model
model = YOLO(r"C:\Users\HP\.vscode\drone_detection\drone_detection_system\best.pt")

# Perform tracking and display the image
result = model.track(source=r"C:\Users\HP\.vscode\drone_detection\drone_detection_system\test\pexels-joseph-redfield-8459631 (1080p).mp4", conf=0.3, iou=0.5, show=True)

# Wait indefinitely until a key is pressed
cv2.waitKey(0)

# When a key is pressed, close all OpenCV windows
cv2.destroyAllWindows()
