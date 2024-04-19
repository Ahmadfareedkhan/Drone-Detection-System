import cv2
from ultralytics import YOLO

# Load the YOLO model with the correct path to your model weights
model_path = r"C:\Users\HP\.vscode\drone_detection\drone_detection_system\best.pt"
model = YOLO(model_path)

def process_frame(frame):
    # Convert the color space from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Perform detection with the correct image size parameter
    results = model(frame_rgb, imgsz=640)  # Adjust 'imgsz' as needed for different resolutions
    
    # Render results on the frame
    # frame_results = results.render()[0]
    
    # Convert back to BGR for OpenCV
    frame_bgr = cv2.cvtColor(results, cv2.COLOR_RGB2BGR)
    return frame_bgr

def show_source(source):
    # Initialize video capture
    cap = cv2.VideoCapture(source)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process the frame
        processed_frame = process_frame(frame)
        
        # Display the processed frame
        cv2.imshow('YOLO Object Detection', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Set the source for video, image, or live camera
# source_video = r"path_to_video.mp4"
# source_image = r"path_to_image.jpg"
source_cam = 0  # Webcam index (usually 0 for the default camera)

# For demonstration, you can uncomment one of these lines based on your needs:

# To show an image
image = cv2.imread(source_image)
processed_image = process_frame(image)
cv2.imshow('YOLO Object Detection', processed_image)

# To show video or live cam feed
show_source(source_cam)  # Replace 'source_video' with 'source_cam' to use webcam

cv2.waitKey(0)
cv2.destroyAllWindows()


