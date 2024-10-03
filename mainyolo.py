import numpy as np
import cv2
import openpyxl
from flask import Flask
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import base64
import time
import cvzone
import math
from ultralytics import YOLO

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables for fall detection
imageLink = ""
isFall = False
fallTime = ""
model = YOLO('yolov8s.pt')

# Load class names
classnames = []
with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines()

def process_frame(frame):
    """Process video frame and detect fall using YOLO model."""
    global isFall, fallTime, imageLink
    
    frame = cv2.resize(frame, (980, 740))
    results = model(frame)
    
    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            confidence = box.conf[0]
            class_detect = box.cls[0]
            class_detect = int(class_detect)
            class_detect = classnames[class_detect]
            conf = math.ceil(confidence * 100)
            
            # Calculate bounding box dimensions
            height = y2 - y1
            width = x2 - x1
            threshold = height - width

            if conf > 80 and class_detect == 'person':
                cvzone.cornerRect(frame, [x1, y1, width, height], l=30, rt=6)
                cvzone.putTextRect(frame, f'{class_detect}', [x1 + 8, y1 - 12], thickness=2, scale=2)
                
                # Detect fall: If height < width, fall is detected
                if threshold < 0:
                    cvzone.putTextRect(frame, 'Fall Detected', [x1, y1 - 40], thickness=2, scale=2)
                    if not isFall:
                        isFall = True
                        fallTime = time.strftime('%Y-%m-%d %H:%M:%S')
                        imageLink = './outputs/fallImage.png'
                        cv2.imwrite(imageLink, frame)  # Save the frame as fall image

            else:
                isFall = False  # Reset if no fall is detected

@socketio.on('video_frame')
def handle_video_stream(data):
    """Receive base64-encoded frame from frontend, process it, and send back the fall data."""
    global isFall, fallTime, imageLink

    # Decode base64 frame
    base64_frame = data['frame']
    nparr = np.frombuffer(base64.b64decode(base64_frame), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Process the frame using YOLO-based fall detection
    process_frame(frame)

    # Send fall data back to frontend
    emit('fall_update', {'isFall': isFall, 'time': fallTime, 'imageLink': imageLink})

@socketio.on('get_fall_data')
def handle_get_fall_data():
    """Send the current fall status when requested."""
    emit('health_update', {'isFall': isFall, 'time': fallTime, 'imageLink': imageLink})

if __name__ == '__main__':
    try:
        socketio.run(app, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("Server interrupted")
