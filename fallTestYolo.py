import cv2
import cvzone
import math
import time
import os
from ultralytics import YOLO

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load YOLO model
model = YOLO('yolov8s.pt')

# Load class names
classnames = []
with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines()

# Create output directory if it doesn't exist
output_dir = './outputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Variables to store fall detection data
isFall = False
fallTime = ""

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.resize(frame, (980, 740))
    
    # Get YOLO detection results
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
                # Draw bounding box and label
                cvzone.cornerRect(frame, [x1, y1, width, height], l=30, rt=6)
                cvzone.putTextRect(frame, f'{class_detect}', [x1 + 8, y1 - 12], thickness=2, scale=2)
                
                # Detect fall: If height < width, fall is detected
                if threshold < 0:
                    # Display "Fall Detected" on the frame
                    cvzone.putTextRect(frame, 'Fall Detected', [x1, y1 - 40], thickness=2, scale=2)

                    # If it's the first detection, store fall time and save the image
                    if not isFall:
                        isFall = True
                        fallTime = time.strftime('%Y-%m-%d %H:%M:%S')
                        image_path = os.path.join(output_dir, 'fallImage.png')
                        cv2.imwrite(image_path, frame)
                
            else:
                isFall = False

    # Display the fall time if a fall is detected
    if isFall:
        cv2.putText(frame, f"Fall Time: {fallTime}", (10, 700), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the video feed
    cv2.imshow('Test Feed', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
